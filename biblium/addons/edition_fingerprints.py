# -*- coding: utf-8 -*-
"""
biblium.addons.edition_fingerprints
===================================

Age-adjusted "edition fingerprint" estimation via per-concept logistic
regression.

Motivation
----------
Raw prevalence of a concept per cited edition confounds the effect of
the cited edition itself with the effect of the *citing* year: 4th-ed
citers are more recent, so any concept whose base rate is growing over
time will appear over-represented in the 4th ed even if the underlying
edition-effect is zero. Fitting

    logit P(concept_c = 1 | i) = a + f(edition_i) + g(year_i) + controls_i

for each concept, and reporting the edition dummies as adjusted odds
ratios against a chosen base edition, isolates the pure edition effect.

Public functions
----------------
compute_edition_fingerprints_adjusted
    Long-form (concept x edition) table with raw prevalence, adjusted OR,
    95% CI, edition-effect Wald p-value, and N per edition.

plot_fingerprint_comparison
    Two-panel heatmap: raw prevalence (%) vs adjusted OR (log-scale
    diverging colormap around 1).

Author: Lan Umek
Version: 2.16.0 (merged into 2.19)
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Backend selection: statsmodels first, sklearn L2 fallback
# ---------------------------------------------------------------------------
try:
    import statsmodels.api as sm  # noqa: F401
    _HAVE_SM = True
except Exception:  # pragma: no cover
    _HAVE_SM = False

try:
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False

try:
    from biblium.utilsbib import fit_logit as _biblium_fit_logit
    _HAVE_FIRTH = True
except Exception:  # pragma: no cover
    _HAVE_FIRTH = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_design(
    df: pd.DataFrame,
    edition_col: str,
    year_col: str,
    doc_type_col: Optional[str],
    discipline_col: Optional[str],
    control_year: bool,
    control_doc_type: bool,
    control_discipline: bool,
    base_edition,
    editions_ordered: List,
):
    """Return (X_df, edition_dummy_cols, meta) for later logit fits."""
    parts = []
    # Edition dummies with an explicit reference category
    ed = df[edition_col].astype(object)
    dummies = pd.get_dummies(ed, prefix="ed", drop_first=False)
    ref_col = f"ed_{base_edition}"
    if ref_col in dummies.columns:
        dummies = dummies.drop(columns=[ref_col])
    # Order dummy columns by the requested edition order
    ordered_dummy_cols = [f"ed_{e}" for e in editions_ordered
                          if e != base_edition and f"ed_{e}" in dummies.columns]
    dummies = dummies[ordered_dummy_cols].astype(float)
    parts.append(dummies)

    if control_year:
        yr = pd.to_numeric(df[year_col], errors="coerce")
        yr = yr - yr.mean()
        parts.append(pd.DataFrame({"year_c": yr.values}, index=df.index))

    if control_doc_type and doc_type_col and doc_type_col in df.columns:
        dt = pd.get_dummies(df[doc_type_col].astype(object),
                            prefix="dt", drop_first=True).astype(float)
        parts.append(dt)

    if control_discipline and discipline_col and discipline_col in df.columns:
        dd = pd.get_dummies(df[discipline_col].astype(object),
                            prefix="disc", drop_first=True).astype(float)
        parts.append(dd)

    X = pd.concat(parts, axis=1)
    # Constant column
    X.insert(0, "const", 1.0)
    return X, ordered_dummy_cols


def _fit_logit_sm(y: np.ndarray, X: pd.DataFrame):
    """Fit logit via statsmodels; return (params, cov, converged)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.Logit(y, X.values).fit(
                disp=False, method="newton", maxiter=100
            )
            return res.params, res.cov_params(), True, "sm"
        except Exception:
            try:
                res = sm.Logit(y, X.values).fit_regularized(
                    disp=False, alpha=1.0, L1_wt=0.0, maxiter=200
                )
                # cov not available for regularized fit
                p = np.asarray(res.params)
                return p, None, False, "sm_l2"
            except Exception:
                return None, None, False, "sm_fail"


def _fit_logit_sk(y: np.ndarray, X: pd.DataFrame):
    """Sklearn L2 fallback; no analytic SE."""
    Xa = X.values
    # Fit *without* the leading constant column, sklearn adds its own intercept
    model = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=500, fit_intercept=True,
    )
    model.fit(Xa[:, 1:], y)
    coef = np.concatenate([[float(model.intercept_[0])],
                           model.coef_.ravel()])
    return coef, None, True, "sk_l2"


def _fit_logit_firth(y: np.ndarray, X: pd.DataFrame):
    """Firth-penalised logit (via biblium.utilsbib).

    Provides bias-reduced coefficients AND standard errors + p-values,
    which is exactly what we need for age-adjusted fingerprints when
    statsmodels Logit fails to converge.
    """
    # X already contains an intercept column. Tell fit_logit not to add another.
    res = _biblium_fit_logit(
        X.values, y,
        method="firth",
        add_intercept=False,
        ci_method="wald",
        alpha=0.05,
        max_iter=200,
    )
    if res is None or res.coef is None:
        return None, None, False, "firth_fail"
    p = res.coef.shape[0]
    # Build a diagonal cov from Firth SEs (Wald CIs); full cov not exposed by adapter
    cov = np.diag(np.where(np.isfinite(res.se), res.se ** 2, np.nan))
    return res.coef, cov, bool(res.converged), "firth"


def _wald_p(coef: float, se: Optional[float]) -> float:
    if se is None or not np.isfinite(se) or se <= 0:
        return np.nan
    from scipy.stats import norm
    z = coef / se
    return float(2 * (1 - norm.cdf(abs(z))))


def _joint_wald_p(coef_vec: np.ndarray,
                  cov: Optional[np.ndarray]) -> float:
    """Joint Wald test that all elements of coef_vec are zero."""
    if cov is None or coef_vec.size == 0:
        return np.nan
    try:
        from scipy.stats import chi2
        inv = np.linalg.pinv(cov)
        stat = float(coef_vec @ inv @ coef_vec)
        df = int(coef_vec.size)
        return float(1 - chi2.cdf(stat, df))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_edition_fingerprints_adjusted(
    df: pd.DataFrame,
    edition_col: str,
    concept_matrix: pd.DataFrame,
    year_col: str = "Year",
    doc_type_col: Optional[str] = None,
    discipline_col: Optional[str] = None,
    min_docs_per_edition: int = 30,
    control_year: bool = True,
    control_doc_type: bool = False,
    control_discipline: bool = False,
    base_edition=None,
) -> pd.DataFrame:
    """
    Adjusted edition fingerprints via per-concept logistic regression.

    Model (one fit per concept)::

        logit P(concept = 1 | i) = a + sum_e beta_e * I[edition_i = e]
                                     + gamma * (year_i - mean_year)
                                     + controls_i

    with one edition dummy dropped as the reference category.

    Parameters
    ----------
    df : DataFrame
        Citing-document metadata (must contain ``edition_col`` and
        ``year_col``; also any control columns you enable).
    edition_col : str
        Column in ``df`` giving the attributed edition per doc.
        Docs with NaN / missing edition are excluded.
    concept_matrix : DataFrame (docs x concepts), bool or 0/1
        Aligned to ``df.index``.
    year_col : str, default "Year"
    doc_type_col, discipline_col : str, optional
        Extra control columns; enable with the corresponding
        ``control_*`` flag.
    min_docs_per_edition : int, default 30
        Drop editions with fewer attributable docs before fitting.
    control_year : bool, default True
    control_doc_type, control_discipline : bool, default False
    base_edition : optional
        Which edition to use as the reference category. Defaults to the
        smallest key after sorting (typically "1" = first ed).

    Returns
    -------
    DataFrame with columns::

        concept | edition | raw_pct | adjusted_OR |
        ci_low_95 | ci_high_95 | p_edition | n
    """
    if concept_matrix.shape[0] != df.shape[0]:
        raise ValueError(
            "concept_matrix rows must equal df rows; "
            f"got {concept_matrix.shape[0]} vs {df.shape[0]}."
        )

    work = df.copy()
    work[edition_col] = work[edition_col].astype(object)
    mask = work[edition_col].notna() & (work[edition_col].astype(str) != "")
    if control_year:
        mask &= pd.to_numeric(work[year_col], errors="coerce").notna()

    work = work.loc[mask].copy()
    cm = concept_matrix.loc[mask].copy().astype(bool)

    # Drop editions with too few docs
    counts = work[edition_col].value_counts()
    keep_eds = counts[counts >= min_docs_per_edition].index.tolist()
    if not keep_eds:
        raise ValueError(
            "No edition has >= min_docs_per_edition documents "
            f"(min_docs_per_edition={min_docs_per_edition})."
        )
    work = work[work[edition_col].isin(keep_eds)].copy()
    cm = cm.loc[work.index].copy()

    editions_ordered = sorted(keep_eds, key=lambda x: str(x))
    if base_edition is None or base_edition not in editions_ordered:
        base_edition = editions_ordered[0]

    X, dummy_cols = _build_design(
        work, edition_col=edition_col, year_col=year_col,
        doc_type_col=doc_type_col, discipline_col=discipline_col,
        control_year=control_year,
        control_doc_type=control_doc_type,
        control_discipline=control_discipline,
        base_edition=base_edition,
        editions_ordered=editions_ordered,
    )

    n_per_ed = work[edition_col].value_counts().to_dict()
    rows = []

    if not (_HAVE_SM or _HAVE_SK):
        raise ImportError(
            "Neither statsmodels nor scikit-learn is available; cannot fit."
        )
    if not _HAVE_SM:
        warnings.warn(
            "statsmodels missing; falling back to sklearn L2 logistic "
            "regression. Confidence intervals and p-values will be NaN.",
            RuntimeWarning,
        )

    for concept in cm.columns:
        y = cm[concept].astype(int).values
        raw_by_ed = {}
        for e in editions_ordered:
            m = (work[edition_col] == e).values
            raw_by_ed[e] = float(y[m].mean() * 100) if m.any() else np.nan

        # If concept is degenerate (constant), skip fitting
        if len(np.unique(y)) < 2:
            for e in editions_ordered:
                rows.append({
                    "concept": concept, "edition": e,
                    "raw_pct": raw_by_ed[e],
                    "adjusted_OR": np.nan,
                    "ci_low_95": np.nan, "ci_high_95": np.nan,
                    "p_edition": np.nan,
                    "n": int(n_per_ed.get(e, 0)),
                })
            continue

        params = cov = None
        method_used = ""
        converged = False
        if _HAVE_SM:
            params, cov, converged, method_used = _fit_logit_sm(y, X)
            # If sm returned an unregularised newton fit with cov, keep it.
            # Otherwise (l2 fallback or fail) try Firth to get SEs + p-values.
            if (method_used in ("sm_l2", "sm_fail") or cov is None) and _HAVE_FIRTH:
                p2, c2, cv2, mu2 = _fit_logit_firth(y, X)
                if p2 is not None and c2 is not None:
                    params, cov, converged, method_used = p2, c2, cv2, mu2
        elif _HAVE_FIRTH:
            params, cov, converged, method_used = _fit_logit_firth(y, X)
        if params is None and _HAVE_SK:
            params, cov, converged, method_used = _fit_logit_sk(y, X)

        if params is None:
            for e in editions_ordered:
                rows.append({
                    "concept": concept, "edition": e,
                    "raw_pct": raw_by_ed[e],
                    "adjusted_OR": np.nan,
                    "ci_low_95": np.nan, "ci_high_95": np.nan,
                    "p_edition": np.nan,
                    "n": int(n_per_ed.get(e, 0)),
                })
            continue

        # Map coefficient index -> column name
        colnames = X.columns.tolist()
        coef_by_col = dict(zip(colnames, params))
        se_by_col = {c: np.nan for c in colnames}
        if cov is not None:
            se_vec = np.sqrt(np.clip(np.diag(cov), 0, None))
            se_by_col = dict(zip(colnames, se_vec))

        # Joint p-value on all edition dummies
        ed_coef_vec = np.array([coef_by_col[c] for c in dummy_cols])
        ed_cov = None
        if cov is not None:
            idx = [colnames.index(c) for c in dummy_cols]
            ed_cov = cov[np.ix_(idx, idx)]
        p_edition = _joint_wald_p(ed_coef_vec, ed_cov)

        # Base edition row (OR = 1 by construction)
        rows.append({
            "concept": concept, "edition": base_edition,
            "raw_pct": raw_by_ed[base_edition],
            "adjusted_OR": 1.0,
            "ci_low_95": 1.0, "ci_high_95": 1.0,
            "p_edition": p_edition,
            "n": int(n_per_ed.get(base_edition, 0)),
        })
        for e in editions_ordered:
            if e == base_edition:
                continue
            col = f"ed_{e}"
            b = float(coef_by_col.get(col, np.nan))
            se = float(se_by_col.get(col, np.nan))
            OR = float(np.exp(b)) if np.isfinite(b) else np.nan
            if np.isfinite(se) and se > 0:
                lo = float(np.exp(b - 1.96 * se))
                hi = float(np.exp(b + 1.96 * se))
            else:
                lo = hi = np.nan
            rows.append({
                "concept": concept, "edition": e,
                "raw_pct": raw_by_ed[e],
                "adjusted_OR": OR,
                "ci_low_95": lo, "ci_high_95": hi,
                "p_edition": p_edition,
                "n": int(n_per_ed.get(e, 0)),
            })

    out = pd.DataFrame(rows, columns=[
        "concept", "edition", "raw_pct",
        "adjusted_OR", "ci_low_95", "ci_high_95",
        "p_edition", "n",
    ])

    # --- Benjamini-Hochberg FDR correction on the per-concept edition
    # p-values. p_edition is a JOINT test per concept (same value repeated
    # across that concept's edition rows), so we correct at the CONCEPT
    # level: one p per concept -> BH -> q value mapped back to all rows.
    per_concept = (out.dropna(subset=["p_edition"])
                      .groupby("concept")["p_edition"].first())
    if len(per_concept) > 0:
        pvals = per_concept.sort_values()
        m_tests = len(pvals)
        ranks = np.arange(1, m_tests + 1)
        raw_q = pvals.values * m_tests / ranks
        # enforce monotonicity from the largest rank down
        q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
        q_map = dict(zip(pvals.index, np.clip(q_sorted, 0, 1)))
        out["q_edition_fdr"] = out["concept"].map(q_map)
        out["significant_fdr_05"] = out["q_edition_fdr"] < 0.05
    else:
        out["q_edition_fdr"] = np.nan
        out["significant_fdr_05"] = False

    out.attrs["base_edition"] = base_edition
    out.attrs["method"] = "statsmodels" if _HAVE_SM else "sklearn_l2"
    out.attrs["fdr_method"] = "benjamini_hochberg_concept_level"
    return out


def plot_fingerprint_comparison(
    fingerprints_df: pd.DataFrame,
    out_path,
    top_n_concepts: int = 20,
    title: str = "Edition fingerprints: raw vs age-adjusted",
):
    """
    Two-panel heatmap: raw prevalence (%) vs adjusted OR (log-scale,
    diverging RdBu_r around 1).

    Rows are sorted by variance across editions (adjusted-OR panel).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    df = fingerprints_df.copy()

    # Pivot to concept x edition
    raw = df.pivot_table(index="concept", columns="edition",
                         values="raw_pct", aggfunc="mean")
    adj = df.pivot_table(index="concept", columns="edition",
                         values="adjusted_OR", aggfunc="mean")

    # Rank concepts by variance in log(OR)
    log_or = np.log(adj.replace(0, np.nan))
    variance = log_or.var(axis=1, skipna=True).fillna(0)
    order = variance.sort_values(ascending=False).head(top_n_concepts).index
    raw = raw.loc[order]
    adj = adj.loc[order]

    ed_cols = sorted(adj.columns, key=lambda x: str(x))
    raw = raw.reindex(columns=ed_cols)
    adj = adj.reindex(columns=ed_cols)

    fig, axes = plt.subplots(
        1, 2, figsize=(11, max(4.5, 0.35 * len(order) + 2)),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.35},
    )

    # Panel A: raw prevalence
    ax = axes[0]
    im = ax.imshow(raw.values, aspect="auto", cmap="Blues",
                   vmin=0, vmax=float(np.nanmax(raw.values)) or 1.0)
    ax.set_xticks(range(len(ed_cols)))
    ax.set_xticklabels([str(c) for c in ed_cols], rotation=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(_wrap_labels(order.tolist()), fontsize=8)
    ax.set_title("Raw prevalence (%)", fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # Panel B: adjusted OR on log scale, diverging around 1
    ax = axes[1]
    adj_plot = adj.copy()
    finite = adj_plot.values[np.isfinite(adj_plot.values)]
    if finite.size == 0:
        vmin, vmax = 0.5, 2.0
    else:
        lo = max(min(finite[finite > 0].min() if (finite > 0).any() else 0.5,
                     0.5), 1e-3)
        hi = max(finite.max(), 2.0)
        m = max(hi, 1.0 / lo)
        vmin, vmax = 1.0 / m, m
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(adj_plot.values, aspect="auto", cmap="RdBu_r",
                   norm=norm)
    ax.set_xticks(range(len(ed_cols)))
    ax.set_xticklabels([str(c) for c in ed_cols], rotation=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["" for _ in order])
    ax.set_title("Adjusted OR (log)", fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def _wrap_labels(labels, width: int = 28):
    import textwrap
    return [textwrap.fill(str(x), width=width) for x in labels]


__all__ = [
    "compute_edition_fingerprints_adjusted",
    "plot_fingerprint_comparison",
]
