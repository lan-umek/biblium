# -*- coding: utf-8 -*-
"""
GroupYearTrendMixin — testi monotonega trenda članstva v skupinah skozi
leta. Year je urejenostna spremenljivka, zato klasični chi-square / CA na
year × group ne prepozna smeri trenda. Tu uporabimo:

- Mann-Kendall (Kendall's tau na (year, yearly_proportion)) — neparametrični
  test za monotoni trend
- linearna regresija (year → yearly_proportion) — efekt v percentage points
  per year + Wald p-value
- opcijsko: logistična regresija (per-doc, doc_in_group ~ year) — samo če je
  statsmodels nameščen

Output:
- DataFrame z eno vrstico na skupino: tau, MK p, slope (pp/year), slope p,
  BH-popravljene p-vrednosti
- opcijski plot: line chart yearly proportions + linearna trend črta + zvezdice

@author: Lan + biblium 2.16
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Helpers
# =============================================================================

def _kendall_tau(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Kendall's tau-b (rank correlation) in dvostranska p-vrednost.
    Uporablja scipy, če je na voljo, sicer enostavna implementacija."""
    try:
        from scipy.stats import kendalltau
        tau, p = kendalltau(x, y, nan_policy="omit")
        return float(tau), float(p)
    except Exception:
        # Fallback: naivni O(n^2)
        x = np.asarray(x); y = np.asarray(y)
        m = ~(np.isnan(x) | np.isnan(y))
        x, y = x[m], y[m]
        n = len(x)
        if n < 2:
            return 0.0, 1.0
        concordant = discordant = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                dx, dy = x[j] - x[i], y[j] - y[i]
                s = np.sign(dx) * np.sign(dy)
                concordant += int(s > 0)
                discordant += int(s < 0)
        denom = 0.5 * n * (n - 1)
        tau = (concordant - discordant) / denom if denom else 0.0
        # Aproksimativna p-vrednost (normalna), brez tie correction
        var = (2 * (2 * n + 5)) / (9 * n * (n - 1)) if n > 1 else 1.0
        z = tau / np.sqrt(var) if var > 0 else 0.0
        try:
            from math import erf, sqrt
            p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        except Exception:
            p = 1.0
        return float(tau), float(p)


def _linear_slope_p(x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None
                    ) -> tuple[float, float, float]:
    """OLS slope, slope SE, slope p (Wald). Z opcijo uteži (n_year)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return 0.0, np.nan, 1.0
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)
        w = np.where(w > 0, w, 0)
    sw = w.sum()
    if sw == 0:
        return 0.0, np.nan, 1.0
    xm = (w * x).sum() / sw
    ym = (w * y).sum() / sw
    sxx = (w * (x - xm) ** 2).sum()
    sxy = (w * (x - xm) * (y - ym)).sum()
    if sxx <= 0:
        return 0.0, np.nan, 1.0
    slope = sxy / sxx
    intercept = ym - slope * xm
    yhat = intercept + slope * x
    resid = y - yhat
    # Weighted residual variance with df = n - 2
    dof = max(n - 2, 1)
    sigma2 = (w * resid ** 2).sum() / dof
    se = np.sqrt(sigma2 / sxx) if sigma2 > 0 else np.nan
    if se and se > 0 and not np.isnan(se):
        t = slope / se
        try:
            from scipy.stats import t as student_t
            p = 2 * (1 - student_t.cdf(abs(t), df=dof))
        except Exception:
            # fallback: normalna aproksimacija
            from math import erf, sqrt
            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    else:
        p = 1.0
    return float(slope), float(se), float(p)


def _bh_adjust(pvals: Iterable[float]) -> list[float]:
    """Benjamini-Hochberg popravek p-vrednosti."""
    p = np.asarray(list(pvals), dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    # monotone non-increasing from the right
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


# =============================================================================
# Mixin
# =============================================================================

class GroupYearTrendMixin:
    """Adds analyze_year_trends + plot_year_trends to BiblioGroup."""

    def analyze_year_trends(
        self,
        year_col: str = "Year",
        year_range: Optional[Tuple[int, int]] = None,
        min_year_n: int = 5,
        multiple_testing: str = "bh",
        plot: bool = True,
        filename: str = "year_trend",
        figsize: Tuple[float, float] = (12, 7),
    ) -> pd.DataFrame:
        """
        Per-group monotonic trend analysis over years (Mann-Kendall +
        linear regression on yearly proportions).

        For each group g and year y compute:
            p_{g,y} = (# docs in g published in y) / (# docs in y)
        Then:
        - Mann-Kendall tau(year, p_{g,·}) → tau, p-value
        - OLS slope of p_{g,·} ~ year, weighted by # docs in y → slope, p

        Parameters
        ----------
        year_col : str
            Column with publication year (must be in self.df).
        year_range : (int, int), optional
            Restrict to [start, end] inclusive.
        min_year_n : int, default 5
            Skip years where total docs < min_year_n (noisy proportions).
        multiple_testing : {"bh", "none"}, default "bh"
            Correction across groups.
        plot : bool, default True
            Save line chart with trend annotations.
        filename : str, default "year_trend"
            Base filename in <res_folder>/plots/ (slika) and <res_folder>/tables/.

        Returns
        -------
        pd.DataFrame
            Columns: group, n_papers, n_years, year_min, year_max,
                     mk_tau, mk_p, mk_p_adj, slope_pp_per_year, slope_se,
                     slope_p, slope_p_adj, mean_share_pct
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka. Najprej zgradi skupine.")

        if year_col not in self.df.columns:
            raise ValueError(f"Stolpec {year_col!r} ne obstaja v self.df.")

        years_all = pd.to_numeric(self.df[year_col], errors="coerce")
        valid_mask = years_all.notna()
        if year_range is not None:
            yr_lo, yr_hi = int(year_range[0]), int(year_range[1])
            valid_mask &= (years_all >= yr_lo) & (years_all <= yr_hi)

        df_y = self.df.loc[valid_mask].copy()
        df_y["__year__"] = years_all.loc[valid_mask].astype(int)

        gm = self.group_matrix.loc[df_y.index].astype(int)
        groups = list(gm.columns)

        # Total docs per year
        n_per_year = df_y["__year__"].value_counts().sort_index()
        n_per_year = n_per_year[n_per_year >= min_year_n]
        years_kept = n_per_year.index.values

        if len(years_kept) < 3:
            raise ValueError(
                f"Premalo let z >= {min_year_n} dokumenti ({len(years_kept)} let). "
                f"Spremeni year_range ali zmanjšaj min_year_n."
            )

        # For each group, yearly proportion
        rows = []
        proportions = pd.DataFrame(index=years_kept, columns=groups, dtype=float)

        for g in groups:
            in_g = gm[g].astype(bool)
            n_g_per_year = df_y.loc[in_g, "__year__"].value_counts().reindex(years_kept).fillna(0)
            n_g_per_year = n_g_per_year.astype(int)
            prop = n_g_per_year / n_per_year
            proportions[g] = prop.values

            tau, p_mk = _kendall_tau(years_kept.astype(float), prop.values)
            slope, slope_se, slope_p = _linear_slope_p(
                years_kept.astype(float), prop.values, weights=n_per_year.values
            )

            rows.append({
                "group": g,
                "n_papers": int(in_g.sum()),
                "n_years": int(len(years_kept)),
                "year_min": int(years_kept.min()),
                "year_max": int(years_kept.max()),
                "mk_tau": tau,
                "mk_p": p_mk,
                "slope_pp_per_year": float(slope * 100),
                "slope_se": float((slope_se or 0) * 100),
                "slope_p": slope_p,
                "mean_share_pct": float(prop.mean() * 100),
            })

        result = pd.DataFrame(rows)

        if multiple_testing == "bh":
            result["mk_p_adj"] = _bh_adjust(result["mk_p"].fillna(1.0).tolist())
            result["slope_p_adj"] = _bh_adjust(result["slope_p"].fillna(1.0).tolist())
        else:
            result["mk_p_adj"] = result["mk_p"]
            result["slope_p_adj"] = result["slope_p"]

        # Reorder
        cols = [
            "group", "n_papers", "n_years", "year_min", "year_max",
            "mean_share_pct",
            "mk_tau", "mk_p", "mk_p_adj",
            "slope_pp_per_year", "slope_se", "slope_p", "slope_p_adj",
        ]
        result = result[cols].sort_values("slope_pp_per_year", ascending=False).reset_index(drop=True)

        # Save
        res_folder = getattr(self, "res_folder", None)
        if res_folder is not None and filename:
            tables_dir = os.path.join(res_folder, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            tab_path = os.path.join(tables_dir, f"{filename}.xlsx")
            with pd.ExcelWriter(tab_path) as w:
                result.to_excel(w, sheet_name="trend_summary", index=False)
                proportions.round(5).to_excel(w, sheet_name="proportions_year")
                n_per_year.to_frame("Documents").to_excel(w, sheet_name="docs_per_year")

        # Plot
        if plot:
            _plot_year_trends(
                proportions=proportions,
                trend_df=result,
                res_folder=res_folder,
                filename=filename,
                figsize=figsize,
                dpi=getattr(self, "dpi", 200),
            )

        self.year_trend_results = result
        self.year_trend_proportions = proportions
        return result


# =============================================================================
# Plot
# =============================================================================

    # =============================================================================
    # classify_concept_lifecycle — 5-fazni klasifikator
    # =============================================================================
    def classify_concept_lifecycle(
        self,
        year_col: str = "Year",
        year_range: Optional[Tuple[int, int]] = None,
        min_year_n: int = 5,
        recent_window: int = 3,
        emerging_first_year_quantile: float = 0.66,
        burst_ratio: float = 2.0,
        rising_alpha: float = 0.05,
        plot: bool = True,
        filename: str = "concept_lifecycle",
        figsize: Tuple[float, float] = (10, 8),
    ) -> pd.DataFrame:
        """
        Klasificira vsak koncept v eno od 5 faz: Burst, Emerging, Growing,
        Mature, Declining (po Mann-Kendall + recent/prior burst score).

        Returns
        -------
        pd.DataFrame z stolpci:
          concept, n_total, first_year, peak_year, peak_count, last_3y_share,
          tau, p_value, burst_score, phase
        """
        import os as _os
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka.")
        if year_col not in self.df.columns:
            raise ValueError(f"Stolpec {year_col!r} ne obstaja.")

        years = pd.to_numeric(self.df[year_col], errors="coerce")
        gm = self.group_matrix.astype(int)
        ok = years.notna()
        years = years[ok].astype(int)
        gm = gm.loc[ok]

        if year_range is not None:
            lo, hi = year_range
            sel = (years >= lo) & (years <= hi)
            years = years[sel]
            gm = gm.loc[sel.values if isinstance(sel, pd.Series) else sel]

        if len(years) == 0:
            return pd.DataFrame()

        yr_min = int(years.min())
        yr_max = int(years.max())
        emerging_threshold_year = int(
            yr_min + (yr_max - yr_min) * emerging_first_year_quantile
        )

        by_year = gm.groupby(years.values).sum().sort_index()

        rows = []
        for c in gm.columns:
            series = by_year[c]
            nonzero = series[series > 0]
            if len(nonzero) < 2:
                continue
            first_year = int(nonzero.index.min())
            peak_year = int(nonzero.idxmax())
            peak_count = int(nonzero.max())
            n_total = int(series.sum())

            yrs_arr = series.index.values.astype(float)
            vals = series.values.astype(float)
            tau, pval = _kendall_tau(yrs_arr, vals)

            if len(series) >= 2 * recent_window:
                recent_mean = float(series.iloc[-recent_window:].mean())
                prior_mean = float(series.iloc[-2 * recent_window:-recent_window].mean())
            else:
                recent_mean = float(series.iloc[-recent_window:].mean()) if len(series) >= recent_window else 0.0
                prior_mean = float(series.iloc[:-recent_window].mean()) if len(series) > recent_window else 1e-9
            prior_mean = max(prior_mean, 1e-9)
            burst_score = recent_mean / prior_mean

            recent_share = float(series.iloc[-recent_window:].sum()) / max(1, n_total)

            if burst_score >= burst_ratio:
                phase = "Burst"
            elif first_year >= emerging_threshold_year and tau > 0 and pval < rising_alpha:
                phase = "Emerging"
            elif tau > 0 and pval < rising_alpha:
                phase = "Growing"
            elif tau < 0 and pval < rising_alpha:
                phase = "Declining"
            else:
                phase = "Mature"

            rows.append({
                "concept": c,
                "n_total": n_total,
                "first_year": first_year,
                "peak_year": peak_year,
                "peak_count": peak_count,
                "last_3y_share": round(recent_share, 3),
                "tau": round(tau, 3),
                "p_value": round(pval, 4),
                "burst_score": round(burst_score, 2),
                "phase": phase,
            })

        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            phase_order = {"Burst": 0, "Emerging": 1, "Growing": 2,
                           "Mature": 3, "Declining": 4}
            df_out["_po"] = df_out["phase"].map(phase_order).fillna(99).astype(int)
            df_out = df_out.sort_values(
                ["_po", "n_total"], ascending=[True, False]
            ).drop(columns="_po").reset_index(drop=True)

        res_folder = getattr(self, "res_folder", None)
        if res_folder is not None and filename:
            tables_dir = _os.path.join(res_folder, "tables")
            _os.makedirs(tables_dir, exist_ok=True)
            df_out.to_excel(_os.path.join(tables_dir, f"{filename}.xlsx"),
                            index=False)

        if plot and not df_out.empty:
            import matplotlib.pyplot as plt
            phase_colors = {
                "Burst":     "#d62728",
                "Emerging":  "#ff7f0e",
                "Growing":   "#2ca02c",
                "Mature":    "#1f77b4",
                "Declining": "#7f7f7f",
            }
            fig, ax = plt.subplots(figsize=figsize)
            sizes = (df_out["peak_count"].clip(lower=1)).pow(0.5) * 18
            for phase, group in df_out.groupby("phase"):
                ax.scatter(
                    group["tau"], np.log10(group["n_total"].clip(lower=1)),
                    s=sizes.loc[group.index], color=phase_colors.get(phase, "#888"),
                    alpha=0.75, edgecolor="white", linewidth=0.8, label=phase
                )
            for _, r in df_out.iterrows():
                ax.text(r["tau"] + 0.005, np.log10(max(1, r["n_total"])),
                        r["concept"], fontsize=8, va="center")
            ax.axvline(0, color="grey", lw=0.5, linestyle="--")
            ax.set_xlabel("Mann-Kendall tau (trend over years)")
            ax.set_ylabel("log10(n_total papers)")
            ax.set_title("Concept lifecycle classification", fontsize=11)
            ax.grid(False)
            ax.legend(title="Phase", loc="best", fontsize=9)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            fig.tight_layout()
            if res_folder:
                plots_dir = _os.path.join(res_folder, "plots")
                _os.makedirs(plots_dir, exist_ok=True)
                fig.savefig(_os.path.join(plots_dir, f"{filename}.png"),
                            dpi=getattr(self, "dpi", 200), bbox_inches="tight")
            plt.close(fig)

        self.concept_lifecycle_df = df_out
        return df_out



def _plot_year_trends(
    proportions: pd.DataFrame,
    trend_df: pd.DataFrame,
    res_folder: Optional[str],
    filename: str,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    years = proportions.index.values.astype(int)
    fig, ax = plt.subplots(figsize=figsize)

    # Iz trend_df vzemi label z slope + zvezdice glede na p_adj
    label_map = {}
    for _, r in trend_df.iterrows():
        slope = r["slope_pp_per_year"]
        p_adj = r["slope_p_adj"]
        stars = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else ""))
        sign = "+" if slope >= 0 else ""
        label_map[r["group"]] = f"{r['group']}  ({sign}{slope:.2f} pp/y{stars})"

    # Sortiraj skupine po slope (upadajoče)
    ordered_groups = trend_df.sort_values("slope_pp_per_year", ascending=False)["group"].tolist()

    for g in ordered_groups:
        if g not in proportions.columns:
            continue
        y = proportions[g].values * 100  # v procentih
        ax.plot(years, y, marker="o", linewidth=2, label=label_map.get(g, g))

    ax.set_xlabel("Year")
    ax.set_ylabel("% of yearly documents in group")
    ax.set_title("Group share over time — Mann-Kendall + linear trend"
                 "  (***p<.001  **p<.01  *p<.05)")
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(False)
    fig.tight_layout()

    if res_folder is not None and filename:
        plots_dir = os.path.join(res_folder, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, f"{filename}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Alternative visualizations of the group-year trend matrix
# (reviewer feedback C4: the line plot with 17 concepts is unreadable;
# offer heatmap, slope chart and small-multiples)
# ---------------------------------------------------------------------------

def plot_group_year_trend_heatmap(
    proportions: pd.DataFrame,
    out_path: str,
    title: str = "Group share over time (% of yearly docs)",
    cmap: str = "RdBu_r",
    annotate: bool = False,
    dpi: int = 200,
) -> None:
    """Alternative C4(a): heatmap concept x year, signed deviation from mean.

    Each cell shows (share_in_year - mean_share_over_years) in percentage
    points. Reading the row tells you when a concept was hotter/colder than
    its long-run average.
    """
    import matplotlib.pyplot as plt
    if proportions is None or proportions.empty:
        return
    data = proportions.fillna(0).astype(float) * 100.0  # to percent
    centred = data.sub(data.mean(axis=0), axis=1)  # year x concept
    arr = centred.T.values  # concept x year
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(centred)),
                                    max(5, 0.4 * arr.shape[0])))
    vmax = float(np.nanmax(np.abs(arr))) or 0.1
    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(arr.shape[1]))
    ax.set_xticklabels([str(int(y)) for y in data.index], rotation=45, ha="right")
    ax.set_yticks(range(arr.shape[0]))
    ax.set_yticklabels(list(centred.columns), fontsize=8)
    ax.set_title(title + "  (signed deviation from concept mean, pp)")
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="pp vs mean")
    if annotate and arr.shape[0] * arr.shape[1] <= 400:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if abs(v) >= vmax * 0.4:
                    col = "white" if abs(v) >= vmax * 0.65 else "black"
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                            fontsize=6, color=col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_group_year_trend_slope(
    proportions: pd.DataFrame,
    out_path: str,
    n_periods: int = 2,
    title: str = "Group share -- first vs last period",
    dpi: int = 200,
) -> None:
    """Alternative C4(b): slope chart comparing first vs last period share.

    Splits years into ``n_periods`` equal buckets and plots a line per
    concept going from its share in the first bucket to the last.
    """
    import matplotlib.pyplot as plt
    if proportions is None or proportions.empty:
        return
    NAVY = "#1f3a93"
    RED = "#c0392b"
    data = proportions.fillna(0).astype(float) * 100.0
    years_sorted = sorted(data.index)
    if len(years_sorted) < 2:
        return
    cut = max(2, n_periods)
    chunks = np.array_split(years_sorted, cut)
    period_labels = [f"{c[0]}-{c[-1]}" for c in chunks if len(c) > 0]
    period_means = pd.DataFrame({
        period_labels[i]: data.loc[list(chunks[i])].mean(axis=0)
        for i in range(len(period_labels))
    })  # rows=concept, cols=period
    period_means = period_means.dropna(how="all")
    if period_means.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(6, 0.4 * len(period_means))))
    xpos = list(range(len(period_labels)))
    for concept, row in period_means.iterrows():
        ys = row.values.tolist()
        rising = ys[-1] >= ys[0]
        col = NAVY if rising else RED
        ax.plot(xpos, ys, marker="o", color=col, alpha=0.85, linewidth=1.5)
        ax.text(xpos[-1] + 0.05, ys[-1], f"  {concept}",
                fontsize=8, va="center", color=col)
    ax.set_xticks(xpos)
    ax.set_xticklabels(period_labels)
    ax.set_ylabel("Share (%)")
    ax.set_title(title)
    ax.grid(False)
    ax.set_xlim(-0.2, len(xpos) + 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_group_year_trend_small_multiples(
    proportions: pd.DataFrame,
    out_path: str,
    title: str = "Group share over time -- small multiples",
    n_cols: int = 4,
    dpi: int = 200,
) -> None:
    """Alternative C4(c): one mini chart per concept on a grid.

    A 4xN grid (default 4 cols) with a single line per concept; reader can
    scan all concepts at once without spaghetti overlap.
    """
    import matplotlib.pyplot as plt
    if proportions is None or proportions.empty:
        return
    NAVY = "#1f3a93"
    concepts = list(proportions.columns)
    n_cols = max(2, min(int(n_cols), 6))
    n_rows = int(np.ceil(len(concepts) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 2.0 * n_rows),
        sharex=True, sharey=False,
    )
    axes = np.atleast_2d(axes)
    years = proportions.index.values.astype(int)
    for i, concept in enumerate(concepts):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        y = proportions[concept].fillna(0).astype(float).values * 100
        ax.plot(years, y, color=NAVY, linewidth=1.5)
        ax.fill_between(years, 0, y, color=NAVY, alpha=0.15)
        ax.set_title(str(concept)[:38], fontsize=8)
        ax.grid(False)
        ax.tick_params(labelsize=6)
    # blank the leftover axes
    for j in range(len(concepts), n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
