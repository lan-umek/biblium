# -*- coding: utf-8 -*-
"""
biblium.reference_patterns
==========================

Extensions of Reference Publication Year Spectroscopy (RPYS) and citation
history patterns of individual cited references (new in 2.19; analogue of the
RPYS / reference-pattern tools in bibliometrix 5.1).

Contents
--------
RPYS
    ``compute_rpys_backward``       RPYS with a centered or backward rolling median
    ``plot_rpys_extended``          bars (citations), median line, deviation area

Cited-reference time series
    ``cited_reference_timeseries``  long table Reference x Citing Year x Citations
    ``classify_reference_patterns`` Hot Paper / Sleeping Beauty / Constant
                                    Performer / Life Cycle / Other
    ``rpys_pattern_summary``        counts, shares and top references per pattern
    ``plot_reference_patterns``     citation curves of the top references coloured
                                    by pattern

All functions are pure and deterministic; plots use matplotlib only and
return ``(fig, ax)``.

Author: Lan Umek
Version: 2.19.0
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "compute_rpys_backward",
    "cited_reference_timeseries",
    "classify_reference_patterns",
    "rpys_pattern_summary",
    "plot_rpys_extended",
    "plot_reference_patterns",
]

PATTERN_ORDER = (
    "Hot Paper",
    "Sleeping Beauty",
    "Constant Performer",
    "Life Cycle",
    "Other",
    "Insufficient data",
)

_TRAILING_YEAR_RE = re.compile(r"\((\d{4})\)\s*\.?\s*$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_reference(ref: str) -> str:
    """Normalisation key of a reference string: lowercase, collapsed
    whitespace, trailing periods removed."""
    key = re.sub(r"\s+", " ", str(ref)).strip().lower()
    return key.rstrip(". ").strip()


def _cited_year(ref: str) -> float:
    """Year in the trailing ``(YYYY)`` of a reference string (NaN if absent)."""
    m = _TRAILING_YEAR_RE.search(str(ref))
    return float(m.group(1)) if m else np.nan


def _finish(fig, ax, title, f_name, dpi, show):
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if f_name:
        fig.savefig(f_name, dpi=dpi, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 1) RPYS with centered / backward median
# ---------------------------------------------------------------------------

def compute_rpys_backward(
    df: pd.DataFrame,
    reference_column: str = "References",
    include_pre_1900: bool = False,
    median_window: int = 5,
    mode: str = "centered",
) -> pd.DataFrame:
    """
    Reference Publication Year Spectroscopy with a selectable rolling median.

    Like :func:`biblium.utilsbib.compute_rpys` but ``mode`` selects how the
    median baseline is computed:

    * ``"centered"`` -- median of the ``median_window`` years centred on the
      year (as in Marx et al. 2014 and ``compute_rpys``);
    * ``"backward"`` -- median of the *preceding* ``median_window`` years
      (the year itself and ``median_window - 1`` earlier years; pandas rolling
      window without centering). The backward variant does not use future
      information and is the convention of some RPYS tools.

    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic data with a reference column (Scopus-style strings).
    reference_column : str
        Column with reference strings separated by ``;``.
    include_pre_1900 : bool
        Also count cited years in 17xx/18xx.
    median_window : int
        Width of the rolling window (years).
    mode : {"centered", "backward"}
        Position of the rolling window.

    Returns
    -------
    pd.DataFrame
        Columns: ``Cited Year``, ``Citations``, ``Median``, ``Deviation``
        (Citations - Median) and ``Deviation %`` (Deviation / Median * 100,
        NaN where the median is 0). Missing years in the range are filled
        with 0 citations.
    """
    from biblium.utilsbib import compute_reference_spectrogram_scopus

    if mode not in ("centered", "backward"):
        raise ValueError("mode must be 'centered' or 'backward'")
    cols = ["Cited Year", "Citations", "Median", "Deviation", "Deviation %"]
    if reference_column not in df.columns:
        return pd.DataFrame(columns=cols)
    spec = compute_reference_spectrogram_scopus(
        df, reference_column=reference_column, include_pre_1900=include_pre_1900
    )
    if spec.empty:
        return pd.DataFrame(columns=cols)
    out = spec.reset_index()
    yr_min, yr_max = int(out["Cited Year"].min()), int(out["Cited Year"].max())
    full = pd.DataFrame({"Cited Year": range(yr_min, yr_max + 1)})
    out = full.merge(out, on="Cited Year", how="left")
    out["Citations"] = out["Citations"].fillna(0).astype(int)
    out["Median"] = _rolling_median(out["Citations"], median_window, mode)
    out["Deviation"] = out["Citations"] - out["Median"]
    med = out["Median"].replace(0, np.nan)
    out["Deviation %"] = out["Deviation"] / med * 100.0
    return out[cols]


def _rolling_median(values: pd.Series, window: int, mode: str) -> pd.Series:
    center = mode == "centered"
    return values.rolling(window, center=center, min_periods=1).median()


# ---------------------------------------------------------------------------
# 2) Cited-reference time series
# ---------------------------------------------------------------------------

def cited_reference_timeseries(
    df: pd.DataFrame,
    reference_column: str = "References",
    year_col: str = "Year",
    sep: str = ";",
    min_citations: int = 3,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Citation history of individual cited references within the corpus.

    For every reference cited by at least ``min_citations`` documents (in
    total), the number of citing documents per citing (publication) year is
    returned. A reference is counted once per citing document.

    Parameters
    ----------
    df : pd.DataFrame
        Corpus with a reference column and a publication-year column.
    reference_column, year_col : str
        Column names.
    sep : str
        Separator of references inside a cell.
    min_citations : int
        Minimum total number of citing documents for a reference to be kept.
    normalize : bool
        If True, reference strings are matched on a normalised key (lowercase,
        collapsed whitespace, trailing periods removed) and the most frequent
        original spelling is reported as ``Reference``. If False, strings are
        matched verbatim (after stripping).

    Returns
    -------
    pd.DataFrame
        Long table with columns ``Reference``, ``Cited Year`` (year parsed from
        the trailing ``(YYYY)`` of the reference, NaN if absent), ``Citing
        Year`` and ``Citations``. Only (reference, year) pairs with at least one
        citation are listed; sorted by Reference and Citing Year.
    """
    cols = ["Reference", "Cited Year", "Citing Year", "Citations"]
    if reference_column not in df.columns or year_col not in df.columns:
        return pd.DataFrame(columns=cols)
    years = _to_numeric(df[year_col])
    refs = df[reference_column]

    per_year: Dict[str, Counter] = defaultdict(Counter)
    spellings: Dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()

    for idx in df.index:
        y = years.loc[idx]
        cell = refs.loc[idx]
        if pd.isna(y) or pd.isna(cell):
            continue
        seen = set()
        for tok in str(cell).split(sep):
            tok = tok.strip()
            if not tok or tok.lower() == "nan":
                continue
            key = _normalize_reference(tok) if normalize else tok
            if key in seen:
                continue
            seen.add(key)
            per_year[key][int(y)] += 1
            spellings[key][tok] += 1
            totals[key] += 1

    rows = []
    for key, total in totals.items():
        if total < min_citations:
            continue
        # most frequent spelling; ties broken alphabetically for determinism
        label = sorted(spellings[key].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        cy = _cited_year(label)
        for yr, n in sorted(per_year[key].items()):
            rows.append({"Reference": label, "Cited Year": cy, "Citing Year": yr, "Citations": n})
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out = out.sort_values(["Reference", "Citing Year"], kind="mergesort").reset_index(drop=True)
    out["Citing Year"] = out["Citing Year"].astype(int)
    out["Citations"] = out["Citations"].astype(int)
    return out


# ---------------------------------------------------------------------------
# 3) Classification of citation patterns
# ---------------------------------------------------------------------------

def _series_stats(ts_ref: pd.DataFrame) -> Dict[str, float]:
    """Descriptive statistics of one reference's citation history."""
    g = ts_ref.groupby("Citing Year")["Citations"].sum()
    g = g[g > 0]
    first, last = int(g.index.min()), int(g.index.max())
    full = pd.Series(0, index=range(first, last + 1), dtype=float)
    full.loc[g.index] = g.values.astype(float)
    total = float(full.sum())
    span = last - first + 1
    mean = total / span
    std = float(full.std(ddof=0))
    cv = std / mean if mean > 0 else np.nan
    peak_year = int(full.idxmax())  # first year with the maximum
    peak = float(full.max())
    # dormancy: consecutive years with <= 1 citation from the first citing year
    dormancy = 0
    for v in full.values:
        if v <= 1:
            dormancy += 1
        else:
            break
    return {
        "First citing year": first,
        "Last citing year": last,
        "Span": span,
        "N citing years": int((full > 0).sum()),
        "Total citations": int(total),
        "Peak year": peak_year,
        "Peak citations": int(peak),
        "Mean per year": mean,
        "CV": cv,
        "Dormancy years": int(dormancy),
        "_full": full,
    }


def classify_reference_patterns(
    ts: pd.DataFrame,
    min_years: int = 3,
    current_year: Optional[int] = None,
    hot_recent_window: int = 3,
    hot_recent_share: float = 0.6,
    constant_cv_max: float = 0.6,
    constant_min_span: int = 8,
    sb_dormancy_years: int = 5,
    sb_awake_ratio: float = 3.0,
) -> pd.DataFrame:
    """
    Classify the citation history of each cited reference.

    Input is the long table of :func:`cited_reference_timeseries`. For each
    reference, the citation series over its span (first to last citing year,
    missing years filled with 0) is summarised and labelled with one of the
    following patterns. The rules are applied **in this order** and the first
    matching rule wins:

    0. ``"Insufficient data"`` -- fewer than ``min_years`` distinct citing years.
    1. ``"Hot Paper"`` -- share of citations received in the last
       ``hot_recent_window`` years (years ``current_year - window + 1`` to
       ``current_year``) is >= ``hot_recent_share`` **and** the last citing year
       is >= ``current_year - 1``.
    2. ``"Sleeping Beauty"`` -- ``Dormancy years`` (consecutive years with <= 1
       citation from the first citing year, before the first year with >= 2
       citations) >= ``sb_dormancy_years`` **and** the reference awoke (there is
       a year with >= 2 citations) **and** the mean citations per year after
       awakening is >= ``sb_awake_ratio`` x the mean during dormancy, where the
       dormancy mean is floored at 0.5 to avoid division by zero.
    3. ``"Constant Performer"`` -- ``Span`` >= ``constant_min_span`` **and**
       ``CV`` (population std / mean over all years of the span, zeros
       included) <= ``constant_cv_max``.
    4. ``"Life Cycle"`` -- the peak year lies strictly inside the span (not
       the first nor the last citing year) **and** the mean citations of the
       last two years of the span is <= 50 % of the peak citations.
    5. ``"Other"`` -- none of the above.

    Parameters
    ----------
    ts : pd.DataFrame
        Long table with ``Reference``, ``Citing Year``, ``Citations`` and
        optionally ``Cited Year``.
    min_years : int
        Minimum number of distinct citing years for classification.
    current_year : int, optional
        Reference year for the "recent" window; the maximum citing year in
        ``ts`` if None.
    hot_recent_window, hot_recent_share, constant_cv_max, constant_min_span,
    sb_dormancy_years, sb_awake_ratio
        Rule thresholds (see above).

    Returns
    -------
    pd.DataFrame
        One row per reference with columns ``Reference``, ``Cited Year``,
        ``First citing year``, ``Last citing year``, ``Span``,
        ``N citing years``, ``Total citations``, ``Peak year``,
        ``Peak citations``, ``Mean per year``, ``CV``, ``Recent share``
        (share of citations in the last ``hot_recent_window`` years),
        ``Dormancy years``, ``Pattern``. Sorted by ``Total citations``
        descending.
    """
    cols = [
        "Reference", "Cited Year", "First citing year", "Last citing year", "Span",
        "N citing years", "Total citations", "Peak year", "Peak citations",
        "Mean per year", "CV", "Recent share", "Dormancy years", "Pattern",
    ]
    if ts is None or ts.empty:
        return pd.DataFrame(columns=cols)
    ts = ts.copy()
    ts["Citing Year"] = _to_numeric(ts["Citing Year"])
    ts["Citations"] = _to_numeric(ts["Citations"]).fillna(0)
    ts = ts.dropna(subset=["Citing Year"])
    ts["Citing Year"] = ts["Citing Year"].astype(int)
    if ts.empty:
        return pd.DataFrame(columns=cols)
    if current_year is None:
        current_year = int(ts["Citing Year"].max())
    current_year = int(current_year)
    has_cited_year = "Cited Year" in ts.columns

    rows = []
    for ref, sub in ts.groupby("Reference", sort=True):
        if (sub["Citations"] > 0).sum() == 0:
            continue
        st = _series_stats(sub)
        full: pd.Series = st.pop("_full")
        total = st["Total citations"]
        recent_lo = current_year - hot_recent_window + 1
        recent = float(full.loc[(full.index >= recent_lo) & (full.index <= current_year)].sum())
        recent_share = recent / total if total > 0 else np.nan

        if st["N citing years"] < min_years:
            pattern = "Insufficient data"
        elif recent_share >= hot_recent_share and st["Last citing year"] >= current_year - 1:
            pattern = "Hot Paper"
        elif _is_sleeping_beauty(full, st["Dormancy years"], sb_dormancy_years, sb_awake_ratio):
            pattern = "Sleeping Beauty"
        elif st["Span"] >= constant_min_span and pd.notna(st["CV"]) and st["CV"] <= constant_cv_max:
            pattern = "Constant Performer"
        elif _is_life_cycle(full, st):
            pattern = "Life Cycle"
        else:
            pattern = "Other"

        row = {"Reference": ref}
        if has_cited_year:
            cy = sub["Cited Year"].dropna()
            row["Cited Year"] = float(cy.iloc[0]) if len(cy) else np.nan
        else:
            row["Cited Year"] = _cited_year(ref)
        row.update(st)
        row["Recent share"] = recent_share
        row["Pattern"] = pattern
        rows.append(row)

    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out = out.sort_values(["Total citations", "Reference"], ascending=[False, True],
                          kind="mergesort").reset_index(drop=True)
    return out


def _is_sleeping_beauty(full: pd.Series, dormancy: int, min_dormancy: int, ratio: float) -> bool:
    if dormancy < min_dormancy or dormancy >= len(full):
        return False
    vals = full.values.astype(float)
    dorm_mean = max(float(vals[:dormancy].mean()), 0.5)
    awake_mean = float(vals[dormancy:].mean())
    return awake_mean >= ratio * dorm_mean


def _is_life_cycle(full: pd.Series, st: Dict[str, float]) -> bool:
    first, last, peak_year = st["First citing year"], st["Last citing year"], st["Peak year"]
    if not (first < peak_year < last):
        return False
    tail = float(full.values[-2:].mean())
    return tail <= 0.5 * st["Peak citations"]


# ---------------------------------------------------------------------------
# 4) Summary per pattern
# ---------------------------------------------------------------------------

def rpys_pattern_summary(classified: pd.DataFrame) -> pd.DataFrame:
    """
    Count, share and top references per citation pattern.

    Parameters
    ----------
    classified : pd.DataFrame
        Output of :func:`classify_reference_patterns`.

    Returns
    -------
    pd.DataFrame
        Columns ``Pattern``, ``N``, ``%`` and ``Top references`` (the three
        references with most total citations in the pattern, joined by
        ``'; '``). Patterns are listed in the canonical order
        (Hot Paper, Sleeping Beauty, Constant Performer, Life Cycle, Other,
        Insufficient data); patterns without references are omitted.
    """
    cols = ["Pattern", "N", "%", "Top references"]
    if classified is None or classified.empty:
        return pd.DataFrame(columns=cols)
    n_all = len(classified)
    rows = []
    present = list(classified["Pattern"].unique())
    order = [p for p in PATTERN_ORDER if p in present] + sorted(p for p in present if p not in PATTERN_ORDER)
    for pat in order:
        sub = classified[classified["Pattern"] == pat]
        top = sub.sort_values(["Total citations", "Reference"], ascending=[False, True], kind="mergesort")
        rows.append({
            "Pattern": pat,
            "N": int(len(sub)),
            "%": 100.0 * len(sub) / n_all,
            "Top references": "; ".join(map(str, top["Reference"].head(3))),
        })
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# 5) Plots
# ---------------------------------------------------------------------------

def plot_rpys_extended(
    rpys_df: pd.DataFrame,
    ax=None,
    title: Optional[str] = "Reference Publication Year Spectroscopy",
    bar_color: str = "#9ecae1",
    median_color: str = "#08519c",
    deviation_color: str = "#e6550d",
    highlight_years: Optional[Sequence[int]] = None,
    figsize: Tuple[float, float] = (10, 5),
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    """
    Extended RPYS plot: bars for citations, line for the rolling median and the
    deviation (Citations - Median) as a filled area on a twin y-axis.

    Parameters
    ----------
    rpys_df : pd.DataFrame
        Output of :func:`compute_rpys_backward` or
        :func:`biblium.utilsbib.compute_rpys`.
    highlight_years : sequence of int, optional
        Cited years to mark with a vertical dashed line (e.g. RPYS peaks).

    Returns
    -------
    (fig, ax) -- ``ax`` is the primary (citations) axis; the twin axis is
    available as ``ax.deviation_ax``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if rpys_df is None or rpys_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return _finish(fig, ax, title, f_name, dpi, show)

    x = rpys_df["Cited Year"].to_numpy()
    ax.bar(x, rpys_df["Citations"].to_numpy(), color=bar_color, width=0.8, label="Citations")
    ax.plot(x, rpys_df["Median"].to_numpy(), color=median_color, lw=1.6, label="Median")
    ax.set_xlabel("Cited year")
    ax.set_ylabel("Number of cited references")

    ax2 = ax.twinx()
    dev = rpys_df["Deviation"].to_numpy(dtype=float)
    ax2.fill_between(x, 0, dev, color=deviation_color, alpha=0.25, label="Deviation")
    ax2.plot(x, dev, color=deviation_color, lw=0.8, alpha=0.8)
    ax2.axhline(0, color=deviation_color, lw=0.5, alpha=0.5)
    ax2.set_ylabel("Deviation from median")
    ax.deviation_ax = ax2

    if highlight_years:
        for y in highlight_years:
            ax.axvline(y, color="grey", ls="--", lw=0.8)
            ax.annotate(str(int(y)), (y, ax.get_ylim()[1]), ha="center", va="bottom", fontsize=8)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    return _finish(fig, ax, title, f_name, dpi, show)


def plot_reference_patterns(
    classified: pd.DataFrame,
    ts: pd.DataFrame,
    top_n: int = 8,
    ax=None,
    title: Optional[str] = "Citation histories of top cited references",
    cmap: str = "tab10",
    label_len: int = 40,
    figsize: Tuple[float, float] = (10, 6),
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    """
    Citation curves (citing year vs citations) of the ``top_n`` references by
    total citations on a single axis, coloured by their pattern.

    Parameters
    ----------
    classified : pd.DataFrame
        Output of :func:`classify_reference_patterns`.
    ts : pd.DataFrame
        Output of :func:`cited_reference_timeseries` (the same table used for
        classification).
    top_n : int
        Number of references drawn.
    label_len : int
        Maximum length of the reference label in the legend.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if classified is None or classified.empty or ts is None or ts.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return _finish(fig, ax, title, f_name, dpi, show)

    top = classified.sort_values(["Total citations", "Reference"], ascending=[False, True],
                                 kind="mergesort").head(top_n)
    palette = plt.get_cmap(cmap)
    colors = {p: palette(i % palette.N) for i, p in enumerate(PATTERN_ORDER)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    for i, (_, row) in enumerate(top.iterrows()):
        sub = ts[ts["Reference"] == row["Reference"]]
        g = sub.groupby("Citing Year")["Citations"].sum()
        first, last = int(g.index.min()), int(g.index.max())
        full = pd.Series(0, index=range(first, last + 1), dtype=float)
        full.loc[g.index] = g.values.astype(float)
        lab = str(row["Reference"])
        if len(lab) > label_len:
            lab = lab[: label_len - 1] + "…"
        ax.plot(full.index, full.values, marker=markers[i % len(markers)], ms=4, lw=1.3,
                color=colors.get(row["Pattern"], "grey"), label=lab)

    ax.set_xlabel("Citing year")
    ax.set_ylabel("Citations in corpus")
    ax.xaxis.get_major_locator().set_params(integer=True)
    leg1 = ax.legend(loc="upper left", fontsize=7, frameon=False, title="Reference")
    ax.add_artist(leg1)
    present = [p for p in PATTERN_ORDER if p in set(top["Pattern"])]
    handles = [Line2D([0], [0], color=colors[p], lw=3) for p in present]
    ax.legend(handles, present, loc="upper right", fontsize=8, frameon=False, title="Pattern")
    return _finish(fig, ax, title, f_name, dpi, show)
