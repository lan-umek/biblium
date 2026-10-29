"""
discipline_analysis.py - high-level methods for knowledge flows
=================================================================

Implements three analyses used by the "21 knowledge flows" workflow:

A) Corpus discipline profile (counts + % per OpenAlex domain/field/
   subfield/topic, plus single-color horizontal bar plots).

B) Concept x field heatmap: for each user concept (regex pattern set
   per concept), what fraction of its papers fall in each top OpenAlex
   field.

C) Field/subfield dynamics over time: top-N entity raw counts and
   per-year shares as stacked-area charts.

The high-level functions return DataFrames/dicts so callers can save
their own custom outputs. Companion plot helpers do the final
visualization following project style rules (single NAVY color for
single-series bars, no grids, wrapped labels, no truncation).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biblium.utilsbib_modules.plotting_helpers import wrap_label


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _explode_counter(df: pd.DataFrame, col: str, sep: str) -> Counter:
    c = Counter()
    if col not in df.columns:
        return c
    for v in df[col].dropna():
        if isinstance(v, str):
            for x in v.split(sep):
                x = x.strip()
                if x:
                    c[x] += 1
    return c


def _per_doc_membership(df: pd.DataFrame, col: str, sep: str) -> dict:
    out = defaultdict(set)
    if col not in df.columns:
        return out
    for idx, v in df[col].items():
        if isinstance(v, str):
            for x in v.split(sep):
                x = x.strip()
                if x:
                    out[x].add(idx)
    return out


def _resolve_navy() -> str:
    try:
        from biblium.config import plot_config  # type: ignore
        palette = list(plot_config.categorical_palette)
        if palette:
            return palette[0]
    except Exception:
        pass
    return "#1F3864"


# ---------------------------------------------------------------------------
# Concept mask builder (regex with wildcards)
# ---------------------------------------------------------------------------
def build_concept_mask(
    df: pd.DataFrame,
    concepts_path_or_df,
    text_column: str = "Processed Combined Text",
) -> pd.DataFrame:
    """Build a boolean DataFrame (rows=df.index, cols=concept).

    ``concepts_path_or_df`` may be a path to an Excel file or a
    pre-loaded DataFrame whose columns are concept names and whose
    cells contain pattern strings (each cell = one pattern; '*' = \w*
    wildcard).
    """
    if isinstance(concepts_path_or_df, pd.DataFrame):
        cn = concepts_path_or_df
    else:
        cn = pd.read_excel(concepts_path_or_df)

    text = df[text_column].fillna("").astype(str).str.lower()
    out = pd.DataFrame(index=df.index)
    for c in cn.columns:
        pats = [str(p).lower().strip() for p in cn[c].dropna()
                if str(p).strip()]
        parts = []
        for p in pats:
            p2 = re.escape(p).replace(r"\*", r"\w*")
            parts.append(
                rf"\b{p2}\b" if r"\w*" not in p2 else rf"\b{p2}"
            )
        rx = "|".join(parts) if parts else None
        out[c] = (
            text.str.contains(rx, regex=True, case=False, na=False)
            if rx else False
        )
    return out


# ---------------------------------------------------------------------------
# A) Corpus discipline profile
# ---------------------------------------------------------------------------
def analyze_corpus_disciplines(
    df: pd.DataFrame,
    columns_seps: Iterable[tuple] = (
        ("oa_domains", "; "),
        ("oa_fields", "; "),
        ("oa_subfields", "; "),
        ("oa_topics", "|"),
    ),
) -> dict:
    """Compute discipline profiles for one or more multivalue columns.

    Returns ``{column_name -> DataFrame(entity, n_papers, pct_of_corpus)}``
    sorted descending by ``n_papers``.
    """
    n = len(df)
    out = {}
    for col, sep in columns_seps:
        if col not in df.columns:
            continue
        c = _explode_counter(df, col, sep)
        if not c:
            out[col] = pd.DataFrame(
                columns=[col, "n_papers", "pct_of_corpus"]
            )
            continue
        entries = c.most_common()
        rows = pd.DataFrame(entries, columns=[col, "n_papers"])
        rows["pct_of_corpus"] = (rows["n_papers"] / max(n, 1) * 100).round(2)
        out[col] = rows
    return out


def plot_discipline_bars(
    profiles: dict,
    out_folder: Path,
    top_ns: dict | None = None,
    single_color: bool = True,
    file_prefix: str = "A_",
) -> dict:
    """Horizontal bar plot per entity category. One NAVY color per chart.

    ``profiles`` is the dict returned by ``analyze_corpus_disciplines``.
    ``top_ns`` overrides per-column top-N (default: domains all, fields 15,
    subfields 25, topics 25).
    Returns dict {column_name -> Path of saved plot}.
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    defaults = {
        "oa_domains": None,    # show all
        "oa_fields": 15,
        "oa_subfields": 25,
        "oa_topics": 25,
    }
    top_ns = {**defaults, **(top_ns or {})}
    color = _resolve_navy() if single_color else None

    saved = {}
    for col, df_p in profiles.items():
        if df_p is None or len(df_p) == 0:
            continue
        top_n = top_ns.get(col)
        df_top = df_p.head(top_n) if top_n else df_p
        df_top = df_top.iloc[::-1]  # for horizontal bar: smallest at top
        n_rows = len(df_top)
        fig, ax = plt.subplots(
            figsize=(10, max(3.5, 0.45 * n_rows + 1)),
            constrained_layout=True,
        )
        labels = [wrap_label(s, 28) for s in df_top[col]]
        ax.barh(
            labels, df_top["pct_of_corpus"],
            color=color if single_color else None,
            edgecolor="white", linewidth=0.5,
        )
        ax.set_xlabel("% of corpus (papers can be in multiple entities)")
        title_suffix = f"top {top_n}" if top_n else "all"
        ax.set_title(f"OpenAlex {col.replace('oa_', '')} ({title_suffix})")
        xmax = max(df_top["pct_of_corpus"]) * 1.18
        for i, (v, npap) in enumerate(zip(
                df_top["pct_of_corpus"], df_top["n_papers"])):
            ax.text(
                v + xmax * 0.005, i, f"{v:.1f}%  (n={npap:,})",
                va="center", fontsize=8,
            )
        ax.set_xlim(0, xmax)
        ax.grid(False)
        out_file = out_folder / f"{file_prefix}{col}_bar.png"
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved[col] = out_file
    return saved


# ---------------------------------------------------------------------------
# B) Concept x field heatmap
# ---------------------------------------------------------------------------
def compute_concept_field_membership(
    df: pd.DataFrame,
    concepts_dict_or_mask,
    multivalue_col: str = "oa_fields",
    sep: str = "; ",
    top_n: int = 15,
    min_papers_concept: int = 30,
    text_column: str = "Processed Combined Text",
):
    """Compute % of concept papers in each top field.

    ``concepts_dict_or_mask`` may be:
      - a boolean DataFrame indexed by df.index with concepts as columns
      - a DataFrame of patterns (columns = concept, cells = patterns)
      - a path-like to an Excel file with same pattern structure

    Returns (top_fields, heatmap_df) where
    ``heatmap_df`` has columns [concept, n_papers_concept, <top_field_1>, ...]
    """
    # Resolve concept mask
    if (isinstance(concepts_dict_or_mask, pd.DataFrame)
            and concepts_dict_or_mask.index.equals(df.index)
            and concepts_dict_or_mask.dtypes.apply(
                lambda d: d == bool).all()):
        cmask = concepts_dict_or_mask
    else:
        cmask = build_concept_mask(
            df, concepts_dict_or_mask, text_column=text_column,
        )

    fld_members = _per_doc_membership(df, multivalue_col, sep)
    counts = Counter({k: len(v) for k, v in fld_members.items()})
    top_fields = [k for k, _ in counts.most_common(top_n)]

    rows = []
    for c in cmask.columns:
        concept_docs = set(cmask.index[cmask[c].astype(bool)])
        # restrict to docs that are present in df (and have the column data)
        concept_docs &= set(df.index)
        if len(concept_docs) < min_papers_concept:
            continue
        row = {"concept": c, "n_papers_concept": len(concept_docs)}
        for f in top_fields:
            inter = len(concept_docs & fld_members[f])
            row[f] = 100 * inter / len(concept_docs)
        rows.append(row)
    return top_fields, pd.DataFrame(rows)


def plot_concept_field_heatmap(
    heatmap_df: pd.DataFrame,
    top_fields: list,
    out: Path,
    title: str,
    cmap: str = "viridis",
    annotate_min: float = 5.0,
) -> None:
    """Heatmap concepts x fields with % values annotated in cells.

    - rows ordered as in heatmap_df
    - viridis cmap, no grid
    - column labels wrapped (never truncated)
    - cells with value < annotate_min left blank for readability
    """
    if len(heatmap_df) == 0 or not top_fields:
        return
    mat = heatmap_df[top_fields].values.astype(float)
    labels_y = heatmap_df["concept"].tolist()
    labels_x = [wrap_label(f, 18) for f in top_fields]
    fig, ax = plt.subplots(
        figsize=(max(10, len(top_fields) * 0.65),
                 max(6, len(labels_y) * 0.5)),
        constrained_layout=True,
    )
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(top_fields)))
    ax.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v >= annotate_min:
                color = "white" if v > 55 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                         fontsize=7, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("% of concept papers in field", fontsize=9)
    ax.set_title(title)
    ax.grid(False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# C) Field/subfield dynamics over time
# ---------------------------------------------------------------------------
def field_dynamics_over_time(
    df: pd.DataFrame,
    year_col: str,
    multivalue_col: str = "oa_subfields",
    sep: str = "; ",
    top_n: int = 8,
    min_year_count: int = 5,
    year_min: int | None = None,
):
    """Build per-year x top-entity counts and per-year shares.

    Returns (mat_raw, mat_share). mat_raw values are absolute counts;
    mat_share is the per-year share (in %) within the top-N entities of
    that year.
    """
    if year_col not in df.columns or multivalue_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    work = df[[year_col, multivalue_col]].copy()
    work["_yr"] = pd.to_numeric(work[year_col], errors="coerce")
    work = work.dropna(subset=["_yr"]).copy()
    work["_yr"] = work["_yr"].astype(int)
    if year_min is not None:
        work = work[work["_yr"] >= year_min]
    if len(work) == 0:
        return pd.DataFrame(), pd.DataFrame()

    c = _explode_counter(work, multivalue_col, sep)
    top_entities = [k for k, _ in c.most_common(top_n)]
    top_set = set(top_entities)

    years = sorted(work["_yr"].unique())
    mat_raw = pd.DataFrame(0, index=years, columns=top_entities, dtype=int)
    year_totals = pd.Series(0, index=years, dtype=int)

    for yr, grp in work.groupby("_yr"):
        year_totals[yr] = len(grp)
        for v in grp[multivalue_col].dropna():
            if isinstance(v, str):
                for x in v.split(sep):
                    x = x.strip()
                    if x in top_set:
                        mat_raw.loc[yr, x] += 1

    valid = year_totals[year_totals >= min_year_count].index.tolist()
    mat_raw = mat_raw.loc[valid]
    year_totals = year_totals.loc[valid]
    if len(mat_raw) == 0:
        return mat_raw, pd.DataFrame()
    mat_share = mat_raw.div(year_totals, axis=0).fillna(0) * 100
    return mat_raw, mat_share


def plot_dynamics_stacked(
    mat_raw: pd.DataFrame,
    mat_share: pd.DataFrame,
    out_raw: Path,
    out_share: Path,
    title: str,
    cmap_categorical: str = "tab20",
) -> None:
    """Two stacked-area charts: raw counts + per-year share.

    Legend placed outside the axes, wrapped labels (no truncation).
    """
    if len(mat_raw) == 0:
        return
    cols = list(mat_raw.columns)
    cmap = plt.get_cmap(cmap_categorical, max(len(cols), 1))
    colors = [cmap(i % cmap.N) for i in range(len(cols))]
    short_labels = [wrap_label(s, 22) for s in cols]

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.stackplot(
        mat_raw.index, mat_raw[cols].T.values,
        labels=short_labels, colors=colors, alpha=0.85,
        edgecolor="white", linewidth=0.4,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers (in top entities)")
    ax.set_title(f"{title} - absolute counts")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=7,
        title="Entity", title_fontsize=8,
    )
    ax.set_xlim(min(mat_raw.index), max(mat_raw.index))
    ax.grid(False)
    Path(out_raw).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_raw, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if len(mat_share) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.stackplot(
        mat_share.index, mat_share[cols].T.values,
        labels=short_labels, colors=colors, alpha=0.85,
        edgecolor="white", linewidth=0.4,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("% of yearly papers in top entities")
    ax.set_title(f"{title} - relative share")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=7,
        title="Entity", title_fontsize=8,
    )
    ax.set_xlim(min(mat_share.index), max(mat_share.index))
    ax.set_ylim(0, max(mat_share.sum(axis=1).max() * 1.05, 1))
    ax.grid(False)
    fig.savefig(out_share, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "build_concept_mask",
    "analyze_corpus_disciplines",
    "plot_discipline_bars",
    "compute_concept_field_membership",
    "plot_concept_field_heatmap",
    "field_dynamics_over_time",
    "plot_dynamics_stacked",
]
