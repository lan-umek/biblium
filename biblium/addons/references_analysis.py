# -*- coding: utf-8 -*-
"""
References analysis addon -- paper-level bibliographic coupling and reference
log-log scatters.

Functions
---------
- build_paper_bibliographic_coupling(df, refs_col, id_col, min_shared, top_n,
                                       weight_col)
    Build a NetworkX graph of papers connected by shared references.
    Reviewer feedback D1: port the inline pipeline implementation into
    biblium so future pipelines just call this.
- plot_references_year_citations(ref_df, year_col, citations_col, label_col,
                                   color_col, out_path, ...)
    log-log scatter of references (top-cited classics): x = publication
    year, y = times cited (within corpus), size optionally a third metric.
    Reviewer feedback D2: a recurring view that 20_entity_scatters builds
    inline -- centralise it here.

Both helpers follow biblium scatter style v2 (no heavy borders, color
encodes data, no truncated labels).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

try:  # optional
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None  # type: ignore


def _split_refs(s: Union[str, float], sep: str = "|") -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    parts = re.split(r"[|;]\s*", s)
    return [p.strip() for p in parts if p.strip()]


def build_paper_bibliographic_coupling(
    df: pd.DataFrame,
    refs_col: str = "oa_referenced_works",
    id_col: str = "oa_openalex_id",
    min_shared: int = 5,
    top_n: int = 200,
    excluded_refs: Optional[Iterable[str]] = None,
):
    """Build a NetworkX graph of paper bibliographic coupling.

    Parameters
    ----------
    df : DataFrame
        Source corpus. Must contain ``refs_col`` (pipe- or semicolon-separated
        reference IDs) and ``id_col`` (unique paper id).
    min_shared : int
        Minimum number of shared references for an edge.
    top_n : int
        Restrict to the top ``top_n`` most-cited papers (by ``Cited by`` or
        ``oa_cited_by_count``) for tractability.
    excluded_refs : iterable, optional
        Reference IDs to drop from the coupling computation (e.g. the seed
        work that every paper cites).

    Returns
    -------
    networkx.Graph
    """
    if nx is None:  # pragma: no cover
        raise RuntimeError("networkx is required for bibliographic coupling.")
    if refs_col not in df.columns or id_col not in df.columns:
        raise ValueError(f"missing column(s): {refs_col!r} / {id_col!r}")

    cite_col = None
    for c in ("oa_cited_by_count", "Cited by", "citations"):
        if c in df.columns:
            cite_col = c
            break

    work = df.copy()
    work["__refs"] = work[refs_col].apply(_split_refs)
    work = work[work["__refs"].str.len() >= 5]
    if cite_col is not None:
        work["__cit"] = pd.to_numeric(work[cite_col], errors="coerce").fillna(0)
    else:
        work["__cit"] = 0
    work = work.sort_values("__cit", ascending=False).head(top_n)

    excluded = set(excluded_refs or [])
    refs_by_id = {
        str(pid): (set(rs) - excluded)
        for pid, rs in zip(work[id_col].astype(str), work["__refs"])
    }

    G = nx.Graph()
    for _, row in work.iterrows():
        pid = str(row[id_col])
        G.add_node(
            pid,
            title=str(row.get("Title", ""))[:120],
            year=row.get("Year"),
            citations=int(row.get("__cit", 0)),
        )
    paper_ids = list(refs_by_id.keys())
    for i, p1 in enumerate(paper_ids):
        s1 = refs_by_id[p1]
        if not s1:
            continue
        for p2 in paper_ids[i + 1:]:
            shared = len(s1 & refs_by_id[p2])
            if shared >= min_shared:
                G.add_edge(p1, p2, weight=shared)
    return G


def plot_references_year_citations(
    ref_df: pd.DataFrame,
    year_col: str = "year",
    citations_col: str = "citations_in_corpus",
    label_col: Optional[str] = "first_author",
    color_col: Optional[str] = None,
    out_path: Union[str, Path] = "references_year_citations.png",
    title: str = "Top references -- year vs corpus citations",
    figsize: tuple = (12, 8),
    n_labels: int = 25,
) -> None:
    """log-log scatter of references: x=year, y=citations in corpus.

    Color optionally encodes a third dimension (e.g. average year of
    citing papers). Marker style follows biblium scatter v2 (thin white
    edge). Labels textwrap, never truncate.
    """
    import textwrap as _tw
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = ref_df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[citations_col] = pd.to_numeric(df[citations_col], errors="coerce")
    df = df.dropna(subset=[year_col, citations_col])
    df = df[df[citations_col] > 0]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)

    color_kwargs: dict = {}
    sm = None
    if color_col and color_col in df.columns:
        vals = pd.to_numeric(df[color_col], errors="coerce")
        if vals.notna().any():
            color_kwargs = dict(
                c=vals.fillna(vals.median()), cmap="viridis",
            )
            sm = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(vmin=float(vals.min()), vmax=float(vals.max())),
            )
            sm.set_array([])
    if not color_kwargs:
        color_kwargs = dict(c="#1f3a93")

    sc = ax.scatter(
        df[year_col],
        df[citations_col],
        s=60,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        **color_kwargs,
    )
    if sm is not None:
        cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(color_col)

    ax.set_yscale("log")
    ax.set_xlabel(year_col)
    ax.set_ylabel(f"{citations_col} (log)")
    ax.set_title(title)

    if label_col and label_col in df.columns:
        top = df.nlargest(n_labels, citations_col)
        for _, r in top.iterrows():
            lab = _tw.fill(str(r[label_col]), width=22)
            t = ax.text(r[year_col], r[citations_col], lab,
                        fontsize=7, ha="left", va="bottom")
            t.set_path_effects([
                patheffects.withStroke(linewidth=2.0, foreground="white"),
            ])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "build_paper_bibliographic_coupling",
    "plot_references_year_citations",
]
