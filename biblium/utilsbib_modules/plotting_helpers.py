"""
plotting_helpers.py - generic plotting primitives for biblium
==============================================================

Reusable plot helpers used across biblium addons. These functions
encapsulate the project-wide style rules:

- NEVER truncate labels (always textwrap.fill -> multiline)
- single-series bars use one color (NAVY) - color encodes data
- no grid lines (ax.grid(False) or ax.set_axis_off())
- network plots: kk for dense small graphs, spring for sparse/large;
  Louvain community coloring; sqrt size scaling; white-stroke label
  outlines; drop isolates; crop empty whitespace

These are stateless helpers - callers pass matrices, graphs, partitions
and the function does one job (plot + save).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform


# Project default NAVY (matches biblium plot_config first categorical color).
# Kept here as a safe fallback so this module can be imported without the
# wider biblium config being initialised.
_NAVY_FALLBACK = "#1F3864"


def wrap_label(s: Any, width: int = 20) -> str:
    """Wrap a long label onto multiple lines. Never truncate.

    Replaces " and " -> " & " for compactness, then uses textwrap.wrap.
    If after preprocessing the string is shorter than width, returned as-is.
    """
    s = str(s).replace(" and ", " & ")
    if len(s) <= width:
        return s
    return "\n".join(textwrap.wrap(s, width=width)) or s


def _resolve_navy() -> str:
    """Return NAVY from biblium plot_config if available, else fallback."""
    try:
        from biblium.config import plot_config  # type: ignore
        palette = list(plot_config.categorical_palette)
        if palette:
            return palette[0]
    except Exception:
        pass
    return _NAVY_FALLBACK


def plot_jaccard_heatmap(
    matrix: pd.DataFrame,
    out: Path,
    title: str,
    order: list | None = None,
    cmap: str = "viridis",
    figsize: tuple | None = None,
    cbar_label: str = "Jaccard similarity",
    triangular: bool = True,
) -> None:
    """Heatmap of a symmetric similarity matrix.

    - diagonal -> NaN (so it doesn't dominate the colormap)
    - wrapped tick labels (never truncated)
    - viridis cmap, no grid
    - optional row/col re-ordering
    - triangular=True (default) hides the upper triangle + diagonal so the
      symmetric matrix is shown as a lower-triangle only (reviewer feedback,
      avoids redundant information).
    """
    mat = matrix.copy()
    if order is not None:
        mat = mat.loc[order, order]
    n = len(mat)
    if figsize is None:
        figsize = (max(8, n * 0.45), max(7, n * 0.42))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    arr = mat.values.astype(float).copy()
    np.fill_diagonal(arr, np.nan)
    if triangular:
        # Hide the upper triangle (k=0 includes the diagonal)
        iu = np.triu_indices_from(arr, k=0)
        arr[iu] = np.nan
    vmax = np.nanmax(arr) if np.isfinite(np.nanmax(arr)) else 1
    if not vmax or vmax <= 0:
        vmax = 1
    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [wrap_label(s, 18) for s in mat.columns],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels([wrap_label(s, 18) for s in mat.index], fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, fontsize=9)
    ax.set_title(title)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_clustermap_hierarchical(
    matrix: pd.DataFrame,
    out: Path,
    title: str,
    method: str = "ward",
    cmap: str = "viridis",
    cbar_label: str = "Jaccard similarity",
) -> list:
    """Hierarchical clustermap: top dendrogram + heatmap with y-labels on the right.

    Default linkage method is Ward (reviewer feedback). Layout is redesigned
    so dendrograms never overlap row/column labels:
      - top dendrogram lives in its own row above the heatmap
      - y-axis tick labels are moved to the RIGHT side of the heatmap, so
        nothing competes for left-side space
      - no left dendrogram is drawn (top dendrogram already shows the same
        linkage; the left one routinely overlapped left tick labels)

    Returns the leaf order used for re-ordering rows/columns.
    """
    mat = matrix.copy()
    arr = mat.values.astype(float)
    np.fill_diagonal(arr, 1.0)
    dist = 1.0 - arr
    dist = np.clip(dist, 0, 2)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    try:
        Z = sch.linkage(condensed, method=method)
    except Exception:
        # Ward needs Euclidean-like distances; if scipy refuses, fall back.
        Z = sch.linkage(condensed, method="average")
    order_idx = sch.leaves_list(Z)
    order = [mat.index[i] for i in order_idx]
    mat_o = mat.loc[order, order]

    navy = _resolve_navy()

    n = len(mat)
    fig = plt.figure(
        figsize=(max(12, n * 0.45), max(10, n * 0.42))
    )
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1, 5], hspace=0.04,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])

    sch.dendrogram(
        Z, ax=ax_top, orientation="top", no_labels=True,
        color_threshold=0, above_threshold_color=navy,
    )
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)

    arr_o = mat_o.values.astype(float).copy()
    np.fill_diagonal(arr_o, np.nan)
    vmax = np.nanmax(arr_o) if np.isfinite(np.nanmax(arr_o)) else 1
    if not vmax or vmax <= 0:
        vmax = 1
    im = ax_main.imshow(arr_o, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax_main.set_xticks(range(len(mat_o)))
    ax_main.set_xticklabels(
        [wrap_label(s, 16) for s in mat_o.columns],
        rotation=45, ha="right", fontsize=7,
    )
    ax_main.set_yticks(range(len(mat_o)))
    ax_main.set_yticklabels(
        [wrap_label(s, 16) for s in mat_o.index], fontsize=7,
    )
    # Move y-axis tick labels to the right so the dendrogram column to the
    # left is uncontested.
    ax_main.yaxis.tick_right()
    ax_main.yaxis.set_label_position("right")
    ax_main.grid(False)
    cb = fig.colorbar(
        im, ax=ax_main, fraction=0.03, pad=0.10, location="left",
    )
    cb.set_label(cbar_label, fontsize=9)
    fig.suptitle(f"{title}  (linkage: {method})", y=0.995, fontsize=11)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return order


def plot_network_auto(
    G: nx.Graph,
    partition: dict,
    out: Path,
    title: str,
    node_size_attr: str | None = None,
    layout: str = "auto",
    seed: int = 2026,
    edge_alpha: float = 0.35,
    max_labels: int | None = None,
    drop_isolates: bool = True,
    cmap_communities: str = "tab20",
) -> None:
    """Plot a network with Louvain community colors, sqrt-scaled nodes.

    - layout="auto" picks kk for dense small graphs, spring for sparse/big
    - sqrt scaling so node size differences remain readable
    - labels: all if <= max_labels, else top by size
    - white-stroke path effect on labels for readability
    - communities legend outside the plot
    - drop isolates by default (no orphan dots padding the canvas)
    - crops empty whitespace using actual node positions
    """
    if len(G.nodes) == 0:
        return

    G_plot = G.copy() if drop_isolates else G
    if drop_isolates:
        iso = list(nx.isolates(G_plot))
        if iso:
            G_plot.remove_nodes_from(iso)
        if len(G_plot.nodes) == 0:
            return

    n_nodes = len(G_plot.nodes)
    if layout == "auto":
        is_sparse = len(G_plot.edges) < n_nodes * 1.5
        layout_use = "spring" if (n_nodes > 60 or is_sparse) else "kk"
    else:
        layout_use = layout

    if layout_use == "spring":
        pos = nx.spring_layout(
            G_plot, seed=seed,
            k=3.5 / np.sqrt(max(n_nodes, 1)),
            iterations=200,
            weight="weight",
        )
    elif layout_use == "kk":
        try:
            pos = nx.kamada_kawai_layout(G_plot, weight="weight")
        except Exception:
            pos = nx.spring_layout(G_plot, seed=seed)
    else:
        pos = nx.spring_layout(G_plot, seed=seed)

    comms = sorted(set(partition.get(n, 0) for n in G_plot.nodes()))
    cmap = plt.get_cmap(cmap_communities, max(len(comms), 1))
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(comms)}
    node_colors = [color_map[partition.get(n, 0)] for n in G_plot.nodes()]

    if node_size_attr:
        sizes_raw = np.array(
            [G_plot.nodes[n].get(node_size_attr, 1) for n in G_plot.nodes()],
            dtype=float,
        )
        max_s = sizes_raw.max() if sizes_raw.max() > 0 else 1
        sizes = 150 + 1800 * np.sqrt(sizes_raw / max_s)
    else:
        sizes = np.full(n_nodes, 350.0)

    fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)

    weights = np.array(
        [d.get("weight", 1) for _, _, d in G_plot.edges(data=True)]
    )
    if len(weights) and weights.max() > 0:
        ew = 0.4 + 4 * (weights / weights.max())
    else:
        ew = 0.5

    nx.draw_networkx_edges(
        G_plot, pos, ax=ax, width=ew, edge_color="#BBBBBB", alpha=edge_alpha,
    )
    nx.draw_networkx_nodes(
        G_plot, pos, ax=ax, node_size=sizes, node_color=node_colors,
        edgecolors="white", linewidths=0.8, alpha=0.92,
    )

    if max_labels is None:
        max_labels = min(20, n_nodes)
    if n_nodes <= max_labels:
        label_nodes = list(G_plot.nodes())
    else:
        nodes_sorted = sorted(
            zip(G_plot.nodes(), sizes), key=lambda x: -x[1],
        )[:max_labels]
        label_nodes = [n for n, _ in nodes_sorted]
    labels = {n: wrap_label(n, 16) for n in label_nodes}
    text_objs = nx.draw_networkx_labels(
        G_plot, pos, labels, ax=ax, font_size=8,
    )
    for t in text_objs.values():
        t.set_path_effects([
            patheffects.withStroke(linewidth=2.5, foreground="white")
        ])

    ax.set_axis_off()
    ax.set_title(title, fontsize=12)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        pad_x = (max(xs) - min(xs)) * 0.1 if len(xs) > 1 else 0.1
        pad_y = (max(ys) - min(ys)) * 0.1 if len(ys) > 1 else 0.1
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    handles = [
        Patch(color=color_map[c], label=f"Community {c}") for c in comms[:10]
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0),
        frameon=False, fontsize=8, title="Louvain", title_fontsize=9,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "wrap_label",
    "plot_jaccard_heatmap",
    "plot_clustermap_hierarchical",
    "plot_network_auto",
]
