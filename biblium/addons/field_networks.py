"""
field_networks.py - co-occurrence networks for OpenAlex fields/subfields
==========================================================================

High-level "section" methods used by the "22 field networks" workflow.
Each section is a single function that:

  - takes a DataFrame and a few config knobs
  - calls biblium primitives (build_cooccurrence_matrix,
    normalize_symmetric_matrix, matrix_to_network, louvain_partition,
    disparity_filter_backbone, bridging_centralities)
  - saves the resulting tables (Excel) and plots (PNG) via plotting_helpers
  - returns the artifacts (matrices/graphs/partitions) for downstream use

Sections:
  A) cooccurrence_heatmap_section  - field x field Jaccard heatmap
  B) cooccurrence_clustermap_section - top-N entities clustermap
  C) cooccurrence_network_section  - Louvain network (jaccard + raw graphs)
  D) disparity_backbone_section    - Serrano-Boguna-Vespignani backbone
  E) bridging_section              - top bridging nodes (betweenness)
  F) temporal_pair_evolution_section - slope chart of pair-Jaccard evolution
  G) per_concept_subgraphs_section - per-concept subfield subgraphs
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from biblium.utilsbib_modules.network import (
    build_cooccurrence_matrix,
    normalize_symmetric_matrix,
    matrix_to_network,
    louvain_partition,
    disparity_filter_backbone,
    bridging_centralities,
)
from biblium.utilsbib_modules.plotting_helpers import (
    wrap_label,
    plot_jaccard_heatmap,
    plot_clustermap_hierarchical,
    plot_network_auto,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_navy() -> str:
    try:
        from biblium.config import plot_config  # type: ignore
        palette = list(plot_config.categorical_palette)
        if palette:
            return palette[0]
    except Exception:
        pass
    return "#1F3864"


def _restrict_topN(
    df: pd.DataFrame, col: str, sep: str, top_n: int,
) -> tuple[pd.DataFrame, list, str]:
    """Filter multivalue cells to the top-N most frequent values.

    Returns (df_out, top_values, new_column_name). Cells that contain
    none of the top values get None for the new column.
    """
    c = Counter()
    for v in df[col].dropna():
        if isinstance(v, str):
            for x in v.split(sep):
                x = x.strip()
                if x:
                    c[x] += 1
    top = [k for k, _ in c.most_common(top_n)]
    top_set = set(top)
    new_col = col + "_topN"

    def filt(v):
        if not isinstance(v, str):
            return None
        items = [x.strip() for x in v.split(sep) if x.strip() in top_set]
        return sep.join(items) if items else None

    out = df.copy()
    out[new_col] = out[col].apply(filt)
    return out, top, new_col


def _attach_node_papers(
    G: nx.Graph, df: pd.DataFrame, col: str, sep: str,
) -> None:
    """Set ``n_papers`` node attribute = doc count for each entity."""
    counts = (
        df[col].dropna()
        .apply(lambda v: [x.strip() for x in v.split(sep) if x.strip()])
        .explode().value_counts()
    )
    for n in G.nodes():
        G.nodes[n]["n_papers"] = int(counts.get(n, 0))


# ---------------------------------------------------------------------------
# A) Heatmap on all entities
# ---------------------------------------------------------------------------
def cooccurrence_heatmap_section(
    df: pd.DataFrame,
    col: str,
    sep: str,
    min_count: int,
    out_table_prefix: Path,
    out_plot: Path,
    title: str,
) -> pd.DataFrame:
    """Build cooccurrence + Jaccard matrices; save tables and a heatmap.

    Returns the Jaccard DataFrame.
    """
    mat = build_cooccurrence_matrix(df, col, sep=sep, min_count=min_count)
    jacc = normalize_symmetric_matrix(mat, method="jaccard")
    out_table_prefix = Path(out_table_prefix)
    out_table_prefix.parent.mkdir(parents=True, exist_ok=True)
    jacc.to_excel(out_table_prefix.with_suffix("").as_posix() + "_jaccard.xlsx")
    mat.to_excel(
        out_table_prefix.with_suffix("").as_posix() + "_cooccurrence_raw.xlsx"
    )
    plot_jaccard_heatmap(jacc, out_plot, title)
    return jacc


# ---------------------------------------------------------------------------
# B) Clustermap on top-N entities
# ---------------------------------------------------------------------------
def cooccurrence_clustermap_section(
    df: pd.DataFrame,
    col: str,
    sep: str,
    top_n: int,
    min_count: int,
    out_table_prefix: Path,
    out_plot: Path,
    title: str,
) -> tuple[pd.DataFrame, list]:
    """Build cooccurrence on top-N entities, draw hierarchical clustermap.

    Returns (jacc_matrix, leaf_order).
    """
    df_top, _, col_top = _restrict_topN(df, col, sep, top_n)
    mat = build_cooccurrence_matrix(
        df_top, col_top, sep=sep, min_count=min_count,
    )
    jacc = normalize_symmetric_matrix(mat, method="jaccard")
    out_table_prefix = Path(out_table_prefix)
    out_table_prefix.parent.mkdir(parents=True, exist_ok=True)
    jacc.to_excel(out_table_prefix.with_suffix("").as_posix() + "_jaccard.xlsx")
    mat.to_excel(
        out_table_prefix.with_suffix("").as_posix() + "_cooccurrence_raw.xlsx"
    )
    order = plot_clustermap_hierarchical(jacc, out_plot, title)
    return jacc, order


# ---------------------------------------------------------------------------
# C) Network (Louvain). Builds TWO graphs:
#     - G_jac   : weights = jaccard       (for layout / Louvain)
#     - G_raw   : weights = raw cooc count (for downstream D, E)
# ---------------------------------------------------------------------------
def cooccurrence_network_section(
    df: pd.DataFrame,
    col: str,
    sep: str,
    top_n: int,
    min_count: int,
    jaccard_threshold: float,
    out_table_prefix: Path,
    out_plot: Path,
    out_graphml: Path,
    title: str,
) -> tuple[nx.Graph, nx.Graph, dict]:
    """Build dual networks (jaccard + raw) and run Louvain on the jaccard graph.

    Returns (G_jac, G_raw, partition). G_jac has isolates removed (they
    pad the plot). G_raw keeps all nodes (D, E need them).
    """
    df_top, _, col_top = _restrict_topN(df, col, sep, top_n)
    mat_raw = build_cooccurrence_matrix(
        df_top, col_top, sep=sep, min_count=min_count,
    )
    jacc = normalize_symmetric_matrix(mat_raw, method="jaccard")

    out_table_prefix = Path(out_table_prefix)
    out_table_prefix.parent.mkdir(parents=True, exist_ok=True)
    jacc.to_excel(out_table_prefix.with_suffix("").as_posix() + "_jaccard.xlsx")
    mat_raw.to_excel(
        out_table_prefix.with_suffix("").as_posix() + "_cooccurrence_raw.xlsx"
    )

    mat_jac = jacc.copy()
    np.fill_diagonal(mat_jac.values, 0)
    G_jac = matrix_to_network(mat_jac, min_weight=jaccard_threshold)
    G_raw = matrix_to_network(mat_raw, min_weight=1)
    _attach_node_papers(G_jac, df, col, sep)
    _attach_node_papers(G_raw, df, col, sep)

    iso = list(nx.isolates(G_jac))
    if iso:
        G_jac.remove_nodes_from(iso)

    partition = louvain_partition(G_jac)

    part_df = pd.DataFrame(
        [(n, c, G_jac.nodes[n].get("n_papers", 0))
         for n, c in partition.items()],
        columns=["entity", "louvain_community", "n_papers"],
    ).sort_values(
        ["louvain_community", "n_papers"], ascending=[True, False]
    )
    part_df.to_excel(
        out_table_prefix.with_suffix("").as_posix() + "_louvain.xlsx",
        index=False,
    )

    out_graphml = Path(out_graphml)
    out_graphml.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G_jac, str(out_graphml))

    plot_network_auto(
        G_jac, partition, out_plot, title,
        node_size_attr="n_papers", layout="auto",
        max_labels=25, drop_isolates=False,  # already removed above
    )
    return G_jac, G_raw, partition


# ---------------------------------------------------------------------------
# D) Disparity-filter backbone
# ---------------------------------------------------------------------------
def disparity_backbone_section(
    G_raw: nx.Graph,
    alpha: float,
    out_table: Path,
    out_plot: Path,
    out_graphml: Path,
    title: str,
) -> tuple[nx.Graph, dict]:
    """Extract the disparity-filter backbone and re-run Louvain on it."""
    G_bb = disparity_filter_backbone(G_raw, alpha=alpha)
    for n in G_bb.nodes():
        G_bb.nodes[n]["n_papers"] = G_raw.nodes[n].get("n_papers", 0)
    isolates = list(nx.isolates(G_bb))
    G_bb.remove_nodes_from(isolates)
    if len(G_bb.nodes) == 0:
        return G_bb, {}

    partition = louvain_partition(G_bb)

    Path(out_graphml).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G_bb, str(out_graphml))
    bb_df = pd.DataFrame(
        [(n, c, G_bb.nodes[n].get("n_papers", 0))
         for n, c in partition.items()],
        columns=["entity", "louvain_community", "n_papers"],
    ).sort_values(
        ["louvain_community", "n_papers"], ascending=[True, False]
    )
    Path(out_table).parent.mkdir(parents=True, exist_ok=True)
    bb_df.to_excel(out_table, index=False)

    plot_network_auto(
        G_bb, partition, out_plot, title,
        node_size_attr="n_papers", layout="auto",
        edge_alpha=0.55, drop_isolates=False,
    )
    return G_bb, partition


# ---------------------------------------------------------------------------
# E) Bridging (betweenness)
# ---------------------------------------------------------------------------
def bridging_section(
    G_raw: nx.Graph,
    top_n: int,
    out_table: Path,
    out_plot: Path,
    title: str,
) -> pd.DataFrame:
    """Top bridging nodes by betweenness; saves NAVY horizontal bar chart."""
    br = bridging_centralities(G_raw, top_n=top_n)
    Path(out_table).parent.mkdir(parents=True, exist_ok=True)
    br.to_excel(out_table, index=False)
    if len(br) == 0:
        return br

    navy = _resolve_navy()
    top_plot = min(15, len(br))
    br_p = br.head(top_plot).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.barh(
        [wrap_label(s, 22) for s in br_p["node"]],
        br_p["betweenness"],
        color=navy, edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("Betweenness centrality")
    ax.set_title(title)
    for i, (v, br_score, deg) in enumerate(zip(
            br_p["betweenness"], br_p["bridging_score"], br_p["degree"])):
        ax.text(
            v + v * 0.02, i,
            f"  bs={br_score:.3f}  deg={deg}",
            va="center", fontsize=8,
        )
    ax.set_xlim(0, br_p["betweenness"].max() * 1.25)
    ax.grid(False)
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return br


# ---------------------------------------------------------------------------
# F) Temporal pair evolution (slope chart)
# ---------------------------------------------------------------------------
def temporal_pair_evolution_section(
    df: pd.DataFrame,
    col: str,
    sep: str,
    year_col: str,
    periods: Iterable[tuple],
    top_n_pairs: int,
    out_table: Path,
    out_plot: Path,
    title: str,
    top_n_entities: int = 40,
    min_papers_period: int = 30,
    cmap_categorical: str = "tab20",
) -> pd.DataFrame:
    """Evolve top entity-pair Jaccard scores across periods.

    ``periods`` is an iterable of (label, year_lo, year_hi).
    Returns the full evol_df sorted by delta descending.
    """
    work = df.copy()
    work["_yr"] = pd.to_numeric(work[year_col], errors="coerce")
    work = work.dropna(subset=["_yr"]).copy()
    work["_yr"] = work["_yr"].astype(int)

    df_top, _, col_top = _restrict_topN(work, col, sep, top_n_entities)

    period_mats = {}
    period_lbls = []
    for label, lo, hi in periods:
        mask = (df_top["_yr"] >= lo) & (df_top["_yr"] <= hi)
        sub_df = df_top[mask]
        if len(sub_df) < min_papers_period:
            continue
        mat_p = build_cooccurrence_matrix(
            sub_df, col_top, sep=sep, min_count=1,
        )
        jacc_p = normalize_symmetric_matrix(mat_p, method="jaccard")
        period_mats[label] = jacc_p
        period_lbls.append(label)

    if not period_mats:
        return pd.DataFrame()

    all_pairs = set()
    for label, mat in period_mats.items():
        pairs = []
        for i in range(len(mat)):
            for j in range(i + 1, len(mat)):
                val = mat.iloc[i, j]
                if val > 0:
                    pairs.append((mat.index[i], mat.columns[j], val))
        pairs.sort(key=lambda x: x[2], reverse=True)
        all_pairs.update((a, b) for a, b, _ in pairs[:top_n_pairs])

    evol_rows = []
    for a, b in all_pairs:
        row = {"entity_1": a, "entity_2": b}
        vals = []
        for label in period_lbls:
            mat = period_mats[label]
            try:
                v = float(mat.loc[a, b])
            except KeyError:
                v = 0.0
            row[label] = v
            vals.append(v)
        row["delta"] = vals[-1] - vals[0] if len(vals) >= 2 else 0
        evol_rows.append(row)
    evol_df = pd.DataFrame(evol_rows).sort_values(
        "delta", ascending=False,
    ).reset_index(drop=True)
    Path(out_table).parent.mkdir(parents=True, exist_ok=True)
    evol_df.to_excel(out_table, index=False)

    # Take top 10 most dynamic pairs (reviewer feedback: 20 was too crowded)
    top_change = evol_df.reindex(
        evol_df["delta"].abs().sort_values(ascending=False).head(10).index
    ).reset_index(drop=True)
    x_pos = list(range(len(period_lbls)))

    # Larger figure (was 12x8) and extra horizontal padding for labels
    fig, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)
    cmap = plt.get_cmap(cmap_categorical, max(len(top_change), 1))

    # Smart label staggering: collect (y, color, text) tuples for the last
    # period, then nudge overlapping labels vertically so they don't collide.
    # Two labels are considered "close" if |y_a - y_b| <= min_gap.
    label_specs = []
    for i, row in top_change.iterrows():
        y_vals = [row[lbl] for lbl in period_lbls]
        color = cmap(i % cmap.N)
        ax.plot(
            x_pos, y_vals, marker="o", color=color,
            alpha=0.85, linewidth=1.6, markersize=6,
        )
        text = (f"{wrap_label(str(row['entity_1']), 22)}"
                f" - {wrap_label(str(row['entity_2']), 22)}")
        label_specs.append([y_vals[-1], color, text])

    # Sort by y to spread labels: walk top-down, enforce min_gap separation
    if label_specs:
        ys = [s[0] for s in label_specs]
        span = (max(ys) - min(ys)) or max(ys) or 0.05
        min_gap = max(span * 0.045, 0.012)  # ~4.5% of y-range, never below 0.012
        order_ids = sorted(range(len(label_specs)), key=lambda k: label_specs[k][0])
        adj = [label_specs[k][0] for k in order_ids]
        for k in range(1, len(adj)):
            if adj[k] - adj[k - 1] < min_gap:
                adj[k] = adj[k - 1] + min_gap
        for new_y, k in zip(adj, order_ids):
            label_specs[k][0] = new_y

    pad_x = 0.12
    for y_label, color, text in label_specs:
        ax.text(
            x_pos[-1] + pad_x, y_label,
            f"  {text}",
            fontsize=8, va="center", color=color,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(period_lbls)
    ax.set_ylabel("Jaccard similarity")
    # Generous right-side padding so the wrapped labels fit comfortably.
    ax.set_xlim(-0.2, (len(x_pos) - 1) + 2.6)
    ax.set_title(title + "  (top 10 most dynamic pairs)")
    ax.grid(False)
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return evol_df


# ---------------------------------------------------------------------------
# G) Per-concept subgraphs
# ---------------------------------------------------------------------------
def per_concept_subgraphs_section(
    df: pd.DataFrame,
    concept_mask: pd.DataFrame,
    col: str,
    sep: str,
    top_n: int,
    min_papers_concept: int,
    min_cooc: int,
    out_plots_folder: Path,
    title_template: str,
) -> int:
    """For each concept (column in mask), plot a subfield co-occurrence subgraph.

    Returns the number of concept plots successfully written.
    """
    out_plots_folder = Path(out_plots_folder)
    out_plots_folder.mkdir(parents=True, exist_ok=True)
    cmask_local = concept_mask.loc[df.index]
    ok_count = 0
    for c in cmask_local.columns:
        docs = df.index[cmask_local[c].astype(bool)]
        if len(docs) < min_papers_concept:
            continue
        sub_part = df.loc[docs]
        sub_top, _, sub_col = _restrict_topN(sub_part, col, sep, top_n)
        if sub_top[sub_col].notna().sum() < 10:
            continue
        mat_c = build_cooccurrence_matrix(
            sub_top, sub_col, sep=sep, min_count=min_cooc,
        )
        if len(mat_c) < 3:
            continue
        G_c = matrix_to_network(mat_c, min_weight=0)
        ssp = mat_c.sum(axis=1)
        for n in G_c.nodes():
            G_c.nodes[n]["n_papers"] = int(ssp.get(n, 0))
        G_c.remove_nodes_from(list(nx.isolates(G_c)))
        if len(G_c.nodes) < 3:
            continue
        try:
            part = louvain_partition(G_c)
        except Exception:
            part = {n: 0 for n in G_c.nodes()}
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", c).strip("_")
        plot_network_auto(
            G_c, part,
            out_plots_folder / f"G_{safe}.png",
            title_template.format(
                concept=c,
                n_papers=len(docs),
                n_nodes=len(G_c.nodes),
                n_edges=len(G_c.edges),
            ),
            node_size_attr="n_papers", layout="auto",
            drop_isolates=False,
        )
        ok_count += 1
    return ok_count


__all__ = [
    "cooccurrence_heatmap_section",
    "cooccurrence_clustermap_section",
    "cooccurrence_network_section",
    "disparity_backbone_section",
    "bridging_section",
    "temporal_pair_evolution_section",
    "per_concept_subgraphs_section",
]
