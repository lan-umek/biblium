# -*- coding: utf-8 -*-
"""
biblium.network_dynamics
========================

Diachronic (time-sliced) co-word networks, fuzzy document-to-cluster
membership, CiteSpace-style cluster timelines and Chen's (2012) structural
variation analysis (new in 2.19).

Contents
--------
Time-sliced networks
    ``period_graphs``                 co-occurrence graphs per period with a
                                      stable node set and louvain partitions
    ``animate_diachronic_network``    GIF animation with a fixed layout
    ``plot_diachronic_grid``          static small multiples, same layout

Documents and clusters
    ``fuzzy_cluster_membership``      share of a document's keywords per cluster
    ``cluster_document_table``        documents of one cluster

Cluster timeline (CiteSpace)
    ``cluster_timeline``              node table (cluster, year, frequency, label)
    ``plot_cluster_timeline``         horizontal lanes per cluster

Structural variation (Chen 2012)
    ``structural_variation``          modularity change rate, cluster linkage
                                      and centrality divergence per document
    ``plot_structural_variation``     metric vs. citations

Networks are ``networkx.Graph`` objects with an integer node attribute holding
the partition (default ``partition_louvain``) and an edge attribute
``weight``. Partitions are computed with
``networkx.algorithms.community.louvain_communities`` (seeded) so that all
results are deterministic.

References
----------
Chen, C. (2012). Predictive effects of structural variation on citation
counts. JASIST, 63(3), 431-449.

Author: Lan Umek
Version: 2.19.0
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "period_graphs",
    "animate_diachronic_network",
    "plot_diachronic_grid",
    "fuzzy_cluster_membership",
    "cluster_document_table",
    "cluster_timeline",
    "plot_cluster_timeline",
    "structural_variation",
    "plot_structural_variation",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _explode_items(df: pd.DataFrame, col: str, sep: str = ";") -> pd.DataFrame:
    """Long table (``_doc``, ``Item``) from a multi-valued column; empty and
    ``nan`` tokens dropped; duplicated tokens within a document counted once."""
    s = df[col].dropna().astype(str).str.split(sep)
    long = s.explode().str.strip()
    long = long[(long != "") & (long.str.lower() != "nan")]
    out = long.rename("Item").reset_index().rename(columns={"index": "_doc"})
    return out.drop_duplicates(["_doc", "Item"])


def _louvain_partition(G: nx.Graph, seed: int = 42, weight: str = "weight") -> Dict:
    """Seeded louvain partition as ``{node: community id}``; communities are
    numbered by decreasing size (ties: smallest node label first)."""
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_edges() == 0:
        comms = [{n} for n in G.nodes()]
    else:
        comms = nx.algorithms.community.louvain_communities(G, weight=weight, seed=seed)
    comms = sorted((sorted(map(str, c)) for c in comms), key=lambda c: (-len(c), c[0]))
    # map back to original node objects via string label
    label_to_node = {str(n): n for n in G.nodes()}
    part = {}
    for cid, c in enumerate(comms):
        for lab in c:
            part[label_to_node[lab]] = cid
    return part


def _communities_from_attr(G: nx.Graph, attr: str) -> List[set]:
    groups: Dict = defaultdict(set)
    for n, d in G.nodes(data=True):
        groups[d.get(attr, -1)].add(n)
    return [groups[k] for k in sorted(groups)]


def _quantile_periods(years: pd.Series, n_slices: int) -> List[Tuple[int, int]]:
    """Periods with approximately equal numbers of documents (same scheme as
    ``BiblioStats.thematic_evolution``)."""
    qs = np.linspace(0, 1, n_slices + 1)
    edges = sorted(set(int(round(v)) for v in years.quantile(qs).dropna()))
    if len(edges) < 2:
        raise ValueError("Not enough distinct years for slicing")
    periods = []
    for i in range(len(edges) - 1):
        lo = edges[i] if i == 0 else edges[i] + 1
        periods.append((lo, edges[i + 1]))
    return periods


def _finish(fig, ax, title, f_name, dpi, show):
    if title and ax is not None:
        ax.set_title(title)
    fig.tight_layout()
    if f_name:
        fig.savefig(f_name, dpi=dpi, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 1) Period graphs
# ---------------------------------------------------------------------------

def period_graphs(
    df: pd.DataFrame,
    field: str = "Author Keywords",
    year_col: str = "Year",
    periods: Optional[Sequence[Tuple[int, int]]] = None,
    n_slices: int = 3,
    cumulative: bool = True,
    sep: str = ";",
    min_count: int = 2,
    min_weight: int = 1,
    top_n: int = 50,
    seed: int = 42,
) -> Tuple[List[nx.Graph], List[str]]:
    """
    Co-occurrence graphs of a multi-valued field per time period.

    The node set is restricted to the ``top_n`` most frequent items of the
    **whole** corpus so that node identity is stable across periods (nodes of
    period ``i`` are a subset of this global set; a node appears in a period
    graph when it occurs in at least one document of the period).

    Parameters
    ----------
    df : pd.DataFrame
    field : str
        Multi-valued column (items separated by ``sep``).
    year_col : str
    periods : sequence of (lo, hi), optional
        Explicit periods (both bounds inclusive). If None, ``n_slices`` periods
        with approximately equal numbers of documents are derived from year
        quantiles (same scheme as ``BiblioStats.thematic_evolution``).
    n_slices : int
    cumulative : bool
        If True, period ``i`` uses all documents published up to (and
        including) its upper bound; if False, only documents inside the period.
    sep : str
    min_count : int
        Minimum co-occurrence count for a pair to enter the matrix (passed to
        ``build_cooccurrence_matrix``).
    min_weight : int
        Edges with weight <= ``min_weight`` are dropped (``matrix_to_network``).
    top_n : int
        Size of the global node set (most frequent items by number of
        documents in the full corpus).
    seed : int
        Seed of the louvain partition.

    Returns
    -------
    graphs : list of nx.Graph
        One graph per (non-empty) period; node attributes ``freq`` (number of
        documents of the period containing the item) and ``partition_louvain``
        (community id, ints numbered by decreasing community size); edge
        attribute ``weight``.
    labels : list of str
        ``"lo-hi"`` label of each period.
    """
    from biblium.utilsbib_modules.network import build_cooccurrence_matrix, matrix_to_network

    if field not in df.columns:
        raise ValueError(f"Column {field!r} not found")
    years = _to_numeric(df[year_col])
    if periods is None:
        periods = _quantile_periods(years.dropna(), n_slices)

    items = _explode_items(df, field, sep)
    freq_all = items["Item"].value_counts()
    keep = freq_all.sort_index().sort_values(ascending=False, kind="mergesort").head(top_n).index
    keep_set = set(keep)
    items = items[items["Item"].isin(keep_set)]

    graphs: List[nx.Graph] = []
    labels: List[str] = []
    for lo, hi in periods:
        mask = (years <= hi) if cumulative else ((years >= lo) & (years <= hi))
        docs = df.index[mask.fillna(False)]
        sub_items = items[items["_doc"].isin(docs)]
        if sub_items.empty:
            continue
        freq = sub_items["Item"].value_counts()
        # rebuild a reduced column with only the kept items (same separator)
        joined = sub_items.groupby("_doc")["Item"].apply(lambda s: sep.join(sorted(s)))
        tmp = pd.DataFrame({"_f": joined})
        mat = build_cooccurrence_matrix(tmp, "_f", sep=sep, min_count=min_count)
        G = matrix_to_network(mat, min_weight=min_weight)
        for n in G.nodes():
            G.nodes[n]["freq"] = int(freq.get(n, 0))
        part = _louvain_partition(G, seed=seed)
        nx.set_node_attributes(G, part, "partition_louvain")
        G.graph["period"] = f"{lo}-{hi}"
        graphs.append(G)
        labels.append(f"{lo}-{hi}")
    return graphs, labels


# ---------------------------------------------------------------------------
# 2) Animation / small multiples with a fixed layout
# ---------------------------------------------------------------------------

def _union_graph(graphs: Sequence[nx.Graph]) -> nx.Graph:
    U = nx.Graph()
    for G in graphs:
        U.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 1.0))
            if U.has_edge(u, v):
                U[u][v]["weight"] += w
            else:
                U.add_edge(u, v, weight=w)
    return U


def _fixed_layout(graphs: Sequence[nx.Graph], layout: str, seed: int) -> Dict:
    U = _union_graph(graphs)
    if layout == "union" or layout == "spring":
        return nx.spring_layout(U, seed=seed, weight="weight")
    if layout == "circular":
        return nx.circular_layout(U)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(U, weight="weight")
    raise ValueError("layout must be 'union', 'spring', 'circular' or 'kamada_kawai'")


def _draw_period(ax, G: nx.Graph, pos: Dict, partition_attr: str, node_scale: float,
                 cmap, title: str, with_labels: bool = True, font_size: int = 7):
    ax.clear()
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if G.number_of_nodes() == 0:
        return
    nodes = list(G.nodes())
    freq = np.array([float(G.nodes[n].get("freq", 1.0)) for n in nodes])
    fmax = freq.max() if freq.max() > 0 else 1.0
    sizes = node_scale * (0.2 + 0.8 * freq / fmax)
    colors = [cmap(int(G.nodes[n].get(partition_attr, 0)) % cmap.N) for n in nodes]
    weights = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    wmax = max(weights) if weights else 1.0
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3,
                           width=[0.5 + 2.0 * w / wmax for w in weights])
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=sizes, node_color=colors,
                           ax=ax, linewidths=0.5, edgecolors="white")
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size)
    # keep the frame fixed
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 0.1
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def animate_diachronic_network(
    graphs: Sequence[nx.Graph],
    labels: Sequence[str],
    f_name: str,
    partition_attr: str = "partition_louvain",
    layout: str = "union",
    fps: int = 1,
    figsize: Tuple[float, float] = (9, 7),
    node_scale: float = 300,
    seed: int = 42,
    cmap: str = "tab10",
    writer: str = "pillow",
    with_labels: bool = True,
    font_size: int = 7,
    dpi: int = 100,
) -> str:
    """
    Animated (GIF) diachronic network: one frame per period on a fixed layout
    computed on the union of all period graphs.

    Node size is proportional to the node attribute ``freq`` of the period,
    colour to the period's partition and the frame title is the period label.

    Parameters
    ----------
    graphs, labels
        Output of :func:`period_graphs`.
    f_name : str
        Output path (``.gif`` for the pillow writer).
    layout : {"union", "spring", "circular", "kamada_kawai"}
        Layout algorithm applied to the union graph (``"union"`` =
        seeded ``spring_layout``).
    fps : int
        Frames per second.
    writer : str
        ``"pillow"`` (default; GIF) or any writer name accepted by matplotlib
        (e.g. ``"ffmpeg"`` if installed).

    Returns
    -------
    str
        The path written.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if len(graphs) == 0:
        raise ValueError("No graphs to animate")
    if len(labels) != len(graphs):
        raise ValueError("labels must have the same length as graphs")
    pos = _fixed_layout(graphs, layout, seed)
    palette = plt.get_cmap(cmap)
    fig, ax = plt.subplots(figsize=figsize)

    def _update(i):
        _draw_period(ax, graphs[i], pos, partition_attr, node_scale, palette, str(labels[i]),
                     with_labels=with_labels, font_size=font_size)
        return []

    anim = animation.FuncAnimation(fig, _update, frames=len(graphs), interval=1000 / fps, blit=False)
    if writer == "pillow":
        w = animation.PillowWriter(fps=fps)
    else:
        w = writer
    anim.save(f_name, writer=w, dpi=dpi)
    plt.close(fig)
    return f_name


def plot_diachronic_grid(
    graphs: Sequence[nx.Graph],
    labels: Sequence[str],
    ncols: int = 3,
    partition_attr: str = "partition_louvain",
    layout: str = "union",
    figsize_per_panel: Tuple[float, float] = (4.5, 4.0),
    node_scale: float = 200,
    seed: int = 42,
    cmap: str = "tab10",
    with_labels: bool = True,
    font_size: int = 6,
    title: Optional[str] = None,
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    """
    Static small multiples of the period graphs on the same fixed layout as
    :func:`animate_diachronic_network`.

    Returns
    -------
    (fig, axes) -- ``axes`` is a flat numpy array of Axes (unused panels are
    hidden).
    """
    import matplotlib.pyplot as plt

    n = len(graphs)
    if n == 0:
        raise ValueError("No graphs to plot")
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    pos = _fixed_layout(graphs, layout, seed)
    palette = plt.get_cmap(cmap)
    for i, ax in enumerate(axes):
        if i < n:
            _draw_period(ax, graphs[i], pos, partition_attr, node_scale, palette, str(labels[i]),
                         with_labels=with_labels, font_size=font_size)
        else:
            ax.set_axis_off()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if f_name:
        fig.savefig(f_name, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


# ---------------------------------------------------------------------------
# 3) Fuzzy document -> cluster membership
# ---------------------------------------------------------------------------

def fuzzy_cluster_membership(
    df: pd.DataFrame,
    G: nx.Graph,
    field: str = "Author Keywords",
    partition_attr: str = "partition_louvain",
    sep: str = ";",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Fuzzy membership of documents in the clusters of a co-word network.

    For each document the share of its items (keywords) that are nodes of
    ``G`` belonging to each cluster is computed.

    Parameters
    ----------
    df : pd.DataFrame
    G : nx.Graph
        Network with node attribute ``partition_attr``.
    field : str
        Multi-valued column of ``df`` whose items are the nodes of ``G``.
    normalize : bool
        If True, memberships are shares in [0, 1] summing to 1 per document;
        if False, raw counts of the document's items per cluster.

    Returns
    -------
    pd.DataFrame
        Index as ``df``; columns ``Cluster k`` for each cluster id ``k`` of the
        partition (sorted), ``Main cluster`` (cluster with the largest
        membership; ties -> smallest id), ``Main membership``, ``N keywords in
        network`` and ``Entropy`` (Shannon entropy, natural log, of the
        normalised membership vector). Documents without any item in ``G``
        get NaN memberships (and 0 in ``N keywords in network``).
    """
    part = {n: d.get(partition_attr) for n, d in G.nodes(data=True)}
    part = {n: c for n, c in part.items() if c is not None}
    clusters = sorted(set(part.values()))
    ccols = [f"Cluster {c}" for c in clusters]
    out = pd.DataFrame(np.nan, index=df.index, columns=ccols, dtype=float)
    out["Main cluster"] = np.nan
    out["Main membership"] = np.nan
    out["N keywords in network"] = 0
    out["Entropy"] = np.nan
    if field not in df.columns or not clusters:
        return out

    items = _explode_items(df, field, sep)
    items = items[items["Item"].isin(part)]
    if items.empty:
        return out
    items["Cluster"] = items["Item"].map(part)
    counts = items.groupby(["_doc", "Cluster"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=clusters, fill_value=0)
    n_kw = counts.sum(axis=1)
    shares = counts.div(n_kw, axis=0)

    vals = shares if normalize else counts
    out.loc[counts.index, ccols] = vals.to_numpy(dtype=float)
    out.loc[counts.index, "N keywords in network"] = n_kw.to_numpy()
    main = shares.to_numpy().argmax(axis=1)  # first max -> smallest cluster id
    out.loc[counts.index, "Main cluster"] = [clusters[i] for i in main]
    out.loc[counts.index, "Main membership"] = shares.to_numpy().max(axis=1)
    p = shares.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    out.loc[counts.index, "Entropy"] = ent
    out["N keywords in network"] = out["N keywords in network"].astype(int)
    return out


def cluster_document_table(
    membership_df: pd.DataFrame,
    df: pd.DataFrame,
    cluster: int,
    min_membership: float = 0.5,
    cols: Sequence[str] = ("Title", "Year", "Cited by"),
) -> pd.DataFrame:
    """
    Documents belonging to one cluster with membership >= ``min_membership``.

    Returns
    -------
    pd.DataFrame
        Columns: the available ``cols`` of ``df``, ``Membership`` and
        ``Main cluster``; sorted by membership (desc). The index is the
        document index of ``df``.
    """
    col = f"Cluster {cluster}"
    if col not in membership_df.columns:
        raise ValueError(f"Cluster {cluster} not in membership table")
    m = membership_df[col]
    sel = m[m >= min_membership].index
    use = [c for c in cols if c in df.columns]
    out = df.loc[sel, use].copy()
    out["Membership"] = m.loc[sel].to_numpy()
    out["Main cluster"] = membership_df.loc[sel, "Main cluster"].to_numpy()
    return out.sort_values("Membership", ascending=False, kind="mergesort")


# ---------------------------------------------------------------------------
# 4) Cluster timeline
# ---------------------------------------------------------------------------

def cluster_timeline(
    G: nx.Graph,
    df: pd.DataFrame,
    field: str = "Author Keywords",
    year_col: str = "Year",
    partition_attr: str = "partition_louvain",
    sep: str = ";",
    year_stat: str = "mean",
    top_labels: int = 5,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    CiteSpace-style timeline data of a clustered co-word network.

    Parameters
    ----------
    G : nx.Graph
        Network with node attribute ``partition_attr``.
    df : pd.DataFrame
        Corpus used to compute occurrence counts and years of the nodes.
    year_stat : {"mean", "median", "first"}
        Statistic of the publication years of the documents containing the
        node used as its position on the time axis.
    top_labels : int
        Number of nodes per cluster (by Freq) flagged in ``Label``.

    Returns
    -------
    table : pd.DataFrame
        One row per node: ``Cluster``, ``Cluster rank`` (0 = largest cluster),
        ``Node``, ``Freq`` (number of documents containing the node), ``Year``,
        ``Label`` (bool). Sorted by cluster size (desc) then Freq (desc).
    cluster_labels : dict
        ``{cluster id: "top1; top2; top3"}`` (top 3 nodes by Freq).
    """
    if year_stat not in ("mean", "median", "first"):
        raise ValueError("year_stat must be 'mean', 'median' or 'first'")
    part = {n: d.get(partition_attr) for n, d in G.nodes(data=True)}
    items = _explode_items(df, field, sep) if field in df.columns else pd.DataFrame(columns=["_doc", "Item"])
    items = items[items["Item"].isin(part)]
    years = _to_numeric(df[year_col]) if year_col in df.columns else pd.Series(np.nan, index=df.index)
    items["Year"] = items["_doc"].map(years)
    freq = items.groupby("Item").size()
    agg = {"mean": "mean", "median": "median", "first": "min"}[year_stat]
    yr = items.dropna(subset=["Year"]).groupby("Item")["Year"].agg(agg)

    rows = []
    for n in G.nodes():
        rows.append({
            "Cluster": part.get(n),
            "Node": n,
            "Freq": int(freq.get(n, 0)),
            "Year": float(yr.get(n, np.nan)),
        })
    table = pd.DataFrame(rows, columns=["Cluster", "Node", "Freq", "Year"])
    if table.empty:
        table["Cluster rank"] = []
        table["Label"] = []
        return table, {}
    sizes = table.groupby("Cluster").size()
    order = sizes.reset_index().rename(columns={0: "n"})
    order = order.sort_values(["n", "Cluster"], ascending=[False, True], kind="mergesort")
    rank = {c: i for i, c in enumerate(order["Cluster"])}
    table["Cluster rank"] = table["Cluster"].map(rank)
    table["Node"] = table["Node"].astype(str)
    table = table.sort_values(["Cluster rank", "Freq", "Node"], ascending=[True, False, True],
                              kind="mergesort").reset_index(drop=True)
    table["Label"] = table.groupby("Cluster").cumcount() < top_labels
    labels = {
        c: "; ".join(table.loc[table["Cluster"] == c, "Node"].head(3))
        for c in order["Cluster"]
    }
    table = table[["Cluster", "Cluster rank", "Node", "Freq", "Year", "Label"]]
    return table, labels


def plot_cluster_timeline(
    table: pd.DataFrame,
    cluster_labels: Dict[int, str],
    G: Optional[nx.Graph] = None,
    ax=None,
    figsize: Tuple[float, float] = (12, 7),
    max_edges: int = 300,
    node_scale: float = 200,
    cmap: str = "tab10",
    edge_alpha: float = 0.2,
    font_size: int = 7,
    title: Optional[str] = "Cluster timeline",
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    """
    CiteSpace-style timeline: one horizontal lane per cluster (largest on
    top, cluster label on the left), nodes as circles at (Year, lane) sized by
    Freq, labelled nodes annotated; optionally the strongest ``max_edges``
    edges of ``G`` (intra- and inter-cluster) drawn as thin lines.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if table is None or table.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return _finish(fig, ax, title, f_name, dpi, show)

    t = table.dropna(subset=["Year"]).copy()
    ranks = sorted(table["Cluster rank"].unique())
    n_lanes = len(ranks)
    lane = {r: n_lanes - 1 - r for r in ranks}  # largest cluster on top
    t["_y"] = t["Cluster rank"].map(lane)
    palette = plt.get_cmap(cmap)
    fmax = t["Freq"].max() if t["Freq"].max() > 0 else 1.0
    sizes = node_scale * (0.15 + 0.85 * t["Freq"] / fmax)
    colors = [palette(int(c) % palette.N) for c in t["Cluster"]]

    for r in ranks:
        ax.axhline(lane[r], color="lightgrey", lw=0.8, zorder=0)

    if G is not None and max_edges and max_edges > 0:
        pos = {row["Node"]: (row["Year"], row["_y"]) for _, row in t.iterrows()}
        edges = [(str(u), str(v), float(d.get("weight", 1.0))) for u, v, d in G.edges(data=True)
                 if str(u) in pos and str(v) in pos]
        edges.sort(key=lambda e: (-e[2], e[0], e[1]))
        wmax = edges[0][2] if edges else 1.0
        for u, v, w in edges[:max_edges]:
            (x1, y1), (x2, y2) = pos[u], pos[v]
            ax.plot([x1, x2], [y1, y2], color="grey", lw=0.3 + 1.2 * w / wmax,
                    alpha=edge_alpha, zorder=1)

    ax.scatter(t["Year"], t["_y"], s=sizes, c=colors, alpha=0.85, edgecolors="white",
               linewidths=0.5, zorder=2)
    for _, row in t[t["Label"]].iterrows():
        ax.annotate(row["Node"], (row["Year"], row["_y"]), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=font_size, zorder=3)

    rank_to_cluster = dict(zip(table["Cluster rank"], table["Cluster"]))
    ax.set_yticks([lane[r] for r in ranks])
    ax.set_yticklabels([f"#{r} {cluster_labels.get(rank_to_cluster[r], '')}" for r in ranks],
                       fontsize=font_size + 1)
    ax.set_ylim(-0.7, n_lanes - 0.3)
    ax.set_xlabel("Year")
    ax.xaxis.get_major_locator().set_params(integer=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _finish(fig, ax, title, f_name, dpi, show)


# ---------------------------------------------------------------------------
# 5) Structural variation (Chen 2012)
# ---------------------------------------------------------------------------

def _normalize_reference(ref: str) -> str:
    key = re.sub(r"\s+", " ", str(ref)).strip().lower()
    return key.rstrip(". ").strip()


def _doc_references(df: pd.DataFrame, reference_column: str, sep: str) -> Dict:
    """``{doc index: [normalised reference keys (unique, ordered)]}``."""
    out = {}
    for idx, cell in df[reference_column].items():
        if pd.isna(cell):
            out[idx] = []
            continue
        seen, refs = set(), []
        for tok in str(cell).split(sep):
            key = _normalize_reference(tok)
            if key and key != "nan" and key not in seen:
                seen.add(key)
                refs.append(key)
        out[idx] = refs
    return out


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def structural_variation(
    df: pd.DataFrame,
    reference_column: str = "References",
    year_col: str = "Year",
    sep: str = ";",
    min_ref_citations: int = 3,
    top_n_refs: int = 150,
    baseline_window: Optional[int] = None,
    seed: int = 42,
    max_docs_per_year: Optional[int] = None,
    approx_k: Optional[int] = None,
    title_col: str = "Title",
    citations_col: str = "Cited by",
) -> pd.DataFrame:
    """
    Structural variation analysis (Chen 2012) on the reference co-citation
    network.

    The node set is the ``top_n_refs`` most cited references (cited by at
    least ``min_ref_citations`` documents; references are matched on a
    normalised string). For each publication year ``t`` (ascending; the first
    year with a usable baseline network is the first year processed) the
    **baseline** network is the co-citation network of the references of the
    documents published before ``t`` (within ``baseline_window`` years if
    given), with a seeded louvain partition and modularity ``M0`` and
    betweenness centrality computed once per year. Years whose baseline has
    fewer than 3 nodes are skipped.

    For every document published in ``t`` with at least two references in the
    node set, its co-citation links (all pairs of its references; new edges
    with weight 1, existing edges +1) are added to a copy of the baseline and
    three metrics are computed:

    * ``Modularity change rate`` -- ``(M1 - M0) / M0 * 100`` where ``M1`` is
      the modularity of the updated network with the baseline partition held
      fixed (references not in the baseline form new singleton communities).
      Negative values mean the document connects clusters. NaN if ``M0 <= 0``.
    * ``Cluster linkage`` -- share of the document's reference pairs (both
      references baseline nodes) that connect two different baseline clusters.
      NaN if the document has no such pair.
    * ``Centrality divergence`` -- Kullback-Leibler divergence
      ``KL(after || before)`` between the normalised betweenness-centrality
      vectors of the network before and after adding the document's links
      (epsilon 1e-9 added; nodes new to the network have centrality 0 before).

    Parameters
    ----------
    max_docs_per_year : int, optional
        Process at most this many documents per year (first in ``df`` order).
    approx_k : int, optional
        Passed as ``k`` to ``nx.betweenness_centrality`` (sampling of pivots
        for speed, with ``seed``); None = exact.
    title_col, citations_col : str
        Optional columns copied to the output when present.

    Returns
    -------
    pd.DataFrame
        One row per processed document: ``Doc index``, ``Year``, ``Title`` (if
        present), ``N refs in network``, ``Modularity change rate``,
        ``Cluster linkage``, ``Centrality divergence``, ``Cited by`` (if
        present).
    """
    cols = ["Doc index", "Year", "Title", "N refs in network", "Modularity change rate",
            "Cluster linkage", "Centrality divergence", "Cited by"]
    if reference_column not in df.columns or year_col not in df.columns:
        return pd.DataFrame(columns=cols)
    years = _to_numeric(df[year_col])
    doc_refs = _doc_references(df, reference_column, sep)

    counts: Counter = Counter()
    for refs in doc_refs.values():
        counts.update(refs)
    ranked = sorted(((c, r) for r, c in counts.items() if c >= min_ref_citations),
                    key=lambda x: (-x[0], x[1]))
    node_set = {r for _, r in ranked[:top_n_refs]}
    doc_nodes = {idx: [r for r in refs if r in node_set] for idx, refs in doc_refs.items()}

    valid_years = sorted(int(y) for y in years.dropna().unique())
    rows = []
    for t in valid_years:
        lo = t - baseline_window if baseline_window is not None else -np.inf
        base_mask = (years < t) & (years >= lo)
        base_docs = df.index[base_mask.fillna(False)]
        G0 = nx.Graph()
        for idx in base_docs:
            nodes = doc_nodes[idx]
            G0.add_nodes_from(nodes)
            for u, v in combinations(sorted(nodes), 2):
                if G0.has_edge(u, v):
                    G0[u][v]["weight"] += 1
                else:
                    G0.add_edge(u, v, weight=1)
        if G0.number_of_nodes() < 3:
            continue
        part0 = _louvain_partition(G0, seed=seed)
        comms0: Dict[int, set] = defaultdict(set)
        for n, c in part0.items():
            comms0[c].add(n)
        base_comms = [comms0[c] for c in sorted(comms0)]
        M0 = (nx.algorithms.community.modularity(G0, base_comms, weight="weight")
              if G0.number_of_edges() > 0 else 0.0)
        bc0 = nx.betweenness_centrality(G0, k=approx_k, seed=seed)

        year_docs = [idx for idx in df.index[(years == t).fillna(False)] if len(doc_nodes[idx]) >= 2]
        if max_docs_per_year is not None:
            year_docs = year_docs[:max_docs_per_year]
        for idx in year_docs:
            nodes = doc_nodes[idx]
            pairs = list(combinations(sorted(nodes), 2))
            G1 = G0.copy()
            G1.add_nodes_from(nodes)
            for u, v in pairs:
                if G1.has_edge(u, v):
                    G1[u][v]["weight"] += 1
                else:
                    G1.add_edge(u, v, weight=1)
            new_nodes = [n for n in nodes if n not in part0]
            comms1 = [set(c) for c in base_comms] + [{n} for n in new_nodes]
            M1 = nx.algorithms.community.modularity(G1, comms1, weight="weight")
            dM = (M1 - M0) / M0 * 100.0 if M0 > 0 else np.nan
            base_pairs = [(u, v) for u, v in pairs if u in part0 and v in part0]
            if base_pairs:
                cl = sum(1 for u, v in base_pairs if part0[u] != part0[v]) / len(base_pairs)
            else:
                cl = np.nan
            bc1 = nx.betweenness_centrality(G1, k=approx_k, seed=seed)
            order = list(G1.nodes())
            v0 = np.array([bc0.get(n, 0.0) for n in order])
            v1 = np.array([bc1.get(n, 0.0) for n in order])
            ckl = _kl_divergence(v1, v0)
            rows.append({
                "Doc index": idx,
                "Year": t,
                "Title": df.at[idx, title_col] if title_col in df.columns else np.nan,
                "N refs in network": len(nodes),
                "Modularity change rate": dM,
                "Cluster linkage": cl,
                "Centrality divergence": ckl,
                "Cited by": _to_numeric(df[citations_col]).at[idx] if citations_col in df.columns else np.nan,
            })
    out = pd.DataFrame(rows, columns=cols)
    if title_col not in df.columns:
        out = out.drop(columns=["Title"])
    if citations_col not in df.columns:
        out = out.drop(columns=["Cited by"])
    return out.reset_index(drop=True)


def plot_structural_variation(
    sva_df: pd.DataFrame,
    metric: str = "Modularity change rate",
    citations_col: str = "Cited by",
    ax=None,
    top_n_annotate: int = 5,
    label_len: int = 40,
    color: str = "#3182bd",
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    """
    Scatter of a structural-variation metric against citations (log1p),
    with the ``top_n_annotate`` documents by |metric| annotated by Title.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if title is None:
        title = f"{metric} vs. citations"
    if sva_df is None or sva_df.empty or metric not in sva_df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return _finish(fig, ax, title, f_name, dpi, show)

    d = sva_df.dropna(subset=[metric]).copy()
    cit = _to_numeric(d[citations_col]).fillna(0) if citations_col in d.columns else pd.Series(0.0, index=d.index)
    ax.scatter(d[metric], np.log1p(cit), s=25, alpha=0.7, color=color, edgecolors="white", linewidths=0.4)
    ax.set_xlabel(metric)
    ax.set_ylabel("log(1 + citations)")
    if top_n_annotate and "Title" in d.columns:
        top = d.reindex(d[metric].abs().sort_values(ascending=False, kind="mergesort").index).head(top_n_annotate)
        for i, row in top.iterrows():
            lab = str(row["Title"])
            if len(lab) > label_len:
                lab = lab[: label_len - 1] + "…"
            ax.annotate(lab, (row[metric], np.log1p(cit.loc[i])), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
    return _finish(fig, ax, title, f_name, dpi, show)
