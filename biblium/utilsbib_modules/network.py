# -*- coding: utf-8 -*-
"""
Network utilities - graph analysis, partitioning, co-occurrence networks.

This module contains:
- Network construction and analysis
- Community detection algorithms
- Co-occurrence matrix computation
- Network export functions
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

import networkx as nx

# Optional imports
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    community_louvain = None
    LOUVAIN_AVAILABLE = False

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    ig = None
    IGRAPH_AVAILABLE = False


# =============================================================================
# NETWORK CONSTRUCTION
# =============================================================================

def build_cooccurrence_matrix(
    df: pd.DataFrame,
    column: str,
    sep: str = "; ",
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Build a co-occurrence matrix from a column with list values.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column : str
        Column containing list values (separated by sep).
    sep : str
        Separator for list values.
    min_count : int
        Minimum co-occurrence count to include.

    Returns
    -------
    DataFrame
        Symmetric co-occurrence matrix.
    """
    from collections import Counter
    from itertools import combinations
    
    # Extract all items
    all_items = set()
    cooc_counts = Counter()
    
    for val in df[column].dropna():
        items = [i.strip() for i in str(val).split(sep) if i.strip()]
        all_items.update(items)
        
        # Count co-occurrences
        for pair in combinations(sorted(set(items)), 2):
            cooc_counts[pair] += 1
    
    # Build matrix
    items = sorted(all_items)
    matrix = pd.DataFrame(0, index=items, columns=items)
    
    for (i, j), count in cooc_counts.items():
        if count >= min_count:
            matrix.loc[i, j] = count
            matrix.loc[j, i] = count
    
    return matrix


def matrix_to_network(
    matrix: pd.DataFrame,
    min_weight: float = 0,
) -> nx.Graph:
    """
    Convert a co-occurrence matrix to a NetworkX graph.

    Parameters
    ----------
    matrix : DataFrame
        Symmetric matrix with items as index/columns.
    min_weight : float
        Minimum edge weight to include.

    Returns
    -------
    nx.Graph
        Undirected weighted graph.
    """
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(matrix.index)
    
    # Add edges
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            weight = matrix.iloc[i, j]
            if weight > min_weight:
                G.add_edge(
                    matrix.index[i],
                    matrix.columns[j],
                    weight=weight
                )
    
    return G


def normalize_symmetric_matrix(
    matrix,
    method: str = "jaccard",
):
    """
    Normalize a symmetric co-occurrence matrix (poenoteno v 2.18).

    Occurrence counts n_i come from the diagonal when present (self-occurrences),
    otherwise from row sums.

    Methods
    -------
    - "association": c_ij / (n_i * n_j)          (association strength, VOSviewer)
    - "salton" / "cosine": c_ij / sqrt(n_i*n_j)  (Salton cosine)
    - "inclusion": c_ij / min(n_i, n_j)
    - "jaccard": c_ij / (n_i + n_j - c_ij)
    - "row": row-wise proportion
    - "none": copy of the input
    """
    if matrix.empty:
        return pd.DataFrame()
    if method == "none":
        return matrix.copy()
    if method == "row":
        row_sums = matrix.sum(axis=1).replace(0, 1)
        return matrix.div(row_sums, axis=0)

    diag = np.diag(matrix.values).astype(float)
    if diag.sum() == 0:
        diag = matrix.sum(axis=1).values.astype(float)

    vals = matrix.values.astype(float)
    m = len(matrix)
    result = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                result[i, j] = 1.0
                continue
            cooc = vals[i, j]
            ni, nj = diag[i], diag[j]
            if method == "association":
                denom = ni * nj
            elif method in ("salton", "cosine"):
                denom = np.sqrt(ni * nj)
            elif method == "inclusion":
                denom = min(ni, nj)
            elif method == "jaccard":
                denom = ni + nj - cooc
            else:
                raise ValueError(f"Unknown normalization method: {method!r}")
            result[i, j] = cooc / denom if denom > 0 else 0.0
    return pd.DataFrame(result, index=matrix.index, columns=matrix.columns)

def louvain_partition(G: nx.Graph, random_state: int = 42) -> Dict[Any, int]:
    """
    Detect communities using the Louvain algorithm.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    dict
        Mapping from node to community ID.
    """
    if not LOUVAIN_AVAILABLE:
        raise ImportError("python-louvain package not installed")

    return community_louvain.best_partition(G, random_state=random_state)


def greedy_modularity_partition(G: nx.Graph) -> Dict[Any, int]:
    """
    Detect communities using greedy modularity optimization.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    dict
        Mapping from node to community ID.
    """
    communities = nx.community.greedy_modularity_communities(G)
    
    partition = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[node] = i
    
    return partition


def label_propagation_partition(G: nx.Graph) -> Dict[Any, int]:
    """
    Detect communities using label propagation.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    dict
        Mapping from node to community ID.
    """
    communities = nx.community.label_propagation_communities(G)
    
    partition = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[node] = i
    
    return partition


def add_partitions(
    G,
    methods=None,
    seed=42,
    louvain_kwargs=None,
    greedy_kwargs=None,
    label_kwargs=None,
    girvan_kwargs=None,
    k_clique_kwargs=None,
    kernighan_kwargs=None,
    walktrap_kwargs=None,
    edge_betweenness_kwargs=None,
    infomap_kwargs=None,
    leading_kwargs=None,
    leiden_kwargs=None,
    spinglass_kwargs=None,
):
    """
    Compute community partitions and attach them to G as node attributes
    ``partition_<method>`` (poenoteno v 2.18).

    Parameters
    ----------
    G : nx.Graph
    methods : list of str, optional
        Subset of methods to run (e.g. ["louvain"]). None = all available.
        Alias "greedy" -> "greedy_modularity".

    Returns
    -------
    dict
        Mapping requested method name -> partition dict. G is annotated in place.
    """
    from biblium.utilsbib import (
        walktrap_partition, edge_betweenness_partition, infomap_partition,
        leading_eigenvector_partition, leiden_partition, spinglass_partition,
        girvan_newman_partition, k_clique_partition, kernighan_lin_partition,
    )
    # Reproducibility: seed Python/NumPy RNG and route igraph through Python RNG
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import igraph as _ig
        _ig.set_random_number_generator(_random)
    except Exception:
        pass
    configs = {
        "walktrap": (walktrap_partition, walktrap_kwargs or {}),
        "edge_betweenness": (edge_betweenness_partition, edge_betweenness_kwargs or {}),
        "infomap": (infomap_partition, infomap_kwargs or {}),
        "leading_eigenvector": (leading_eigenvector_partition, leading_kwargs or {}),
        "leiden": (leiden_partition, leiden_kwargs or {}),
        "spinglass": (spinglass_partition, spinglass_kwargs or {}),
        "louvain": (louvain_partition, louvain_kwargs or {}),
        "greedy_modularity": (greedy_modularity_partition, greedy_kwargs or {}),
        "label_propagation": (label_propagation_partition, label_kwargs or {}),
        "girvan_newman": (girvan_newman_partition, girvan_kwargs or {}),
        "k_clique": (k_clique_partition, k_clique_kwargs or {}),
        "kernighan_lin": (kernighan_lin_partition, kernighan_kwargs or {}),
    }
    aliases = {"greedy": "greedy_modularity", "label": "label_propagation"}
    requested = list(configs) if methods is None else list(methods)
    results = {}
    for name in requested:
        key = aliases.get(name, name)
        if key not in configs:
            continue
        func, kwargs = configs[key]
        try:
            partition = func(G, **kwargs)
            results[name] = partition
            for node, cid in partition.items():
                G.nodes[node][f"partition_{name}"] = cid
        except Exception:
            pass
    return results

def compute_basic_stats(G: nx.Graph) -> Dict[str, Any]:
    """
    Compute basic network statistics.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    dict
        Network statistics.
    """
    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
    }
    
    if G.number_of_nodes() > 0:
        degrees = [d for _, d in G.degree()]
        stats["avg_degree"] = np.mean(degrees)
        stats["max_degree"] = max(degrees)
        
        if nx.is_connected(G):
            stats["diameter"] = nx.diameter(G)
            stats["avg_path_length"] = nx.average_shortest_path_length(G)
        else:
            # Get largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            stats["largest_component_size"] = len(largest_cc)
            stats["diameter_largest"] = nx.diameter(subG)
    
    return stats


def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    """
    Compute various centrality measures.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    DataFrame
        Centrality measures for each node.
    """
    centralities = {
        "degree": dict(G.degree()),
        "betweenness": nx.betweenness_centrality(G),
        "closeness": nx.closeness_centrality(G),
        "eigenvector": {},
    }
    
    try:
        centralities["eigenvector"] = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception:
        pass
    
    df = pd.DataFrame(centralities)
    df.index.name = "Node"
    
    return df


def nodes_to_dataframe(G: nx.Graph) -> pd.DataFrame:
    """
    Convert node attributes to a DataFrame.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    DataFrame
        Node attributes.
    """
    data = []
    for node, attrs in G.nodes(data=True):
        row = {"Node": node, "Degree": G.degree(node)}
        row.update(attrs)
        data.append(row)
    
    return pd.DataFrame(data)


# =============================================================================
# NETWORK EXPORT
# =============================================================================

def save_network(
    G,
    filename,
    formats=None,
    vector_cols=None,
    partition_attr="partition",
):
    """
    Save a NetworkX graph in various formats, including Pajek (.net, .clu, .vec),
    GraphML, and GEXF.

    Parameters
    ----------
    G : networkx.Graph
        The graph to save.
    filename : str
        Path/prefix for saving files (no extension).
    formats : list[str], optional
        Formats to save ("pajek", "graphml", "gexf"). Defaults to all.
    vector_cols : list[str], optional
        Node attributes to export as .vec files.
    partition_attr : str, optional
        Node attribute prefix to use for .clu partition file(s).
        All node attributes whose names start with this prefix are exported
        as separate .clu files.
    """
    if formats is None:
        formats = ["pajek", "graphml", "gexf"]

    # --- Pajek export (.net, .clu, .vec) ------------------------------------
    if "pajek" in formats:
        # Copy G and convert all node/edge attributes to strings for Pajek.
        # This avoids "Non-string attribute" warnings from NetworkX's Pajek writer.
        G_pajek = G.copy()

        # Node attributes -> strings
        for _, attrs in G_pajek.nodes(data=True):
            for key, value in list(attrs.items()):
                if value is None:
                    # Drop None-valued attributes
                    del attrs[key]
                elif not isinstance(value, str):
                    attrs[key] = str(value)

        # Edge attributes -> strings
        for _, _, attrs in G_pajek.edges(data=True):
            for key, value in list(attrs.items()):
                if value is None:
                    del attrs[key]
                elif not isinstance(value, str):
                    attrs[key] = str(value)

        # Main Pajek network file
        nx.write_pajek(G_pajek, f"{filename}.net")

        # Partition .clu files (use ORIGINAL G so cluster IDs stay numeric)
        if any(
            any(k.startswith(partition_attr) for k in d)
            for _, d in G.nodes(data=True)
        ):
            partition_attrs = set()
            for _, d in G.nodes(data=True):
                partition_attrs.update(
                    k for k in d if k.startswith(partition_attr)
                )

            # Save each partition to its own .clu file
            for attr in sorted(partition_attrs):  # sorted for consistent order
                with open(f"{filename}_{attr}.clu", "w", encoding="utf-8") as f:
                    f.write(f"*Vertices {G.number_of_nodes()}\n")
                    for n in G.nodes():
                        # Pajek expects integers, but str(int) is fine in text file
                        f.write(f"{G.nodes[n].get(attr, 1)}\n")

        # Vector .vec files (again, use ORIGINAL G)
        if vector_cols:
            for vec_col in vector_cols:
                with open(f"{filename}_{vec_col}.vec", "w", encoding="utf-8") as f:
                    f.write(f"*Vertices {G.number_of_nodes()}\n")
                    for n in G.nodes():
                        f.write(f"{G.nodes[n].get(vec_col, 0)}\n")

    # --- GraphML -------------------------------------------------------------
    if "graphml" in formats:
        nx.write_graphml(G, f"{filename}.graphml")

    # --- GEXF ----------------------------------------------------------------
    if "gexf" in formats:
        nx.write_gexf(G, f"{filename}.gexf")

def save_to_pajek(
    G,
    basename,
    partition_attrs = None,
    vector_attrs = None,
):
    """
    Export graph and node attributes to Pajek files.

    Parameters
    ----------
    G : networkx.Graph
    basename : str
        Base filename for .net, .clu, .vec files.
    partition_attrs : str or list of str, optional
    vector_attrs : list of str, optional
    """
    nx.write_pajek(G, f"{basename}.net")
    nodes = list(G.nodes())
    if partition_attrs:
        attrs = [partition_attrs] if isinstance(partition_attrs, str) else partition_attrs
        for attr in attrs:
            fname = f"{basename}.clu" if len(attrs) == 1 else f"{basename}_{attr}.clu"
            with open(fname, "w") as f:
                f.write(f"*Vertices {len(nodes)}\n")
                for n in nodes:
                    f.write(str(G.nodes[n].get(attr, 0)) + "\n")
    if vector_attrs:
        with open(f"{basename}.vec", "w") as f:
            f.write(f"*Vertices {len(nodes)}\n")
            for n in nodes:
                vals = [str(G.nodes[n].get(v, 0)) for v in vector_attrs]
                f.write(" ".join(vals) + "\n")

def disparity_filter_backbone(
    G: nx.Graph,
    alpha: float = 0.05,
    weight_attr: str = "weight",
) -> nx.Graph:
    """
    Extract a network backbone using the disparity filter (Serrano,
    Boguna & Vespignani 2009).

    For each edge (i,j) with weight w_ij, computes a p-value relative
    to the null model of uniformly distributed strengths at each node:

        p_ij(at_i) = (1 - w_ij / s_i)^(k_i - 1)

    where s_i = sum of weights of edges incident to i and k_i = degree.
    Edge is kept if min(p_ij(at_i), p_ij(at_j)) < alpha (significant from
    at least one node's perspective).

    Parameters
    ----------
    G : nx.Graph
        Weighted graph.
    alpha : float, default 0.05
        Significance threshold (smaller = stricter filtering).
    weight_attr : str, default "weight"
        Edge attribute storing weights.

    Returns
    -------
    nx.Graph
        Backbone subgraph with significant edges only.

    References
    ----------
    Serrano, M.A., Boguna, M., Vespignani, A. (2009).
    Extracting the multiscale backbone of complex weighted networks.
    PNAS, 106(16), 6483-6488.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    # Pre-compute strength and degree for each node
    strength = {}
    degree = {}
    for n in G.nodes():
        strength[n] = sum(
            G[n][m].get(weight_attr, 1.0) for m in G.neighbors(n)
        )
        degree[n] = G.degree(n)

    for u, v, d in G.edges(data=True):
        w = d.get(weight_attr, 1.0)
        if w <= 0:
            continue
        # p-value from each endpoint's perspective
        p_u = ((1 - w / strength[u]) ** (degree[u] - 1)
               if degree[u] > 1 and strength[u] > 0 else 1.0)
        p_v = ((1 - w / strength[v]) ** (degree[v] - 1)
               if degree[v] > 1 and strength[v] > 0 else 1.0)
        # Keep edge if significant from at least one perspective
        if min(p_u, p_v) < alpha:
            H.add_edge(u, v, **d)
    return H


def bridging_centralities(
    G: nx.Graph,
    top_n: int = 20,
    weight_attr: str = "weight",
) -> "pd.DataFrame":
    """
    Identify bridging nodes that connect communities by ranking on
    betweenness centrality (with degree and clustering as auxiliaries).

    Parameters
    ----------
    G : nx.Graph
        Input (typically backbone) graph.
    top_n : int
        Top-N nodes to return.
    weight_attr : str
        Edge weight attribute used in betweenness computation.

    Returns
    -------
    DataFrame
        Columns: node, degree, betweenness, clustering, bridging_score.
        Sorted by betweenness descending. Bridging score = betweenness
        x (1 - clustering): high betweenness + low clustering = bridge.
    """
    bw = nx.betweenness_centrality(G, weight=weight_attr, normalized=True)
    cl = nx.clustering(G, weight=weight_attr)
    rows = []
    for n in G.nodes():
        rows.append({
            "node": n,
            "degree": G.degree(n),
            "betweenness": bw.get(n, 0.0),
            "clustering": cl.get(n, 0.0),
            "bridging_score": (bw.get(n, 0.0) * (1 - cl.get(n, 0.0))),
        })
    out = pd.DataFrame(rows).sort_values(
        "betweenness", ascending=False
    ).head(top_n).reset_index(drop=True)
    return out
