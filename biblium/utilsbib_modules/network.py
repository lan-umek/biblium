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

from typing import Dict, List, Optional, Tuple, Union, Any
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
    matrix: pd.DataFrame,
    method: str = "association"
) -> pd.DataFrame:
    """
    Normalize a symmetric co-occurrence matrix.

    Parameters
    ----------
    matrix : DataFrame
        Symmetric matrix.
    method : {"association", "inclusion", "jaccard", "salton"}
        Normalization method.

    Returns
    -------
    DataFrame
        Normalized matrix.
    """
    # Diagonal contains self-occurrences (or we estimate from row sums)
    diag = np.diag(matrix.values)
    if diag.sum() == 0:
        # Estimate from matrix
        diag = matrix.sum(axis=1).values
    
    n = len(matrix)
    result = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                result[i, j] = 1.0
            else:
                cooc = matrix.iloc[i, j]
                ni, nj = diag[i], diag[j]
                
                if method == "association":
                    denom = np.sqrt(ni * nj)
                elif method == "inclusion":
                    denom = min(ni, nj)
                elif method == "jaccard":
                    denom = ni + nj - cooc
                elif method == "salton":
                    denom = np.sqrt(ni * nj)
                else:
                    denom = 1
                
                result[i, j] = cooc / denom if denom > 0 else 0
    
    return pd.DataFrame(result, index=matrix.index, columns=matrix.columns)


# =============================================================================
# COMMUNITY DETECTION
# =============================================================================

def louvain_partition(G: nx.Graph) -> Dict[Any, int]:
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
    
    return community_louvain.best_partition(G)


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
    G: nx.Graph,
    methods: Optional[List[str]] = None
) -> nx.Graph:
    """
    Add community partitions as node attributes.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    methods : list, optional
        Partition methods to apply. Default: ["louvain", "greedy"].

    Returns
    -------
    nx.Graph
        Graph with partition attributes added.
    """
    if methods is None:
        methods = ["greedy"]
        if LOUVAIN_AVAILABLE:
            methods.insert(0, "louvain")
    
    method_funcs = {
        "louvain": louvain_partition if LOUVAIN_AVAILABLE else None,
        "greedy": greedy_modularity_partition,
        "label_propagation": label_propagation_partition,
    }
    
    for method in methods:
        func = method_funcs.get(method)
        if func is None:
            continue
        
        try:
            partition = func(G)
            nx.set_node_attributes(G, partition, f"partition_{method}")
        except Exception:
            pass
    
    return G


# =============================================================================
# NETWORK ANALYSIS
# =============================================================================

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
    G: nx.Graph,
    filepath: str,
    format: str = "graphml"
) -> None:
    """
    Save a network to file.

    Parameters
    ----------
    G : nx.Graph
        Network to save.
    filepath : str
        Output file path.
    format : {"graphml", "gexf", "pajek", "edgelist"}
        Output format.
    """
    if format == "graphml":
        nx.write_graphml(G, filepath)
    elif format == "gexf":
        nx.write_gexf(G, filepath)
    elif format == "pajek":
        nx.write_pajek(G, filepath)
    elif format == "edgelist":
        nx.write_weighted_edgelist(G, filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_to_pajek(
    G: nx.Graph,
    filepath: str,
    partition_attr: Optional[str] = None
) -> None:
    """
    Save network in Pajek format with optional partition.

    Parameters
    ----------
    G : nx.Graph
        Network to save.
    filepath : str
        Output file path.
    partition_attr : str, optional
        Node attribute to use as partition.
    """
    # Write network
    nx.write_pajek(G, filepath)
    
    # Write partition file if specified
    if partition_attr:
        partition_file = filepath.replace(".net", ".clu")
        nodes = list(G.nodes())
        
        with open(partition_file, "w") as f:
            f.write(f"*Vertices {len(nodes)}\n")
            for node in nodes:
                cluster = G.nodes[node].get(partition_attr, 0)
                f.write(f"{cluster}\n")
