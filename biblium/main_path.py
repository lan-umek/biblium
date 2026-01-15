# -*- coding: utf-8 -*-
"""
Main Path Analysis
==================
Analyze the main knowledge flow paths in citation networks.

Main Path Analysis (Hummon & Doreian, 1989) identifies the most significant 
paths of knowledge diffusion in a citation network.

Features:
- Multiple traversal weights: SPC, SPLC, SPNP
- Forward, backward, and global main paths
- Key-route main path (multiple paths)
- Path statistics and visualization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
import warnings

import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class MainPathResult:
    """Result of main path analysis."""
    # Graph info
    n_nodes: int = 0
    n_edges: int = 0
    n_sources: int = 0  # Nodes with no incoming edges
    n_sinks: int = 0    # Nodes with no outgoing edges
    
    # Main paths
    global_main_path: List[str] = field(default_factory=list)
    forward_main_path: List[str] = field(default_factory=list)
    backward_main_path: List[str] = field(default_factory=list)
    key_routes: List[List[str]] = field(default_factory=list)
    
    # Path info
    path_length: int = 0
    path_documents: List[Dict] = field(default_factory=list)
    
    # Weights
    weight_method: str = "SPC"
    edge_weights: Dict[Tuple[str, str], float] = field(default_factory=dict)
    node_weights: Dict[str, float] = field(default_factory=dict)
    
    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)


def compute_traversal_weights(
    G: "nx.DiGraph",
    method: Literal["SPC", "SPLC", "SPNP"] = "SPC",
) -> Dict[Tuple[str, str], float]:
    """
    Compute edge traversal weights for main path analysis.
    
    Parameters
    ----------
    G : nx.DiGraph
        Citation network (directed, acyclic preferred)
    method : str
        Weight method:
        - "SPC": Search Path Count - number of paths through edge
        - "SPLC": Search Path Link Count - normalized by path length  
        - "SPNP": Search Path Node Pair - normalized by node pairs
    
    Returns
    -------
    dict
        Edge weights {(u, v): weight}
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required")
    
    # Ensure DAG by condensing cycles
    if not nx.is_directed_acyclic_graph(G):
        # Condense strongly connected components
        G = nx.condensation(G)
    
    # Find sources (no incoming) and sinks (no outgoing)
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    if not sources or not sinks:
        return {}
    
    # Count paths from sources to each node
    paths_from_source = {n: 0 for n in G.nodes()}
    for source in sources:
        paths_from_source[source] = 1
    
    # Topological traversal
    for node in nx.topological_sort(G):
        for pred in G.predecessors(node):
            paths_from_source[node] += paths_from_source[pred]
    
    # Count paths from each node to sinks
    paths_to_sink = {n: 0 for n in G.nodes()}
    for sink in sinks:
        paths_to_sink[sink] = 1
    
    # Reverse topological traversal
    for node in reversed(list(nx.topological_sort(G))):
        for succ in G.successors(node):
            paths_to_sink[node] += paths_to_sink[succ]
    
    # Total paths through network
    total_paths = sum(paths_from_source[sink] for sink in sinks)
    
    # Compute edge weights
    edge_weights = {}
    
    for u, v in G.edges():
        # SPC: paths through this edge
        spc = paths_from_source[u] * paths_to_sink[v]
        
        if method == "SPC":
            edge_weights[(u, v)] = spc
        elif method == "SPLC":
            # Normalize by total paths
            edge_weights[(u, v)] = spc / total_paths if total_paths > 0 else 0
        elif method == "SPNP":
            # Normalize by number of source-sink pairs
            n_pairs = len(sources) * len(sinks)
            edge_weights[(u, v)] = spc / n_pairs if n_pairs > 0 else 0
    
    return edge_weights


def find_global_main_path(
    G: "nx.DiGraph",
    edge_weights: Dict[Tuple[str, str], float],
) -> List[str]:
    """
    Find the global main path (highest total weight path from source to sink).
    
    Uses dynamic programming to find the path with maximum sum of edge weights.
    """
    if not edge_weights:
        return []
    
    # Ensure DAG
    if not nx.is_directed_acyclic_graph(G):
        G = nx.condensation(G)
    
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    if not sources or not sinks:
        return []
    
    # Dynamic programming: max weight to reach each node
    max_weight = {n: float('-inf') for n in G.nodes()}
    predecessor = {n: None for n in G.nodes()}
    
    for source in sources:
        max_weight[source] = 0
    
    for node in nx.topological_sort(G):
        for succ in G.successors(node):
            edge_w = edge_weights.get((node, succ), 0)
            new_weight = max_weight[node] + edge_w
            if new_weight > max_weight[succ]:
                max_weight[succ] = new_weight
                predecessor[succ] = node
    
    # Find best sink
    best_sink = max(sinks, key=lambda s: max_weight[s])
    
    if max_weight[best_sink] == float('-inf'):
        return []
    
    # Backtrack to get path
    path = []
    node = best_sink
    while node is not None:
        path.append(node)
        node = predecessor[node]
    
    return list(reversed(path))


def find_forward_main_path(
    G: "nx.DiGraph",
    edge_weights: Dict[Tuple[str, str], float],
    start_node: str = None,
) -> List[str]:
    """
    Find forward main path: always follow highest-weight outgoing edge.
    """
    if not edge_weights:
        return []
    
    # Start from source with highest weight edges
    if start_node is None:
        sources = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not sources:
            return []
        
        # Find source with highest total outgoing weight
        best_source = max(sources, key=lambda s: sum(
            edge_weights.get((s, succ), 0) for succ in G.successors(s)
        ))
        start_node = best_source
    
    path = [start_node]
    current = start_node
    visited = {start_node}
    
    while True:
        successors = list(G.successors(current))
        if not successors:
            break
        
        # Find highest weight successor
        best_succ = None
        best_weight = -1
        for succ in successors:
            if succ not in visited:
                w = edge_weights.get((current, succ), 0)
                if w > best_weight:
                    best_weight = w
                    best_succ = succ
        
        if best_succ is None:
            break
        
        path.append(best_succ)
        visited.add(best_succ)
        current = best_succ
    
    return path


def find_backward_main_path(
    G: "nx.DiGraph",
    edge_weights: Dict[Tuple[str, str], float],
    end_node: str = None,
) -> List[str]:
    """
    Find backward main path: trace back from sink following highest weights.
    """
    if not edge_weights:
        return []
    
    # Start from sink with highest weight edges
    if end_node is None:
        sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
        if not sinks:
            return []
        
        # Find sink with highest total incoming weight
        best_sink = max(sinks, key=lambda s: sum(
            edge_weights.get((pred, s), 0) for pred in G.predecessors(s)
        ))
        end_node = best_sink
    
    path = [end_node]
    current = end_node
    visited = {end_node}
    
    while True:
        predecessors = list(G.predecessors(current))
        if not predecessors:
            break
        
        # Find highest weight predecessor
        best_pred = None
        best_weight = -1
        for pred in predecessors:
            if pred not in visited:
                w = edge_weights.get((pred, current), 0)
                if w > best_weight:
                    best_weight = w
                    best_pred = pred
        
        if best_pred is None:
            break
        
        path.append(best_pred)
        visited.add(best_pred)
        current = best_pred
    
    return list(reversed(path))


def find_key_routes(
    G: "nx.DiGraph",
    edge_weights: Dict[Tuple[str, str], float],
    threshold: float = 0.5,
    max_routes: int = 5,
) -> List[List[str]]:
    """
    Find key-route main paths: multiple significant paths.
    
    Parameters
    ----------
    G : nx.DiGraph
        Citation network
    edge_weights : dict
        Edge traversal weights
    threshold : float
        Weight threshold as fraction of max weight
    max_routes : int
        Maximum number of routes to return
    
    Returns
    -------
    list
        List of paths
    """
    if not edge_weights:
        return []
    
    # Find max edge weight
    max_weight = max(edge_weights.values()) if edge_weights else 0
    if max_weight == 0:
        return []
    
    cutoff = threshold * max_weight
    
    # Create subgraph with only high-weight edges
    high_weight_edges = [(u, v) for (u, v), w in edge_weights.items() if w >= cutoff]
    
    if not high_weight_edges:
        return []
    
    subG = G.edge_subgraph(high_weight_edges).copy()
    
    # Find all paths from sources to sinks in subgraph
    sources = [n for n in subG.nodes() if subG.in_degree(n) == 0]
    sinks = [n for n in subG.nodes() if subG.out_degree(n) == 0]
    
    if not sources or not sinks:
        return []
    
    routes = []
    for source in sources:
        for sink in sinks:
            try:
                # Find all simple paths (limit to avoid explosion)
                for path in nx.all_simple_paths(subG, source, sink, cutoff=50):
                    if len(path) >= 3:  # At least 3 nodes
                        routes.append(path)
                        if len(routes) >= max_routes * 3:
                            break
            except nx.NetworkXNoPath:
                continue
        if len(routes) >= max_routes * 3:
            break
    
    # Sort by path weight and return top routes
    def path_weight(path):
        return sum(edge_weights.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
    
    routes.sort(key=path_weight, reverse=True)
    
    # Remove duplicates and return top
    seen = set()
    unique_routes = []
    for route in routes:
        key = tuple(route)
        if key not in seen:
            seen.add(key)
            unique_routes.append(route)
            if len(unique_routes) >= max_routes:
                break
    
    return unique_routes


def compute_main_path_analysis(
    G: "nx.DiGraph",
    method: Literal["SPC", "SPLC", "SPNP"] = "SPC",
    node_data: Dict[str, Dict] = None,
    verbose: bool = True,
) -> MainPathResult:
    """
    Perform complete main path analysis.
    
    Parameters
    ----------
    G : nx.DiGraph
        Citation network (edges go from cited to citing)
    method : str
        Traversal weight method: SPC, SPLC, SPNP
    node_data : dict
        Optional node attributes {node_id: {attr: value}}
    verbose : bool
        Print progress
    
    Returns
    -------
    MainPathResult
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for main path analysis")
    
    result = MainPathResult(weight_method=method)
    
    # Basic stats
    result.n_nodes = G.number_of_nodes()
    result.n_edges = G.number_of_edges()
    
    if result.n_nodes == 0:
        if verbose:
            print("Empty graph")
        return result
    
    # Handle cycles by condensation if needed
    original_G = G
    condensed_to_original = None  # Maps condensed node ID -> list of original nodes
    
    if not nx.is_directed_acyclic_graph(G):
        if verbose:
            print("Graph has cycles, condensing...")
        condensed_G = nx.condensation(G)
        
        # Build mapping from condensed node to original nodes
        condensed_to_original = {}
        for cnode in condensed_G.nodes():
            members = list(condensed_G.nodes[cnode]['members'])
            condensed_to_original[cnode] = members
        
        G = condensed_G
    
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    result.n_sources = len(sources)
    result.n_sinks = len(sinks)
    
    if verbose:
        print(f"Main Path Analysis ({method})")
        print(f"  Nodes: {result.n_nodes}, Edges: {result.n_edges}")
        print(f"  Sources: {result.n_sources}, Sinks: {result.n_sinks}")
    
    # Compute traversal weights on (possibly condensed) graph
    edge_weights = compute_traversal_weights(G, method=method)
    
    if not edge_weights:
        if verbose:
            print("  No valid paths found")
        return result
    
    # Find main paths on (possibly condensed) graph
    condensed_global = find_global_main_path(G, edge_weights)
    condensed_forward = find_forward_main_path(G, edge_weights)
    condensed_backward = find_backward_main_path(G, edge_weights)
    condensed_routes = find_key_routes(G, edge_weights)
    
    # Function to expand condensed path back to original nodes
    def expand_path(condensed_path):
        if condensed_to_original is None:
            return condensed_path
        
        expanded = []
        for cnode in condensed_path:
            members = condensed_to_original.get(cnode, [cnode])
            if members:
                # Pick representative: prefer by year if node_data available
                if node_data and len(members) > 1:
                    members_sorted = sorted(members, 
                        key=lambda m: node_data.get(m, {}).get('year', 2000))
                    expanded.append(members_sorted[0])
                else:
                    expanded.append(members[0])
        return expanded
    
    # Map paths back to original node IDs
    result.global_main_path = expand_path(condensed_global)
    result.forward_main_path = expand_path(condensed_forward)
    result.backward_main_path = expand_path(condensed_backward)
    result.key_routes = [expand_path(route) for route in condensed_routes]
    
    # Compute node weights on ORIGINAL graph
    node_weights = {}
    for node in original_G.nodes():
        in_deg = original_G.in_degree(node)
        out_deg = original_G.out_degree(node)
        node_weights[node] = in_deg + out_deg
    result.node_weights = node_weights
    
    # Store edge weights mapped back to original if needed
    result.edge_weights = {}
    for (u, v), w in edge_weights.items():
        if condensed_to_original:
            u_orig = condensed_to_original.get(u, [u])[0]
            v_orig = condensed_to_original.get(v, [v])[0]
            result.edge_weights[(u_orig, v_orig)] = w
        else:
            result.edge_weights[(u, v)] = w
    
    # Use global main path as primary
    main_path = result.global_main_path
    result.path_length = len(main_path)
    
    if verbose:
        print(f"  Main path length: {result.path_length}")
        if main_path:
            path_preview = [str(n)[:20] for n in main_path[:5]]
            print(f"  Path: {' → '.join(path_preview)}...")
    
    # Extract document info for main path
    if node_data and main_path:
        for node in main_path:
            if node in node_data:
                result.path_documents.append(node_data[node])
    
    # Statistics
    result.statistics = {
        'total_traversal_count': sum(edge_weights.values()),
        'max_edge_weight': max(edge_weights.values()) if edge_weights else 0,
        'avg_edge_weight': np.mean(list(edge_weights.values())) if edge_weights else 0,
        'max_node_weight': max(node_weights.values()) if node_weights else 0,
        'avg_path_length': result.path_length,
    }
    
    return result


def plot_main_path(
    G: "nx.DiGraph",
    main_path: List[str],
    node_data: Dict[str, Dict] = None,
    layout: Literal["chronological", "spring", "kamada_kawai"] = "chronological",
    figsize: Tuple[int, int] = (12, 8),
    highlight_color: str = "#e74c3c",
    node_color: str = "#3498db",
    title: str = "Main Path Analysis",
) -> "matplotlib.figure.Figure":
    """
    Visualize main path in citation network.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No data to display", ha='center', va='center')
        return fig
    
    # Filter main_path to only nodes that exist in G
    valid_main_path = [n for n in main_path if n in G.nodes()]
    
    # Compute layout
    if layout == "chronological" and node_data:
        # Position by year (x) and spread (y)
        pos = {}
        years = {}
        for node in G.nodes():
            if node in node_data:
                years[node] = node_data[node].get('year', 2000)
            else:
                years[node] = 2000
        
        if years:
            min_year = min(years.values())
            max_year = max(years.values())
            year_range = max(1, max_year - min_year)
        else:
            min_year, max_year, year_range = 2000, 2000, 1
        
        # Group by year
        year_groups = {}
        for node, year in years.items():
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(node)
        
        for year, nodes in year_groups.items():
            x = (year - min_year) / year_range if year_range > 0 else 0.5
            n_nodes = len(nodes)
            for i, node in enumerate(nodes):
                y = (i + 1) / (n_nodes + 1)
                pos[node] = (x, y)
    elif layout == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, k=2, iterations=50)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw all edges (light)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, 
                          edge_color='gray', arrows=True, arrowsize=10)
    
    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, 
                          node_size=100, alpha=0.6)
    
    # Highlight main path - only for nodes/edges that exist in graph and layout
    if valid_main_path and len(valid_main_path) > 1:
        # Only create edges for consecutive nodes that both exist in pos and graph
        path_edges = []
        for i in range(len(valid_main_path) - 1):
            u, v = valid_main_path[i], valid_main_path[i+1]
            if u in pos and v in pos:
                # Check if edge exists (in either direction for visualization)
                if G.has_edge(u, v) or G.has_edge(v, u):
                    path_edges.append((u, v))
        
        if path_edges:
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax,
                                  edge_color=highlight_color, width=3, 
                                  arrows=True, arrowsize=15)
        
        # Only highlight nodes that exist in layout
        path_nodes_in_pos = [n for n in valid_main_path if n in pos]
        if path_nodes_in_pos:
            nx.draw_networkx_nodes(G, pos, nodelist=path_nodes_in_pos, ax=ax,
                                  node_color=highlight_color, node_size=200)
            
            # Labels for main path nodes
            labels = {n: str(n)[:15] for n in path_nodes_in_pos}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.grid(False)
    
    plt.tight_layout()
    return fig
