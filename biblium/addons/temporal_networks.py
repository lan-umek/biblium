# -*- coding: utf-8 -*-
"""
Temporal Network Analysis Module for Bibliometric Analysis

This module provides functions to analyze how bibliometric networks evolve over time.
It includes snapshot generation, temporal metrics, community tracking, and visualizations.

Features:
- Snapshot network generation (co-authorship, keyword co-occurrence, citation, country collaboration)
- Temporal node/edge metrics (persistence, centrality trajectories, lifecycle stages)
- Community dynamics (birth, death, merge, split detection)
- Network evolution statistics
- Rich visualizations (animated networks, timeline plots, alluvial diagrams)

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, Set
from itertools import combinations
import json

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import entropy, spearmanr, pearsonr
from scipy.spatial.distance import cosine, jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import seaborn as sns

# Import core biblium bridge
try:
    from biblium.addons.core_utils import get_core_function, CORE_AVAILABLE, use_core_or_fallback
except ImportError:
    CORE_AVAILABLE = False
    def get_core_function(name): return None
    def use_core_or_fallback(name, fallback, *args, **kwargs): return fallback(*args, **kwargs)

# Optional imports
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

try:
    from pyvis.network import Network as PyvisNetwork
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# =============================================================================
# DEFAULT COLORMAPS (user-configurable)
# =============================================================================

CMAP_CONTINUOUS = "viridis"  # For continuous/sequential data
CATEGORICAL_COLOR = "lightblue"      # For categorical/discrete data

def set_default_cmaps(continuous: str = None):
    """Set default colormap for continuous data."""
    global CMAP_CONTINUOUS
    if continuous:
        CMAP_CONTINUOUS = continuous

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TemporalNode:
    """Represents a node's properties across time."""
    node_id: str
    first_appearance: str  # Period string
    last_appearance: str
    active_periods: List[str]
    degree_trajectory: Dict[str, int]
    centrality_trajectories: Dict[str, Dict[str, float]]  # metric_name -> {period: value}
    community_trajectory: Dict[str, Any]  # period -> community_id
    total_activity: int  # Sum of degrees across all periods
    
    def is_active_in(self, period: str) -> bool:
        return period in self.active_periods
    
    def get_lifespan(self, all_periods: List[str]) -> int:
        """Number of periods between first and last appearance."""
        if self.first_appearance not in all_periods or self.last_appearance not in all_periods:
            return 0
        start_idx = all_periods.index(self.first_appearance)
        end_idx = all_periods.index(self.last_appearance)
        return end_idx - start_idx + 1
    
    def get_activity_ratio(self, all_periods: List[str]) -> float:
        """Fraction of lifespan periods where node was active."""
        lifespan = self.get_lifespan(all_periods)
        if lifespan == 0:
            return 0.0
        return len(self.active_periods) / lifespan

@dataclass
class TemporalEdge:
    """Represents an edge's properties across time."""
    source: str
    target: str
    first_appearance: str
    last_appearance: str
    active_periods: List[str]
    weight_trajectory: Dict[str, float]  # period -> weight
    total_weight: float
    
    def get_persistence(self, all_periods: List[str]) -> float:
        """Fraction of possible periods where edge existed."""
        if not all_periods:
            return 0.0
        return len(self.active_periods) / len(all_periods)
    
    def is_recurring(self) -> bool:
        """Check if edge disappeared and reappeared."""
        if len(self.active_periods) < 2:
            return False
        # Check for gaps in active periods
        # This is a simplified check - assumes periods are ordered in active_periods
        return True  # Simplified; full implementation would check for actual gaps

@dataclass
class CommunityEvent:
    """Represents a community lifecycle event."""
    event_type: str  # "birth", "death", "merge", "split", "grow", "shrink", "stable"
    period: str
    communities_before: List[Any]  # Community IDs before event
    communities_after: List[Any]  # Community IDs after event
    nodes_involved: Set[str]
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NetworkSnapshot:
    """A network at a specific time period."""
    period: str
    start_year: int
    end_year: int
    graph: nx.Graph
    n_nodes: int
    n_edges: int
    density: float
    communities: Optional[Dict[str, int]] = None  # node -> community_id
    node_metrics: Optional[pd.DataFrame] = None
    
    def get_community_sizes(self) -> Dict[int, int]:
        """Get size of each community."""
        if self.communities is None:
            return {}
        sizes = Counter(self.communities.values())
        return dict(sizes)

@dataclass
class TemporalNetworkAnalysis:
    """Container for complete temporal network analysis results."""
    network_type: str  # "coauthorship", "keyword", "citation", "country"
    snapshots: Dict[str, NetworkSnapshot]
    periods: List[str]
    temporal_nodes: Dict[str, TemporalNode]
    temporal_edges: Dict[Tuple[str, str], TemporalEdge]
    community_events: List[CommunityEvent]
    global_metrics: pd.DataFrame  # Period-level network metrics
    parameters: Dict[str, Any]
    
    def get_node(self, node_id: str) -> Optional[TemporalNode]:
        return self.temporal_nodes.get(node_id)
    
    def get_edge(self, source: str, target: str) -> Optional[TemporalEdge]:
        key = (source, target) if (source, target) in self.temporal_edges else (target, source)
        return self.temporal_edges.get(key)
    
    def get_snapshot(self, period: str) -> Optional[NetworkSnapshot]:
        return self.snapshots.get(period)
    
    def get_persistent_nodes(self, min_ratio: float = 0.8) -> List[str]:
        """Get nodes active in at least min_ratio of their lifespan."""
        persistent = []
        for node_id, tnode in self.temporal_nodes.items():
            if tnode.get_activity_ratio(self.periods) >= min_ratio:
                persistent.append(node_id)
        return persistent
    
    def get_persistent_edges(self, min_ratio: float = 0.5) -> List[Tuple[str, str]]:
        """Get edges that persist across at least min_ratio of periods."""
        persistent = []
        for edge_key, tedge in self.temporal_edges.items():
            if tedge.get_persistence(self.periods) >= min_ratio:
                persistent.append(edge_key)
        return persistent
    
    def to_summary_df(self) -> pd.DataFrame:
        """Get summary DataFrame of network evolution."""
        return self.global_metrics.copy()

# =============================================================================
# CORE FUNCTIONS: SNAPSHOT GENERATION
# =============================================================================

def create_time_windows(
    df: pd.DataFrame,
    year_col: str = "Year",
    window_size: int = 5,
    window_type: str = "fixed",
    slide_step: int = 1,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None) -> List[Tuple[int, int, pd.DataFrame]]:
    """
    Split dataframe into time windows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with year column.
    year_col : str
        Name of the year column.
    window_size : int
        Size of each window in years.
    window_type : str
        "fixed" for non-overlapping, "sliding" for overlapping windows.
    slide_step : int
        Step size for sliding windows.
    min_year, max_year : int, optional
        Override automatic year range detection.
    
    Returns
    -------
    List of (start_year, end_year, subset_df) tuples.
    """
    df = df.dropna(subset=[year_col]).copy()
    df[year_col] = df[year_col].astype(int)
    
    if min_year is None:
        min_year = df[year_col].min()
    if max_year is None:
        max_year = df[year_col].max()
    
    windows = []
    
    if window_type == "fixed":
        start = min_year
        while start <= max_year:
            end = min(start + window_size - 1, max_year)
            mask = (df[year_col] >= start) & (df[year_col] <= end)
            subset = df[mask].copy()
            if len(subset) > 0:
                windows.append((start, end, subset))
            start = end + 1
    else:  # sliding
        start = min_year
        while start + window_size - 1 <= max_year:
            end = start + window_size - 1
            mask = (df[year_col] >= start) & (df[year_col] <= end)
            subset = df[mask].copy()
            if len(subset) > 0:
                windows.append((start, end, subset))
            start += slide_step
    
    return windows

def build_cooccurrence_network(
    df: pd.DataFrame,
    column: str,
    sep: str = "; ",
    min_weight: int = 1,
    normalize_weights: bool = False) -> nx.Graph:
    """
    Build a co-occurrence network from a list-type column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column containing list-type data (keywords, authors, etc.).
    sep : str
        Separator for splitting string values.
    min_weight : int
        Minimum edge weight to include.
    normalize_weights : bool
        If True, normalize weights by total co-occurrences.
    
    Returns
    -------
    nx.Graph
        Undirected weighted graph.
    """
    G = nx.Graph()
    edge_weights = Counter()
    
    for _, row in df.iterrows():
        value = row.get(column)
        if pd.isna(value):
            continue
        
        # Parse items
        if isinstance(value, str):
            items = [x.strip() for x in value.split(sep) if x.strip()]
        elif isinstance(value, (list, tuple, set)):
            items = [str(x).strip() for x in value if x]
        else:
            continue
        
        # Add nodes
        for item in items:
            if item not in G:
                G.add_node(item)
        
        # Count co-occurrences
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                if item1 != item2:
                    edge = tuple(sorted([item1, item2]))
                    edge_weights[edge] += 1
    
    # Add edges
    total_weight = sum(edge_weights.values()) if normalize_weights else 1
    
    for (u, v), weight in edge_weights.items():
        if weight >= min_weight:
            w = weight / total_weight if normalize_weights else weight
            G.add_edge(u, v, weight=w)
    
    return G

def build_citation_network_from_df(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    refs_col: str = "References",
    sep: str = "; ",
    keep_internal_only: bool = True) -> nx.DiGraph:
    """
    Build a citation network from document references.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_col : str
        Column with document identifiers.
    refs_col : str
        Column with references (list of cited document IDs).
    sep : str
        Separator for reference strings.
    keep_internal_only : bool
        If True, only keep edges where both nodes are in the dataset.
    
    Returns
    -------
    nx.DiGraph
        Directed citation network (edges point from citing to cited).
    """
    G = nx.DiGraph()
    
    # Get all document IDs
    all_ids = set(df[id_col].dropna().astype(str))
    
    for _, row in df.iterrows():
        doc_id = str(row.get(id_col, ""))
        refs = row.get(refs_col)
        
        if pd.isna(refs) or not doc_id:
            continue
        
        # Add citing document
        G.add_node(doc_id)
        
        # Parse references
        if isinstance(refs, str):
            ref_list = [x.strip() for x in refs.split(sep) if x.strip()]
        elif isinstance(refs, (list, tuple)):
            ref_list = [str(x).strip() for x in refs if x]
        else:
            continue
        
        # Add edges
        for ref in ref_list:
            if keep_internal_only and ref not in all_ids:
                continue
            G.add_node(ref)
            G.add_edge(doc_id, ref)  # citing -> cited
    
    return G

def generate_network_snapshots(
    df: pd.DataFrame,
    network_type: str = "keyword",
    column: str = None,
    year_col: str = "Year",
    window_size: int = 5,
    window_type: str = "fixed",
    sep: str = "; ",
    min_weight: int = 1,
    min_docs_per_window: int = 20,
    compute_communities: bool = True,
    community_method: str = "louvain",
    top_n_nodes: Optional[int] = None,
    **kwargs) -> Dict[str, NetworkSnapshot]:
    """
    Generate network snapshots across time periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    network_type : str
        Type of network: "keyword", "coauthorship", "citation", "country".
    column : str, optional
        Column to use for network construction. Auto-detected if None.
    year_col : str
        Year column name.
    window_size : int
        Time window size in years.
    window_type : str
        "fixed" or "sliding".
    sep : str
        Separator for list-type columns.
    min_weight : int
        Minimum edge weight.
    min_docs_per_window : int
        Minimum documents per window.
    compute_communities : bool
        Whether to detect communities.
    community_method : str
        Community detection method: "louvain", "label_propagation", "greedy_modularity".
    top_n_nodes : int, optional
        If set, keep only top N nodes by total degree across periods.
    **kwargs
        Additional arguments for network construction.
    
    Returns
    -------
    Dict mapping period strings to NetworkSnapshot objects.
    """
    # Auto-detect column based on network type
    if column is None:
        column_map = {
            "keyword": [
                "Processed Author Keywords", "Author Keywords", "Keywords",
                "Processed Index Keywords", "Index Keywords",
                "Processed Author and Index Keywords", "Author and Index Keywords",
                "author_keywords", "keywords", "index_keywords",
            ],
            "coauthorship": [
                "Author full names", "Authors", "authors", 
                "authorships.author.display_name", "author_names",
            ],
            "country": [
                "Countries of Authors", "Country", "countries",
                "authorships.countries", "author_countries",
            ],
            "citation": [
                "References", "referenced_works", "references",
                "cited_references", "bibliography",
            ],
        }
        
        # Try exact matches first
        for col in column_map.get(network_type, []):
            if col in df.columns:
                column = col
                break
        
        # Try case-insensitive partial matching
        if column is None:
            search_terms = {
                "keyword": ["keyword", "kw"],
                "coauthorship": ["author", "writer"],
                "country": ["countr", "nation"],
                "citation": ["refer", "cited", "bibliog"],
            }
            for col in df.columns:
                col_lower = col.lower()
                for term in search_terms.get(network_type, []):
                    if term in col_lower:
                        column = col
                        break
                if column:
                    break
        
        if column is None:
            # Provide helpful error message
            available_cols = [c for c in df.columns if df[c].dtype == 'object'][:20]
            raise ValueError(
                f"Could not auto-detect column for network_type='{network_type}'.\n"
                f"Please specify the column explicitly using the 'column' parameter.\n"
                f"Available text columns (first 20): {available_cols}"
            )
        
        print(f"  Auto-detected column: '{column}'")
    
    # Create time windows
    windows = create_time_windows(df, year_col, window_size, window_type)
    windows = [(s, e, d) for s, e, d in windows if len(d) >= min_docs_per_window]
    
    if not windows:
        raise ValueError("No time windows with sufficient data.")
    
    snapshots = {}
    
    for start_year, end_year, subset_df in windows:
        period = f"{start_year}-{end_year}"
        
        # Build network based on type
        if network_type == "citation":
            G = build_citation_network_from_df(subset_df, sep=sep, **kwargs)
            # Convert to undirected for some metrics
            G_undirected = G.to_undirected()
        else:
            G = build_cooccurrence_network(subset_df, column, sep=sep, min_weight=min_weight, **kwargs)
            G_undirected = G
        
        # Filter to top N nodes if requested
        if top_n_nodes and len(G) > top_n_nodes:
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.keys(), key=lambda x: -degrees[x])[:top_n_nodes]
            G = G.subgraph(top_nodes).copy()
            G_undirected = G_undirected.subgraph(top_nodes).copy() if network_type == "citation" else G
        
        # Compute basic metrics
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G) if n_nodes > 1 else 0.0
        
        # Detect communities
        communities = None
        if compute_communities and n_nodes > 0:
            communities = detect_communities(G_undirected, method=community_method)
        
        # Create snapshot
        snapshot = NetworkSnapshot(
            period=period,
            start_year=start_year,
            end_year=end_year,
            graph=G,
            n_nodes=n_nodes,
            n_edges=n_edges,
            density=density,
            communities=communities)
        
        snapshots[period] = snapshot
    
    return snapshots

def detect_communities(
    G: nx.Graph,
    method: str = "louvain",
    resolution: float = 1.0) -> Dict[str, int]:
    """
    Detect communities in a network.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph (should be undirected for most methods).
    method : str
        Detection method: "louvain", "label_propagation", "greedy_modularity".
    resolution : float
        Resolution parameter for Louvain.
    
    Returns
    -------
    Dict mapping node to community ID.
    """
    if G.number_of_nodes() == 0:
        return {}
    
    # Handle disconnected components
    if not nx.is_connected(G):
        # Process each component separately
        communities = {}
        community_offset = 0
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(component) < 3:
                # Assign small components their own community
                for node in component:
                    communities[node] = community_offset
                community_offset += 1
            else:
                sub_communities = _detect_communities_connected(subgraph, method, resolution)
                for node, comm in sub_communities.items():
                    communities[node] = comm + community_offset
                community_offset += max(sub_communities.values()) + 1 if sub_communities else 1
        return communities
    
    return _detect_communities_connected(G, method, resolution)

def _detect_communities_connected(
    G: nx.Graph,
    method: str,
    resolution: float) -> Dict[str, int]:
    """Detect communities in a connected graph."""
    if method == "louvain":
        if LOUVAIN_AVAILABLE:
            return community_louvain.best_partition(G, resolution=resolution)
        else:
            # Fallback to greedy modularity
            method = "greedy_modularity"
    
    if method == "label_propagation":
        communities = nx.community.label_propagation_communities(G)
        result = {}
        for i, comm in enumerate(communities):
            for node in comm:
                result[node] = i
        return result
    
    if method == "greedy_modularity":
        communities = nx.community.greedy_modularity_communities(G)
        result = {}
        for i, comm in enumerate(communities):
            for node in comm:
                result[node] = i
        return result
    
    raise ValueError(f"Unknown community detection method: {method}")

# =============================================================================
# TEMPORAL METRICS
# =============================================================================

def compute_temporal_node_metrics(
    snapshots: Dict[str, NetworkSnapshot],
    periods: List[str],
    centrality_measures: List[str] = None) -> Dict[str, TemporalNode]:
    """
    Compute temporal metrics for all nodes across snapshots.
    
    Parameters
    ----------
    snapshots : Dict[str, NetworkSnapshot]
        Network snapshots by period.
    periods : List[str]
        Ordered list of periods.
    centrality_measures : List[str], optional
        Which centrality measures to compute. Default: ["degree", "betweenness"].
    
    Returns
    -------
    Dict mapping node ID to TemporalNode object.
    """
    if centrality_measures is None:
        centrality_measures = ["degree", "betweenness"]
    
    # Collect all nodes across all periods
    all_nodes = set()
    for snapshot in snapshots.values():
        all_nodes.update(snapshot.graph.nodes())
    
    temporal_nodes = {}
    
    for node_id in all_nodes:
        # Track activity
        active_periods = []
        degree_trajectory = {}
        centrality_trajectories = {m: {} for m in centrality_measures}
        community_trajectory = {}
        
        for period in periods:
            if period not in snapshots:
                continue
            
            snapshot = snapshots[period]
            G = snapshot.graph
            
            if node_id in G:
                active_periods.append(period)
                degree_trajectory[period] = G.degree(node_id)
                
                if snapshot.communities and node_id in snapshot.communities:
                    community_trajectory[period] = snapshot.communities[node_id]
                
                # Compute centralities
                if "degree" in centrality_measures:
                    centrality_trajectories["degree"][period] = G.degree(node_id)
                
                if "betweenness" in centrality_measures and G.number_of_nodes() > 2:
                    try:
                        bc = nx.betweenness_centrality(G)
                        centrality_trajectories["betweenness"][period] = bc.get(node_id, 0)
                    except:
                        centrality_trajectories["betweenness"][period] = 0
                
                if "closeness" in centrality_measures and G.number_of_nodes() > 1:
                    try:
                        cc = nx.closeness_centrality(G)
                        centrality_trajectories["closeness"][period] = cc.get(node_id, 0)
                    except:
                        centrality_trajectories["closeness"][period] = 0
                
                if "eigenvector" in centrality_measures and G.number_of_nodes() > 1:
                    try:
                        ec = nx.eigenvector_centrality_numpy(G)
                        centrality_trajectories["eigenvector"][period] = ec.get(node_id, 0)
                    except:
                        centrality_trajectories["eigenvector"][period] = 0
                
                if "pagerank" in centrality_measures:
                    try:
                        pr = nx.pagerank(G)
                        centrality_trajectories["pagerank"][period] = pr.get(node_id, 0)
                    except:
                        centrality_trajectories["pagerank"][period] = 0
        
        if active_periods:
            temporal_nodes[node_id] = TemporalNode(
                node_id=node_id,
                first_appearance=active_periods[0],
                last_appearance=active_periods[-1],
                active_periods=active_periods,
                degree_trajectory=degree_trajectory,
                centrality_trajectories=centrality_trajectories,
                community_trajectory=community_trajectory,
                total_activity=sum(degree_trajectory.values()))
    
    return temporal_nodes

def compute_temporal_edge_metrics(
    snapshots: Dict[str, NetworkSnapshot],
    periods: List[str]) -> Dict[Tuple[str, str], TemporalEdge]:
    """
    Compute temporal metrics for all edges across snapshots.
    
    Parameters
    ----------
    snapshots : Dict[str, NetworkSnapshot]
        Network snapshots by period.
    periods : List[str]
        Ordered list of periods.
    
    Returns
    -------
    Dict mapping (source, target) tuple to TemporalEdge object.
    """
    # Collect all edges across all periods
    all_edges = set()
    for snapshot in snapshots.values():
        for u, v in snapshot.graph.edges():
            edge = tuple(sorted([str(u), str(v)]))
            all_edges.add(edge)
    
    temporal_edges = {}
    
    for edge in all_edges:
        u, v = edge
        active_periods = []
        weight_trajectory = {}
        
        for period in periods:
            if period not in snapshots:
                continue
            
            G = snapshots[period].graph
            
            # Check if edge exists (handle both directions for undirected)
            if G.has_edge(u, v):
                active_periods.append(period)
                weight = G[u][v].get("weight", 1)
                weight_trajectory[period] = weight
            elif G.has_edge(v, u):
                active_periods.append(period)
                weight = G[v][u].get("weight", 1)
                weight_trajectory[period] = weight
        
        if active_periods:
            temporal_edges[edge] = TemporalEdge(
                source=u,
                target=v,
                first_appearance=active_periods[0],
                last_appearance=active_periods[-1],
                active_periods=active_periods,
                weight_trajectory=weight_trajectory,
                total_weight=sum(weight_trajectory.values()))
    
    return temporal_edges

def compute_global_network_metrics(
    snapshots: Dict[str, NetworkSnapshot],
    periods: List[str]) -> pd.DataFrame:
    """
    Compute global network metrics for each period.
    
    Parameters
    ----------
    snapshots : Dict[str, NetworkSnapshot]
        Network snapshots by period.
    periods : List[str]
        Ordered list of periods.
    
    Returns
    -------
    pd.DataFrame with metrics per period.
    """
    records = []
    
    for period in periods:
        if period not in snapshots:
            continue
        
        snapshot = snapshots[period]
        G = snapshot.graph
        
        record = {
            "Period": period,
            "Start Year": snapshot.start_year,
            "End Year": snapshot.end_year,
            "N_Nodes": snapshot.n_nodes,
            "N_Edges": snapshot.n_edges,
            "Density": snapshot.density,
        }
        
        # Additional metrics
        if G.number_of_nodes() > 0:
            # Average degree
            degrees = [d for n, d in G.degree()]
            record["Avg_Degree"] = np.mean(degrees) if degrees else 0
            record["Max_Degree"] = max(degrees) if degrees else 0
            
            # Clustering
            try:
                record["Avg_Clustering"] = nx.average_clustering(G)
            except:
                record["Avg_Clustering"] = 0
            
            # Components
            if G.is_directed():
                n_components = nx.number_weakly_connected_components(G)
                largest = max(nx.weakly_connected_components(G), key=len) if n_components > 0 else set()
            else:
                n_components = nx.number_connected_components(G)
                largest = max(nx.connected_components(G), key=len) if n_components > 0 else set()
            
            record["N_Components"] = n_components
            record["Largest_Component_Size"] = len(largest)
            record["Largest_Component_Ratio"] = len(largest) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
            
            # Communities
            if snapshot.communities:
                n_communities = len(set(snapshot.communities.values()))
                record["N_Communities"] = n_communities
                
                # Modularity
                try:
                    if n_communities > 1:
                        community_sets = defaultdict(set)
                        for node, comm in snapshot.communities.items():
                            community_sets[comm].add(node)
                        record["Modularity"] = nx.community.modularity(G, community_sets.values())
                    else:
                        record["Modularity"] = 0
                except:
                    record["Modularity"] = None
            else:
                record["N_Communities"] = None
                record["Modularity"] = None
        
        records.append(record)
    
    return pd.DataFrame(records)

# =============================================================================
# COMMUNITY DYNAMICS
# =============================================================================

def track_community_evolution(
    snapshots: Dict[str, NetworkSnapshot],
    periods: List[str],
    match_threshold: float = 0.5) -> Tuple[List[CommunityEvent], pd.DataFrame]:
    """
    Track community evolution and detect events (birth, death, merge, split).
    
    Parameters
    ----------
    snapshots : Dict[str, NetworkSnapshot]
        Network snapshots with communities detected.
    periods : List[str]
        Ordered list of periods.
    match_threshold : float
        Jaccard threshold for matching communities across periods.
    
    Returns
    -------
    Tuple of (list of CommunityEvent, DataFrame of community tracking).
    """
    events = []
    tracking_records = []
    
    # Build community member sets per period
    period_communities = {}
    for period in periods:
        if period not in snapshots or snapshots[period].communities is None:
            continue
        
        comm_members = defaultdict(set)
        for node, comm_id in snapshots[period].communities.items():
            comm_members[comm_id].add(node)
        
        period_communities[period] = dict(comm_members)
    
    # Track across consecutive periods
    prev_period = None
    prev_communities = None
    
    for period in periods:
        if period not in period_communities:
            continue
        
        current_communities = period_communities[period]
        
        if prev_communities is None:
            # First period - all communities are births
            for comm_id, members in current_communities.items():
                events.append(CommunityEvent(
                    event_type="birth",
                    period=period,
                    communities_before=[],
                    communities_after=[comm_id],
                    nodes_involved=members,
                    details={"size": len(members)}
                ))
                
                tracking_records.append({
                    "Period": period,
                    "Community_ID": comm_id,
                    "Size": len(members),
                    "Event": "birth",
                    "Matched From": None,
                })
        else:
            # Match communities using Jaccard similarity
            matches = match_communities(prev_communities, current_communities, match_threshold)
            
            # Analyze matches to detect events
            matched_prev = set()
            matched_curr = set()
            
            for prev_id, curr_id, jaccard in matches:
                matched_prev.add(prev_id)
                matched_curr.add(curr_id)
                
                prev_size = len(prev_communities[prev_id])
                curr_size = len(current_communities[curr_id])
                
                # Determine event type
                if curr_size > prev_size * 1.5:
                    event_type = "grow"
                elif curr_size < prev_size * 0.67:
                    event_type = "shrink"
                else:
                    event_type = "stable"
                
                tracking_records.append({
                    "Period": period,
                    "Community_ID": curr_id,
                    "Size": curr_size,
                    "Event": event_type,
                    "Matched From": prev_id,
                    "Jaccard": jaccard,
                })
            
            # Deaths: previous communities with no match
            for prev_id in set(prev_communities.keys()) - matched_prev:
                events.append(CommunityEvent(
                    event_type="death",
                    period=period,
                    communities_before=[prev_id],
                    communities_after=[],
                    nodes_involved=prev_communities[prev_id],
                    details={"prev_period": prev_period}
                ))
            
            # Births: current communities with no match
            for curr_id in set(current_communities.keys()) - matched_curr:
                members = current_communities[curr_id]
                events.append(CommunityEvent(
                    event_type="birth",
                    period=period,
                    communities_before=[],
                    communities_after=[curr_id],
                    nodes_involved=members,
                    details={"size": len(members)}
                ))
                
                tracking_records.append({
                    "Period": period,
                    "Community_ID": curr_id,
                    "Size": len(members),
                    "Event": "birth",
                    "Matched From": None,
                })
            
            # Detect merges and splits
            detect_merge_split_events(
                prev_communities, current_communities, 
                matched_prev, matched_curr, events, period
            )
        
        prev_period = period
        prev_communities = current_communities
    
    return events, pd.DataFrame(tracking_records)

def match_communities(
    communities1: Dict[int, Set[str]],
    communities2: Dict[int, Set[str]],
    threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Match communities across two periods using Jaccard similarity.
    
    Returns list of (comm1_id, comm2_id, jaccard_score) tuples.
    """
    matches = []
    
    for id1, members1 in communities1.items():
        best_match = None
        best_jaccard = 0
        
        for id2, members2 in communities2.items():
            intersection = len(members1 & members2)
            union = len(members1 | members2)
            jaccard = intersection / union if union > 0 else 0
            
            if jaccard > best_jaccard and jaccard >= threshold:
                best_jaccard = jaccard
                best_match = id2
        
        if best_match is not None:
            matches.append((id1, best_match, best_jaccard))
    
    return matches

def detect_merge_split_events(
    prev_communities: Dict[int, Set[str]],
    curr_communities: Dict[int, Set[str]],
    matched_prev: Set[int],
    matched_curr: Set[int],
    events: List[CommunityEvent],
    period: str) -> None:
    """Detect merge and split events between periods."""
    # Check for merges: multiple prev communities contributing to one curr
    curr_sources = defaultdict(list)
    for prev_id, prev_members in prev_communities.items():
        for curr_id, curr_members in curr_communities.items():
            overlap = len(prev_members & curr_members)
            if overlap > len(prev_members) * 0.3:  # Significant overlap
                curr_sources[curr_id].append((prev_id, overlap))
    
    for curr_id, sources in curr_sources.items():
        if len(sources) > 1:
            # Multiple sources - potential merge
            total_overlap = sum(s[1] for s in sources)
            if total_overlap > len(curr_communities[curr_id]) * 0.5:
                events.append(CommunityEvent(
                    event_type="merge",
                    period=period,
                    communities_before=[s[0] for s in sources],
                    communities_after=[curr_id],
                    nodes_involved=curr_communities[curr_id],
                    details={"sources": sources}
                ))
    
    # Check for splits: one prev community contributing to multiple curr
    prev_destinations = defaultdict(list)
    for prev_id, prev_members in prev_communities.items():
        for curr_id, curr_members in curr_communities.items():
            overlap = len(prev_members & curr_members)
            if overlap > len(curr_members) * 0.3:  # Significant overlap
                prev_destinations[prev_id].append((curr_id, overlap))
    
    for prev_id, destinations in prev_destinations.items():
        if len(destinations) > 1:
            # Multiple destinations - potential split
            total_overlap = sum(d[1] for d in destinations)
            if total_overlap > len(prev_communities[prev_id]) * 0.5:
                events.append(CommunityEvent(
                    event_type="split",
                    period=period,
                    communities_before=[prev_id],
                    communities_after=[d[0] for d in destinations],
                    nodes_involved=prev_communities[prev_id],
                    details={"destinations": destinations}
                ))

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_temporal_network(
    df: pd.DataFrame,
    network_type: str = "keyword",
    column: str = None,
    year_col: str = "Year",
    window_size: int = 5,
    window_type: str = "fixed",
    sep: str = "; ",
    min_weight: int = 1,
    min_docs_per_window: int = 20,
    compute_communities: bool = True,
    community_method: str = "louvain",
    centrality_measures: List[str] = None,
    top_n_nodes: Optional[int] = None,
    **kwargs) -> TemporalNetworkAnalysis:
    """
    Perform complete temporal network analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    network_type : str
        Type of network: "keyword", "coauthorship", "citation", "country".
    column : str, optional
        Column to use for network construction.
    year_col : str
        Year column name.
    window_size : int
        Time window size in years.
    window_type : str
        "fixed" or "sliding".
    sep : str
        Separator for list-type columns.
    min_weight : int
        Minimum edge weight.
    min_docs_per_window : int
        Minimum documents per window.
    compute_communities : bool
        Whether to detect communities.
    community_method : str
        Community detection method.
    centrality_measures : List[str], optional
        Centrality measures to compute.
    top_n_nodes : int, optional
        If set, keep only top N nodes by total degree.
    **kwargs
        Additional arguments.
    
    Returns
    -------
    TemporalNetworkAnalysis
        Complete temporal network analysis results.
    """
    print(f"Analyzing temporal {network_type} network...")
    
    # Generate snapshots
    print("  Generating network snapshots...")
    snapshots = generate_network_snapshots(
        df=df,
        network_type=network_type,
        column=column,
        year_col=year_col,
        window_size=window_size,
        window_type=window_type,
        sep=sep,
        min_weight=min_weight,
        min_docs_per_window=min_docs_per_window,
        compute_communities=compute_communities,
        community_method=community_method,
        top_n_nodes=top_n_nodes,
        **kwargs)
    
    periods = sorted(snapshots.keys())
    print(f"  Generated {len(periods)} snapshots")
    
    # Compute temporal node metrics
    print("  Computing temporal node metrics...")
    temporal_nodes = compute_temporal_node_metrics(
        snapshots, periods, centrality_measures
    )
    print(f"  Tracked {len(temporal_nodes)} nodes")
    
    # Compute temporal edge metrics
    print("  Computing temporal edge metrics...")
    temporal_edges = compute_temporal_edge_metrics(snapshots, periods)
    print(f"  Tracked {len(temporal_edges)} edges")
    
    # Track community evolution
    community_events = []
    if compute_communities:
        print("  Tracking community evolution...")
        community_events, _ = track_community_evolution(snapshots, periods)
        print(f"  Detected {len(community_events)} community events")
    
    # Compute global metrics
    print("  Computing global network metrics...")
    global_metrics = compute_global_network_metrics(snapshots, periods)
    
    # Create analysis object
    analysis = TemporalNetworkAnalysis(
        network_type=network_type,
        snapshots=snapshots,
        periods=periods,
        temporal_nodes=temporal_nodes,
        temporal_edges=temporal_edges,
        community_events=community_events,
        global_metrics=global_metrics,
        parameters={
            "column": column,
            "window_size": window_size,
            "window_type": window_type,
            "sep": sep,
            "min_weight": min_weight,
            "min_docs_per_window": min_docs_per_window,
            "compute_communities": compute_communities,
            "community_method": community_method,
            "top_n_nodes": top_n_nodes,
        })
    
    print("  Analysis complete!")
    return analysis

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_network_evolution(
    analysis: TemporalNetworkAnalysis,
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot network metrics evolution over time.
    
    Parameters
    ----------
    analysis : TemporalNetworkAnalysis
        Temporal analysis results.
    metrics : List[str], optional
        Metrics to plot. Default: nodes, edges, density, clustering.
    figsize : Tuple[int, int]
        Figure size.
    title : str, optional
        Custom title.
    save_path : str, optional
        Path to save figure.
    dpi : int
        Resolution.
    
    Returns
    -------
    matplotlib Figure.
    """
    if metrics is None:
        metrics = ["N_Nodes", "N_Edges", "Density", "Avg_Clustering"]
    
    df = analysis.global_metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        raise ValueError("No requested metrics available in data.")
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    for _ax in (axes if hasattr(axes, "__iter__") else [axes]): _ax.grid(False)
    if n_metrics == 1:
        axes = [axes]
    
    x = range(len(df))
    
    colors = "lightblue"
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        values = df[metric].values
        
        ax.plot(x, values, marker='o', linewidth=2, color="lightblue", markersize=8)
        
        
        ax.set_ylabel(metric.replace("_", " "), fontsize=11)
        ax
        
        # Add trend line
        if len(values) > 2:
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), '--', color='gray', alpha=0.5, linewidth=1)
    
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(df["Period"].values, rotation=45, ha='right')
    axes[-1].set_xlabel("Time Period", fontsize=12)
    
    if title is None:
        title = f"Temporal {analysis.network_type.title()} Network Evolution"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    # Disable grids
    for _ax in axes.flat:
        _
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_node_trajectories(
    analysis: TemporalNetworkAnalysis,
    nodes: List[str] = None,
    metric: str = "degree",
    top_n: int = 10,
    figsize: Tuple[int, int] = (14, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot centrality trajectories for selected nodes.
    
    Parameters
    ----------
    analysis : TemporalNetworkAnalysis
        Temporal analysis results.
    nodes : List[str], optional
        Specific nodes to plot. If None, select top_n by total activity.
    metric : str
        Centrality metric to plot.
    top_n : int
        Number of top nodes to show if nodes not specified.
    figsize : Tuple[int, int]
        Figure size.
    title : str, optional
        Custom title.
    save_path : str, optional
        Path to save figure.
    dpi : int
        Resolution.
    
    Returns
    -------
    matplotlib Figure.
    """
    # Select nodes
    if nodes is None:
        # Get top nodes by total activity
        node_activity = [(nid, tnode.total_activity) 
                        for nid, tnode in analysis.temporal_nodes.items()]
        node_activity.sort(key=lambda x: -x[1])
        nodes = [n[0] for n in node_activity[:top_n]]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    periods = analysis.periods
    x = range(len(periods))
    
    colors = "lightblue"
    
    for i, node_id in enumerate(nodes):
        if node_id not in analysis.temporal_nodes:
            continue
        
        tnode = analysis.temporal_nodes[node_id]
        
        if metric == "degree":
            trajectory = tnode.degree_trajectory
        elif metric in tnode.centrality_trajectories:
            trajectory = tnode.centrality_trajectories[metric]
        else:
            continue
        
        # Build y values (NaN for missing periods)
        y_values = [trajectory.get(p, np.nan) for p in periods]
        
        # Truncate long labels
        label = node_id[:30] + "..." if len(str(node_id)) > 30 else node_id
        
        ax.plot(x, y_values, marker='o', linewidth=2, color="lightblue", 
               label=label, alpha=0.8, markersize=6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    
    if title is None:
        title = f"Node {metric.title()} Trajectories"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax
    
    # Disable grids
    for _ax in axes.flat:
        _
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_edge_persistence(
    analysis: TemporalNetworkAnalysis,
    figsize: Tuple[int, int] = (12, 8),
    bins: int = 20,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """
    Plot distribution of edge persistence (how long edges last).
    
    Parameters
    ----------
    analysis : TemporalNetworkAnalysis
        Temporal analysis results.
    figsize : Tuple[int, int]
        Figure size.
    bins : int
        Number of histogram bins.
    title : str, optional
        Custom title.
    save_path : str, optional
        Path to save figure.
    dpi : int
        Resolution.
    
    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    # Calculate persistence ratios
    persistence_ratios = [
        tedge.get_persistence(analysis.periods)
        for tedge in analysis.temporal_edges.values()
    ]
    
    # Calculate absolute durations
    durations = [len(tedge.active_periods) for tedge in analysis.temporal_edges.values()]
    
    # Plot persistence ratio distribution
    ax.hist(persistence_ratios, bins=bins, color="lightblue", edgecolor='white', alpha=0.7)
    ax.set_xlabel("Persistence Ratio", fontsize=11)
    ax.set_ylabel("Number of Edges", fontsize=11)
    ax.set_title("Edge Persistence Distribution", fontsize=12)
    
    # Plot duration distribution
    ax.hist(durations, bins=min(bins, len(analysis.periods)), 
            color="lightblue", edgecolor='white', alpha=0.7)
    ax.set_xlabel("Active Periods", fontsize=11)
    ax.set_ylabel("Number of Edges", fontsize=11)
    ax.set_title("Edge Duration Distribution", fontsize=12)
    
    if title is None:
        title = "Edge Temporal Patterns"
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_community_evolution(
    analysis: TemporalNetworkAnalysis,
    figsize: Tuple[int, int] = (16, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot community sizes and evolution over time (alluvial-style).
    
    Parameters
    ----------
    analysis : TemporalNetworkAnalysis
        Temporal analysis results.
    figsize : Tuple[int, int]
        Figure size.
    title : str, optional
        Custom title.
    save_path : str, optional
        Path to save figure.
    dpi : int
        Resolution.
    
    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    periods = analysis.periods
    
    # Top plot: Community sizes over time
    
    # Get community sizes per period
    period_comm_sizes = {}
    for period in periods:
        if period in analysis.snapshots:
            snapshot = analysis.snapshots[period]
            if snapshot.communities:
                sizes = snapshot.get_community_sizes()
                period_comm_sizes[period] = sorted(sizes.values(), reverse=True)
            else:
                period_comm_sizes[period] = []
    
    # Stacked bar chart of community sizes
    x = np.arange(len(periods))
    max_communities = max(len(sizes) for sizes in period_comm_sizes.values()) if period_comm_sizes else 0
    
    colors = "lightblue"
    
    bottom = np.zeros(len(periods))
    for comm_idx in range(max_communities):
        heights = []
        for period in periods:
            sizes = period_comm_sizes.get(period, [])
            if comm_idx < len(sizes):
                heights.append(sizes[comm_idx])
            else:
                heights.append(0)
        
        ax.bar(x, heights, bottom=bottom, color="lightblue", 
               edgecolor='white', width=0.8, label=f'Community {comm_idx+1}')
        bottom += heights
    
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.set_ylabel("Community Size (nodes)", fontsize=11)
    ax.set_title("Community Sizes Over Time", fontsize=12)
    ax
    
    if max_communities <= 10:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    
    # Bottom plot: Community events timeline
    
    event_types = ["birth", "death", "merge", "split", "grow", "shrink"]
    event_colors = {
        "birth": "green",
        "death": "red",
        "merge": "purple",
        "split": "orange",
        "grow": "lightgreen",
        "shrink": "lightcoral",
    }
    
    # Count events per period
    event_counts = {period: {et: 0 for et in event_types} for period in periods}
    for event in analysis.community_events:
        if event.period in event_counts and event.event_type in event_counts[event.period]:
            event_counts[event.period][event.event_type] += 1
    
    # Grouped bar chart
    bar_width = 0.12
    for i, event_type in enumerate(event_types):
        counts = [event_counts[p][event_type] for p in periods]
        offset = (i - len(event_types)/2 + 0.5) * bar_width
        ax.bar(x + offset, counts, bar_width, 
               color=event_colors[event_type], label=event_type.title())
    
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.set_ylabel("Event Count", fontsize=11)
    ax.set_title("Community Events", fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax
    
    if title is None:
        title = "Community Evolution Analysis"
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_network_snapshots(
    analysis: TemporalNetworkAnalysis,
    periods: List[str] = None,
    layout: str = "spring",
    node_size_by: str = "degree",
    color_by: str = "community",
    figsize_per_plot: Tuple[int, int] = (8, 8),
    max_cols: int = 4,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """
    Plot network snapshots in a grid.
    
    Parameters
    ----------
    analysis : TemporalNetworkAnalysis
        Temporal analysis results.
    periods : List[str], optional
        Specific periods to plot. If None, plot all.
    layout : str
        Network layout: "spring", "kamada_kawai", "circular".
    node_size_by : str
        Attribute for node sizing: "degree", "betweenness", "uniform".
    color_by : str
        Attribute for node coloring: "community", "degree", "uniform".
    figsize_per_plot : Tuple[int, int]
        Size per subplot.
    max_cols : int
        Maximum columns in grid.
    title : str, optional
        Custom title.
    save_path : str, optional
        Path to save figure.
    dpi : int
        Resolution.
    
    Returns
    -------
    matplotlib Figure.
    """
    if periods is None:
        periods = analysis.periods
    
    n_plots = len(periods)
    n_cols = min(n_plots, max_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(figsize_per_plot[0]*n_cols, figsize_per_plot[1]*n_rows))
    
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    layout_funcs = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
        "random": nx.random_layout,
    }
    layout_func = layout_funcs.get(layout, nx.spring_layout)
    
    for idx, period in enumerate(periods):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if period not in analysis.snapshots:
            ax.set_visible(False)
            continue
        
        snapshot = analysis.snapshots[period]
        G = snapshot.graph
        
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
            ax.set_title(period, fontsize=11)
            ax.axis('off')
            continue
        
        # Compute layout
        try:
            pos = layout_func(G)
        except:
            pos = nx.spring_layout(G)
        
        # Node sizes
        if node_size_by == "degree":
            degrees = dict(G.degree())
            max_deg = max(degrees.values()) if degrees else 1
            node_sizes = [300 + 700 * degrees.get(n, 0) / max_deg for n in G.nodes()]
        elif node_size_by == "betweenness":
            try:
                bc = nx.betweenness_centrality(G)
                max_bc = max(bc.values()) if bc else 1
                node_sizes = [300 + 700 * bc.get(n, 0) / max(max_bc, 1e-10) for n in G.nodes()]
            except:
                node_sizes = [400] * G.number_of_nodes()
        else:
            node_sizes = [400] * G.number_of_nodes()
        
        # Node colors
        if color_by == "community" and snapshot.communities:
            communities = [snapshot.communities.get(n, 0) for n in G.nodes()]
            n_comm = len(set(communities))
            # Using single color
            node_colors = [cmap(c % 20) for c in communities]
        elif color_by == "degree":
            degrees = [G.degree(n) for n in G.nodes()]
            norm = Normalize(vmin=min(degrees), vmax=max(degrees))
            node_colors = "lightblue"
        else:
            node_colors = 'lightblue'
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                              node_color=node_colors, alpha=0.7)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.5)
        
        # Add labels for high-degree nodes only
        degrees = dict(G.degree())
        if degrees:
            threshold = np.percentile(list(degrees.values()), 90)
            labels = {n: n[:15] for n, d in degrees.items() if d >= threshold}
            if labels:
                nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        ax.set_title(f"{period}\n({snapshot.n_nodes} nodes, {snapshot.n_edges} edges)", 
                    fontsize=11)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    if title is None:
        title = f"Temporal {analysis.network_type.title()} Network Snapshots"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    # Disable grids
    for _ax in axes.flat:
        _
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_node_lifecycle(
    analysis: TemporalNetworkAnalysis,
    nodes: List[str] = None,
    top_n: int = 30,
    figsize: Tuple[int, int] = (16, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot node lifecycle (appearance/disappearance timeline).
    
    Parameters
    ----------
    analysis : TemporalNetworkAnalysis
        Temporal analysis results.
    nodes : List[str], optional
        Specific nodes to show. If None, select top_n by activity.
    top_n : int
        Number of nodes if not specified.
    figsize : Tuple[int, int]
        Figure size.
    title : str, optional
        Custom title.
    save_path : str, optional
        Path to save figure.
    dpi : int
        Resolution.
    
    Returns
    -------
    matplotlib Figure.
    """
    if nodes is None:
        # Get top nodes by total activity
        node_activity = [(nid, tnode.total_activity) 
                        for nid, tnode in analysis.temporal_nodes.items()]
        node_activity.sort(key=lambda x: -x[1])
        nodes = [n[0] for n in node_activity[:top_n]]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    periods = analysis.periods
    period_to_x = {p: i for i, p in enumerate(periods)}
    
    y_pos = 0
    y_labels = []
    
    for node_id in nodes:
        if node_id not in analysis.temporal_nodes:
            continue
        
        tnode = analysis.temporal_nodes[node_id]
        
        # Draw activity line
        for period in tnode.active_periods:
            x = period_to_x.get(period)
            if x is not None:
                # Size by degree
                degree = tnode.degree_trajectory.get(period, 1)
                size = 50 + 10 * degree
                ax.scatter(x, y_pos, s=size, c='steelblue', alpha=0.7, zorder=3)
        
        # Connect active periods
        active_x = [period_to_x[p] for p in tnode.active_periods if p in period_to_x]
        if len(active_x) > 1:
            ax.plot([min(active_x), max(active_x)], [y_pos, y_pos], 
                   c='lightblue', linewidth=2, alpha=0.5, zorder=1)
        
        # Truncate label
        label = str(node_id)[:25] + "..." if len(str(node_id)) > 25 else str(node_id)
        y_labels.append(label)
        y_pos += 1
    
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha='right')
    
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Node", fontsize=12)
    
    if title is None:
        title = "Node Lifecycle Timeline"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax
    ax.set_xlim(-0.5, len(periods) - 0.5)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_network_comparison(
    analysis: TemporalNetworkAnalysis,
    period1: str,
    period2: str,
    figsize: Tuple[int, int] = (16, 8),
    layout: str = "spring",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Compare two network snapshots side by side, highlighting changes.
    
    Parameters
    ----------
    analysis : TemporalNetworkAnalysis
        Temporal analysis results.
    period1, period2 : str
        Periods to compare.
    figsize : Tuple[int, int]
        Figure size.
    layout : str
        Network layout.
    title : str, optional
        Custom title.
    save_path : str, optional
        Path to save figure.
    dpi : int
        Resolution.
    
    Returns
    -------
    matplotlib Figure.
    """
    if period1 not in analysis.snapshots or period2 not in analysis.snapshots:
        raise ValueError("One or both periods not found in analysis.")
    
    G1 = analysis.snapshots[period1].graph
    G2 = analysis.snapshots[period2].graph
    
    nodes1 = set(G1.nodes())
    nodes2 = set(G2.nodes())
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    
    # Categorize nodes
    new_nodes = nodes2 - nodes1
    lost_nodes = nodes1 - nodes2
    persistent_nodes = nodes1 & nodes2
    
    # Categorize edges
    new_edges = edges2 - edges1
    lost_edges = edges1 - edges2
    persistent_edges = edges1 & edges2
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    # Compute shared layout using union graph
    G_union = nx.compose(G1, G2)
    layout_funcs = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
    }
    pos = layout_funcs.get(layout, nx.spring_layout)(G_union)
    
    # Plot period 1
    colors1 = ['lightblue' if n in persistent_nodes else 'lightcoral' for n in G1.nodes()]
    nx.draw_networkx_nodes(G1, pos, ax=ax, node_color=colors1, node_size=200, alpha=0.7)
    nx.draw_networkx_edges(G1, pos, ax=ax, alpha=0.3, width=0.5)
    ax.set_title(f"{period1}\n({len(G1.nodes())} nodes, {len(G1.edges())} edges)", fontsize=11)
    ax.axis('off')
    
    # Plot period 2
    colors = ['lightblue' if n in persistent_nodes else 'lightgreen' for n in G2.nodes()]
    nx.draw_networkx_nodes(G2, pos, ax=ax, node_color=colors, node_size=200, alpha=0.7)
    nx.draw_networkx_edges(G2, pos, ax=ax, alpha=0.3, width=0.5)
    ax.set_title(f"{period2}\n({len(G2.nodes())} nodes, {len(G2.edges())} edges)", fontsize=11)
    ax.axis('off')
    
    # Plot difference summary
    categories = ['Persistent\nNodes', 'New\nNodes', 'Lost\nNodes', 
                 'Persistent\nEdges', 'New\nEdges', 'Lost\nEdges']
    values = [len(persistent_nodes), len(new_nodes), len(lost_nodes),
             len(persistent_edges), len(new_edges), len(lost_edges)]
    colors = ['steelblue', 'green', 'red', 'steelblue', 'green', 'red']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='white')
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Network Changes", fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=10)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', label='Persistent'),
        mpatches.Patch(facecolor='lightgreen', label='New'),
        mpatches.Patch(facecolor='lightcoral', label='Lost'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    if title is None:
        title = f"Network Comparison: {period1} vs {period2}"
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def create_animated_network(
    analysis: TemporalNetworkAnalysis,
    output_path: str = "network_evolution.gif",
    figsize: Tuple[int, int] = (10, 10),
    layout: str = "spring",
    interval: int = 1000,
    fps: int = 1,
    dpi: int = 100) -> str:
    """
    Create an animated GIF of network evolution.
    
    Parameters
    ----------
    analysis : TemporalNetworkAnalysis
        Temporal analysis results.
    output_path : str
        Output file path.
    figsize : Tuple[int, int]
        Figure size.
    layout : str
        Network layout.
    interval : int
        Milliseconds between frames.
    fps : int
        Frames per second for GIF.
    dpi : int
        Resolution.
    
    Returns
    -------
    str
        Path to saved animation.
    """
    # Compute consistent layout using union graph
    G_union = nx.Graph()
    for snapshot in analysis.snapshots.values():
        G_union = nx.compose(G_union, snapshot.graph)
    
    layout_funcs = {
        "spring": lambda G: nx.spring_layout(G, seed=42),
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
    }
    pos = layout_funcs.get(layout, lambda G: nx.spring_layout(G, seed=42))(G_union)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    def update(frame_idx):
        ax.clear()
        
        period = analysis.periods[frame_idx]
        snapshot = analysis.snapshots.get(period)
        
        if snapshot is None or snapshot.graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, f"{period}\nNo data", ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
        
        G = snapshot.graph
        
        # Node colors by community
        if snapshot.communities:
            communities = [snapshot.communities.get(n, 0) for n in G.nodes()]
            # Using single color
            node_colors = [cmap(c % 12) for c in communities]
        else:
            node_colors = 'lightblue'
        
        # Node sizes by degree
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        node_sizes = [100 + 400 * degrees.get(n, 0) / max_deg for n in G.nodes()]
        
        # Get positions for current nodes
        current_pos = {n: pos[n] for n in G.nodes() if n in pos}
        
        # Draw
        nx.draw_networkx_nodes(G, current_pos, ax=ax, node_size=node_sizes,
                              node_color=node_colors, alpha=0.7)
        nx.draw_networkx_edges(G, current_pos, ax=ax, alpha=0.3, width=0.5)
        
        ax.set_title(f"{period}\n({snapshot.n_nodes} nodes, {snapshot.n_edges} edges)", 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(analysis.periods), interval=interval
    )
    
    # Save as GIF
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)
    
    print(f"Animation saved to {output_path}")
    return output_path

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_network_summary(analysis: TemporalNetworkAnalysis) -> pd.DataFrame:
    """Get a summary of network evolution statistics."""
    records = []
    
    for period in analysis.periods:
        snapshot = analysis.snapshots.get(period)
        if snapshot is None:
            continue
        
        records.append({
            "Period": period,
            "Nodes": snapshot.n_nodes,
            "Edges": snapshot.n_edges,
            "Density": round(snapshot.density, 4),
            "Communities": len(set(snapshot.communities.values())) if snapshot.communities else None,
        })
    
    return pd.DataFrame(records)

def get_node_summary(analysis: TemporalNetworkAnalysis, top_n: int = 50) -> pd.DataFrame:
    """Get summary of top nodes by activity."""
    records = []
    
    for node_id, tnode in analysis.temporal_nodes.items():
        records.append({
            "Node": node_id,
            "First Appearance": tnode.first_appearance,
            "Last Appearance": tnode.last_appearance,
            "Active Periods": len(tnode.active_periods),
            "Total Activity": tnode.total_activity,
            "Activity Ratio": round(tnode.get_activity_ratio(analysis.periods), 3),
            "Avg Degree": round(np.mean(list(tnode.degree_trajectory.values())), 2) if tnode.degree_trajectory else 0,
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values("Total Activity", ascending=False).head(top_n)
    return df.reset_index(drop=True)

def get_edge_summary(analysis: TemporalNetworkAnalysis, top_n: int = 50) -> pd.DataFrame:
    """Get summary of top edges by persistence and weight."""
    records = []
    
    for (u, v), tedge in analysis.temporal_edges.items():
        records.append({
            "Source": u,
            "Target": v,
            "First Appearance": tedge.first_appearance,
            "Last Appearance": tedge.last_appearance,
            "Active Periods": len(tedge.active_periods),
            "Persistence": round(tedge.get_persistence(analysis.periods), 3),
            "Total Weight": tedge.total_weight,
            "Avg Weight": round(np.mean(list(tedge.weight_trajectory.values())), 2) if tedge.weight_trajectory else 0,
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values("Total Weight", ascending=False).head(top_n)
    return df.reset_index(drop=True)

def export_temporal_network(
    analysis: TemporalNetworkAnalysis,
    output_dir: str,
    formats: List[str] = None) -> Dict[str, str]:
    """
    Export temporal network data to various formats.
    
    Parameters
    ----------
    analysis : TemporalNetworkAnalysis
        Temporal analysis results.
    output_dir : str
        Output directory.
    formats : List[str], optional
        Formats to export: "excel", "graphml", "gexf", "json".
    
    Returns
    -------
    Dict mapping format to file path.
    """
    if formats is None:
        formats = ["excel", "graphml"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    if "excel" in formats:
        # Export summaries to Excel
        excel_path = os.path.join(output_dir, "temporal_network_analysis.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            analysis.global_metrics.to_excel(writer, sheet_name="Network_Evolution", index=False)
            get_node_summary(analysis, top_n=500).to_excel(writer, sheet_name="Node_Summary", index=False)
            get_edge_summary(analysis, top_n=500).to_excel(writer, sheet_name="Edge_Summary", index=False)
        paths["excel"] = excel_path
        print(f"Excel export: {excel_path}")
    
    if "graphml" in formats:
        # Export each snapshot as GraphML
        graphml_dir = output_dir
        os.makedirs(graphml_dir, exist_ok=True)
        for period, snapshot in analysis.snapshots.items():
            safe_period = period.replace("-", "_")
            path = os.path.join(graphml_dir, f"network_{safe_period}.graphml")
            nx.write_graphml(snapshot.graph, path)
        paths["graphml"] = graphml_dir
        print(f"GraphML exports: {graphml_dir}")
    
    if "gexf" in formats:
        # Export each snapshot as GEXF (supports dynamic networks in Gephi)
        gexf_dir = output_dir
        os.makedirs(gexf_dir, exist_ok=True)
        for period, snapshot in analysis.snapshots.items():
            safe_period = period.replace("-", "_")
            path = os.path.join(gexf_dir, f"network_{safe_period}.gexf")
            nx.write_gexf(snapshot.graph, path)
        paths["gexf"] = gexf_dir
        print(f"GEXF exports: {gexf_dir}")
    
    if "json" in formats:
        # Export analysis metadata as JSON
        json_path = os.path.join(output_dir, "temporal_network_metadata.json")
        metadata = {
            "network_type": analysis.network_type,
            "periods": analysis.periods,
            "parameters": analysis.parameters,
            "n_nodes_total": len(analysis.temporal_nodes),
            "n_edges_total": len(analysis.temporal_edges),
            "n_community_events": len(analysis.community_events),
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        paths["json"] = json_path
        print(f"JSON metadata: {json_path}")
    
    return paths

# =============================================================================
# INTEGRATION WITH BIBLIOANALYSIS CLASS
# =============================================================================

def add_temporal_network_methods(cls):
    """
    Add temporal network methods to BiblioAnalysis class.
    
    Usage:
        from temporal_networks import add_temporal_network_methods
        add_temporal_network_methods(BiblioAnalysis)
    """
    
    def analyze_temporal_network_method(
        self,
        network_type: str = "keyword",
        column: str = None,
        window_size: int = 5,
        save_results: bool = True,
        **kwargs
    ) -> TemporalNetworkAnalysis:
        """
        Analyze temporal network evolution.
        
        Parameters
        ----------
        network_type : str
            Type: "keyword", "coauthorship", "citation", "country".
        column : str, optional
            Column for network construction.
        window_size : int
            Time window size in years.
        save_results : bool
            Whether to save results.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        TemporalNetworkAnalysis
        """
        sep = getattr(self, 'default_separator', '; ')
        
        self.temporal_network = analyze_temporal_network(
            self.df,
            network_type=network_type,
            column=column,
            window_size=window_size,
            sep=sep,
            **kwargs
        )
        
        self.temporal_network_summary = self.temporal_network.to_summary_df()
        
        if save_results and hasattr(self, 'res_folder') and self.res_folder:
            output_dir = self.res_folder
            export_temporal_network(self.temporal_network, output_dir)
        
        return self.temporal_network
    
    def plot_temporal_network_method(
        self,
        plot_type: str = "evolution",
        save: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Create temporal network visualizations.
        
        Parameters
        ----------
        plot_type : str
            Type: "evolution", "trajectories", "persistence", "communities",
            "snapshots", "lifecycle", "comparison".
        save : bool
            Whether to save plot.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        matplotlib Figure.
        """
        if not hasattr(self, 'temporal_network'):
            raise ValueError("Run analyze_temporal_network first.")
        
        save_path = None
        if save and hasattr(self, 'res_folder') and self.res_folder:
            save_path = os.path.join(self.res_folder, f"temporal_{plot_type}")
        
        plot_functions = {
            "evolution": plot_network_evolution,
            "trajectories": plot_node_trajectories,
            "persistence": plot_edge_persistence,
            "communities": plot_community_evolution,
            "snapshots": plot_network_snapshots,
            "lifecycle": plot_node_lifecycle,
        }
        
        if plot_type not in plot_functions:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        return plot_functions[plot_type](self.temporal_network, save_path=save_path, **kwargs)
    
    cls.analyze_temporal_network = analyze_temporal_network_method
    cls.plot_temporal_network = plot_temporal_network_method
    
    return cls

# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import biblium as bb
    ba = bb.BiblioAnalysis(f_name="data\\open alex dataset.csv", db="oa")
    
    print("Temporal Networks Analysis")
    print("=" * 60)
    
    # Run analysis
    result = analyze_temporal_network(
        df=ba.df,
        authors_col="Authors",
        year_col="Year",
        citations_col="Cited by",
        window_size=3,
        verbose=True)
    
    # Print summary
    print(f"\nAnalyzed {len(result.snapshots)} time periods")
    
    # Visualizations
    print("\nGenerating plots...")
    plot_network_evolution(result, save_path="results/network_evolution")
    plot_node_trajectories(result, save_path="results/node_trajectories")
    
    # Export
    print("\nExporting results...")
    export_temporal_network(result, "results")
    
    print("\nDone!")
