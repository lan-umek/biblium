# -*- coding: utf-8 -*-
"""
SDG Network Analysis Module

This module provides network-based analysis of SDG relationships,
including co-occurrence networks, collaboration patterns, and
identification of bridge papers connecting multiple SDGs.

Features:
1. SDG Co-occurrence Networks - Which SDGs appear together
2. SDG Collaboration Networks - Country/institution collaboration by SDG
3. SDG Bridge Papers - Papers connecting multiple SDGs
4. SDG Clusters - Community detection in SDG networks
5. SDG Centrality Analysis - Which SDGs are most central/connected
6. Temporal SDG Networks - How SDG relationships evolve
7. SDG Knowledge Flow - Citation-based SDG relationships

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import seaborn as sns

# Network libraries
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("networkx not available. Install with: pip install networkx")

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

# =============================================================================
# SDG DEFINITIONS (shared with sdg_drift)
# =============================================================================

SDG_NAMES = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health and Well-being",
    4: "Quality Education", 5: "Gender Equality", 6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy", 8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure", 10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities", 12: "Responsible Consumption and Production",
    13: "Climate Action", 14: "Life Below Water", 15: "Life on Land",
    16: "Peace, Justice and Strong Institutions", 17: "Partnerships for the Goals",
}

SDG_SHORT_NAMES = {
    1: "Poverty", 2: "Hunger", 3: "Health", 4: "Education", 5: "Gender",
    6: "Water", 7: "Energy", 8: "Work", 9: "Industry", 10: "Inequality",
    11: "Cities", 12: "Consumption", 13: "Climate", 14: "Oceans",
    15: "Land", 16: "Peace", 17: "Partnerships"
}

SDG_COLORS = {
    1: "#E5243B", 2: "#DDA63A", 3: "#4C9F38", 4: "#C5192D", 5: "#FF3A21",
    6: "#26BDE2", 7: "#FCC30B", 8: "#A21942", 9: "#FD6925", 10: "#DD1367",
    11: "#FD9D24", 12: "#BF8B2E", 13: "#3F7E44", 14: "#0A97D9", 15: "#56C02B",
    16: "#00689D", 17: "#19486A"
}

SDG_PERSPECTIVES = {
    "Life": [3],
    "Social": [1, 2, 4, 5, 10],
    "Economic": [7, 8, 9, 11, 12],
    "Planet": [6, 13, 14, 15],
    "Peace": [16],
    "Partnership": [17],
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SDGNetworkMetrics:
    """Metrics for SDG network analysis."""
    sdg: int
    degree: int
    weighted_degree: float
    betweenness: float
    closeness: float
    eigenvector: float
    clustering: float
    community: int
    is_bridge: bool  # High betweenness relative to degree
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "SDG": self.sdg,
            "Name": SDG_SHORT_NAMES.get(self.sdg, ""),
            "Degree": self.degree,
            "Weighted Degree": self.weighted_degree,
            "Betweenness": self.betweenness,
            "Closeness": self.closeness,
            "Eigenvector": self.eigenvector,
            "Clustering": self.clustering,
            "Community": self.community,
            "Is Bridge": self.is_bridge,
        }


@dataclass
class BridgePaper:
    """A paper that bridges multiple SDGs."""
    doc_id: Any
    title: str
    year: int
    sdgs: List[int]
    n_sdgs: int
    perspectives_covered: List[str]
    n_perspectives: int
    bridge_score: float  # Based on diversity of SDGs covered
    citations: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Doc_ID": self.doc_id,
            "Title": self.title,
            "Year": self.year,
            "SDGs": self.sdgs,
            "N_SDGs": self.n_sdgs,
            "Perspectives": self.perspectives_covered,
            "N_Perspectives": self.n_perspectives,
            "Bridge Score": self.bridge_score,
            "Citations": self.citations,
        }


@dataclass
class SDGCluster:
    """A cluster of related SDGs."""
    cluster_id: int
    sdgs: List[int]
    name: str  # Auto-generated or user-defined
    dominant_perspective: str
    internal_density: float
    external_connections: Dict[int, float]  # Other cluster -> connection strength
    key_terms: List[str]


@dataclass
class SDGNetworkAnalysis:
    """Complete SDG network analysis results."""
    # Network objects
    cooccurrence_network: Any  # nx.Graph
    collaboration_network: Any  # nx.Graph (countries)
    
    # Metrics
    sdg_metrics: Dict[int, SDGNetworkMetrics]
    
    # Bridge papers
    bridge_papers: List[BridgePaper]
    top_bridge_papers: List[BridgePaper]
    
    # Clusters
    sdg_clusters: List[SDGCluster]
    
    # Matrices
    cooccurrence_matrix: pd.DataFrame
    normalized_cooccurrence: pd.DataFrame  # Jaccard or cosine normalized
    
    # Summary statistics
    network_density: float
    average_clustering: float
    n_communities: int
    modularity: float
    
    # Temporal (if available)
    temporal_networks: Dict[str, Any]  # Period -> network
    
    def get_metrics_df(self) -> pd.DataFrame:
        """Get SDG metrics as DataFrame."""
        records = [m.to_dict() for m in self.sdg_metrics.values()]
        return pd.DataFrame(records).sort_values("Weighted Degree", ascending=False)
    
    def get_bridge_papers_df(self) -> pd.DataFrame:
        """Get bridge papers as DataFrame."""
        records = [p.to_dict() for p in self.bridge_papers]
        df = pd.DataFrame(records)
        return df.sort_values("Bridge Score", ascending=False) if not df.empty and "Bridge Score" in df.columns else df
    
    def get_top_connections(self, n: int = 20) -> pd.DataFrame:
        """Get top SDG connections."""
        connections = []
        matrix = self.cooccurrence_matrix
        for i in matrix.index:
            for j in matrix.columns:
                if i < j:
                    connections.append({
                        "SDG_1": i,
                        "SDG_2": j,
                        "Name_1": SDG_SHORT_NAMES.get(i, ""),
                        "Name_2": SDG_SHORT_NAMES.get(j, ""),
                        "Co-occurrences": matrix.loc[i, j],
                        "Normalized": self.normalized_cooccurrence.loc[i, j],
                    })
        df = pd.DataFrame(connections)
        return df.sort_values("Co-occurrences", ascending=False).head(n)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sdg_columns(df: pd.DataFrame) -> List[str]:
    """Find SDG indicator columns in DataFrame."""
    import re
    sdg_cols = []
    for col in df.columns:
        if col.startswith("SDG") and any(c.isdigit() for c in col):
            sdg_cols.append(col)
    return sorted(sdg_cols)


def extract_sdg_number(col_name: str) -> int:
    """Extract SDG number from column name."""
    import re
    match = re.search(r'(\d+)', col_name)
    return int(match.group(1)) if match else 0


def get_paper_sdgs(row: pd.Series, sdg_cols: List[str]) -> List[int]:
    """Get list of SDGs for a paper."""
    sdgs = []
    for col in sdg_cols:
        if row.get(col, 0) == 1:
            sdgs.append(extract_sdg_number(col))
    return sdgs


def get_perspectives_for_sdgs(sdgs: List[int]) -> List[str]:
    """Get perspectives covered by a list of SDGs."""
    perspectives = set()
    for sdg in sdgs:
        for persp, persp_sdgs in SDG_PERSPECTIVES.items():
            if sdg in persp_sdgs:
                perspectives.add(persp)
    return list(perspectives)


def compute_bridge_score(sdgs: List[int]) -> float:
    """
    Compute bridge score based on SDG diversity.
    Higher score = more diverse SDG coverage across perspectives.
    """
    if len(sdgs) <= 1:
        return 0.0
    
    perspectives = get_perspectives_for_sdgs(sdgs)
    n_perspectives = len(perspectives)
    n_sdgs = len(sdgs)
    
    # Score based on: number of SDGs, number of perspectives, and spread
    # Max theoretical: 17 SDGs, 6 perspectives
    sdg_component = min(n_sdgs / 5, 1.0)  # Cap at 5 SDGs
    perspective_component = n_perspectives / 6
    
    # Bonus for covering distant perspectives
    distant_pairs = 0
    perspective_distances = {
        ("Social", "Planet"): 1, ("Social", "Economic"): 0.5,
        ("Economic", "Planet"): 0.8, ("Life", "Peace"): 1,
        ("Partnership", "Planet"): 0.7,
    }
    for p1, p2 in combinations(perspectives, 2):
        key = tuple(sorted([p1, p2]))
        distant_pairs += perspective_distances.get(key, 0.3)
    
    distance_component = min(distant_pairs / 3, 1.0)
    
    return (sdg_component * 0.3 + perspective_component * 0.4 + distance_component * 0.3)


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def build_sdg_cooccurrence_network(
    df: pd.DataFrame,
    sdg_cols: List[str] = None,
    min_cooccurrence: int = 1,
    normalize: str = "jaccard"  # "jaccard", "cosine", "none"
) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Build SDG co-occurrence network.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with SDG indicator columns.
    sdg_cols : List[str]
        SDG columns. Auto-detected if None.
    min_cooccurrence : int
        Minimum co-occurrences to create edge.
    normalize : str
        Normalization method for edge weights.
    
    Returns
    -------
    Tuple of (nx.Graph, raw_matrix, normalized_matrix)
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx required for network analysis")
    
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    sdg_numbers = [extract_sdg_number(col) for col in sdg_cols]
    n_sdgs = len(sdg_numbers)
    
    # Compute co-occurrence matrix
    raw_matrix = np.zeros((n_sdgs, n_sdgs))
    
    for idx, row in df.iterrows():
        paper_sdgs = []
        for i, col in enumerate(sdg_cols):
            if row.get(col, 0) == 1:
                paper_sdgs.append(i)
        
        # Count co-occurrences
        for i, j in combinations(paper_sdgs, 2):
            raw_matrix[i, j] += 1
            raw_matrix[j, i] += 1
        
        # Diagonal = total count per SDG
        for i in paper_sdgs:
            raw_matrix[i, i] += 1
    
    raw_df = pd.DataFrame(raw_matrix, index=sdg_numbers, columns=sdg_numbers)
    
    # Normalize
    if normalize == "jaccard":
        norm_matrix = np.zeros_like(raw_matrix)
        for i in range(n_sdgs):
            for j in range(n_sdgs):
                if i != j:
                    union = raw_matrix[i, i] + raw_matrix[j, j] - raw_matrix[i, j]
                    norm_matrix[i, j] = raw_matrix[i, j] / union if union > 0 else 0
                else:
                    norm_matrix[i, j] = 1.0
    elif normalize == "cosine":
        norm_matrix = np.zeros_like(raw_matrix)
        for i in range(n_sdgs):
            for j in range(n_sdgs):
                denom = np.sqrt(raw_matrix[i, i] * raw_matrix[j, j])
                norm_matrix[i, j] = raw_matrix[i, j] / denom if denom > 0 else 0
    else:
        norm_matrix = raw_matrix.copy()
    
    norm_df = pd.DataFrame(norm_matrix, index=sdg_numbers, columns=sdg_numbers)
    
    # Build network
    G = nx.Graph()
    
    # Add nodes
    for sdg in sdg_numbers:
        G.add_node(sdg, 
                   name=SDG_SHORT_NAMES.get(sdg, f"SDG {sdg}"),
                   full_name=SDG_NAMES.get(sdg, f"SDG {sdg}"),
                   color=SDG_COLORS.get(sdg, "#888888"),
                   count=int(raw_df.loc[sdg, sdg]))
    
    # Add edges
    for i, sdg_i in enumerate(sdg_numbers):
        for j, sdg_j in enumerate(sdg_numbers):
            if i < j and raw_matrix[i, j] >= min_cooccurrence:
                G.add_edge(sdg_i, sdg_j,
                          weight=norm_matrix[i, j],
                          raw_count=int(raw_matrix[i, j]))
    
    return G, raw_df, norm_df


def compute_sdg_network_metrics(G: Any) -> Dict[int, SDGNetworkMetrics]:
    """Compute centrality and other metrics for each SDG."""
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx required")
    
    # Compute centralities
    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight='weight'))
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector = {n: 0 for n in G.nodes()}
    
    clustering = nx.clustering(G, weight='weight')
    
    # Community detection
    if LOUVAIN_AVAILABLE and len(G.nodes()) > 1:
        try:
            communities = community_louvain.best_partition(G)
        except:
            communities = {n: 0 for n in G.nodes()}
    else:
        communities = {n: 0 for n in G.nodes()}
    
    # Identify bridges (high betweenness relative to degree)
    max_betweenness = max(betweenness.values()) if betweenness else 1
    max_degree = max(degree.values()) if degree else 1
    
    metrics = {}
    for sdg in G.nodes():
        norm_betweenness = betweenness[sdg] / max_betweenness if max_betweenness > 0 else 0
        norm_degree = degree[sdg] / max_degree if max_degree > 0 else 0
        is_bridge = norm_betweenness > 0.5 and norm_degree < 0.7
        
        metrics[sdg] = SDGNetworkMetrics(
            sdg=sdg,
            degree=degree[sdg],
            weighted_degree=weighted_degree[sdg],
            betweenness=betweenness[sdg],
            closeness=closeness[sdg],
            eigenvector=eigenvector.get(sdg, 0),
            clustering=clustering[sdg],
            community=communities.get(sdg, 0),
            is_bridge=is_bridge
        )
    
    return metrics


def identify_bridge_papers(
    df: pd.DataFrame,
    sdg_cols: List[str] = None,
    min_sdgs: int = 3,
    min_perspectives: int = 2,
    top_n: int = 100,
    id_col: str = None,
    title_col: str = "Title",
    year_col: str = "Year",
    citations_col: str = "Cited by"
) -> List[BridgePaper]:
    """
    Identify papers that bridge multiple SDGs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with SDG indicators.
    min_sdgs : int
        Minimum number of SDGs to qualify as bridge.
    min_perspectives : int
        Minimum perspectives covered.
    top_n : int
        Number of top bridge papers to return.
    
    Returns
    -------
    List of BridgePaper objects.
    """
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    # Auto-detect ID column
    if id_col is None:
        for candidate in ["EID", "DOI", "unique-id", "id", "ID"]:
            if candidate in df.columns:
                id_col = candidate
                break
    
    bridge_papers = []
    
    for idx, row in df.iterrows():
        sdgs = get_paper_sdgs(row, sdg_cols)
        n_sdgs = len(sdgs)
        
        if n_sdgs < min_sdgs:
            continue
        
        perspectives = get_perspectives_for_sdgs(sdgs)
        n_perspectives = len(perspectives)
        
        if n_perspectives < min_perspectives:
            continue
        
        bridge_score = compute_bridge_score(sdgs)
        
        doc_id = row.get(id_col, idx) if id_col else idx
        title = row.get(title_col, "Unknown")
        year = int(row.get(year_col, 0)) if pd.notna(row.get(year_col)) else 0
        citations = int(row.get(citations_col, 0)) if pd.notna(row.get(citations_col)) else 0
        
        bridge_papers.append(BridgePaper(
            doc_id=doc_id,
            title=title[:200] if isinstance(title, str) else str(title)[:200],
            year=year,
            sdgs=sorted(sdgs),
            n_sdgs=n_sdgs,
            perspectives_covered=perspectives,
            n_perspectives=n_perspectives,
            bridge_score=bridge_score,
            citations=citations
        ))
    
    # Sort by bridge score
    bridge_papers.sort(key=lambda x: -x.bridge_score)
    
    return bridge_papers[:top_n]


def detect_sdg_clusters(
    G: Any,
    method: str = "louvain",
    n_clusters: int = None
) -> List[SDGCluster]:
    """
    Detect clusters of related SDGs.
    
    Parameters
    ----------
    G : nx.Graph
        SDG co-occurrence network.
    method : str
        "louvain", "hierarchical", or "spectral"
    n_clusters : int
        Number of clusters (for hierarchical/spectral).
    
    Returns
    -------
    List of SDGCluster objects.
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx required")
    
    # Get community assignments
    if method == "louvain" and LOUVAIN_AVAILABLE:
        try:
            partition = community_louvain.best_partition(G)
        except:
            partition = {n: 0 for n in G.nodes()}
    elif method == "hierarchical":
        # Use hierarchical clustering on adjacency matrix
        adj_matrix = nx.to_numpy_array(G)
        if n_clusters is None:
            n_clusters = min(4, len(G.nodes()) // 2)
        
        if len(G.nodes()) > 1:
            Z = linkage(adj_matrix, method='ward')
            labels = fcluster(Z, n_clusters, criterion='maxclust')
            partition = {node: labels[i] for i, node in enumerate(G.nodes())}
        else:
            partition = {n: 0 for n in G.nodes()}
    else:
        partition = {n: 0 for n in G.nodes()}
    
    # Group SDGs by cluster
    cluster_sdgs = defaultdict(list)
    for sdg, cluster_id in partition.items():
        cluster_sdgs[cluster_id].append(sdg)
    
    clusters = []
    for cluster_id, sdgs in cluster_sdgs.items():
        # Determine dominant perspective
        perspective_counts = Counter()
        for sdg in sdgs:
            for persp, persp_sdgs in SDG_PERSPECTIVES.items():
                if sdg in persp_sdgs:
                    perspective_counts[persp] += 1
        
        dominant = perspective_counts.most_common(1)[0][0] if perspective_counts else "Mixed"
        
        # Compute internal density
        subgraph = G.subgraph(sdgs)
        if len(sdgs) > 1:
            internal_density = nx.density(subgraph)
        else:
            internal_density = 1.0
        
        # Auto-generate name
        if len(sdgs) <= 3:
            name = " + ".join([SDG_SHORT_NAMES.get(s, str(s)) for s in sorted(sdgs)])
        else:
            name = f"{dominant} Cluster ({len(sdgs)} SDGs)"
        
        clusters.append(SDGCluster(
            cluster_id=cluster_id,
            sdgs=sorted(sdgs),
            name=name,
            dominant_perspective=dominant,
            internal_density=internal_density,
            external_connections={},
            key_terms=[]
        ))
    
    return clusters


def build_sdg_collaboration_network(
    df: pd.DataFrame,
    sdg_cols: List[str] = None,
    country_col: str = "Countries",
    sep: str = "; ",
    min_papers: int = 5
) -> Dict[int, Any]:
    """
    Build country collaboration networks for each SDG.
    
    Returns dict mapping SDG number to country collaboration network.
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx required")
    
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    if country_col not in df.columns:
        warnings.warn(f"Country column '{country_col}' not found")
        return {}
    
    sdg_networks = {}
    
    for sdg_col in sdg_cols:
        sdg_num = extract_sdg_number(sdg_col)
        sdg_df = df[df[sdg_col] == 1]
        
        if len(sdg_df) < min_papers:
            continue
        
        # Build country co-occurrence
        country_pairs = Counter()
        country_counts = Counter()
        
        for _, row in sdg_df.iterrows():
            countries_str = row.get(country_col, "")
            if pd.isna(countries_str) or not countries_str:
                continue
            
            countries = [c.strip() for c in str(countries_str).split(sep) if c.strip()]
            countries = list(set(countries))  # Unique
            
            for c in countries:
                country_counts[c] += 1
            
            for c1, c2 in combinations(sorted(countries), 2):
                country_pairs[(c1, c2)] += 1
        
        # Build network
        G = nx.Graph()
        
        for country, count in country_counts.items():
            if count >= 2:
                G.add_node(country, count=count)
        
        for (c1, c2), count in country_pairs.items():
            if c1 in G.nodes() and c2 in G.nodes() and count >= 1:
                G.add_edge(c1, c2, weight=count)
        
        sdg_networks[sdg_num] = G
    
    return sdg_networks


def analyze_temporal_sdg_networks(
    df: pd.DataFrame,
    sdg_cols: List[str] = None,
    year_col: str = "Year",
    window_size: int = 5,
    min_docs: int = 50
) -> Dict[str, Any]:
    """
    Build SDG networks for different time periods.
    
    Returns dict mapping period string to network.
    """
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    df_clean = df.dropna(subset=[year_col]).copy()
    df_clean[year_col] = df_clean[year_col].astype(int)
    
    min_year = df_clean[year_col].min()
    max_year = df_clean[year_col].max()
    
    temporal_networks = {}
    
    start = min_year
    while start <= max_year:
        end = min(start + window_size - 1, max_year)
        period = f"{start}-{end}"
        
        mask = (df_clean[year_col] >= start) & (df_clean[year_col] <= end)
        subset = df_clean[mask]
        
        if len(subset) >= min_docs:
            G, _, _ = build_sdg_cooccurrence_network(subset, sdg_cols)
            temporal_networks[period] = G
        
        start += window_size
    
    return temporal_networks


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_sdg_networks(
    df: pd.DataFrame,
    sdg_cols: List[str] = None,
    country_col: str = None,
    year_col: str = "Year",
    min_cooccurrence: int = 1,
    min_bridge_sdgs: int = 3,
    analyze_temporal: bool = True,
    window_size: int = 5,
    verbose: bool = True
) -> SDGNetworkAnalysis:
    """
    Comprehensive SDG network analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with SDG indicator columns.
    sdg_cols : List[str]
        SDG columns. Auto-detected if None.
    country_col : str
        Country column for collaboration analysis.
    year_col : str
        Year column for temporal analysis.
    min_cooccurrence : int
        Minimum co-occurrences for network edges.
    min_bridge_sdgs : int
        Minimum SDGs for bridge paper identification.
    analyze_temporal : bool
        Whether to build temporal networks.
    window_size : int
        Window size for temporal analysis.
    verbose : bool
        Print progress.
    
    Returns
    -------
    SDGNetworkAnalysis
        Complete network analysis results.
    """
    if verbose:
        print("="*60)
        print("SDG NETWORK ANALYSIS")
        print("="*60)
    
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    if not sdg_cols:
        raise ValueError("No SDG columns found. Run identify_sdgs() first.")
    
    if verbose:
        print(f"Found {len(sdg_cols)} SDG columns")
        print(f"Analyzing {len(df)} documents")
    
    # Auto-detect country column
    if country_col is None:
        for candidate in ["Countries", "Countries of Authors", "Country", "authorships.countries"]:
            if candidate in df.columns:
                country_col = candidate
                break
    
    # Build co-occurrence network
    if verbose:
        print("\nBuilding SDG co-occurrence network...")
    
    G, raw_matrix, norm_matrix = build_sdg_cooccurrence_network(
        df, sdg_cols, min_cooccurrence=min_cooccurrence
    )
    
    if verbose:
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Compute metrics
    if verbose:
        print("Computing network metrics...")
    
    sdg_metrics = compute_sdg_network_metrics(G)
    
    # Identify bridge papers
    if verbose:
        print("Identifying bridge papers...")
    
    bridge_papers = identify_bridge_papers(df, sdg_cols, min_sdgs=min_bridge_sdgs)
    top_bridge = bridge_papers[:20]
    
    if verbose:
        print(f"  Found {len(bridge_papers)} bridge papers")
    
    # Detect clusters
    if verbose:
        print("Detecting SDG clusters...")
    
    clusters = detect_sdg_clusters(G)
    
    if verbose:
        print(f"  Found {len(clusters)} clusters")
        for cluster in clusters:
            print(f"    {cluster.name}: SDGs {cluster.sdgs}")
    
    # Build collaboration networks
    collab_network = None
    if country_col:
        if verbose:
            print("Building collaboration networks...")
        collab_networks = build_sdg_collaboration_network(df, sdg_cols, country_col)
        # Merge into single network with SDG attributes
        # For now, just store the dict
        collab_network = collab_networks
    
    # Temporal analysis
    temporal_networks = {}
    if analyze_temporal and year_col in df.columns:
        if verbose:
            print("Analyzing temporal evolution...")
        temporal_networks = analyze_temporal_sdg_networks(
            df, sdg_cols, year_col, window_size
        )
        if verbose:
            print(f"  Built {len(temporal_networks)} temporal networks")
    
    # Compute summary statistics
    network_density = nx.density(G) if G.number_of_nodes() > 1 else 0
    avg_clustering = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
    
    communities = set(m.community for m in sdg_metrics.values())
    n_communities = len(communities)
    
    # Modularity
    if LOUVAIN_AVAILABLE and G.number_of_nodes() > 1:
        try:
            partition = community_louvain.best_partition(G)
            modularity = community_louvain.modularity(partition, G)
        except:
            modularity = 0
    else:
        modularity = 0
    
    if verbose:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nNetwork Statistics:")
        print(f"  Density: {network_density:.3f}")
        print(f"  Avg Clustering: {avg_clustering:.3f}")
        print(f"  Communities: {n_communities}")
        print(f"  Modularity: {modularity:.3f}")
        
        print(f"\nTop Connected SDGs:")
        metrics_df = pd.DataFrame([m.to_dict() for m in sdg_metrics.values()])
        top_3 = metrics_df.nlargest(3, 'Weighted_Degree')
        for _, row in top_3.iterrows():
            print(f"  SDG {int(row['SDG'])}: {row['Name']} (degree={row['Weighted_Degree']:.2f})")
    
    return SDGNetworkAnalysis(
        cooccurrence_network=G,
        collaboration_network=collab_network,
        sdg_metrics=sdg_metrics,
        bridge_papers=bridge_papers,
        top_bridge_papers=top_bridge,
        sdg_clusters=clusters,
        cooccurrence_matrix=raw_matrix,
        normalized_cooccurrence=norm_matrix,
        network_density=network_density,
        average_clustering=avg_clustering,
        n_communities=n_communities,
        modularity=modularity,
        temporal_networks=temporal_networks
    )


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_sdg_network(
    analysis: SDGNetworkAnalysis,
    layout: str = "spring",
    node_size_by: str = "count",  # "count", "degree", "betweenness"
    edge_width_scale: float = 5.0,
    figsize: Tuple[int, int] = (14, 12),
    title: str = "SDG Co-occurrence Network",
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Visualize SDG co-occurrence network.
    """
    G = analysis.cooccurrence_network
    
    if G.number_of_nodes() == 0:
        print("Empty network")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=2)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Node sizes
    if node_size_by == "count":
        sizes = [G.nodes[n].get('count', 100) for n in G.nodes()]
        max_size = max(sizes) if sizes else 1
        sizes = [s / max_size * 2000 + 200 for s in sizes]
    elif node_size_by == "degree":
        degrees = dict(G.degree(weight='weight'))
        max_deg = max(degrees.values()) if degrees else 1
        sizes = [degrees[n] / max_deg * 2000 + 200 for n in G.nodes()]
    else:
        betweenness = nx.betweenness_centrality(G)
        max_bet = max(betweenness.values()) if betweenness else 1
        sizes = [betweenness[n] / max_bet * 2000 + 200 for n in G.nodes()]
    
    # Node colors
    colors = [G.nodes[n].get('color', '#888888') for n in G.nodes()]
    
    # Edge widths
    edge_weights = [G[u][v].get('weight', 0.1) * edge_width_scale for u, v in G.edges()]
    
    # Draw
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, alpha=0.9, ax=ax)
    
    # Labels
    labels = {n: f"SDG {n}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_sdg_cooccurrence_heatmap(
    analysis: SDGNetworkAnalysis,
    normalized: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "YlOrRd",
    title: str = "SDG Co-occurrence Matrix",
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot SDG co-occurrence as heatmap.
    """
    matrix = analysis.normalized_cooccurrence if normalized else analysis.cooccurrence_matrix
    
    # Add SDG names to index
    new_index = [f"SDG {i}: {SDG_SHORT_NAMES.get(i, '')}" for i in matrix.index]
    new_cols = [f"SDG {i}" for i in matrix.columns]
    
    plot_matrix = matrix.copy()
    plot_matrix.index = new_index
    plot_matrix.columns = new_cols
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    sns.heatmap(
        plot_matrix,
        cmap=cmap,
        annot=True if len(matrix) <= 10 else False,
        fmt=".2f" if normalized else ".0f",
        ax=ax,
        cbar_kws={"label": "Jaccard Similarity" if normalized else "Co-occurrences"}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_bridge_papers_summary(
    analysis: SDGNetworkAnalysis,
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8),
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """Plot top bridge papers by score."""
    papers = analysis.top_bridge_papers[:top_n]
    
    if not papers:
        print("No bridge papers found")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    scores = [p.bridge_score for p in papers]
    titles = [p.title[:50] + "..." if len(p.title) > 50 else p.title for p in papers]
    
    # Color by number of SDGs
    colors = plt.cm.YlOrRd([p.n_sdgs / 10 for p in papers])
    
    bars = ax.barh(range(len(papers)), scores, color=colors)
    ax.set_yticks(range(len(papers)))
    ax.set_yticklabels(titles, fontsize=8)
    ax.set_xlabel("Bridge Score")
    ax.set_title("Top Bridge Papers (Connecting Multiple SDGs)", fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    # Add SDG count annotation
    for i, (bar, paper) in enumerate(zip(bars, papers)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{paper.n_sdgs} SDGs", va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_sdg_network_analysis(
    ba,  # BiblioAnalysis object
    min_cooccurrence: int = 1,
    analyze_temporal: bool = True,
    verbose: bool = True
) -> SDGNetworkAnalysis:
    """
    Convenience function to run SDG network analysis.
    
    Usage:
        from biblium.addons.sdg_networks import run_sdg_network_analysis
        
        ba.identify_sdgs()
        networks = run_sdg_network_analysis(ba)
        
        # View results
        print(networks.get_metrics_df())
        print(networks.get_top_connections())
        print(networks.get_bridge_papers_df())
    """
    sdg_cols = get_sdg_columns(ba.df)
    if not sdg_cols:
        raise ValueError("No SDG columns found. Run ba.identify_sdgs() first.")
    
    return analyze_sdg_networks(
        ba.df,
        sdg_cols=sdg_cols,
        analyze_temporal=analyze_temporal,
        min_cooccurrence=min_cooccurrence,
        verbose=verbose
    )
