# -*- coding: utf-8 -*-
"""
Disruption Index Module for Biblium
===================================

This module implements the Disruption Index (CD Index) and related metrics
for measuring whether research consolidates or disrupts existing knowledge.

The Disruption Index was introduced by Funk & Owen-Smith (2017) and refined
by Wu, Wang & Evans (2019) in "Large teams develop and small teams disrupt 
science and technology" (Nature).

Key metrics:
- CD Index: Core disruption index ranging from -1 (consolidating) to +1 (disruptive)
- DI Index: Simplified disruption index
- CD5 Index: 5-year window disruption index

Aggregation levels:
- Document-level: Individual paper disruption
- Source-level: Average disruption by journal/source
- Author-level: Average disruption by author
- Country-level: Average disruption by country
- Institution-level: Average disruption by affiliation

@author: Claude (Anthropic) for Lan.Umek
@version: 2.7.0
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import TwoSlopeNorm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# =============================================================================
# CITATION NETWORK BUILDING
# =============================================================================

def build_citation_network_from_refs(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    refs_col: str = "References",
    sep: str = "; ",
    normalize_ids: bool = True,
) -> Dict[str, Set[str]]:
    """
    Build citation network from references column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    id_col : str
        Column containing document IDs.
    refs_col : str
        Column containing references (semicolon-separated).
    sep : str
        Separator for parsing references.
    normalize_ids : bool
        Whether to normalize IDs (lowercase, strip).
    
    Returns
    -------
    Dict[str, Set[str]]
        Dictionary mapping paper_id -> set of paper_ids it cites.
    """
    network = {}
    
    for _, row in df.iterrows():
        doc_id = str(row.get(id_col, ""))
        if not doc_id or doc_id == "nan":
            continue
            
        if normalize_ids:
            doc_id = doc_id.strip().lower()
        
        refs_str = row.get(refs_col, "")
        if pd.isna(refs_str) or not refs_str:
            network[doc_id] = set()
            continue
        
        refs = set()
        for ref in str(refs_str).split(sep):
            ref = ref.strip()
            if ref and ref != "nan":
                if normalize_ids:
                    ref = ref.lower()
                refs.add(ref)
        
        network[doc_id] = refs
    
    return network


def build_reverse_citation_index(
    citation_network: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    """
    Build reverse citation index (who cites whom).
    
    Parameters
    ----------
    citation_network : Dict[str, Set[str]]
        Forward citation network {paper_id: set of refs}.
    
    Returns
    -------
    Dict[str, Set[str]]
        Reverse index {paper_id: set of papers that cite it}.
    """
    cited_by = defaultdict(set)
    
    for paper, refs in citation_network.items():
        for ref in refs:
            cited_by[ref].add(paper)
    
    return dict(cited_by)


# =============================================================================
# CORE DISRUPTION INDEX COMPUTATION
# =============================================================================

def compute_disruption_for_paper(
    doc_id: str,
    focal_refs: Set[str],
    citing_papers: Set[str],
    citation_network: Dict[str, Set[str]],
    cited_by: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """
    Compute disruption metrics for a single paper.
    
    The CD Index formula: CD = (n_i - n_j) / (n_i + n_j + n_k)
    
    Where:
    - n_i = citing papers that cite focal but NOT its references
    - n_j = citing papers that cite BOTH focal AND its references  
    - n_k = papers that cite focal's references but NOT focal
    
    Parameters
    ----------
    doc_id : str
        Document identifier.
    focal_refs : Set[str]
        References of the focal paper.
    citing_papers : Set[str]
        Papers that cite the focal paper.
    citation_network : Dict[str, Set[str]]
        Forward citation network.
    cited_by : Dict[str, Set[str]]
        Reverse citation index.
    
    Returns
    -------
    Dict with disruption metrics.
    """
    if not citing_papers:
        return {
            'doc_id': doc_id,
            'cd_index': np.nan,
            'di_index': np.nan,
            'n_citing': 0,
            'n_i': 0,
            'n_j': 0,
            'n_k': 0,
            'interpretation': 'uncited',
        }
    
    # Count n_i, n_j for citing papers
    n_i = 0  # Cite focal, don't cite focal's refs
    n_j = 0  # Cite focal AND focal's refs
    
    for citing_paper in citing_papers:
        citing_refs = citation_network.get(citing_paper, set())
        
        # Does this citing paper also cite any of focal's references?
        cites_focal_refs = bool(citing_refs & focal_refs)
        
        if cites_focal_refs:
            n_j += 1
        else:
            n_i += 1
    
    # Count n_k: papers that cite focal's references but not focal
    papers_citing_refs = set()
    for ref in focal_refs:
        ref_citers = cited_by.get(ref, set())
        papers_citing_refs.update(ref_citers)
    
    # Exclude papers that cite focal and focal itself
    n_k = len(papers_citing_refs - citing_papers - {doc_id})
    
    # Compute CD index
    denominator = n_i + n_j + n_k
    if denominator > 0:
        cd_index = (n_i - n_j) / denominator
    else:
        cd_index = 0.0
    
    # Simplified DI index (only considers citing papers)
    if n_i + n_j > 0:
        di_index = (n_i - n_j) / (n_i + n_j)
    else:
        di_index = 0.0
    
    # Interpret
    if cd_index >= 0.25:
        interpretation = "disruptive"
    elif cd_index <= -0.25:
        interpretation = "consolidating"
    else:
        interpretation = "neutral"
    
    return {
        'doc_id': doc_id,
        'cd_index': round(cd_index, 4),
        'di_index': round(di_index, 4),
        'n_citing': len(citing_papers),
        'n_i': n_i,
        'n_j': n_j,
        'n_k': n_k,
        'interpretation': interpretation,
    }


def compute_document_disruption(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    refs_col: str = "References",
    sep: str = "; ",
    citation_network: Dict[str, Set[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute Disruption Index for all documents.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    id_col : str
        Column with document IDs.
    refs_col : str
        Column with references.
    sep : str
        Separator for parsing references.
    citation_network : Dict[str, Set[str]], optional
        Pre-computed citation network.
    verbose : bool
        Print progress.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with disruption metrics for each document.
    """
    if verbose:
        print("Computing Document Disruption Index...")
    
    # Build citation network if not provided
    if citation_network is None:
        if verbose:
            print("  Building citation network from references...")
        citation_network = build_citation_network_from_refs(
            df, id_col, refs_col, sep
        )
    
    # Build reverse citation index
    if verbose:
        print("  Building reverse citation index...")
    cited_by = build_reverse_citation_index(citation_network)
    
    # Get all paper IDs in dataset
    all_papers = set()
    for doc_id in df[id_col].dropna():
        all_papers.add(str(doc_id).strip().lower())
    
    # Diagnostic info
    internal_citations = sum(
        1 for paper in all_papers 
        if paper in cited_by and cited_by[paper] & all_papers
    )
    
    if verbose:
        print(f"  Papers in dataset: {len(all_papers)}")
        print(f"  Papers with internal citations: {internal_citations}")
        
        if internal_citations < len(all_papers) * 0.1:
            print("  NOTE: Few internal citations detected.")
            print("    CD Index requires papers to cite each other within the dataset.")
            print("    Papers citing only external sources will have CD = NaN.")
    
    # Compute metrics for each paper
    results = []
    
    for _, row in df.iterrows():
        doc_id = str(row.get(id_col, ""))
        if not doc_id or doc_id == "nan":
            continue
        
        doc_id_norm = doc_id.strip().lower()
        
        # Get focal paper's references
        focal_refs = citation_network.get(doc_id_norm, set())
        
        # Get papers that cite the focal paper (internal only)
        citing_papers = cited_by.get(doc_id_norm, set()) & all_papers
        
        # Compute disruption
        metrics = compute_disruption_for_paper(
            doc_id_norm, focal_refs, citing_papers,
            citation_network, cited_by
        )
        
        # Use original doc_id for output
        metrics['doc_id'] = doc_id
        results.append(metrics)
    
    if verbose:
        print(f"  Computed disruption for {len(results)} documents")
    
    return pd.DataFrame(results)


# =============================================================================
# AGGREGATED DISRUPTION (SOURCES, AUTHORS, COUNTRIES, ETC.)
# =============================================================================

def aggregate_disruption_by_entity(
    disruption_df: pd.DataFrame,
    main_df: pd.DataFrame,
    entity_col: str,
    id_col: str = "unique-id",
    sep: str = "; ",
    min_docs: int = 1,
    metrics: List[str] = None,
) -> pd.DataFrame:
    """
    Aggregate disruption metrics by entity (source, author, country, etc.).
    
    Parameters
    ----------
    disruption_df : pd.DataFrame
        Document-level disruption metrics.
    main_df : pd.DataFrame
        Main bibliographic dataframe with entity column.
    entity_col : str
        Column to aggregate by.
    id_col : str
        Column with document IDs.
    sep : str
        Separator for multi-value columns.
    min_docs : int
        Minimum documents for entity to be included.
    metrics : List[str], optional
        Which metrics to aggregate. Default: cd_index, di_index.
    
    Returns
    -------
    pd.DataFrame
        Entity-level disruption statistics.
    """
    if metrics is None:
        metrics = ['cd_index', 'di_index']
    
    # Start with only the columns we need from main_df to avoid conflicts
    needed_cols = [id_col, entity_col]
    needed_cols = [c for c in needed_cols if c in main_df.columns]
    merged = main_df[needed_cols].copy()
    
    # Normalize doc_id for joining
    merged['_doc_id_norm'] = merged[id_col].astype(str).str.strip().str.lower()
    disruption_copy = disruption_df.copy()
    disruption_copy['_doc_id_norm'] = disruption_copy['doc_id'].astype(str).str.strip().str.lower()
    
    # Get columns to merge from disruption_df
    cols_to_merge = ['_doc_id_norm'] + [c for c in metrics + ['interpretation', 'n_citing'] 
                                         if c in disruption_copy.columns]
    
    merged = merged.merge(
        disruption_copy[cols_to_merge],
        on='_doc_id_norm',
        how='left'
    )
    
    # Handle multi-value columns (e.g., Authors)
    if entity_col not in merged.columns:
        raise ValueError(f"Column '{entity_col}' not found in dataframe")
    
    # Check if column contains separators (only for string columns)
    sample = merged[entity_col].dropna().head(100)
    is_multi = False
    if sample.dtype == 'object' or str(sample.dtype) == 'string':
        try:
            is_multi = sample.str.contains(sep, regex=False).any() if len(sample) > 0 else False
        except (AttributeError, TypeError):
            is_multi = False
    
    if is_multi:
        # Explode multi-value column
        merged['_entity_list'] = merged[entity_col].fillna('').str.split(sep)
        merged = merged.explode('_entity_list')
        merged['_entity_list'] = merged['_entity_list'].str.strip()
        merged = merged[merged['_entity_list'] != '']
        entity_col_use = '_entity_list'
    else:
        entity_col_use = entity_col
    
    # Build aggregation dictionary
    agg_dict = {}
    
    # Count documents
    agg_dict['_doc_id_norm'] = 'count'
    
    # Add metrics that exist in the merged dataframe
    metrics_present = [m for m in metrics if m in merged.columns]
    for metric in metrics_present:
        agg_dict[metric] = ['mean', 'median', 'std', 'min', 'max']
    
    if 'n_citing' in merged.columns:
        agg_dict['n_citing'] = 'sum'
    
    # Group and aggregate
    grouped = merged.groupby(entity_col_use).agg(agg_dict)
    
    # Flatten column names - handle both MultiIndex and regular Index
    if isinstance(grouped.columns, pd.MultiIndex):
        new_columns = []
        for col in grouped.columns:
            if col[1]:
                new_columns.append(f'{col[0]}_{col[1]}')
            else:
                new_columns.append(col[0])
        grouped.columns = new_columns
    else:
        # Single aggregation per column - rename directly
        rename_map = {}
        for col in grouped.columns:
            if col == '_doc_id_norm':
                rename_map[col] = 'n_documents'
            elif col == 'n_citing':
                rename_map[col] = 'total_citations'
        grouped = grouped.rename(columns=rename_map)
    
    # Rename count column (for MultiIndex case)
    if '_doc_id_norm_count' in grouped.columns:
        grouped = grouped.rename(columns={'_doc_id_norm_count': 'n_documents'})
    if 'n_citing_sum' in grouped.columns:
        grouped = grouped.rename(columns={'n_citing_sum': 'total_citations'})
    
    # Filter by min_docs
    grouped = grouped[grouped['n_documents'] >= min_docs]
    
    # Sort by mean cd_index
    if 'cd_index_mean' in grouped.columns:
        grouped = grouped.sort_values('cd_index_mean', ascending=False)
    
    # Reset index
    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={entity_col_use: entity_col})
    
    # Add interpretation counts
    if 'interpretation' in merged.columns:
        interp_counts = merged.groupby(entity_col_use)['interpretation'].value_counts().unstack(fill_value=0)
        interp_counts = interp_counts.reset_index()
        interp_counts = interp_counts.rename(columns={entity_col_use: entity_col})
        
        # Merge interpretation counts
        for col in ['disruptive', 'neutral', 'consolidating', 'uncited']:
            if col in interp_counts.columns:
                grouped = grouped.merge(
                    interp_counts[[entity_col, col]],
                    on=entity_col,
                    how='left'
                )
                grouped[f'n_{col}'] = grouped[col].fillna(0).astype(int)
                grouped = grouped.drop(columns=[col])
    
    return grouped


def compute_source_disruption(
    disruption_df: pd.DataFrame,
    main_df: pd.DataFrame,
    source_col: str = "Source title",
    id_col: str = "unique-id",
    min_docs: int = 3,
) -> pd.DataFrame:
    """
    Compute average Disruption Index by source/journal.
    
    Parameters
    ----------
    disruption_df : pd.DataFrame
        Document-level disruption metrics.
    main_df : pd.DataFrame
        Main bibliographic dataframe.
    source_col : str
        Column containing source/journal names.
    id_col : str
        Column with document IDs.
    min_docs : int
        Minimum documents for source to be included.
    
    Returns
    -------
    pd.DataFrame
        Source-level disruption statistics.
    """
    return aggregate_disruption_by_entity(
        disruption_df, main_df, source_col, id_col,
        sep="; ", min_docs=min_docs
    )


def compute_author_disruption(
    disruption_df: pd.DataFrame,
    main_df: pd.DataFrame,
    author_col: str = "Authors",
    id_col: str = "unique-id",
    sep: str = "; ",
    min_docs: int = 2,
) -> pd.DataFrame:
    """
    Compute average Disruption Index by author.
    
    Parameters
    ----------
    disruption_df : pd.DataFrame
        Document-level disruption metrics.
    main_df : pd.DataFrame
        Main bibliographic dataframe.
    author_col : str
        Column containing author names.
    id_col : str
        Column with document IDs.
    sep : str
        Separator for author names.
    min_docs : int
        Minimum documents for author to be included.
    
    Returns
    -------
    pd.DataFrame
        Author-level disruption statistics.
    """
    return aggregate_disruption_by_entity(
        disruption_df, main_df, author_col, id_col,
        sep=sep, min_docs=min_docs
    )


def compute_country_disruption(
    disruption_df: pd.DataFrame,
    main_df: pd.DataFrame,
    country_col: str = "Countries",
    id_col: str = "unique-id",
    sep: str = "; ",
    min_docs: int = 5,
) -> pd.DataFrame:
    """
    Compute average Disruption Index by country.
    
    Parameters
    ----------
    disruption_df : pd.DataFrame
        Document-level disruption metrics.
    main_df : pd.DataFrame
        Main bibliographic dataframe.
    country_col : str
        Column containing country names.
    id_col : str
        Column with document IDs.
    sep : str
        Separator for country names.
    min_docs : int
        Minimum documents for country to be included.
    
    Returns
    -------
    pd.DataFrame
        Country-level disruption statistics.
    """
    return aggregate_disruption_by_entity(
        disruption_df, main_df, country_col, id_col,
        sep=sep, min_docs=min_docs
    )


def compute_affiliation_disruption(
    disruption_df: pd.DataFrame,
    main_df: pd.DataFrame,
    affiliation_col: str = "Affiliations",
    id_col: str = "unique-id",
    sep: str = "; ",
    min_docs: int = 3,
) -> pd.DataFrame:
    """
    Compute average Disruption Index by affiliation/institution.
    
    Parameters
    ----------
    disruption_df : pd.DataFrame
        Document-level disruption metrics.
    main_df : pd.DataFrame
        Main bibliographic dataframe.
    affiliation_col : str
        Column containing affiliation names.
    id_col : str
        Column with document IDs.
    sep : str
        Separator for affiliation names.
    min_docs : int
        Minimum documents for affiliation to be included.
    
    Returns
    -------
    pd.DataFrame
        Affiliation-level disruption statistics.
    """
    return aggregate_disruption_by_entity(
        disruption_df, main_df, affiliation_col, id_col,
        sep=sep, min_docs=min_docs
    )


def compute_year_disruption(
    disruption_df: pd.DataFrame,
    main_df: pd.DataFrame,
    year_col: str = "Year",
    id_col: str = "unique-id",
) -> pd.DataFrame:
    """
    Compute average Disruption Index by year.
    
    Parameters
    ----------
    disruption_df : pd.DataFrame
        Document-level disruption metrics.
    main_df : pd.DataFrame
        Main bibliographic dataframe.
    year_col : str
        Column containing publication year.
    id_col : str
        Column with document IDs.
    
    Returns
    -------
    pd.DataFrame
        Year-level disruption statistics.
    """
    return aggregate_disruption_by_entity(
        disruption_df, main_df, year_col, id_col,
        sep="|||", min_docs=1  # No separator for year
    )


# =============================================================================
# ADD DISRUPTION TO EXISTING DATAFRAMES
# =============================================================================

def add_disruption_to_df(
    df: pd.DataFrame,
    disruption_df: pd.DataFrame,
    id_col: str = "unique-id",
    columns: List[str] = None,
) -> pd.DataFrame:
    """
    Add disruption columns to an existing dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Target dataframe.
    disruption_df : pd.DataFrame
        Document-level disruption metrics.
    id_col : str
        Column with document IDs.
    columns : List[str], optional
        Which disruption columns to add. Default: all.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with disruption columns added.
    """
    if columns is None:
        columns = ['cd_index', 'di_index', 'interpretation', 'n_citing', 'n_i', 'n_j', 'n_k']
    
    # Keep only requested columns
    cols_to_add = ['doc_id'] + [c for c in columns if c in disruption_df.columns]
    disruption_subset = disruption_df[cols_to_add].copy()
    
    # Normalize for joining
    df = df.copy()
    df['_doc_id_join'] = df[id_col].astype(str).str.strip().str.lower()
    disruption_subset['_doc_id_join'] = disruption_subset['doc_id'].astype(str).str.strip().str.lower()
    
    # Merge
    result = df.merge(
        disruption_subset.drop(columns=['doc_id']),
        on='_doc_id_join',
        how='left'
    )
    
    # Clean up
    result = result.drop(columns=['_doc_id_join'])
    
    return result


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_disruption_distribution(
    disruption_df: pd.DataFrame,
    metric: str = 'cd_index',
    title: str = None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = None,
    ax: plt.Axes = None,
    show_stats: bool = True,
) -> plt.Figure:
    """
    Plot distribution of disruption index.
    
    Parameters
    ----------
    disruption_df : pd.DataFrame
        Disruption metrics dataframe.
    metric : str
        Which metric to plot.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size.
    color : str, optional
        Bar color.
    ax : plt.Axes, optional
        Existing axes to plot on.
    show_stats : bool
        Show statistics box.
    
    Returns
    -------
    plt.Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get valid data
    data = disruption_df[metric].dropna()
    
    if len(data) == 0:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Histogram
    bins = np.linspace(-1, 1, 41)
    n, bins_edges, patches = ax.hist(data, bins=bins, color=color or '#3498db', 
                                      alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Color bars based on value
    for i, patch in enumerate(patches):
        bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
        if bin_center > 0.25:
            patch.set_facecolor('#27ae60')  # Green for disruptive
        elif bin_center < -0.25:
            patch.set_facecolor('#e74c3c')  # Red for consolidating
        else:
            patch.set_facecolor('#95a5a6')  # Gray for neutral
    
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=0.25, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.25, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels
    ax.set_xlabel(f'{metric.replace("_", " ").title()}', fontsize=11)
    ax.set_ylabel('Number of Documents', fontsize=11)
    ax.set_title(title or f'Distribution of {metric.replace("_", " ").title()}', fontsize=13)
    ax.set_xlim(-1.05, 1.05)
    
    # Statistics box
    if show_stats:
        stats_text = (
            f'n = {len(data):,}\n'
            f'Mean = {data.mean():.3f}\n'
            f'Median = {data.median():.3f}\n'
            f'Std = {data.std():.3f}'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#27ae60', label='Disruptive (> 0.25)'),
        mpatches.Patch(facecolor='#95a5a6', label='Neutral'),
        mpatches.Patch(facecolor='#e74c3c', label='Consolidating (< -0.25)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_disruption_by_entity(
    entity_df: pd.DataFrame,
    entity_col: str,
    metric: str = 'cd_index_mean',
    top_n: int = 20,
    title: str = None,
    figsize: Tuple[int, int] = (12, 8),
    ax: plt.Axes = None,
    show_error: bool = True,
) -> plt.Figure:
    """
    Plot disruption index by entity (bar chart).
    
    Parameters
    ----------
    entity_df : pd.DataFrame
        Entity-level disruption statistics.
    entity_col : str
        Column with entity names.
    metric : str
        Which metric to plot.
    top_n : int
        Number of entities to show.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size.
    ax : plt.Axes, optional
        Existing axes.
    show_error : bool
        Show error bars (std).
    
    Returns
    -------
    plt.Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Sort and get top N
    if metric in entity_df.columns:
        df_sorted = entity_df.nlargest(top_n, metric).copy()
    else:
        df_sorted = entity_df.head(top_n).copy()
    
    # Truncate long names
    df_sorted[entity_col] = df_sorted[entity_col].astype(str).str[:50]
    
    # Colors based on value
    colors = []
    for val in df_sorted[metric]:
        if val > 0.25:
            colors.append('#27ae60')
        elif val < -0.25:
            colors.append('#e74c3c')
        else:
            colors.append('#3498db')
    
    # Bar plot
    y_pos = range(len(df_sorted))
    bars = ax.barh(y_pos, df_sorted[metric], color=colors, alpha=0.8, edgecolor='white')
    
    # Error bars
    std_col = metric.replace('mean', 'std')
    if show_error and std_col in df_sorted.columns:
        ax.errorbar(df_sorted[metric], y_pos, xerr=df_sorted[std_col],
                   fmt='none', ecolor='gray', alpha=0.5, capsize=3)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted[entity_col], fontsize=9)
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_title(title or f'Disruption Index by {entity_col}', fontsize=13)
    
    # Reference line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add document count as text
    if 'n_documents' in df_sorted.columns:
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax.text(ax.get_xlim()[1], i, f'  n={int(row["n_documents"])}',
                   va='center', fontsize=8, color='gray')
    
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_disruption_over_time(
    year_df: pd.DataFrame,
    year_col: str = "Year",
    metric: str = 'cd_index_mean',
    title: str = None,
    figsize: Tuple[int, int] = (12, 6),
    ax: plt.Axes = None,
    show_ci: bool = True,
) -> plt.Figure:
    """
    Plot disruption index trend over time.
    
    Parameters
    ----------
    year_df : pd.DataFrame
        Year-level disruption statistics.
    year_col : str
        Column with years.
    metric : str
        Which metric to plot.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size.
    ax : plt.Axes, optional
        Existing axes.
    show_ci : bool
        Show confidence interval (±1 std).
    
    Returns
    -------
    plt.Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Sort by year
    df_sorted = year_df.sort_values(year_col).copy()
    
    years = df_sorted[year_col].astype(int)
    values = df_sorted[metric]
    
    # Line plot
    ax.plot(years, values, 'o-', color='#3498db', linewidth=2, markersize=6, label='Mean CD Index')
    
    # Confidence interval
    std_col = metric.replace('mean', 'std')
    if show_ci and std_col in df_sorted.columns:
        ax.fill_between(years, 
                       values - df_sorted[std_col],
                       values + df_sorted[std_col],
                       alpha=0.2, color='#3498db', label='±1 Std')
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=0.25, color='green', linestyle='--', linewidth=1, alpha=0.3, label='Disruptive threshold')
    ax.axhline(y=-0.25, color='red', linestyle='--', linewidth=1, alpha=0.3, label='Consolidating threshold')
    
    # Labels
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_title(title or 'Disruption Index Over Time', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    
    # Set y limits
    ax.set_ylim(-1, 1)
    
    plt.tight_layout()
    return fig


def plot_disruption_scatter(
    disruption_df: pd.DataFrame,
    main_df: pd.DataFrame,
    id_col: str = "unique-id",
    x_col: str = "Year",
    y_col: str = "cd_index",
    size_col: str = "Cited by",
    color_by: str = "interpretation",
    title: str = None,
    figsize: Tuple[int, int] = (12, 8),
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Scatter plot of disruption vs. another variable.
    
    Parameters
    ----------
    disruption_df : pd.DataFrame
        Disruption metrics.
    main_df : pd.DataFrame
        Main bibliographic dataframe.
    id_col : str
        Document ID column.
    x_col : str
        Column for x-axis.
    y_col : str
        Column for y-axis (from disruption_df).
    size_col : str
        Column for point size.
    color_by : str
        Column to color by.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size.
    ax : plt.Axes, optional
        Existing axes.
    
    Returns
    -------
    plt.Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Merge dataframes
    merged = add_disruption_to_df(main_df, disruption_df, id_col)
    
    # Get valid data
    if x_col not in merged.columns or y_col not in merged.columns:
        ax.text(0.5, 0.5, f"Column not found: {x_col} or {y_col}", 
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    data = merged[[x_col, y_col, color_by]].dropna()
    
    # Size
    if size_col in merged.columns:
        sizes = merged.loc[data.index, size_col].fillna(10)
        sizes = np.clip(sizes, 10, 1000)
        sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1) * 200 + 20
    else:
        sizes = 50
    
    # Colors
    color_map = {
        'disruptive': '#27ae60',
        'neutral': '#95a5a6',
        'consolidating': '#e74c3c',
        'uncited': '#bdc3c7',
    }
    colors = data[color_by].map(color_map).fillna('#95a5a6')
    
    # Scatter
    ax.scatter(data[x_col], data[y_col], c=colors, s=sizes, alpha=0.6, edgecolor='white')
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=0.25, color='green', linestyle='--', linewidth=1, alpha=0.3)
    ax.axhline(y=-0.25, color='red', linestyle='--', linewidth=1, alpha=0.3)
    
    # Labels
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=11)
    ax.set_title(title or f'{y_col.replace("_", " ").title()} vs {x_col}', fontsize=13)
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor=c, label=l) 
                       for l, c in color_map.items() if l in data[color_by].values]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_disruption_comparison(
    entity_dfs: Dict[str, pd.DataFrame],
    metric: str = 'cd_index_mean',
    title: str = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Compare disruption distributions across entity types.
    
    Parameters
    ----------
    entity_dfs : Dict[str, pd.DataFrame]
        Dictionary mapping entity type to disruption dataframe.
    metric : str
        Which metric to compare.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    plt.Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    data_to_plot = []
    labels = []
    
    for entity_type, df in entity_dfs.items():
        if metric in df.columns:
            data_to_plot.append(df[metric].dropna().values)
            labels.append(entity_type)
    
    if not data_to_plot:
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Box plot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    colors = ['#3498db', '#e74c3c', '#27ae60', '#9b59b6', '#f39c12']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=0.25, color='green', linestyle='--', linewidth=1, alpha=0.3)
    ax.axhline(y=-0.25, color='red', linestyle='--', linewidth=1, alpha=0.3)
    
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_title(title or 'Disruption Index Comparison', fontsize=13)
    
    plt.tight_layout()
    return fig


# =============================================================================
# MIXIN CLASS FOR INTEGRATION WITH BIBLIOSTATS
# =============================================================================

class DisruptionMixin:
    """
    Mixin class providing disruption index methods for BiblioStats.
    
    This mixin adds methods to compute disruption metrics and adds them
    to the main df and entity count dataframes.
    """
    
    # Storage attributes
    disruption_df: pd.DataFrame = None
    
    def compute_disruption_index(
        self,
        id_col: str = None,
        refs_col: str = None,
        sep: str = None,
        verbose: bool = True,
        add_to_df: bool = True,
        add_to_counts: bool = True,
    ) -> pd.DataFrame:
        """
        Compute Disruption Index (CD Index) for all documents.
        
        The CD Index measures whether a paper consolidates or disrupts existing
        knowledge:
        - CD = +1: Fully disruptive (citers ignore the paper's references)
        - CD = 0: Neutral  
        - CD = -1: Fully consolidating (citers always cite the paper's references)
        
        Parameters
        ----------
        id_col : str, optional
            Column with document IDs. Auto-detected if not provided.
        refs_col : str, optional
            Column with references. Auto-detected if not provided.
        sep : str, optional
            Separator for references. Default: "; "
        verbose : bool
            Print progress messages.
        add_to_df : bool
            Add disruption columns to self.df.
        add_to_counts : bool
            Add disruption columns to entity count dataframes.
        
        Returns
        -------
        pd.DataFrame
            Document-level disruption metrics.
        
        Notes
        -----
        The CD Index is computed based on citations within the dataset. This works
        best when:
        
        - **OpenAlex**: References contain OpenAlex IDs that can be directly matched.
          This provides the most accurate results.
        
        - **Scopus/Web of Science**: References are formatted citation strings.
          The algorithm attempts to match references to documents in the dataset
          by comparing IDs (DOI, EID, UT). Results depend on how many documents
          in your dataset cite each other.
        
        For best results with Scopus/WoS data, use a focused dataset where papers
        are likely to cite each other (e.g., papers from a specific research area
        or journal over time).
        
        Examples
        --------
        >>> bib = BiblioAnalysis("data.csv", db="scopus")
        >>> di = bib.compute_disruption_index()
        >>> print(di[['doc_id', 'cd_index', 'interpretation']].head())
        """
        # Auto-detect columns
        if id_col is None:
            id_col = self.mapping.get("unique-id", "unique-id")
            if id_col not in self.df.columns:
                for alt in ["DOI", "EID", "unique-id", "UT"]:
                    if alt in self.df.columns:
                        id_col = alt
                        break
        
        if refs_col is None:
            refs_col = self.mapping.get("References", "References")
            if refs_col not in self.df.columns:
                for alt in ["References", "Cited References", "CR"]:
                    if alt in self.df.columns:
                        refs_col = alt
                        break
        
        if sep is None:
            sep = getattr(self, 'default_separator', '; ')
        
        # Compute disruption
        self.disruption_df = compute_document_disruption(
            self.df,
            id_col=id_col,
            refs_col=refs_col,
            sep=sep,
            verbose=verbose,
        )
        
        # Add to main dataframe
        if add_to_df:
            self.df = add_disruption_to_df(
                self.df, self.disruption_df, id_col,
                columns=['cd_index', 'di_index', 'interpretation']
            )
        
        # Add to entity count dataframes
        if add_to_counts:
            self._add_disruption_to_counts(verbose=verbose)
        
        # Save if res_folder is set
        if hasattr(self, 'res_folder') and self.res_folder:
            try:
                from biblium import utilsbib
                utilsbib.to_excel_fancy(
                    self.disruption_df,
                    f_name=os.path.join(self.res_folder, "tables", "disruption_documents.xlsx"),
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not save disruption table: {e}")
        
        return self.disruption_df
    
    def _add_disruption_to_counts(self, verbose: bool = True):
        """Add disruption metrics to existing count dataframes."""
        if self.disruption_df is None:
            return
        
        id_col = self.mapping.get("unique-id", "unique-id")
        sep = getattr(self, 'default_separator', '; ')
        
        # Sources
        if hasattr(self, 'sources_counts_df') and self.sources_counts_df is not None:
            source_col = self.mapping.get("Source_title", "Source title")
            try:
                source_di = aggregate_disruption_by_entity(
                    self.disruption_df, self.df, source_col, id_col, sep="|||", min_docs=1
                )
                # Merge with existing counts - match on source name
                src_name_col = 'Source' if 'Source' in self.sources_counts_df.columns else source_col
                di_cols = ['cd_index_mean', 'cd_index_median', 'cd_index_std', 'di_index_mean']
                di_cols = [c for c in di_cols if c in source_di.columns]
                if di_cols:
                    source_di = source_di.rename(columns={source_col: src_name_col})
                    self.sources_counts_df = self.sources_counts_df.merge(
                        source_di[[src_name_col] + di_cols],
                        on=src_name_col, how='left'
                    )
                    if verbose:
                        print(f"  Added disruption to sources_counts_df")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not add disruption to sources: {e}")
        
        # Authors
        if hasattr(self, 'authors_counts_df') and self.authors_counts_df is not None:
            author_col = self.mapping.get("Authors", "Authors")
            try:
                author_di = aggregate_disruption_by_entity(
                    self.disruption_df, self.df, author_col, id_col, sep=sep, min_docs=1
                )
                # Match on author name - try different column names
                auth_name_col = None
                for col_name in ['Author', 'Authors', author_col]:
                    if col_name in self.authors_counts_df.columns:
                        auth_name_col = col_name
                        break
                
                if auth_name_col is None:
                    if verbose:
                        print(f"  Warning: Could not find author name column in authors_counts_df")
                else:
                    di_cols = ['cd_index_mean', 'cd_index_median', 'cd_index_std', 'di_index_mean']
                    di_cols = [c for c in di_cols if c in author_di.columns]
                    if di_cols:
                        author_di = author_di.rename(columns={author_col: auth_name_col})
                        self.authors_counts_df = self.authors_counts_df.merge(
                            author_di[[auth_name_col] + di_cols],
                            on=auth_name_col, how='left'
                        )
                        if verbose:
                            print(f"  Added disruption to authors_counts_df")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not add disruption to authors: {e}")
        
        # Countries
        if hasattr(self, 'all_countries_counts_df') and self.all_countries_counts_df is not None:
            # Try to find country column
            country_col = None
            for col in ["Countries", "Country", "All Countries", "Countries of Authors"]:
                if col in self.df.columns:
                    country_col = col
                    break
            
            if country_col:
                try:
                    country_di = aggregate_disruption_by_entity(
                        self.disruption_df, self.df, country_col, id_col, sep=sep, min_docs=1
                    )
                    # Match on country name
                    cnt_name_col = 'Country' if 'Country' in self.all_countries_counts_df.columns else country_col
                    di_cols = ['cd_index_mean', 'cd_index_median', 'cd_index_std', 'di_index_mean']
                    di_cols = [c for c in di_cols if c in country_di.columns]
                    if di_cols:
                        country_di = country_di.rename(columns={country_col: cnt_name_col})
                        self.all_countries_counts_df = self.all_countries_counts_df.merge(
                            country_di[[cnt_name_col] + di_cols],
                            on=cnt_name_col, how='left'
                        )
                        if verbose:
                            print(f"  Added disruption to all_countries_counts_df")
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not add disruption to countries: {e}")
    
    def get_disruption_by_year(self) -> pd.DataFrame:
        """
        Get disruption index aggregated by year.
        
        Returns
        -------
        pd.DataFrame
            Year-level disruption statistics.
        """
        if self.disruption_df is None:
            self.compute_disruption_index()
        
        year_col = self.mapping.get("Year", "Year")
        id_col = self.mapping.get("unique-id", "unique-id")
        
        return aggregate_disruption_by_entity(
            self.disruption_df, self.df, year_col, id_col,
            sep="|||", min_docs=1
        )
    
    def get_top_disruptive(self, n: int = 20) -> pd.DataFrame:
        """
        Get top N most disruptive papers.
        
        Parameters
        ----------
        n : int
            Number of papers to return.
        
        Returns
        -------
        pd.DataFrame
            Top disruptive papers with metadata.
        """
        if self.disruption_df is None:
            self.compute_disruption_index()
        
        # Get top disruptive
        top = self.disruption_df.nlargest(n, 'cd_index')
        
        # Merge with main df for metadata
        id_col = self.mapping.get("unique-id", "unique-id")
        title_col = self.mapping.get("Title", "Title")
        year_col = self.mapping.get("Year", "Year")
        cit_col = self.mapping.get("Cited_by", "Cited by")
        
        self.df['_doc_id_join'] = self.df[id_col].astype(str).str.strip().str.lower()
        top['_doc_id_join'] = top['doc_id'].astype(str).str.strip().str.lower()
        
        cols_to_get = [c for c in [id_col, title_col, year_col, cit_col, '_doc_id_join'] 
                       if c in self.df.columns]
        
        result = top.merge(
            self.df[cols_to_get].drop_duplicates('_doc_id_join'),
            on='_doc_id_join', how='left'
        ).drop(columns=['_doc_id_join'])
        
        self.df = self.df.drop(columns=['_doc_id_join'])
        
        return result
    
    def get_top_consolidating(self, n: int = 20) -> pd.DataFrame:
        """
        Get top N most consolidating papers.
        
        Parameters
        ----------
        n : int
            Number of papers to return.
        
        Returns
        -------
        pd.DataFrame
            Top consolidating papers with metadata.
        """
        if self.disruption_df is None:
            self.compute_disruption_index()
        
        # Get top consolidating (lowest cd_index)
        top = self.disruption_df.nsmallest(n, 'cd_index')
        
        # Merge with main df for metadata
        id_col = self.mapping.get("unique-id", "unique-id")
        title_col = self.mapping.get("Title", "Title")
        year_col = self.mapping.get("Year", "Year")
        cit_col = self.mapping.get("Cited_by", "Cited by")
        
        self.df['_doc_id_join'] = self.df[id_col].astype(str).str.strip().str.lower()
        top['_doc_id_join'] = top['doc_id'].astype(str).str.strip().str.lower()
        
        cols_to_get = [c for c in [id_col, title_col, year_col, cit_col, '_doc_id_join'] 
                       if c in self.df.columns]
        
        result = top.merge(
            self.df[cols_to_get].drop_duplicates('_doc_id_join'),
            on='_doc_id_join', how='left'
        ).drop(columns=['_doc_id_join'])
        
        self.df = self.df.drop(columns=['_doc_id_join'])
        
        return result
    
    def get_source_disruption(self, min_docs: int = 3) -> pd.DataFrame:
        """
        Get average Disruption Index by source/journal.
        
        Parameters
        ----------
        min_docs : int
            Minimum documents for source to be included.
        
        Returns
        -------
        pd.DataFrame
            Source-level disruption statistics.
        """
        if self.disruption_df is None:
            self.compute_disruption_index()
        
        source_col = self.mapping.get("Source_title", "Source title")
        id_col = self.mapping.get("unique-id", "unique-id")
        
        return aggregate_disruption_by_entity(
            self.disruption_df, self.df,
            entity_col=source_col,
            id_col=id_col,
            sep="|||",  # Sources typically don't have separators
            min_docs=min_docs,
        )
    
    def get_author_disruption(self, min_docs: int = 2) -> pd.DataFrame:
        """
        Get average Disruption Index by author.
        
        Parameters
        ----------
        min_docs : int
            Minimum documents for author to be included.
        
        Returns
        -------
        pd.DataFrame
            Author-level disruption statistics.
        """
        if self.disruption_df is None:
            self.compute_disruption_index()
        
        author_col = self.mapping.get("Authors", "Authors")
        id_col = self.mapping.get("unique-id", "unique-id")
        sep = getattr(self, 'default_separator', '; ')
        
        return aggregate_disruption_by_entity(
            self.disruption_df, self.df,
            entity_col=author_col,
            id_col=id_col,
            sep=sep,
            min_docs=min_docs,
        )
    
    def get_country_disruption(self, min_docs: int = 5) -> pd.DataFrame:
        """
        Get average Disruption Index by country.
        
        Parameters
        ----------
        min_docs : int
            Minimum documents for country to be included.
        
        Returns
        -------
        pd.DataFrame
            Country-level disruption statistics.
        """
        if self.disruption_df is None:
            self.compute_disruption_index()
        
        # Try to find country column
        country_col = None
        for col in ["Countries", "Country", "All Countries", "CA Country"]:
            if col in self.df.columns:
                country_col = col
                break
        
        if country_col is None:
            country_col = self.mapping.get("Countries", "Countries")
        
        id_col = self.mapping.get("unique-id", "unique-id")
        sep = getattr(self, 'default_separator', '; ')
        
        return aggregate_disruption_by_entity(
            self.disruption_df, self.df,
            entity_col=country_col,
            id_col=id_col,
            sep=sep,
            min_docs=min_docs,
        )
    
    def get_affiliation_disruption(self, min_docs: int = 3) -> pd.DataFrame:
        """
        Get average Disruption Index by affiliation/institution.
        
        Parameters
        ----------
        min_docs : int
            Minimum documents for affiliation to be included.
        
        Returns
        -------
        pd.DataFrame
            Affiliation-level disruption statistics.
        """
        if self.disruption_df is None:
            self.compute_disruption_index()
        
        affiliation_col = self.mapping.get("Affiliations", "Affiliations")
        id_col = self.mapping.get("unique-id", "unique-id")
        sep = getattr(self, 'default_separator', '; ')
        
        return aggregate_disruption_by_entity(
            self.disruption_df, self.df,
            entity_col=affiliation_col,
            id_col=id_col,
            sep=sep,
            min_docs=min_docs,
        )
    
    def get_year_disruption(self) -> pd.DataFrame:
        """
        Get average Disruption Index by year.
        
        Returns
        -------
        pd.DataFrame
            Year-level disruption statistics.
        """
        if self.disruption_df is None:
            self.compute_disruption_index()
        
        year_col = self.mapping.get("Year", "Year")
        id_col = self.mapping.get("unique-id", "unique-id")
        
        return aggregate_disruption_by_entity(
            self.disruption_df, self.df,
            entity_col=year_col,
            id_col=id_col,
            sep="|||",  # Years don't have separators
            min_docs=1,
        )
