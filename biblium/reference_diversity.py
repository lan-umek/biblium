# -*- coding: utf-8 -*-
"""
Reference Diversity Analysis
============================
Analyze diversity metrics of references/citations in bibliometric datasets.

Metrics include:
- Reference count and density
- Source diversity (journals/venues)
- Topic/Field diversity (interdisciplinarity)
- Age diversity (temporal spread of references)
- Geographic diversity (countries of referenced works)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
import math

import numpy as np
import pandas as pd


class DiversityLevel(Enum):
    """Diversity level classification."""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"


@dataclass
class ReferenceDiversityMetrics:
    """Diversity metrics for a single paper's references."""
    # Paper identification
    doi: str = ""
    title: str = ""
    pub_year: int = 0
    
    # Basic counts
    reference_count: int = 0
    unique_sources: int = 0
    unique_fields: int = 0
    unique_topics: int = 0
    unique_countries: int = 0
    
    # Age metrics
    median_ref_age: float = 0.0
    mean_ref_age: float = 0.0
    ref_age_std: float = 0.0
    oldest_ref_age: int = 0
    newest_ref_age: int = 0
    
    # Diversity indices
    source_diversity: float = 0.0  # Shannon entropy for sources
    field_diversity: float = 0.0   # Shannon entropy for fields
    topic_diversity: float = 0.0   # Shannon entropy for topics
    
    # Interdisciplinarity
    rao_stirling_index: float = 0.0  # Rao-Stirling interdisciplinarity
    
    # Self-citation
    self_citation_rate: float = 0.0
    self_citation_count: int = 0
    
    # Classification
    diversity_level: str = "Moderate"
    
    # Raw data for detailed analysis
    sources_distribution: Dict[str, int] = field(default_factory=dict)
    fields_distribution: Dict[str, int] = field(default_factory=dict)
    topics_distribution: Dict[str, int] = field(default_factory=dict)
    ref_years: List[int] = field(default_factory=list)


@dataclass
class ReferenceDiversityResult:
    """Result of reference diversity analysis."""
    metrics: List[ReferenceDiversityMetrics]
    
    # Aggregate statistics
    n_papers: int = 0
    n_analyzed: int = 0
    n_with_references: int = 0
    
    # Dataset-level diversity
    avg_reference_count: float = 0.0
    avg_source_diversity: float = 0.0
    avg_field_diversity: float = 0.0
    avg_ref_age: float = 0.0
    avg_self_citation_rate: float = 0.0
    
    # Distribution of diversity levels
    diversity_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Data source info
    data_source: str = "dataset"  # "dataset", "openalex", "partial"
    api_calls_made: int = 0


def compute_shannon_entropy(distribution: Dict[str, int]) -> float:
    """
    Compute Shannon entropy (diversity index).
    
    H = -sum(p_i * log(p_i))
    
    Returns value between 0 (no diversity) and log(n) (max diversity).
    Normalized to 0-1 range.
    """
    if not distribution:
        return 0.0
    
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    
    n_categories = len(distribution)
    if n_categories <= 1:
        return 0.0
    
    entropy = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    
    # Normalize by max possible entropy
    max_entropy = math.log(n_categories)
    if max_entropy > 0:
        return entropy / max_entropy
    return 0.0


def compute_simpson_diversity(distribution: Dict[str, int]) -> float:
    """
    Compute Simpson's diversity index.
    
    D = 1 - sum(p_i^2)
    
    Returns value between 0 (no diversity) and 1 (max diversity).
    """
    if not distribution:
        return 0.0
    
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    
    sum_squared = sum((count / total) ** 2 for count in distribution.values())
    return 1 - sum_squared


def compute_rao_stirling(
    distribution: Dict[str, int],
    distance_matrix: Optional[Dict[Tuple[str, str], float]] = None
) -> float:
    """
    Compute Rao-Stirling interdisciplinarity index.
    
    RS = sum_i sum_j (d_ij * p_i * p_j) for i != j
    
    Where d_ij is the distance between categories i and j.
    If no distance matrix provided, assumes uniform distance of 1.
    """
    if not distribution or len(distribution) <= 1:
        return 0.0
    
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    
    categories = list(distribution.keys())
    n = len(categories)
    
    rs_index = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            p_i = distribution[categories[i]] / total
            p_j = distribution[categories[j]] / total
            
            # Get distance (default to 1 if not in matrix)
            if distance_matrix:
                d_ij = distance_matrix.get((categories[i], categories[j]), 
                       distance_matrix.get((categories[j], categories[i]), 1.0))
            else:
                d_ij = 1.0
            
            rs_index += d_ij * p_i * p_j * 2  # *2 because we only iterate upper triangle
    
    return rs_index


def classify_diversity_level(
    source_div: float,
    field_div: float,
    ref_count: int
) -> str:
    """Classify overall diversity level."""
    if ref_count < 5:
        return DiversityLevel.VERY_LOW.value
    
    # Weighted average of diversities
    avg_div = (source_div * 0.4 + field_div * 0.6)
    
    if avg_div >= 0.8:
        return DiversityLevel.VERY_HIGH.value
    elif avg_div >= 0.6:
        return DiversityLevel.HIGH.value
    elif avg_div >= 0.4:
        return DiversityLevel.MODERATE.value
    elif avg_div >= 0.2:
        return DiversityLevel.LOW.value
    else:
        return DiversityLevel.VERY_LOW.value


def _parse_pipe_separated(value) -> List[str]:
    """Parse pipe-separated string to list."""
    if pd.isna(value) or not value:
        return []
    
    if isinstance(value, list):
        return value
    
    return [x.strip() for x in str(value).split('|') if x.strip()]


def _extract_openalex_id(url: str) -> str:
    """Extract OpenAlex ID from URL."""
    if not url:
        return ""
    # https://openalex.org/W1234567 -> W1234567
    return url.split('/')[-1] if '/' in url else url


def fetch_reference_details_openalex(
    reference_ids: List[str],
    max_refs: int = 100,
    verbose: bool = False
) -> Dict[str, Dict]:
    """
    Fetch details for referenced works from OpenAlex API.
    
    Parameters
    ----------
    reference_ids : list
        List of OpenAlex work IDs (e.g., ['W1234', 'W5678'])
    max_refs : int
        Maximum references to fetch per paper
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        {work_id: {'year': int, 'source': str, 'field': str, 'topic': str, 'country': str}}
    """
    import requests
    import time
    
    if not reference_ids:
        return {}
    
    # Limit to max_refs
    ref_ids = reference_ids[:max_refs]
    
    # Extract just the IDs (remove URL prefix if present)
    clean_ids = [_extract_openalex_id(r) for r in ref_ids if r]
    clean_ids = [r for r in clean_ids if r.startswith('W')]
    
    if not clean_ids:
        return {}
    
    results = {}
    
    # Batch query (OpenAlex supports up to 50 IDs per request)
    batch_size = 50
    for i in range(0, len(clean_ids), batch_size):
        batch = clean_ids[i:i+batch_size]
        
        # Build filter for batch
        ids_filter = '|'.join(batch)
        url = f"https://api.openalex.org/works?filter=openalex_id:{ids_filter}&per-page={len(batch)}&select=id,publication_year,primary_location,primary_topic,authorships"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for work in data.get('results', []):
                    work_id = _extract_openalex_id(work.get('id', ''))
                    
                    # Extract fields
                    year = work.get('publication_year', 0)
                    
                    source = ""
                    loc = work.get('primary_location', {})
                    if loc and loc.get('source'):
                        source = loc['source'].get('display_name', '')
                    
                    field = ""
                    topic_name = ""
                    topic = work.get('primary_topic', {})
                    if topic:
                        topic_name = topic.get('display_name', '')
                        if topic.get('field'):
                            field = topic['field'].get('display_name', '')
                    
                    # Get country from first author's institution
                    country = ""
                    authorships = work.get('authorships', [])
                    if authorships:
                        for auth in authorships:
                            insts = auth.get('institutions', [])
                            if insts:
                                country = insts[0].get('country_code', '')
                                if country:
                                    break
                    
                    results[work_id] = {
                        'year': year,
                        'source': source,
                        'field': field,
                        'topic': topic_name,
                        'country': country
                    }
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            if verbose:
                print(f"  ⚠️ API error: {e}")
    
    return results


def analyze_paper_references(
    row: pd.Series,
    current_year: int,
    ref_details: Dict[str, Dict],
    ref_col: str = "referenced_works",
    author_col: str = "Authors",
    ref_authors_available: bool = False
) -> ReferenceDiversityMetrics:
    """
    Analyze reference diversity for a single paper.
    
    Parameters
    ----------
    row : pd.Series
        Paper data
    current_year : int
        Reference year for age calculations
    ref_details : dict
        Pre-fetched reference details from API
    ref_col : str
        Column name for referenced works
    author_col : str
        Column name for authors
    ref_authors_available : bool
        Whether reference author data is available for self-citation detection
        
    Returns
    -------
    ReferenceDiversityMetrics
    """
    metrics = ReferenceDiversityMetrics()
    
    # Get paper info
    metrics.doi = str(row.get('doi', row.get('DOI', '')))
    metrics.title = str(row.get('title', row.get('Title', '')))[:200]
    
    try:
        metrics.pub_year = int(row.get('publication_year', row.get('Year', 0)))
    except:
        metrics.pub_year = 0
    
    # Parse references
    refs = _parse_pipe_separated(row.get(ref_col, ''))
    metrics.reference_count = len(refs)
    
    if not refs:
        metrics.diversity_level = DiversityLevel.VERY_LOW.value
        return metrics
    
    # Collect distributions
    sources = {}
    fields = {}
    topics = {}
    countries = {}
    ref_years = []
    
    for ref in refs:
        ref_id = _extract_openalex_id(ref)
        
        if ref_id in ref_details:
            detail = ref_details[ref_id]
            
            # Year
            year = detail.get('year', 0)
            if year and year > 1800:
                ref_years.append(year)
            
            # Source
            source = detail.get('source', '')
            if source:
                sources[source] = sources.get(source, 0) + 1
            
            # Field
            field = detail.get('field', '')
            if field:
                fields[field] = fields.get(field, 0) + 1
            
            # Topic
            topic = detail.get('topic', '')
            if topic:
                topics[topic] = topics.get(topic, 0) + 1
            
            # Country
            country = detail.get('country', '')
            if country:
                countries[country] = countries.get(country, 0) + 1
    
    # Store distributions
    metrics.sources_distribution = sources
    metrics.fields_distribution = fields
    metrics.topics_distribution = topics
    metrics.ref_years = ref_years
    
    # Unique counts
    metrics.unique_sources = len(sources)
    metrics.unique_fields = len(fields)
    metrics.unique_topics = len(topics)
    metrics.unique_countries = len(countries)
    
    # Age metrics
    if ref_years:
        ages = [current_year - y for y in ref_years if y > 0]
        if ages:
            metrics.median_ref_age = float(np.median(ages))
            metrics.mean_ref_age = float(np.mean(ages))
            metrics.ref_age_std = float(np.std(ages))
            metrics.oldest_ref_age = max(ages)
            metrics.newest_ref_age = min(ages)
    
    # Diversity indices
    metrics.source_diversity = compute_shannon_entropy(sources)
    metrics.field_diversity = compute_shannon_entropy(fields)
    metrics.topic_diversity = compute_shannon_entropy(topics)
    
    # Rao-Stirling (simplified - uniform distance)
    metrics.rao_stirling_index = compute_rao_stirling(fields)
    
    # Classification
    metrics.diversity_level = classify_diversity_level(
        metrics.source_diversity,
        metrics.field_diversity,
        metrics.reference_count
    )
    
    return metrics


def analyze_reference_diversity(
    df: pd.DataFrame,
    use_openalex: bool = True,
    max_papers: int = 100,
    max_refs_per_paper: int = 50,
    current_year: int = None,
    verbose: bool = True,
    stop_flag: Callable[[], bool] = None
) -> ReferenceDiversityResult:
    """
    Analyze reference diversity for a bibliometric dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliometric dataset with referenced_works column
    use_openalex : bool
        Fetch reference details from OpenAlex API
    max_papers : int
        Maximum papers to analyze
    max_refs_per_paper : int
        Maximum references to fetch per paper
    current_year : int, optional
        Reference year for age calculations
    verbose : bool
        Print progress
    stop_flag : callable, optional
        Function that returns True if analysis should stop
        
    Returns
    -------
    ReferenceDiversityResult
    """
    import datetime
    
    if current_year is None:
        current_year = datetime.datetime.now().year
    
    def should_stop():
        return stop_flag is not None and stop_flag()
    
    # Find reference column
    ref_col = None
    for c in ["referenced_works", "References", "Cited References", "references"]:
        if c in df.columns:
            ref_col = c
            break
    
    if ref_col is None:
        if verbose:
            print("⚠️ No referenced_works column found")
        return ReferenceDiversityResult(
            metrics=[],
            n_papers=len(df),
            n_analyzed=0,
            data_source="none"
        )
    
    # Check how many papers have references
    has_refs = df[ref_col].notna() & (df[ref_col] != '')
    n_with_refs = has_refs.sum()
    
    if verbose:
        print(f"Analyzing reference diversity...")
        print(f"  Reference column: {ref_col}")
        print(f"  Papers with references: {n_with_refs}/{len(df)}")
    
    if n_with_refs == 0:
        return ReferenceDiversityResult(
            metrics=[],
            n_papers=len(df),
            n_analyzed=0,
            n_with_references=0,
            data_source="none"
        )
    
    # Limit to papers with references
    df_with_refs = df[has_refs].head(max_papers)
    
    if should_stop():
        return ReferenceDiversityResult(
            metrics=[], n_papers=len(df), n_analyzed=0
        )
    
    # Collect all reference IDs for batch fetching
    all_ref_ids = set()
    for _, row in df_with_refs.iterrows():
        refs = _parse_pipe_separated(row.get(ref_col, ''))
        for ref in refs[:max_refs_per_paper]:
            ref_id = _extract_openalex_id(ref)
            if ref_id:
                all_ref_ids.add(ref_id)
    
    if verbose:
        print(f"  Unique references to fetch: {len(all_ref_ids)}")
    
    # Fetch reference details from OpenAlex
    ref_details = {}
    api_calls = 0
    
    if use_openalex and all_ref_ids:
        if verbose:
            print(f"  Fetching reference details from OpenAlex...")
        
        # Batch fetch
        ref_list = list(all_ref_ids)
        batch_size = 50
        
        for i in range(0, len(ref_list), batch_size):
            if should_stop():
                if verbose:
                    print("  ⏹ Stopped by user")
                break
            
            batch = ref_list[i:i+batch_size]
            batch_details = fetch_reference_details_openalex(
                batch, max_refs=len(batch), verbose=verbose
            )
            ref_details.update(batch_details)
            api_calls += 1
            
            if verbose and (i + batch_size) % 200 == 0:
                print(f"    Fetched {min(i + batch_size, len(ref_list))}/{len(ref_list)} references...")
        
        if verbose:
            print(f"  ✓ Retrieved details for {len(ref_details)} references")
    
    # Analyze each paper
    metrics_list = []
    diversity_counts = {level.value: 0 for level in DiversityLevel}
    
    for idx, (_, row) in enumerate(df_with_refs.iterrows()):
        if should_stop():
            if verbose:
                print(f"  ⏹ Stopped at paper {idx}")
            break
        
        if idx % 50 == 0 and verbose:
            print(f"  Processing paper {idx + 1}/{len(df_with_refs)}...")
        
        metrics = analyze_paper_references(
            row, current_year, ref_details, ref_col
        )
        metrics_list.append(metrics)
        diversity_counts[metrics.diversity_level] = diversity_counts.get(metrics.diversity_level, 0) + 1
    
    # Compute aggregates
    n_analyzed = len(metrics_list)
    
    if n_analyzed > 0:
        avg_ref_count = np.mean([m.reference_count for m in metrics_list])
        avg_source_div = np.mean([m.source_diversity for m in metrics_list if m.reference_count > 0])
        avg_field_div = np.mean([m.field_diversity for m in metrics_list if m.reference_count > 0])
        avg_ref_age = np.mean([m.mean_ref_age for m in metrics_list if m.mean_ref_age > 0])
        avg_self_cite = np.mean([m.self_citation_rate for m in metrics_list])
    else:
        avg_ref_count = avg_source_div = avg_field_div = avg_ref_age = avg_self_cite = 0.0
    
    data_source = "openalex" if ref_details else "dataset"
    
    if verbose:
        print(f"  ✓ Analyzed {n_analyzed} documents")
        print(f"  Average references: {avg_ref_count:.1f}")
        print(f"  Average source diversity: {avg_source_div:.3f}")
        print(f"  Average field diversity: {avg_field_div:.3f}")
    
    return ReferenceDiversityResult(
        metrics=metrics_list,
        n_papers=len(df),
        n_analyzed=n_analyzed,
        n_with_references=n_with_refs,
        avg_reference_count=avg_ref_count,
        avg_source_diversity=avg_source_div,
        avg_field_diversity=avg_field_div,
        avg_ref_age=avg_ref_age,
        avg_self_citation_rate=avg_self_cite,
        diversity_distribution=diversity_counts,
        data_source=data_source,
        api_calls_made=api_calls
    )


# Convenience function for quick analysis
def quick_reference_diversity(df: pd.DataFrame, **kwargs) -> ReferenceDiversityResult:
    """Quick reference diversity analysis with sensible defaults."""
    return analyze_reference_diversity(
        df,
        use_openalex=True,
        max_papers=100,
        max_refs_per_paper=30,
        verbose=True,
        **kwargs
    )


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_diversity_distribution(
    result: ReferenceDiversityResult,
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
    colors: List[str] = None,
):
    """
    Plot distribution of diversity levels.
    
    Parameters
    ----------
    result : ReferenceDiversityResult
        Analysis result.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple
        Figure size if creating new figure.
    colors : list, optional
        Colors for bars.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    dist = result.diversity_distribution
    levels = list(dist.keys())
    counts = list(dist.values())
    
    if colors is None:
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
    
    bars = ax.bar(levels, counts, color=colors[:len(levels)], edgecolor='white')
    ax.set_xlabel("Diversity Level")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Distribution of Reference Diversity Levels")
    
    # Remove gridlines
    ax.grid(False)
    ax.set_axisbelow(True)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_source_diversity(
    result: ReferenceDiversityResult,
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = '#3498db',
    bins: int = 20,
):
    """
    Plot source diversity distribution.
    
    Parameters
    ----------
    result : ReferenceDiversityResult
        Analysis result.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.
    color : str
        Histogram color.
    bins : int
        Number of bins.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    diversities = [m.source_diversity for m in result.metrics if m.reference_count > 0]
    
    if diversities:
        ax.hist(diversities, bins=bins, color=color, edgecolor='white', alpha=0.8)
        mean_val = sum(diversities) / len(diversities)
        ax.axvline(mean_val, color='red', linestyle='--', 
                  label=f'Mean: {mean_val:.3f}')
        ax.set_xlabel("Source Diversity Index (Shannon)")
        ax.set_ylabel("Number of Papers")
        ax.set_title("Distribution of Source Diversity")
        ax.legend()
    
    # Remove gridlines
    ax.grid(False)
    
    plt.tight_layout()
    return fig


def plot_field_diversity(
    result: ReferenceDiversityResult,
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = '#9b59b6',
    bins: int = 20,
):
    """
    Plot field diversity distribution.
    
    Parameters
    ----------
    result : ReferenceDiversityResult
        Analysis result.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.
    color : str
        Histogram color.
    bins : int
        Number of bins.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    diversities = [m.field_diversity for m in result.metrics if m.reference_count > 0]
    
    if diversities:
        ax.hist(diversities, bins=bins, color=color, edgecolor='white', alpha=0.8)
        mean_val = sum(diversities) / len(diversities)
        ax.axvline(mean_val, color='red', linestyle='--',
                  label=f'Mean: {mean_val:.3f}')
        ax.set_xlabel("Field Diversity Index (Shannon)")
        ax.set_ylabel("Number of Papers")
        ax.set_title("Distribution of Field/Interdisciplinary Diversity")
        ax.legend()
    
    # Remove gridlines
    ax.grid(False)
    
    plt.tight_layout()
    return fig


def plot_reference_age_distribution(
    result: ReferenceDiversityResult,
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = '#1abc9c',
    bins: int = 30,
    current_year: int = None,
):
    """
    Plot reference age distribution.
    
    Parameters
    ----------
    result : ReferenceDiversityResult
        Analysis result.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.
    color : str
        Histogram color.
    bins : int
        Number of bins.
    current_year : int, optional
        Reference year for age calculation.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import datetime
    
    if current_year is None:
        current_year = datetime.datetime.now().year
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    ages = []
    for m in result.metrics:
        ages.extend(m.ref_years)
    
    if ages:
        ages_calc = [current_year - y for y in ages if y > 1800]
        
        if ages_calc:
            ax.hist(ages_calc, bins=bins, color=color, edgecolor='white', alpha=0.8)
            ax.set_xlabel("Reference Age (years)")
            ax.set_ylabel("Frequency")
            ax.set_title("Age Distribution of References")
            
            median_age = sorted(ages_calc)[len(ages_calc)//2]
            ax.axvline(median_age, color='red', linestyle='--',
                      label=f'Median: {median_age} years')
            ax.legend()
    
    # Remove gridlines
    ax.grid(False)
    
    plt.tight_layout()
    return fig


def plot_diversity_by_year(
    result: ReferenceDiversityResult,
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot diversity trends by publication year.
    
    Parameters
    ----------
    result : ReferenceDiversityResult
        Analysis result.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    year_data = {}
    for m in result.metrics:
        if m.pub_year and m.pub_year > 1900:
            if m.pub_year not in year_data:
                year_data[m.pub_year] = {'source': [], 'field': []}
            year_data[m.pub_year]['source'].append(m.source_diversity)
            year_data[m.pub_year]['field'].append(m.field_diversity)
    
    if year_data:
        years = sorted(year_data.keys())
        source_means = [sum(year_data[y]['source'])/len(year_data[y]['source']) 
                      if year_data[y]['source'] else 0 for y in years]
        field_means = [sum(year_data[y]['field'])/len(year_data[y]['field'])
                     if year_data[y]['field'] else 0 for y in years]
        
        ax.plot(years, source_means, 'o-', label='Source Diversity', color='#3498db')
        ax.plot(years, field_means, 's-', label='Field Diversity', color='#9b59b6')
        ax.set_xlabel("Publication Year")
        ax.set_ylabel("Average Diversity Index")
        ax.set_title("Reference Diversity Trends Over Time")
        ax.legend()
    
    # Remove gridlines
    ax.grid(False)
    
    plt.tight_layout()
    return fig


def get_top_diverse_papers(
    result: ReferenceDiversityResult,
    n: int = 15,
    sort_by: str = "combined",
) -> pd.DataFrame:
    """
    Get top most diverse papers as a DataFrame.
    
    Parameters
    ----------
    result : ReferenceDiversityResult
        Analysis result.
    n : int
        Number of top papers to return.
    sort_by : str
        Sort criterion: "combined", "source", "field", "topic"
        
    Returns
    -------
    pd.DataFrame
        DataFrame with top diverse papers.
    """
    if sort_by == "source":
        key_func = lambda m: m.source_diversity
    elif sort_by == "field":
        key_func = lambda m: m.field_diversity
    elif sort_by == "topic":
        key_func = lambda m: m.topic_diversity
    else:  # combined
        key_func = lambda m: m.source_diversity + m.field_diversity
    
    sorted_metrics = sorted(
        result.metrics,
        key=key_func,
        reverse=True
    )[:n]
    
    data = []
    for rank, m in enumerate(sorted_metrics, 1):
        data.append({
            'Rank': rank,
            'Title': m.title,
            'Year': m.pub_year,
            'References': m.reference_count,
            'Unique Sources': m.unique_sources,
            'Unique Fields': m.unique_fields,
            'Source Diversity': round(m.source_diversity, 3),
            'Field Diversity': round(m.field_diversity, 3),
            'Combined': round(m.source_diversity + m.field_diversity, 3),
            'Median Ref Age': round(m.median_ref_age, 1),
            'Level': m.diversity_level,
            'DOI': m.doi,
        })
    
    return pd.DataFrame(data)


def add_diversity_to_dataframe(
    df: pd.DataFrame,
    result: ReferenceDiversityResult,
    doi_col: str = None,
    title_col: str = None,
) -> pd.DataFrame:
    """
    Add reference diversity metrics to the original dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original bibliometric dataframe.
    result : ReferenceDiversityResult
        Analysis result from analyze_reference_diversity().
    doi_col : str, optional
        Name of DOI column. Auto-detected if not provided.
    title_col : str, optional
        Name of title column. Auto-detected if not provided.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new columns added:
        - Ref_Source_Diversity
        - Ref_Field_Diversity
        - Ref_Topic_Diversity
        - Ref_Rao_Stirling
        - Ref_Unique_Sources
        - Ref_Unique_Fields
        - Ref_Median_Age
        - Ref_Mean_Age
        - Ref_Diversity_Level
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Create a mapping from DOI/title to metrics
    metrics_data = {}
    for m in result.metrics:
        # Use DOI as primary key, fall back to title
        key = m.doi if m.doi else m.title
        if key:
            metrics_data[key] = {
                'Ref_Source_Diversity': round(m.source_diversity, 4),
                'Ref_Field_Diversity': round(m.field_diversity, 4),
                'Ref_Topic_Diversity': round(m.topic_diversity, 4),
                'Ref_Rao_Stirling': round(m.rao_stirling_index, 4),
                'Ref_Unique_Sources': m.unique_sources,
                'Ref_Unique_Fields': m.unique_fields,
                'Ref_Median_Age': round(m.median_ref_age, 1),
                'Ref_Mean_Age': round(m.mean_ref_age, 1),
                'Ref_Diversity_Level': m.diversity_level,
            }
    
    # Initialize new columns
    new_cols = [
        'Ref_Source_Diversity', 'Ref_Field_Diversity', 'Ref_Topic_Diversity',
        'Ref_Rao_Stirling', 'Ref_Unique_Sources', 'Ref_Unique_Fields',
        'Ref_Median_Age', 'Ref_Mean_Age', 'Ref_Diversity_Level'
    ]
    
    for col in new_cols:
        df[col] = None
    
    # Auto-detect DOI and title columns if not provided
    if doi_col is None:
        for c in df.columns:
            if c.lower() == 'doi':
                doi_col = c
                break
    
    if title_col is None:
        for c in df.columns:
            c_lower = c.lower()
            if c_lower in ['title', 'document title', 'ti']:
                title_col = c
                break
    
    # Map metrics to rows
    matched = 0
    for idx, row in df.iterrows():
        key = None
        if doi_col and pd.notna(row.get(doi_col)):
            key = str(row[doi_col])
        if key not in metrics_data and title_col and pd.notna(row.get(title_col)):
            key = str(row[title_col])
        
        if key and key in metrics_data:
            for col, val in metrics_data[key].items():
                df.at[idx, col] = val
            matched += 1
    
    print(f"  ✓ Added diversity metrics to {matched}/{len(df)} rows")
    
    return df
