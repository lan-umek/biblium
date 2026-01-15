# -*- coding: utf-8 -*-
"""
Comparative Analysis Tools Module for Bibliometric Analysis

This module provides tools for comparing bibliometric datasets, fields,
institutions, countries, time periods, and research groups.

Features implemented:
1. Dataset Comparison - Compare two or more bibliometric datasets
2. Field/Discipline Comparison - Cross-field analysis
3. Institutional Comparison - Compare universities, research centers
4. Country/Region Comparison - Geographic research output analysis
5. Temporal Comparison - Compare different time periods
6. Author Group Comparison - Compare research groups or cohorts
7. Journal Comparison - Compare publication venues
8. Benchmark Analysis - Compare against field averages
9. Similarity Analysis - Measure dataset overlap and similarity
10. Radar/Spider Charts - Multi-dimensional comparisons
11. Statistical Significance Testing - Rigorous group comparisons

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine, jaccard
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import seaborn as sns

# Optional imports
try:
    import statsmodels.api as sm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

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
class ComparisonMetrics:
    """Metrics for a single entity being compared."""
    name: str
    n_papers: int
    n_authors: int
    n_sources: int
    total_citations: int
    mean_citations: float
    median_citations: float
    h_index: int
    citations_per_paper: float
    authors_per_paper: float
    international_collab_rate: float
    open_access_rate: float
    year_range: Tuple[int, int]
    top_keywords: List[Tuple[str, int]]
    top_sources: List[Tuple[str, int]]
    top_authors: List[Tuple[str, int]]
    growth_rate: float
    additional_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class PairwiseComparison:
    """Results of comparing two entities."""
    entity1: str
    entity2: str
    metrics_comparison: pd.DataFrame
    statistical_tests: Dict[str, Dict[str, float]]
    keyword_overlap: float
    author_overlap: float
    source_overlap: float
    citation_ratio: float
    size_ratio: float
    similarity_score: float
    significant_differences: List[str]

@dataclass
class MultipleComparison:
    """Results of comparing multiple entities."""
    entities: List[str]
    metrics_df: pd.DataFrame
    normalized_metrics_df: pd.DataFrame
    rankings: pd.DataFrame
    statistical_tests: Dict[str, Dict[str, float]]
    pairwise_similarities: pd.DataFrame
    clusters: Dict[str, int]
    summary_statistics: Dict[str, pd.DataFrame]

@dataclass
class BenchmarkResult:
    """Results of benchmarking against reference."""
    entity_name: str
    benchmark_name: str
    metrics_comparison: pd.DataFrame
    percentile_ranks: Dict[str, float]
    above_benchmark: List[str]
    below_benchmark: List[str]
    relative_performance: Dict[str, float]
    overall_score: float

@dataclass
class ComparativeAnalysisResult:
    """Container for complete comparative analysis results."""
    comparison_type: str
    entities: List[str]
    entity_metrics: Dict[str, ComparisonMetrics]
    pairwise_comparisons: List[PairwiseComparison]
    multiple_comparison: Optional[MultipleComparison]
    benchmark_results: Optional[List[BenchmarkResult]]
    temporal_comparison: Optional[pd.DataFrame]
    parameters: Dict[str, Any]

# =============================================================================
# METRIC EXTRACTION
# =============================================================================

def extract_comparison_metrics(
    df: pd.DataFrame,
    name: str,
    authors_col: str = "Authors",
    year_col: str = "Year",
    citations_col: str = "Cited by",
    source_col: str = "Source title",
    keywords_col: str = "Author Keywords",
    affiliation_col: str = "Affiliations",
    oa_col: str = None,
    sep: str = "; ") -> ComparisonMetrics:
    """
    Extract comparison metrics from a dataset or subset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    name : str
        Name for this entity.
    Various column parameters.
    sep : str
        Separator for multi-valued fields.
    
    Returns
    -------
    ComparisonMetrics
    """
    n_papers = len(df)
    
    if n_papers == 0:
        return ComparisonMetrics(
            name=name, n_papers=0, n_authors=0, n_sources=0,
            total_citations=0, mean_citations=0, median_citations=0,
            h_index=0, citations_per_paper=0, authors_per_paper=0,
            international_collab_rate=0, open_access_rate=0,
            year_range=(0, 0), top_keywords=[], top_sources=[],
            top_authors=[], growth_rate=0)
    
    # Citations
    citations = df[citations_col].fillna(0).astype(int) if citations_col in df.columns else pd.Series([0] * n_papers)
    total_citations = citations.sum()
    mean_citations = citations.mean()
    median_citations = citations.median()
    
    # H-index
    sorted_cit = sorted(citations, reverse=True)
    h_index = sum(1 for i, c in enumerate(sorted_cit) if c >= i + 1)
    
    # Authors
    author_counts = Counter()
    authors_per_paper_list = []
    
    for _, row in df.iterrows():
        authors = row.get(authors_col, "")
        if pd.notna(authors) and authors:
            if isinstance(authors, str):
                author_list = [a.strip() for a in authors.split(sep) if a.strip()]
                authors_per_paper_list.append(len(author_list))
                for a in author_list:
                    author_counts[a] += 1
    
    n_authors = len(author_counts)
    authors_per_paper = np.mean(authors_per_paper_list) if authors_per_paper_list else 0
    
    # Sources
    source_counts = Counter()
    if source_col in df.columns:
        for source in df[source_col].dropna():
            source_counts[str(source).strip()] += 1
    n_sources = len(source_counts)
    
    # Keywords
    keyword_counts = Counter()
    if keywords_col in df.columns:
        for _, row in df.iterrows():
            kws = row.get(keywords_col, "")
            if pd.notna(kws) and kws:
                if isinstance(kws, str):
                    for kw in kws.split(sep):
                        kw = kw.strip().lower()
                        if kw:
                            keyword_counts[kw] += 1
    
    # Year range
    if year_col in df.columns:
        years = df[year_col].dropna().astype(int)
        year_range = (int(years.min()), int(years.max())) if len(years) > 0 else (0, 0)
        
        # Growth rate
        year_counts = years.value_counts().sort_index()
        if len(year_counts) >= 2:
            x = np.arange(len(year_counts))
            y = year_counts.values
            slope, _, _, _, _ = stats.linregress(x, y)
            growth_rate = slope / np.mean(y) if np.mean(y) > 0 else 0
        else:
            growth_rate = 0
    else:
        year_range = (0, 0)
        growth_rate = 0
    
    # International collaboration (detect multiple countries in affiliations)
    international_collab = 0
    if affiliation_col in df.columns:
        for _, row in df.iterrows():
            aff = row.get(affiliation_col, "")
            if pd.notna(aff) and aff:
                # Simple heuristic: count unique country patterns
                countries = set()
                for pattern in [r"\b[A-Z]{2}\b", r"USA|UK|China|Germany|France|Japan|Canada|Australia|India|Brazil"]:
                    matches = re.findall(pattern, str(aff))
                    countries.update(matches)
                if len(countries) > 1:
                    international_collab += 1
        international_collab_rate = international_collab / n_papers
    else:
        international_collab_rate = 0
    
    # Open access rate
    if oa_col and oa_col in df.columns:
        oa_count = df[oa_col].apply(lambda x: str(x).lower() in ["true", "yes", "1", "gold", "green", "hybrid", "bronze"]).sum()
        open_access_rate = oa_count / n_papers
    else:
        open_access_rate = 0
    
    return ComparisonMetrics(
        name=name,
        n_papers=n_papers,
        n_authors=n_authors,
        n_sources=n_sources,
        total_citations=int(total_citations),
        mean_citations=round(mean_citations, 2),
        median_citations=round(median_citations, 1),
        h_index=h_index,
        citations_per_paper=round(total_citations / n_papers, 2) if n_papers > 0 else 0,
        authors_per_paper=round(authors_per_paper, 2),
        international_collab_rate=round(international_collab_rate, 3),
        open_access_rate=round(open_access_rate, 3),
        year_range=year_range,
        top_keywords=keyword_counts.most_common(20),
        top_sources=source_counts.most_common(10),
        top_authors=author_counts.most_common(10),
        growth_rate=round(growth_rate, 4))

# =============================================================================
# PAIRWISE COMPARISON
# =============================================================================

def compare_two_entities(
    metrics1: ComparisonMetrics,
    metrics2: ComparisonMetrics,
    df1: pd.DataFrame = None,
    df2: pd.DataFrame = None,
    citations_col: str = "Cited by",
    alpha: float = 0.05) -> PairwiseComparison:
    """
    Compare two entities in detail.
    
    Parameters
    ----------
    metrics1, metrics2 : ComparisonMetrics
        Metrics for each entity.
    df1, df2 : pd.DataFrame, optional
        Original dataframes for statistical tests.
    citations_col : str
        Citations column.
    alpha : float
        Significance level.
    
    Returns
    -------
    PairwiseComparison
    """
    # Metrics comparison dataframe
    metrics_data = {
        "Metric": [
            "Papers", "Authors", "Sources", "Total Citations",
            "Mean Citations", "Median Citations", "H-index",
            "Authors/Paper", "International Collab %", "OA Rate %",
            "Growth Rate"
        ],
        metrics1.name: [
            metrics1.n_papers, metrics1.n_authors, metrics1.n_sources,
            metrics1.total_citations, metrics1.mean_citations,
            metrics1.median_citations, metrics1.h_index,
            metrics1.authors_per_paper, metrics1.international_collab_rate * 100,
            metrics1.open_access_rate * 100, metrics1.growth_rate
        ],
        metrics2.name: [
            metrics2.n_papers, metrics2.n_authors, metrics2.n_sources,
            metrics2.total_citations, metrics2.mean_citations,
            metrics2.median_citations, metrics2.h_index,
            metrics2.authors_per_paper, metrics2.international_collab_rate * 100,
            metrics2.open_access_rate * 100, metrics2.growth_rate
        ],
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df["Difference"] = metrics_df[metrics2.name] - metrics_df[metrics1.name]
    metrics_df["Ratio"] = metrics_df[metrics2.name] / (metrics_df[metrics1.name] + 1e-10)
    
    # Statistical tests
    statistical_tests = {}
    significant_differences = []
    
    if df1 is not None and df2 is not None and citations_col in df1.columns and citations_col in df2.columns:
        cit1 = df1[citations_col].fillna(0).astype(float)
        cit2 = df2[citations_col].fillna(0).astype(float)
        
        # Mann-Whitney U test (non-parametric)
        if len(cit1) >= 5 and len(cit2) >= 5:
            stat, p = stats.mannwhitneyu(cit1, cit2, alternative="two-sided")
            statistical_tests["mann_whitney"] = {"statistic": stat, "p_value": p}
            if p < alpha:
                significant_differences.append("citations (Mann-Whitney)")
        
        # T-test (parametric)
        if len(cit1) >= 5 and len(cit2) >= 5:
            stat, p = stats.ttest_ind(cit1, cit2)
            statistical_tests["t_test"] = {"statistic": stat, "p_value": p}
            if p < alpha:
                significant_differences.append("citations (t-test)")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((cit1.var() + cit2.var()) / 2)
        cohens_d = (cit1.mean() - cit2.mean()) / pooled_std if pooled_std > 0 else 0
        statistical_tests["effect_size"] = {"cohens_d": cohens_d}
    
    # Keyword overlap (Jaccard)
    kw1 = set(k for k, _ in metrics1.top_keywords)
    kw2 = set(k for k, _ in metrics2.top_keywords)
    keyword_overlap = len(kw1 & kw2) / len(kw1 | kw2) if kw1 | kw2 else 0
    
    # Author overlap
    auth1 = set(a for a, _ in metrics1.top_authors)
    auth2 = set(a for a, _ in metrics2.top_authors)
    author_overlap = len(auth1 & auth2) / len(auth1 | auth2) if auth1 | auth2 else 0
    
    # Source overlap
    src1 = set(s for s, _ in metrics1.top_sources)
    src2 = set(s for s, _ in metrics2.top_sources)
    source_overlap = len(src1 & src2) / len(src1 | src2) if src1 | src2 else 0
    
    # Ratios
    citation_ratio = metrics2.total_citations / (metrics1.total_citations + 1)
    size_ratio = metrics2.n_papers / (metrics1.n_papers + 1)
    
    # Overall similarity score
    similarity_score = (keyword_overlap + author_overlap + source_overlap) / 3
    
    return PairwiseComparison(
        entity1=metrics1.name,
        entity2=metrics2.name,
        metrics_comparison=metrics_df,
        statistical_tests=statistical_tests,
        keyword_overlap=round(keyword_overlap, 3),
        author_overlap=round(author_overlap, 3),
        source_overlap=round(source_overlap, 3),
        citation_ratio=round(citation_ratio, 3),
        size_ratio=round(size_ratio, 3),
        similarity_score=round(similarity_score, 3),
        significant_differences=significant_differences)

# =============================================================================
# MULTIPLE ENTITY COMPARISON
# =============================================================================

def compare_multiple_entities(
    metrics_list: List[ComparisonMetrics],
    perform_clustering: bool = True,
    n_clusters: int = 3) -> MultipleComparison:
    """
    Compare multiple entities simultaneously.
    
    Parameters
    ----------
    metrics_list : List[ComparisonMetrics]
        List of metrics for each entity.
    perform_clustering : bool
        Whether to cluster entities.
    n_clusters : int
        Number of clusters.
    
    Returns
    -------
    MultipleComparison
    """
    entities = [m.name for m in metrics_list]
    
    # Create metrics dataframe
    data = []
    for m in metrics_list:
        data.append({
            "Entity": m.name,
            "Papers": m.n_papers,
            "Authors": m.n_authors,
            "Sources": m.n_sources,
            "Total Citations": m.total_citations,
            "Mean Citations": m.mean_citations,
            "Median Citations": m.median_citations,
            "H_Index": m.h_index,
            "Authors_per_Paper": m.authors_per_paper,
            "Intl Collab Rate": m.international_collab_rate,
            "OA_Rate": m.open_access_rate,
            "Growth Rate": m.growth_rate,
        })
    
    metrics_df = pd.DataFrame(data)
    metrics_df = metrics_df.set_index("Entity")
    
    # Normalize metrics
    scaler = MinMaxScaler()
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    normalized_values = scaler.fit_transform(metrics_df[numeric_cols])
    normalized_df = pd.DataFrame(
        normalized_values,
        index=metrics_df.index,
        columns=numeric_cols
    )
    
    # Rankings
    rankings = metrics_df[numeric_cols].rank(ascending=False).astype(int)
    rankings.columns = [f"{c}_Rank" for c in rankings.columns]
    
    # Pairwise similarities
    similarity_matrix = cosine_similarity(normalized_df)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=entities,
        columns=entities
    )
    
    # Clustering
    clusters = {}
    if perform_clustering and len(metrics_list) >= n_clusters:
        try:
            Z = linkage(normalized_df, method="ward")
            cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")
            clusters = {entity: int(label) for entity, label in zip(entities, cluster_labels)}
        except:
            clusters = {entity: 1 for entity in entities}
    else:
        clusters = {entity: 1 for entity in entities}
    
    # Statistical tests (Kruskal-Wallis would need raw data)
    statistical_tests = {}
    
    # Summary statistics per metric
    summary_statistics = {}
    for col in numeric_cols:
        summary_statistics[col] = pd.DataFrame({
            "mean": [metrics_df[col].mean()],
            "std": [metrics_df[col].std()],
            "min": [metrics_df[col].min()],
            "max": [metrics_df[col].max()],
            "median": [metrics_df[col].median()],
        })
    
    return MultipleComparison(
        entities=entities,
        metrics_df=metrics_df.reset_index(),
        normalized_metrics_df=normalized_df.reset_index(),
        rankings=rankings.reset_index(),
        statistical_tests=statistical_tests,
        pairwise_similarities=similarity_df,
        clusters=clusters,
        summary_statistics=summary_statistics)

# =============================================================================
# BENCHMARK ANALYSIS
# =============================================================================

def benchmark_against_reference(
    entity_metrics: ComparisonMetrics,
    reference_metrics: ComparisonMetrics,
    reference_name: str = "Benchmark") -> BenchmarkResult:
    """
    Benchmark an entity against a reference.
    
    Parameters
    ----------
    entity_metrics : ComparisonMetrics
        Metrics for entity to benchmark.
    reference_metrics : ComparisonMetrics
        Reference/benchmark metrics.
    reference_name : str
        Name for the benchmark.
    
    Returns
    -------
    BenchmarkResult
    """
    # Metrics to compare
    metrics_to_compare = {
        "Mean Citations": (entity_metrics.mean_citations, reference_metrics.mean_citations),
        "H_Index": (entity_metrics.h_index, reference_metrics.h_index),
        "Authors_per_Paper": (entity_metrics.authors_per_paper, reference_metrics.authors_per_paper),
        "Intl Collab Rate": (entity_metrics.international_collab_rate, reference_metrics.international_collab_rate),
        "OA_Rate": (entity_metrics.open_access_rate, reference_metrics.open_access_rate),
        "Growth Rate": (entity_metrics.growth_rate, reference_metrics.growth_rate),
    }
    
    comparison_data = []
    above_benchmark = []
    below_benchmark = []
    relative_performance = {}
    
    for metric, (entity_val, ref_val) in metrics_to_compare.items():
        diff = entity_val - ref_val
        rel_perf = entity_val / ref_val if ref_val != 0 else 0
        
        comparison_data.append({
            "Metric": metric,
            "Entity": entity_val,
            "Benchmark": ref_val,
            "Difference": round(diff, 3),
            "Relative": round(rel_perf, 3),
        })
        
        relative_performance[metric] = rel_perf
        
        if entity_val > ref_val:
            above_benchmark.append(metric)
        elif entity_val < ref_val:
            below_benchmark.append(metric)
    
    metrics_df = pd.DataFrame(comparison_data)
    
    # Percentile ranks (simplified - would need distribution data for real percentiles)
    percentile_ranks = {
        metric: min(100, max(0, 50 + (entity_val - ref_val) / (ref_val + 1e-10) * 50))
        for metric, (entity_val, ref_val) in metrics_to_compare.items()
    }
    
    # Overall score (average relative performance)
    overall_score = np.mean(list(relative_performance.values()))
    
    return BenchmarkResult(
        entity_name=entity_metrics.name,
        benchmark_name=reference_name,
        metrics_comparison=metrics_df,
        percentile_ranks=percentile_ranks,
        above_benchmark=above_benchmark,
        below_benchmark=below_benchmark,
        relative_performance=relative_performance,
        overall_score=round(overall_score, 3))

# =============================================================================
# TEMPORAL COMPARISON
# =============================================================================

def compare_time_periods(
    df: pd.DataFrame,
    year_col: str = "Year",
    citations_col: str = "Cited by",
    authors_col: str = "Authors",
    keywords_col: str = "Author Keywords",
    periods: List[Tuple[int, int]] = None,
    sep: str = "; ",
    verbose: bool = True) -> pd.DataFrame:
    """
    Compare metrics across different time periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    year_col : str
        Year column.
    periods : List[Tuple[int, int]], optional
        List of (start_year, end_year) tuples. Auto-generated if None.
    verbose : bool
        Print progress.
    
    Returns
    -------
    DataFrame with period comparisons.
    """
    if verbose:
        print("Comparing time periods...")
    
    years = df[year_col].dropna().astype(int)
    min_year, max_year = years.min(), years.max()
    
    # Auto-generate periods if not provided
    if periods is None:
        span = max_year - min_year
        if span <= 10:
            period_length = 2
        elif span <= 20:
            period_length = 5
        else:
            period_length = 10
        
        periods = []
        start = min_year
        while start < max_year:
            end = min(start + period_length - 1, max_year)
            periods.append((start, end))
            start = end + 1
    
    # Calculate metrics for each period
    period_metrics = []
    
    for start, end in periods:
        period_df = df[(df[year_col] >= start) & (df[year_col] <= end)]
        
        if len(period_df) == 0:
            continue
        
        period_name = f"{start}-{end}"
        metrics = extract_comparison_metrics(
            period_df, period_name,
            authors_col=authors_col,
            year_col=year_col,
            citations_col=citations_col,
            keywords_col=keywords_col,
            sep=sep)
        
        period_metrics.append({
            "Period": period_name,
            "Start Year": start,
            "End Year": end,
            "Papers": metrics.n_papers,
            "Authors": metrics.n_authors,
            "Total Citations": metrics.total_citations,
            "Mean Citations": metrics.mean_citations,
            "H_Index": metrics.h_index,
            "Authors_per_Paper": metrics.authors_per_paper,
        })
    
    result_df = pd.DataFrame(period_metrics)
    
    # Calculate period-over-period changes
    if len(result_df) > 1:
        result_df["Papers_Change"] = result_df["Papers"].pct_change() * 100
        result_df["Citations_Change"] = result_df["Mean_Citations"].pct_change() * 100
    
    if verbose:
        print(f"  Compared {len(result_df)} time periods")
    
    return result_df

# =============================================================================
# GROUP COMPARISON BY COLUMN
# =============================================================================

def compare_by_column(
    df: pd.DataFrame,
    group_col: str,
    authors_col: str = "Authors",
    year_col: str = "Year",
    citations_col: str = "Cited by",
    source_col: str = "Source title",
    keywords_col: str = "Author Keywords",
    sep: str = "; ",
    top_n: int = 10,
    verbose: bool = True) -> ComparativeAnalysisResult:
    """
    Compare groups defined by a column (e.g., country, institution, field).
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    group_col : str
        Column to group by.
    top_n : int
        Number of top groups to compare.
    verbose : bool
        Print progress.
    
    Returns
    -------
    ComparativeAnalysisResult
    """
    if verbose:
        print(f"Comparing by '{group_col}'...")
    
    # Get group counts
    if sep in str(df[group_col].iloc[0]) if len(df) > 0 else False:
        # Multi-valued column
        group_counts = Counter()
        for _, row in df.iterrows():
            val = row.get(group_col, "")
            if pd.notna(val) and val:
                for item in str(val).split(sep):
                    item = item.strip()
                    if item:
                        group_counts[item] += 1
    else:
        group_counts = df[group_col].value_counts().to_dict()
    
    # Get top groups
    top_groups = [g for g, _ in sorted(group_counts.items(), key=lambda x: -x[1])[:top_n]]
    
    if verbose:
        print(f"  Found {len(group_counts)} groups, analyzing top {len(top_groups)}")
    
    # Extract metrics for each group
    entity_metrics = {}
    group_dfs = {}
    
    for group in top_groups:
        if sep in str(df[group_col].iloc[0]) if len(df) > 0 else False:
            # Filter for multi-valued column
            mask = df[group_col].apply(
                lambda x: group in str(x).split(sep) if pd.notna(x) else False
            )
        else:
            mask = df[group_col] == group
        
        group_df = df[mask]
        group_dfs[group] = group_df
        
        metrics = extract_comparison_metrics(
            group_df, group,
            authors_col=authors_col,
            year_col=year_col,
            citations_col=citations_col,
            source_col=source_col,
            keywords_col=keywords_col,
            sep=sep)
        entity_metrics[group] = metrics
    
    # Pairwise comparisons
    pairwise_comparisons = []
    for g1, g2 in combinations(top_groups, 2):
        comparison = compare_two_entities(
            entity_metrics[g1], entity_metrics[g2],
            group_dfs[g1], group_dfs[g2],
            citations_col=citations_col)
        pairwise_comparisons.append(comparison)
    
    # Multiple comparison
    multiple_comparison = compare_multiple_entities(list(entity_metrics.values()))
    
    if verbose:
        print(f"  Completed {len(pairwise_comparisons)} pairwise comparisons")
    
    return ComparativeAnalysisResult(
        comparison_type=f"by_{group_col}",
        entities=top_groups,
        entity_metrics=entity_metrics,
        pairwise_comparisons=pairwise_comparisons,
        multiple_comparison=multiple_comparison,
        benchmark_results=None,
        temporal_comparison=None,
        parameters={"group_col": group_col, "top_n": top_n})

# =============================================================================
# DATASET COMPARISON
# =============================================================================

def compare_datasets(
    datasets: Dict[str, pd.DataFrame],
    authors_col: str = "Authors",
    year_col: str = "Year",
    citations_col: str = "Cited by",
    source_col: str = "Source title",
    keywords_col: str = "Author Keywords",
    sep: str = "; ",
    verbose: bool = True) -> ComparativeAnalysisResult:
    """
    Compare multiple bibliometric datasets.
    
    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary mapping dataset names to dataframes.
    Various column parameters.
    verbose : bool
        Print progress.
    
    Returns
    -------
    ComparativeAnalysisResult
    """
    if verbose:
        print(f"Comparing {len(datasets)} datasets...")
    
    # Extract metrics for each dataset
    entity_metrics = {}
    
    for name, df in datasets.items():
        metrics = extract_comparison_metrics(
            df, name,
            authors_col=authors_col,
            year_col=year_col,
            citations_col=citations_col,
            source_col=source_col,
            keywords_col=keywords_col,
            sep=sep)
        entity_metrics[name] = metrics
        
        if verbose:
            print(f"  {name}: {metrics.n_papers} papers, {metrics.total_citations} citations")
    
    # Pairwise comparisons
    pairwise_comparisons = []
    dataset_names = list(datasets.keys())
    
    for n1, n2 in combinations(dataset_names, 2):
        comparison = compare_two_entities(
            entity_metrics[n1], entity_metrics[n2],
            datasets[n1], datasets[n2],
            citations_col=citations_col)
        pairwise_comparisons.append(comparison)
    
    # Multiple comparison
    multiple_comparison = compare_multiple_entities(list(entity_metrics.values()))
    
    return ComparativeAnalysisResult(
        comparison_type="datasets",
        entities=dataset_names,
        entity_metrics=entity_metrics,
        pairwise_comparisons=pairwise_comparisons,
        multiple_comparison=multiple_comparison,
        benchmark_results=None,
        temporal_comparison=None,
        parameters={"n_datasets": len(datasets)})

# =============================================================================
# SIMILARITY ANALYSIS
# =============================================================================

def calculate_dataset_similarity(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_col: str = "DOI",
    authors_col: str = "Authors",
    keywords_col: str = "Author Keywords",
    source_col: str = "Source title",
    sep: str = "; ") -> Dict[str, float]:
    """
    Calculate various similarity metrics between two datasets.
    
    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Dataframes to compare.
    Various column parameters.
    
    Returns
    -------
    Dict of similarity metrics.
    """
    similarities = {}
    
    # Document overlap (by DOI or ID)
    if id_col in df1.columns and id_col in df2.columns:
        ids1 = set(df1[id_col].dropna().astype(str))
        ids2 = set(df2[id_col].dropna().astype(str))
        if ids1 | ids2:
            similarities["document_overlap"] = len(ids1 & ids2) / len(ids1 | ids2)
        else:
            similarities["document_overlap"] = 0
    
    # Author overlap
    def get_authors(df, col, sep):
        authors = set()
        for val in df[col].dropna():
            if isinstance(val, str):
                for a in val.split(sep):
                    a = a.strip()
                    if a:
                        authors.add(a.lower())
        return authors
    
    if authors_col in df1.columns and authors_col in df2.columns:
        auth1 = get_authors(df1, authors_col, sep)
        auth2 = get_authors(df2, authors_col, sep)
        if auth1 | auth2:
            similarities["author_overlap"] = len(auth1 & auth2) / len(auth1 | auth2)
        else:
            similarities["author_overlap"] = 0
    
    # Keyword overlap
    def get_keywords(df, col, sep):
        keywords = set()
        for val in df[col].dropna():
            if isinstance(val, str):
                for k in val.split(sep):
                    k = k.strip().lower()
                    if k:
                        keywords.add(k)
        return keywords
    
    if keywords_col in df1.columns and keywords_col in df2.columns:
        kw1 = get_keywords(df1, keywords_col, sep)
        kw2 = get_keywords(df2, keywords_col, sep)
        if kw1 | kw2:
            similarities["keyword_overlap"] = len(kw1 & kw2) / len(kw1 | kw2)
        else:
            similarities["keyword_overlap"] = 0
    
    # Source overlap
    if source_col in df1.columns and source_col in df2.columns:
        src1 = set(df1[source_col].dropna().astype(str).str.lower())
        src2 = set(df2[source_col].dropna().astype(str).str.lower())
        if src1 | src2:
            similarities["source_overlap"] = len(src1 & src2) / len(src1 | src2)
        else:
            similarities["source_overlap"] = 0
    
    # Overall similarity
    valid_sims = [v for v in similarities.values() if v is not None]
    similarities["overall_similarity"] = np.mean(valid_sims) if valid_sims else 0
    
    return similarities

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_comparison_overview(
    result: ComparativeAnalysisResult,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot papers and citations comparison."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    # Papers and Citations comparison
    x = range(len(result.entities))
    width = 0.35
    
    papers = [result.entity_metrics[e].n_papers for e in result.entities]
    citations = [result.entity_metrics[e].total_citations / 100 for e in result.entities]
    
    ax.bar([i - width/2 for i in x], papers, width, label="Papers", color="lightblue")
    ax.bar([i + width/2 for i in x], citations, width, label="Citations (÷100)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([e[:20] for e in result.entities], rotation=45, ha="right")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Papers and Citations by Entity", fontsize=12)
    ax.legend()
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_overview.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_mean_citations_comparison(
    result: ComparativeAnalysisResult,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot mean citations comparison."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    mean_cit = [result.entity_metrics[e].mean_citations for e in result.entities]
    
    bars = ax.bar(range(len(result.entities)), mean_cit, color="lightblue")
    ax.set_xticks(range(len(result.entities)))
    ax.set_xticklabels([e[:20] for e in result.entities], rotation=45, ha="right")
    ax.set_ylabel("Mean Citations", fontsize=11)
    ax.set_title("Mean Citations per Paper", fontsize=12)
    
    for bar, val in zip(bars, mean_cit):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_mean_citations.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_normalized_metrics_heatmap(
    result: ComparativeAnalysisResult,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot normalized metrics heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    mc = result.multiple_comparison
    norm_df = mc.normalized_metrics_df.set_index("Entity")
    
    sns.heatmap(norm_df, annot=True, fmt=".2f", cmap=cmap or "viridis", ax=ax,
               cbar_kws={"shrink": 0.8})
    ax.set_title("Normalized Metrics Heatmap", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_normalized.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_similarity_matrix(
    result: ComparativeAnalysisResult,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot pairwise similarity matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    mc = result.multiple_comparison
    sim_df = mc.pairwise_similarities
    
    mask = np.triu(np.ones_like(sim_df, dtype=bool), k=1)
    sns.heatmap(sim_df, annot=True, fmt=".2f", cmap=cmap or "viridis", ax=ax,
               mask=mask, cbar_kws={"shrink": 0.8})
    ax.set_title("Pairwise Similarity Matrix", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_similarity.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_radar_comparison(
    result: ComparativeAnalysisResult,
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot radar/spider chart comparison."""
    if metrics is None:
        metrics = ["Papers", "Mean_Citations", "H_Index", "Authors_per_Paper", 
                  "Intl_Collab_Rate", "Growth_Rate"]
    
    mc = result.multiple_comparison
    norm_df = mc.normalized_metrics_df.set_index("Entity")
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in norm_df.columns]
    
    if len(available_metrics) < 3:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "Insufficient metrics for radar chart", ha="center", va="center")
        ax.axis("off")
        return fig
    
    # Setup radar chart
    n_metrics = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.grid(False)
    
    colors = "lightblue"
    
    for i, entity in enumerate(result.entities):
        values = norm_df.loc[entity, available_metrics].values.tolist()
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, "o-", linewidth=2, label=entity[:20], color="lightblue")
        ax.fill(angles, values, alpha=0.1, color="lightblue")
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    
    plt.title("Multi-dimensional Comparison", fontsize=14, fontweight="bold", y=1.1)
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_temporal_comparison(
    temporal_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot papers over time."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    ax.bar(temporal_df["Period"], temporal_df["Papers"], color="lightblue", edgecolor="white")
    ax.set_xlabel("Period", fontsize=11)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title("Publication Output by Period", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_temporal_citations(
    temporal_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot mean citations over time."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    ax.plot(temporal_df["Period"], temporal_df["Mean_Citations"], marker="o", 
            linewidth=2, color="lightblue", markersize=8)
    ax.set_xlabel("Period", fontsize=11)
    ax.set_ylabel("Mean Citations", fontsize=11)
    ax.set_title("Citation Impact by Period", fontsize=12)
    ax.set_xticks(range(len(temporal_df)))
    ax.set_xticklabels(temporal_df["Period"], rotation=45)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_temporal_hindex(
    temporal_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot H-index over time."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    ax.plot(temporal_df["Period"], temporal_df["H_Index"], marker="s",
            linewidth=2, color="lightblue", markersize=8)
    ax.set_xlabel("Period", fontsize=11)
    ax.set_ylabel("H-Index", fontsize=11)
    ax.set_title("H-Index by Period", fontsize=12)
    ax.set_xticks(range(len(temporal_df)))
    ax.set_xticklabels(temporal_df["Period"], rotation=45)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_benchmark_comparison(
    benchmark_result: BenchmarkResult,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot benchmark side-by-side comparison."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    df = benchmark_result.metrics_comparison
    
    # Side-by-side comparison
    x = range(len(df))
    width = 0.35
    
    ax.barh([i - width/2 for i in x], df["Entity"], width, 
            label=benchmark_result.entity_name, color="lightblue")
    ax.barh([i + width/2 for i in x], df["Benchmark"], width,
            label=benchmark_result.benchmark_name, color="steelblue")
    
    ax.set_yticks(x)
    ax.set_yticklabels(df["Metric"])
    ax.set_xlabel("Value", fontsize=11)
    ax.set_title("Entity vs Benchmark", fontsize=12)
    ax.legend()
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_relative_performance(
    benchmark_result: BenchmarkResult,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot relative performance vs benchmark."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    rel_perf = list(benchmark_result.relative_performance.values())
    metrics = list(benchmark_result.relative_performance.keys())
    
    bars = ax.barh(range(len(metrics)), rel_perf, color="lightblue")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Relative Performance (1 = benchmark)", fontsize=11)
    ax.set_title(f"Performance vs Benchmark (Overall: {benchmark_result.overall_score:.2f}x)", fontsize=12)
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_ranking_comparison(
    result: ComparativeAnalysisResult,
    top_metrics: int = 8,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot ranking comparison across entities."""
    mc = result.multiple_comparison
    rankings = mc.rankings.set_index("Entity")
    
    # Select top metrics
    metric_cols = rankings.columns[:top_metrics]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    x = np.arange(len(metric_cols))
    width = 0.8 / len(result.entities)
    colors = "lightblue"
    
    for i, entity in enumerate(result.entities):
        ranks = rankings.loc[entity, metric_cols].values
        offset = (i - len(result.entities) / 2 + 0.5) * width
        ax.bar(x + offset, ranks, width, label=entity[:20], color="lightblue")
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_Rank", "") for m in metric_cols], rotation=45, ha="right")
    ax.set_ylabel("Rank (1 = best)", fontsize=11)
    ax.set_title("Ranking Comparison Across Metrics", fontsize=12)
    ax.legend(loc="upper right")
    ax.invert_yaxis()  # Lower rank is better
    
    # Disable grids
    for _ax in axes.flat:
        _
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_comparison_results(
    result: ComparativeAnalysisResult,
    output_dir: str) -> Dict[str, str]:
    """Export comparative analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    excel_path = os.path.join(output_dir, "comparative_analysis.xlsx")
    
    with pd.ExcelWriter(excel_path) as writer:
        # Entity metrics
        metrics_data = []
        for name, m in result.entity_metrics.items():
            metrics_data.append({
                "Entity": name,
                "Papers": m.n_papers,
                "Authors": m.n_authors,
                "Sources": m.n_sources,
                "Total Citations": m.total_citations,
                "Mean Citations": m.mean_citations,
                "H_Index": m.h_index,
                "Authors_per_Paper": m.authors_per_paper,
                "Intl Collab Rate": m.international_collab_rate,
                "OA_Rate": m.open_access_rate,
                "Growth Rate": m.growth_rate,
            })
        pd.DataFrame(metrics_data).to_excel(writer, sheet_name="Entity_Metrics", index=False)
        
        # Multiple comparison
        if result.multiple_comparison:
            result.multiple_comparison.metrics_df.to_excel(
                writer, sheet_name="Metrics_Comparison", index=False
            )
            result.multiple_comparison.rankings.to_excel(
                writer, sheet_name="Rankings", index=False
            )
            result.multiple_comparison.pairwise_similarities.to_excel(
                writer, sheet_name="Similarities"
            )
        
        # Pairwise comparisons
        if result.pairwise_comparisons:
            pairwise_data = []
            for pc in result.pairwise_comparisons:
                pairwise_data.append({
                    "Entity1": pc.entity1,
                    "Entity2": pc.entity2,
                    "Keyword Overlap": pc.keyword_overlap,
                    "Author Overlap": pc.author_overlap,
                    "Source Overlap": pc.source_overlap,
                    "Citation Ratio": pc.citation_ratio,
                    "Similarity Score": pc.similarity_score,
                })
            pd.DataFrame(pairwise_data).to_excel(
                writer, sheet_name="Pairwise_Comparisons", index=False
            )
        
        # Temporal comparison
        if result.temporal_comparison is not None:
            result.temporal_comparison.to_excel(
                writer, sheet_name="Temporal_Comparison", index=False
            )
    
    print(f"Exported to: {excel_path}")
    return {"excel": excel_path}

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_comparative_analysis(
    df: pd.DataFrame,
    comparison_type: str = "field",
    group_col: str = None,
    authors_col: str = "Authors",
    year_col: str = "Year",
    citations_col: str = "Cited by",
    source_col: str = "Source title",
    keywords_col: str = "Author Keywords",
    sep: str = "; ",
    top_n: int = 10,
    include_temporal: bool = True,
    verbose: bool = True) -> ComparativeAnalysisResult:
    """
    Run comprehensive comparative analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    comparison_type : str
        Type of comparison: "field", "country", "institution", "source", "custom".
    group_col : str, optional
        Column to group by (required for "custom" type).
    Various column parameters.
    top_n : int
        Number of top entities to compare.
    include_temporal : bool
        Include temporal comparison.
    verbose : bool
        Print progress.
    
    Returns
    -------
    ComparativeAnalysisResult
    """
    if verbose:
        print("=" * 50)
        print("Comparative Analysis")
        print("=" * 50)
    
    # Determine grouping column
    column_map = {
        "field": "Subject Area",
        "country": "Country",
        "institution": "Affiliations",
        "source": source_col,
        "author": authors_col,
        "custom": group_col,
    }
    
    actual_group_col = column_map.get(comparison_type, group_col)
    
    if actual_group_col is None or actual_group_col not in df.columns:
        # Try to find a suitable column
        for col in ["Subject Area", "Country", "Affiliations", source_col]:
            if col in df.columns:
                actual_group_col = col
                break
    
    if actual_group_col is None or actual_group_col not in df.columns:
        raise ValueError(f"Column for comparison not found. Available: {df.columns.tolist()}")
    
    if verbose:
        print(f"  Comparison type: {comparison_type}")
        print(f"  Group column: {actual_group_col}")
    
    # Run comparison
    result = compare_by_column(
        df,
        actual_group_col,
        authors_col=authors_col,
        year_col=year_col,
        citations_col=citations_col,
        source_col=source_col,
        keywords_col=keywords_col,
        sep=sep,
        top_n=top_n,
        verbose=verbose)
    
    # Add temporal comparison
    if include_temporal and year_col in df.columns:
        if verbose:
            print("\n--- Temporal Comparison ---")
        result.temporal_comparison = compare_time_periods(
            df, year_col, citations_col, authors_col, keywords_col, sep=sep, verbose=verbose
        )
    
    result.comparison_type = comparison_type
    
    if verbose:
        print("\n" + "=" * 50)
        print("Analysis complete!")
    
    return result

# =============================================================================
# CLASS INTEGRATION
# =============================================================================

def add_comparative_methods(cls):
    """Add comparative analysis methods to BiblioAnalysis class."""
    
    def run_comparative_analysis_method(
        self,
        comparison_type: str = "source",
        group_col: str = None,
        top_n: int = 10,
        **kwargs
    ) -> ComparativeAnalysisResult:
        """Run comparative analysis."""
        self.comparative_results = run_comparative_analysis(
            self.df,
            comparison_type=comparison_type,
            group_col=group_col,
            authors_col="Authors",
            year_col="Year",
            citations_col="Cited by",
            source_col="Source title",
            keywords_col="Author Keywords",
            top_n=top_n,
            **kwargs
        )
        
        if hasattr(self, "res_folder") and self.res_folder:
            export_comparison_results(
                self.comparative_results,
                self.res_folder
            )
        
        return self.comparative_results
    
    def plot_comparative_method(
        self,
        plot_type: str = "overview",
        save: bool = True,
        **kwargs
    ) -> plt.Figure:
        """Create comparative visualizations."""
        if not hasattr(self, "comparative_results"):
            raise ValueError("Run run_comparative_analysis first")
        
        save_path = None
        if save and hasattr(self, "res_folder") and self.res_folder:
            save_path = os.path.join(self.res_folder, f"comparative_{plot_type}")
        
        result = self.comparative_results
        
        if plot_type == "overview":
            return plot_comparison_overview(result, save_path=save_path, **kwargs)
        elif plot_type == "radar":
            return plot_radar_comparison(result, save_path=save_path, **kwargs)
        elif plot_type == "temporal" and result.temporal_comparison is not None:
            return plot_temporal_comparison(result.temporal_comparison, save_path=save_path, **kwargs)
        elif plot_type == "ranking":
            return plot_ranking_comparison(result, save_path=save_path, **kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
    
    cls.run_comparative_analysis = run_comparative_analysis_method
    cls.plot_comparative = plot_comparative_method
    
    return cls
# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import biblium as bb
    ba = bb.BiblioAnalysis(f_name="data\\open alex dataset.csv", db="oa")
    
    print("Comparative Analysis")
    print("=" * 60)
    
    # Run analysis
    result = run_comparative_analysis(
        df=ba.df,
        comparison_type="source",
        authors_col="Authors",
        year_col="Year",
        citations_col="Cited by",
        source_col="Source title",
        top_n=10,
        verbose=True)
    
    # Print summary
    print(f"\nCompared {len(result.entities)} entities")
    
    # Visualizations
    print("\nGenerating plots...")
    plot_comparison_overview(result, save_path="results/comparison_overview")
    plot_radar_comparison(result, save_path="results/radar_comparison")
    
    # Export
    print("\nExporting results...")
    export_comparison_results(result, "results")
    
    print("\nDone!")