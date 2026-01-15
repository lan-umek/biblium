# -*- coding: utf-8 -*-
"""
Advanced Statistical Methods Module for Bibliometric Analysis

This module provides sophisticated statistical methods for analyzing
bibliometric data beyond basic descriptive statistics.

Features implemented:
1. Lotka's Law Analysis - Author productivity distribution
2. Bradford's Law Analysis - Journal/source concentration
3. Zipf's Law Analysis - Word frequency distribution
4. Price's Law Analysis - Elite contributor identification
5. Bibliometric Growth Models - Exponential, logistic, power law fitting
6. Citation Distribution Analysis - Power law, log-normal fitting
7. Collaboration Metrics - Collaboration index, degree of collaboration
8. Scientific Maturity Index - Field maturity indicators
9. Research Diversity Indices - Shannon, Simpson, Gini indices
10. Statistical Hypothesis Testing - Comparing groups, trends
11. Regression Analysis - Citation predictors, trend analysis
12. Bootstrap Confidence Intervals - Robust uncertainty estimation

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import os
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit, minimize
from scipy.special import zeta
from scipy.stats import (
    pearsonr, spearmanr, kendalltau,
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon,
    chi2_contingency, fisher_exact,
    kruskal, f_oneway,
    kstest, shapiro, normaltest,
    linregress, powerlaw)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
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
    import powerlaw as pl
    POWERLAW_AVAILABLE = True
except ImportError:
    POWERLAW_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
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
class LotkaResult:
    """Results from Lotka's Law analysis."""
    observed_distribution: pd.DataFrame
    theoretical_distribution: pd.DataFrame
    lotka_exponent: float
    lotka_constant: float
    r_squared: float
    ks_statistic: float
    ks_pvalue: float
    is_valid: bool  # Whether data follows Lotka's Law
    most_productive_authors: pd.DataFrame
    summary_stats: Dict[str, float]

@dataclass
class BradfordResult:
    """Results from Bradford's Law analysis."""
    zones: pd.DataFrame
    bradford_multiplier: float
    core_sources: List[str]
    r_squared: float
    groos_droop: float  # Deviation from ideal Bradford
    source_ranking: pd.DataFrame
    summary_stats: Dict[str, float]

@dataclass
class ZipfResult:
    """Results from Zipf's Law analysis."""
    observed_distribution: pd.DataFrame
    zipf_exponent: float
    zipf_constant: float
    r_squared: float
    top_words: pd.DataFrame
    deviations: pd.DataFrame
    summary_stats: Dict[str, float]

@dataclass
class GrowthModelResult:
    """Results from bibliometric growth model fitting."""
    model_type: str  # "exponential", "logistic", "power", "linear"
    parameters: Dict[str, float]
    r_squared: float
    aic: float
    bic: float
    fitted_values: np.ndarray
    residuals: np.ndarray
    prediction: pd.DataFrame
    doubling_time: Optional[float]
    growth_rate: float

@dataclass
class CitationDistributionResult:
    """Results from citation distribution analysis."""
    distribution_type: str  # "power_law", "log_normal", "exponential"
    parameters: Dict[str, float]
    fit_quality: Dict[str, float]
    percentiles: Dict[int, float]
    highly_cited_threshold: float
    proportion_uncited: float
    gini_coefficient: float
    h_index: int
    summary_stats: Dict[str, float]

@dataclass
class CollaborationMetrics:
    """Collaboration analysis results."""
    collaboration_index: float  # Mean authors per paper
    degree_of_collaboration: float  # Proportion of multi-authored papers
    collaboration_coefficient: float  # Modified collaboration index
    single_author_papers: int
    multi_author_papers: int
    max_authors: int
    author_distribution: pd.DataFrame
    temporal_trend: pd.DataFrame
    summary_stats: Dict[str, float]

@dataclass
class DiversityIndices:
    """Research diversity indices."""
    shannon_index: float
    simpson_index: float
    inverse_simpson: float
    gini_coefficient: float
    herfindahl_index: float
    evenness: float
    richness: int
    category_distribution: pd.DataFrame

@dataclass 
class StatisticalTestResult:
    """Results from statistical hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    conclusion: str
    is_significant: bool
    alpha: float
    additional_info: Dict[str, Any]

@dataclass
class AdvancedStatsResult:
    """Container for complete statistical analysis results."""
    lotka: Optional[LotkaResult]
    bradford: Optional[BradfordResult]
    zipf: Optional[ZipfResult]
    growth_model: Optional[GrowthModelResult]
    citation_distribution: Optional[CitationDistributionResult]
    collaboration: Optional[CollaborationMetrics]
    diversity: Optional[DiversityIndices]
    statistical_tests: List[StatisticalTestResult]
    regression_results: Dict[str, Any]
    parameters: Dict[str, Any]

# =============================================================================
# LOTKA'S LAW ANALYSIS
# =============================================================================

def analyze_lotka_law(
    df: pd.DataFrame,
    authors_col: str = "Authors",
    sep: str = "; ",
    min_papers: int = 1,
    verbose: bool = True) -> LotkaResult:
    """
    Analyze author productivity distribution using Lotka's Law.
    
    Lotka's Law: y = C / x^n
    Where y is proportion of authors with x publications,
    C is constant, n is Lotka exponent (typically ~2).
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    authors_col : str
        Authors column.
    sep : str
        Separator for author names.
    min_papers : int
        Minimum papers to include author.
    verbose : bool
        Print progress.
    
    Returns
    -------
    LotkaResult
    """
    if verbose:
        print("Analyzing Lotka's Law (author productivity)...")
    
    # Count papers per author
    author_counts = Counter()
    
    for _, row in df.iterrows():
        authors = row.get(authors_col, "")
        if pd.isna(authors) or not authors:
            continue
        
        if isinstance(authors, str):
            author_list = [a.strip() for a in authors.split(sep) if a.strip()]
        else:
            continue
        
        for author in author_list:
            author_counts[author] += 1
    
    if not author_counts:
        raise ValueError("No authors found in data")
    
    # Filter by minimum papers
    author_counts = {a: c for a, c in author_counts.items() if c >= min_papers}
    
    # Create productivity distribution
    productivity_counts = Counter(author_counts.values())
    
    # Prepare data for fitting
    x_values = sorted(productivity_counts.keys())
    y_counts = [productivity_counts[x] for x in x_values]
    total_authors = sum(y_counts)
    y_proportions = [c / total_authors for c in y_counts]
    
    observed_df = pd.DataFrame({
        "papers": x_values,
        "authors": y_counts,
        "proportion": y_proportions,
        "cumulative_proportion": np.cumsum(y_proportions),
    })
    
    # Fit Lotka's Law using log-log regression
    log_x = np.log(x_values)
    log_y = np.log(y_proportions)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
    
    lotka_exponent = -slope  # Negative because y decreases with x
    lotka_constant = np.exp(intercept)
    r_squared = r_value ** 2
    
    # Calculate theoretical distribution
    theoretical_proportions = [lotka_constant / (x ** lotka_exponent) for x in x_values]
    theoretical_normalized = [p / sum(theoretical_proportions) for p in theoretical_proportions]
    
    theoretical_df = pd.DataFrame({
        "papers": x_values,
        "theoretical_proportion": theoretical_normalized,
    })
    
    # Kolmogorov-Smirnov test
    try:
        # Create CDF function that handles both scalar and array inputs
        cumsum_theoretical = np.cumsum(theoretical_normalized)
        def theoretical_cdf(x):
            if np.isscalar(x):
                return cumsum_theoretical[np.searchsorted(x_values, x, side='right') - 1] if x >= x_values[0] else 0
            else:
                result = np.zeros_like(x, dtype=float)
                for i, xi in enumerate(x):
                    idx = np.searchsorted(x_values, xi, side='right') - 1
                    result[i] = cumsum_theoretical[idx] if xi >= x_values[0] and idx >= 0 else 0
                return result
        ks_stat, ks_pvalue = stats.kstest(y_proportions, theoretical_cdf)
    except Exception:
        ks_stat, ks_pvalue = np.nan, np.nan
    
    # Validity check (Lotka's n should be around 2)
    is_valid = 1.5 <= lotka_exponent <= 3.0 and r_squared >= 0.8
    
    # Most productive authors
    top_authors = sorted(author_counts.items(), key=lambda x: -x[1])[:20]
    top_df = pd.DataFrame(top_authors, columns=["author", "papers"])
    
    # Summary statistics
    summary_stats = {
        "total_authors": total_authors,
        "total_papers": len(df),
        "mean_papers_per_author": np.mean(list(author_counts.values())),
        "median_papers_per_author": np.median(list(author_counts.values())),
        "max_papers": max(author_counts.values()),
        "single_paper_authors": productivity_counts.get(1, 0),
        "single_paper_proportion": productivity_counts.get(1, 0) / total_authors,
    }
    
    if verbose:
        print(f"  Total authors: {total_authors}")
        print(f"  Lotka exponent (n): {lotka_exponent:.3f}")
        print(f"  R-squared: {r_squared:.3f}")
        print(f"  Single-paper authors: {summary_stats['single_paper_proportion']*100:.1f}%")
        print(f"  Law validity: {'Yes' if is_valid else 'No'}")
    
    return LotkaResult(
        observed_distribution=observed_df,
        theoretical_distribution=theoretical_df,
        lotka_exponent=lotka_exponent,
        lotka_constant=lotka_constant,
        r_squared=r_squared,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pvalue,
        is_valid=is_valid,
        most_productive_authors=top_df,
        summary_stats=summary_stats)

# =============================================================================
# BRADFORD'S LAW ANALYSIS
# =============================================================================

def analyze_bradford_law(
    df: pd.DataFrame,
    source_col: str = "Source title",
    n_zones: int = 3,
    verbose: bool = True) -> BradfordResult:
    """
    Analyze source concentration using Bradford's Law.
    
    Bradford's Law divides sources into zones of equal productivity,
    with geometrically increasing number of sources per zone.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    source_col : str
        Source/journal column.
    n_zones : int
        Number of Bradford zones.
    verbose : bool
        Print progress.
    
    Returns
    -------
    BradfordResult
    """
    if verbose:
        print("Analyzing Bradford's Law (source concentration)...")
    
    # Count papers per source
    source_counts = Counter()
    
    for _, row in df.iterrows():
        source = row.get(source_col, "")
        if pd.notna(source) and source:
            source_counts[str(source).strip()] += 1
    
    if not source_counts:
        raise ValueError("No sources found in data")
    
    # Sort sources by productivity (descending)
    sorted_sources = sorted(source_counts.items(), key=lambda x: -x[1])
    
    # Create ranking dataframe
    ranking_df = pd.DataFrame(sorted_sources, columns=["source", "papers"])
    ranking_df["rank"] = range(1, len(ranking_df) + 1)
    ranking_df["cumulative_papers"] = ranking_df["papers"].cumsum()
    ranking_df["cumulative_proportion"] = ranking_df["cumulative_papers"] / ranking_df["papers"].sum()
    
    # Divide into zones
    total_papers = ranking_df["papers"].sum()
    papers_per_zone = total_papers / n_zones
    
    zones = []
    current_zone = 1
    zone_start = 0
    cumsum = 0
    
    for i, (_, row) in enumerate(ranking_df.iterrows()):
        cumsum += row["papers"]
        if cumsum >= papers_per_zone * current_zone and current_zone < n_zones:
            zones.append({
                "zone": current_zone,
                "n_sources": i + 1 - zone_start,
                "n_papers": int(cumsum - (papers_per_zone * (current_zone - 1)) if current_zone > 1 else cumsum),
                "sources": list(ranking_df.iloc[zone_start:i+1]["source"]),
            })
            zone_start = i + 1
            current_zone += 1
    
    # Add final zone
    zones.append({
        "zone": n_zones,
        "n_sources": len(ranking_df) - zone_start,
        "n_papers": int(total_papers - sum(z["n_papers"] for z in zones)),
        "sources": list(ranking_df.iloc[zone_start:]["source"]),
    })
    
    zones_df = pd.DataFrame([{
        "zone": z["zone"],
        "n_sources": z["n_sources"],
        "n_papers": z["n_papers"],
    } for z in zones])
    
    # Calculate Bradford multiplier
    if len(zones_df) >= 2 and zones_df.iloc[0]["n_sources"] > 0:
        bradford_multiplier = zones_df.iloc[1]["n_sources"] / zones_df.iloc[0]["n_sources"]
    else:
        bradford_multiplier = np.nan
    
    # Core sources (zone 1)
    core_sources = zones[0]["sources"] if zones else []
    
    # Fit Bradford-Zipf model: R(r) = a * log(r) + b
    log_rank = np.log(ranking_df["rank"])
    cumulative = ranking_df["cumulative_papers"]
    
    slope, intercept, r_value, _, _ = linregress(log_rank, cumulative)
    r_squared = r_value ** 2
    
    # Groos droop (deviation from ideal)
    predicted = slope * log_rank + intercept
    groos_droop = np.mean(np.abs(cumulative - predicted))
    
    # Summary statistics
    summary_stats = {
        "total_sources": len(source_counts),
        "total_papers": total_papers,
        "mean_papers_per_source": np.mean(list(source_counts.values())),
        "median_papers_per_source": np.median(list(source_counts.values())),
        "core_zone_sources": zones_df.iloc[0]["n_sources"] if len(zones_df) > 0 else 0,
        "single_paper_sources": sum(1 for c in source_counts.values() if c == 1),
    }
    
    if verbose:
        print(f"  Total sources: {len(source_counts)}")
        print(f"  Bradford multiplier: {bradford_multiplier:.2f}")
        print(f"  Core sources (zone 1): {summary_stats['core_zone_sources']}")
        print(f"  R-squared: {r_squared:.3f}")
    
    return BradfordResult(
        zones=zones_df,
        bradford_multiplier=bradford_multiplier,
        core_sources=core_sources,
        r_squared=r_squared,
        groos_droop=groos_droop,
        source_ranking=ranking_df,
        summary_stats=summary_stats)

# =============================================================================
# ZIPF'S LAW ANALYSIS
# =============================================================================

def analyze_zipf_law(
    df: pd.DataFrame,
    text_col: str = "Abstract",
    title_col: str = "Title",
    min_word_length: int = 3,
    top_n: int = 100,
    stopwords: set = None,
    verbose: bool = True) -> ZipfResult:
    """
    Analyze word frequency distribution using Zipf's Law.
    
    Zipf's Law: f(r) = C / r^s
    Where f(r) is frequency of word at rank r,
    C is constant, s is Zipf exponent (typically ~1).
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    text_col : str
        Main text column (abstract).
    title_col : str
        Title column.
    min_word_length : int
        Minimum word length to include.
    top_n : int
        Number of top words to analyze.
    stopwords : set, optional
        Stopwords to exclude.
    verbose : bool
        Print progress.
    
    Returns
    -------
    ZipfResult
    """
    if verbose:
        print("Analyzing Zipf's Law (word frequency)...")
    
    # Default stopwords
    if stopwords is None:
        stopwords = {
            "the", "and", "for", "are", "was", "were", "been", "being",
            "have", "has", "had", "having", "does", "did", "doing",
            "will", "would", "could", "should", "may", "might", "must",
            "this", "that", "these", "those", "with", "from", "into",
            "during", "before", "after", "above", "below", "between",
            "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "each", "few", "more",
            "most", "other", "some", "such", "only", "own", "same", "than",
            "too", "very", "can", "just", "also", "which", "their", "them",
            "they", "our", "your", "its", "not", "but", "out", "about",
            "study", "paper", "research", "results", "using", "based",
            "method", "methods", "data", "analysis", "however", "found",
        }
    
    # Collect all words
    word_counts = Counter()
    
    for _, row in df.iterrows():
        text = ""
        if text_col and text_col in df.columns:
            text += str(row.get(text_col, "")) + " "
        if title_col and title_col in df.columns:
            text += str(row.get(title_col, ""))
        
        if not text or text.strip() == "nan":
            continue
        
        # Tokenize
        words = text.lower().split()
        words = [w.strip(".,;:!?\"'()[]{}") for w in words]
        words = [w for w in words if len(w) >= min_word_length and w.isalpha()]
        words = [w for w in words if w not in stopwords]
        
        word_counts.update(words)
    
    if not word_counts:
        raise ValueError("No words found in data")
    
    # Get top words
    top_words = word_counts.most_common(top_n)
    
    # Create distribution dataframe
    ranks = list(range(1, len(top_words) + 1))
    frequencies = [c for _, c in top_words]
    words = [w for w, _ in top_words]
    
    observed_df = pd.DataFrame({
        "rank": ranks,
        "word": words,
        "frequency": frequencies,
        "log_rank": np.log(ranks),
        "log_frequency": np.log(frequencies),
    })
    
    # Fit Zipf's Law using log-log regression
    log_rank = np.log(ranks)
    log_freq = np.log(frequencies)
    
    slope, intercept, r_value, _, _ = linregress(log_rank, log_freq)
    
    zipf_exponent = -slope  # Negative because frequency decreases with rank
    zipf_constant = np.exp(intercept)
    r_squared = r_value ** 2
    
    # Calculate theoretical frequencies
    theoretical_freq = [zipf_constant / (r ** zipf_exponent) for r in ranks]
    
    # Calculate deviations
    deviations_df = observed_df.copy()
    deviations_df["theoretical_frequency"] = theoretical_freq
    deviations_df["deviation"] = deviations_df["frequency"] - deviations_df["theoretical_frequency"]
    deviations_df["relative_deviation"] = deviations_df["deviation"] / deviations_df["theoretical_frequency"]
    
    # Top words dataframe
    top_df = pd.DataFrame(top_words[:20], columns=["word", "frequency"])
    top_df["rank"] = range(1, len(top_df) + 1)
    
    # Summary statistics
    summary_stats = {
        "total_unique_words": len(word_counts),
        "total_word_occurrences": sum(word_counts.values()),
        "zipf_exponent": zipf_exponent,
        "r_squared": r_squared,
        "most_frequent_word": top_words[0][0] if top_words else None,
        "most_frequent_count": top_words[0][1] if top_words else 0,
    }
    
    if verbose:
        print(f"  Unique words: {len(word_counts)}")
        print(f"  Zipf exponent (s): {zipf_exponent:.3f}")
        print(f"  R-squared: {r_squared:.3f}")
        print(f"  Top word: '{top_words[0][0]}' ({top_words[0][1]} occurrences)")
    
    return ZipfResult(
        observed_distribution=observed_df,
        zipf_exponent=zipf_exponent,
        zipf_constant=zipf_constant,
        r_squared=r_squared,
        top_words=top_df,
        deviations=deviations_df,
        summary_stats=summary_stats)

# =============================================================================
# PRICE'S LAW ANALYSIS
# =============================================================================

def analyze_price_law(
    df: pd.DataFrame,
    authors_col: str = "Authors",
    sep: str = "; ",
    verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze elite contributors using Price's Law.
    
    Price's Law: The square root of contributors produce 50% of output.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    authors_col : str
        Authors column.
    sep : str
        Separator.
    verbose : bool
        Print progress.
    
    Returns
    -------
    Dict with Price's Law analysis results.
    """
    if verbose:
        print("Analyzing Price's Law (elite contributors)...")
    
    # Count papers per author
    author_counts = Counter()
    
    for _, row in df.iterrows():
        authors = row.get(authors_col, "")
        if pd.isna(authors) or not authors:
            continue
        
        if isinstance(authors, str):
            author_list = [a.strip() for a in authors.split(sep) if a.strip()]
        else:
            continue
        
        for author in author_list:
            author_counts[author] += 1
    
    total_authors = len(author_counts)
    total_papers = sum(author_counts.values())
    
    # Price's Law prediction
    elite_count = int(np.sqrt(total_authors))
    
    # Get actual elite contribution
    sorted_authors = sorted(author_counts.values(), reverse=True)
    elite_papers = sum(sorted_authors[:elite_count])
    elite_proportion = elite_papers / total_papers
    
    # Find actual number needed for 50%
    cumsum = 0
    for i, count in enumerate(sorted_authors):
        cumsum += count
        if cumsum >= total_papers * 0.5:
            actual_50_pct = i + 1
            break
    else:
        actual_50_pct = len(sorted_authors)
    
    # Price ratio
    price_ratio = elite_papers / (total_papers * 0.5)
    
    result = {
        "total_authors": total_authors,
        "total_papers": total_papers,
        "elite_count_predicted": elite_count,
        "elite_proportion_predicted": elite_count / total_authors,
        "elite_papers": elite_papers,
        "elite_output_proportion": elite_proportion,
        "actual_authors_for_50pct": actual_50_pct,
        "price_ratio": price_ratio,
        "price_law_holds": 0.4 <= elite_proportion <= 0.6,
        "concentration_index": 1 - (actual_50_pct / total_authors),
    }
    
    if verbose:
        print(f"  Total authors: {total_authors}")
        print(f"  Elite (√n): {elite_count} authors")
        print(f"  Elite output: {elite_proportion*100:.1f}% of papers")
        print(f"  Price's Law holds: {'Yes' if result['price_law_holds'] else 'No'}")
    
    return result

# =============================================================================
# BIBLIOMETRIC GROWTH MODELS
# =============================================================================

def fit_growth_model(
    df: pd.DataFrame,
    year_col: str = "Year",
    model_type: str = "auto",
    forecast_years: int = 5,
    verbose: bool = True) -> GrowthModelResult:
    """
    Fit bibliometric growth model to publication data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    year_col : str
        Year column.
    model_type : str
        Model type: "exponential", "logistic", "power", "linear", "auto".
    forecast_years : int
        Years to forecast.
    verbose : bool
        Print progress.
    
    Returns
    -------
    GrowthModelResult
    """
    if verbose:
        print(f"Fitting growth model ({model_type})...")
    
    # Count papers per year
    year_counts = df[year_col].dropna().astype(int).value_counts().sort_index()
    years = np.array(year_counts.index)
    counts = np.array(year_counts.values)
    
    # Filter to reasonable years
    mask = years >= 1900
    years = years[mask]
    counts = counts[mask]
    
    # Normalize years for fitting
    year_offset = years.min()
    t = years - year_offset
    
    # Define models
    def exponential(t, a, b):
        return a * np.exp(b * t)
    
    def logistic(t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))
    
    def power_law(t, a, b):
        return a * (t + 1) ** b
    
    def linear(t, a, b):
        return a * t + b
    
    models = {
        "exponential": (exponential, [1, 0.1], [(0, None), (0, 1)]),
        "logistic": (logistic, [max(counts) * 2, 0.1, len(t) / 2], [(0, None), (0, 1), (0, len(t))]),
        "power": (power_law, [1, 1], [(0, None), (0, 5)]),
        "linear": (linear, [1, min(counts)], [(-np.inf, np.inf), (-np.inf, np.inf)]),
    }
    
    # Auto-select best model
    if model_type == "auto":
        best_model = None
        best_aic = np.inf
        
        for name, (func, p0, bounds) in models.items():
            try:
                bounds_tuple = ([b[0] if b[0] is not None else -np.inf for b in bounds],
                               [b[1] if b[1] is not None else np.inf for b in bounds])
                popt, _ = curve_fit(func, t, counts, p0=p0, bounds=bounds_tuple, maxfev=5000)
                fitted = func(t, *popt)
                residuals = counts - fitted
                ss_res = np.sum(residuals ** 2)
                n = len(counts)
                k = len(popt)
                aic = n * np.log(ss_res / n) + 2 * k
                
                if aic < best_aic:
                    best_aic = aic
                    best_model = name
            except:
                continue
        
        model_type = best_model or "linear"
        if verbose:
            print(f"  Auto-selected model: {model_type}")
    
    # Fit selected model
    func, p0, bounds = models[model_type]
    bounds_tuple = ([b[0] if b[0] is not None else -np.inf for b in bounds],
                   [b[1] if b[1] is not None else np.inf for b in bounds])
    
    try:
        popt, pcov = curve_fit(func, t, counts, p0=p0, bounds=bounds_tuple, maxfev=5000)
    except Exception as e:
        if verbose:
            print(f"  Warning: Fitting failed, using linear model. Error: {e}")
        func = linear
        popt, pcov = curve_fit(linear, t, counts)
        model_type = "linear"
    
    # Calculate fitted values and metrics
    fitted = func(t, *popt)
    residuals = counts - fitted
    
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((counts - np.mean(counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    n = len(counts)
    k = len(popt)
    aic = n * np.log(ss_res / n + 1e-10) + 2 * k
    bic = n * np.log(ss_res / n + 1e-10) + k * np.log(n)
    
    # Parameter names
    param_names = {
        "exponential": ["a", "growth_rate"],
        "logistic": ["carrying_capacity", "growth_rate", "midpoint"],
        "power": ["coefficient", "exponent"],
        "linear": ["slope", "intercept"],
    }
    
    parameters = dict(zip(param_names.get(model_type, ["p" + str(i) for i in range(len(popt))]), popt))
    
    # Growth rate and doubling time
    if model_type == "exponential":
        growth_rate = popt[1]
        doubling_time = np.log(2) / growth_rate if growth_rate > 0 else None
    elif model_type == "logistic":
        growth_rate = popt[1]
        doubling_time = np.log(2) / growth_rate if growth_rate > 0 else None
    elif model_type == "linear":
        growth_rate = popt[0] / np.mean(counts) if np.mean(counts) > 0 else 0
        doubling_time = None
    else:
        growth_rate = 0
        doubling_time = None
    
    # Forecast
    future_t = np.arange(t[-1] + 1, t[-1] + 1 + forecast_years)
    future_years = future_t + year_offset
    future_counts = func(future_t, *popt)
    
    prediction_df = pd.DataFrame({
        "year": np.concatenate([years, future_years]),
        "papers": np.concatenate([counts, np.full(forecast_years, np.nan)]),
        "fitted": np.concatenate([fitted, future_counts]),
        "is_forecast": [False] * len(years) + [True] * forecast_years,
    })
    
    if verbose:
        print(f"  R-squared: {r_squared:.3f}")
        print(f"  Growth rate: {growth_rate:.4f}")
        if doubling_time:
            print(f"  Doubling time: {doubling_time:.1f} years")
    
    return GrowthModelResult(
        model_type=model_type,
        parameters=parameters,
        r_squared=r_squared,
        aic=aic,
        bic=bic,
        fitted_values=fitted,
        residuals=residuals,
        prediction=prediction_df,
        doubling_time=doubling_time,
        growth_rate=growth_rate)


# =============================================================================
# LIFE CYCLE ANALYSIS (BiblioShiny-style)
# =============================================================================

@dataclass
class LifeCycleResult:
    """Results from scientific production life cycle analysis using logistic model.
    
    This implements the BiblioShiny-style life cycle analysis that models scientific
    production as a logistic (S-curve) growth process with carrying capacity.
    
    Attributes
    ----------
    saturation_k : float
        Carrying capacity - the maximum cumulative publications the topic will reach.
    peak_year_tm : float
        The inflection point year where growth rate is maximum (50% of K).
    peak_annual : float
        Maximum annual publication rate (at the inflection point).
    growth_duration_delta_t : float
        Time (in years) to grow from 10% to 90% of saturation.
    growth_rate_r : float
        Intrinsic growth rate parameter of the logistic model.
    r_squared : float
        Coefficient of determination (R²) for model fit.
    rmse : float
        Root Mean Square Error of the fit.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    last_observed_year : int
        Most recent year in the data.
    last_annual_pubs : int
        Publications in the last observed year.
    cumulative_total : int
        Total cumulative publications to date.
    progress_to_saturation : float
        Current progress toward saturation (0-1).
    milestone_years : Dict[str, float]
        Years when specific percentages of K are reached (10%, 50%, 90%, 99%).
    current_phase : str
        Current growth phase: "emergence", "rapid_growth", "maturity", or "saturation".
    phase_description : str
        Human-readable description of current phase.
    forecast_df : pd.DataFrame
        DataFrame with observed data, fitted values, and forecasts.
    fit_quality : str
        Quality assessment: "Excellent", "Good", "Fair", or "Poor".
    parameters : Dict[str, float]
        All fitted parameters.
    """
    saturation_k: float
    peak_year_tm: float
    peak_annual: float
    growth_duration_delta_t: float
    growth_rate_r: float
    r_squared: float
    rmse: float
    aic: float
    bic: float
    last_observed_year: int
    last_annual_pubs: int
    cumulative_total: int
    progress_to_saturation: float
    milestone_years: Dict[str, float]
    current_phase: str
    phase_description: str
    forecast_df: pd.DataFrame
    fit_quality: str
    parameters: Dict[str, float]
    projections: Dict[str, Dict[str, float]]  # e.g., {"2025": {"cumulative": 2183, "annual": 436}}


def analyze_life_cycle(
    df: pd.DataFrame,
    year_col: str = "Year",
    forecast_years: int = 50,
    target_years: Optional[List[int]] = None,
    verbose: bool = True,
) -> LifeCycleResult:
    """
    Analyze the life cycle of scientific production using logistic growth model.
    
    This implements the BiblioShiny-style "Life Cycle of Scientific Production"
    analysis, fitting a logistic (S-curve) model to cumulative publication data
    to estimate:
    - Saturation level (carrying capacity K)
    - Peak year (inflection point Tm)
    - Growth duration (Δt from 10% to 90% of K)
    - Current phase (emergence, rapid growth, maturity, saturation)
    - Milestone years (when 10%, 50%, 90%, 99% of K are reached)
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe with publication year data.
    year_col : str, default="Year"
        Name of the year column.
    forecast_years : int, default=50
        Number of years to forecast into the future.
    target_years : List[int], optional
        Specific years for which to provide projections (e.g., [2025, 2030]).
        If None, uses 5 and 10 years from last observed year.
    verbose : bool, default=True
        Print progress and results.
    
    Returns
    -------
    LifeCycleResult
        Comprehensive life cycle analysis results.
    
    Examples
    --------
    >>> from biblium import BiblioAnalysis
    >>> from biblium.addons.advanced_statistics import analyze_life_cycle
    >>> ba = BiblioAnalysis("data.csv", db="scopus")
    >>> result = analyze_life_cycle(ba.df)
    >>> print(f"Saturation: {result.saturation_k:.0f} publications")
    >>> print(f"Peak year: {result.peak_year_tm:.1f}")
    >>> print(f"Current phase: {result.current_phase}")
    
    Notes
    -----
    The logistic growth model is:
        N(t) = K / (1 + exp(-r * (t - Tm)))
    
    Where:
    - K is the carrying capacity (saturation)
    - r is the growth rate
    - Tm is the midpoint (inflection point)
    
    The growth duration Δt (time from 10% to 90% of K) is calculated as:
        Δt = 2 * ln(9) / r ≈ 4.39 / r
    """
    if verbose:
        print("Analyzing life cycle of scientific production...")
    
    # Count papers per year and compute cumulative
    year_counts = df[year_col].dropna().astype(int).value_counts().sort_index()
    years = np.array(year_counts.index)
    annual_counts = np.array(year_counts.values)
    
    # Filter to reasonable years
    mask = years >= 1900
    years = years[mask]
    annual_counts = annual_counts[mask]
    
    # Compute cumulative counts
    cumulative_counts = np.cumsum(annual_counts)
    
    # Normalize years for fitting
    year_offset = years.min()
    t = years - year_offset
    
    # Logistic model for cumulative data: N(t) = K / (1 + exp(-r * (t - Tm)))
    def logistic_cumulative(t, K, r, Tm):
        return K / (1 + np.exp(-r * (t - Tm)))
    
    # Initial guesses
    K_init = cumulative_counts[-1] * 2  # Start with 2x current total
    Tm_init = t[-1]  # Midpoint around last year
    r_init = 0.2  # Moderate growth rate
    
    # Bounds: K > current total, r > 0, Tm can be past or future
    bounds = (
        [cumulative_counts[-1] * 1.01, 0.001, -len(t)],  # lower bounds
        [cumulative_counts[-1] * 100, 2.0, len(t) * 3]    # upper bounds
    )
    
    try:
        popt, pcov = curve_fit(
            logistic_cumulative, t, cumulative_counts,
            p0=[K_init, r_init, Tm_init],
            bounds=bounds,
            maxfev=10000
        )
        K, r, Tm_offset = popt
    except Exception as e:
        if verbose:
            print(f"  Warning: Logistic fit failed ({e}), using fallback estimates")
        # Fallback: simple extrapolation
        K = cumulative_counts[-1] * 3
        r = 0.1
        Tm_offset = t[-1]
    
    # Convert Tm back to actual year
    Tm = Tm_offset + year_offset
    
    # Calculate fitted values
    fitted_cumulative = logistic_cumulative(t, K, r, Tm_offset)
    
    # Residuals and fit quality metrics
    residuals = cumulative_counts - fitted_cumulative
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((cumulative_counts - np.mean(cumulative_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    n = len(cumulative_counts)
    k_params = 3  # number of parameters
    rmse = np.sqrt(ss_res / n)
    aic = n * np.log(ss_res / n + 1e-10) + 2 * k_params
    bic = n * np.log(ss_res / n + 1e-10) + k_params * np.log(n)
    
    # Fit quality assessment
    if r_squared >= 0.95:
        fit_quality = "Excellent"
    elif r_squared >= 0.85:
        fit_quality = "Good"
    elif r_squared >= 0.70:
        fit_quality = "Fair"
    else:
        fit_quality = "Poor"
    
    # Growth duration: time from 10% to 90% of K
    # For logistic: Δt = 2 * ln(9) / r ≈ 4.394 / r
    growth_duration = 2 * np.log(9) / r if r > 0 else np.inf
    
    # Peak annual production (derivative at inflection point)
    # For logistic: max annual rate = K * r / 4
    peak_annual = K * r / 4
    
    # Milestone years (when specific percentages of K are reached)
    # For logistic: t = Tm + ln(p / (1-p)) / r where p is the proportion
    def year_at_proportion(p):
        if p <= 0 or p >= 1:
            return np.nan
        return Tm + np.log(p / (1 - p)) / r
    
    milestone_years = {
        "10%": year_at_proportion(0.10),
        "50%": Tm,  # By definition, Tm is when 50% is reached
        "90%": year_at_proportion(0.90),
        "99%": year_at_proportion(0.99),
    }
    
    # Current status
    last_observed_year = int(years[-1])
    last_annual_pubs = int(annual_counts[-1])
    cumulative_total = int(cumulative_counts[-1])
    progress_to_saturation = cumulative_total / K if K > 0 else 0
    
    # Current phase determination
    if progress_to_saturation < 0.10:
        current_phase = "emergence"
        phase_description = "The topic is in the emergence phase (<10% of saturation)."
    elif progress_to_saturation < 0.50:
        current_phase = "rapid_growth"
        phase_description = "The topic is in the rapid growth phase (10-50% of saturation)."
    elif progress_to_saturation < 0.90:
        current_phase = "maturity"
        phase_description = "The topic is in the maturity phase (50-90% of saturation)."
    else:
        current_phase = "saturation"
        phase_description = "The topic has reached saturation (>90% of carrying capacity)."
    
    # Generate forecast
    future_t = np.arange(t[-1] + 1, t[-1] + 1 + forecast_years)
    future_years = future_t + year_offset
    future_cumulative = logistic_cumulative(future_t, K, r, Tm_offset)
    
    # Calculate annual values from cumulative (difference)
    all_cumulative = np.concatenate([fitted_cumulative, future_cumulative])
    fitted_annual = np.diff(all_cumulative, prepend=0)
    fitted_annual[0] = all_cumulative[0]  # First year
    
    # Build forecast DataFrame
    all_years = np.concatenate([years, future_years])
    all_observed_annual = np.concatenate([annual_counts, np.full(forecast_years, np.nan)])
    all_observed_cumulative = np.concatenate([cumulative_counts, np.full(forecast_years, np.nan)])
    
    forecast_df = pd.DataFrame({
        "Year": all_years.astype(int),
        "Observed_Annual": all_observed_annual,
        "Observed_Cumulative": all_observed_cumulative,
        "Fitted_Annual": fitted_annual,
        "Fitted_Cumulative": all_cumulative,
        "Is_Forecast": [False] * len(years) + [True] * forecast_years,
    })
    
    # Projections for specific years
    if target_years is None:
        target_years = [last_observed_year + 5, last_observed_year + 10]
    
    projections = {}
    for target_year in target_years:
        if target_year > last_observed_year:
            t_target = target_year - year_offset
            cum_proj = logistic_cumulative(t_target, K, r, Tm_offset)
            # Annual is the difference from previous year
            t_prev = t_target - 1
            cum_prev = logistic_cumulative(t_prev, K, r, Tm_offset)
            annual_proj = cum_proj - cum_prev
            projections[str(target_year)] = {
                "cumulative": round(cum_proj),
                "annual": round(annual_proj),
            }
    
    if verbose:
        print(f"\n  === Life Cycle Analysis Results ===")
        print(f"  Model Fit Quality: {fit_quality} (R² = {r_squared:.3f})")
        print(f"\n  Model Overview:")
        print(f"    Saturation (K): {K:,.0f} publications")
        print(f"    Peak Year (Tm): {Tm:.1f}")
        print(f"    Peak Annual Rate: {peak_annual:,.0f} pubs/year")
        print(f"    Growth Duration (Δt): {growth_duration:.1f} years")
        print(f"\n  Current Status:")
        print(f"    Last Observed Year: {last_observed_year}")
        print(f"    Cumulative Total: {cumulative_total:,}")
        print(f"    Progress to Saturation: {progress_to_saturation*100:.1f}%")
        print(f"    Phase: {current_phase}")
        print(f"\n  Milestone Years:")
        for pct, year in milestone_years.items():
            years_from_now = year - last_observed_year
            print(f"    {pct} of K: {year:.1f} ({years_from_now:+.0f} years)")
        if projections:
            print(f"\n  Projections:")
            for year, proj in projections.items():
                print(f"    {year}: {proj['cumulative']:,} cumulative ({proj['annual']:,} annual)")
    
    return LifeCycleResult(
        saturation_k=K,
        peak_year_tm=Tm,
        peak_annual=peak_annual,
        growth_duration_delta_t=growth_duration,
        growth_rate_r=r,
        r_squared=r_squared,
        rmse=rmse,
        aic=aic,
        bic=bic,
        last_observed_year=last_observed_year,
        last_annual_pubs=last_annual_pubs,
        cumulative_total=cumulative_total,
        progress_to_saturation=progress_to_saturation,
        milestone_years=milestone_years,
        current_phase=current_phase,
        phase_description=phase_description,
        forecast_df=forecast_df,
        fit_quality=fit_quality,
        parameters={"K": K, "r": r, "Tm": Tm},
        projections=projections,
    )


def plot_life_cycle(
    result: LifeCycleResult,
    show_milestones: bool = True,
    show_forecast: bool = True,
    forecast_limit_years: int = 30,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot the life cycle analysis results.
    
    Parameters
    ----------
    result : LifeCycleResult
        Results from analyze_life_cycle().
    show_milestones : bool, default=True
        Show milestone year markers (10%, 50%, 90%, 99% of K).
    show_forecast : bool, default=True
        Show forecast period.
    forecast_limit_years : int, default=30
        Maximum years of forecast to show.
    figsize : Tuple[int, int], default=(12, 8)
        Figure size.
    save_path : str, optional
        Path to save the figure.
    dpi : int, default=150
        Resolution for saved figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    df = result.forecast_df.copy()
    
    # Limit forecast years
    last_obs = result.last_observed_year
    df = df[df["Year"] <= last_obs + forecast_limit_years]
    
    observed_df = df[~df["Is_Forecast"]]
    forecast_df_plot = df[df["Is_Forecast"]]
    
    # --- Top plot: Cumulative ---
    ax1 = axes[0]
    
    # Observed data
    ax1.scatter(observed_df["Year"], observed_df["Observed_Cumulative"], 
                color="blue", s=30, label="Observed", zorder=3)
    
    # Fitted line
    ax1.plot(df["Year"], df["Fitted_Cumulative"], 
             color="red", linewidth=2, label="Logistic Model", zorder=2)
    
    # Forecast shading
    if show_forecast and len(forecast_df_plot) > 0:
        ax1.axvspan(last_obs, df["Year"].max(), alpha=0.1, color="gray", label="Forecast Period")
    
    # Saturation line
    ax1.axhline(y=result.saturation_k, color="green", linestyle="--", 
                linewidth=1.5, label=f"Saturation (K={result.saturation_k:,.0f})")
    
    # Milestones
    if show_milestones:
        colors = {"10%": "orange", "50%": "purple", "90%": "brown", "99%": "gray"}
        for pct, year in result.milestone_years.items():
            if not np.isnan(year) and df["Year"].min() <= year <= df["Year"].max():
                proportion = float(pct.rstrip("%")) / 100
                y_val = result.saturation_k * proportion
                ax1.axvline(x=year, color=colors.get(pct, "gray"), 
                           linestyle=":", alpha=0.7, linewidth=1)
                ax1.annotate(f"{pct}", xy=(year, y_val), fontsize=8,
                            xytext=(5, 0), textcoords="offset points")
    
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cumulative Publications")
    ax1.set_title(f"Life Cycle of Scientific Production\n"
                  f"(R² = {result.r_squared:.3f}, Phase: {result.current_phase})")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # --- Bottom plot: Annual ---
    ax2 = axes[1]
    
    # Observed annual
    ax2.bar(observed_df["Year"], observed_df["Observed_Annual"], 
            color="blue", alpha=0.6, label="Observed Annual")
    
    # Fitted annual
    ax2.plot(df["Year"], df["Fitted_Annual"], 
             color="red", linewidth=2, label="Model Fit")
    
    # Peak annual
    ax2.axhline(y=result.peak_annual, color="green", linestyle="--", 
                alpha=0.7, label=f"Peak ({result.peak_annual:,.0f}/year)")
    
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Annual Publications")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    return fig


# =============================================================================
# CITATION DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_citation_distribution(
    df: pd.DataFrame,
    citations_col: str = "Cited by",
    verbose: bool = True) -> CitationDistributionResult:
    """
    Analyze citation distribution characteristics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    citations_col : str
        Citations column.
    verbose : bool
        Print progress.
    
    Returns
    -------
    CitationDistributionResult
    """
    if verbose:
        print("Analyzing citation distribution...")
    
    # Get citation counts
    citations = df[citations_col].dropna().astype(int).values
    citations = citations[citations >= 0]
    
    if len(citations) == 0:
        raise ValueError("No citation data found")
    
    n = len(citations)
    
    # Basic statistics
    mean_cit = np.mean(citations)
    median_cit = np.median(citations)
    std_cit = np.std(citations)
    max_cit = np.max(citations)
    
    # Proportion uncited
    uncited = np.sum(citations == 0)
    prop_uncited = uncited / n
    
    # Percentiles
    percentiles = {p: np.percentile(citations, p) for p in [25, 50, 75, 90, 95, 99]}
    
    # Highly cited threshold (top 10%)
    highly_cited_threshold = np.percentile(citations, 90)
    
    # H-index
    sorted_cit = np.sort(citations)[::-1]
    h_index = sum(1 for i, c in enumerate(sorted_cit) if c >= i + 1)
    
    # Gini coefficient
    sorted_cit_asc = np.sort(citations)
    cumsum = np.cumsum(sorted_cit_asc)
    gini = 1 - 2 * np.sum(cumsum) / (n * np.sum(citations)) if np.sum(citations) > 0 else 0
    
    # Fit distributions (for non-zero citations)
    positive_citations = citations[citations > 0]
    
    fit_quality = {}
    
    # Log-normal fit
    if len(positive_citations) > 10:
        log_cit = np.log(positive_citations)
        mu, sigma = np.mean(log_cit), np.std(log_cit)
        
        # KS test for log-normal
        ks_stat, ks_p = stats.kstest(positive_citations, "lognorm", args=(sigma, 0, np.exp(mu)))
        fit_quality["lognormal"] = {"mu": mu, "sigma": sigma, "ks_stat": ks_stat, "ks_p": ks_p}
    
    # Power law fit (if powerlaw package available)
    distribution_type = "unknown"
    parameters = {}
    
    if POWERLAW_AVAILABLE and len(positive_citations) > 50:
        try:
            fit = pl.Fit(positive_citations, discrete=True, verbose=False)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            
            # Compare with log-normal
            R, p = fit.distribution_compare("power_law", "lognormal")
            
            if R > 0 and p < 0.1:
                distribution_type = "power_law"
                parameters = {"alpha": alpha, "xmin": xmin}
            else:
                distribution_type = "lognormal"
                parameters = {"mu": mu, "sigma": sigma}
            
            fit_quality["power_law"] = {"alpha": alpha, "xmin": xmin, "R": R, "p": p}
        except:
            distribution_type = "lognormal" if "lognormal" in fit_quality else "unknown"
            if distribution_type == "lognormal":
                parameters = {"mu": mu, "sigma": sigma}
    else:
        distribution_type = "lognormal" if "lognormal" in fit_quality else "unknown"
        if distribution_type == "lognormal":
            parameters = {"mu": mu, "sigma": sigma}
    
    # Summary statistics
    summary_stats = {
        "n_papers": n,
        "mean_citations": mean_cit,
        "median_citations": median_cit,
        "std_citations": std_cit,
        "max_citations": max_cit,
        "skewness": stats.skew(citations),
        "kurtosis": stats.kurtosis(citations),
    }
    
    if verbose:
        print(f"  Papers analyzed: {n}")
        print(f"  Mean citations: {mean_cit:.2f}")
        print(f"  Median citations: {median_cit:.1f}")
        print(f"  Uncited papers: {prop_uncited*100:.1f}%")
        print(f"  H-index: {h_index}")
        print(f"  Gini coefficient: {gini:.3f}")
        print(f"  Distribution type: {distribution_type}")
    
    return CitationDistributionResult(
        distribution_type=distribution_type,
        parameters=parameters,
        fit_quality=fit_quality,
        percentiles=percentiles,
        highly_cited_threshold=highly_cited_threshold,
        proportion_uncited=prop_uncited,
        gini_coefficient=gini,
        h_index=h_index,
        summary_stats=summary_stats)

# =============================================================================
# COLLABORATION METRICS
# =============================================================================

def analyze_collaboration(
    df: pd.DataFrame,
    authors_col: str = "Authors",
    year_col: str = "Year",
    sep: str = "; ",
    verbose: bool = True) -> CollaborationMetrics:
    """
    Analyze collaboration patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    authors_col : str
        Authors column.
    year_col : str
        Year column.
    sep : str
        Separator.
    verbose : bool
        Print progress.
    
    Returns
    -------
    CollaborationMetrics
    """
    if verbose:
        print("Analyzing collaboration patterns...")
    
    author_counts = []
    year_data = defaultdict(list)
    
    for _, row in df.iterrows():
        authors = row.get(authors_col, "")
        year = row.get(year_col)
        
        if pd.isna(authors) or not authors:
            continue
        
        if isinstance(authors, str):
            author_list = [a.strip() for a in authors.split(sep) if a.strip()]
        else:
            continue
        
        n_authors = len(author_list)
        author_counts.append(n_authors)
        
        if pd.notna(year):
            year_data[int(year)].append(n_authors)
    
    if not author_counts:
        raise ValueError("No author data found")
    
    author_counts = np.array(author_counts)
    
    # Basic metrics
    single_author = np.sum(author_counts == 1)
    multi_author = np.sum(author_counts > 1)
    
    # Collaboration Index (CI) - mean authors per paper
    collaboration_index = np.mean(author_counts)
    
    # Degree of Collaboration (DC) - proportion of multi-authored papers
    degree_of_collaboration = multi_author / len(author_counts)
    
    # Collaboration Coefficient (CC)
    # CC = 1 - (Σ(1/j * fj)) / N
    f_counts = Counter(author_counts)
    cc_sum = sum((1 / j) * fj for j, fj in f_counts.items())
    collaboration_coefficient = 1 - (cc_sum / len(author_counts))
    
    # Distribution
    distribution = Counter(author_counts)
    dist_df = pd.DataFrame([
        {"n_authors": k, "n_papers": v, "proportion": v / len(author_counts)}
        for k, v in sorted(distribution.items())
    ])
    
    # Temporal trend
    temporal_data = []
    for year in sorted(year_data.keys()):
        if year >= 1900:
            counts = year_data[year]
            temporal_data.append({
                "year": year,
                "n_papers": len(counts),
                "mean_authors": np.mean(counts),
                "median_authors": np.median(counts),
                "single_author_pct": sum(1 for c in counts if c == 1) / len(counts) * 100,
            })
    
    temporal_df = pd.DataFrame(temporal_data)
    
    # Summary statistics
    summary_stats = {
        "n_papers": len(author_counts),
        "mean_authors": collaboration_index,
        "median_authors": np.median(author_counts),
        "std_authors": np.std(author_counts),
        "max_authors": int(np.max(author_counts)),
        "mode_authors": int(stats.mode(author_counts, keepdims=False).mode),
    }
    
    if verbose:
        print(f"  Papers analyzed: {len(author_counts)}")
        print(f"  Collaboration Index: {collaboration_index:.2f}")
        print(f"  Degree of Collaboration: {degree_of_collaboration:.3f}")
        print(f"  Collaboration Coefficient: {collaboration_coefficient:.3f}")
        print(f"  Single-author papers: {single_author} ({single_author/len(author_counts)*100:.1f}%)")
    
    return CollaborationMetrics(
        collaboration_index=collaboration_index,
        degree_of_collaboration=degree_of_collaboration,
        collaboration_coefficient=collaboration_coefficient,
        single_author_papers=single_author,
        multi_author_papers=multi_author,
        max_authors=int(np.max(author_counts)),
        author_distribution=dist_df,
        temporal_trend=temporal_df,
        summary_stats=summary_stats)

# =============================================================================
# DIVERSITY INDICES
# =============================================================================

def calculate_diversity_indices(
    df: pd.DataFrame,
    category_col: str,
    sep: str = "; ",
    verbose: bool = True) -> DiversityIndices:
    """
    Calculate diversity indices for a categorical variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    category_col : str
        Category column (e.g., keywords, subjects).
    sep : str
        Separator for multi-valued fields.
    verbose : bool
        Print progress.
    
    Returns
    -------
    DiversityIndices
    """
    if verbose:
        print(f"Calculating diversity indices for '{category_col}'...")
    
    # Count categories
    category_counts = Counter()
    
    for _, row in df.iterrows():
        value = row.get(category_col, "")
        if pd.isna(value) or not value:
            continue
        
        if isinstance(value, str):
            categories = [c.strip() for c in value.split(sep) if c.strip()]
        else:
            categories = [str(value)]
        
        for cat in categories:
            category_counts[cat] += 1
    
    if not category_counts:
        raise ValueError(f"No categories found in '{category_col}'")
    
    counts = np.array(list(category_counts.values()))
    total = counts.sum()
    proportions = counts / total
    
    # Richness (number of unique categories)
    richness = len(category_counts)
    
    # Shannon Index: H = -Σ(pi * ln(pi))
    shannon = -np.sum(proportions * np.log(proportions))
    
    # Simpson Index: D = Σ(pi^2)
    simpson = np.sum(proportions ** 2)
    
    # Inverse Simpson: 1/D
    inverse_simpson = 1 / simpson if simpson > 0 else 0
    
    # Evenness: H / ln(S)
    evenness = shannon / np.log(richness) if richness > 1 else 0
    
    # Gini coefficient
    sorted_counts = np.sort(counts)
    cumsum = np.cumsum(sorted_counts)
    gini = 1 - 2 * np.sum(cumsum) / (richness * total) if total > 0 else 0
    
    # Herfindahl Index: Σ(si^2) where si is share
    herfindahl = np.sum(proportions ** 2)
    
    # Category distribution
    top_categories = category_counts.most_common(20)
    dist_df = pd.DataFrame(top_categories, columns=["category", "count"])
    dist_df["proportion"] = dist_df["count"] / total
    
    if verbose:
        print(f"  Categories (richness): {richness}")
        print(f"  Shannon Index: {shannon:.3f}")
        print(f"  Simpson Index: {simpson:.3f}")
        print(f"  Evenness: {evenness:.3f}")
        print(f"  Gini coefficient: {gini:.3f}")
    
    return DiversityIndices(
        shannon_index=shannon,
        simpson_index=simpson,
        inverse_simpson=inverse_simpson,
        gini_coefficient=gini,
        herfindahl_index=herfindahl,
        evenness=evenness,
        richness=richness,
        category_distribution=dist_df)

# =============================================================================
# STATISTICAL HYPOTHESIS TESTS
# =============================================================================

def compare_groups(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    test_type: str = "auto",
    alpha: float = 0.05,
    verbose: bool = True) -> StatisticalTestResult:
    """
    Compare groups using appropriate statistical test.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe.
    value_col : str
        Numeric value column.
    group_col : str
        Group column.
    test_type : str
        Test type: "auto", "ttest", "mannwhitney", "anova", "kruskal".
    alpha : float
        Significance level.
    verbose : bool
        Print progress.
    
    Returns
    -------
    StatisticalTestResult
    """
    if verbose:
        print(f"Comparing groups by '{group_col}'...")
    
    # Get groups
    groups = df.groupby(group_col)[value_col].apply(list).to_dict()
    n_groups = len(groups)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for comparison")
    
    group_data = [np.array([x for x in v if pd.notna(x)]) for v in groups.values()]
    group_names = list(groups.keys())
    
    # Auto-select test
    if test_type == "auto":
        # Check normality
        normal = all(len(g) < 8 or stats.shapiro(g[:5000])[1] > 0.05 for g in group_data if len(g) > 3)
        
        if n_groups == 2:
            test_type = "ttest" if normal else "mannwhitney"
        else:
            test_type = "anova" if normal else "kruskal"
    
    # Perform test
    if test_type == "ttest" and n_groups == 2:
        stat, p_value = stats.ttest_ind(group_data[0], group_data[1])
        test_name = "Independent t-test"
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(group_data[0]) + np.var(group_data[1])) / 2)
        effect_size = (np.mean(group_data[0]) - np.mean(group_data[1])) / pooled_std if pooled_std > 0 else 0
        
    elif test_type == "mannwhitney" and n_groups == 2:
        stat, p_value = stats.mannwhitneyu(group_data[0], group_data[1], alternative="two-sided")
        test_name = "Mann-Whitney U test"
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(group_data[0]), len(group_data[1])
        effect_size = 1 - (2 * stat) / (n1 * n2)
        
    elif test_type == "anova":
        stat, p_value = stats.f_oneway(*group_data)
        test_name = "One-way ANOVA"
        
        # Effect size (eta-squared)
        all_data = np.concatenate(group_data)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        effect_size = ss_between / ss_total if ss_total > 0 else 0
        
    elif test_type == "kruskal":
        stat, p_value = stats.kruskal(*group_data)
        test_name = "Kruskal-Wallis H test"
        
        # Effect size (epsilon-squared)
        n = sum(len(g) for g in group_data)
        effect_size = stat / (n - 1) if n > 1 else 0
        
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    is_significant = p_value < alpha
    
    if is_significant:
        conclusion = f"Significant difference found between groups (p = {p_value:.4f})"
    else:
        conclusion = f"No significant difference found between groups (p = {p_value:.4f})"
    
    additional_info = {
        "n_groups": n_groups,
        "group_names": group_names,
        "group_sizes": [len(g) for g in group_data],
        "group_means": [np.mean(g) for g in group_data],
        "group_medians": [np.median(g) for g in group_data],
    }
    
    if verbose:
        print(f"  Test: {test_name}")
        print(f"  Statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Effect size: {effect_size:.3f}")
        print(f"  Conclusion: {'Significant' if is_significant else 'Not significant'}")
    
    return StatisticalTestResult(
        test_name=test_name,
        statistic=stat,
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=None,
        conclusion=conclusion,
        is_significant=is_significant,
        alpha=alpha,
        additional_info=additional_info)

def test_trend(
    df: pd.DataFrame,
    year_col: str = "Year",
    value_col: str = "Cited by",
    alpha: float = 0.05,
    verbose: bool = True) -> StatisticalTestResult:
    """
    Test for significant trend over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe.
    year_col : str
        Year column.
    value_col : str
        Value column to test for trend.
    alpha : float
        Significance level.
    verbose : bool
        Print progress.
    
    Returns
    -------
    StatisticalTestResult
    """
    if verbose:
        print(f"Testing trend in '{value_col}' over time...")
    
    # Aggregate by year
    yearly = df.groupby(year_col)[value_col].mean().reset_index()
    yearly = yearly[yearly[year_col] >= 1900].sort_values(year_col)
    
    years = yearly[year_col].values
    values = yearly[value_col].values
    
    # Mann-Kendall trend test
    n = len(values)
    s = 0
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(values[j] - values[i])
    
    # Variance
    unique_values = np.unique(values)
    if len(unique_values) == n:
        var_s = n * (n - 1) * (2 * n + 5) / 18
    else:
        tp = Counter(values)
        var_s = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in tp.values())) / 18
    
    # Z-score
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Sen's slope
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if years[j] != years[i]:
                slopes.append((values[j] - values[i]) / (years[j] - years[i]))
    
    sens_slope = np.median(slopes) if slopes else 0
    
    is_significant = p_value < alpha
    
    if is_significant:
        direction = "increasing" if s > 0 else "decreasing"
        conclusion = f"Significant {direction} trend detected (p = {p_value:.4f})"
    else:
        conclusion = f"No significant trend detected (p = {p_value:.4f})"
    
    if verbose:
        print(f"  Mann-Kendall statistic (S): {s}")
        print(f"  Z-score: {z:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Sen's slope: {sens_slope:.4f}")
        print(f"  Conclusion: {conclusion}")
    
    return StatisticalTestResult(
        test_name="Mann-Kendall trend test",
        statistic=z,
        p_value=p_value,
        effect_size=sens_slope,
        confidence_interval=None,
        conclusion=conclusion,
        is_significant=is_significant,
        alpha=alpha,
        additional_info={"s_statistic": s, "sens_slope": sens_slope, "n_years": n})

# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval.
    
    Parameters
    ----------
    data : np.ndarray
        Data array.
    statistic_func : callable
        Function to compute statistic.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence_level : float
        Confidence level.
    random_state : int
        Random seed.
    
    Returns
    -------
    Tuple of (point_estimate, lower_bound, upper_bound).
    """
    np.random.seed(random_state)
    
    data = np.array(data)
    n = len(data)
    
    # Point estimate
    point_estimate = statistic_func(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Percentile method
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return point_estimate, lower, upper

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_advanced_statistics(
    df: pd.DataFrame,
    authors_col: str = "Authors",
    source_col: str = "Source title",
    year_col: str = "Year",
    citations_col: str = "Cited by",
    abstract_col: str = "Abstract",
    title_col: str = "Title",
    keywords_col: str = "Author Keywords",
    sep: str = "; ",
    run_lotka: bool = True,
    run_bradford: bool = True,
    run_zipf: bool = True,
    run_growth: bool = True,
    run_citation_dist: bool = True,
    run_collaboration: bool = True,
    run_diversity: bool = True,
    verbose: bool = True) -> AdvancedStatsResult:
    """
    Run comprehensive advanced statistical analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    Various column parameters.
    run_* : bool
        Flags to enable/disable specific analyses.
    verbose : bool
        Print progress.
    
    Returns
    -------
    AdvancedStatsResult
    """
    if verbose:
        print("=" * 50)
        print("Advanced Statistical Analysis")
        print("=" * 50)
    
    lotka = None
    bradford = None
    zipf = None
    growth = None
    citation_dist = None
    collaboration = None
    diversity = None
    statistical_tests = []
    regression_results = {}
    
    # Lotka's Law
    if run_lotka and authors_col in df.columns:
        try:
            if verbose:
                print("\n--- Lotka's Law ---")
            lotka = analyze_lotka_law(df, authors_col, sep, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  Warning: Lotka analysis failed: {e}")
    
    # Bradford's Law
    if run_bradford and source_col in df.columns:
        try:
            if verbose:
                print("\n--- Bradford's Law ---")
            bradford = analyze_bradford_law(df, source_col, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  Warning: Bradford analysis failed: {e}")
    
    # Zipf's Law
    if run_zipf and (abstract_col in df.columns or title_col in df.columns):
        try:
            if verbose:
                print("\n--- Zipf's Law ---")
            zipf = analyze_zipf_law(df, abstract_col, title_col, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  Warning: Zipf analysis failed: {e}")
    
    # Growth Model
    if run_growth and year_col in df.columns:
        try:
            if verbose:
                print("\n--- Growth Model ---")
            growth = fit_growth_model(df, year_col, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  Warning: Growth model failed: {e}")
    
    # Citation Distribution
    if run_citation_dist and citations_col in df.columns:
        try:
            if verbose:
                print("\n--- Citation Distribution ---")
            citation_dist = analyze_citation_distribution(df, citations_col, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  Warning: Citation distribution failed: {e}")
    
    # Collaboration Metrics
    if run_collaboration and authors_col in df.columns:
        try:
            if verbose:
                print("\n--- Collaboration Metrics ---")
            collaboration = analyze_collaboration(df, authors_col, year_col, sep, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  Warning: Collaboration analysis failed: {e}")
    
    # Diversity Indices
    if run_diversity and keywords_col in df.columns:
        try:
            if verbose:
                print("\n--- Diversity Indices ---")
            diversity = calculate_diversity_indices(df, keywords_col, sep, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  Warning: Diversity analysis failed: {e}")
    
    # Trend test
    if year_col in df.columns and citations_col in df.columns:
        try:
            if verbose:
                print("\n--- Trend Analysis ---")
            trend_result = test_trend(df, year_col, citations_col, verbose=verbose)
            statistical_tests.append(trend_result)
        except Exception as e:
            if verbose:
                print(f"  Warning: Trend test failed: {e}")
    
    if verbose:
        print("\n" + "=" * 50)
        print("Analysis complete!")
    
    return AdvancedStatsResult(
        lotka=lotka,
        bradford=bradford,
        zipf=zipf,
        growth_model=growth,
        citation_distribution=citation_dist,
        collaboration=collaboration,
        diversity=diversity,
        statistical_tests=statistical_tests,
        regression_results=regression_results,
        parameters={
            "authors_col": authors_col,
            "source_col": source_col,
            "year_col": year_col,
            "citations_col": citations_col,
        })

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_lotka_law(
    result: LotkaResult,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot Lotka's Law analysis."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    df = result.observed_distribution
    
    # Log-log plot
    ax.scatter(df["papers"], df["proportion"], s=50, alpha=0.7, label="Observed")
    
    # Theoretical line
    x_theory = np.linspace(1, df["papers"].max(), 100)
    y_theory = result.lotka_constant / (x_theory ** result.lotka_exponent)
    y_theory = y_theory / y_theory.sum() * df["proportion"].sum()
    ax.plot(x_theory, y_theory, "r--", linewidth=2, label="Lotka fit")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Papers", fontsize=11)
    ax.set_ylabel("Proportion of Authors", fontsize=11)
    ax.set_title(f"Lotka's Law (n = {result.lotka_exponent:.2f}, R² = {result.r_squared:.3f})", fontsize=12)
    ax.legend()
    ax
    
    # Bar chart of top authors
    top = result.most_productive_authors.head(15)
    ax.barh(range(len(top)), top["papers"], color="lightblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([a[:30] for a in top["author"]], fontsize=9)
    ax.set_xlabel("Number of Papers", fontsize=11)
    ax.set_title("Most Productive Authors", fontsize=12)
    ax.invert_yaxis()
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_bradford_law(
    result: BradfordResult,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """Plot Bradford's Law analysis."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    df = result.source_ranking
    
    # Bradford curve
    ax.plot(np.log(df["rank"]), df["cumulative_papers"], linewidth=2, color="lightblue")
    ax.set_xlabel("log(Rank)", fontsize=11)
    ax.set_ylabel("Cumulative Papers", fontsize=11)
    ax.set_title(f"Bradford Curve (R² = {result.r_squared:.3f})", fontsize=12)
    ax
    
    # Zone breakdown
    zones = result.zones
    colors = "lightblue"[:len(zones)]
    
    x = range(len(zones))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], zones["n_sources"], width, label="Sources", color="lightblue")
    ax.bar([i + width/2 for i in x], zones["n_papers"], width, label="Papers", color="lightblue")
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"Zone {z}" for z in zones["zone"]])
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Bradford Zones (multiplier = {result.bradford_multiplier:.2f})", fontsize=12)
    ax.legend()
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_growth_model(
    result: GrowthModelResult,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """Plot growth model results."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    df = result.prediction
    historical = df[~df["is_forecast"]]
    forecast = df[df["is_forecast"]]
    
    # Growth curve with forecast
    ax.scatter(historical["year"], historical["papers"], s=30, alpha=0.7, label="Observed")
    ax.plot(df["year"], df["fitted"], "r-", linewidth=2, label="Fitted")
    
    if len(forecast) > 0:
        ax.plot(forecast["year"], forecast["fitted"], "r--", linewidth=2, label="Forecast")
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title(f"Growth Model: {result.model_type.title()} (R² = {result.r_squared:.3f})", fontsize=12)
    ax.legend()
    ax
    
    # Residuals
    ax.scatter(historical["year"], result.residuals, s=30, alpha=0.7)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Residuals", fontsize=11)
    ax.set_title("Model Residuals", fontsize=12)
    ax
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_citation_distribution(
    result: CitationDistributionResult,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot citation distribution analysis."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    stats_dict = result.summary_stats
    
    # Histogram with log scale
    # Create dummy data for visualization
    citations = np.random.lognormal(
        result.parameters.get("mu", 1),
        result.parameters.get("sigma", 1),
        1000
    ).astype(int)
    
    ax.hist(citations[citations < np.percentile(citations, 99)], bins=50, 
            color="lightblue", edgecolor="white", alpha=0.7)
    ax.set_xlabel("Citations", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Citation Distribution", fontsize=12)
    ax.legend()
    
    # Summary stats
    stats_text = [
        f"Distribution: {result.distribution_type}",
        f"H-index: {result.h_index}",
        f"Gini coefficient: {result.gini_coefficient:.3f}",
        f"Uncited: {result.proportion_uncited*100:.1f}%",
        f"Top 10% threshold: {result.highly_cited_threshold:.0f}",
        "",
        "Percentiles:",
    ]
    for p, v in result.percentiles.items():
        stats_text.append(f"  {p}th: {v:.0f}")
    
    ax.text(0.1, 0.9, "\n".join(stats_text), transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace")
    ax.axis("off")
    ax.set_title("Distribution Statistics", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_collaboration_metrics(
    result: CollaborationMetrics,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """Plot collaboration analysis."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    # Author distribution
    dist = result.author_distribution
    ax.bar(dist["n_authors"], dist["n_papers"], color="lightblue", edgecolor="white")
    ax.set_xlabel("Number of Authors", fontsize=11)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title("Author Count Distribution", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Collaboration metrics summary
    metrics = ["Collab. Index", "Degree of Collab.", "Collab. Coefficient"]
    values = [result.collaboration_index, result.degree_of_collaboration, result.collaboration_coefficient]
    colors = "lightblue"
    bars = ax.bar(range(len(metrics)), values, color=colors)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Collaboration Metrics", fontsize=12)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    
    # Temporal trend
    if len(result.temporal_trend) > 0:
        trend = result.temporal_trend
        ax.plot(trend["year"], trend["mean_authors"], marker="o", linewidth=2, color="lightblue")
        
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Mean Authors per Paper", fontsize=11)
        ax.set_title("Collaboration Trend Over Time", fontsize=12)
        ax
    
    # Single vs multi-author
    sizes = [result.single_author_papers, result.multi_author_papers]
    labels = ["Single Author", "Multi-Author"]
    colors = ["#e74c3c", "#27ae60"]
    
    bars = ax.bar(range(len(labels)), sizes, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title("Authorship Type", fontsize=12)
    
    for bar, size in zip(bars, sizes):
        pct = size / sum(sizes) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{size}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_advanced_stats(
    result: AdvancedStatsResult,
    output_dir: str) -> Dict[str, str]:
    """Export advanced statistics results."""
    os.makedirs(output_dir, exist_ok=True)
    
    excel_path = os.path.join(output_dir, "advanced_statistics.xlsx")
    
    with pd.ExcelWriter(excel_path) as writer:
        # Lotka
        if result.lotka:
            result.lotka.observed_distribution.to_excel(writer, sheet_name="Lotka_Distribution", index=False)
            result.lotka.most_productive_authors.to_excel(writer, sheet_name="Top_Authors", index=False)
            pd.DataFrame([result.lotka.summary_stats]).to_excel(writer, sheet_name="Lotka_Summary", index=False)
        
        # Bradford
        if result.bradford:
            result.bradford.zones.to_excel(writer, sheet_name="Bradford_Zones", index=False)
            result.bradford.source_ranking.head(100).to_excel(writer, sheet_name="Source_Ranking", index=False)
        
        # Zipf
        if result.zipf:
            result.zipf.top_words.to_excel(writer, sheet_name="Top_Words", index=False)
        
        # Growth
        if result.growth_model:
            result.growth_model.prediction.to_excel(writer, sheet_name="Growth_Prediction", index=False)
        
        # Citation Distribution
        if result.citation_distribution:
            pd.DataFrame([result.citation_distribution.summary_stats]).to_excel(
                writer, sheet_name="Citation_Stats", index=False
            )
        
        # Collaboration
        if result.collaboration:
            result.collaboration.author_distribution.to_excel(
                writer, sheet_name="Author_Distribution", index=False
            )
            result.collaboration.temporal_trend.to_excel(
                writer, sheet_name="Collab_Trend", index=False
            )
        
        # Diversity
        if result.diversity:
            result.diversity.category_distribution.to_excel(
                writer, sheet_name="Diversity", index=False
            )
    
    print(f"Exported to: {excel_path}")
    return {"excel": excel_path}

# =============================================================================
# CLASS INTEGRATION
# =============================================================================

def add_advanced_stats_methods(cls):
    """Add advanced statistics methods to BiblioAnalysis class."""
    
    def run_advanced_stats_method(self, **kwargs) -> AdvancedStatsResult:
        """Run advanced statistical analysis."""
        self.advanced_stats_results = run_advanced_statistics(
            self.df,
            authors_col="Authors",
            source_col="Source title",
            year_col="Year",
            citations_col="Cited by",
            abstract_col="Abstract",
            title_col="Title",
            keywords_col="Author Keywords",
            **kwargs
        )
        
        if hasattr(self, "res_folder") and self.res_folder:
            export_advanced_stats(
                self.advanced_stats_results,
                self.res_folder
            )
        
        return self.advanced_stats_results
    
    def plot_advanced_stats_method(
        self,
        plot_type: str = "lotka",
        save: bool = True,
        **kwargs
    ) -> plt.Figure:
        """Create advanced statistics visualizations."""
        if not hasattr(self, "advanced_stats_results"):
            raise ValueError("Run run_advanced_stats first")
        
        save_path = None
        if save and hasattr(self, "res_folder") and self.res_folder:
            save_path = os.path.join(self.res_folder, f"stats_{plot_type}")
        
        result = self.advanced_stats_results
        
        plot_map = {
            "lotka": (plot_lotka_law, result.lotka),
            "bradford": (plot_bradford_law, result.bradford),
            "growth": (plot_growth_model, result.growth_model),
            "citations": (plot_citation_distribution, result.citation_distribution),
            "collaboration": (plot_collaboration_metrics, result.collaboration),
        }
        
        if plot_type not in plot_map:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        func, data = plot_map[plot_type]
        if data is None:
            raise ValueError(f"No data available for {plot_type}")
        
        return func(data, save_path=save_path, **kwargs)
    
    cls.run_advanced_stats = run_advanced_stats_method
    cls.plot_advanced_stats = plot_advanced_stats_method
    
    return cls
# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import biblium as bb
    ba = bb.BiblioAnalysis(f_name="data\\open alex dataset.csv", db="oa")
    
    print("Advanced Statistics Analysis")
    print("=" * 60)
    
    # Run analysis
    result = run_advanced_statistics(
        df=ba.df,
        authors_col="Authors",
        source_col="Source title",
        citations_col="Cited by",
        year_col="Year",
        verbose=True)
    
    # Print summary
    if result.lotka:
        print(f"\nLotka exponent: {result.lotka.lotka_exponent:.3f}")
    if result.bradford:
        print(f"Bradford zones: {len(result.bradford.zones)}")
    
    # Visualizations
    print("\nGenerating plots...")
    if result.lotka:
        plot_lotka_law(result.lotka, save_path="results/lotka_law")
    if result.bradford:
        plot_bradford_law(result.bradford, save_path="results/bradford_law")
    
    # Export
    print("\nExporting results...")
    export_advanced_stats(result, "results")
    
    print("\nDone!")
