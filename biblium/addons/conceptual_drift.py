# -*- coding: utf-8 -*-
"""
Conceptual Drift Detection Module for Bibliometric Analysis

This module provides functions to detect and visualize how scientific concepts
evolve over time. It tracks semantic shifts in terminology by analyzing
changes in word co-occurrence patterns and contextual usage.

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr, pearsonr
from scipy.spatial.distance import cosine, jensenshannon
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
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
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# =============================================================================
# DEFAULT COLORMAPS (user-configurable)
# =============================================================================

CMAP_CONTINUOUS = "viridis"  # For continuous/sequential data
CATEGORICAL_COLOR = "lightblue"      # Single color for categorical

def set_default_cmaps(continuous: str = None, discrete: str = None):
    """Set default colormaps for all plots in this module."""
    global CMAP_CONTINUOUS, CMAP_DISCRETE
    if continuous:
        CMAP_CONTINUOUS = continuous
    if discrete:
        CMAP_DISCRETE = discrete

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DriftResult:
    """Container for drift analysis results for a single term."""
    term: str
    periods: List[str]
    drift_scores: List[float]  # Drift vs previous period
    cumulative_drift: List[float]  # Drift vs first period
    context_words: Dict[str, List[str]]  # Period -> top context words
    emerging_words: Dict[str, List[str]]  # Period -> newly prominent words
    fading_words: Dict[str, List[str]]  # Period -> declining words
    doc_counts: Dict[str, int]  # Period -> document count
    field_distribution: Optional[Dict[str, Dict[str, float]]] = None  # Period -> field -> proportion
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a summary DataFrame."""
        records = []
        for i, period in enumerate(self.periods):
            records.append({
                "Term": self.term,
                "Period": period,
                "Drift_vs_Previous": self.drift_scores[i] if i > 0 else None,
                "Drift_vs_Baseline": self.cumulative_drift[i],
                "Document Count": self.doc_counts.get(period, 0),
                "Top Context Words": "; ".join(self.context_words.get(period, [])[:10]),
                "Emerging Words": "; ".join(self.emerging_words.get(period, [])[:5]),
                "Fading Words": "; ".join(self.fading_words.get(period, [])[:5]),
            })
        return pd.DataFrame(records)

@dataclass
class ConceptualDriftAnalysis:
    """Container for full drift analysis results."""
    term_results: Dict[str, DriftResult]
    periods: List[str]
    method: str
    parameters: Dict[str, Any]
    global_vocabulary: List[str]
    period_stats: Dict[str, Dict[str, Any]]  # Period -> stats
    
    def get_summary_df(self) -> pd.DataFrame:
        """Get summary DataFrame for all terms."""
        dfs = [result.to_dataframe() for result in self.term_results.values()]
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def get_drift_matrix(self) -> pd.DataFrame:
        """Get matrix of drift scores (terms x periods)."""
        data = {}
        for term, result in self.term_results.items():
            data[term] = result.drift_scores
        return pd.DataFrame(data, index=self.periods).T
    
    def get_high_drift_terms(self, threshold: float = 0.3) -> List[str]:
        """Get terms with high average drift."""
        high_drift = []
        for term, result in self.term_results.items():
            valid_scores = [s for s in result.drift_scores if s is not None]
            if valid_scores and np.mean(valid_scores) > threshold:
                high_drift.append((term, np.mean(valid_scores)))
        return [t[0] for t in sorted(high_drift, key=lambda x: -x[1])]
    
    def get_stable_terms(self, threshold: float = 0.1) -> List[str]:
        """Get terms with low drift (stable concepts)."""
        stable = []
        for term, result in self.term_results.items():
            valid_scores = [s for s in result.drift_scores if s is not None]
            if valid_scores and np.mean(valid_scores) < threshold:
                stable.append((term, np.mean(valid_scores)))
        return [t[0] for t in sorted(stable, key=lambda x: x[1])]

# =============================================================================
# CORE COMPUTATION FUNCTIONS
# =============================================================================

def create_time_windows(
    df: pd.DataFrame,
    year_col: str = "Year",
    window_size: int = 5,
    window_type: str = "fixed",  # "fixed" or "sliding"
    slide_step: int = 1,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
) -> List[Tuple[int, int, pd.DataFrame]]:
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
        Step size for sliding windows (ignored if window_type="fixed").
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

def compute_cooccurrence_profile(
    texts: Iterable[str],
    target_term: str,
    top_n: int = 100,
    window_size: int = 10,
    min_count: int = 2,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute co-occurrence profile for a target term.
    
    Parameters
    ----------
    texts : Iterable[str]
        Collection of text documents.
    target_term : str
        Term to compute co-occurrences for.
    top_n : int
        Number of top co-occurring terms to return.
    window_size : int
        Context window size (words on each side).
    min_count : int
        Minimum co-occurrence count to include.
    normalize : bool
        If True, normalize counts to probabilities.
    
    Returns
    -------
    Dict mapping co-occurring terms to their scores.
    """
    cooccur = Counter()
    target_lower = target_term.lower()
    target_count = 0
    
    for text in texts:
        if not isinstance(text, str):
            continue
        
        words = text.lower().split()
        
        for i, word in enumerate(words):
            if word == target_lower or target_lower in word:
                target_count += 1
                # Get context window
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                context = words[start:i] + words[i+1:end]
                
                for ctx_word in context:
                    if ctx_word != target_lower and len(ctx_word) > 2:
                        cooccur[ctx_word] += 1
    
    # Filter by minimum count
    cooccur = {k: v for k, v in cooccur.items() if v >= min_count}
    
    # Normalize if requested
    if normalize and cooccur:
        total = sum(cooccur.values())
        cooccur = {k: v / total for k, v in cooccur.items()}
    
    # Return top N
    sorted_items = sorted(cooccur.items(), key=lambda x: -x[1])
    return dict(sorted_items[:top_n])

def compute_drift_score(
    profile1: Dict[str, float],
    profile2: Dict[str, float],
    method: str = "jensen_shannon",
) -> float:
    """
    Compute drift score between two co-occurrence profiles.
    
    Parameters
    ----------
    profile1, profile2 : Dict[str, float]
        Co-occurrence profiles to compare.
    method : str
        Distance method: "jensen_shannon", "cosine", "jaccard", "rank_correlation".
    
    Returns
    -------
    float
        Drift score (0 = identical, 1 = maximally different).
    """
    if not profile1 or not profile2:
        return 1.0  # Maximum drift if one profile is empty
    
    # Get union of all terms
    all_terms = set(profile1.keys()) | set(profile2.keys())
    
    if method == "jensen_shannon":
        # Convert to probability distributions
        vec1 = np.array([profile1.get(t, 0) for t in all_terms])
        vec2 = np.array([profile2.get(t, 0) for t in all_terms])
        
        # Normalize to sum to 1
        vec1 = vec1 / (vec1.sum() + 1e-10)
        vec2 = vec2 / (vec2.sum() + 1e-10)
        
        # Add small epsilon to avoid log(0)
        vec1 = vec1 + 1e-10
        vec2 = vec2 + 1e-10
        vec1 = vec1 / vec1.sum()
        vec2 = vec2 / vec2.sum()
        
        return float(jensenshannon(vec1, vec2))
    
    elif method == "cosine":
        vec1 = np.array([profile1.get(t, 0) for t in all_terms])
        vec2 = np.array([profile2.get(t, 0) for t in all_terms])
        
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 1.0
        
        return float(cosine(vec1, vec2))
    
    elif method == "jaccard":
        set1 = set(profile1.keys())
        set2 = set(profile2.keys())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 1.0
        
        return 1.0 - (intersection / union)
    
    elif method == "rank_correlation":
        # Spearman correlation of ranks
        common_terms = set(profile1.keys()) & set(profile2.keys())
        
        if len(common_terms) < 3:
            return 1.0
        
        ranks1 = [profile1[t] for t in common_terms]
        ranks2 = [profile2[t] for t in common_terms]
        
        corr, _ = spearmanr(ranks1, ranks2)
        
        # Convert correlation to distance (1 = perfect correlation -> 0 drift)
        return (1 - corr) / 2
    
    else:
        raise ValueError(f"Unknown method: {method}")

def identify_emerging_fading_terms(
    profile_old: Dict[str, float],
    profile_new: Dict[str, float],
    emergence_threshold: float = 2.0,
    fading_threshold: float = 0.5,
    top_n: int = 10,
) -> Tuple[List[str], List[str]]:
    """
    Identify terms that emerged or faded between two profiles.
    
    Parameters
    ----------
    profile_old, profile_new : Dict[str, float]
        Co-occurrence profiles from earlier and later periods.
    emergence_threshold : float
        Minimum ratio (new/old) to be considered emerging.
    fading_threshold : float
        Maximum ratio (new/old) to be considered fading.
    top_n : int
        Maximum number of terms to return per category.
    
    Returns
    -------
    Tuple of (emerging_terms, fading_terms).
    """
    all_terms = set(profile_old.keys()) | set(profile_new.keys())
    
    emerging = []
    fading = []
    
    for term in all_terms:
        old_score = profile_old.get(term, 0)
        new_score = profile_new.get(term, 0)
        
        # Handle new appearances
        if old_score < 1e-6 and new_score > 1e-6:
            emerging.append((term, float('inf'), new_score))
            continue
        
        # Handle disappearances
        if new_score < 1e-6 and old_score > 1e-6:
            fading.append((term, 0, old_score))
            continue
        
        if old_score > 1e-6:
            ratio = new_score / old_score
            
            if ratio >= emergence_threshold:
                emerging.append((term, ratio, new_score))
            elif ratio <= fading_threshold:
                fading.append((term, ratio, old_score))
    
    # Sort by strength of change
    emerging = sorted(emerging, key=lambda x: -x[2])[:top_n]
    fading = sorted(fading, key=lambda x: -x[2])[:top_n]
    
    return [t[0] for t in emerging], [t[0] for t in fading]

def compute_conceptual_drift(
    df: pd.DataFrame,
    target_terms: List[str],
    text_col: str = "Processed Abstract",
    year_col: str = "Year",
    window_size: int = 5,
    window_type: str = "fixed",
    method: str = "jensen_shannon",
    top_n_context: int = 100,
    min_docs_per_window: int = 50,
    min_term_occurrences: int = 10,
    context_window_words: int = 10,
    field_col: Optional[str] = None,
) -> ConceptualDriftAnalysis:
    """
    Compute conceptual drift for target terms over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe with text and year columns.
    target_terms : List[str]
        Terms to analyze for drift.
    text_col : str
        Column containing text to analyze.
    year_col : str
        Column containing publication year.
    window_size : int
        Size of time windows in years.
    window_type : str
        "fixed" or "sliding" windows.
    method : str
        Drift computation method.
    top_n_context : int
        Number of context words to track per period.
    min_docs_per_window : int
        Minimum documents required per window.
    min_term_occurrences : int
        Minimum term occurrences per window.
    context_window_words : int
        Word window size for co-occurrence.
    field_col : str, optional
        Column containing field/discipline classification.
    
    Returns
    -------
    ConceptualDriftAnalysis
        Complete drift analysis results.
    """
    # Create time windows
    windows = create_time_windows(
        df, year_col, window_size, window_type
    )
    
    # Filter windows with insufficient data
    windows = [(s, e, d) for s, e, d in windows if len(d) >= min_docs_per_window]
    
    if not windows:
        raise ValueError("No time windows with sufficient data.")
    
    periods = [f"{s}-{e}" for s, e, _ in windows]
    
    # Initialize results
    term_results = {}
    period_stats = {}
    
    # Compute stats per period
    for start, end, subset in windows:
        period = f"{start}-{end}"
        period_stats[period] = {
            "n_docs": len(subset),
            "year_range": (start, end),
        }
    
    # Analyze each target term
    for term in target_terms:
        term_lower = term.lower()
        
        profiles = {}
        doc_counts = {}
        field_distributions = {} if field_col else None
        
        # Compute profile for each period
        for start, end, subset in windows:
            period = f"{start}-{end}"
            
            # Filter to documents containing the term
            if text_col in subset.columns:
                mask = subset[text_col].fillna("").str.lower().str.contains(
                    term_lower, regex=False
                )
                term_docs = subset[mask]
            else:
                term_docs = pd.DataFrame()
            
            doc_counts[period] = len(term_docs)
            
            if len(term_docs) >= min_term_occurrences:
                texts = term_docs[text_col].dropna().tolist()
                
                profiles[period] = compute_cooccurrence_profile(
                    texts,
                    term,
                    top_n=top_n_context,
                    window_size=context_window_words,
                )
                
                # Compute field distribution if available
                if field_col and field_col in term_docs.columns:
                    field_counts = term_docs[field_col].value_counts(normalize=True)
                    field_distributions[period] = field_counts.to_dict()
            else:
                profiles[period] = {}
        
        # Compute drift scores
        drift_scores = [None]  # First period has no "previous"
        cumulative_drift = [0.0]  # First period is baseline
        context_words = {}
        emerging_words = {}
        fading_words = {}
        
        baseline_profile = profiles.get(periods[0], {})
        context_words[periods[0]] = list(baseline_profile.keys())[:top_n_context]
        emerging_words[periods[0]] = []
        fading_words[periods[0]] = []
        
        prev_profile = baseline_profile
        
        for i, period in enumerate(periods[1:], 1):
            current_profile = profiles.get(period, {})
            
            # Drift vs previous
            drift_vs_prev = compute_drift_score(prev_profile, current_profile, method)
            drift_scores.append(drift_vs_prev)
            
            # Drift vs baseline
            drift_vs_base = compute_drift_score(baseline_profile, current_profile, method)
            cumulative_drift.append(drift_vs_base)
            
            # Context words
            context_words[period] = list(current_profile.keys())[:top_n_context]
            
            # Emerging and fading
            emerging, fading = identify_emerging_fading_terms(
                prev_profile, current_profile
            )
            emerging_words[period] = emerging
            fading_words[period] = fading
            
            prev_profile = current_profile
        
        term_results[term] = DriftResult(
            term=term,
            periods=periods,
            drift_scores=drift_scores,
            cumulative_drift=cumulative_drift,
            context_words=context_words,
            emerging_words=emerging_words,
            fading_words=fading_words,
            doc_counts=doc_counts,
            field_distribution=field_distributions,
        )
    
    # Build global vocabulary
    all_context_words = set()
    for result in term_results.values():
        for words in result.context_words.values():
            all_context_words.update(words)
    
    return ConceptualDriftAnalysis(
        term_results=term_results,
        periods=periods,
        method=method,
        parameters={
            "window_size": window_size,
            "window_type": window_type,
            "top_n_context": top_n_context,
            "min_docs_per_window": min_docs_per_window,
            "min_term_occurrences": min_term_occurrences,
            "context_window_words": context_window_words,
        },
        global_vocabulary=sorted(all_context_words),
        period_stats=period_stats,
    )

# =============================================================================
# EMBEDDING-BASED DRIFT (Optional, requires gensim)
# =============================================================================

def compute_embedding_drift(
    df: pd.DataFrame,
    target_terms: List[str],
    text_col: str = "Processed Abstract",
    year_col: str = "Year",
    window_size: int = 5,
    embedding_dim: int = 100,
    min_count: int = 5,
    window_words: int = 5,
    min_docs_per_window: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute drift using Word2Vec embeddings trained per time period.
    
    Requires gensim to be installed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_terms : List[str]
        Terms to track.
    text_col : str
        Text column name.
    year_col : str
        Year column name.
    window_size : int
        Time window size in years.
    embedding_dim : int
        Word2Vec embedding dimensions.
    min_count : int
        Minimum word frequency for Word2Vec.
    window_words : int
        Word2Vec context window.
    min_docs_per_window : int
        Minimum documents per period.
    
    Returns
    -------
    Dict with drift information per term.
    """
    if not GENSIM_AVAILABLE:
        raise ImportError("gensim is required for embedding-based drift. Install with: pip install gensim")
    
    windows = create_time_windows(df, year_col, window_size, "fixed")
    windows = [(s, e, d) for s, e, d in windows if len(d) >= min_docs_per_window]
    
    if len(windows) < 2:
        raise ValueError("Need at least 2 time windows for embedding drift.")
    
    periods = [f"{s}-{e}" for s, e, _ in windows]
    
    # Train embeddings per period
    models = {}
    for start, end, subset in windows:
        period = f"{start}-{end}"
        
        # Prepare sentences
        sentences = []
        for text in subset[text_col].dropna():
            if isinstance(text, str):
                sentences.append(text.lower().split())
        
        if len(sentences) < 100:
            continue
        
        # Train Word2Vec
        model = Word2Vec(
            sentences,
            vector_size=embedding_dim,
            window=window_words,
            min_count=min_count,
            workers=4,
            epochs=10,
        )
        models[period] = model
    
    # Compute drift for each term
    results = {}
    
    for term in target_terms:
        term_lower = term.lower()
        term_data = {
            "periods": [],
            "vectors": [],
            "drift_scores": [],
            "nearest_neighbors": {},
        }
        
        prev_vector = None
        
        for period in periods:
            if period not in models:
                continue
            
            model = models[period]
            
            if term_lower not in model.wv:
                continue
            
            vector = model.wv[term_lower]
            term_data["periods"].append(period)
            term_data["vectors"].append(vector)
            
            # Get nearest neighbors
            try:
                neighbors = model.wv.most_similar(term_lower, topn=10)
                term_data["nearest_neighbors"][period] = [n[0] for n in neighbors]
            except KeyError:
                term_data["nearest_neighbors"][period] = []
            
            # Compute drift vs previous
            if prev_vector is not None:
                drift = cosine(prev_vector, vector)
                term_data["drift_scores"].append(drift)
            else:
                term_data["drift_scores"].append(None)
            
            prev_vector = vector
        
        results[term] = term_data
    
    return results

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_drift_timeline(
    analysis: ConceptualDriftAnalysis,
    terms: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
    cmap: str = "tab10",
    show_cumulative: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    Plot drift scores over time for multiple terms.
    
    Parameters
    ----------
    analysis : ConceptualDriftAnalysis
        Results from compute_conceptual_drift.
    terms : List[str], optional
        Specific terms to plot. If None, plot all.
    figsize : Tuple[int, int]
        Figure size.
    cmap : str
        Colormap name.
    show_cumulative : bool
        If True, show cumulative drift from baseline.
    title : str, optional
        Custom title.
    save_path : str, optional
        Path to save figure (without extension).
    dpi : int
        Resolution for saved figure.
    
    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    if terms is None:
        terms = list(analysis.term_results.keys())
    
    colors = "lightblue"
    
    periods = analysis.periods
    x_positions = np.arange(len(periods))
    
    for i, term in enumerate(terms):
        if term not in analysis.term_results:
            continue
        
        result = analysis.term_results[term]
        
        if show_cumulative:
            y_values = result.cumulative_drift
            ylabel = "Cumulative Drift (vs Baseline)"
        else:
            y_values = result.drift_scores
            ylabel = "Drift Score (vs Previous Period)"
        
        # Replace None with NaN for plotting
        y_values = [v if v is not None else np.nan for v in y_values]
        
        ax.plot(
            x_positions, y_values,
            marker='o', linewidth=2, markersize=8,
            color="lightblue", label=term, alpha=0.8
        )
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title is None:
        title = "Conceptual Drift Over Time"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_drift_heatmap(
    analysis: ConceptualDriftAnalysis,
    terms: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = None,
    annotate: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    Plot heatmap of drift scores (terms x periods).
    
    Parameters
    ----------
    analysis : ConceptualDriftAnalysis
        Results from compute_conceptual_drift.
    terms : List[str], optional
        Specific terms to include.
    figsize : Tuple[int, int]
        Figure size.
    cmap : str
        Colormap for heatmap.
    annotate : bool
        Whether to annotate cells with values.
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
    # Build drift matrix
    if terms is None:
        terms = list(analysis.term_results.keys())
    
    periods = analysis.periods
    
    data = []
    for term in terms:
        if term in analysis.term_results:
            scores = analysis.term_results[term].drift_scores
            scores = [s if s is not None else np.nan for s in scores]
            data.append(scores)
        else:
            data.append([np.nan] * len(periods))
    
    matrix = np.array(data)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    im = ax.imshow(matrix, cmap=cmap or CMAP_CONTINUOUS, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Drift Score", fontsize=11)
    
    # Set ticks
    ax.set_xticks(np.arange(len(periods)))
    ax.set_yticks(np.arange(len(terms)))
    ax.set_xticklabels(periods, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(terms, fontsize=10)
    
    # Annotate
    if annotate:
        for i in range(len(terms)):
            for j in range(len(periods)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                           color=text_color, fontsize=9)
    
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Term", fontsize=12)
    
    if title is None:
        title = "Conceptual Drift Heatmap"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_context_evolution(
    analysis: ConceptualDriftAnalysis,
    term: str,
    top_n: int = 15,
    figsize: Tuple[int, int] = (16, 10),
    cmap: str = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    Plot evolution of context words for a single term over time.
    
    Shows a heatmap of context word importance across periods.
    
    Parameters
    ----------
    analysis : ConceptualDriftAnalysis
        Results from compute_conceptual_drift.
    term : str
        Term to visualize.
    top_n : int
        Number of top context words to show.
    figsize : Tuple[int, int]
        Figure size.
    cmap : str
        Colormap.
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
    if term not in analysis.term_results:
        raise ValueError(f"Term '{term}' not found in analysis.")
    
    result = analysis.term_results[term]
    periods = result.periods
    
    # Collect all context words and their scores
    all_words = set()
    word_scores = {}
    
    for period in periods:
        context = result.context_words.get(period, [])
        for i, word in enumerate(context[:top_n]):
            all_words.add(word)
            if word not in word_scores:
                word_scores[word] = {}
            # Use inverse rank as score (higher = more important)
            word_scores[word][period] = top_n - i
    
    # Select top words based on total importance
    word_totals = {w: sum(scores.values()) for w, scores in word_scores.items()}
    top_words = sorted(word_totals.keys(), key=lambda x: -word_totals[x])[:top_n]
    
    # Build matrix
    matrix = np.zeros((len(top_words), len(periods)))
    for i, word in enumerate(top_words):
        for j, period in enumerate(periods):
            matrix[i, j] = word_scores.get(word, {}).get(period, 0)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    im = ax.imshow(matrix, cmap=cmap or CMAP_CONTINUOUS, aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Importance Rank (higher = more prominent)", fontsize=10)
    
    ax.set_xticks(np.arange(len(periods)))
    ax.set_yticks(np.arange(len(top_words)))
    ax.set_xticklabels(periods, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(top_words, fontsize=10)
    
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Context Word", fontsize=12)
    
    if title is None:
        title = f"Context Evolution for '{term}'"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_emerging_fading_terms(
    analysis: ConceptualDriftAnalysis,
    term: str,
    figsize: Tuple[int, int] = (14, 8),
    n_terms: int = 5,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    Visualize emerging and fading context terms over time.
    
    Parameters
    ----------
    analysis : ConceptualDriftAnalysis
        Results from compute_conceptual_drift.
    term : str
        Term to visualize.
    figsize : Tuple[int, int]
        Figure size.
    n_terms : int
        Number of emerging/fading terms to show per period.
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
    if term not in analysis.term_results:
        raise ValueError(f"Term '{term}' not found in analysis.")
    
    result = analysis.term_results[term]
    periods = result.periods[1:]  # Skip first period (no previous)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    # Emerging terms
    y_pos = np.arange(len(periods))
    
    for i, period in enumerate(periods):
        emerging = result.emerging_words.get(period, [])[:n_terms]
        text = ", ".join(emerging) if emerging else "(none)"
        ax.text(0.05, i, text, fontsize=10, va='center', ha='left',
                transform=ax.get_yaxis_transform())
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(periods)
    ax.set_xlim(0, 1)
    ax.set_title("Emerging Terms", fontsize=12, fontweight='bold', color="lightblue")
    ax.set_xlabel("")
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Fading terms
    for i, period in enumerate(periods):
        fading = result.fading_words.get(period, [])[:n_terms]
        text = ", ".join(fading) if fading else "(none)"
        ax.text(0.05, i, text, fontsize=10, va='center', ha='left',
                transform=ax.get_yaxis_transform())
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(periods)
    ax.set_xlim(0, 1)
    ax.set_title("Fading Terms", fontsize=12, fontweight='bold', color="lightblue")
    ax.set_xlabel("")
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    if title is None:
        title = f"Emerging and Fading Context for '{term}'"
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_drift_trajectory_2d(
    analysis: ConceptualDriftAnalysis,
    terms: Optional[List[str]] = None,
    method: str = "pca",  # "pca" or "tsne" or "umap"
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = None,
    show_arrows: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    Plot 2D trajectory of terms through semantic space over time.
    
    Reduces context profiles to 2D and shows how terms move.
    
    Parameters
    ----------
    analysis : ConceptualDriftAnalysis
        Results from compute_conceptual_drift.
    terms : List[str], optional
        Terms to plot.
    method : str
        Dimensionality reduction method.
    figsize : Tuple[int, int]
        Figure size.
    cmap : str
        Colormap for time encoding.
    show_arrows : bool
        Whether to show movement arrows.
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
    if terms is None:
        terms = list(analysis.term_results.keys())
    
    periods = analysis.periods
    vocabulary = analysis.global_vocabulary
    
    if not vocabulary:
        raise ValueError("No vocabulary available for trajectory plot.")
    
    # Build feature matrix: each row is a term-period combination
    vectors = []
    labels = []  # (term, period, period_index)
    
    for term in terms:
        if term not in analysis.term_results:
            continue
        result = analysis.term_results[term]
        
        for i, period in enumerate(periods):
            context = result.context_words.get(period, [])
            
            # Create vector from context words
            vec = np.zeros(len(vocabulary))
            for j, word in enumerate(context):
                if word in vocabulary:
                    idx = vocabulary.index(word)
                    vec[idx] = len(context) - j  # Weight by rank
            
            if vec.sum() > 0:  # Only include non-empty vectors
                vectors.append(vec)
                labels.append((term, period, i))
    
    if len(vectors) < 3:
        raise ValueError("Insufficient data for trajectory plot.")
    
    X = np.array(vectors)
    
    # Dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_2d = reducer.fit_transform(X)
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn required. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    # Color by time
    time_colors = "lightblue"
    term_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Group by term
    term_indices = {term: [] for term in terms}
    for i, (term, period, period_idx) in enumerate(labels):
        term_indices[term].append((i, period_idx))
    
    # Plot each term's trajectory
    for t_idx, term in enumerate(terms):
        if term not in term_indices or not term_indices[term]:
            continue
        
        indices = sorted(term_indices[term], key=lambda x: x[1])
        marker = term_markers[t_idx % len(term_markers)]
        
        for i, (idx, period_idx) in enumerate(indices):
            ax.scatter(
                X_2d[idx, 0], X_2d[idx, 1],
                c=["lightblue"],
                marker=marker, s=150, alpha=0.8,
                edgecolors='black', linewidths=1
            )
            
            # Draw arrows between consecutive points
            if show_arrows and i > 0:
                prev_idx = indices[i-1][0]
                ax.annotate(
                    '', xy=(X_2d[idx, 0], X_2d[idx, 1]),
                    xytext=(X_2d[prev_idx, 0], X_2d[prev_idx, 1]),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1.5)
                )
        
        # Label the term at its final position
        if indices:
            final_idx = indices[-1][0]
            ax.annotate(
                term, (X_2d[final_idx, 0], X_2d[final_idx, 1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
    
    # Add time colorbar
    sm = ScalarMappable(cmap=cmap or CMAP_CONTINUOUS, norm=Normalize(vmin=0, vmax=len(periods)-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_ticks(np.arange(len(periods)))
    cbar.set_ticklabels(periods)
    cbar.set_label("Time Period", fontsize=11)
    
    # Legend for term markers
    legend_elements = [
        Line2D([0], [0], marker=term_markers[i % len(term_markers)],
               color='w', markerfacecolor='gray', markersize=10,
               label=term)
        for i, term in enumerate(terms) if term in term_indices
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=12)
    ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=12)
    
    if title is None:
        title = "Conceptual Drift Trajectories in Semantic Space"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_drift_comparison_radar(
    analysis: ConceptualDriftAnalysis,
    terms: List[str],
    period: Optional[str] = None,
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    Radar plot comparing drift characteristics across terms.
    
    Parameters
    ----------
    analysis : ConceptualDriftAnalysis
        Results from compute_conceptual_drift.
    terms : List[str]
        Terms to compare.
    period : str, optional
        Specific period to analyze. If None, use averages.
    metrics : List[str], optional
        Metrics to include. Default: drift magnitude, volatility, etc.
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
    # Compute metrics for each term
    term_metrics = {}
    
    for term in terms:
        if term not in analysis.term_results:
            continue
        
        result = analysis.term_results[term]
        valid_drifts = [d for d in result.drift_scores if d is not None]
        
        if not valid_drifts:
            continue
        
        term_metrics[term] = {
            "Avg Drift": np.mean(valid_drifts),
            "Max Drift": np.max(valid_drifts),
            "Drift Volatility": np.std(valid_drifts),
            "Total Cumulative": result.cumulative_drift[-1] if result.cumulative_drift else 0,
            "Doc Coverage": np.mean(list(result.doc_counts.values())),
        }
    
    if not term_metrics:
        raise ValueError("No valid metrics computed for specified terms.")
    
    # Normalize metrics to [0, 1] for radar chart
    metric_names = list(list(term_metrics.values())[0].keys())
    
    for metric in metric_names:
        values = [term_metrics[t][metric] for t in term_metrics]
        max_val = max(values) if max(values) > 0 else 1
        for t in term_metrics:
            term_metrics[t][metric] /= max_val
    
    # Create radar chart
    num_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    ax.grid(False)
    
    colors = "lightblue"
    
    for i, (term, metrics_dict) in enumerate(term_metrics.items()):
        values = [metrics_dict[m] for m in metric_names]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, color="lightblue", label=term)
        ax.fill(angles, values, alpha=0.1, color="lightblue")
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    if title is None:
        title = "Drift Characteristics Comparison"
    ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_drift_velocity(
    analysis: ConceptualDriftAnalysis,
    terms: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    Plot drift velocity (rate of change) and acceleration over time.
    
    Parameters
    ----------
    analysis : ConceptualDriftAnalysis
        Results from compute_conceptual_drift.
    terms : List[str], optional
        Terms to plot.
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
    if terms is None:
        terms = list(analysis.term_results.keys())[:5]  # Limit for clarity
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    periods = analysis.periods
    x_positions = np.arange(len(periods))
    
    colors = "lightblue"
    
    for i, term in enumerate(terms):
        if term not in analysis.term_results:
            continue
        
        result = analysis.term_results[term]
        drifts = result.drift_scores
        
        # Replace None with NaN
        drifts = np.array([d if d is not None else np.nan for d in drifts])
        
        # Velocity (first derivative) - drift itself is already "velocity"
        ax.plot(x_positions, drifts, marker='o', linewidth=2,
                color="lightblue", label=term, alpha=0.8)
        
        # Acceleration (second derivative)
        acceleration = np.diff(np.nan_to_num(drifts, nan=0))
        accel_x = x_positions[1:]
        ax.plot(accel_x, acceleration, marker='s', linewidth=2,
                color="lightblue", label=term, alpha=0.8)
    
    # Configure velocity plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.set_xlabel("Time Period", fontsize=11)
    ax.set_ylabel("Drift Magnitude", fontsize=11)
    ax.set_title("Drift Velocity", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax
    
    # Configure acceleration plot
    ax.set_xticks(x_positions[1:])
    ax.set_xticklabels(periods[1:], rotation=45, ha='right')
    ax.set_xlabel("Time Period", fontsize=11)
    ax.set_ylabel("Drift Acceleration", fontsize=11)
    ax.set_title("Drift Acceleration (Speeding Up / Slowing Down)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax
    
    if title is None:
        title = "Drift Dynamics Analysis"
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

def plot_document_coverage(
    analysis: ConceptualDriftAnalysis,
    terms: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
    stacked: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    Plot document counts for terms over time.
    
    Parameters
    ----------
    analysis : ConceptualDriftAnalysis
        Results from compute_conceptual_drift.
    terms : List[str], optional
        Terms to plot.
    figsize : Tuple[int, int]
        Figure size.
    stacked : bool
        Whether to use stacked area chart.
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
    if terms is None:
        terms = list(analysis.term_results.keys())
    
    periods = analysis.periods
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    if stacked:
        # Prepare data for stacked area
        data = []
        for term in terms:
            if term in analysis.term_results:
                counts = [analysis.term_results[term].doc_counts.get(p, 0) for p in periods]
                data.append(counts)
        
        ax.stackplot(range(len(periods)), data, labels=terms, alpha=0.7)
    else:
        colors = "lightblue"
        
        for i, term in enumerate(terms):
            if term not in analysis.term_results:
                continue
            
            result = analysis.term_results[term]
            counts = [result.doc_counts.get(p, 0) for p in periods]
            
            ax.plot(range(len(periods)), counts, marker='o', linewidth=2,
                   color="lightblue", label=term, alpha=0.8)
    
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Document Count", fontsize=12)
    
    if title is None:
        title = "Term Occurrence Over Time"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}.(png/svg/pdf)")
    
    return fig

# =============================================================================
# INTEGRATION WITH BIBSTATS CLASS
# =============================================================================

def add_drift_methods_to_class(cls):
    """
    Decorator/function to add conceptual drift methods to BiblioStats class.
    
    Usage:
        from conceptual_drift import add_drift_methods_to_class
        add_drift_methods_to_class(BiblioAnalysis)
    
    Or as decorator:
        @add_drift_methods_to_class
        class BiblioAnalysis(BiblioStats):
            pass
    """
    
    def compute_conceptual_drift_method(
        self,
        target_terms: List[str],
        text_col: str = None,
        year_col: str = "Year",
        window_size: int = 5,
        method: str = "jensen_shannon",
        top_n_context: int = 100,
        min_docs_per_window: int = 50,
        save_results: bool = True,
        **kwargs
    ) -> ConceptualDriftAnalysis:
        """
        Compute conceptual drift for specified terms.
        
        Parameters
        ----------
        target_terms : List[str]
            Terms to analyze.
        text_col : str, optional
            Text column to use. Auto-detected if None.
        year_col : str
            Year column name.
        window_size : int
            Time window size in years.
        method : str
            Drift computation method.
        top_n_context : int
            Number of context words to track.
        min_docs_per_window : int
            Minimum documents per window.
        save_results : bool
            Whether to save results to Excel.
        **kwargs
            Additional arguments passed to compute_conceptual_drift.
        
        Returns
        -------
        ConceptualDriftAnalysis
        """
        # Auto-detect text column
        if text_col is None:
            for col in ["Processed Abstract", "Processed Combined Text", "Abstract", "Title"]:
                if col in self.df.columns:
                    text_col = col
                    break
        
        if text_col is None:
            raise ValueError("No suitable text column found.")
        
        # Compute drift
        self.drift_analysis = compute_conceptual_drift(
            self.df,
            target_terms,
            text_col=text_col,
            year_col=year_col,
            window_size=window_size,
            method=method,
            top_n_context=top_n_context,
            min_docs_per_window=min_docs_per_window,
            **kwargs
        )
        
        # Store summary DataFrame
        self.drift_summary_df = self.drift_analysis.get_summary_df()
        
        # Save to Excel if requested
        if save_results and hasattr(self, 'res_folder') and self.res_folder:
            output_path = os.path.join(self.res_folder, "tables", "conceptual_drift.xlsx")
            self.drift_summary_df.to_excel(output_path, index=False)
            print(f"Drift analysis saved to {output_path}")
        
        return self.drift_analysis
    
    def plot_conceptual_drift(
        self,
        plot_type: str = "timeline",
        terms: Optional[List[str]] = None,
        save: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Create conceptual drift visualizations.
        
        Parameters
        ----------
        plot_type : str
            Type of plot: "timeline", "heatmap", "context", "trajectory",
            "emerging_fading", "radar", "velocity", "coverage".
        terms : List[str], optional
            Terms to include.
        save : bool
            Whether to save plot to results folder.
        **kwargs
            Additional arguments for specific plot functions.
        
        Returns
        -------
        matplotlib Figure.
        """
        if not hasattr(self, 'drift_analysis'):
            raise ValueError("Run compute_conceptual_drift first.")
        
        # Determine save path
        save_path = None
        if save and hasattr(self, 'res_folder') and self.res_folder:
            save_path = os.path.join(self.res_folder, f"drift_{plot_type}")
        
        plot_functions = {
            "timeline": plot_drift_timeline,
            "heatmap": plot_drift_heatmap,
            "context": plot_context_evolution,
            "trajectory": plot_drift_trajectory_2d,
            "emerging_fading": plot_emerging_fading_terms,
            "radar": plot_drift_comparison_radar,
            "velocity": plot_drift_velocity,
            "coverage": plot_document_coverage,
        }
        
        if plot_type not in plot_functions:
            raise ValueError(f"Unknown plot_type: {plot_type}. Choose from: {list(plot_functions.keys())}")
        
        plot_func = plot_functions[plot_type]
        
        # Handle single-term plots
        if plot_type in ["context", "emerging_fading"]:
            if terms is None or len(terms) == 0:
                terms = [list(self.drift_analysis.term_results.keys())[0]]
            return plot_func(self.drift_analysis, terms[0], save_path=save_path, **kwargs)
        
        return plot_func(self.drift_analysis, terms=terms, save_path=save_path, **kwargs)
    
    # Attach methods to class
    cls.compute_conceptual_drift = compute_conceptual_drift_method
    cls.plot_conceptual_drift = plot_conceptual_drift
    
    return cls

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def classify_drift_pattern(
    drift_scores: List[float],
    cumulative_drift: List[float],
) -> str:
    """
    Classify the drift pattern of a term.
    
    Returns one of:
    - "stable": Low drift throughout
    - "gradual_evolution": Steady, moderate drift
    - "punctuated_equilibrium": Long stability then sudden shift
    - "volatile": High variance in drift
    - "accelerating": Drift rate increasing
    - "decelerating": Drift rate decreasing
    """
    valid_scores = [s for s in drift_scores if s is not None]
    
    if not valid_scores:
        return "unknown"
    
    mean_drift = np.mean(valid_scores)
    std_drift = np.std(valid_scores)
    max_drift = np.max(valid_scores)
    
    # Thresholds
    if mean_drift < 0.1:
        return "stable"
    
    if std_drift > 0.2:
        return "volatile"
    
    if max_drift > 0.5 and std_drift > 0.15:
        return "punctuated_equilibrium"
    
    # Check for acceleration/deceleration
    if len(valid_scores) >= 3:
        first_half = np.mean(valid_scores[:len(valid_scores)//2])
        second_half = np.mean(valid_scores[len(valid_scores)//2:])
        
        if second_half > first_half * 1.5:
            return "accelerating"
        elif second_half < first_half * 0.5:
            return "decelerating"
    
    return "gradual_evolution"

def find_drift_events(
    analysis: ConceptualDriftAnalysis,
    threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Identify significant drift events (sudden changes).
    
    Parameters
    ----------
    analysis : ConceptualDriftAnalysis
        Drift analysis results.
    threshold : float
        Minimum drift score to count as an event.
    
    Returns
    -------
    DataFrame with columns: Term, Period, Drift_Score, Top_Emerging, Top_Fading.
    """
    events = []
    
    for term, result in analysis.term_results.items():
        for i, period in enumerate(result.periods[1:], 1):
            drift = result.drift_scores[i]
            
            if drift is not None and drift >= threshold:
                events.append({
                    "Term": term,
                    "Period": period,
                    "Drift Score": drift,
                    "Top Emerging": "; ".join(result.emerging_words.get(period, [])[:3]),
                    "Top Fading": "; ".join(result.fading_words.get(period, [])[:3]),
                })
    
    return pd.DataFrame(events).sort_values("Drift Score", ascending=False)

def compare_drift_across_fields(
    df: pd.DataFrame,
    target_terms: List[str],
    field_col: str,
    text_col: str = "Processed Abstract",
    year_col: str = "Year",
    window_size: int = 5,
) -> Dict[str, ConceptualDriftAnalysis]:
    """
    Compare how terms drift differently across research fields.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_terms : List[str]
        Terms to analyze.
    field_col : str
        Column containing field/discipline classification.
    text_col : str
        Text column.
    year_col : str
        Year column.
    window_size : int
        Time window size.
    
    Returns
    -------
    Dict mapping field name to ConceptualDriftAnalysis.
    """
    results = {}
    
    for field in df[field_col].dropna().unique():
        field_df = df[df[field_col] == field]
        
        if len(field_df) < 100:
            continue
        
        try:
            analysis = compute_conceptual_drift(
                field_df,
                target_terms,
                text_col=text_col,
                year_col=year_col,
                window_size=window_size,
                min_docs_per_window=20,
                min_term_occurrences=5,
            )
            results[field] = analysis
        except Exception as e:
            print(f"Could not analyze field '{field}': {e}")
    
    return results


def export_drift_results(
    analysis: ConceptualDriftAnalysis,
    output_dir: str,
) -> Dict[str, str]:
    """Export drift analysis results to files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # Export drift matrix
    drift_matrix = analysis.get_drift_matrix()
    csv_path = os.path.join(output_dir, "drift_matrix.csv")
    drift_matrix.to_csv(csv_path)
    paths["drift_matrix"] = csv_path
    
    # Export summary
    summary_df = analysis.get_summary_df()
    if len(summary_df) > 0:
        summary_path = os.path.join(output_dir, "drift_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        paths["summary"] = summary_path
    
    # Export high drift terms
    high_drift = analysis.get_high_drift_terms()
    if high_drift:
        high_drift_path = os.path.join(output_dir, "high_drift_terms.csv")
        pd.DataFrame({"term": high_drift}).to_csv(high_drift_path, index=False)
        paths["high_drift"] = high_drift_path
    
    print(f"Exported drift results to: {output_dir}")
    return paths


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import biblium as bb
    ba = bb.BiblioAnalysis(f_name="data\\open alex dataset.csv", db="oa")
    
    print("Conceptual Drift Analysis")
    print("=" * 60)
    
    # Run analysis
    result = compute_conceptual_drift(
        df=ba.df,
        target_terms=["machine learning", "deep learning", "neural network"],
        text_col="Abstract",
        year_col="Year",
        window_size=3,
    )
    
    # Print summary
    print(f"\nAnalyzed {len(result.periods)} time periods")
    print(f"Terms tracked: {len(result.term_results)}")
    
    # Visualizations
    print("\nGenerating plots...")
    plot_drift_heatmap(result, save_path="results/drift_heatmap")
    plot_drift_timeline(result, save_path="results/drift_timeline")
    
    # Export
    print("\nExporting results...")
    export_drift_results(result, "results")
    
    print("\nDone!")
