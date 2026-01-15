# -*- coding: utf-8 -*-
"""
Impact & Influence Metrics Module for Bibliometric Analysis

This module provides advanced metrics for measuring scientific impact and influence
beyond simple citation counts. It implements both established and cutting-edge
bibliometric indicators.

Metrics implemented:
1. Disruption Index (CD Index) - Measures consolidating vs disruptive papers
2. Novelty/Atypicality Measures - Unusual keyword/reference combinations
3. Citation Pattern Classifications - Evergreens, Flash-in-pan, Heartbeat patterns
4. Field-Normalized Metrics - FWCI, percentile rankings, SNIP-style normalization
5. Citation Velocity & Momentum - First/second derivatives of citation curves
6. Reference Diversity Metrics - How diverse are a paper's references?
7. Interdisciplinary Impact - Cross-field citation flows
8. Knowledge Consolidation vs Disruption spectrum

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
from scipy.stats import percentileofscore, zscore, spearmanr, pearsonr, entropy
from scipy.spatial.distance import cosine, jensenshannon
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
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
    from scipy.stats import powerlaw, lognorm, expon
    SCIPY_DISTRIBUTIONS = True
except ImportError:
    SCIPY_DISTRIBUTIONS = False

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
class DisruptionMetrics:
    """Container for disruption-related metrics of a paper."""
    doc_id: Any
    cd_index: float  # Core disruption index [-1, 1]
    cd5_index: Optional[float]  # 5-year disruption
    di_index: float  # Disruption index (simplified)
    n_citing: int  # Papers citing focal paper
    n_i: int  # Citing papers that cite focal but not references
    n_j: int  # Citing papers that cite both focal and references
    n_k: int  # Citing papers that cite references but not focal
    interpretation: str  # "consolidating", "neutral", "disruptive"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'cd_index': self.cd_index,
            'cd5_index': self.cd5_index,
            'di_index': self.di_index,
            'n_citing': self.n_citing,
            'n_i': self.n_i,
            'n_j': self.n_j,
            'n_k': self.n_k,
            'interpretation': self.interpretation,
        }

@dataclass
class NoveltyMetrics:
    """Container for novelty/atypicality metrics of a paper."""
    doc_id: Any
    combinatorial_novelty: float  # Uzzi-style: unusual combinations
    atypicality_score: float  # How atypical are the references/keywords
    conventionality_score: float  # How conventional (opposite of novelty)
    novelty_percentile: float  # Percentile within cohort
    novel_combinations: List[Tuple[str, str, float]]  # Most unusual pairs
    interpretation: str  # "highly_novel", "novel", "typical", "conventional"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'combinatorial_novelty': self.combinatorial_novelty,
            'atypicality_score': self.atypicality_score,
            'conventionality_score': self.conventionality_score,
            'novelty_percentile': self.novelty_percentile,
            'top_novel_combinations': self.novel_combinations[:5],
            'interpretation': self.interpretation,
        }

@dataclass
class CitationPattern:
    """Classification of citation accumulation pattern."""
    doc_id: Any
    pattern_type: str  # "sleeping_beauty", "evergreen", "flash_in_pan", "heartbeat", "delayed", "standard"
    peak_year: Optional[int]
    peak_citations: int
    years_to_peak: Optional[int]
    decay_rate: Optional[float]  # Exponential decay rate after peak
    resurgence_count: int  # Number of citation resurgences
    citation_half_life: Optional[float]  # Years to accumulate half of citations
    velocity_at_peak: float  # Citation rate at peak
    current_trend: str  # "increasing", "decreasing", "stable", "dormant"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'pattern_type': self.pattern_type,
            'peak_year': self.peak_year,
            'peak_citations': self.peak_citations,
            'years_to_peak': self.years_to_peak,
            'decay_rate': self.decay_rate,
            'resurgence_count': self.resurgence_count,
            'citation_half_life': self.citation_half_life,
            'velocity_at_peak': self.velocity_at_peak,
            'current_trend': self.current_trend,
        }

@dataclass
class FieldNormalizedMetrics:
    """Container for field-normalized impact metrics."""
    doc_id: Any
    raw_citations: int
    expected_citations: float  # Expected for field-year
    fwci: float  # Field-Weighted Citation Impact
    percentile_in_field: float  # Percentile within field
    percentile_in_year: float  # Percentile within publication year
    percentile_in_field_year: float  # Percentile within field-year cohort
    z_score: float  # Standard deviations from mean
    is_highly_cited: bool  # Top 10%
    is_top_1_percent: bool
    citation_class: str  # "uncited", "low", "average", "high", "exceptional"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'raw_citations': self.raw_citations,
            'expected_citations': self.expected_citations,
            'fwci': self.fwci,
            'percentile_in_field': self.percentile_in_field,
            'percentile_in_year': self.percentile_in_year,
            'percentile_in_field_year': self.percentile_in_field_year,
            'z_score': self.z_score,
            'is_highly_cited': self.is_highly_cited,
            'is_top_1_percent': self.is_top_1_percent,
            'citation_class': self.citation_class,
        }

@dataclass
class ImpactAnalysisResult:
    """Container for complete impact analysis results."""
    disruption_metrics: pd.DataFrame
    novelty_metrics: pd.DataFrame
    citation_patterns: pd.DataFrame
    field_normalized: pd.DataFrame
    velocity_metrics: pd.DataFrame
    summary_statistics: Dict[str, Any]
    parameters: Dict[str, Any]
    
    def get_disruptive_papers(self, threshold: float = 0.5) -> pd.DataFrame:
        """Get papers with high disruption index."""
        if self.disruption_metrics is None or len(self.disruption_metrics) == 0:
            print("Warning: No disruption metrics available. Run with compute_disruption=True.")
            return pd.DataFrame()
        if 'cd_index' not in self.disruption_metrics.columns:
            print("Warning: cd_index column not found.")
            return pd.DataFrame()
        return self.disruption_metrics[self.disruption_metrics['cd_index'] >= threshold]
    
    def get_consolidating_papers(self, threshold: float = -0.5) -> pd.DataFrame:
        """Get papers that consolidate existing knowledge."""
        if self.disruption_metrics is None or len(self.disruption_metrics) == 0:
            print("Warning: No disruption metrics available. Run with compute_disruption=True.")
            return pd.DataFrame()
        if 'cd_index' not in self.disruption_metrics.columns:
            print("Warning: cd_index column not found.")
            return pd.DataFrame()
        return self.disruption_metrics[self.disruption_metrics['cd_index'] <= threshold]
    
    def get_novel_papers(self, percentile: float = 90) -> pd.DataFrame:
        """Get highly novel papers."""
        if self.novelty_metrics is None or len(self.novelty_metrics) == 0:
            print("Warning: No novelty metrics available. Run with compute_novelty=True.")
            return pd.DataFrame()
        if 'novelty_percentile' not in self.novelty_metrics.columns:
            print("Warning: novelty_percentile column not found. Novelty computation may have failed.")
            return pd.DataFrame()
        return self.novelty_metrics[self.novelty_metrics['novelty_percentile'] >= percentile]
    
    def get_highly_cited(self, percentile: float = 90) -> pd.DataFrame:
        """Get highly cited papers by field-normalized metrics."""
        if self.field_normalized is None or len(self.field_normalized) == 0:
            print("Warning: No field-normalized metrics available. Run with compute_normalized=True.")
            return pd.DataFrame()
        if 'percentile_in_field_year' not in self.field_normalized.columns:
            print("Warning: percentile_in_field_year column not found.")
            return pd.DataFrame()
        return self.field_normalized[self.field_normalized['percentile_in_field_year'] >= percentile]
    
    def to_summary_df(self) -> pd.DataFrame:
        """Merge all metrics into a single DataFrame."""
        df = self.disruption_metrics[['doc_id', 'cd_index', 'interpretation']].copy()
        df = df.rename(columns={'interpretation': 'disruption_type'})
        
        if 'doc_id' in self.novelty_metrics.columns:
            novelty_cols = ['doc_id', 'combinatorial_novelty', 'novelty_percentile']
            df = df.merge(self.novelty_metrics[novelty_cols], on='doc_id', how='left')
        
        if 'doc_id' in self.field_normalized.columns:
            fn_cols = ['doc_id', 'fwci', 'percentile_in_field_year', 'citation_class']
            df = df.merge(self.field_normalized[fn_cols], on='doc_id', how='left')
        
        if 'doc_id' in self.citation_patterns.columns:
            cp_cols = ['doc_id', 'pattern_type', 'current_trend']
            df = df.merge(self.citation_patterns[cp_cols], on='doc_id', how='left')
        
        return df

# =============================================================================
# DISRUPTION INDEX (CD INDEX)
# =============================================================================

def compute_disruption_index(
    df: pd.DataFrame,
    citation_network: Dict[str, Set[str]] = None,
    id_col: str = "unique-id",
    refs_col: str = "References",
    citations_col: str = "Cited by",
    sep: str = "; ",
    use_cd5: bool = True,
    year_col: str = "Year",
    verbose: bool = True) -> pd.DataFrame:
    """
    Compute the Disruption Index (CD Index) for papers.
    
    The CD Index measures whether a paper consolidates or disrupts existing knowledge:
    - CD = +1: Fully disruptive (citers ignore the paper's references)
    - CD = 0: Neutral
    - CD = -1: Fully consolidating (citers always cite the paper's references)
    
    Formula: CD = (n_i - n_j) / (n_i + n_j + n_k)
    where:
    - n_i = citing papers that cite focal but NOT its references
    - n_j = citing papers that cite BOTH focal AND its references
    - n_k = citing papers that cite references but NOT focal
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    citation_network : Dict[str, Set[str]], optional
        Pre-computed citation network {paper_id: set of papers it cites}.
        If None, built from refs_col.
    id_col : str
        Column with document IDs.
    refs_col : str
        Column with references.
    citations_col : str
        Column with citing paper IDs or count.
    sep : str
        Separator for parsing references.
    use_cd5 : bool
        Also compute 5-year disruption index.
    year_col : str
        Year column for CD5 calculation.
    verbose : bool
        Print progress.
    
    Returns
    -------
    pd.DataFrame with disruption metrics.
    """
    if verbose:
        print("Computing Disruption Index (CD Index)...")
    
    # Build citation network if not provided
    if citation_network is None:
        if verbose:
            print("  Building citation network...")
        citation_network = build_citation_network(df, id_col, refs_col, sep)
    
    # Build reverse citation network (who cites whom)
    if verbose:
        print("  Building reverse citation index...")
    cited_by = defaultdict(set)
    for paper, refs in citation_network.items():
        for ref in refs:
            cited_by[ref].add(paper)
    
    # Get all paper IDs
    all_papers = set(df[id_col].dropna().astype(str))
    
    # Diagnostic: Check how many internal citations exist
    internal_citations = sum(1 for paper in all_papers if paper in cited_by and cited_by[paper] & all_papers)
    if verbose:
        print(f"  Papers in dataset: {len(all_papers)}")
        print(f"  Papers with internal citations: {internal_citations}")
        if internal_citations < len(all_papers) * 0.1:
            print("  WARNING: Very few internal citations detected!")
            print("    CD Index requires papers to cite each other within the dataset.")
            print("    Most papers will have CD Index = NaN (marked as 'uncited').")
            print("    This is normal for focused datasets where papers cite external sources.")
    
    # Compute metrics for each paper
    results = []
    
    for idx, row in df.iterrows():
        doc_id = str(row.get(id_col, ""))
        if not doc_id:
            continue
        
        # Get focal paper's references
        focal_refs = citation_network.get(doc_id, set())
        
        # Get papers that cite the focal paper
        citing_papers = cited_by.get(doc_id, set())
        
        if not citing_papers:
            # No citations - cannot compute disruption
            results.append({
                'doc_id': doc_id,
                'cd_index': np.nan,
                'cd5_index': np.nan,
                'di_index': np.nan,
                'n_citing': 0,
                'n_i': 0,
                'n_j': 0,
                'n_k': 0,
                'interpretation': 'uncited',
            })
            continue
        
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
        n_k = 0
        for ref in focal_refs:
            ref_citers = cited_by.get(ref, set())
            for ref_citer in ref_citers:
                if ref_citer not in citing_papers and ref_citer != doc_id:
                    n_k += 1
        
        # Avoid double counting n_k
        n_k = len({citer for ref in focal_refs 
                   for citer in cited_by.get(ref, set()) 
                   if citer not in citing_papers and citer != doc_id})
        
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
        
        results.append({
            'doc_id': doc_id,
            'cd_index': round(cd_index, 4),
            'cd5_index': np.nan,  # TODO: implement time-windowed version
            'di_index': round(di_index, 4),
            'n_citing': len(citing_papers),
            'n_i': n_i,
            'n_j': n_j,
            'n_k': n_k,
            'interpretation': interpretation,
        })
    
    if verbose:
        print(f"  Computed disruption index for {len(results)} papers")
    
    return pd.DataFrame(results)

def build_citation_network(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    refs_col: str = "References",
    sep: str = "; ") -> Dict[str, Set[str]]:
    """
    Build citation network from dataframe.
    
    Returns dict mapping paper_id -> set of paper_ids it cites.
    """
    network = {}
    
    for _, row in df.iterrows():
        doc_id = str(row.get(id_col, ""))
        refs = row.get(refs_col)
        
        if not doc_id:
            continue
        
        if pd.isna(refs):
            network[doc_id] = set()
            continue
        
        # Parse references
        if isinstance(refs, str):
            ref_list = [r.strip() for r in refs.split(sep) if r.strip()]
        elif isinstance(refs, (list, tuple)):
            ref_list = [str(r).strip() for r in refs if r]
        else:
            ref_list = []
        
        network[doc_id] = set(ref_list)
    
    return network

# =============================================================================
# NOVELTY / ATYPICALITY MEASURES
# =============================================================================

def compute_novelty_metrics(
    df: pd.DataFrame,
    combination_col: str = "References",
    id_col: str = "unique-id",
    year_col: str = "Year",
    sep: str = "; ",
    baseline_years: int = 5,
    min_combinations: int = 3,
    verbose: bool = True) -> pd.DataFrame:
    """
    Compute novelty/atypicality metrics based on unusual combinations.
    
    Implements Uzzi et al. style combinatorial novelty:
    - Measures how unusual the pairwise combinations in a paper are
    - Compares to baseline co-occurrence frequencies
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    combination_col : str
        Column containing items to analyze (references, keywords, etc.).
    id_col : str
        Document ID column.
    year_col : str
        Publication year column.
    sep : str
        Separator for parsing items.
    baseline_years : int
        Years of prior data to use as baseline.
    min_combinations : int
        Minimum number of pairs needed to compute novelty.
    verbose : bool
        Print progress.
    
    Returns
    -------
    pd.DataFrame with novelty metrics.
    """
    if verbose:
        print("Computing Novelty/Atypicality metrics...")
    
    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    
    # Parse items for each document
    if verbose:
        print("  Parsing item combinations...")
    
    doc_items = {}
    for _, row in df.iterrows():
        doc_id = str(row.get(id_col, ""))
        items = row.get(combination_col)
        year = row.get(year_col)
        
        if not doc_id or pd.isna(items):
            continue
        
        if isinstance(items, str):
            item_list = [i.strip() for i in items.split(sep) if i.strip()]
        elif isinstance(items, (list, tuple)):
            item_list = [str(i).strip() for i in items if i]
        else:
            continue
        
        if len(item_list) >= 2:
            doc_items[doc_id] = {
                'items': item_list,
                'year': year,
            }
    
    if verbose:
        print(f"  Parsed {len(doc_items)} documents with valid items")
    
    # Build baseline co-occurrence frequencies
    if verbose:
        print("  Building co-occurrence baseline...")
    
    cooc_counts = Counter()
    item_counts = Counter()
    total_pairs = 0
    
    # Count all co-occurrences
    for doc_id, data in doc_items.items():
        items = data['items']
        for item in items:
            item_counts[item] += 1
        
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                pair = tuple(sorted([item1, item2]))
                cooc_counts[pair] += 1
                total_pairs += 1
    
    if total_pairs == 0:
        if verbose:
            print("  Warning: No co-occurrences found")
        return pd.DataFrame()
    
    # Compute expected co-occurrence under independence
    total_items = sum(item_counts.values())
    
    def expected_cooccurrence(item1: str, item2: str) -> float:
        """Expected co-occurrence under independence assumption."""
        p1 = item_counts.get(item1, 0) / total_items
        p2 = item_counts.get(item2, 0) / total_items
        return p1 * p2 * total_pairs
    
    # Compute novelty for each document
    if verbose:
        print("  Computing novelty scores...")
    
    results = []
    all_novelty_scores = []
    
    for doc_id, data in doc_items.items():
        items = data['items']
        year = data['year']
        
        # Get all pairs
        pairs = list(combinations(items, 2))
        
        if len(pairs) < min_combinations:
            results.append({
                'doc_id': doc_id,
                'combinatorial_novelty': np.nan,
                'atypicality_score': np.nan,
                'conventionality_score': np.nan,
                'novelty_percentile': np.nan,
                'novel_combinations': [],
                'interpretation': 'insufficient_data',
            })
            continue
        
        # Compute novelty for each pair
        pair_novelties = []
        novel_combinations = []
        
        for item1, item2 in pairs:
            pair = tuple(sorted([item1, item2]))
            observed = cooc_counts.get(pair, 0)
            expected = expected_cooccurrence(item1, item2)
            
            if expected > 0:
                # Log ratio (negative = novel, positive = conventional)
                ratio = observed / (expected + 1)  # +1 smoothing
                novelty = -np.log(ratio + 0.01)  # Higher = more novel
            else:
                novelty = 5.0  # Very novel if expected is 0
            
            pair_novelties.append(novelty)
            novel_combinations.append((item1, item2, novelty))
        
        # Aggregate scores
        # Use 10th percentile (most novel combinations)
        combinatorial_novelty = np.percentile(pair_novelties, 90) if pair_novelties else 0
        atypicality_score = np.mean(pair_novelties) if pair_novelties else 0
        conventionality_score = np.percentile(pair_novelties, 10) if pair_novelties else 0
        
        # Sort novel combinations by novelty
        novel_combinations.sort(key=lambda x: -x[2])
        
        all_novelty_scores.append(combinatorial_novelty)
        
        results.append({
            'doc_id': doc_id,
            'combinatorial_novelty': round(combinatorial_novelty, 4),
            'atypicality_score': round(atypicality_score, 4),
            'conventionality_score': round(conventionality_score, 4),
            'novelty_percentile': 0,  # Filled in next step
            'novel_combinations': novel_combinations[:10],
            'interpretation': '',  # Filled in next step
        })
    
    # Compute percentiles
    if all_novelty_scores:
        for result in results:
            if not np.isnan(result['combinatorial_novelty']):
                result['novelty_percentile'] = round(
                    percentileofscore(all_novelty_scores, result['combinatorial_novelty']), 1
                )
                
                # Interpret
                pct = result['novelty_percentile']
                if pct >= 90:
                    result['interpretation'] = 'highly_novel'
                elif pct >= 75:
                    result['interpretation'] = 'novel'
                elif pct >= 25:
                    result['interpretation'] = 'typical'
                else:
                    result['interpretation'] = 'conventional'
    
    if verbose:
        print(f"  Computed novelty for {len(results)} papers")
    
    return pd.DataFrame(results)

# =============================================================================
# CITATION PATTERN CLASSIFICATION
# =============================================================================

def classify_citation_patterns(
    df: pd.DataFrame,
    citation_history_col: str = None,
    citations_col: str = "Cited by",
    year_col: str = "Year",
    id_col: str = "unique-id",
    current_year: int = None,
    min_years: int = 5,
    verbose: bool = True) -> pd.DataFrame:
    """
    Classify papers by their citation accumulation pattern.
    
    Patterns detected:
    - sleeping_beauty: Long dormancy followed by awakening
    - evergreen: Steady citation accumulation over time
    - flash_in_pan: Quick spike then rapid decay
    - heartbeat: Periodic resurgences
    - delayed: Slow start but eventually successful
    - standard: Typical peak-and-decay pattern
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    citation_history_col : str, optional
        Column with citation counts per year (dict or list).
    citations_col : str
        Total citations column.
    year_col : str
        Publication year column.
    id_col : str
        Document ID column.
    current_year : int, optional
        Current year for calculations.
    min_years : int
        Minimum years since publication to classify.
    verbose : bool
        Print progress.
    
    Returns
    -------
    pd.DataFrame with citation pattern classifications.
    """
    if verbose:
        print("Classifying citation patterns...")
    
    if current_year is None:
        current_year = pd.Timestamp.now().year
    
    results = []
    
    for _, row in df.iterrows():
        doc_id = str(row.get(id_col, ""))
        pub_year = row.get(year_col)
        total_citations = row.get(citations_col, 0)
        
        if not doc_id or pd.isna(pub_year):
            continue
        
        pub_year = int(pub_year)
        years_since_pub = current_year - pub_year
        
        if years_since_pub < min_years:
            results.append({
                'doc_id': doc_id,
                'pattern_type': 'too_recent',
                'peak_year': None,
                'peak_citations': 0,
                'years_to_peak': None,
                'decay_rate': None,
                'resurgence_count': 0,
                'citation_half_life': None,
                'velocity_at_peak': 0,
                'current_trend': 'unknown',
            })
            continue
        
        # Try to get citation history
        citation_history = None
        if citation_history_col and citation_history_col in df.columns:
            hist = row.get(citation_history_col)
            if isinstance(hist, dict):
                citation_history = hist
            elif isinstance(hist, (list, tuple)):
                # Assume list from pub_year onwards
                citation_history = {pub_year + i: c for i, c in enumerate(hist)}
        
        # If no history, create synthetic based on total and typical patterns
        if citation_history is None:
            citation_history = _synthesize_citation_history(
                total_citations, pub_year, current_year
            )
        
        # Analyze the citation history
        pattern = _analyze_citation_history(citation_history, pub_year, current_year)
        pattern['doc_id'] = doc_id
        
        results.append(pattern)
    
    if verbose:
        print(f"  Classified {len(results)} papers")
    
    return pd.DataFrame(results)

def _synthesize_citation_history(
    total_citations: int,
    pub_year: int,
    current_year: int) -> Dict[int, int]:
    """
    Synthesize a plausible citation history when actual data is unavailable.
    Uses typical citation accumulation pattern (log-normal like).
    """
    years = list(range(pub_year, current_year + 1))
    n_years = len(years)
    
    if n_years == 0 or total_citations == 0:
        return {y: 0 for y in years}
    
    # Typical pattern: peak around year 3-5, then decay
    peak_year_offset = min(3, n_years - 1)
    
    weights = []
    for i, year in enumerate(years):
        offset = i - peak_year_offset
        if offset <= 0:
            # Rising phase
            w = np.exp(-0.5 * (offset / 2) ** 2)
        else:
            # Decay phase
            w = np.exp(-0.3 * offset)
        weights.append(w)
    
    # Normalize and distribute citations
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    citations = np.round(weights * total_citations).astype(int)
    
    # Adjust to match total
    diff = total_citations - citations.sum()
    if diff != 0:
        citations[np.argmax(weights)] += diff
    
    return {year: int(c) for year, c in zip(years, citations)}

def _analyze_citation_history(
    citation_history: Dict[int, int],
    pub_year: int,
    current_year: int) -> Dict[str, Any]:
    """Analyze citation history to determine pattern type."""
    years = sorted(citation_history.keys())
    citations = [citation_history.get(y, 0) for y in years]
    
    if not citations or sum(citations) == 0:
        return {
            'pattern_type': 'uncited',
            'peak_year': None,
            'peak_citations': 0,
            'years_to_peak': None,
            'decay_rate': None,
            'resurgence_count': 0,
            'citation_half_life': None,
            'velocity_at_peak': 0,
            'current_trend': 'dormant',
        }
    
    # Find peak
    peak_idx = np.argmax(citations)
    peak_year = years[peak_idx]
    peak_citations = citations[peak_idx]
    years_to_peak = peak_year - pub_year
    
    # Citation velocity (first derivative)
    velocities = np.diff(citations) if len(citations) > 1 else [0]
    velocity_at_peak = velocities[peak_idx - 1] if peak_idx > 0 else citations[0]
    
    # Current trend (last 3 years)
    recent_citations = citations[-3:] if len(citations) >= 3 else citations
    if len(recent_citations) >= 2:
        recent_trend = np.polyfit(range(len(recent_citations)), recent_citations, 1)[0]
        if recent_trend > 0.5:
            current_trend = 'increasing'
        elif recent_trend < -0.5:
            current_trend = 'decreasing'
        elif np.mean(recent_citations) < 1:
            current_trend = 'dormant'
        else:
            current_trend = 'stable'
    else:
        current_trend = 'unknown'
    
    # Decay rate (after peak)
    decay_rate = None
    if peak_idx < len(citations) - 2:
        post_peak = citations[peak_idx:]
        if post_peak[0] > 0:
            try:
                x = np.arange(len(post_peak))
                log_citations = np.log(np.array(post_peak) + 1)
                decay_rate = -np.polyfit(x, log_citations, 1)[0]
            except:
                pass
    
    # Citation half-life
    total = sum(citations)
    cumsum = np.cumsum(citations)
    half_life = None
    for i, cs in enumerate(cumsum):
        if cs >= total / 2:
            half_life = i + 1
            break
    
    # Count resurgences (local maxima after initial peak)
    resurgence_count = 0
    if len(citations) > 5:
        for i in range(peak_idx + 2, len(citations) - 1):
            if citations[i] > citations[i-1] and citations[i] > citations[i+1]:
                if citations[i] > 0.3 * peak_citations:
                    resurgence_count += 1
    
    # Classify pattern
    pattern_type = _classify_pattern(
        citations, years_to_peak, decay_rate, resurgence_count,
        peak_citations, current_trend, len(years)
    )
    
    return {
        'pattern_type': pattern_type,
        'peak_year': peak_year,
        'peak_citations': peak_citations,
        'years_to_peak': years_to_peak,
        'decay_rate': round(decay_rate, 4) if decay_rate else None,
        'resurgence_count': resurgence_count,
        'citation_half_life': half_life,
        'velocity_at_peak': velocity_at_peak,
        'current_trend': current_trend,
    }

def _classify_pattern(
    citations: List[int],
    years_to_peak: int,
    decay_rate: Optional[float],
    resurgence_count: int,
    peak_citations: int,
    current_trend: str,
    total_years: int) -> str:
    """Classify the citation pattern based on features."""
    
    # Early dormancy check (Sleeping Beauty)
    if total_years >= 10:
        early_citations = citations[:5]
        late_citations = citations[-5:]
        
        if sum(early_citations) < sum(late_citations) * 0.1:
            if years_to_peak >= 7:
                return 'sleeping_beauty'
    
    # Heartbeat pattern (multiple resurgences)
    if resurgence_count >= 2:
        return 'heartbeat'
    
    # Flash in the pan (quick peak, rapid decay)
    if years_to_peak <= 2 and decay_rate and decay_rate > 0.5:
        if current_trend in ['dormant', 'decreasing']:
            return 'flash_in_pan'
    
    # Delayed success
    if years_to_peak >= 5 and current_trend in ['stable', 'increasing']:
        return 'delayed'
    
    # Evergreen (steady accumulation, no major decay)
    if decay_rate is not None and decay_rate < 0.1:
        if current_trend in ['stable', 'increasing']:
            return 'evergreen'
    
    # Standard pattern
    return 'standard'

# =============================================================================
# FIELD-NORMALIZED METRICS
# =============================================================================

def compute_field_normalized_metrics(
    df: pd.DataFrame,
    citations_col: str = "Cited by",
    field_col: str = "Subject Area",
    year_col: str = "Year",
    id_col: str = "unique-id",
    sep: str = "; ",
    verbose: bool = True) -> pd.DataFrame:
    """
    Compute field-normalized citation metrics.
    
    Implements:
    - FWCI (Field-Weighted Citation Impact)
    - Percentile rankings within field-year cohorts
    - Z-scores
    - Citation classes
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    citations_col : str
        Total citations column.
    field_col : str
        Field/subject area column.
    year_col : str
        Publication year column.
    id_col : str
        Document ID column.
    sep : str
        Separator for multi-valued fields.
    verbose : bool
        Print progress.
    
    Returns
    -------
    pd.DataFrame with field-normalized metrics.
    """
    if verbose:
        print("Computing field-normalized metrics...")
    
    df = df.copy()
    
    # Ensure numeric columns
    df[citations_col] = pd.to_numeric(df[citations_col], errors='coerce').fillna(0)
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    
    # Parse primary field (use first if multiple)
    def get_primary_field(x):
        if pd.isna(x):
            return "Unknown"
        if isinstance(x, str):
            parts = x.split(sep)
            return parts[0].strip() if parts else "Unknown"
        return str(x)
    
    df['_primary_field'] = df[field_col].apply(get_primary_field) if field_col in df.columns else "Unknown"
    
    # Compute expected citations per field-year
    if verbose:
        print("  Computing field-year baselines...")
    
    field_year_stats = df.groupby(['_primary_field', year_col])[citations_col].agg(['mean', 'std', 'count'])
    field_year_stats.columns = ['expected', 'std', 'count']
    field_year_stats = field_year_stats.reset_index()
    
    # Year-only stats
    year_stats = df.groupby(year_col)[citations_col].agg(['mean', 'std']).reset_index()
    year_stats.columns = [year_col, 'year_expected', 'year_std']
    
    # Field-only stats
    field_stats = df.groupby('_primary_field')[citations_col].agg(['mean', 'std']).reset_index()
    field_stats.columns = ['_primary_field', 'field_expected', 'field_std']
    
    # Compute metrics for each paper
    results = []
    
    # Pre-compute percentile data
    year_citations = df.groupby(year_col)[citations_col].apply(list).to_dict()
    field_citations = df.groupby('_primary_field')[citations_col].apply(list).to_dict()
    field_year_citations = df.groupby(['_primary_field', year_col])[citations_col].apply(list).to_dict()
    
    for _, row in df.iterrows():
        doc_id = str(row.get(id_col, ""))
        citations = row[citations_col]
        year = row[year_col]
        field = row['_primary_field']
        
        if not doc_id or pd.isna(year):
            continue
        
        # Get expected citations for field-year
        fy_mask = (field_year_stats['_primary_field'] == field) & (field_year_stats[year_col] == year)
        fy_row = field_year_stats[fy_mask]
        
        if len(fy_row) > 0:
            expected = fy_row['expected'].values[0]
            std = fy_row['std'].values[0]
        else:
            # Fallback to year average
            y_mask = year_stats[year_col] == year
            if y_mask.any():
                expected = year_stats[y_mask]['year_expected'].values[0]
                std = year_stats[y_mask]['year_std'].values[0]
            else:
                expected = df[citations_col].mean()
                std = df[citations_col].std()
        
        # FWCI
        if expected > 0:
            fwci = citations / expected
        else:
            fwci = 0 if citations == 0 else float('inf')
        
        # Z-score
        if std and std > 0:
            z_score = (citations - expected) / std
        else:
            z_score = 0
        
        # Percentiles
        fy_key = (field, year)
        if fy_key in field_year_citations:
            pct_field_year = percentileofscore(field_year_citations[fy_key], citations)
        else:
            pct_field_year = 50.0
        
        if year in year_citations:
            pct_year = percentileofscore(year_citations[year], citations)
        else:
            pct_year = 50.0
        
        if field in field_citations:
            pct_field = percentileofscore(field_citations[field], citations)
        else:
            pct_field = 50.0
        
        # Citation class
        if citations == 0:
            citation_class = 'uncited'
        elif pct_field_year >= 99:
            citation_class = 'exceptional'
        elif pct_field_year >= 90:
            citation_class = 'high'
        elif pct_field_year >= 50:
            citation_class = 'average'
        else:
            citation_class = 'low'
        
        results.append({
            'doc_id': doc_id,
            'raw_citations': int(citations),
            'expected_citations': round(expected, 2),
            'fwci': round(fwci, 3),
            'percentile_in_field': round(pct_field, 1),
            'percentile_in_year': round(pct_year, 1),
            'percentile_in_field_year': round(pct_field_year, 1),
            'z_score': round(z_score, 2),
            'is_highly_cited': pct_field_year >= 90,
            'is_top_1_percent': pct_field_year >= 99,
            'citation_class': citation_class,
        })
    
    if verbose:
        print(f"  Computed metrics for {len(results)} papers")
    
    return pd.DataFrame(results)

# =============================================================================
# CITATION VELOCITY & MOMENTUM
# =============================================================================

def compute_citation_velocity(
    df: pd.DataFrame,
    citation_history_col: str = None,
    citations_col: str = "Cited by",
    year_col: str = "Year",
    id_col: str = "unique-id",
    current_year: int = None,
    verbose: bool = True) -> pd.DataFrame:
    """
    Compute citation velocity (rate of change) and momentum metrics.
    
    Metrics computed:
    - current_velocity: Recent citation rate
    - acceleration: Change in velocity (second derivative)
    - momentum: Weighted recent citations
    - trajectory_direction: Gaining/losing/stable
    - predicted_next_year: Simple extrapolation
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    citation_history_col : str, optional
        Column with citation counts per year.
    citations_col : str
        Total citations column.
    year_col : str
        Publication year column.
    id_col : str
        Document ID column.
    current_year : int, optional
        Current year for calculations.
    verbose : bool
        Print progress.
    
    Returns
    -------
    pd.DataFrame with velocity metrics.
    """
    if verbose:
        print("Computing citation velocity and momentum...")
    
    if current_year is None:
        current_year = pd.Timestamp.now().year
    
    results = []
    
    for _, row in df.iterrows():
        doc_id = str(row.get(id_col, ""))
        pub_year = row.get(year_col)
        total_citations = row.get(citations_col, 0)
        
        if not doc_id or pd.isna(pub_year):
            continue
        
        pub_year = int(pub_year)
        age = current_year - pub_year
        
        if age < 2:
            results.append({
                'doc_id': doc_id,
                'age_years': age,
                'total_citations': int(total_citations),
                'avg_annual_citations': total_citations / max(age, 1),
                'current_velocity': np.nan,
                'acceleration': np.nan,
                'momentum': np.nan,
                'trajectory_direction': 'too_recent',
                'predicted_next_year': np.nan,
            })
            continue
        
        # Get or synthesize citation history
        citation_history = None
        if citation_history_col and citation_history_col in df.columns:
            hist = row.get(citation_history_col)
            if isinstance(hist, dict):
                citation_history = hist
            elif isinstance(hist, (list, tuple)):
                citation_history = {pub_year + i: c for i, c in enumerate(hist)}
        
        if citation_history is None:
            citation_history = _synthesize_citation_history(
                total_citations, pub_year, current_year
            )
        
        # Convert to time series
        years = sorted(citation_history.keys())
        citations = [citation_history.get(y, 0) for y in years]
        
        # Compute velocities (first derivative)
        if len(citations) >= 2:
            velocities = np.diff(citations)
            current_velocity = velocities[-1] if len(velocities) > 0 else 0
            
            # Acceleration (second derivative)
            if len(velocities) >= 2:
                accelerations = np.diff(velocities)
                acceleration = accelerations[-1]
            else:
                acceleration = 0
        else:
            current_velocity = citations[0] if citations else 0
            acceleration = 0
        
        # Momentum (exponentially weighted recent citations)
        if len(citations) >= 3:
            weights = np.exp(np.linspace(-2, 0, min(5, len(citations))))
            weights = weights / weights.sum()
            recent = citations[-len(weights):]
            momentum = np.dot(recent, weights[-len(recent):])
        else:
            momentum = np.mean(citations) if citations else 0
        
        # Trajectory direction
        if current_velocity > 0.5:
            trajectory_direction = 'gaining'
        elif current_velocity < -0.5:
            trajectory_direction = 'losing'
        else:
            trajectory_direction = 'stable'
        
        # Simple prediction for next year
        if len(citations) >= 3:
            x = np.arange(len(citations[-5:]))
            y = citations[-5:]
            slope = np.polyfit(x, y, 1)[0]
            predicted = max(0, citations[-1] + slope)
        else:
            predicted = np.mean(citations) if citations else 0
        
        results.append({
            'doc_id': doc_id,
            'age_years': age,
            'total_citations': int(total_citations),
            'avg_annual_citations': round(total_citations / max(age, 1), 2),
            'current_velocity': round(current_velocity, 2),
            'acceleration': round(acceleration, 2),
            'momentum': round(momentum, 2),
            'trajectory_direction': trajectory_direction,
            'predicted_next_year': round(predicted, 1),
        })
    
    if verbose:
        print(f"  Computed velocity for {len(results)} papers")
    
    return pd.DataFrame(results)

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_impact_metrics(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    citations_col: str = "Cited by",
    refs_col: str = "References",
    year_col: str = "Year",
    field_col: str = "Subject Area",
    keywords_col: str = None,
    sep: str = "; ",
    compute_disruption: bool = True,
    compute_novelty: bool = True,
    compute_patterns: bool = True,
    compute_normalized: bool = True,
    compute_velocity: bool = True,
    verbose: bool = True) -> ImpactAnalysisResult:
    """
    Perform comprehensive impact and influence analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    id_col : str
        Document ID column.
    citations_col : str
        Total citations column.
    refs_col : str
        References column.
    year_col : str
        Publication year column.
    field_col : str
        Field/subject area column.
    keywords_col : str, optional
        Keywords column for novelty analysis.
    sep : str
        Separator for multi-valued fields.
    compute_disruption : bool
        Compute disruption index.
    compute_novelty : bool
        Compute novelty metrics.
    compute_patterns : bool
        Classify citation patterns.
    compute_normalized : bool
        Compute field-normalized metrics.
    compute_velocity : bool
        Compute citation velocity.
    verbose : bool
        Print progress.
    
    Returns
    -------
    ImpactAnalysisResult
        Complete impact analysis results.
    """
    if verbose:
        print("=" * 50)
        print("Impact & Influence Metrics Analysis")
        print("=" * 50)
    
    # Initialize results
    disruption_df = pd.DataFrame()
    novelty_df = pd.DataFrame()
    patterns_df = pd.DataFrame()
    normalized_df = pd.DataFrame()
    velocity_df = pd.DataFrame()
    
    # Build citation network once for reuse
    citation_network = None
    if compute_disruption or compute_novelty:
        if refs_col in df.columns:
            citation_network = build_citation_network(df, id_col, refs_col, sep)
    
    # Compute disruption index
    if compute_disruption and refs_col in df.columns:
        disruption_df = compute_disruption_index(
            df, citation_network, id_col, refs_col, citations_col, sep,
            verbose=verbose
        )
    
    # Compute novelty metrics
    if compute_novelty:
        # Use keywords if available, otherwise references
        novelty_col = None
        if keywords_col and keywords_col in df.columns:
            novelty_col = keywords_col
        elif refs_col and refs_col in df.columns:
            novelty_col = refs_col
        
        if novelty_col is not None:
            try:
                novelty_df = compute_novelty_metrics(
                    df, novelty_col, id_col, year_col, sep,
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: Novelty computation failed: {e}")
                novelty_df = pd.DataFrame()
        else:
            if verbose:
                print("  Skipping novelty: No suitable column found (keywords or references)")
            novelty_df = pd.DataFrame()
    
    # Classify citation patterns
    if compute_patterns:
        patterns_df = classify_citation_patterns(
            df, citations_col=citations_col, year_col=year_col, id_col=id_col,
            verbose=verbose
        )
    
    # Compute field-normalized metrics
    if compute_normalized:
        normalized_df = compute_field_normalized_metrics(
            df, citations_col, field_col, year_col, id_col, sep,
            verbose=verbose
        )
    
    # Compute citation velocity
    if compute_velocity:
        velocity_df = compute_citation_velocity(
            df, citations_col=citations_col, year_col=year_col, id_col=id_col,
            verbose=verbose
        )
    
    # Compute summary statistics
    summary = _compute_summary_statistics(
        disruption_df, novelty_df, patterns_df, normalized_df, velocity_df
    )
    
    result = ImpactAnalysisResult(
        disruption_metrics=disruption_df,
        novelty_metrics=novelty_df,
        citation_patterns=patterns_df,
        field_normalized=normalized_df,
        velocity_metrics=velocity_df,
        summary_statistics=summary,
        parameters={
            'id_col': id_col,
            'citations_col': citations_col,
            'refs_col': refs_col,
            'year_col': year_col,
            'field_col': field_col,
            'keywords_col': keywords_col,
        })
    
    if verbose:
        print("\n" + "=" * 50)
        print("Analysis complete!")
        _print_summary(summary)
    
    return result

def _compute_summary_statistics(
    disruption_df: pd.DataFrame,
    novelty_df: pd.DataFrame,
    patterns_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    velocity_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics across all metrics."""
    summary = {}
    
    if len(disruption_df) > 0 and 'cd_index' in disruption_df.columns:
        valid_cd = disruption_df['cd_index'].dropna()
        summary['disruption'] = {
            'mean_cd_index': round(valid_cd.mean(), 3),
            'median_cd_index': round(valid_cd.median(), 3),
            'n_disruptive': len(disruption_df[disruption_df['interpretation'] == 'disruptive']),
            'n_consolidating': len(disruption_df[disruption_df['interpretation'] == 'consolidating']),
            'n_neutral': len(disruption_df[disruption_df['interpretation'] == 'neutral']),
        }
    
    if len(novelty_df) > 0 and 'combinatorial_novelty' in novelty_df.columns:
        valid_novelty = novelty_df['combinatorial_novelty'].dropna()
        summary['novelty'] = {
            'mean_novelty': round(valid_novelty.mean(), 3),
            'n_highly_novel': len(novelty_df[novelty_df['interpretation'] == 'highly_novel']),
            'n_novel': len(novelty_df[novelty_df['interpretation'] == 'novel']),
            'n_conventional': len(novelty_df[novelty_df['interpretation'] == 'conventional']),
        }
    
    if len(patterns_df) > 0 and 'pattern_type' in patterns_df.columns:
        pattern_counts = patterns_df['pattern_type'].value_counts().to_dict()
        summary['patterns'] = pattern_counts
    
    if len(normalized_df) > 0 and 'fwci' in normalized_df.columns:
        summary['normalized'] = {
            'mean_fwci': round(normalized_df['fwci'].mean(), 3),
            'median_fwci': round(normalized_df['fwci'].median(), 3),
            'n_highly_cited': len(normalized_df[normalized_df['is_highly_cited']]),
            'n_top_1_percent': len(normalized_df[normalized_df['is_top_1_percent']]),
        }
    
    if len(velocity_df) > 0 and 'trajectory_direction' in velocity_df.columns:
        trajectory_counts = velocity_df['trajectory_direction'].value_counts().to_dict()
        summary['velocity'] = {
            'trajectories': trajectory_counts,
            'mean_velocity': round(velocity_df['current_velocity'].dropna().mean(), 2),
        }
    
    return summary

def _print_summary(summary: Dict[str, Any]) -> None:
    """Print summary statistics."""
    print("\nSummary Statistics:")
    print("-" * 30)
    
    if 'disruption' in summary:
        d = summary['disruption']
        print(f"\nDisruption Index:")
        print(f"  Mean CD Index: {d['mean_cd_index']}")
        print(f"  Disruptive papers: {d['n_disruptive']}")
        print(f"  Consolidating papers: {d['n_consolidating']}")
    
    if 'novelty' in summary:
        n = summary['novelty']
        print(f"\nNovelty:")
        print(f"  Mean novelty: {n['mean_novelty']}")
        print(f"  Highly novel papers: {n['n_highly_novel']}")
    
    if 'patterns' in summary:
        print(f"\nCitation Patterns:")
        for pattern, count in summary['patterns'].items():
            print(f"  {pattern}: {count}")
    
    if 'normalized' in summary:
        fn = summary['normalized']
        print(f"\nField-Normalized Metrics:")
        print(f"  Mean FWCI: {fn['mean_fwci']}")
        print(f"  Highly cited (top 10%): {fn['n_highly_cited']}")
        print(f"  Top 1%: {fn['n_top_1_percent']}")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_disruption_histogram(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot histogram of disruption index values."""
    if result.disruption_metrics is None or len(result.disruption_metrics) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        ax.text(0.5, 0.5, "Disruption metrics not available",
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    df = result.disruption_metrics
    valid_cd = df['cd_index'].dropna()
    
    if len(valid_cd) < 5:
        total = len(df)
        n_nan = df['cd_index'].isna().sum()
        ax.text(0.5, 0.5, f"Insufficient valid CD Index values\n\n"
                f"Total papers: {total}\n"
                f"Papers with CD Index: {len(valid_cd)}\n"
                f"Papers without (uncited in dataset): {n_nan}\n\n"
                f"CD Index requires papers to cite\neach other within the dataset.",
               ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.set_title("CD Index Distribution - Insufficient Data", fontsize=12)
        ax.axis('off')
    else:
        bins = np.linspace(-1, 1, 41)
        ax.hist(valid_cd, bins=bins, color="lightblue", edgecolor='white', alpha=0.7)
        ax.set_xlabel("CD Index", fontsize=12)
        ax.set_ylabel("Number of Papers", fontsize=12)
        ax.set_title(title or f"Disruption Index Distribution (n={len(valid_cd)})", fontsize=13)
        ax.set_xlim(-1, 1)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_histogram.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_disruption_classification(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot paper classification by disruption type."""
    if result.disruption_metrics is None or len(result.disruption_metrics) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        ax.text(0.5, 0.5, "Disruption metrics not available",
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    df = result.disruption_metrics
    interp_counts = df['interpretation'].value_counts()
    
    bars = ax.bar(range(len(interp_counts)), interp_counts.values, color="lightblue", edgecolor='white')
    ax.set_xticks(range(len(interp_counts)))
    ax.set_xticklabels(interp_counts.index, rotation=45, ha='right')
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title(title or "Paper Classification by Disruption", fontsize=13)
    
    total = interp_counts.sum()
    for bar, count in zip(bars, interp_counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_classification.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_disruption_distribution(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Dict[str, plt.Figure]:
    """
    Plot disruption distribution as separate figures.
    
    Returns dict with keys: 'histogram', 'classification'
    """
    return {
        "histogram": plot_disruption_histogram(result, figsize, title, save_path, dpi),
        "classification": plot_disruption_classification(result, figsize, title, save_path, dpi),
    }

def plot_novelty_vs_impact(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot novelty vs citation impact scatter plot.
    """
    # Check if required data is available
    if result.novelty_metrics is None or len(result.novelty_metrics) == 0:
        print("Warning: No novelty metrics available. Cannot create novelty vs impact plot.")
        print("  Run analyze_impact_metrics with compute_novelty=True and a valid keywords column.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "Novelty metrics not available\n\nRun with compute_novelty=True\nand provide keywords_col parameter",
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title("Novelty vs Impact - Data Not Available")
        ax.axis('off')
        return fig
    
    if result.field_normalized is None or len(result.field_normalized) == 0:
        print("Warning: No field-normalized metrics available. Cannot create novelty vs impact plot.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "Field-normalized metrics not available\n\nRun with compute_normalized=True",
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title("Novelty vs Impact - Data Not Available")
        ax.axis('off')
        return fig
    
    if 'combinatorial_novelty' not in result.novelty_metrics.columns:
        print("Warning: combinatorial_novelty column not found in novelty metrics.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "Novelty computation incomplete",
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    # Merge novelty and normalized metrics
    df = result.novelty_metrics.merge(
        result.field_normalized[['doc_id', 'fwci', 'percentile_in_field_year']],
        on='doc_id',
        how='inner'
    )
    
    if len(df) == 0:
        print("Warning: No matching documents between novelty and field-normalized metrics.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "No matching documents found",
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    # Filter valid data
    df = df.dropna(subset=['combinatorial_novelty', 'fwci'])
    df = df[df['fwci'] < df['fwci'].quantile(0.99)]  # Remove extreme outliers
    
    scatter = ax.scatter(
        df['combinatorial_novelty'],
        df['fwci'],
        c=df['percentile_in_field_year'],
        cmap=cmap or CMAP_CONTINUOUS,
        alpha=0.6,
        s=50,
        edgecolors='white',
        linewidths=0.5)
    
    plt.colorbar(scatter, ax=ax, label='Percentile in Field-Year')
    
    # Add quadrant lines
    novelty_median = df['combinatorial_novelty'].median()
    fwci_median = df['fwci'].median()
    
    # Label quadrants
    ax.text(0.95, 0.95, 'High Novelty\nHigh Impact', transform=ax.transAxes,
           ha='right', va='top', fontsize=10, alpha=0.7)
    ax.text(0.05, 0.95, 'Low Novelty\nHigh Impact', transform=ax.transAxes,
           ha='left', va='top', fontsize=10, alpha=0.7)
    ax.text(0.95, 0.05, 'High Novelty\nLow Impact', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=10, alpha=0.7)
    ax.text(0.05, 0.05, 'Low Novelty\nLow Impact', transform=ax.transAxes,
           ha='left', va='bottom', fontsize=10, alpha=0.7)
    
    ax.set_xlabel("Combinatorial Novelty", fontsize=12)
    ax.set_ylabel("Field-Weighted Citation Impact (FWCI)", fontsize=12)
    
    if title is None:
        title = "Novelty vs Impact"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_pattern_distribution(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot citation pattern type distribution."""
    df = result.citation_patterns
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    pattern_counts = df['pattern_type'].value_counts()
    
    bars = ax.bar(range(len(pattern_counts)), pattern_counts.values, color="lightblue")
    ax.set_xticks(range(len(pattern_counts)))
    ax.set_xticklabels(pattern_counts.index, rotation=45, ha='right')
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title(title or "Citation Pattern Distribution", fontsize=12)
    
    for bar, count in zip(bars, pattern_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontsize=9)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_patterns.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_trend_distribution(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot current citation trend distribution."""
    df = result.citation_patterns
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    trend_counts = df['current_trend'].value_counts()
    
    bars = ax.bar(range(len(trend_counts)), trend_counts.values, color="lightblue", edgecolor='white')
    ax.set_xticks(range(len(trend_counts)))
    ax.set_xticklabels(trend_counts.index, rotation=45, ha='right')
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title(title or "Current Citation Trends", fontsize=12)
    
    total = trend_counts.sum()
    for bar, count in zip(bars, trend_counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_trends.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_citation_patterns(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Dict[str, plt.Figure]:
    """
    Plot citation patterns as separate figures.
    
    Returns dict with keys: 'patterns', 'trends'
    """
    return {
        "patterns": plot_pattern_distribution(result, figsize, title, save_path, dpi),
        "trends": plot_trend_distribution(result, figsize, title, save_path, dpi),
    }

def plot_fwci_histogram(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot FWCI histogram."""
    df = result.field_normalized
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    fwci_values = df['fwci'].dropna()
    fwci_capped = fwci_values.clip(upper=fwci_values.quantile(0.95))
    
    ax.hist(fwci_capped, bins=50, color="lightblue", edgecolor='white', alpha=0.7)
    ax.set_xlabel("FWCI (Field-Weighted Citation Impact)", fontsize=11)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title(title or "FWCI Distribution", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_fwci.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_citation_classes(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot citation class distribution."""
    df = result.field_normalized
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    class_counts = df['citation_class'].value_counts()
    class_order = ['exceptional', 'high', 'average', 'low', 'uncited']
    class_counts = class_counts.reindex([c for c in class_order if c in class_counts.index])
    
    bars = ax.bar(range(len(class_counts)), class_counts.values, color="lightblue")
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title(title or "Citation Classes", fontsize=12)
    
    total = class_counts.sum()
    for bar, count in zip(bars, class_counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_classes.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_fwci_distribution(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> Dict[str, plt.Figure]:
    """
    Plot FWCI distribution as separate figures.
    
    Returns dict with keys: 'fwci', 'classes'
    """
    return {
        "fwci": plot_fwci_histogram(result, figsize, title, save_path, dpi),
        "classes": plot_citation_classes(result, figsize, title, save_path, dpi),
    }

def plot_disruption_vs_citations(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot disruption index vs citation count.
    """
    # Check if required data is available
    if result.disruption_metrics is None or len(result.disruption_metrics) == 0:
        print("Warning: No disruption metrics available.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "Disruption metrics not available",
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    if result.field_normalized is None or len(result.field_normalized) == 0:
        print("Warning: No field-normalized metrics available.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "Field-normalized metrics not available",
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    # Merge disruption and normalized metrics
    df = result.disruption_metrics.merge(
        result.field_normalized[['doc_id', 'raw_citations', 'fwci']],
        on='doc_id',
        how='inner'
    )
    
    df = df.dropna(subset=['cd_index', 'raw_citations'])
    
    if len(df) == 0:
        print("Warning: No valid data after merging disruption and citation metrics.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "No valid data for plot\n(all CD indices may be NaN)",
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    # Color by interpretation
    colors = {
        'disruptive': '#27ae60',
        'neutral': '#f1c40f',
        'consolidating': '#e74c3c',
    }
    
    for interp, color in colors.items():
        mask = df['interpretation'] == interp
        if mask.any():
            ax.scatter(
                df[mask]['cd_index'],
                df[mask]['raw_citations'],
                c=color,
                label=interp.title(),
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidths=0.5)
    
    ax.set_xlabel("CD Index (Disruption)", fontsize=12)
    ax.set_ylabel("Total Citations (log scale)", fontsize=12)
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlim(-1.1, 1.1)
    
    if title is None:
        title = "Disruption vs Citations"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend()
    ax
    
    # Disable grids
    for _ax in axes.flat:
        _
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_velocity_histogram(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot citation velocity distribution."""
    df = result.velocity_metrics
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    velocity = df['current_velocity'].dropna()
    velocity_clipped = velocity.clip(lower=-10, upper=10)
    
    ax.hist(velocity_clipped, bins=40, color="lightblue", edgecolor='white', alpha=0.7)
    ax.set_xlabel("Current Velocity", fontsize=11)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title(title or "Citation Velocity Distribution", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_velocity.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_trajectory_distribution(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot trajectory directions distribution."""
    df = result.velocity_metrics
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    traj_counts = df['trajectory_direction'].value_counts()
    
    bars = ax.bar(range(len(traj_counts)), traj_counts.values, color="lightblue", edgecolor='white')
    ax.set_xticks(range(len(traj_counts)))
    ax.set_xticklabels(traj_counts.index, rotation=45, ha='right')
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title(title or "Trajectory Directions", fontsize=12)
    
    total = traj_counts.sum()
    for bar, count in zip(bars, traj_counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_trajectory.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_velocity_vs_age(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot velocity vs paper age."""
    df = result.velocity_metrics
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    valid = df.dropna(subset=['age_years', 'current_velocity'])
    valid = valid[valid['age_years'] <= 30]
    
    ax.scatter(valid['age_years'], valid['current_velocity'], alpha=0.6, s=20, c="lightblue")
    ax.set_xlabel("Paper Age (years)", fontsize=11)
    ax.set_ylabel("Current Velocity", fontsize=11)
    ax.set_title(title or "Velocity vs Paper Age", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_velocity_age.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_momentum_vs_citations(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot momentum vs total citations."""
    df = result.velocity_metrics
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    valid = df.dropna(subset=['total_citations', 'momentum'])
    valid = valid[valid['total_citations'] > 0]
    
    ax.scatter(valid['total_citations'], valid['momentum'], alpha=0.6, s=20, c="lightblue")
    ax.set_xscale('log')
    ax.set_xlabel("Total Citations (log scale)", fontsize=11)
    ax.set_ylabel("Citation Momentum", fontsize=11)
    ax.set_title(title or "Momentum vs Total Citations", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_momentum.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_velocity_analysis(
    result: ImpactAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> Dict[str, plt.Figure]:
    """
    Plot velocity analysis as separate figures.
    
    Returns dict with keys: 'velocity', 'trajectory', 'velocity_age', 'momentum'
    """
    return {
        "velocity": plot_velocity_histogram(result, figsize, title, save_path, dpi),
        "trajectory": plot_trajectory_distribution(result, figsize, title, save_path, dpi),
        "velocity_age": plot_velocity_vs_age(result, figsize, title, save_path, dpi),
        "momentum": plot_momentum_vs_citations(result, figsize, title, save_path, dpi),
    }

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_impact_results(
    result: ImpactAnalysisResult,
    output_dir: str,
    formats: List[str] = None) -> Dict[str, str]:
    """
    Export impact analysis results.
    
    Parameters
    ----------
    result : ImpactAnalysisResult
        Analysis results.
    output_dir : str
        Output directory.
    formats : List[str], optional
        Formats: "excel", "csv", "json".
    
    Returns
    -------
    Dict mapping format to file path.
    """
    if formats is None:
        formats = ["excel"]
    
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    
    if "excel" in formats:
        excel_path = os.path.join(output_dir, "impact_metrics.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            # Combined summary
            result.to_summary_df().to_excel(writer, sheet_name="Summary", index=False)
            
            # Individual metric sheets
            if len(result.disruption_metrics) > 0:
                result.disruption_metrics.to_excel(writer, sheet_name="Disruption", index=False)
            
            if len(result.novelty_metrics) > 0:
                # Exclude complex columns for Excel
                novelty_simple = result.novelty_metrics.drop(columns=['novel_combinations'], errors='ignore')
                novelty_simple.to_excel(writer, sheet_name="Novelty", index=False)
            
            if len(result.citation_patterns) > 0:
                result.citation_patterns.to_excel(writer, sheet_name="Patterns", index=False)
            
            if len(result.field_normalized) > 0:
                result.field_normalized.to_excel(writer, sheet_name="Normalized", index=False)
            
            if len(result.velocity_metrics) > 0:
                result.velocity_metrics.to_excel(writer, sheet_name="Velocity", index=False)
        
        paths["excel"] = excel_path
        print(f"Excel export: {excel_path}")
    
    if "json" in formats:
        json_path = os.path.join(output_dir, "impact_metrics.json")
        
        export_data = {
            'parameters': result.parameters,
            'summary_statistics': result.summary_statistics,
        }
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        paths["json"] = json_path
        print(f"JSON export: {json_path}")
    
    if "csv" in formats:
        csv_dir = output_dir
        os.makedirs(csv_dir, exist_ok=True)
        
        result.to_summary_df().to_csv(os.path.join(csv_dir, "summary.csv"), index=False)
        
        if len(result.disruption_metrics) > 0:
            result.disruption_metrics.to_csv(os.path.join(csv_dir, "disruption.csv"), index=False)
        
        paths["csv"] = csv_dir
        print(f"CSV exports: {csv_dir}")
    
    return paths

# =============================================================================
# INTEGRATION WITH BIBLIOANALYSIS CLASS
# =============================================================================

def add_impact_methods(cls):
    """
    Add impact analysis methods to BiblioAnalysis class.
    
    Usage:
        from impact_metrics import add_impact_methods
        add_impact_methods(BiblioAnalysis)
    """
    
    def analyze_impact_method(
        self,
        compute_disruption: bool = True,
        compute_novelty: bool = True,
        compute_patterns: bool = True,
        compute_normalized: bool = True,
        compute_velocity: bool = True,
        save_results: bool = True,
        **kwargs
    ) -> ImpactAnalysisResult:
        """
        Analyze impact and influence metrics.
        
        Parameters
        ----------
        compute_disruption : bool
            Compute disruption index.
        compute_novelty : bool
            Compute novelty metrics.
        compute_patterns : bool
            Classify citation patterns.
        compute_normalized : bool
            Compute field-normalized metrics.
        compute_velocity : bool
            Compute citation velocity.
        save_results : bool
            Whether to save results.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        ImpactAnalysisResult
        """
        # Get column names from class attributes
        id_col = getattr(self, 'id_var', 'unique-id')
        citations_col = getattr(self, 'citations_var', 'Cited by')
        refs_col = getattr(self, 'refs_var', 'References')
        year_col = getattr(self, 'year_var', 'Year')
        field_col = getattr(self, 'field_var', 'Subject Area')
        keywords_col = getattr(self, 'kw_var', None)
        sep = getattr(self, 'default_separator', '; ')
        
        self.impact_metrics = analyze_impact_metrics(
            self.df,
            id_col=id_col,
            citations_col=citations_col,
            refs_col=refs_col,
            year_col=year_col,
            field_col=field_col,
            keywords_col=keywords_col,
            sep=sep,
            compute_disruption=compute_disruption,
            compute_novelty=compute_novelty,
            compute_patterns=compute_patterns,
            compute_normalized=compute_normalized,
            compute_velocity=compute_velocity,
            **kwargs
        )
        
        if save_results and hasattr(self, 'res_folder') and self.res_folder:
            output_dir = self.res_folder
            export_impact_results(self.impact_metrics, output_dir)
        
        return self.impact_metrics
    
    def plot_impact_method(
        self,
        plot_type: str = "disruption",
        save: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Create impact analysis visualizations.
        
        Parameters
        ----------
        plot_type : str
            Type: "disruption", "novelty_impact", "patterns", "fwci",
            "disruption_citations", "velocity".
        save : bool
            Whether to save plot.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        matplotlib Figure.
        """
        if not hasattr(self, 'impact_metrics'):
            raise ValueError("Run analyze_impact first.")
        
        save_path = None
        if save and hasattr(self, 'res_folder') and self.res_folder:
            save_path = os.path.join(self.res_folder, f"impact_{plot_type}")
        
        plot_functions = {
            "disruption": plot_disruption_distribution,
            "novelty_impact": plot_novelty_vs_impact,
            "patterns": plot_citation_patterns,
            "fwci": plot_fwci_distribution,
            "disruption_citations": plot_disruption_vs_citations,
            "velocity": plot_velocity_analysis,
        }
        
        if plot_type not in plot_functions:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        return plot_functions[plot_type](self.impact_metrics, save_path=save_path, **kwargs)
    
    cls.analyze_impact = analyze_impact_method
    cls.plot_impact = plot_impact_method
    
    return cls

# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import biblium as bb
    ba = bb.BiblioAnalysis(f_name="data\\open alex dataset.csv", db="oa")
    
    print("Impact Metrics Analysis")
    print("=" * 60)
    
    # Run analysis
    result = analyze_impact_metrics(
        df=ba.df,
        citations_col="Cited by",
        year_col="Year",
        refs_col="References" if "References" in ba.df.columns else None,
        verbose=True)
    
    # Print summary
    print(f"\nAnalyzed {len(result.disruption_metrics)} papers")
    
    # Visualizations
    print("\nGenerating plots...")
    plot_disruption_distribution(result, save_path="results/disruption_distribution")
    plot_fwci_distribution(result, save_path="results/fwci_distribution")
    
    # Export
    print("\nExporting results...")
    export_impact_results(result, "results")
    
    print("\nDone!")
