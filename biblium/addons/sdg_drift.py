# -*- coding: utf-8 -*-
"""
SDG Conceptual Drift Analysis Module

This module extends the conceptual_drift addon with SDG-specific analysis,
allowing researchers to track how Sustainable Development Goals concepts
evolve over time, how different SDG perspectives interact, and identify
emerging/fading SDG-related terminology.

Features:
1. SDG-specific drift analysis - Track how each SDG's conceptual meaning evolves
2. SDG perspective comparison - Compare Life, Social, Economic, Peace, Partnership, Planet
3. Cross-SDG concept flow - How concepts migrate between SDGs over time
4. SDG keyword emergence - Track emerging SDG-related terminology
5. SDG co-occurrence evolution - How SDG combinations change over time
6. SDG framing analysis - How SDGs are framed in different contexts

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
from scipy.stats import entropy, spearmanr, chi2_contingency
from scipy.spatial.distance import cosine, jensenshannon
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import seaborn as sns

# Try to import conceptual_drift base module
try:
    from biblium.addons.conceptual_drift import (
        create_time_windows,
        compute_cooccurrence_profile,
        compute_drift_score,
        DriftResult,
        ConceptualDriftAnalysis
    )
    DRIFT_AVAILABLE = True
except ImportError:
    DRIFT_AVAILABLE = False

# =============================================================================
# SDG DEFINITIONS
# =============================================================================

SDG_NAMES = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health and Well-being",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy",
    8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure",
    10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life on Land",
    16: "Peace, Justice and Strong Institutions",
    17: "Partnerships for the Goals",
}

SDG_SHORT_NAMES = {
    1: "Poverty", 2: "Hunger", 3: "Health", 4: "Education", 5: "Gender",
    6: "Water", 7: "Energy", 8: "Work", 9: "Industry", 10: "Inequality",
    11: "Cities", 12: "Consumption", 13: "Climate", 14: "Oceans",
    15: "Land", 16: "Peace", 17: "Partnerships"
}

# SDG Perspectives (Wedding Cake model / similar groupings)
SDG_PERSPECTIVES = {
    "Life": [3],  # Health
    "Social": [1, 2, 4, 5, 10],  # Poverty, Hunger, Education, Gender, Inequality
    "Economic": [7, 8, 9, 11, 12],  # Energy, Work, Industry, Cities, Consumption
    "Planet": [6, 13, 14, 15],  # Water, Climate, Oceans, Land
    "Peace": [16],  # Peace & Justice
    "Partnership": [17],  # Partnerships
}

# Alternative: Five Ps grouping
SDG_FIVE_PS = {
    "People": [1, 2, 3, 4, 5],
    "Prosperity": [7, 8, 9, 10, 11],
    "Planet": [6, 12, 13, 14, 15],
    "Peace": [16],
    "Partnership": [17],
}

# Core SDG-related terms for tracking
SDG_CORE_TERMS = {
    1: ["poverty", "poor", "income", "deprivation", "social protection", "vulnerable"],
    2: ["hunger", "food security", "nutrition", "agriculture", "malnutrition", "famine"],
    3: ["health", "mortality", "disease", "healthcare", "well-being", "mental health"],
    4: ["education", "learning", "school", "literacy", "skills", "training"],
    5: ["gender", "women", "equality", "empowerment", "discrimination", "female"],
    6: ["water", "sanitation", "hygiene", "drinking water", "wastewater", "WASH"],
    7: ["energy", "renewable", "electricity", "clean energy", "solar", "wind"],
    8: ["employment", "economic growth", "decent work", "labor", "jobs", "GDP"],
    9: ["infrastructure", "innovation", "industry", "technology", "R&D", "manufacturing"],
    10: ["inequality", "inclusion", "discrimination", "migration", "social mobility"],
    11: ["urban", "cities", "housing", "transport", "sustainable cities", "slums"],
    12: ["consumption", "production", "waste", "recycling", "circular economy", "SCP"],
    13: ["climate", "emissions", "carbon", "mitigation", "adaptation", "global warming"],
    14: ["ocean", "marine", "fisheries", "coastal", "sea", "aquatic"],
    15: ["biodiversity", "forest", "land degradation", "ecosystem", "species", "deforestation"],
    16: ["peace", "justice", "institutions", "governance", "corruption", "violence"],
    17: ["partnership", "cooperation", "ODA", "capacity building", "technology transfer", "trade"],
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SDGDriftResult:
    """Results for a single SDG's conceptual drift."""
    sdg_number: int
    sdg_name: str
    periods: List[str]
    drift_scores: List[float]
    cumulative_drift: List[float]
    doc_counts: Dict[str, int]
    keyword_evolution: Dict[str, List[Tuple[str, float]]]  # Period -> [(word, weight), ...]
    emerging_terms: Dict[str, List[str]]
    fading_terms: Dict[str, List[str]]
    co_occurring_sdgs: Dict[str, Dict[int, float]]  # Period -> {SDG: proportion}
    perspective_alignment: Dict[str, str]  # Period -> dominant perspective
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to summary DataFrame."""
        records = []
        for i, period in enumerate(self.periods):
            records.append({
                "SDG": self.sdg_number,
                "SDG_Name": self.sdg_name,
                "Period": period,
                "Drift_vs_Previous": self.drift_scores[i] if i > 0 else None,
                "Drift_vs_Baseline": self.cumulative_drift[i],
                "Document Count": self.doc_counts.get(period, 0),
                "Top Keywords": "; ".join([w for w, _ in self.keyword_evolution.get(period, [])[:10]]),
                "Emerging Terms": "; ".join(self.emerging_terms.get(period, [])[:5]),
                "Fading Terms": "; ".join(self.fading_terms.get(period, [])[:5]),
            })
        return pd.DataFrame(records)


@dataclass
class SDGPerspectiveDrift:
    """Results for perspective-level drift analysis."""
    perspective: str
    sdgs: List[int]
    periods: List[str]
    drift_scores: List[float]
    internal_coherence: Dict[str, float]  # Period -> coherence score
    dominant_themes: Dict[str, List[str]]  # Period -> top themes
    cross_perspective_similarity: Dict[str, Dict[str, float]]  # Period -> {other_perspective: similarity}


@dataclass 
class SDGConceptFlowResult:
    """Results for concept flow between SDGs."""
    concept: str
    periods: List[str]
    sdg_associations: Dict[str, Dict[int, float]]  # Period -> {SDG: strength}
    primary_sdg_trajectory: List[int]  # Which SDG dominates each period
    flow_events: List[Dict[str, Any]]  # Migration events


@dataclass
class SDGDriftAnalysis:
    """Container for complete SDG drift analysis."""
    sdg_results: Dict[int, SDGDriftResult]
    perspective_results: Dict[str, SDGPerspectiveDrift]
    concept_flows: Dict[str, SDGConceptFlowResult]
    periods: List[str]
    parameters: Dict[str, Any]
    
    # Summary statistics
    most_drifting_sdgs: List[Tuple[int, float]]
    most_stable_sdgs: List[Tuple[int, float]]
    sdg_cooccurrence_evolution: Dict[str, pd.DataFrame]  # Period -> co-occurrence matrix
    emerging_sdg_combinations: List[Tuple[Tuple[int, int], str, float]]  # New SDG pairs
    
    def get_sdg_summary_df(self) -> pd.DataFrame:
        """Get summary DataFrame for all SDGs."""
        dfs = [result.to_dataframe() for result in self.sdg_results.values()]
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def get_drift_ranking(self) -> pd.DataFrame:
        """Rank SDGs by average drift."""
        rankings = []
        for sdg, result in self.sdg_results.items():
            valid_scores = [s for s in result.drift_scores if s is not None and not np.isnan(s)]
            if valid_scores:
                rankings.append({
                    "SDG": sdg,
                    "Name": result.sdg_name,
                    "Avg Drift": np.mean(valid_scores),
                    "Max Drift": np.max(valid_scores),
                    "Total Drift": result.cumulative_drift[-1] if result.cumulative_drift else 0,
                })
        return pd.DataFrame(rankings).sort_values("Avg Drift", ascending=False)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sdg_columns(df: pd.DataFrame) -> List[str]:
    """Find SDG indicator columns in DataFrame."""
    sdg_cols = []
    for col in df.columns:
        # Match SDG01, SDG02, ..., SDG17 or SDG1, SDG2, etc.
        if col.startswith("SDG") and any(c.isdigit() for c in col):
            sdg_cols.append(col)
    return sorted(sdg_cols)


def extract_sdg_number(col_name: str) -> int:
    """Extract SDG number from column name."""
    import re
    match = re.search(r'(\d+)', col_name)
    if match:
        return int(match.group(1))
    return 0


def get_sdg_documents(df: pd.DataFrame, sdg_number: int, sdg_cols: List[str] = None) -> pd.DataFrame:
    """Get documents associated with a specific SDG."""
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    # Find the column for this SDG
    target_col = None
    for col in sdg_cols:
        if extract_sdg_number(col) == sdg_number:
            target_col = col
            break
    
    if target_col is None:
        return pd.DataFrame()
    
    return df[df[target_col] == 1].copy()


def compute_sdg_cooccurrence_matrix(df: pd.DataFrame, sdg_cols: List[str] = None) -> pd.DataFrame:
    """Compute SDG co-occurrence matrix."""
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    if not sdg_cols:
        return pd.DataFrame()
    
    # Get SDG numbers
    sdg_numbers = [extract_sdg_number(col) for col in sdg_cols]
    
    # Compute co-occurrence
    matrix = np.zeros((len(sdg_cols), len(sdg_cols)))
    
    for i, col_i in enumerate(sdg_cols):
        for j, col_j in enumerate(sdg_cols):
            if i == j:
                matrix[i, j] = df[col_i].sum()
            else:
                matrix[i, j] = ((df[col_i] == 1) & (df[col_j] == 1)).sum()
    
    return pd.DataFrame(matrix, index=sdg_numbers, columns=sdg_numbers)


def compute_keyword_profile(
    df: pd.DataFrame,
    text_col: str,
    top_n: int = 50,
    min_df: int = 3,
    max_df: float = 0.8,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Dict[str, float]:
    """Compute TF-IDF keyword profile for a document set."""
    texts = df[text_col].dropna().astype(str).tolist()
    
    if len(texts) < min_df:
        return {}
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=top_n * 2,
            min_df=min(min_df, len(texts) - 1),
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Get mean TF-IDF scores
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        feature_names = vectorizer.get_feature_names_out()
        
        # Return top keywords
        top_indices = mean_scores.argsort()[-top_n:][::-1]
        return {feature_names[i]: mean_scores[i] for i in top_indices}
    except Exception as e:
        warnings.warn(f"Keyword extraction failed: {e}")
        return {}


def compute_profile_drift(profile1: Dict[str, float], profile2: Dict[str, float]) -> float:
    """Compute drift between two keyword profiles using Jensen-Shannon divergence."""
    if not profile1 or not profile2:
        return np.nan
    
    # Get union of all terms
    all_terms = set(profile1.keys()) | set(profile2.keys())
    
    if len(all_terms) == 0:
        return np.nan
    
    # Create vectors
    vec1 = np.array([profile1.get(t, 0) for t in all_terms])
    vec2 = np.array([profile2.get(t, 0) for t in all_terms])
    
    # Normalize to probability distributions
    if vec1.sum() > 0:
        vec1 = vec1 / vec1.sum()
    if vec2.sum() > 0:
        vec2 = vec2 / vec2.sum()
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    vec1 = vec1 + eps
    vec2 = vec2 + eps
    vec1 = vec1 / vec1.sum()
    vec2 = vec2 / vec2.sum()
    
    return jensenshannon(vec1, vec2)


def identify_emerging_fading(
    profile_old: Dict[str, float],
    profile_new: Dict[str, float],
    threshold_ratio: float = 2.0
) -> Tuple[List[str], List[str]]:
    """Identify emerging and fading terms between two periods."""
    emerging = []
    fading = []
    
    all_terms = set(profile_old.keys()) | set(profile_new.keys())
    
    for term in all_terms:
        old_score = profile_old.get(term, 0)
        new_score = profile_new.get(term, 0)
        
        if old_score == 0 and new_score > 0:
            emerging.append((term, new_score))
        elif new_score == 0 and old_score > 0:
            fading.append((term, old_score))
        elif old_score > 0:
            ratio = new_score / old_score
            if ratio >= threshold_ratio:
                emerging.append((term, new_score))
            elif ratio <= 1 / threshold_ratio:
                fading.append((term, old_score))
    
    emerging = [t for t, _ in sorted(emerging, key=lambda x: -x[1])]
    fading = [t for t, _ in sorted(fading, key=lambda x: -x[1])]
    
    return emerging, fading


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def analyze_sdg_drift(
    df: pd.DataFrame,
    text_col: str = "Processed Abstract",
    year_col: str = "Year",
    sdg_cols: List[str] = None,
    window_size: int = 5,
    min_docs_per_window: int = 20,
    top_keywords: int = 30,
    perspective_grouping: str = "perspectives",  # "perspectives" or "five_ps"
    verbose: bool = True
) -> SDGDriftAnalysis:
    """
    Analyze conceptual drift for SDGs over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe with SDG indicator columns.
    text_col : str
        Column containing text for analysis.
    year_col : str
        Column containing publication year.
    sdg_cols : List[str], optional
        SDG indicator columns. Auto-detected if None.
    window_size : int
        Size of time windows in years.
    min_docs_per_window : int
        Minimum documents required per window.
    top_keywords : int
        Number of top keywords to track per SDG.
    perspective_grouping : str
        "perspectives" for 6-way grouping, "five_ps" for 5 Ps grouping.
    verbose : bool
        Print progress.
    
    Returns
    -------
    SDGDriftAnalysis
        Complete drift analysis results.
    """
    if verbose:
        print("="*60)
        print("SDG CONCEPTUAL DRIFT ANALYSIS")
        print("="*60)
    
    # Auto-detect SDG columns
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    if not sdg_cols:
        raise ValueError("No SDG columns found. Run identify_sdgs() first or provide sdg_cols.")
    
    if verbose:
        print(f"Found {len(sdg_cols)} SDG columns: {sdg_cols}")
    
    # Create time windows
    df_clean = df.dropna(subset=[year_col, text_col]).copy()
    df_clean[year_col] = df_clean[year_col].astype(int)
    
    min_year = df_clean[year_col].min()
    max_year = df_clean[year_col].max()
    
    windows = []
    start = min_year
    while start <= max_year:
        end = min(start + window_size - 1, max_year)
        mask = (df_clean[year_col] >= start) & (df_clean[year_col] <= end)
        subset = df_clean[mask].copy()
        if len(subset) >= min_docs_per_window:
            windows.append((f"{start}-{end}", subset))
        start += window_size
    
    if verbose:
        print(f"Created {len(windows)} time windows")
        for period, subset in windows:
            print(f"  {period}: {len(subset)} documents")
    
    periods = [w[0] for w in windows]
    
    # Select perspective grouping
    perspective_groups = SDG_PERSPECTIVES if perspective_grouping == "perspectives" else SDG_FIVE_PS
    
    # ==========================================================================
    # ANALYZE EACH SDG
    # ==========================================================================
    
    sdg_results = {}
    
    for sdg_col in sdg_cols:
        sdg_num = extract_sdg_number(sdg_col)
        if sdg_num < 1 or sdg_num > 17:
            continue
        
        if verbose:
            print(f"\nAnalyzing SDG {sdg_num}: {SDG_NAMES.get(sdg_num, 'Unknown')}...")
        
        # Compute profiles for each period
        profiles = {}
        doc_counts = {}
        co_occurring_sdgs = {}
        
        for period, subset in windows:
            sdg_docs = subset[subset[sdg_col] == 1]
            doc_counts[period] = len(sdg_docs)
            
            if len(sdg_docs) >= 5:
                profiles[period] = compute_keyword_profile(
                    sdg_docs, text_col, top_n=top_keywords
                )
                
                # Compute co-occurring SDGs
                cooccur = {}
                for other_col in sdg_cols:
                    other_num = extract_sdg_number(other_col)
                    if other_num != sdg_num:
                        cooccur[other_num] = sdg_docs[other_col].mean()
                co_occurring_sdgs[period] = cooccur
            else:
                profiles[period] = {}
                co_occurring_sdgs[period] = {}
        
        # Compute drift scores
        drift_scores = [None]  # First period has no drift
        cumulative_drift = [0.0]
        
        baseline_profile = profiles.get(periods[0], {})
        
        for i in range(1, len(periods)):
            prev_profile = profiles.get(periods[i-1], {})
            curr_profile = profiles.get(periods[i], {})
            
            # Drift vs previous
            drift = compute_profile_drift(prev_profile, curr_profile)
            drift_scores.append(drift)
            
            # Drift vs baseline
            cum_drift = compute_profile_drift(baseline_profile, curr_profile)
            cumulative_drift.append(cum_drift)
        
        # Identify emerging/fading terms
        emerging_terms = {}
        fading_terms = {}
        
        for i in range(1, len(periods)):
            prev_profile = profiles.get(periods[i-1], {})
            curr_profile = profiles.get(periods[i], {})
            emerging, fading = identify_emerging_fading(prev_profile, curr_profile)
            emerging_terms[periods[i]] = emerging[:10]
            fading_terms[periods[i]] = fading[:10]
        
        # Keyword evolution
        keyword_evolution = {
            period: list(profiles.get(period, {}).items())[:top_keywords]
            for period in periods
        }
        
        # Determine perspective alignment
        perspective_alignment = {}
        for persp, sdg_list in perspective_groups.items():
            if sdg_num in sdg_list:
                for period in periods:
                    perspective_alignment[period] = persp
                break
        
        sdg_results[sdg_num] = SDGDriftResult(
            sdg_number=sdg_num,
            sdg_name=SDG_NAMES.get(sdg_num, f"SDG {sdg_num}"),
            periods=periods,
            drift_scores=drift_scores,
            cumulative_drift=cumulative_drift,
            doc_counts=doc_counts,
            keyword_evolution=keyword_evolution,
            emerging_terms=emerging_terms,
            fading_terms=fading_terms,
            co_occurring_sdgs=co_occurring_sdgs,
            perspective_alignment=perspective_alignment
        )
    
    # ==========================================================================
    # ANALYZE PERSPECTIVES
    # ==========================================================================
    
    if verbose:
        print("\nAnalyzing SDG perspectives...")
    
    perspective_results = {}
    
    for persp, sdg_list in perspective_groups.items():
        # Get documents for this perspective
        persp_cols = [col for col in sdg_cols if extract_sdg_number(col) in sdg_list]
        
        if not persp_cols:
            continue
        
        profiles = {}
        internal_coherence = {}
        
        for period, subset in windows:
            # Documents with any SDG in this perspective
            mask = subset[persp_cols].any(axis=1)
            persp_docs = subset[mask]
            
            if len(persp_docs) >= 10:
                profiles[period] = compute_keyword_profile(persp_docs, text_col, top_n=top_keywords)
                
                # Internal coherence: average pairwise similarity between SDGs
                sdg_profiles = []
                for col in persp_cols:
                    sdg_docs = persp_docs[persp_docs[col] == 1]
                    if len(sdg_docs) >= 5:
                        sdg_profiles.append(compute_keyword_profile(sdg_docs, text_col))
                
                if len(sdg_profiles) >= 2:
                    similarities = []
                    for i, j in combinations(range(len(sdg_profiles)), 2):
                        sim = 1 - compute_profile_drift(sdg_profiles[i], sdg_profiles[j])
                        if not np.isnan(sim):
                            similarities.append(sim)
                    internal_coherence[period] = np.mean(similarities) if similarities else np.nan
                else:
                    internal_coherence[period] = np.nan
        
        # Compute drift
        drift_scores = [None]
        for i in range(1, len(periods)):
            prev = profiles.get(periods[i-1], {})
            curr = profiles.get(periods[i], {})
            drift_scores.append(compute_profile_drift(prev, curr))
        
        # Dominant themes
        dominant_themes = {
            period: [w for w, _ in list(profiles.get(period, {}).items())[:10]]
            for period in periods
        }
        
        perspective_results[persp] = SDGPerspectiveDrift(
            perspective=persp,
            sdgs=sdg_list,
            periods=periods,
            drift_scores=drift_scores,
            internal_coherence=internal_coherence,
            dominant_themes=dominant_themes,
            cross_perspective_similarity={}  # TODO: compute cross-perspective similarity
        )
    
    # ==========================================================================
    # ANALYZE CONCEPT FLOWS (SDG Core Terms)
    # ==========================================================================
    
    if verbose:
        print("\nAnalyzing concept flows across SDGs...")
    
    concept_flows = {}
    
    # Track a subset of core terms
    all_core_terms = []
    for terms in SDG_CORE_TERMS.values():
        all_core_terms.extend(terms[:3])  # Top 3 terms per SDG
    all_core_terms = list(set(all_core_terms))[:20]  # Limit to 20 terms
    
    for concept in all_core_terms:
        sdg_associations = {}
        
        for period, subset in windows:
            # Find which SDGs this concept appears in
            associations = {}
            
            for sdg_col in sdg_cols:
                sdg_num = extract_sdg_number(sdg_col)
                sdg_docs = subset[subset[sdg_col] == 1]
                
                if len(sdg_docs) > 0:
                    # Check how often concept appears in this SDG's documents
                    concept_count = sdg_docs[text_col].str.lower().str.contains(
                        concept.lower(), na=False
                    ).sum()
                    associations[sdg_num] = concept_count / len(sdg_docs)
            
            sdg_associations[period] = associations
        
        # Primary SDG trajectory
        primary_trajectory = []
        for period in periods:
            assoc = sdg_associations.get(period, {})
            if assoc:
                primary_sdg = max(assoc, key=assoc.get)
                primary_trajectory.append(primary_sdg)
            else:
                primary_trajectory.append(0)
        
        concept_flows[concept] = SDGConceptFlowResult(
            concept=concept,
            periods=periods,
            sdg_associations=sdg_associations,
            primary_sdg_trajectory=primary_trajectory,
            flow_events=[]  # TODO: detect flow events
        )
    
    # ==========================================================================
    # COMPUTE SUMMARY STATISTICS
    # ==========================================================================
    
    # Rank SDGs by drift
    drift_ranking = []
    for sdg_num, result in sdg_results.items():
        valid = [s for s in result.drift_scores if s is not None and not np.isnan(s)]
        if valid:
            drift_ranking.append((sdg_num, np.mean(valid)))
    
    drift_ranking.sort(key=lambda x: -x[1])
    most_drifting = drift_ranking[:5]
    most_stable = drift_ranking[-5:][::-1]
    
    # SDG co-occurrence evolution
    cooccur_evolution = {}
    for period, subset in windows:
        cooccur_evolution[period] = compute_sdg_cooccurrence_matrix(subset, sdg_cols)
    
    # Emerging SDG combinations
    emerging_combinations = []
    if len(windows) >= 2:
        first_cooccur = cooccur_evolution.get(periods[0], pd.DataFrame())
        last_cooccur = cooccur_evolution.get(periods[-1], pd.DataFrame())
        
        if not first_cooccur.empty and not last_cooccur.empty:
            for i in first_cooccur.index:
                for j in first_cooccur.columns:
                    if i < j:
                        first_val = first_cooccur.loc[i, j] if i in first_cooccur.index else 0
                        last_val = last_cooccur.loc[i, j] if i in last_cooccur.index else 0
                        
                        if first_val > 0:
                            change = (last_val - first_val) / first_val
                        elif last_val > 0:
                            change = float('inf')
                        else:
                            change = 0
                        
                        if change > 0.5:  # 50% increase
                            emerging_combinations.append(((i, j), periods[-1], change))
    
    emerging_combinations.sort(key=lambda x: -x[2] if x[2] != float('inf') else 1000)
    
    if verbose:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nMost drifting SDGs:")
        for sdg, drift in most_drifting[:3]:
            print(f"  SDG {sdg} ({SDG_SHORT_NAMES.get(sdg, '')}): {drift:.3f}")
        print(f"\nMost stable SDGs:")
        for sdg, drift in most_stable[:3]:
            print(f"  SDG {sdg} ({SDG_SHORT_NAMES.get(sdg, '')}): {drift:.3f}")
    
    return SDGDriftAnalysis(
        sdg_results=sdg_results,
        perspective_results=perspective_results,
        concept_flows=concept_flows,
        periods=periods,
        parameters={
            "text_col": text_col,
            "year_col": year_col,
            "window_size": window_size,
            "min_docs_per_window": min_docs_per_window,
            "perspective_grouping": perspective_grouping,
        },
        most_drifting_sdgs=most_drifting,
        most_stable_sdgs=most_stable,
        sdg_cooccurrence_evolution=cooccur_evolution,
        emerging_sdg_combinations=emerging_combinations[:10],
    )


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_sdg_drift_heatmap(
    analysis: SDGDriftAnalysis,
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = "YlOrRd",
    title: str = "SDG Conceptual Drift Over Time",
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot heatmap of SDG drift scores over time.
    """
    # Build matrix
    sdg_nums = sorted(analysis.sdg_results.keys())
    periods = analysis.periods
    
    matrix = []
    for sdg in sdg_nums:
        result = analysis.sdg_results[sdg]
        # Use cumulative drift for visualization
        matrix.append(result.cumulative_drift)
    
    df_matrix = pd.DataFrame(
        matrix,
        index=[f"SDG {s}: {SDG_SHORT_NAMES.get(s, '')}" for s in sdg_nums],
        columns=periods
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    sns.heatmap(
        df_matrix,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Cumulative Drift"}
    )
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("SDG")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_sdg_drift_trajectories(
    analysis: SDGDriftAnalysis,
    sdgs: List[int] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "SDG Drift Trajectories",
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot drift trajectories for selected SDGs.
    """
    if sdgs is None:
        # Plot top 5 most drifting
        sdgs = [s for s, _ in analysis.most_drifting_sdgs[:5]]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(sdgs)))
    
    for i, sdg in enumerate(sdgs):
        if sdg in analysis.sdg_results:
            result = analysis.sdg_results[sdg]
            ax.plot(
                result.periods,
                result.cumulative_drift,
                marker='o',
                linewidth=2,
                color=colors[i],
                label=f"SDG {sdg}: {SDG_SHORT_NAMES.get(sdg, '')}"
            )
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Cumulative Drift (from baseline)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_sdg_cooccurrence_evolution(
    analysis: SDGDriftAnalysis,
    figsize: Tuple[int, int] = (16, 5),
    cmap: str = "Blues",
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot SDG co-occurrence matrices for first and last periods.
    """
    periods = list(analysis.sdg_cooccurrence_evolution.keys())
    
    if len(periods) < 2:
        print("Need at least 2 periods for evolution plot")
        return None
    
    first_period = periods[0]
    last_period = periods[-1]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for _ax in (axes if hasattr(axes, "__iter__") else [axes]): _ax.grid(False)
    
    for ax, period in zip(axes, [first_period, last_period]):
        matrix = analysis.sdg_cooccurrence_evolution[period]
        
        if matrix.empty:
            continue
        
        # Normalize by diagonal (self-occurrence)
        norm_matrix = matrix.copy()
        for i in matrix.index:
            if matrix.loc[i, i] > 0:
                norm_matrix.loc[i, :] = matrix.loc[i, :] / matrix.loc[i, i]
        
        sns.heatmap(
            norm_matrix,
            cmap=cmap,
            annot=False,
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Co-occurrence Rate"}
        )
        ax.set_title(f"SDG Co-occurrence: {period}")
    
    plt.suptitle("SDG Co-occurrence Evolution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_perspective_drift(
    analysis: SDGDriftAnalysis,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "SDG Perspective Drift",
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot drift for SDG perspectives.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    colors = {
        "Life": "#e41a1c",
        "Social": "#377eb8",
        "Economic": "#4daf4a",
        "Planet": "#984ea3",
        "Peace": "#ff7f00",
        "Partnership": "#ffff33",
        "People": "#377eb8",
        "Prosperity": "#4daf4a",
    }
    
    for persp, result in analysis.perspective_results.items():
        drift_vals = [d if d is not None else 0 for d in result.drift_scores]
        cumulative = np.cumsum([0] + drift_vals[1:])
        
        ax.plot(
            result.periods,
            cumulative,
            marker='s',
            linewidth=2,
            markersize=8,
            color=colors.get(persp, "gray"),
            label=persp
        )
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Cumulative Drift")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_emerging_terms_wordcloud(
    analysis: SDGDriftAnalysis,
    sdg: int,
    period: str = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot word cloud of emerging terms for an SDG.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud package not installed")
        return None
    
    if sdg not in analysis.sdg_results:
        print(f"SDG {sdg} not found in results")
        return None
    
    result = analysis.sdg_results[sdg]
    
    if period is None:
        period = result.periods[-1]
    
    keywords = dict(result.keyword_evolution.get(period, []))
    
    if not keywords:
        print(f"No keywords for SDG {sdg} in period {period}")
        return None
    
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=50
    ).generate_from_frequencies(keywords)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"SDG {sdg} Keywords: {period}", fontsize=14, fontweight="bold")
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_sdg_drift_analysis(
    ba,  # BiblioAnalysis object
    window_size: int = 5,
    min_docs: int = 20,
    verbose: bool = True
) -> SDGDriftAnalysis:
    """
    Convenience function to run SDG drift analysis on BiblioAnalysis object.
    
    Usage:
        from biblium.addons.sdg_drift import run_sdg_drift_analysis
        
        # Make sure SDGs are identified first
        ba.identify_sdgs()
        
        # Run drift analysis
        drift = run_sdg_drift_analysis(ba)
        
        # View results
        print(drift.get_sdg_summary_df())
        
        # Plot
        from biblium.addons.sdg_drift import plot_sdg_drift_heatmap
        plot_sdg_drift_heatmap(drift)
    """
    # Check for required columns
    text_col = None
    for candidate in ["Processed Abstract", "Abstract"]:
        if candidate in ba.df.columns:
            text_col = candidate
            break
    
    if text_col is None:
        raise ValueError("No text column found. Process text first with ba.process_text_vars()")
    
    # Check for SDG columns
    sdg_cols = get_sdg_columns(ba.df)
    if not sdg_cols:
        raise ValueError("No SDG columns found. Run ba.identify_sdgs() first.")
    
    return analyze_sdg_drift(
        ba.df,
        text_col=text_col,
        year_col="Year",
        sdg_cols=sdg_cols,
        window_size=window_size,
        min_docs_per_window=min_docs,
        verbose=verbose
    )
