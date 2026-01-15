# -*- coding: utf-8 -*-
"""
Research Gaps Identifier Module

This module provides tools for systematically identifying under-researched
areas in bibliometric datasets, with special focus on SDG research gaps.

Features:
1. SDG Combination Gaps - Under-researched SDG pairs/triples
2. Geographic Gaps - Regions under-researching specific topics
3. Methodological Gaps - Under-used methods for certain topics
4. Temporal Gaps - Topics losing research attention
5. Thematic Gaps - Under-explored subtopics within fields
6. Cross-Domain Gaps - Missing interdisciplinary connections
7. Gap Prioritization - Rank gaps by importance/opportunity

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns

# =============================================================================
# SDG DEFINITIONS
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

# Expected/ideal SDG relationships (based on literature)
# Higher values = stronger expected connection
SDG_EXPECTED_CONNECTIONS = {
    (1, 2): 0.8, (1, 3): 0.6, (1, 4): 0.7, (1, 5): 0.6, (1, 8): 0.7, (1, 10): 0.8,
    (2, 3): 0.7, (2, 6): 0.6, (2, 12): 0.6, (2, 13): 0.7, (2, 15): 0.6,
    (3, 6): 0.7, (3, 13): 0.5,
    (4, 5): 0.7, (4, 8): 0.6,
    (5, 8): 0.5, (5, 10): 0.7, (5, 16): 0.5,
    (6, 7): 0.5, (6, 13): 0.6, (6, 14): 0.6, (6, 15): 0.5,
    (7, 8): 0.6, (7, 9): 0.7, (7, 11): 0.6, (7, 12): 0.6, (7, 13): 0.8,
    (8, 9): 0.7, (8, 12): 0.5,
    (9, 11): 0.6, (9, 12): 0.5,
    (11, 12): 0.6, (11, 13): 0.6,
    (12, 13): 0.7, (12, 14): 0.5, (12, 15): 0.5,
    (13, 14): 0.7, (13, 15): 0.7,
    (14, 15): 0.6,
    (16, 17): 0.5,
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ResearchGap:
    """A single research gap."""
    gap_type: str  # "sdg_combination", "geographic", "methodological", "temporal", "thematic"
    description: str
    entities: List[Any]  # SDGs, countries, methods, etc.
    current_count: int
    expected_count: float
    gap_score: float  # 0-1, higher = bigger gap
    priority_score: float  # Based on importance and feasibility
    evidence: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Type": self.gap_type,
            "Description": self.description,
            "Entities": str(self.entities),
            "Current Count": self.current_count,
            "Expected Count": self.expected_count,
            "Gap Score": self.gap_score,
            "Priority Score": self.priority_score,
            "Recommendations": "; ".join(self.recommendations),
        }


@dataclass
class SDGCombinationGap:
    """Gap in SDG combination research."""
    sdgs: Tuple[int, ...]
    sdg_names: List[str]
    observed_count: int
    expected_count: float
    expected_ratio: float  # Based on individual SDG frequencies
    gap_ratio: float  # expected / observed
    statistical_significance: float  # p-value
    related_combinations: List[Tuple[int, ...]]  # Similar but researched
    potential_synergies: List[str]
    potential_tradeoffs: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "SDGs": self.sdgs,
            "Names": " + ".join(self.sdg_names),
            "Observed": self.observed_count,
            "Expected": self.expected_count,
            "Gap Ratio": self.gap_ratio,
            "P_Value": self.statistical_significance,
            "Potential Synergies": "; ".join(self.potential_synergies[:3]),
        }


@dataclass
class GeographicGap:
    """Geographic research gap."""
    region: str
    sdgs_underrepresented: List[int]
    sdgs_overrepresented: List[int]
    total_papers: int
    global_share: float
    research_focus_bias: Dict[int, float]  # SDG -> deviation from global average
    recommended_focus: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Region": self.region,
            "Total Papers": self.total_papers,
            "Global Share": self.global_share,
            "Underrepresented_SDGs": self.sdgs_underrepresented,
            "Overrepresented_SDGs": self.sdgs_overrepresented,
            "Recommended Focus": self.recommended_focus,
        }


@dataclass
class MethodologicalGap:
    """Methodological research gap."""
    method: str
    sdgs_rarely_using: List[int]
    sdgs_commonly_using: List[int]
    total_usage: int
    usage_distribution: Dict[int, float]  # SDG -> proportion using this method
    recommended_applications: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Method": self.method,
            "Total Usage": self.total_usage,
            "Rarely Used For": [SDG_SHORT_NAMES.get(s, str(s)) for s in self.sdgs_rarely_using],
            "Commonly Used For": [SDG_SHORT_NAMES.get(s, str(s)) for s in self.sdgs_commonly_using],
        }


@dataclass
class TemporalGap:
    """Temporal research gap - declining attention."""
    topic: str  # SDG or keyword
    peak_year: int
    peak_count: int
    recent_count: int
    decline_rate: float  # Percentage decline from peak
    still_relevant: bool  # Based on citations or other indicators
    recommended_revival: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Topic": self.topic,
            "Peak Year": self.peak_year,
            "Peak Count": self.peak_count,
            "Recent Count": self.recent_count,
            "Decline Rate": self.decline_rate,
            "Still Relevant": self.still_relevant,
        }


@dataclass
class ResearchGapsAnalysis:
    """Complete research gaps analysis results."""
    sdg_combination_gaps: List[SDGCombinationGap]
    geographic_gaps: List[GeographicGap]
    methodological_gaps: List[MethodologicalGap]
    temporal_gaps: List[TemporalGap]
    all_gaps: List[ResearchGap]
    
    # Summary
    top_priority_gaps: List[ResearchGap]
    gap_summary: Dict[str, int]  # Type -> count
    
    # Matrices
    sdg_gap_matrix: pd.DataFrame  # SDG pair -> gap score
    geographic_sdg_matrix: pd.DataFrame  # Region x SDG -> gap indicator
    
    def get_sdg_gaps_df(self) -> pd.DataFrame:
        """Get SDG combination gaps as DataFrame."""
        records = [g.to_dict() for g in self.sdg_combination_gaps]
        df = pd.DataFrame(records)
        return df.sort_values("Gap Ratio", ascending=False) if not df.empty and "Gap Ratio" in df.columns else df
    
    def get_geographic_gaps_df(self) -> pd.DataFrame:
        """Get geographic gaps as DataFrame."""
        records = [g.to_dict() for g in self.geographic_gaps]
        return pd.DataFrame(records)
    
    def get_all_gaps_df(self) -> pd.DataFrame:
        """Get all gaps as DataFrame."""
        records = [g.to_dict() for g in self.all_gaps]
        return pd.DataFrame(records).sort_values("Priority Score", ascending=False)
    
    def get_top_recommendations(self, n: int = 10) -> List[str]:
        """Get top research recommendations."""
        recommendations = []
        for gap in self.top_priority_gaps[:n]:
            recommendations.extend(gap.recommendations[:2])
        return recommendations[:n]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sdg_columns(df: pd.DataFrame) -> List[str]:
    """Find SDG indicator columns."""
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
    """Get SDGs for a paper."""
    return [extract_sdg_number(col) for col in sdg_cols if row.get(col, 0) == 1]


def compute_expected_cooccurrence(
    count_i: int,
    count_j: int,
    total: int
) -> float:
    """Compute expected co-occurrence under independence."""
    if total == 0:
        return 0
    p_i = count_i / total
    p_j = count_j / total
    return p_i * p_j * total


# =============================================================================
# SDG COMBINATION GAP ANALYSIS
# =============================================================================

def identify_sdg_combination_gaps(
    df: pd.DataFrame,
    sdg_cols: List[str] = None,
    min_expected: float = 5.0,
    gap_threshold: float = 2.0,
    max_combination_size: int = 3
) -> List[SDGCombinationGap]:
    """
    Identify under-researched SDG combinations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with SDG indicators.
    sdg_cols : List[str]
        SDG columns. Auto-detected if None.
    min_expected : float
        Minimum expected count to consider.
    gap_threshold : float
        Minimum gap ratio (expected/observed) to flag.
    max_combination_size : int
        Maximum SDGs in combination (2 or 3).
    
    Returns
    -------
    List of SDGCombinationGap objects.
    """
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    total_papers = len(df)
    sdg_numbers = [extract_sdg_number(col) for col in sdg_cols]
    
    # Count individual SDGs
    sdg_counts = {}
    for col in sdg_cols:
        sdg_num = extract_sdg_number(col)
        sdg_counts[sdg_num] = df[col].sum()
    
    # Count combinations
    combination_counts = Counter()
    for _, row in df.iterrows():
        sdgs = tuple(sorted(get_paper_sdgs(row, sdg_cols)))
        if len(sdgs) >= 2:
            # Count all pairs
            for pair in combinations(sdgs, 2):
                combination_counts[pair] += 1
            # Count triples if requested
            if max_combination_size >= 3 and len(sdgs) >= 3:
                for triple in combinations(sdgs, 3):
                    combination_counts[triple] += 1
    
    gaps = []
    
    # Analyze pairs
    for i, sdg_i in enumerate(sdg_numbers):
        for j, sdg_j in enumerate(sdg_numbers):
            if i >= j:
                continue
            
            pair = (sdg_i, sdg_j)
            observed = combination_counts.get(pair, 0)
            
            # Expected under independence
            expected = compute_expected_cooccurrence(
                sdg_counts.get(sdg_i, 0),
                sdg_counts.get(sdg_j, 0),
                total_papers
            )
            
            # Adjust by known expected connections
            key = tuple(sorted(pair))
            expected_modifier = SDG_EXPECTED_CONNECTIONS.get(key, 0.3)
            adjusted_expected = expected * (1 + expected_modifier)
            
            if adjusted_expected < min_expected:
                continue
            
            gap_ratio = adjusted_expected / max(observed, 1)
            
            if gap_ratio >= gap_threshold:
                # Statistical test
                contingency = [
                    [observed, sdg_counts.get(sdg_i, 0) - observed],
                    [sdg_counts.get(sdg_j, 0) - observed, total_papers - sdg_counts.get(sdg_i, 0) - sdg_counts.get(sdg_j, 0) + observed]
                ]
                try:
                    _, p_value, _, _ = chi2_contingency(contingency)
                except:
                    p_value = 1.0
                
                # Find related combinations that ARE researched
                related = []
                for other_pair, count in combination_counts.items():
                    if len(other_pair) == 2:
                        if sdg_i in other_pair or sdg_j in other_pair:
                            if count > observed * 2:
                                related.append(other_pair)
                
                gaps.append(SDGCombinationGap(
                    sdgs=pair,
                    sdg_names=[SDG_SHORT_NAMES.get(sdg_i, ""), SDG_SHORT_NAMES.get(sdg_j, "")],
                    observed_count=observed,
                    expected_count=adjusted_expected,
                    expected_ratio=expected / total_papers if total_papers > 0 else 0,
                    gap_ratio=gap_ratio,
                    statistical_significance=p_value,
                    related_combinations=related[:5],
                    potential_synergies=[
                        f"Combining {SDG_SHORT_NAMES.get(sdg_i, '')} and {SDG_SHORT_NAMES.get(sdg_j, '')} research",
                    ],
                    potential_tradeoffs=[]
                ))
    
    # Sort by gap ratio
    gaps.sort(key=lambda x: -x.gap_ratio)
    
    return gaps


# =============================================================================
# GEOGRAPHIC GAP ANALYSIS
# =============================================================================

def identify_geographic_gaps(
    df: pd.DataFrame,
    sdg_cols: List[str] = None,
    country_col: str = "Countries",
    sep: str = "; ",
    min_papers: int = 20,
    deviation_threshold: float = 0.3
) -> List[GeographicGap]:
    """
    Identify geographic research gaps.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with SDG indicators and country column.
    country_col : str
        Column with country names.
    min_papers : int
        Minimum papers for a region to be analyzed.
    deviation_threshold : float
        Minimum deviation from global average to flag.
    
    Returns
    -------
    List of GeographicGap objects.
    """
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    if country_col not in df.columns:
        warnings.warn(f"Country column '{country_col}' not found")
        return []
    
    sdg_numbers = [extract_sdg_number(col) for col in sdg_cols]
    
    # Compute global SDG distribution
    global_dist = {}
    total_sdg_papers = 0
    for col in sdg_cols:
        sdg_num = extract_sdg_number(col)
        count = df[col].sum()
        global_dist[sdg_num] = count
        total_sdg_papers += count
    
    # Normalize global distribution
    global_norm = {sdg: count / total_sdg_papers for sdg, count in global_dist.items()}
    
    # Analyze by country/region
    country_data = defaultdict(lambda: {"papers": 0, "sdg_counts": Counter()})
    
    for _, row in df.iterrows():
        countries_str = row.get(country_col, "")
        if pd.isna(countries_str) or not countries_str:
            continue
        
        countries = [c.strip() for c in str(countries_str).split(sep) if c.strip()]
        sdgs = get_paper_sdgs(row, sdg_cols)
        
        for country in set(countries):
            country_data[country]["papers"] += 1
            for sdg in sdgs:
                country_data[country]["sdg_counts"][sdg] += 1
    
    gaps = []
    total_papers = len(df)
    
    for country, data in country_data.items():
        if data["papers"] < min_papers:
            continue
        
        # Compute country SDG distribution
        country_total = sum(data["sdg_counts"].values())
        if country_total == 0:
            continue
        
        country_norm = {sdg: data["sdg_counts"].get(sdg, 0) / country_total 
                       for sdg in sdg_numbers}
        
        # Find deviations
        underrepresented = []
        overrepresented = []
        research_bias = {}
        
        for sdg in sdg_numbers:
            global_prop = global_norm.get(sdg, 0)
            country_prop = country_norm.get(sdg, 0)
            
            if global_prop > 0:
                deviation = (country_prop - global_prop) / global_prop
                research_bias[sdg] = deviation
                
                if deviation < -deviation_threshold:
                    underrepresented.append(sdg)
                elif deviation > deviation_threshold:
                    overrepresented.append(sdg)
        
        if underrepresented:
            gaps.append(GeographicGap(
                region=country,
                sdgs_underrepresented=sorted(underrepresented),
                sdgs_overrepresented=sorted(overrepresented),
                total_papers=data["papers"],
                global_share=data["papers"] / total_papers,
                research_focus_bias=research_bias,
                recommended_focus=underrepresented[:5]
            ))
    
    # Sort by total papers (more impactful gaps first)
    gaps.sort(key=lambda x: -x.total_papers)
    
    return gaps


# =============================================================================
# METHODOLOGICAL GAP ANALYSIS
# =============================================================================

# Common methodology keywords
METHODOLOGY_KEYWORDS = {
    "quantitative": ["regression", "statistical", "survey", "experiment", "quantitative", 
                     "correlation", "anova", "t-test", "chi-square", "sampling"],
    "qualitative": ["qualitative", "interview", "case study", "ethnograph", "grounded theory",
                    "phenomenolog", "narrative", "thematic analysis", "focus group"],
    "mixed_methods": ["mixed method", "triangulation", "sequential", "concurrent design"],
    "systematic_review": ["systematic review", "meta-analysis", "scoping review", "prisma",
                          "literature review"],
    "modeling": ["simulation", "agent-based", "system dynamics", "optimization", "model"],
    "machine_learning": ["machine learning", "deep learning", "neural network", "random forest",
                         "classification", "clustering", "nlp", "text mining"],
    "gis_spatial": ["gis", "spatial analysis", "remote sensing", "geospatial", "mapping"],
    "participatory": ["participatory", "stakeholder", "co-design", "community-based", "action research"],
}


def identify_methodological_gaps(
    df: pd.DataFrame,
    sdg_cols: List[str] = None,
    text_col: str = "Abstract",
    methods: Dict[str, List[str]] = None,
    min_usage: int = 10,
    gap_threshold: float = 0.3
) -> List[MethodologicalGap]:
    """
    Identify methodological gaps - methods rarely used for certain SDGs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with SDG indicators and text column.
    text_col : str
        Column containing abstract or methods text.
    methods : Dict
        Method categories and keywords. Uses defaults if None.
    min_usage : int
        Minimum papers using a method to analyze.
    gap_threshold : float
        Threshold for flagging under-use.
    
    Returns
    -------
    List of MethodologicalGap objects.
    """
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    if text_col not in df.columns:
        warnings.warn(f"Text column '{text_col}' not found")
        return []
    
    if methods is None:
        methods = METHODOLOGY_KEYWORDS
    
    sdg_numbers = [extract_sdg_number(col) for col in sdg_cols]
    
    # Detect methods in papers
    method_papers = defaultdict(set)  # method -> set of paper indices
    sdg_papers = defaultdict(set)  # sdg -> set of paper indices
    
    for idx, row in df.iterrows():
        text = str(row.get(text_col, "")).lower()
        paper_sdgs = get_paper_sdgs(row, sdg_cols)
        
        for sdg in paper_sdgs:
            sdg_papers[sdg].add(idx)
        
        for method_name, keywords in methods.items():
            if any(kw in text for kw in keywords):
                method_papers[method_name].add(idx)
    
    gaps = []
    
    for method_name, paper_indices in method_papers.items():
        if len(paper_indices) < min_usage:
            continue
        
        # Compute method usage by SDG
        usage_distribution = {}
        for sdg in sdg_numbers:
            sdg_set = sdg_papers.get(sdg, set())
            if len(sdg_set) > 0:
                overlap = len(paper_indices & sdg_set)
                usage_distribution[sdg] = overlap / len(sdg_set)
            else:
                usage_distribution[sdg] = 0
        
        # Find SDGs with low usage
        avg_usage = np.mean(list(usage_distribution.values()))
        
        rarely_using = []
        commonly_using = []
        
        for sdg, usage in usage_distribution.items():
            if sdg_papers.get(sdg, set()):  # Only if SDG has papers
                if usage < avg_usage * (1 - gap_threshold):
                    rarely_using.append(sdg)
                elif usage > avg_usage * (1 + gap_threshold):
                    commonly_using.append(sdg)
        
        if rarely_using:
            gaps.append(MethodologicalGap(
                method=method_name,
                sdgs_rarely_using=rarely_using,
                sdgs_commonly_using=commonly_using,
                total_usage=len(paper_indices),
                usage_distribution=usage_distribution,
                recommended_applications=[
                    f"Apply {method_name} to SDG {sdg} ({SDG_SHORT_NAMES.get(sdg, '')})"
                    for sdg in rarely_using[:3]
                ]
            ))
    
    return gaps


# =============================================================================
# TEMPORAL GAP ANALYSIS
# =============================================================================

def identify_temporal_gaps(
    df: pd.DataFrame,
    sdg_cols: List[str] = None,
    year_col: str = "Year",
    citations_col: str = "Cited by",
    decline_threshold: float = 0.3,
    recent_years: int = 3
) -> List[TemporalGap]:
    """
    Identify topics with declining research attention that may still be relevant.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with SDG indicators.
    year_col : str
        Year column.
    citations_col : str
        Citations column for relevance check.
    decline_threshold : float
        Minimum decline rate to flag.
    recent_years : int
        Years to consider as "recent".
    
    Returns
    -------
    List of TemporalGap objects.
    """
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    if year_col not in df.columns:
        warnings.warn(f"Year column '{year_col}' not found")
        return []
    
    df_clean = df.dropna(subset=[year_col]).copy()
    df_clean[year_col] = df_clean[year_col].astype(int)
    
    max_year = df_clean[year_col].max()
    recent_start = max_year - recent_years + 1
    
    gaps = []
    
    # Analyze each SDG
    for col in sdg_cols:
        sdg_num = extract_sdg_number(col)
        sdg_df = df_clean[df_clean[col] == 1]
        
        if len(sdg_df) < 20:
            continue
        
        # Yearly counts
        yearly_counts = sdg_df.groupby(year_col).size()
        
        if len(yearly_counts) < 3:
            continue
        
        # Find peak
        peak_year = yearly_counts.idxmax()
        peak_count = yearly_counts.max()
        
        # Recent count
        recent_mask = sdg_df[year_col] >= recent_start
        recent_count = recent_mask.sum()
        recent_avg = recent_count / recent_years
        
        # Peak period average (3 years around peak)
        peak_years = [y for y in yearly_counts.index if abs(y - peak_year) <= 1]
        peak_avg = yearly_counts[peak_years].mean()
        
        # Decline rate
        if peak_avg > 0:
            decline_rate = (peak_avg - recent_avg) / peak_avg
        else:
            decline_rate = 0
        
        if decline_rate >= decline_threshold and peak_year < recent_start:
            # Check if still relevant (based on citations)
            still_relevant = False
            if citations_col in df_clean.columns:
                recent_sdg = sdg_df[sdg_df[year_col] >= recent_start]
                if len(recent_sdg) > 0:
                    avg_citations = recent_sdg[citations_col].mean()
                    overall_avg = sdg_df[citations_col].mean()
                    still_relevant = avg_citations >= overall_avg * 0.5
            
            gaps.append(TemporalGap(
                topic=f"SDG {sdg_num}: {SDG_SHORT_NAMES.get(sdg_num, '')}",
                peak_year=int(peak_year),
                peak_count=int(peak_count),
                recent_count=int(recent_count),
                decline_rate=decline_rate,
                still_relevant=still_relevant,
                recommended_revival=[
                    f"Revisit SDG {sdg_num} with new methodologies",
                    f"Connect SDG {sdg_num} to emerging topics"
                ]
            ))
    
    # Sort by decline rate
    gaps.sort(key=lambda x: -x.decline_rate)
    
    return gaps


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def identify_research_gaps(
    df: pd.DataFrame,
    sdg_cols: List[str] = None,
    country_col: str = None,
    text_col: str = None,
    year_col: str = "Year",
    citations_col: str = "Cited by",
    analyze_sdg_combinations: bool = True,
    analyze_geographic: bool = True,
    analyze_methodological: bool = True,
    analyze_temporal: bool = True,
    verbose: bool = True
) -> ResearchGapsAnalysis:
    """
    Comprehensive research gaps analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with SDG indicators.
    sdg_cols : List[str]
        SDG columns. Auto-detected if None.
    country_col : str
        Country column for geographic analysis.
    text_col : str
        Text column for methodological analysis.
    year_col : str
        Year column for temporal analysis.
    analyze_* : bool
        Which analyses to run.
    verbose : bool
        Print progress.
    
    Returns
    -------
    ResearchGapsAnalysis
        Complete gaps analysis.
    """
    if verbose:
        print("="*60)
        print("RESEARCH GAPS ANALYSIS")
        print("="*60)
    
    if sdg_cols is None:
        sdg_cols = get_sdg_columns(df)
    
    if not sdg_cols:
        raise ValueError("No SDG columns found. Run identify_sdgs() first.")
    
    # Auto-detect columns
    if country_col is None:
        for candidate in ["Countries", "Countries of Authors", "Country"]:
            if candidate in df.columns:
                country_col = candidate
                break
    
    if text_col is None:
        for candidate in ["Abstract", "Processed Abstract", "Title"]:
            if candidate in df.columns:
                text_col = candidate
                break
    
    if verbose:
        print(f"Analyzing {len(df)} documents")
        print(f"SDG columns: {len(sdg_cols)}")
    
    # Run analyses
    sdg_combination_gaps = []
    geographic_gaps = []
    methodological_gaps = []
    temporal_gaps = []
    all_gaps = []
    
    # SDG Combinations
    if analyze_sdg_combinations:
        if verbose:
            print("\nIdentifying SDG combination gaps...")
        sdg_combination_gaps = identify_sdg_combination_gaps(df, sdg_cols)
        if verbose:
            print(f"  Found {len(sdg_combination_gaps)} combination gaps")
        
        # Convert to general gaps
        for gap in sdg_combination_gaps[:20]:
            all_gaps.append(ResearchGap(
                gap_type="sdg_combination",
                description=f"Under-researched: {gap.sdg_names[0]} + {gap.sdg_names[1]}",
                entities=list(gap.sdgs),
                current_count=gap.observed_count,
                expected_count=gap.expected_count,
                gap_score=min(gap.gap_ratio / 10, 1.0),
                priority_score=min(gap.gap_ratio / 10, 1.0) * 0.8,
                evidence={"gap_ratio": gap.gap_ratio, "p_value": gap.statistical_significance},
                recommendations=[
                    f"Investigate synergies between SDG {gap.sdgs[0]} and SDG {gap.sdgs[1]}",
                    f"Explore potential tradeoffs in {gap.sdg_names[0]}-{gap.sdg_names[1]} interactions"
                ]
            ))
    
    # Geographic
    if analyze_geographic and country_col:
        if verbose:
            print("\nIdentifying geographic gaps...")
        geographic_gaps = identify_geographic_gaps(df, sdg_cols, country_col)
        if verbose:
            print(f"  Found {len(geographic_gaps)} geographic gaps")
        
        for gap in geographic_gaps[:10]:
            all_gaps.append(ResearchGap(
                gap_type="geographic",
                description=f"{gap.region}: under-researching {len(gap.sdgs_underrepresented)} SDGs",
                entities=gap.sdgs_underrepresented,
                current_count=gap.total_papers,
                expected_count=0,
                gap_score=len(gap.sdgs_underrepresented) / 17,
                priority_score=gap.global_share * len(gap.sdgs_underrepresented) / 17,
                evidence={"global_share": gap.global_share},
                recommendations=[
                    f"Increase {gap.region} research focus on SDG {gap.sdgs_underrepresented[0]}" 
                    if gap.sdgs_underrepresented else ""
                ]
            ))
    
    # Methodological
    if analyze_methodological and text_col:
        if verbose:
            print("\nIdentifying methodological gaps...")
        methodological_gaps = identify_methodological_gaps(df, sdg_cols, text_col)
        if verbose:
            print(f"  Found {len(methodological_gaps)} methodological gaps")
        
        for gap in methodological_gaps:
            all_gaps.append(ResearchGap(
                gap_type="methodological",
                description=f"{gap.method} rarely used for SDGs: {[SDG_SHORT_NAMES.get(s, s) for s in gap.sdgs_rarely_using[:3]]}",
                entities=gap.sdgs_rarely_using,
                current_count=gap.total_usage,
                expected_count=0,
                gap_score=len(gap.sdgs_rarely_using) / 17,
                priority_score=0.5 * len(gap.sdgs_rarely_using) / 17,
                evidence={"usage_distribution": gap.usage_distribution},
                recommendations=gap.recommended_applications
            ))
    
    # Temporal
    if analyze_temporal and year_col in df.columns:
        if verbose:
            print("\nIdentifying temporal gaps...")
        temporal_gaps = identify_temporal_gaps(df, sdg_cols, year_col, citations_col)
        if verbose:
            print(f"  Found {len(temporal_gaps)} temporal gaps")
        
        for gap in temporal_gaps:
            priority = gap.decline_rate * (0.7 if gap.still_relevant else 0.3)
            all_gaps.append(ResearchGap(
                gap_type="temporal",
                description=f"{gap.topic} declining ({gap.decline_rate:.0%} from peak)",
                entities=[gap.topic],
                current_count=gap.recent_count,
                expected_count=gap.peak_count,
                gap_score=gap.decline_rate,
                priority_score=priority,
                evidence={"peak_year": gap.peak_year, "Still Relevant": gap.still_relevant},
                recommendations=gap.recommended_revival
            ))
    
    # Sort all gaps by priority
    all_gaps.sort(key=lambda x: -x.priority_score)
    top_priority = all_gaps[:20]
    
    # Build SDG gap matrix
    sdg_numbers = sorted([extract_sdg_number(col) for col in sdg_cols])
    gap_matrix = pd.DataFrame(0.0, index=sdg_numbers, columns=sdg_numbers)
    
    for gap in sdg_combination_gaps:
        if len(gap.sdgs) == 2:
            i, j = gap.sdgs
            gap_matrix.loc[i, j] = gap.gap_ratio
            gap_matrix.loc[j, i] = gap.gap_ratio
    
    # Summary
    gap_summary = {
        "sdg_combination": len(sdg_combination_gaps),
        "geographic": len(geographic_gaps),
        "methodological": len(methodological_gaps),
        "temporal": len(temporal_gaps),
        "total": len(all_gaps)
    }
    
    if verbose:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nGaps Found:")
        for gap_type, count in gap_summary.items():
            print(f"  {gap_type}: {count}")
        
        print(f"\nTop Priority Gaps:")
        for gap in top_priority[:5]:
            print(f"  - {gap.description}")
    
    return ResearchGapsAnalysis(
        sdg_combination_gaps=sdg_combination_gaps,
        geographic_gaps=geographic_gaps,
        methodological_gaps=methodological_gaps,
        temporal_gaps=temporal_gaps,
        all_gaps=all_gaps,
        top_priority_gaps=top_priority,
        gap_summary=gap_summary,
        sdg_gap_matrix=gap_matrix,
        geographic_sdg_matrix=pd.DataFrame()  # TODO
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_sdg_gap_matrix(
    analysis: ResearchGapsAnalysis,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "YlOrRd",
    title: str = "SDG Combination Research Gaps",
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """Plot SDG gap matrix."""
    matrix = analysis.sdg_gap_matrix
    
    # Add names to index
    new_index = [f"SDG {i}" for i in matrix.index]
    plot_matrix = matrix.copy()
    plot_matrix.index = new_index
    plot_matrix.columns = new_index
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    # Mask diagonal
    mask = np.eye(len(matrix), dtype=bool)
    
    sns.heatmap(
        plot_matrix,
        cmap=cmap,
        mask=mask,
        annot=True,
        fmt=".1f",
        ax=ax,
        cbar_kws={"label": "Gap Ratio (higher = bigger gap)"}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_gap_summary(
    analysis: ResearchGapsAnalysis,
    figsize: Tuple[int, int] = (12, 8),
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """Plot top priority gaps."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    top_gaps = analysis.top_priority_gaps[:15]
    
    if not top_gaps:
        ax.text(0.5, 0.5, "No gaps identified", ha='center', va='center', fontsize=14)
        return fig
    
    descs = [g.description[:60] + "..." if len(g.description) > 60 else g.description 
             for g in top_gaps]
    scores = [g.priority_score for g in top_gaps]
    type_colors = {
        "sdg_combination": '#e41a1c',
        "geographic": '#377eb8',
        "methodological": '#4daf4a',
        "temporal": '#984ea3'
    }
    colors = [type_colors.get(g.gap_type, '#888888') for g in top_gaps]
    
    bars = ax.barh(range(len(top_gaps)), scores, color=colors)
    ax.set_yticks(range(len(top_gaps)))
    ax.set_yticklabels(descs, fontsize=9)
    ax.set_xlabel("Priority Score")
    ax.set_title("Top Research Gaps by Priority", fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    # Legend for gap types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.replace('_', ' ').title()) 
                       for t, c in type_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_gap_analysis(
    ba,  # BiblioAnalysis object
    verbose: bool = True
) -> ResearchGapsAnalysis:
    """
    Convenience function to run research gaps analysis.
    
    Usage:
        from biblium.addons.research_gaps import run_gap_analysis
        
        ba.identify_sdgs()
        gaps = run_gap_analysis(ba)
        
        # View results
        print(gaps.get_sdg_gaps_df())
        print(gaps.get_all_gaps_df())
        print(gaps.get_top_recommendations())
    """
    sdg_cols = get_sdg_columns(ba.df)
    if not sdg_cols:
        raise ValueError("No SDG columns found. Run ba.identify_sdgs() first.")
    
    return identify_research_gaps(
        ba.df,
        sdg_cols=sdg_cols,
        verbose=verbose
    )
