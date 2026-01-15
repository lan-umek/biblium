# -*- coding: utf-8 -*-
"""
Citation Pattern Classification for Biblium
============================================

Classify papers based on their citation trajectory over time:
- Evergreen/Classic: Sustained citations over many years
- Flash-in-the-pan: Quick burst then rapid decline  
- Delayed Recognition: Initially ignored, later discovered
- Sleeping Beauty: Extreme delayed recognition
- Normal: Typical citation decay pattern

Data Sources:
- OpenAlex API: Provides actual yearly citation counts (recommended)
- Estimation: Approximates patterns from total citations and age (less accurate)

@author: Lan.Umek
@version: 2.9.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import warnings
import time

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class CitationPattern(Enum):
    """Citation trajectory pattern types."""
    EVERGREEN = "Evergreen"
    FLASH_IN_THE_PAN = "Flash-in-the-pan"
    DELAYED_RECOGNITION = "Delayed Recognition"
    SLEEPING_BEAUTY = "Sleeping Beauty"
    NORMAL = "Normal"
    UNCITED = "Uncited"
    TOO_RECENT = "Too Recent"


@dataclass
class CitationTrajectory:
    """Citation trajectory data for a single paper."""
    doi: str
    title: str
    pub_year: int
    total_citations: int
    counts_by_year: Dict[int, int]  # {year: count}
    
    # Computed metrics
    peak_year: int = 0
    peak_citations: int = 0
    years_to_peak: int = 0
    citation_half_life: float = 0.0
    decay_rate: float = 0.0
    early_citations_pct: float = 0.0  # % in first 3 years
    late_citations_pct: float = 0.0   # % after year 5
    awakening_year: Optional[int] = None
    pattern: CitationPattern = CitationPattern.NORMAL
    confidence: float = 1.0  # 1.0 for OpenAlex, lower for estimated
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "DOI": self.doi,
            "Title": self.title[:80] + "..." if len(self.title) > 80 else self.title,
            "Year": self.pub_year,
            "Total Citations": self.total_citations,
            "Pattern": self.pattern.value,
            "Peak Year": self.peak_year,
            "Peak Citations": self.peak_citations,
            "Years to Peak": self.years_to_peak,
            "Half-life": round(self.citation_half_life, 1),
            "Decay Rate": round(self.decay_rate, 3),
            "Early Citations %": round(self.early_citations_pct, 1),
            "Late Citations %": round(self.late_citations_pct, 1),
            "Confidence": round(self.confidence, 2),
        }


@dataclass
class CitationPatternResult:
    """Complete citation pattern analysis result."""
    trajectories: List[CitationTrajectory]
    pattern_counts: Dict[str, int]
    data_source: str  # "openalex" or "estimated"
    n_papers: int
    n_analyzed: int
    current_year: int
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all trajectories to DataFrame."""
        if not self.trajectories:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trajectories])
    
    def get_pattern_summary(self) -> pd.DataFrame:
        """Get summary by pattern type."""
        data = []
        total = sum(self.pattern_counts.values())
        for pattern, count in sorted(self.pattern_counts.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            data.append({
                "Pattern": pattern,
                "Count": count,
                "Percentage": round(pct, 1),
            })
        return pd.DataFrame(data)
    
    def get_papers_by_pattern(self, pattern: Union[str, CitationPattern]) -> List[CitationTrajectory]:
        """Get all papers with a specific pattern."""
        if isinstance(pattern, str):
            pattern = CitationPattern(pattern)
        return [t for t in self.trajectories if t.pattern == pattern]
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "CITATION PATTERN CLASSIFICATION",
            "=" * 50,
            f"Data source: {self.data_source.upper()}",
            f"Papers analyzed: {self.n_analyzed} / {self.n_papers}",
            f"Reference year: {self.current_year}",
            "",
            "PATTERN DISTRIBUTION",
            "-" * 50,
        ]
        
        total = sum(self.pattern_counts.values())
        for pattern, count in sorted(self.pattern_counts.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 5)
            lines.append(f"  {pattern:20s}: {count:5d} ({pct:5.1f}%) {bar}")
        
        if self.data_source == "estimated":
            lines.extend([
                "",
                "⚠️  WARNING: Results are ESTIMATED from total citations.",
                "    For accurate classification, use OpenAlex data source.",
            ])
        
        return "\n".join(lines)


# =============================================================================
# OPENALEX FETCHING
# =============================================================================

def fetch_citation_history_openalex(
    df: pd.DataFrame,
    doi_col: str = None,
    title_col: str = None,
    max_papers: int = 500,
    delay: float = 0.1,
    verbose: bool = True,
) -> Dict[str, Dict[int, int]]:
    """
    Fetch citation history from OpenAlex API.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with DOIs or titles.
    doi_col : str, optional
        Column containing DOIs.
    title_col : str, optional
        Column containing titles (fallback).
    max_papers : int
        Maximum papers to fetch.
    delay : float
        Delay between API calls.
    verbose : bool
        Print progress.
        
    Returns
    -------
    dict
        {doi_or_id: {year: citation_count}}
    """
    import urllib.request
    import urllib.parse
    import json
    
    # Find DOI column
    if doi_col is None:
        doi_candidates = ["DOI", "doi", "DI", "Doi"]
        for c in doi_candidates:
            if c in df.columns:
                doi_col = c
                break
    
    # Find title column
    if title_col is None:
        title_candidates = ["Title", "TI", "Article Title", "Document Title"]
        for c in title_candidates:
            if c in df.columns:
                title_col = c
                break
    
    results = {}
    papers_to_fetch = df.head(max_papers)
    
    if verbose:
        print(f"Fetching citation history from OpenAlex for up to {len(papers_to_fetch)} papers...")
    
    for idx, row in papers_to_fetch.iterrows():
        # Try DOI first
        doi = None
        if doi_col and doi_col in row and pd.notna(row[doi_col]):
            doi = str(row[doi_col]).strip()
            if doi:
                # Clean DOI
                doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        
        # Build query
        if doi:
            url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}"
            key = doi
        elif title_col and title_col in row and pd.notna(row[title_col]):
            title = str(row[title_col]).strip()
            if not title:
                continue
            encoded_title = urllib.parse.quote(title[:200])
            url = f"https://api.openalex.org/works?filter=title.search:{encoded_title}&per_page=1"
            key = title[:100]
        else:
            continue
        
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Biblium/2.9 (mailto:research@example.com)"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            # Handle search results vs direct lookup
            if "results" in data:
                if data["results"]:
                    work = data["results"][0]
                else:
                    continue
            else:
                work = data
            
            # Extract counts_by_year
            counts_by_year = {}
            if "counts_by_year" in work:
                for entry in work["counts_by_year"]:
                    year = entry.get("year")
                    count = entry.get("cited_by_count", 0)
                    if year:
                        counts_by_year[year] = count
            
            if counts_by_year:
                results[key] = counts_by_year
            
            time.sleep(delay)
            
        except Exception as e:
            if verbose and idx % 50 == 0:
                print(f"  Warning at paper {idx}: {e}")
            continue
        
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1} papers, {len(results)} with citation history")
    
    if verbose:
        print(f"  ✓ Retrieved citation history for {len(results)} papers")
    
    return results


# =============================================================================
# PATTERN CLASSIFICATION
# =============================================================================

def compute_trajectory_metrics(
    counts_by_year: Dict[int, int],
    pub_year: int,
    current_year: int,
) -> dict:
    """
    Compute citation trajectory metrics from yearly counts.
    
    Parameters
    ----------
    counts_by_year : dict
        {year: citation_count}
    pub_year : int
        Publication year.
    current_year : int
        Current/reference year.
        
    Returns
    -------
    dict
        Trajectory metrics.
    """
    if not counts_by_year:
        return {
            "peak_year": pub_year,
            "peak_citations": 0,
            "years_to_peak": 0,
            "citation_half_life": 0.0,
            "decay_rate": 0.0,
            "early_citations_pct": 0.0,
            "late_citations_pct": 0.0,
            "awakening_year": None,
        }
    
    # Get citations in chronological order
    years = sorted(counts_by_year.keys())
    total = sum(counts_by_year.values())
    
    if total == 0:
        return {
            "peak_year": pub_year,
            "peak_citations": 0,
            "years_to_peak": 0,
            "citation_half_life": 0.0,
            "decay_rate": 0.0,
            "early_citations_pct": 0.0,
            "late_citations_pct": 0.0,
            "awakening_year": None,
        }
    
    # Find peak year
    peak_year = max(counts_by_year.keys(), key=lambda y: counts_by_year[y])
    peak_citations = counts_by_year[peak_year]
    years_to_peak = peak_year - pub_year
    
    # Early vs late citations
    early_citations = sum(counts_by_year.get(y, 0) for y in range(pub_year, pub_year + 4))
    late_citations = sum(counts_by_year.get(y, 0) for y in range(pub_year + 5, current_year + 1))
    
    early_pct = (early_citations / total * 100) if total > 0 else 0
    late_pct = (late_citations / total * 100) if total > 0 else 0
    
    # Citation half-life (years until 50% of citations received)
    cumsum = 0
    half_life = 0
    for year in sorted(counts_by_year.keys()):
        cumsum += counts_by_year[year]
        if cumsum >= total / 2:
            half_life = year - pub_year
            break
    
    # Decay rate (after peak)
    decay_rate = 0.0
    post_peak_years = [y for y in years if y > peak_year]
    if len(post_peak_years) >= 2 and peak_citations > 0:
        post_peak_counts = [counts_by_year[y] for y in post_peak_years]
        if post_peak_counts[0] > 0:
            # Simple exponential decay estimate
            decay_rate = 1 - (post_peak_counts[-1] / post_peak_counts[0]) ** (1 / len(post_peak_years))
    
    # Awakening detection (for delayed recognition)
    awakening_year = None
    if years_to_peak >= 5:
        # Look for year when citations first exceed threshold
        threshold = max(2, total * 0.1)
        cumsum = 0
        for year in sorted(counts_by_year.keys()):
            cumsum += counts_by_year[year]
            if counts_by_year[year] >= threshold and year > pub_year + 3:
                awakening_year = year
                break
    
    return {
        "peak_year": peak_year,
        "peak_citations": peak_citations,
        "years_to_peak": years_to_peak,
        "citation_half_life": half_life,
        "decay_rate": decay_rate,
        "early_citations_pct": early_pct,
        "late_citations_pct": late_pct,
        "awakening_year": awakening_year,
    }


def classify_pattern(
    metrics: dict,
    total_citations: int,
    paper_age: int,
) -> Tuple[CitationPattern, float]:
    """
    Classify citation pattern based on trajectory metrics.
    
    Parameters
    ----------
    metrics : dict
        Trajectory metrics from compute_trajectory_metrics.
    total_citations : int
        Total citation count.
    paper_age : int
        Years since publication.
        
    Returns
    -------
    tuple
        (CitationPattern, confidence)
    """
    # Handle edge cases
    if total_citations == 0:
        return CitationPattern.UNCITED, 1.0
    
    if paper_age < 3:
        return CitationPattern.TOO_RECENT, 1.0
    
    years_to_peak = metrics["years_to_peak"]
    early_pct = metrics["early_citations_pct"]
    late_pct = metrics["late_citations_pct"]
    decay_rate = metrics["decay_rate"]
    half_life = metrics["citation_half_life"]
    awakening_year = metrics["awakening_year"]
    
    # Classification rules
    confidence = 0.8
    
    # Sleeping Beauty: Very delayed recognition
    if awakening_year and years_to_peak >= 8 and early_pct < 10:
        return CitationPattern.SLEEPING_BEAUTY, 0.9
    
    # Delayed Recognition: Moderate delay
    if years_to_peak >= 5 and early_pct < 25 and late_pct > 50:
        return CitationPattern.DELAYED_RECOGNITION, 0.85
    
    # Flash-in-the-pan: Quick burst, rapid decline
    if years_to_peak <= 2 and early_pct > 60 and decay_rate > 0.3:
        return CitationPattern.FLASH_IN_THE_PAN, 0.85
    
    # Evergreen: Sustained citations
    if half_life >= 5 and late_pct >= 30 and decay_rate < 0.15:
        return CitationPattern.EVERGREEN, 0.85
    
    # Also check for steady accumulation
    if paper_age >= 8 and late_pct >= 25 and early_pct < 50:
        return CitationPattern.EVERGREEN, 0.75
    
    # Normal: Default
    return CitationPattern.NORMAL, 0.7


def estimate_pattern_from_totals(
    total_citations: int,
    pub_year: int,
    current_year: int,
    field_avg_citations: float = None,
) -> Tuple[CitationPattern, Dict[int, int], float]:
    """
    Estimate citation pattern when yearly data unavailable.
    
    This is a rough approximation based on typical citation curves.
    
    Parameters
    ----------
    total_citations : int
        Total citation count.
    pub_year : int
        Publication year.
    current_year : int
        Current year.
    field_avg_citations : float, optional
        Average citations for papers of similar age.
        
    Returns
    -------
    tuple
        (pattern, estimated_counts_by_year, confidence)
    """
    paper_age = current_year - pub_year
    
    if total_citations == 0:
        return CitationPattern.UNCITED, {}, 0.5
    
    if paper_age < 3:
        return CitationPattern.TOO_RECENT, {}, 0.3
    
    # Create estimated distribution (typical decay curve)
    # Most papers peak 2-3 years after publication
    estimated_counts = {}
    
    if paper_age <= 5:
        # Recent paper - assume front-loaded
        weights = [0.15, 0.25, 0.25, 0.20, 0.15][:paper_age]
    else:
        # Older paper - typical decay curve
        weights = []
        for i in range(paper_age):
            # Exponential decay with peak at year 2
            if i <= 2:
                w = 0.15 + 0.1 * i
            else:
                w = 0.35 * (0.8 ** (i - 2))
            weights.append(max(0.02, w))
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Distribute citations
    for i, w in enumerate(weights):
        year = pub_year + i
        estimated_counts[year] = int(total_citations * w)
    
    # Adjust to match total
    diff = total_citations - sum(estimated_counts.values())
    if diff != 0 and estimated_counts:
        peak_year = max(estimated_counts.keys(), key=lambda y: estimated_counts[y])
        estimated_counts[peak_year] += diff
    
    # Simple classification based on age and citations
    citations_per_year = total_citations / paper_age if paper_age > 0 else 0
    
    if paper_age >= 10 and citations_per_year >= 2:
        pattern = CitationPattern.EVERGREEN
        confidence = 0.4
    elif paper_age >= 8 and citations_per_year < 0.5:
        pattern = CitationPattern.FLASH_IN_THE_PAN
        confidence = 0.35
    elif paper_age >= 10 and total_citations > 50:
        # Could be delayed recognition
        pattern = CitationPattern.DELAYED_RECOGNITION
        confidence = 0.3
    else:
        pattern = CitationPattern.NORMAL
        confidence = 0.5
    
    return pattern, estimated_counts, confidence


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def _parse_counts_by_year(value) -> Dict[int, int]:
    """
    Parse counts_by_year from various formats.
    
    Handles:
    - List of dicts: [{"year": 2023, "cited_by_count": 5}, ...]
    - Dict: {2023: 5, 2022: 3, ...}
    - JSON string
    """
    import json
    
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    
    # Already a dict
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    
    # String - try to parse as JSON
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except:
            return {}
    
    # List of dicts (OpenAlex format)
    if isinstance(value, list):
        result = {}
        for item in value:
            if isinstance(item, dict):
                year = item.get("year")
                count = item.get("cited_by_count", item.get("count", 0))
                if year:
                    result[int(year)] = int(count)
        return result
    
    return {}


def _parse_split_counts_by_year(years_str, counts_str) -> Dict[int, int]:
    """
    Parse counts_by_year from OpenAlex split format.
    
    OpenAlex datasets may have two columns:
    - counts_by_year.year: "2025|2024|2023|2022"
    - counts_by_year.cited_by_count: "100|200|150|50"
    
    Parameters
    ----------
    years_str : str
        Pipe-separated year values.
    counts_str : str
        Pipe-separated citation count values.
        
    Returns
    -------
    dict
        {year: citation_count}
    """
    if not years_str or not counts_str:
        return {}
    
    try:
        years_str = str(years_str).strip()
        counts_str = str(counts_str).strip()
        
        # Handle pipe-separated format
        if '|' in years_str:
            years = years_str.split('|')
            counts = counts_str.split('|')
        # Handle comma-separated format
        elif ',' in years_str:
            years = years_str.split(',')
            counts = counts_str.split(',')
        # Handle semicolon-separated format
        elif ';' in years_str:
            years = years_str.split(';')
            counts = counts_str.split(';')
        else:
            # Single value
            years = [years_str]
            counts = [counts_str]
        
        result = {}
        for y, c in zip(years, counts):
            try:
                year = int(float(y.strip()))
                count = int(float(c.strip()))
                result[year] = count
            except (ValueError, TypeError):
                continue
        
        return result
    except Exception:
        return {}


def analyze_citation_patterns(
    df: pd.DataFrame,
    use_openalex: bool = True,
    max_papers: int = 500,
    current_year: int = None,
    min_age: int = 3,
    verbose: bool = True,
    stop_flag: callable = None,
) -> CitationPatternResult:
    """
    Analyze citation patterns for papers in dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliometric dataset.
    use_openalex : bool
        Fetch actual citation history from OpenAlex API.
        Ignored if dataset already contains counts_by_year column.
    max_papers : int
        Maximum papers to analyze (for API limits).
    current_year : int, optional
        Reference year. Defaults to current year.
    min_age : int
        Minimum paper age to analyze.
    verbose : bool
        Print progress.
    stop_flag : callable, optional
        Function that returns True if analysis should stop.
        
    Returns
    -------
    CitationPatternResult
        Complete analysis result.
        
    Notes
    -----
    If the dataset already contains a 'counts_by_year' column (e.g., from OpenAlex),
    the function will use that data directly without making API calls.
    """
    import datetime
    
    # Helper to check stop
    def should_stop():
        return stop_flag is not None and stop_flag()
    
    if current_year is None:
        current_year = datetime.datetime.now().year
    
    # Find required columns
    year_col = None
    for c in ["Year", "PY", "Publication Year", "publication_year"]:
        if c in df.columns:
            year_col = c
            break
    
    cite_col = None
    for c in ["Cited by", "Times Cited", "Citations", "TC", "Cited By", "cited_by_count"]:
        if c in df.columns:
            cite_col = c
            break
    
    doi_col = None
    for c in ["DOI", "doi", "DI"]:
        if c in df.columns:
            doi_col = c
            break
    
    title_col = None
    for c in ["Title", "TI", "Article Title", "Document Title", "title"]:
        if c in df.columns:
            title_col = c
            break
    
    # Check for embedded counts_by_year column (OpenAlex datasets)
    counts_by_year_col = None
    for c in ["counts_by_year", "Counts by Year", "citation_counts_by_year"]:
        if c in df.columns:
            counts_by_year_col = c
            break
    
    # Check for OpenAlex split format: counts_by_year.year and counts_by_year.cited_by_count
    # Note: BiblioPlot may rename counts_by_year.cited_by_count to "Citations by Year"
    counts_year_col = None
    counts_cited_col = None
    for c in df.columns:
        c_lower = c.lower()
        if c_lower == "counts_by_year.year":
            counts_year_col = c
        elif c_lower == "counts_by_year.cited_by_count":
            counts_cited_col = c
        elif c_lower == "citations by year" and counts_cited_col is None:
            # BiblioPlot renames counts_by_year.cited_by_count to "Citations by Year"
            counts_cited_col = c
    
    has_split_counts = counts_year_col is not None and counts_cited_col is not None
    
    if year_col is None:
        raise ValueError("Year column not found")
    
    # Determine data source
    has_embedded_counts = False
    if counts_by_year_col:
        # Check if column has actual data
        sample = df[counts_by_year_col].dropna().head(10)
        if len(sample) > 0:
            parsed = [_parse_counts_by_year(v) for v in sample]
            if any(p for p in parsed):  # At least one non-empty dict
                has_embedded_counts = True
    
    if verbose:
        print(f"Analyzing citation patterns...")
        print(f"  Year column: {year_col}")
        print(f"  Citation column: {cite_col}")
        print(f"  DOI column: {doi_col}")
        if has_embedded_counts:
            print(f"  ✓ Found embedded counts_by_year column: {counts_by_year_col}")
            print(f"    (Using dataset citation history - no API calls needed)")
        elif has_split_counts:
            print(f"  ✓ Found OpenAlex split format (counts_by_year.year + counts_by_year.cited_by_count)")
    
    # Fetch citation history from OpenAlex if requested AND not already in dataset
    citation_history = {}
    data_source = "estimated"
    
    if has_embedded_counts:
        # Use embedded data - no API needed
        data_source = "dataset"
        if verbose:
            print(f"  Using citation history from dataset")
    elif has_split_counts:
        # Use split format data - no API needed
        data_source = "dataset"
        if verbose:
            print(f"  Using citation history from dataset (split format)")
    elif use_openalex and (doi_col or title_col):
        try:
            citation_history = fetch_citation_history_openalex(
                df, doi_col=doi_col, title_col=title_col,
                max_papers=max_papers, verbose=verbose
            )
            if citation_history:
                data_source = "openalex"
        except Exception as e:
            if verbose:
                print(f"  ⚠️ OpenAlex fetch failed: {e}")
                print(f"  Falling back to estimation...")
    
    # Check stop before processing
    if should_stop():
        if verbose:
            print("  ⏹ Analysis stopped by user")
        return CitationPatternResult(
            trajectories=[], pattern_counts={p.value: 0 for p in CitationPattern},
            data_source=data_source, n_papers=len(df), n_analyzed=0,
            current_year=current_year
        )
    
    # Process papers
    trajectories = []
    pattern_counts = {p.value: 0 for p in CitationPattern}
    
    for idx, row in df.iterrows():
        # Check stop flag periodically
        if idx % 100 == 0 and should_stop():
            if verbose:
                print(f"  ⏹ Analysis stopped by user at paper {idx}")
            break
        
        # Get basic info
        try:
            pub_year = int(row[year_col])
        except:
            continue
        
        paper_age = current_year - pub_year
        if paper_age < min_age:
            continue
        
        total_cites = 0
        if cite_col and pd.notna(row.get(cite_col)):
            try:
                total_cites = int(row[cite_col])
            except:
                pass
        
        doi = ""
        if doi_col and pd.notna(row.get(doi_col)):
            doi = str(row[doi_col]).strip()
            doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        
        title = ""
        if title_col and pd.notna(row.get(title_col)):
            title = str(row[title_col]).strip()
        
        # Get citation history - try embedded first, then split format, then API results, then estimate
        counts_by_year = None
        confidence = 0.5
        
        # 1. Try embedded counts_by_year column
        if has_embedded_counts and counts_by_year_col:
            embedded = row.get(counts_by_year_col)
            if pd.notna(embedded):
                counts_by_year = _parse_counts_by_year(embedded)
                if counts_by_year:
                    confidence = 1.0
        
        # 2. Try OpenAlex split format
        if not counts_by_year and has_split_counts:
            years_str = row.get(counts_year_col)
            counts_str = row.get(counts_cited_col)
            if pd.notna(years_str) and pd.notna(counts_str):
                counts_by_year = _parse_split_counts_by_year(years_str, counts_str)
                if counts_by_year:
                    confidence = 1.0
        
        # 3. Try API-fetched citation history
        if not counts_by_year:
            key = doi if doi else title[:100]
            if key in citation_history:
                counts_by_year = citation_history[key]
                confidence = 1.0
        
        # 4. Compute metrics and classify
        if counts_by_year:
            # Update total_cites from counts_by_year if not set
            if total_cites == 0:
                total_cites = sum(counts_by_year.values())
            
            metrics = compute_trajectory_metrics(counts_by_year, pub_year, current_year)
            pattern, conf = classify_pattern(metrics, total_cites, paper_age)
            confidence *= conf
        else:
            # Estimate from totals
            pattern, counts_by_year, confidence = estimate_pattern_from_totals(
                total_cites, pub_year, current_year
            )
            metrics = compute_trajectory_metrics(counts_by_year, pub_year, current_year)
        
        # Create trajectory object
        traj = CitationTrajectory(
            doi=doi,
            title=title,
            pub_year=pub_year,
            total_citations=total_cites,
            counts_by_year=counts_by_year,
            peak_year=metrics["peak_year"],
            peak_citations=metrics["peak_citations"],
            years_to_peak=metrics["years_to_peak"],
            citation_half_life=metrics["citation_half_life"],
            decay_rate=metrics["decay_rate"],
            early_citations_pct=metrics["early_citations_pct"],
            late_citations_pct=metrics["late_citations_pct"],
            awakening_year=metrics["awakening_year"],
            pattern=pattern,
            confidence=confidence,
        )
        
        trajectories.append(traj)
        pattern_counts[pattern.value] += 1
    
    if verbose:
        print(f"  ✓ Analyzed {len(trajectories)} papers")
        if data_source == "estimated":
            print(f"  ⚠️ WARNING: Results are ESTIMATED. Use OpenAlex for accurate classification.")
    
    return CitationPatternResult(
        trajectories=trajectories,
        pattern_counts=pattern_counts,
        data_source=data_source,
        n_papers=len(df),
        n_analyzed=len(trajectories),
        current_year=current_year,
    )


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_pattern_distribution(
    result: CitationPatternResult,
    figsize: Tuple[float, float] = (10, 6),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Plot distribution of citation patterns as horizontal bar chart.
    
    Parameters
    ----------
    result : CitationPatternResult
        Analysis result.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    filename : str, optional
        Save to file.
    dpi : int
        Resolution.
        
    Returns
    -------
    tuple
        (fig, ax)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Prepare data - exclude zero counts, sort descending
    patterns = []
    counts = []
    for pattern, count in sorted(result.pattern_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            patterns.append(pattern)
            counts.append(count)
    
    if not patterns:
        raise ValueError("No patterns to plot")
    
    # Reverse so highest count is at top of horizontal bar chart
    patterns = patterns[::-1]
    counts = counts[::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(patterns))
    ax.barh(y_pos, counts, color="steelblue", edgecolor="none")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(patterns)
    ax.set_xlabel("Number of Documents", fontsize=11)
    ax.set_ylabel("Citation Pattern", fontsize=11)
    
    # Add count labels
    total = sum(counts)
    for i, (count, pattern) in enumerate(zip(counts, patterns)):
        pct = count / total * 100
        ax.text(count + max(counts) * 0.01, i, f" {count} ({pct:.1f}%)", 
               va="center", fontsize=9)
    
    # Extend x-axis for labels
    ax.set_xlim(0, max(counts) * 1.2)
    
    title_text = title or "Citation Pattern Distribution"
    if result.data_source == "estimated":
        title_text += "\n(Estimated - Use OpenAlex for Accurate Results)"
    ax.set_title(title_text, fontsize=12, fontweight="bold")
    
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_pattern_by_year(
    result: CitationPatternResult,
    figsize: Tuple[float, float] = (12, 6),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Plot pattern distribution by publication year as stacked area chart.
    
    Parameters
    ----------
    result : CitationPatternResult
        Analysis result.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    filename : str, optional
        Save to file.
    dpi : int
        Resolution.
        
    Returns
    -------
    tuple
        (fig, ax)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Build year-pattern matrix
    year_pattern = {}
    for traj in result.trajectories:
        year = traj.pub_year
        pattern = traj.pattern.value
        if year not in year_pattern:
            year_pattern[year] = {}
        year_pattern[year][pattern] = year_pattern[year].get(pattern, 0) + 1
    
    if not year_pattern:
        raise ValueError("No data to plot")
    
    years = sorted(year_pattern.keys())
    patterns = [p.value for p in CitationPattern if any(p.value in year_pattern.get(y, {}) for y in years)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create stacked bar chart
    bottom = np.zeros(len(years))
    colors = plt.cm.Set2(np.linspace(0, 1, len(patterns)))
    
    for pattern, color in zip(patterns, colors):
        values = [year_pattern.get(y, {}).get(pattern, 0) for y in years]
        ax.bar(years, values, bottom=bottom, label=pattern, color=color, edgecolor="none", width=0.8)
        bottom += values
    
    ax.set_xlabel("Publication Year", fontsize=11)
    ax.set_ylabel("Number of Documents", fontsize=11)
    
    title_text = title or "Citation Patterns by Publication Year"
    if result.data_source == "estimated":
        title_text += "\n(Estimated)"
    ax.set_title(title_text, fontsize=12, fontweight="bold")
    
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_trajectory_examples(
    result: CitationPatternResult,
    pattern: Union[str, CitationPattern] = None,
    n_examples: int = 5,
    figsize: Tuple[float, float] = (10, 6),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Plot example citation trajectories for a specific pattern.
    
    Parameters
    ----------
    result : CitationPatternResult
        Analysis result.
    pattern : str or CitationPattern, optional
        Pattern to show examples for. If None, shows top cited.
    n_examples : int
        Number of examples to plot.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    filename : str, optional
        Save to file.
    dpi : int
        Resolution.
        
    Returns
    -------
    tuple
        (fig, ax)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Select papers
    if pattern:
        if isinstance(pattern, str):
            pattern = CitationPattern(pattern)
        papers = [t for t in result.trajectories if t.pattern == pattern]
        pattern_name = pattern.value
    else:
        papers = result.trajectories
        pattern_name = "Top Cited"
    
    # Sort by total citations and take top n
    papers = sorted(papers, key=lambda x: -x.total_citations)[:n_examples]
    
    if not papers:
        raise ValueError(f"No papers found for pattern: {pattern_name}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(papers)))
    
    for paper, color in zip(papers, colors):
        if not paper.counts_by_year:
            continue
        
        years = sorted(paper.counts_by_year.keys())
        counts = [paper.counts_by_year[y] for y in years]
        
        # Create short label
        label = paper.title[:40] + "..." if len(paper.title) > 40 else paper.title
        label = f"{label} ({paper.pub_year})"
        
        ax.plot(years, counts, 'o-', color=color, linewidth=2, markersize=4, label=label)
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Citations", fontsize=11)
    
    title_text = title or f"Citation Trajectories: {pattern_name}"
    ax.set_title(title_text, fontsize=12, fontweight="bold")
    
    # Legend below the graph
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), 
              fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    # Make room for legend below
    fig.subplots_adjust(bottom=0.25)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_metrics_comparison(
    result: CitationPatternResult,
    metric: str = "half_life",
    figsize: Tuple[float, float] = (10, 6),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Plot box plot comparing a metric across patterns.
    
    Parameters
    ----------
    result : CitationPatternResult
        Analysis result.
    metric : str
        Metric to compare: 'half_life', 'years_to_peak', 'early_pct', 'decay_rate'
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    filename : str, optional
        Save to file.
    dpi : int
        Resolution.
        
    Returns
    -------
    tuple
        (fig, ax)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Map metric names
    metric_map = {
        "half_life": ("citation_half_life", "Citation Half-life (years)"),
        "years_to_peak": ("years_to_peak", "Years to Peak"),
        "early_pct": ("early_citations_pct", "Early Citations (%)"),
        "late_pct": ("late_citations_pct", "Late Citations (%)"),
        "decay_rate": ("decay_rate", "Decay Rate"),
    }
    
    if metric not in metric_map:
        raise ValueError(f"Unknown metric: {metric}. Use one of {list(metric_map.keys())}")
    
    attr_name, metric_label = metric_map[metric]
    
    # Collect data by pattern
    pattern_data = {}
    for traj in result.trajectories:
        pattern = traj.pattern.value
        if pattern in ["Uncited", "Too Recent"]:
            continue
        value = getattr(traj, attr_name)
        if pattern not in pattern_data:
            pattern_data[pattern] = []
        pattern_data[pattern].append(value)
    
    if not pattern_data:
        raise ValueError("No data to plot")
    
    # Sort patterns by median value
    patterns = sorted(pattern_data.keys(), 
                     key=lambda p: np.median(pattern_data[p]) if pattern_data[p] else 0)
    data = [pattern_data[p] for p in patterns]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bp = ax.boxplot(data, labels=patterns, patch_artist=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor("lightsteelblue")
        patch.set_edgecolor("steelblue")
    
    for median in bp['medians']:
        median.set_color("darkblue")
        median.set_linewidth(2)
    
    ax.set_xlabel("Citation Pattern", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    
    title_text = title or f"{metric_label} by Citation Pattern"
    ax.set_title(title_text, fontsize=12, fontweight="bold")
    
    ax.tick_params(axis='x', rotation=30)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_pattern_scatter(
    result: CitationPatternResult,
    x_metric: str = "years_to_peak",
    y_metric: str = "early_citations_pct",
    figsize: Tuple[float, float] = (10, 8),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Scatter plot of papers colored by pattern.
    
    Parameters
    ----------
    result : CitationPatternResult
        Analysis result.
    x_metric : str
        Metric for x-axis.
    y_metric : str
        Metric for y-axis.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    filename : str, optional
        Save to file.
    dpi : int
        Resolution.
        
    Returns
    -------
    tuple
        (fig, ax)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    metric_labels = {
        "years_to_peak": "Years to Peak",
        "citation_half_life": "Citation Half-life",
        "early_citations_pct": "Early Citations (%)",
        "late_citations_pct": "Late Citations (%)",
        "decay_rate": "Decay Rate",
        "total_citations": "Total Citations",
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by pattern
    pattern_colors = {
        CitationPattern.EVERGREEN: "forestgreen",
        CitationPattern.FLASH_IN_THE_PAN: "orangered",
        CitationPattern.DELAYED_RECOGNITION: "royalblue",
        CitationPattern.SLEEPING_BEAUTY: "purple",
        CitationPattern.NORMAL: "gray",
        CitationPattern.UNCITED: "lightgray",
        CitationPattern.TOO_RECENT: "lightgray",
    }
    
    for pattern in CitationPattern:
        papers = [t for t in result.trajectories if t.pattern == pattern]
        if not papers or pattern in [CitationPattern.UNCITED, CitationPattern.TOO_RECENT]:
            continue
        
        x_vals = [getattr(p, x_metric) for p in papers]
        y_vals = [getattr(p, y_metric) for p in papers]
        
        ax.scatter(x_vals, y_vals, c=pattern_colors.get(pattern, "gray"),
                  label=pattern.value, alpha=0.6, s=30, edgecolors="none")
    
    ax.set_xlabel(metric_labels.get(x_metric, x_metric), fontsize=11)
    ax.set_ylabel(metric_labels.get(y_metric, y_metric), fontsize=11)
    
    title_text = title or "Citation Pattern Classification"
    ax.set_title(title_text, fontsize=12, fontweight="bold")
    
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums and classes
    "CitationPattern",
    "CitationTrajectory",
    "CitationPatternResult",
    # Main functions
    "analyze_citation_patterns",
    "fetch_citation_history_openalex",
    "compute_trajectory_metrics",
    "classify_pattern",
    "estimate_pattern_from_totals",
    # Visualization
    "plot_pattern_distribution",
    "plot_pattern_by_year",
    "plot_trajectory_examples",
    "plot_metrics_comparison",
    "plot_pattern_scatter",
]
