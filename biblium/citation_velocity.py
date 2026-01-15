# -*- coding: utf-8 -*-
"""
Citation Velocity & Momentum Analysis for Biblium
=================================================

Analyze how fast papers are accumulating citations and whether
that rate is accelerating or decelerating.

Metrics:
- Current Velocity: Citations in recent years (rate)
- Average Velocity: Total citations / paper age
- Peak Velocity: Maximum yearly citations
- Momentum: Acceleration or deceleration of velocity
- Trend: Rising, Stable, or Declining

Data Sources:
- OpenAlex datasets with counts_by_year column (recommended)
- OpenAlex API fetch (for other databases)
- Estimation not supported (requires actual yearly data)

@author: Lan.Umek
@version: 2.9.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class VelocityTrend(Enum):
    """Citation velocity trend classification."""
    ACCELERATING = "Accelerating"
    STABLE = "Stable"
    DECELERATING = "Decelerating"
    RISING = "Rising"
    DECLINING = "Declining"
    NEW = "New"  # Too recent to determine
    DORMANT = "Dormant"  # Very low citations


@dataclass
class CitationVelocityMetrics:
    """Citation velocity metrics for a single paper."""
    doi: str
    title: str
    pub_year: int
    total_citations: int
    paper_age: int
    counts_by_year: Dict[int, int]
    
    # Velocity metrics
    current_velocity: float  # Citations per year in recent period
    average_velocity: float  # Total citations / age
    peak_velocity: int  # Max yearly citations
    peak_year: int  # Year of peak velocity
    recent_citations: int  # Citations in last N years
    
    # Momentum metrics
    momentum: float  # Change in velocity (positive = accelerating)
    momentum_pct: float  # Momentum as percentage change
    trend: VelocityTrend
    
    # Additional metrics
    velocity_ratio: float  # Current velocity / average velocity
    years_since_peak: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "DOI": self.doi,
            "Title": self.title[:60] + "..." if len(self.title) > 60 else self.title,
            "Year": self.pub_year,
            "Age": self.paper_age,
            "Total Citations": self.total_citations,
            "Current Velocity": round(self.current_velocity, 2),
            "Average Velocity": round(self.average_velocity, 2),
            "Peak Velocity": self.peak_velocity,
            "Peak Year": self.peak_year,
            "Recent Citations": self.recent_citations,
            "Momentum": round(self.momentum, 2),
            "Momentum %": round(self.momentum_pct, 1),
            "Trend": self.trend.value,
            "Velocity Ratio": round(self.velocity_ratio, 2),
            "Years Since Peak": self.years_since_peak,
        }


@dataclass
class CitationVelocityResult:
    """Complete citation velocity analysis result."""
    metrics: List[CitationVelocityMetrics]
    trend_counts: Dict[str, int]
    n_papers: int
    n_analyzed: int
    current_year: int
    recent_window: int  # Years used for "recent" calculations
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all metrics to DataFrame."""
        if not self.metrics:
            return pd.DataFrame()
        return pd.DataFrame([m.to_dict() for m in self.metrics])
    
    def get_trend_summary(self) -> pd.DataFrame:
        """Get summary by trend type."""
        data = []
        total = sum(self.trend_counts.values())
        for trend, count in sorted(self.trend_counts.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            data.append({
                "Trend": trend,
                "Count": count,
                "Percentage": round(pct, 1),
            })
        return pd.DataFrame(data)
    
    def get_top_accelerating(self, n: int = 10) -> List[CitationVelocityMetrics]:
        """Get papers with highest positive momentum."""
        accelerating = [m for m in self.metrics if m.momentum > 0]
        return sorted(accelerating, key=lambda x: -x.momentum)[:n]
    
    def get_top_velocity(self, n: int = 10) -> List[CitationVelocityMetrics]:
        """Get papers with highest current velocity."""
        return sorted(self.metrics, key=lambda x: -x.current_velocity)[:n]
    
    def get_rising_stars(self, n: int = 10, max_age: int = 5) -> List[CitationVelocityMetrics]:
        """Get recent papers with high acceleration."""
        recent = [m for m in self.metrics if m.paper_age <= max_age and m.momentum > 0]
        return sorted(recent, key=lambda x: -x.momentum)[:n]
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "CITATION VELOCITY & MOMENTUM ANALYSIS",
            "=" * 50,
            f"Documents analyzed: {self.n_analyzed} / {self.n_papers}",
            f"Reference year: {self.current_year}",
            f"Recent window: {self.recent_window} years",
            "",
            "TREND DISTRIBUTION",
            "-" * 50,
        ]
        
        total = sum(self.trend_counts.values())
        for trend, count in sorted(self.trend_counts.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 5)
            lines.append(f"  {trend:15s}: {count:5d} ({pct:5.1f}%) {bar}")
        
        # Add velocity statistics
        if self.metrics:
            velocities = [m.current_velocity for m in self.metrics]
            lines.extend([
                "",
                "VELOCITY STATISTICS",
                "-" * 50,
                f"  Mean current velocity: {np.mean(velocities):.2f} citations/year",
                f"  Median current velocity: {np.median(velocities):.2f} citations/year",
                f"  Max current velocity: {max(velocities):.2f} citations/year",
            ])
            
            # Top accelerating
            top_acc = self.get_top_accelerating(3)
            if top_acc:
                lines.extend([
                    "",
                    "TOP ACCELERATING",
                    "-" * 50,
                ])
                for m in top_acc:
                    lines.append(f"  • {m.title[:50]}... (+{m.momentum:.1f}/year)")
        
        return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_counts_by_year(value) -> Dict[int, int]:
    """
    Parse counts_by_year from various formats.
    """
    import json
    
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except:
            return {}
    
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


def compute_velocity_metrics(
    counts_by_year: Dict[int, int],
    pub_year: int,
    current_year: int,
    recent_window: int = 3,
) -> dict:
    """
    Compute citation velocity and momentum metrics.
    
    Parameters
    ----------
    counts_by_year : dict
        {year: citation_count}
    pub_year : int
        Publication year.
    current_year : int
        Current/reference year.
    recent_window : int
        Years to consider as "recent" for velocity calculation.
        
    Returns
    -------
    dict
        Velocity and momentum metrics.
    """
    paper_age = current_year - pub_year
    total_citations = sum(counts_by_year.values()) if counts_by_year else 0
    
    if not counts_by_year or paper_age <= 0:
        return {
            "current_velocity": 0.0,
            "average_velocity": 0.0,
            "peak_velocity": 0,
            "peak_year": pub_year,
            "recent_citations": 0,
            "momentum": 0.0,
            "momentum_pct": 0.0,
            "trend": VelocityTrend.NEW,
            "velocity_ratio": 0.0,
            "years_since_peak": 0,
        }
    
    # Average velocity
    average_velocity = total_citations / paper_age if paper_age > 0 else 0
    
    # Peak velocity and year
    peak_year = max(counts_by_year.keys(), key=lambda y: counts_by_year[y])
    peak_velocity = counts_by_year[peak_year]
    years_since_peak = current_year - peak_year
    
    # Recent citations and current velocity
    recent_years = range(current_year - recent_window + 1, current_year + 1)
    recent_citations = sum(counts_by_year.get(y, 0) for y in recent_years)
    current_velocity = recent_citations / recent_window
    
    # Earlier period velocity (for momentum calculation)
    earlier_years = range(current_year - 2 * recent_window + 1, current_year - recent_window + 1)
    earlier_citations = sum(counts_by_year.get(y, 0) for y in earlier_years)
    earlier_velocity = earlier_citations / recent_window if paper_age > recent_window else 0
    
    # Momentum (change in velocity)
    momentum = current_velocity - earlier_velocity
    momentum_pct = (momentum / earlier_velocity * 100) if earlier_velocity > 0 else 0
    
    # Velocity ratio
    velocity_ratio = current_velocity / average_velocity if average_velocity > 0 else 0
    
    # Determine trend
    if paper_age < recent_window:
        trend = VelocityTrend.NEW
    elif total_citations < 5:
        trend = VelocityTrend.DORMANT
    elif momentum > 1:
        trend = VelocityTrend.ACCELERATING
    elif momentum < -1:
        trend = VelocityTrend.DECELERATING
    elif velocity_ratio > 1.2:
        trend = VelocityTrend.RISING
    elif velocity_ratio < 0.5:
        trend = VelocityTrend.DECLINING
    else:
        trend = VelocityTrend.STABLE
    
    return {
        "current_velocity": current_velocity,
        "average_velocity": average_velocity,
        "peak_velocity": peak_velocity,
        "peak_year": peak_year,
        "recent_citations": recent_citations,
        "momentum": momentum,
        "momentum_pct": momentum_pct,
        "trend": trend,
        "velocity_ratio": velocity_ratio,
        "years_since_peak": years_since_peak,
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_citation_velocity(
    df: pd.DataFrame,
    use_openalex: bool = False,
    max_papers: int = 500,
    current_year: int = None,
    recent_window: int = 3,
    min_age: int = 2,
    verbose: bool = True,
    stop_flag: callable = None,
) -> CitationVelocityResult:
    """
    Analyze citation velocity and momentum for papers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliometric dataset.
    use_openalex : bool
        Fetch citation history from OpenAlex API if not in dataset.
    max_papers : int
        Maximum papers to fetch from API.
    current_year : int, optional
        Reference year. Defaults to current year.
    recent_window : int
        Years to consider as "recent" for velocity calculation.
    min_age : int
        Minimum paper age to analyze.
    verbose : bool
        Print progress.
    stop_flag : callable, optional
        Function that returns True if analysis should stop.
        
    Returns
    -------
    CitationVelocityResult
        Complete analysis result.
        
    Notes
    -----
    This analysis requires yearly citation counts. If the dataset doesn't
    contain counts_by_year data, use_openalex=True to fetch from API.
    Unlike Citation Patterns, estimation is not supported for velocity analysis.
    """
    import datetime
    
    # Helper to check stop
    def should_stop():
        return stop_flag is not None and stop_flag()
    
    if current_year is None:
        current_year = datetime.datetime.now().year
    
    # Find columns
    year_col = None
    for c in ["Year", "PY", "Publication Year", "publication_year"]:
        if c in df.columns:
            year_col = c
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
    
    cite_col = None
    for c in ["Cited by", "Times Cited", "Citations", "TC", "cited_by_count"]:
        if c in df.columns:
            cite_col = c
            break
    
    # Check for embedded counts_by_year (single column with JSON/dict)
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
    
    has_embedded_counts = False
    if counts_by_year_col:
        sample = df[counts_by_year_col].dropna().head(10)
        if len(sample) > 0:
            parsed = [_parse_counts_by_year(v) for v in sample]
            if any(p for p in parsed):
                has_embedded_counts = True
    
    if verbose:
        print(f"Analyzing citation velocity...")
        print(f"  Year column: {year_col}")
        print(f"  Recent window: {recent_window} years")
        if has_embedded_counts:
            print(f"  ✓ Using embedded counts_by_year column")
        elif has_split_counts:
            print(f"  ✓ Using OpenAlex split format (counts_by_year.year + counts_by_year.cited_by_count)")
    
    # Check stop before API fetch
    if should_stop():
        if verbose:
            print("  ⏹ Analysis stopped by user")
        return CitationVelocityResult(
            metrics=[], trend_counts={t.value: 0 for t in VelocityTrend},
            n_papers=len(df), n_analyzed=0, current_year=current_year,
            recent_window=recent_window
        )
    
    # Fetch from OpenAlex if needed
    citation_history = {}
    
    if not has_embedded_counts and not has_split_counts and use_openalex and (doi_col or title_col):
        try:
            from biblium.citation_patterns import fetch_citation_history_openalex
            citation_history = fetch_citation_history_openalex(
                df, doi_col=doi_col, title_col=title_col,
                max_papers=max_papers, verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"  ⚠️ OpenAlex fetch failed: {e}")
    
    # Process papers
    metrics_list = []
    trend_counts = {t.value: 0 for t in VelocityTrend}
    skipped_no_data = 0
    
    for idx, row in df.iterrows():
        # Check stop flag periodically
        if idx % 100 == 0 and should_stop():
            if verbose:
                print(f"  ⏹ Analysis stopped by user at paper {idx}")
            break
        
        try:
            pub_year = int(row[year_col])
        except:
            continue
        
        paper_age = current_year - pub_year
        if paper_age < min_age:
            continue
        
        # Get DOI and title
        doi = ""
        if doi_col and pd.notna(row.get(doi_col)):
            doi = str(row[doi_col]).strip()
            doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        
        title = ""
        if title_col and pd.notna(row.get(title_col)):
            title = str(row[title_col]).strip()
        
        # Get counts_by_year
        counts_by_year = None
        
        # Try embedded column first
        if has_embedded_counts and counts_by_year_col:
            embedded = row.get(counts_by_year_col)
            if pd.notna(embedded):
                counts_by_year = _parse_counts_by_year(embedded)
        
        # Try OpenAlex split format (counts_by_year.year and counts_by_year.cited_by_count)
        if not counts_by_year and has_split_counts:
            years_str = row.get(counts_year_col)
            counts_str = row.get(counts_cited_col)
            if pd.notna(years_str) and pd.notna(counts_str):
                counts_by_year = _parse_split_counts_by_year(years_str, counts_str)
        
        # Try API-fetched data
        if not counts_by_year:
            key = doi if doi else title[:100]
            if key in citation_history:
                counts_by_year = citation_history[key]
        
        # Skip if no yearly data available
        if not counts_by_year:
            skipped_no_data += 1
            continue
        
        # Get total citations
        total_cites = sum(counts_by_year.values())
        if cite_col and pd.notna(row.get(cite_col)):
            try:
                total_cites = max(total_cites, int(row[cite_col]))
            except:
                pass
        
        # Compute metrics
        vel_metrics = compute_velocity_metrics(
            counts_by_year, pub_year, current_year, recent_window
        )
        
        metrics = CitationVelocityMetrics(
            doi=doi,
            title=title,
            pub_year=pub_year,
            total_citations=total_cites,
            paper_age=paper_age,
            counts_by_year=counts_by_year,
            current_velocity=vel_metrics["current_velocity"],
            average_velocity=vel_metrics["average_velocity"],
            peak_velocity=vel_metrics["peak_velocity"],
            peak_year=vel_metrics["peak_year"],
            recent_citations=vel_metrics["recent_citations"],
            momentum=vel_metrics["momentum"],
            momentum_pct=vel_metrics["momentum_pct"],
            trend=vel_metrics["trend"],
            velocity_ratio=vel_metrics["velocity_ratio"],
            years_since_peak=vel_metrics["years_since_peak"],
        )
        
        metrics_list.append(metrics)
        trend_counts[metrics.trend.value] += 1
    
    if verbose:
        print(f"  ✓ Analyzed {len(metrics_list)} documents")
        if skipped_no_data > 0:
            print(f"  ⚠️ Skipped {skipped_no_data} documents (no yearly citation data)")
    
    return CitationVelocityResult(
        metrics=metrics_list,
        trend_counts=trend_counts,
        n_papers=len(df),
        n_analyzed=len(metrics_list),
        current_year=current_year,
        recent_window=recent_window,
    )


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_trend_distribution(
    result: CitationVelocityResult,
    figsize: Tuple[float, float] = (10, 6),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Plot distribution of velocity trends as horizontal bar chart.
    
    Parameters
    ----------
    result : CitationVelocityResult
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
    trends = []
    counts = []
    for trend, count in sorted(result.trend_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            trends.append(trend)
            counts.append(count)
    
    if not trends:
        raise ValueError("No trends to plot")
    
    # Reverse for horizontal bar chart (highest at top)
    trends = trends[::-1]
    counts = counts[::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(trends))
    ax.barh(y_pos, counts, color="steelblue", edgecolor="none")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(trends)
    ax.set_xlabel("Number of Documents", fontsize=11)
    ax.set_ylabel("Velocity Trend", fontsize=11)
    
    # Add count labels
    total = sum(counts)
    for i, count in enumerate(counts):
        pct = count / total * 100
        ax.text(count + max(counts) * 0.01, i, f" {count} ({pct:.1f}%)", 
               va="center", fontsize=9)
    
    ax.set_xlim(0, max(counts) * 1.2)
    ax.set_title(title or "Citation Velocity Trend Distribution", fontsize=12, fontweight="bold")
    
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_velocity_distribution(
    result: CitationVelocityResult,
    figsize: Tuple[float, float] = (10, 6),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Plot histogram of current velocities.
    
    Parameters
    ----------
    result : CitationVelocityResult
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
    
    velocities = [m.current_velocity for m in result.metrics if m.current_velocity > 0]
    
    if not velocities:
        raise ValueError("No velocity data to plot")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use log scale bins for better visualization
    ax.hist(velocities, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    
    ax.set_xlabel("Current Velocity (citations/year)", fontsize=11)
    ax.set_ylabel("Number of Documents", fontsize=11)
    ax.set_title(title or "Distribution of Citation Velocities", fontsize=12, fontweight="bold")
    
    # Add statistics
    mean_vel = np.mean(velocities)
    median_vel = np.median(velocities)
    ax.axvline(mean_vel, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_vel:.1f}")
    ax.axvline(median_vel, color="orange", linestyle="--", linewidth=1.5, label=f"Median: {median_vel:.1f}")
    
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_velocity_vs_age(
    result: CitationVelocityResult,
    figsize: Tuple[float, float] = (10, 6),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Scatter plot of velocity vs paper age.
    
    Parameters
    ----------
    result : CitationVelocityResult
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
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by trend
    trend_colors = {
        VelocityTrend.ACCELERATING: "green",
        VelocityTrend.RISING: "lightgreen",
        VelocityTrend.STABLE: "steelblue",
        VelocityTrend.DECLINING: "orange",
        VelocityTrend.DECELERATING: "red",
        VelocityTrend.NEW: "gray",
        VelocityTrend.DORMANT: "lightgray",
    }
    
    for trend in VelocityTrend:
        papers = [m for m in result.metrics if m.trend == trend]
        if not papers:
            continue
        
        ages = [m.paper_age for m in papers]
        velocities = [m.current_velocity for m in papers]
        
        ax.scatter(ages, velocities, c=trend_colors.get(trend, "gray"),
                  label=trend.value, alpha=0.6, s=30, edgecolors="none")
    
    ax.set_xlabel("Paper Age (years)", fontsize=11)
    ax.set_ylabel("Current Velocity (citations/year)", fontsize=11)
    ax.set_title(title or "Citation Velocity vs Paper Age", fontsize=12, fontweight="bold")
    
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_momentum_distribution(
    result: CitationVelocityResult,
    figsize: Tuple[float, float] = (10, 6),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Plot histogram of momentum values.
    
    Parameters
    ----------
    result : CitationVelocityResult
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
    
    momenta = [m.momentum for m in result.metrics]
    
    if not momenta:
        raise ValueError("No momentum data to plot")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color positive and negative differently
    pos_momenta = [m for m in momenta if m > 0]
    neg_momenta = [m for m in momenta if m <= 0]
    
    bins = np.linspace(min(momenta), max(momenta), 31)
    
    ax.hist(pos_momenta, bins=bins, color="green", edgecolor="white", alpha=0.7, label="Accelerating")
    ax.hist(neg_momenta, bins=bins, color="red", edgecolor="white", alpha=0.7, label="Decelerating")
    
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    
    ax.set_xlabel("Momentum (change in citations/year)", fontsize=11)
    ax.set_ylabel("Number of Documents", fontsize=11)
    ax.set_title(title or "Distribution of Citation Momentum", fontsize=12, fontweight="bold")
    
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_top_accelerating(
    result: CitationVelocityResult,
    n: int = 15,
    figsize: Tuple[float, float] = (10, 8),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Plot top accelerating papers.
    
    Parameters
    ----------
    result : CitationVelocityResult
        Analysis result.
    n : int
        Number of papers to show.
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
    
    top = result.get_top_accelerating(n)
    
    if not top:
        raise ValueError("No accelerating papers found")
    
    # Reverse for horizontal bar chart
    top = top[::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = [f"{m.title[:35]}... ({m.pub_year})" if len(m.title) > 35 else f"{m.title} ({m.pub_year})" for m in top]
    values = [m.momentum for m in top]
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="green", edgecolor="none")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Momentum (citations/year change)", fontsize=11)
    ax.set_title(title or f"Top {len(top)} Accelerating Documents", fontsize=12, fontweight="bold")
    
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_top_velocity(
    result: CitationVelocityResult,
    n: int = 15,
    figsize: Tuple[float, float] = (10, 8),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Plot papers with highest current velocity.
    
    Parameters
    ----------
    result : CitationVelocityResult
        Analysis result.
    n : int
        Number of papers to show.
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
    
    top = result.get_top_velocity(n)
    
    if not top:
        raise ValueError("No velocity data found")
    
    # Reverse for horizontal bar chart
    top = top[::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = [f"{m.title[:35]}... ({m.pub_year})" if len(m.title) > 35 else f"{m.title} ({m.pub_year})" for m in top]
    values = [m.current_velocity for m in top]
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="steelblue", edgecolor="none")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Current Velocity (citations/year)", fontsize=11)
    ax.set_title(title or f"Top {len(top)} Highest Velocity Documents", fontsize=12, fontweight="bold")
    
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_velocity_trajectories(
    result: CitationVelocityResult,
    n_examples: int = 5,
    selection: str = "top_velocity",
    figsize: Tuple[float, float] = (10, 6),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Plot citation trajectories for selected papers.
    
    Parameters
    ----------
    result : CitationVelocityResult
        Analysis result.
    n_examples : int
        Number of papers to plot.
    selection : str
        How to select papers: 'top_velocity', 'top_accelerating', 'rising_stars'
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
    if selection == "top_accelerating":
        papers = result.get_top_accelerating(n_examples)
        selection_label = "Top Accelerating"
    elif selection == "rising_stars":
        papers = result.get_rising_stars(n_examples)
        selection_label = "Rising Stars"
    else:
        papers = result.get_top_velocity(n_examples)
        selection_label = "Highest Velocity"
    
    if not papers:
        raise ValueError(f"No papers found for selection: {selection}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(papers)))
    
    for paper, color in zip(papers, colors):
        if not paper.counts_by_year:
            continue
        
        years = sorted(paper.counts_by_year.keys())
        counts = [paper.counts_by_year[y] for y in years]
        
        label = paper.title[:35] + "..." if len(paper.title) > 35 else paper.title
        label = f"{label} ({paper.pub_year})"
        
        ax.plot(years, counts, 'o-', color=color, linewidth=2, markersize=4, label=label)
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Citations", fontsize=11)
    ax.set_title(title or f"Citation Trajectories: {selection_label}", fontsize=12, fontweight="bold")
    
    # Legend below
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), 
              fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(bottom=0.25)
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums and classes
    "VelocityTrend",
    "CitationVelocityMetrics",
    "CitationVelocityResult",
    # Main functions
    "analyze_citation_velocity",
    "compute_velocity_metrics",
    # Visualization
    "plot_trend_distribution",
    "plot_velocity_distribution",
    "plot_velocity_vs_age",
    "plot_momentum_distribution",
    "plot_top_accelerating",
    "plot_top_velocity",
    "plot_velocity_trajectories",
]
