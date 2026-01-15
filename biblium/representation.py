# -*- coding: utf-8 -*-
"""
Relative Representation Analysis for Biblium

Compare analyzed dataset's distribution against a global/reference distribution
to identify over- and under-representation.

Features:
- Fetch reference data from OpenAlex API (Year, Country, SDG, Document Type, etc.)
- Compute percentage point differences
- Chi-square test for statistical significance
- Visualize with diverging colormap (blue=over, red=under)

@author: Lan.Umek
@version: 2.9.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass
import warnings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ChiSquareResult:
    """Result of chi-square test for distribution comparison."""
    chi2: float
    p_value: float
    degrees_of_freedom: int
    significant: bool  # p < 0.05
    interpretation: str
    
    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return f"χ² = {self.chi2:.2f}, df = {self.degrees_of_freedom}, p = {self.p_value:.4f} ({sig})"


@dataclass  
class ReferenceBenchmarkResult:
    """Complete result of reference benchmark analysis."""
    comparison_df: pd.DataFrame
    category: str
    reference_source: str  # "OpenAlex" or "Custom"
    chi_square: Optional[ChiSquareResult]
    n_over_represented: int
    n_under_represented: int
    n_as_expected: int
    threshold: float
    warning_message: Optional[str]
    
    def summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            f"Reference Benchmark: {self.category}",
            f"Reference: {self.reference_source}",
            f"Categories: {len(self.comparison_df)}",
            f"Threshold: ±{self.threshold} pp",
            "",
            f"Over-represented: {self.n_over_represented}",
            f"Under-represented: {self.n_under_represented}",
            f"As expected: {self.n_as_expected}",
        ]
        
        if self.chi_square:
            lines.extend(["", f"Chi-square test: {self.chi_square}"])
        
        if self.warning_message:
            lines.extend(["", f"⚠️ {self.warning_message}"])
        
        return "\n".join(lines)


# =============================================================================
# WARNING MESSAGES
# =============================================================================

OPENALEX_WARNING = (
    "Reference data from OpenAlex represents global research patterns across all "
    "disciplines and sources. This may differ from coverage in Scopus or Web of Science. "
    "For database-specific comparisons, provide custom reference data exported from "
    "your target database."
)

DATABASE_COVERAGE_NOTES = {
    "scopus": "Scopus has different coverage than OpenAlex, especially for older publications and non-English journals.",
    "wos": "Web of Science focuses on high-impact journals and may have different SDG classifications.",
    "pubmed": "PubMed is biomedical-focused; global comparisons may not be meaningful for non-medical research.",
    "oa": "OpenAlex data compared to itself - no coverage bias expected.",
}


def get_reference_warning(database: str = None) -> str:
    """
    Get appropriate warning message for reference data.
    
    Parameters
    ----------
    database : str, optional
        Source database of analyzed data (scopus, wos, pubmed, oa).
        
    Returns
    -------
    str
        Warning message about potential coverage differences.
    """
    base_warning = OPENALEX_WARNING
    
    if database and database.lower() in DATABASE_COVERAGE_NOTES:
        db_note = DATABASE_COVERAGE_NOTES[database.lower()]
        return f"{base_warning}\n\nNote for {database}: {db_note}"
    
    return base_warning


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def chi_square_test(
    observed_counts: np.ndarray,
    reference_counts: np.ndarray,
    min_expected: float = 5.0,
) -> ChiSquareResult:
    """
    Perform chi-square test comparing observed to reference distribution.
    
    Parameters
    ----------
    observed_counts : array-like
        Observed counts for each category.
    reference_counts : array-like
        Reference counts (will be scaled to match observed total).
    min_expected : float
        Minimum expected count per cell. Categories below this are merged
        or test may be unreliable.
        
    Returns
    -------
    ChiSquareResult
        Test results with interpretation.
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return ChiSquareResult(
            chi2=np.nan,
            p_value=np.nan,
            degrees_of_freedom=0,
            significant=False,
            interpretation="scipy not available for chi-square test"
        )
    
    observed = np.array(observed_counts, dtype=float)
    reference = np.array(reference_counts, dtype=float)
    
    # Scale reference to match observed total
    observed_total = observed.sum()
    reference_total = reference.sum()
    
    if reference_total == 0 or observed_total == 0:
        return ChiSquareResult(
            chi2=np.nan,
            p_value=np.nan,
            degrees_of_freedom=0,
            significant=False,
            interpretation="Cannot compute: zero total counts"
        )
    
    expected = reference * (observed_total / reference_total)
    
    # Check minimum expected counts
    low_expected = (expected < min_expected).sum()
    if low_expected > len(expected) * 0.2:
        interpretation = (
            f"Warning: {low_expected} categories have expected count < {min_expected}. "
            "Results may be unreliable. Consider merging small categories."
        )
    else:
        interpretation = ""
    
    # Perform test
    chi2, p_value = scipy_stats.chisquare(observed, expected)
    df = len(observed) - 1
    significant = p_value < 0.05
    
    if significant:
        interpretation += "Distribution differs significantly from reference (p < 0.05)."
    else:
        interpretation += "Distribution does not differ significantly from reference."
    
    return ChiSquareResult(
        chi2=float(chi2),
        p_value=float(p_value),
        degrees_of_freedom=df,
        significant=significant,
        interpretation=interpretation.strip()
    )


# =============================================================================
# OPENALEX REFERENCE DATA FETCHERS
# =============================================================================

# SDG mapping (OpenAlex uses these IDs)
SDG_NAMES = {
    "https://metadata.un.org/sdg/1": "SDG 1: No Poverty",
    "https://metadata.un.org/sdg/2": "SDG 2: Zero Hunger",
    "https://metadata.un.org/sdg/3": "SDG 3: Good Health",
    "https://metadata.un.org/sdg/4": "SDG 4: Quality Education",
    "https://metadata.un.org/sdg/5": "SDG 5: Gender Equality",
    "https://metadata.un.org/sdg/6": "SDG 6: Clean Water",
    "https://metadata.un.org/sdg/7": "SDG 7: Clean Energy",
    "https://metadata.un.org/sdg/8": "SDG 8: Economic Growth",
    "https://metadata.un.org/sdg/9": "SDG 9: Industry & Innovation",
    "https://metadata.un.org/sdg/10": "SDG 10: Reduced Inequalities",
    "https://metadata.un.org/sdg/11": "SDG 11: Sustainable Cities",
    "https://metadata.un.org/sdg/12": "SDG 12: Responsible Consumption",
    "https://metadata.un.org/sdg/13": "SDG 13: Climate Action",
    "https://metadata.un.org/sdg/14": "SDG 14: Life Below Water",
    "https://metadata.un.org/sdg/15": "SDG 15: Life on Land",
    "https://metadata.un.org/sdg/16": "SDG 16: Peace & Justice",
    "https://metadata.un.org/sdg/17": "SDG 17: Partnerships",
}

# Short SDG names for plotting
SDG_SHORT = {
    "https://metadata.un.org/sdg/1": "SDG 1",
    "https://metadata.un.org/sdg/2": "SDG 2",
    "https://metadata.un.org/sdg/3": "SDG 3",
    "https://metadata.un.org/sdg/4": "SDG 4",
    "https://metadata.un.org/sdg/5": "SDG 5",
    "https://metadata.un.org/sdg/6": "SDG 6",
    "https://metadata.un.org/sdg/7": "SDG 7",
    "https://metadata.un.org/sdg/8": "SDG 8",
    "https://metadata.un.org/sdg/9": "SDG 9",
    "https://metadata.un.org/sdg/10": "SDG 10",
    "https://metadata.un.org/sdg/11": "SDG 11",
    "https://metadata.un.org/sdg/12": "SDG 12",
    "https://metadata.un.org/sdg/13": "SDG 13",
    "https://metadata.un.org/sdg/14": "SDG 14",
    "https://metadata.un.org/sdg/15": "SDG 15",
    "https://metadata.un.org/sdg/16": "SDG 16",
    "https://metadata.un.org/sdg/17": "SDG 17",
}


def fetch_openalex_reference(
    group_by: str,
    year_range: Tuple[int, int] = None,
    additional_filter: str = None,
    top_n: int = 200,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch reference distribution from OpenAlex API.
    
    Parameters
    ----------
    group_by : str
        What to group by. Options:
        - "publication_year" : By year
        - "authorships.countries" : By country
        - "sustainable_development_goals.id" : By SDG
        - "type" : By document type
        - "open_access.is_oa" : By open access status
        - "primary_location.source.id" : By source/journal
        - "authorships.institutions.country_code" : By institution country
    year_range : tuple of (start_year, end_year), optional
        Filter by year range.
    additional_filter : str, optional
        Additional OpenAlex filter string.
    top_n : int
        Maximum number of groups to return.
    timeout : int
        Request timeout in seconds.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns [category_name, 'Count', 'Percentage'].
    """
    base_url = "https://api.openalex.org/works"
    
    params = {
        "group_by": group_by,
        "per_page": top_n,
    }
    
    # Build filter
    filters = []
    if year_range:
        filters.append(f"publication_year:{year_range[0]}-{year_range[1]}")
    if additional_filter:
        filters.append(additional_filter)
    
    if filters:
        params["filter"] = ",".join(filters)
    
    try:
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("group_by", []):
            key = item.get("key", "Unknown")
            display_name = item.get("key_display_name", key)
            count = item.get("count", 0)
            
            # Use short SDG names if applicable
            if group_by == "sustainable_development_goals.id" and key in SDG_SHORT:
                display_name = SDG_SHORT[key]
            
            results.append({
                "Category": display_name,
                "Count": count,
                "_key": key,  # Keep original key for reference
            })
        
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            total = df["Count"].sum()
            df["Percentage"] = (df["Count"] / total * 100) if total > 0 else 0
        
        return df
        
    except Exception as e:
        warnings.warn(f"Failed to fetch OpenAlex data: {e}")
        return pd.DataFrame(columns=["Category", "Count", "Percentage"])


def fetch_openalex_yearly_counts(
    start_year: int = 2000,
    end_year: int = 2024,
    additional_filter: str = None,
) -> pd.DataFrame:
    """
    Fetch global publication counts by year from OpenAlex.
    
    Parameters
    ----------
    start_year : int
        Start year.
    end_year : int
        End year.
    additional_filter : str, optional
        Additional filter (e.g., concept, country).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Year', 'Count', 'Percentage'].
    """
    df = fetch_openalex_reference(
        group_by="publication_year",
        year_range=(start_year, end_year),
        additional_filter=additional_filter,
        top_n=end_year - start_year + 5,
    )
    
    if len(df) > 0:
        df = df.rename(columns={"Category": "Year"})
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]
        df = df.sort_values("Year").reset_index(drop=True)
    
    return df


def fetch_openalex_country_counts(
    year_range: Tuple[int, int] = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Fetch global publication counts by country from OpenAlex.
    """
    df = fetch_openalex_reference(
        group_by="authorships.countries",
        year_range=year_range,
        top_n=top_n,
    )
    
    if len(df) > 0:
        df = df.rename(columns={"Category": "Country"})
    
    return df


# Fallback global SDG distribution (approximate from OpenAlex 2020-2024 data)
# Used when API is unavailable
FALLBACK_SDG_DISTRIBUTION = {
    "SDG 1": 1850000,    # No Poverty
    "SDG 2": 2950000,    # Zero Hunger
    "SDG 3": 18500000,   # Good Health (largest)
    "SDG 4": 3200000,    # Quality Education
    "SDG 5": 1450000,    # Gender Equality
    "SDG 6": 1650000,    # Clean Water
    "SDG 7": 2850000,    # Clean Energy
    "SDG 8": 2100000,    # Economic Growth
    "SDG 9": 3500000,    # Industry & Innovation
    "SDG 10": 1250000,   # Reduced Inequalities
    "SDG 11": 2450000,   # Sustainable Cities
    "SDG 12": 1950000,   # Responsible Consumption
    "SDG 13": 3650000,   # Climate Action
    "SDG 14": 1150000,   # Life Below Water
    "SDG 15": 1850000,   # Life on Land
    "SDG 16": 1550000,   # Peace & Justice
    "SDG 17": 950000,    # Partnerships
}


def fetch_openalex_sdg_counts(
    year_range: Tuple[int, int] = None,
    use_fallback_on_error: bool = True,
) -> pd.DataFrame:
    """
    Fetch global publication counts by SDG from OpenAlex.
    
    Parameters
    ----------
    year_range : tuple, optional
        Year range filter.
    use_fallback_on_error : bool, default True
        If True and API fails, use hardcoded fallback distribution.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['SDG', 'Count', 'Percentage'].
    """
    df = fetch_openalex_reference(
        group_by="sustainable_development_goals.id",
        year_range=year_range,
        top_n=20,
    )
    
    if len(df) > 0:
        df = df.rename(columns={"Category": "SDG"})
        
        # Standardize SDG labels to "SDG X" format
        if "_key" in df.columns:
            def extract_sdg_label(key):
                if pd.isna(key):
                    return None
                import re
                match = re.search(r'/sdg/(\d+)', str(key))
                if match:
                    return f"SDG {int(match.group(1))}"
                return None
            
            df["SDG"] = df["_key"].apply(extract_sdg_label)
            df = df[df["SDG"].notna()]
        
        # Sort by SDG number
        if len(df) > 0:
            df["_sort"] = df["SDG"].str.extract(r"(\d+)").astype(float)
            df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
            return df
    
    # API failed or returned empty - use fallback
    if use_fallback_on_error:
        print("OpenAlex API returned no SDG data. Using fallback distribution.")
        fallback_data = [{"SDG": k, "Count": v} for k, v in FALLBACK_SDG_DISTRIBUTION.items()]
        df = pd.DataFrame(fallback_data)
        total = df["Count"].sum()
        df["Percentage"] = df["Count"] / total * 100
        # Sort by SDG number
        df["_sort"] = df["SDG"].str.extract(r"(\d+)").astype(float)
        df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
        return df
    
    return df


def fetch_openalex_doctype_counts(
    year_range: Tuple[int, int] = None,
) -> pd.DataFrame:
    """
    Fetch global publication counts by document type from OpenAlex.
    """
    df = fetch_openalex_reference(
        group_by="type",
        year_range=year_range,
        top_n=20,
    )
    
    if len(df) > 0:
        df = df.rename(columns={"Category": "Document Type"})
    
    return df


def fetch_openalex_oa_counts(
    year_range: Tuple[int, int] = None,
) -> pd.DataFrame:
    """
    Fetch global publication counts by open access status from OpenAlex.
    """
    df = fetch_openalex_reference(
        group_by="open_access.is_oa",
        year_range=year_range,
        top_n=10,
    )
    
    if len(df) > 0:
        df = df.rename(columns={"Category": "Open Access"})
        df["Open Access"] = df["Open Access"].map({True: "Open Access", False: "Closed", "true": "Open Access", "false": "Closed"})
    
    return df


# =============================================================================
# RELATIVE REPRESENTATION COMPUTATION
# =============================================================================

def compute_relative_representation(
    observed_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    category_col: str,
    count_col: str = "Count",
    ref_count_col: str = None,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Compute relative representation (percentage point difference) between
    observed and reference distributions.
    
    Parameters
    ----------
    observed_df : pd.DataFrame
        Analyzed dataset's distribution.
    reference_df : pd.DataFrame
        Reference/baseline distribution.
    category_col : str
        Name of the category column.
    count_col : str
        Count column in observed_df.
    ref_count_col : str, optional
        Count column in reference_df.
    threshold : float
        Threshold for classification (pp).
    
    Returns
    -------
    pd.DataFrame
        Analysis with difference in percentage points.
    """
    import re
    
    if ref_count_col is None:
        ref_count_col = count_col
    
    observed = observed_df.copy()
    reference = reference_df.copy()
    
    # Calculate percentages
    obs_total = observed[count_col].sum()
    ref_total = reference[ref_count_col].sum()
    
    observed["Observed %"] = (observed[count_col] / obs_total * 100) if obs_total > 0 else 0
    reference["Reference %"] = (reference[ref_count_col] / ref_total * 100) if ref_total > 0 else 0
    
    # Rename count columns before merge to avoid conflicts
    observed = observed.rename(columns={count_col: "Observed Count"})
    reference = reference.rename(columns={ref_count_col: "Reference Count"})
    
    # Debug: show what we're trying to merge
    print(f"\nMerging on column: {category_col}")
    print(f"Observed categories ({len(observed)}): {sorted(observed[category_col].tolist())}")
    print(f"Reference categories ({len(reference)}): {sorted(reference[category_col].tolist())}")
    
    # Merge - use inner join to only keep categories in BOTH datasets
    merged = pd.merge(
        observed[[category_col, "Observed Count", "Observed %"]],
        reference[[category_col, "Reference Count", "Reference %"]],
        on=category_col,
        how="inner",
    )
    
    print(f"Matched categories: {len(merged)}")
    
    # If no matches, try to fix common formatting issues
    if len(merged) == 0:
        print("\nNo exact matches found. Attempting to standardize labels...")
        
        # Try standardizing both to "SDG X" format
        import re
        def standardize_sdg(val):
            val_str = str(val).strip()
            match = re.search(r"(\d+)", val_str)
            if match:
                return f"SDG {int(match.group(1))}"
            return val_str
        
        observed[category_col] = observed[category_col].apply(standardize_sdg)
        reference[category_col] = reference[category_col].apply(standardize_sdg)
        
        print(f"Standardized observed: {sorted(observed[category_col].tolist())}")
        print(f"Standardized reference: {sorted(reference[category_col].tolist())}")
        
        # Try merge again
        merged = pd.merge(
            observed[[category_col, "Observed Count", "Observed %"]],
            reference[[category_col, "Reference Count", "Reference %"]],
            on=category_col,
            how="inner",
        )
        print(f"Matched after standardization: {len(merged)}")
    
    if len(merged) == 0:
        raise ValueError(
            f"No matching categories found between observed and reference data. "
            f"Check that category labels match exactly."
        )
    
    # Recalculate percentages based on matched categories only
    matched_obs_total = merged["Observed Count"].sum()
    matched_ref_total = merged["Reference Count"].sum()
    
    if matched_obs_total > 0:
        merged["Observed %"] = merged["Observed Count"] / matched_obs_total * 100
    if matched_ref_total > 0:
        merged["Reference %"] = merged["Reference Count"] / matched_ref_total * 100
    
    # Compute differences
    merged["Difference (pp)"] = merged["Observed %"] - merged["Reference %"]
    
    # Relative difference
    merged["Relative Diff (%)"] = np.where(
        merged["Reference %"] > 0,
        (merged["Observed %"] - merged["Reference %"]) / merged["Reference %"] * 100,
        np.where(merged["Observed %"] > 0, np.inf, 0)
    )
    
    # Classification
    def classify(pp_diff):
        if pp_diff > threshold:
            return "Over-represented"
        elif pp_diff < -threshold:
            return "Under-represented"
        else:
            return "As expected"
    
    merged["Representation"] = merged["Difference (pp)"].apply(classify)
    
    # Sort numerically for SDG and Year, alphabetically for others
    def extract_number(val):
        match = re.search(r"(\d+)", str(val))
        return int(match.group(1)) if match else float("inf")
    
    # Check if category looks like it has numbers (SDG, Year, etc.)
    if merged[category_col].astype(str).str.contains(r"\d+").any():
        merged["_sort_key"] = merged[category_col].apply(extract_number)
        merged = merged.sort_values("_sort_key").drop(columns=["_sort_key"])
    else:
        merged = merged.sort_values(category_col)
    
    return merged.reset_index(drop=True)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_relative_representation(
    rep_df: pd.DataFrame,
    category_col: str,
    diff_col: str = "Difference (pp)",
    figsize: Tuple[float, float] = (12, 6),
    title: str = None,
    xlabel: str = None,
    ylabel: str = "Percentage Point Difference",
    cmap: str = "RdBu",  # Red for negative, Blue for positive
    vmin: float = None,
    vmax: float = None,
    annotate: bool = True,
    rotation: int = 0,
    filename: str = None,
    dpi: int = 300,
    ax: plt.Axes = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot relative representation as a bar chart with diverging colormap.
    
    Colors:
    - Blue shades: Over-represented (positive pp difference)
    - Red shades: Under-represented (negative pp difference)
    - Intensity reflects magnitude
    
    Parameters
    ----------
    rep_df : pd.DataFrame
        Output from compute_relative_representation().
    category_col : str
        Category column for x-axis.
    diff_col : str
        Difference column for y-axis and coloring.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    cmap : str
        Diverging colormap name (default: "RdBu" - red/blue).
    vmin, vmax : float
        Color scale limits. If None, auto-determined.
    annotate : bool
        Whether to show values on bars.
    rotation : int
        X-axis label rotation.
    filename : str
        If provided, save the plot.
    dpi : int
        Resolution.
    ax : plt.Axes
        Existing axes.
    
    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    categories = rep_df[category_col].astype(str)
    values = rep_df[diff_col].values
    
    # Handle empty data
    if len(values) == 0:
        print("Warning: No data to plot")
        return fig, ax
    
    # Determine color scale
    max_abs = max(abs(values.min()), abs(values.max())) if len(values) > 0 else 1
    if vmin is None:
        vmin = -max_abs
    if vmax is None:
        vmax = max_abs
    
    # Ensure we have valid range
    if vmin == vmax == 0:
        vmin, vmax = -1, 1
    
    # Create diverging norm centered at 0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap)
    
    # Get colors based on values
    colors = [cmap_obj(norm(v)) for v in values]
    
    # Create bars
    x = np.arange(len(categories))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
    
    # Horizontal line at 0
    ax.axhline(y=0, color="black", linewidth=1, linestyle="-")
    
    # Annotate bars
    if annotate:
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            va = "bottom" if height >= 0 else "top"
            offset = 3 if height >= 0 else -3
            ax.annotate(
                f"{val:+.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center", va=va,
                fontsize=8,
                fontweight="bold",
            )
    
    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=rotation, ha="right" if rotation > 0 else "center")
    
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Remove gridlines
    ax.grid(False)
    ax.set_facecolor("white")
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Percentage Point Difference", fontsize=10)
    
    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    if filename:
        for ext in [".png", ".svg", ".pdf"]:
            fig.savefig(filename + ext, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved to {filename}.(png/svg/pdf)")
    
    return fig, ax


def plot_distribution_comparison(
    rep_df: pd.DataFrame,
    category_col: str,
    figsize: Tuple[float, float] = (12, 6),
    title: str = None,
    rotation: int = 45,
    filename: str = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot observed vs reference distribution as grouped bars.
    
    Parameters
    ----------
    rep_df : pd.DataFrame
        Output from compute_relative_representation().
    category_col : str
        Category column.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    rotation : int
        X-axis label rotation.
    filename : str
        If provided, save the plot.
    dpi : int
        Resolution.
    
    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    categories = rep_df[category_col].astype(str)
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rep_df["Observed %"], width, 
                   label="Analyzed Dataset", color="#3498db", alpha=0.9)
    bars2 = ax.bar(x + width/2, rep_df["Reference %"], width, 
                   label="Global Reference", color="#95a5a6", alpha=0.9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=rotation, ha="right" if rotation > 0 else "center")
    
    ax.set_ylabel("Percentage (%)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    
    ax.legend(loc="upper right")
    
    # Remove gridlines
    ax.grid(False)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    if filename:
        for ext in [".png", ".svg", ".pdf"]:
            fig.savefig(filename + ext, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved to {filename}.(png/svg/pdf)")
    
    return fig, ax


# =============================================================================
# SUPPORTED REFERENCE TYPES
# =============================================================================

SUPPORTED_REFERENCES = {
    "Year": {
        "fetcher": fetch_openalex_yearly_counts,
        "category_col": "Year",
        "description": "Publication year distribution",
    },
    "Country": {
        "fetcher": fetch_openalex_country_counts,
        "category_col": "Country",
        "description": "Country distribution (by author affiliation)",
    },
    "SDG": {
        "fetcher": fetch_openalex_sdg_counts,
        "category_col": "SDG",
        "description": "Sustainable Development Goals distribution",
    },
    "Document Type": {
        "fetcher": fetch_openalex_doctype_counts,
        "category_col": "Document Type",
        "description": "Document type distribution",
    },
    "Open Access": {
        "fetcher": fetch_openalex_oa_counts,
        "category_col": "Open Access",
        "description": "Open access status distribution",
    },
}


def list_supported_references():
    """Print list of supported reference types for auto-fetch."""
    print("Supported reference types for auto-fetch from OpenAlex:")
    print("-" * 60)
    for name, info in SUPPORTED_REFERENCES.items():
        print(f"  • {name}: {info['description']}")
    print("-" * 60)
    print("Usage: ba.get_relative_representation('<type>')")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example with simulated data
    np.random.seed(42)
    
    # Simulated observed data
    observed = pd.DataFrame({
        "Year": list(range(2018, 2025)),
        "Count": [10, 20, 35, 50, 80, 120, 90],
    })
    
    # Simulated reference data
    reference = pd.DataFrame({
        "Year": list(range(2018, 2025)),
        "Count": [1000, 1100, 1200, 1300, 1400, 1500, 1600],
    })
    
    # Compute
    result = compute_relative_representation(observed, reference, "Year")
    print("Relative Representation Analysis:")
    print(result.to_string(index=False))
    
    # Plot
    plot_relative_representation(
        result, "Year",
        title="Analyzed Dataset vs Global Trends",
    )
    plt.show()
    
    # Show supported types
    list_supported_references()
