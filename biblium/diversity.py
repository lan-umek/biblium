# -*- coding: utf-8 -*-
"""
Research Diversity Indices for Biblium

Compute diversity metrics (Shannon, Simpson, Gini) for bibliometric entities
to measure research diversity across sources, authors, countries, keywords, etc.

Features:
- Shannon Index (entropy-based diversity)
- Simpson Index (probability-based diversity)  
- Gini Index (inequality measure)
- Support for all bibliometric entities
- Optional benchmarking against OpenAlex global data
- Radar plot visualization

@author: Lan.Umek
@version: 2.9.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Literal
from dataclasses import dataclass, field
import warnings

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DiversityResult:
    """Result of diversity index computation for a single entity."""
    entity: str
    shannon: float
    shannon_normalized: float  # 0-1 scale (H / ln(n))
    simpson: float  # 1 - D (diversity form)
    simpson_reciprocal: float  # 1/D
    gini: float
    n_categories: int
    n_items: int
    dominant_category: str
    dominant_share: float  # Percentage of dominant category
    
    def to_dict(self) -> dict:
        return {
            "Entity": self.entity,
            "Shannon Index": self.shannon,
            "Shannon (normalized)": self.shannon_normalized,
            "Simpson Diversity": self.simpson,
            "Simpson Reciprocal": self.simpson_reciprocal,
            "Gini Index": self.gini,
            "N Categories": self.n_categories,
            "N Items": self.n_items,
            "Dominant Category": self.dominant_category,
            "Dominant Share (%)": self.dominant_share,
        }


@dataclass
class DiversityAnalysisResult:
    """Complete diversity analysis result for multiple entities."""
    results: Dict[str, DiversityResult]
    benchmark: Optional[Dict[str, DiversityResult]] = None
    benchmark_source: str = ""
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = [r.to_dict() for r in self.results.values()]
        df = pd.DataFrame(rows)
        return df
    
    def to_comparison_dataframe(self) -> pd.DataFrame:
        """Create comparison DataFrame with benchmark if available."""
        rows = []
        for entity, result in self.results.items():
            row = {
                "Entity": entity,
                "Shannon": result.shannon_normalized,
                "Simpson": result.simpson,
                "Gini": result.gini,
            }
            if self.benchmark and entity in self.benchmark:
                bench = self.benchmark[entity]
                row["Shannon (Reference)"] = bench.shannon_normalized
                row["Simpson (Reference)"] = bench.simpson
                row["Gini (Reference)"] = bench.gini
                row["Shannon Δ"] = result.shannon_normalized - bench.shannon_normalized
                row["Simpson Δ"] = result.simpson - bench.simpson
                row["Gini Δ"] = result.gini - bench.gini
            rows.append(row)
        return pd.DataFrame(rows)
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "RESEARCH DIVERSITY ANALYSIS",
            "=" * 50,
            "",
            f"Entities analyzed: {len(self.results)}",
            "",
            "DIVERSITY INDICES BY ENTITY",
            "-" * 50,
            f"{'Entity':<20} {'Shannon':>10} {'Simpson':>10} {'Gini':>10}",
            "-" * 50,
        ]
        
        for entity, r in self.results.items():
            lines.append(
                f"{entity:<20} {r.shannon_normalized:>10.3f} {r.simpson:>10.3f} {r.gini:>10.3f}"
            )
        
        if self.benchmark:
            lines.extend([
                "",
                f"Benchmark: {self.benchmark_source}",
            ])
        
        return "\n".join(lines)


# =============================================================================
# DIVERSITY INDEX COMPUTATIONS
# =============================================================================

def compute_shannon_index(counts: np.ndarray) -> Tuple[float, float]:
    """
    Compute Shannon diversity index (entropy).
    
    H = -Σ(pᵢ × ln(pᵢ))
    
    Parameters
    ----------
    counts : array-like
        Count of items in each category.
        
    Returns
    -------
    tuple
        (H, H_normalized) where H_normalized = H / ln(n)
    """
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]  # Remove zero counts
    
    if len(counts) == 0:
        return 0.0, 0.0
    
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0
    
    proportions = counts / total
    
    # Shannon entropy: H = -Σ(p * ln(p))
    H = -np.sum(proportions * np.log(proportions))
    
    # Normalized (0-1): H / ln(n)
    n = len(counts)
    H_max = np.log(n) if n > 1 else 1
    H_normalized = H / H_max if H_max > 0 else 0
    
    return float(H), float(H_normalized)


def compute_simpson_index(counts: np.ndarray) -> Tuple[float, float]:
    """
    Compute Simpson diversity index.
    
    D = Σ(pᵢ²)  (concentration)
    Simpson Diversity = 1 - D
    Simpson Reciprocal = 1/D
    
    Parameters
    ----------
    counts : array-like
        Count of items in each category.
        
    Returns
    -------
    tuple
        (simpson_diversity, simpson_reciprocal)
    """
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]
    
    if len(counts) == 0:
        return 0.0, 0.0
    
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0
    
    proportions = counts / total
    
    # Simpson's D (concentration)
    D = np.sum(proportions ** 2)
    
    # Simpson's Diversity Index (1 - D)
    simpson_diversity = 1 - D
    
    # Simpson's Reciprocal Index (1/D)
    simpson_reciprocal = 1 / D if D > 0 else float('inf')
    
    return float(simpson_diversity), float(simpson_reciprocal)


def compute_gini_index(counts: np.ndarray) -> float:
    """
    Compute Gini coefficient (inequality measure).
    
    G = (2 × Σᵢ(i × xᵢ)) / (n × Σxᵢ) - (n+1)/n
    
    Parameters
    ----------
    counts : array-like
        Count of items in each category.
        
    Returns
    -------
    float
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    counts = np.array(counts, dtype=float)
    counts = counts[counts >= 0]  # Keep zeros for Gini
    
    if len(counts) == 0:
        return 0.0
    
    # Sort in ascending order
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    total = sorted_counts.sum()
    
    if total == 0 or n == 0:
        return 0.0
    
    # Gini coefficient formula
    cumsum = np.cumsum(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * total) - (n + 1) / n
    
    return float(max(0, min(1, gini)))  # Clamp to [0, 1]


def compute_diversity_indices(counts: np.ndarray, categories: List[str] = None) -> Dict:
    """
    Compute all diversity indices for a distribution.
    
    Parameters
    ----------
    counts : array-like
        Count of items in each category.
    categories : list, optional
        Category names (for identifying dominant category).
        
    Returns
    -------
    dict
        Dictionary with all computed indices.
    """
    counts = np.array(counts, dtype=float)
    
    # Shannon
    shannon, shannon_norm = compute_shannon_index(counts)
    
    # Simpson
    simpson_div, simpson_rec = compute_simpson_index(counts)
    
    # Gini
    gini = compute_gini_index(counts)
    
    # Dominant category
    n_items = int(counts.sum())
    n_categories = int((counts > 0).sum())
    
    if len(counts) > 0 and counts.sum() > 0:
        max_idx = np.argmax(counts)
        dominant_share = (counts[max_idx] / counts.sum()) * 100
        if categories is not None and max_idx < len(categories):
            dominant_category = str(categories[max_idx])
        else:
            dominant_category = f"Category {max_idx}"
    else:
        dominant_category = "N/A"
        dominant_share = 0.0
    
    return {
        "shannon": shannon,
        "shannon_normalized": shannon_norm,
        "simpson": simpson_div,
        "simpson_reciprocal": simpson_rec,
        "gini": gini,
        "n_categories": n_categories,
        "n_items": n_items,
        "dominant_category": dominant_category,
        "dominant_share": dominant_share,
    }


# =============================================================================
# ENTITY EXTRACTION AND COUNTING
# =============================================================================

# Entity configuration: (display_name, possible_column_names, is_multi_valued)
ENTITY_CONFIG = {
    "Sources": (["Source title", "Source", "SO", "Journal", "Publication Name"], True),
    "Authors": (["Authors", "Author full names", "AU", "AF"], True),
    "Countries": (["Countries of Authors", "Countries", "Country", "CU"], True),
    "Affiliations": (["Affiliations", "C1", "Addresses", "Author Affiliations"], True),
    "Author Keywords": (["Author Keywords", "DE", "Keywords"], True),
    "Index Keywords": (["Index Keywords", "ID", "Keywords Plus"], True),
    "Subject Areas": (["Research Areas", "SC", "Subject Area", "Fields", "Web of Science Categories", "WC"], True),
    "Document Types": (["Document Type", "DT", "Type"], False),
    "SDGs": (["SDG"], False),  # Special handling for SDG columns
    "References": (["References", "Cited References", "CR"], True),
    "Years": (["Year", "PY", "Publication Year"], False),
}


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    # Case-insensitive fallback
    df_cols_lower = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in df_cols_lower:
            return df_cols_lower[col.lower()]
    return None


def extract_entity_counts(
    df: pd.DataFrame,
    entity: str,
    separator: str = "; ",
    top_n: int = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract counts for an entity from the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliometric dataset.
    entity : str
        Entity name (Sources, Authors, Countries, etc.)
    separator : str
        Separator for multi-valued fields.
    top_n : int, optional
        Limit to top N categories.
        
    Returns
    -------
    tuple
        (counts array, category names list)
    """
    if entity not in ENTITY_CONFIG:
        raise ValueError(f"Unknown entity: {entity}. Available: {list(ENTITY_CONFIG.keys())}")
    
    candidates, is_multi = ENTITY_CONFIG[entity]
    
    # Special handling for SDGs
    if entity == "SDGs":
        return _extract_sdg_counts(df)
    
    col = find_column(df, candidates)
    if col is None:
        raise ValueError(f"No column found for entity '{entity}'. Tried: {candidates}")
    
    if is_multi:
        # Split and count multi-valued field
        values = df[col].dropna().astype(str)
        all_items = []
        for val in values:
            items = [x.strip() for x in val.split(separator) if x.strip()]
            all_items.extend(items)
        
        from collections import Counter
        counter = Counter(all_items)
    else:
        # Single-valued field
        from collections import Counter
        counter = Counter(df[col].dropna().astype(str))
    
    # Sort by count descending
    sorted_items = counter.most_common(top_n)
    
    if not sorted_items:
        return np.array([]), []
    
    categories = [item[0] for item in sorted_items]
    counts = np.array([item[1] for item in sorted_items])
    
    return counts, categories


def _extract_sdg_counts(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Extract SDG counts from binary SDG columns."""
    import re
    
    # Find SDG columns (SDG01, SDG1, SDG 1, etc.)
    sdg_cols = [c for c in df.columns if c.upper().startswith("SDG") and any(char.isdigit() for char in c)]
    
    if not sdg_cols:
        raise ValueError("No SDG columns found. Run identify_sdgs() first.")
    
    sdg_counts = {}
    for col in sdg_cols:
        match = re.search(r'(\d+)', col)
        if match:
            sdg_num = int(match.group(1))
            sdg_label = f"SDG {sdg_num}"
            # Sum the column
            col_sum = pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
            sdg_counts[sdg_label] = sdg_counts.get(sdg_label, 0) + int(col_sum)
    
    # Sort by SDG number
    sorted_sdgs = sorted(sdg_counts.items(), key=lambda x: int(x[0].split()[1]))
    
    categories = [item[0] for item in sorted_sdgs]
    counts = np.array([item[1] for item in sorted_sdgs])
    
    return counts, categories


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def compute_research_diversity(
    df: pd.DataFrame,
    entities: List[str] = None,
    separator: str = "; ",
    top_n_per_entity: int = None,
) -> DiversityAnalysisResult:
    """
    Compute diversity indices for multiple bibliometric entities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliometric dataset.
    entities : list, optional
        List of entities to analyze. If None, analyzes all available.
    separator : str
        Separator for multi-valued fields.
    top_n_per_entity : int, optional
        Limit categories per entity for computation.
        
    Returns
    -------
    DiversityAnalysisResult
        Complete analysis results.
        
    Example
    -------
    >>> result = compute_research_diversity(df)
    >>> print(result.summary())
    >>> result.to_dataframe()
    """
    if entities is None:
        # Auto-detect available entities
        entities = []
        for entity, (candidates, _) in ENTITY_CONFIG.items():
            if entity == "SDGs":
                # Check for SDG columns
                sdg_cols = [c for c in df.columns if c.upper().startswith("SDG") and any(char.isdigit() for char in c)]
                if sdg_cols:
                    entities.append(entity)
            else:
                if find_column(df, candidates) is not None:
                    entities.append(entity)
    
    results = {}
    
    for entity in entities:
        try:
            counts, categories = extract_entity_counts(
                df, entity, separator=separator, top_n=top_n_per_entity
            )
            
            if len(counts) == 0:
                print(f"Warning: No data for entity '{entity}', skipping.")
                continue
            
            indices = compute_diversity_indices(counts, categories)
            
            results[entity] = DiversityResult(
                entity=entity,
                shannon=indices["shannon"],
                shannon_normalized=indices["shannon_normalized"],
                simpson=indices["simpson"],
                simpson_reciprocal=indices["simpson_reciprocal"],
                gini=indices["gini"],
                n_categories=indices["n_categories"],
                n_items=indices["n_items"],
                dominant_category=indices["dominant_category"],
                dominant_share=indices["dominant_share"],
            )
            
        except Exception as e:
            print(f"Warning: Could not compute diversity for '{entity}': {e}")
    
    return DiversityAnalysisResult(results=results)


# =============================================================================
# OPENALEX BENCHMARK
# =============================================================================

def fetch_openalex_diversity_benchmark(
    year_range: Tuple[int, int] = None,
    entities: List[str] = None,
    timeout: int = 30,
) -> Dict[str, DiversityResult]:
    """
    Fetch global diversity indices from OpenAlex for benchmarking.
    
    Parameters
    ----------
    year_range : tuple, optional
        (start_year, end_year) filter.
    entities : list, optional
        Entities to fetch. Supports: Sources, Countries, Document Types, SDGs
    timeout : int
        API timeout.
        
    Returns
    -------
    dict
        Dictionary of DiversityResult by entity.
    """
    import requests
    
    if entities is None:
        entities = ["Sources", "Countries", "Document Types", "SDGs"]
    
    # Map entities to OpenAlex group_by parameters
    entity_mapping = {
        "Sources": "primary_location.source.id",
        "Countries": "authorships.countries",
        "Document Types": "type",
        "SDGs": "sustainable_development_goals.id",
        "Subject Areas": "topics.domain.id",
    }
    
    results = {}
    base_url = "https://api.openalex.org/works"
    
    for entity in entities:
        if entity not in entity_mapping:
            continue
        
        group_by = entity_mapping[entity]
        
        params = {
            "group_by": group_by,
            "per_page": 200,  # Get top 200 categories
        }
        
        if year_range:
            params["filter"] = f"publication_year:{year_range[0]}-{year_range[1]}"
        
        try:
            response = requests.get(base_url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            
            counts = []
            categories = []
            
            for item in data.get("group_by", []):
                count = item.get("count", 0)
                key = item.get("key_display_name", item.get("key", "Unknown"))
                counts.append(count)
                categories.append(str(key))
            
            if counts:
                counts_arr = np.array(counts)
                indices = compute_diversity_indices(counts_arr, categories)
                
                results[entity] = DiversityResult(
                    entity=entity,
                    shannon=indices["shannon"],
                    shannon_normalized=indices["shannon_normalized"],
                    simpson=indices["simpson"],
                    simpson_reciprocal=indices["simpson_reciprocal"],
                    gini=indices["gini"],
                    n_categories=indices["n_categories"],
                    n_items=indices["n_items"],
                    dominant_category=indices["dominant_category"],
                    dominant_share=indices["dominant_share"],
                )
                
        except Exception as e:
            print(f"Warning: Could not fetch OpenAlex data for '{entity}': {e}")
    
    return results


def compute_research_diversity_with_benchmark(
    df: pd.DataFrame,
    entities: List[str] = None,
    separator: str = "; ",
    fetch_benchmark: bool = True,
    year_range: Tuple[int, int] = None,
) -> DiversityAnalysisResult:
    """
    Compute diversity indices with optional OpenAlex benchmarking.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliometric dataset.
    entities : list, optional
        Entities to analyze.
    separator : str
        Separator for multi-valued fields.
    fetch_benchmark : bool
        Whether to fetch OpenAlex benchmark.
    year_range : tuple, optional
        Year range for benchmark filtering.
        
    Returns
    -------
    DiversityAnalysisResult
        Results with benchmark comparison.
    """
    # Compute local diversity
    result = compute_research_diversity(df, entities=entities, separator=separator)
    
    # Fetch benchmark if requested
    if fetch_benchmark:
        # Determine year range from data if not provided
        if year_range is None and "Year" in df.columns:
            year_range = (int(df["Year"].min()), int(df["Year"].max()))
        
        # Only benchmark entities that OpenAlex supports
        benchmark_entities = [e for e in result.results.keys() 
                           if e in ["Sources", "Countries", "Document Types", "SDGs", "Subject Areas"]]
        
        if benchmark_entities:
            print(f"Fetching OpenAlex benchmark for: {benchmark_entities}...")
            benchmark = fetch_openalex_diversity_benchmark(
                year_range=year_range,
                entities=benchmark_entities,
            )
            result.benchmark = benchmark
            result.benchmark_source = "OpenAlex Global"
            print(f"Benchmark fetched for {len(benchmark)} entities.")
    
    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_diversity_radar(
    result: DiversityAnalysisResult,
    indices: List[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    title: str = "Research Diversity Profile",
    show_benchmark: bool = True,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Create radar plot of diversity indices across entities.
    
    Parameters
    ----------
    result : DiversityAnalysisResult
        Diversity analysis result.
    indices : list, optional
        Which indices to plot. Default: ["Shannon (normalized)", "Simpson", "1-Gini"]
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    show_benchmark : bool
        Whether to show benchmark line.
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
    
    entities = list(result.results.keys())
    n_entities = len(entities)
    
    if n_entities < 3:
        print("Warning: Radar plot requires at least 3 entities. Using bar chart instead.")
        return plot_diversity_bars(result, figsize=figsize, title=title, filename=filename, dpi=dpi)
    
    # Prepare data - use normalized indices (0-1 scale)
    # Shannon normalized is already 0-1
    # Simpson diversity is already 0-1
    # Gini is 0-1 but inverted (low = diverse), so use 1-Gini
    
    shannon_vals = [result.results[e].shannon_normalized for e in entities]
    simpson_vals = [result.results[e].simpson for e in entities]
    gini_inv_vals = [1 - result.results[e].gini for e in entities]  # 1-Gini so higher = more diverse
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, n_entities, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot each index
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    labels = ['Shannon (H\')', 'Simpson (1-D)', 'Equality (1-G)']
    
    for vals, color, label in zip(
        [shannon_vals, simpson_vals, gini_inv_vals],
        colors,
        labels
    ):
        values = vals + vals[:1]  # Close polygon
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=label)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Plot benchmark if available
    if show_benchmark and result.benchmark:
        bench_shannon = []
        bench_simpson = []
        bench_gini_inv = []
        
        for e in entities:
            if e in result.benchmark:
                bench_shannon.append(result.benchmark[e].shannon_normalized)
                bench_simpson.append(result.benchmark[e].simpson)
                bench_gini_inv.append(1 - result.benchmark[e].gini)
            else:
                bench_shannon.append(np.nan)
                bench_simpson.append(np.nan)
                bench_gini_inv.append(np.nan)
        
        # Plot benchmark as dashed lines
        for vals, color, label in zip(
            [bench_shannon, bench_simpson, bench_gini_inv],
            colors,
            ['Shannon (Ref)', 'Simpson (Ref)', 'Equality (Ref)']
        ):
            if not all(np.isnan(vals)):
                values = vals + vals[:1]
                ax.plot(angles, values, '--', linewidth=1.5, color=color, alpha=0.6)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(entities, size=10)
    
    # Set y limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    
    ax.set_title(title, size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(False)  # No gridlines
    ax.spines['polar'].set_visible(False)  # Remove polar spine
    
    plt.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved to {filename}")
    
    return fig, ax


def plot_diversity_bars(
    result: DiversityAnalysisResult,
    figsize: Tuple[float, float] = (12, 6),
    title: str = "Research Diversity Indices",
    show_benchmark: bool = True,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Create grouped bar chart of diversity indices.
    
    Parameters
    ----------
    result : DiversityAnalysisResult
        Diversity analysis result.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    show_benchmark : bool
        Whether to show benchmark bars.
    filename : str
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
    
    entities = list(result.results.keys())
    n_entities = len(entities)
    
    # Prepare data
    shannon_vals = [result.results[e].shannon_normalized for e in entities]
    simpson_vals = [result.results[e].simpson for e in entities]
    gini_inv_vals = [1 - result.results[e].gini for e in entities]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_entities)
    width = 0.25
    
    # Plot bars
    bars1 = ax.bar(x - width, shannon_vals, width, label='Shannon (H\')', color='#3498db', alpha=0.9)
    bars2 = ax.bar(x, simpson_vals, width, label='Simpson (1-D)', color='#e74c3c', alpha=0.9)
    bars3 = ax.bar(x + width, gini_inv_vals, width, label='Equality (1-G)', color='#2ecc71', alpha=0.9)
    
    # Add benchmark markers if available
    if show_benchmark and result.benchmark:
        for i, e in enumerate(entities):
            if e in result.benchmark:
                b = result.benchmark[e]
                ax.scatter(i - width, b.shannon_normalized, marker='_', s=200, color='#2c3e50', linewidths=3, zorder=5)
                ax.scatter(i, b.simpson, marker='_', s=200, color='#2c3e50', linewidths=3, zorder=5)
                ax.scatter(i + width, 1 - b.gini, marker='_', s=200, color='#2c3e50', linewidths=3, zorder=5)
        
        # Add to legend
        ax.scatter([], [], marker='_', s=200, color='#2c3e50', linewidths=3, label='Reference (OpenAlex)')
    
    ax.set_xlabel('Entity', fontsize=11)
    ax.set_ylabel('Index Value (0-1)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(entities, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    ax.grid(False)  # No gridlines
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved to {filename}")
    
    return fig, ax


# =============================================================================
# INTERPRETATION HELPERS
# =============================================================================

def interpret_diversity(result: DiversityResult) -> str:
    """
    Generate human-readable interpretation of diversity indices.
    
    Parameters
    ----------
    result : DiversityResult
        Single entity diversity result.
        
    Returns
    -------
    str
        Interpretation text.
    """
    lines = [f"Diversity Analysis: {result.entity}", "=" * 40]
    
    # Shannon interpretation
    if result.shannon_normalized >= 0.8:
        shannon_interp = "Very high diversity (evenly distributed)"
    elif result.shannon_normalized >= 0.6:
        shannon_interp = "High diversity"
    elif result.shannon_normalized >= 0.4:
        shannon_interp = "Moderate diversity"
    elif result.shannon_normalized >= 0.2:
        shannon_interp = "Low diversity (concentrated)"
    else:
        shannon_interp = "Very low diversity (highly concentrated)"
    
    lines.append(f"Shannon (normalized): {result.shannon_normalized:.3f} - {shannon_interp}")
    
    # Simpson interpretation
    if result.simpson >= 0.8:
        simpson_interp = "High diversity (low probability of same category)"
    elif result.simpson >= 0.5:
        simpson_interp = "Moderate diversity"
    else:
        simpson_interp = "Low diversity (high concentration)"
    
    lines.append(f"Simpson diversity: {result.simpson:.3f} - {simpson_interp}")
    
    # Gini interpretation
    if result.gini <= 0.3:
        gini_interp = "Low inequality (relatively equal distribution)"
    elif result.gini <= 0.5:
        gini_interp = "Moderate inequality"
    elif result.gini <= 0.7:
        gini_interp = "High inequality"
    else:
        gini_interp = "Very high inequality (dominated by few categories)"
    
    lines.append(f"Gini coefficient: {result.gini:.3f} - {gini_interp}")
    
    # Dominant category
    lines.append(f"\nDominant: {result.dominant_category} ({result.dominant_share:.1f}%)")
    lines.append(f"Categories: {result.n_categories}, Items: {result.n_items}")
    
    return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def list_available_entities(df: pd.DataFrame) -> List[str]:
    """List entities available for diversity analysis in the dataframe."""
    available = []
    for entity, (candidates, _) in ENTITY_CONFIG.items():
        if entity == "SDGs":
            sdg_cols = [c for c in df.columns if c.upper().startswith("SDG") and any(char.isdigit() for char in c)]
            if sdg_cols:
                available.append(entity)
        else:
            if find_column(df, candidates) is not None:
                available.append(entity)
    return available


# =============================================================================
# TEMPORAL DIVERSITY ANALYSIS
# =============================================================================

@dataclass
class TemporalDiversityResult:
    """Result of temporal diversity analysis."""
    entity: str
    years: List[int]
    shannon_series: List[float]
    simpson_series: List[float]
    gini_series: List[float]
    n_categories_series: List[int]
    n_items_series: List[int]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            "Year": self.years,
            "Shannon (normalized)": self.shannon_series,
            "Simpson (1-D)": self.simpson_series,
            "Gini": self.gini_series,
            "N Categories": self.n_categories_series,
            "N Items": self.n_items_series,
        })
    
    def get_trend(self, index: str = "shannon") -> Tuple[float, str]:
        """
        Calculate trend direction and slope.
        
        Returns
        -------
        tuple
            (slope, direction) where direction is 'increasing', 'decreasing', or 'stable'
        """
        if index == "shannon":
            values = self.shannon_series
        elif index == "simpson":
            values = self.simpson_series
        elif index == "gini":
            values = self.gini_series
        else:
            raise ValueError(f"Unknown index: {index}")
        
        if len(values) < 2:
            return 0.0, "stable"
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Remove NaN
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0.0, "stable"
        
        x, y = x[mask], y[mask]
        slope = np.polyfit(x, y, 1)[0]
        
        if abs(slope) < 0.005:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return float(slope), direction


@dataclass 
class TemporalDiversityAnalysisResult:
    """Complete temporal diversity analysis for multiple entities."""
    results: Dict[str, TemporalDiversityResult]
    year_range: Tuple[int, int]
    
    def to_dataframe(self, entity: str = None) -> pd.DataFrame:
        """Get DataFrame for specific entity or combined."""
        if entity:
            if entity not in self.results:
                raise ValueError(f"Entity '{entity}' not in results")
            return self.results[entity].to_dataframe()
        
        # Combined DataFrame
        dfs = []
        for ent, res in self.results.items():
            df = res.to_dataframe()
            df["Entity"] = ent
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "TEMPORAL DIVERSITY ANALYSIS",
            "=" * 50,
            f"Year range: {self.year_range[0]} - {self.year_range[1]}",
            f"Entities analyzed: {len(self.results)}",
            "",
            "DIVERSITY TRENDS",
            "-" * 50,
        ]
        
        for entity, res in self.results.items():
            slope, direction = res.get_trend("shannon")
            lines.append(f"{entity}: Shannon {direction} (slope={slope:.4f})")
        
        return "\n".join(lines)


def compute_temporal_diversity(
    df: pd.DataFrame,
    entity: str,
    year_col: str = "Year",
    separator: str = "; ",
    min_items_per_year: int = 5,
) -> TemporalDiversityResult:
    """
    Compute diversity indices over time for a single entity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliometric dataset.
    entity : str
        Entity to analyze (Sources, Authors, Countries, etc.)
    year_col : str
        Column containing publication year.
    separator : str
        Separator for multi-valued fields.
    min_items_per_year : int
        Minimum items required to compute diversity for a year.
        
    Returns
    -------
    TemporalDiversityResult
        Time series of diversity indices.
    """
    if year_col not in df.columns:
        raise ValueError(f"Year column '{year_col}' not found")
    
    # Get unique years
    years = sorted(df[year_col].dropna().astype(int).unique())
    
    shannon_series = []
    simpson_series = []
    gini_series = []
    n_categories_series = []
    n_items_series = []
    valid_years = []
    
    for year in years:
        year_df = df[df[year_col] == year]
        
        if len(year_df) < min_items_per_year:
            continue
        
        try:
            counts, categories = extract_entity_counts(year_df, entity, separator=separator)
            
            if len(counts) == 0:
                continue
            
            indices = compute_diversity_indices(counts, categories)
            
            valid_years.append(year)
            shannon_series.append(indices["shannon_normalized"])
            simpson_series.append(indices["simpson"])
            gini_series.append(indices["gini"])
            n_categories_series.append(indices["n_categories"])
            n_items_series.append(indices["n_items"])
            
        except Exception:
            continue
    
    return TemporalDiversityResult(
        entity=entity,
        years=valid_years,
        shannon_series=shannon_series,
        simpson_series=simpson_series,
        gini_series=gini_series,
        n_categories_series=n_categories_series,
        n_items_series=n_items_series,
    )


def compute_temporal_diversity_multi(
    df: pd.DataFrame,
    entities: List[str] = None,
    year_col: str = "Year",
    separator: str = "; ",
    min_items_per_year: int = 5,
) -> TemporalDiversityAnalysisResult:
    """
    Compute temporal diversity for multiple entities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliometric dataset.
    entities : list, optional
        Entities to analyze. If None, auto-detects available.
    year_col : str
        Column containing publication year.
    separator : str
        Separator for multi-valued fields.
    min_items_per_year : int
        Minimum items per year.
        
    Returns
    -------
    TemporalDiversityAnalysisResult
        Results for all entities.
    """
    if entities is None:
        entities = list_available_entities(df)
        # Remove Years from the list (doesn't make sense for temporal analysis)
        entities = [e for e in entities if e != "Years"]
    
    results = {}
    
    for entity in entities:
        try:
            result = compute_temporal_diversity(
                df, entity, year_col=year_col, 
                separator=separator, min_items_per_year=min_items_per_year
            )
            if len(result.years) > 0:
                results[entity] = result
        except Exception as e:
            print(f"Warning: Could not compute temporal diversity for '{entity}': {e}")
    
    # Get year range
    if year_col in df.columns:
        year_range = (int(df[year_col].min()), int(df[year_col].max()))
    else:
        year_range = (0, 0)
    
    return TemporalDiversityAnalysisResult(results=results, year_range=year_range)


def plot_temporal_diversity(
    result: Union[TemporalDiversityResult, TemporalDiversityAnalysisResult],
    entities: List[str] = None,
    index: str = "shannon",
    figsize: Tuple[float, float] = (12, 6),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Plot diversity indices over time.
    
    Parameters
    ----------
    result : TemporalDiversityResult or TemporalDiversityAnalysisResult
        Temporal diversity result.
    entities : list, optional
        Entities to plot (for multi-entity result).
    index : str
        Which index to plot: 'shannon', 'simpson', 'gini', or 'all'
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
    
    # Handle single entity result
    if isinstance(result, TemporalDiversityResult):
        results = {result.entity: result}
    else:
        results = result.results
    
    if entities:
        results = {k: v for k, v in results.items() if k in entities}
    
    if not results:
        raise ValueError("No results to plot")
    
    # Determine plot layout
    if index == "all":
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0], figsize[1]))
        indices = ["shannon", "simpson", "gini"]
        titles = ["Shannon (H')", "Simpson (1-D)", "Gini (G)"]
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        indices = [index]
        titles = [{"shannon": "Shannon (H')", "simpson": "Simpson (1-D)", "gini": "Gini (G)"}[index]]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for ax, idx, idx_title in zip(axes, indices, titles):
        for (entity, res), color in zip(results.items(), colors):
            if idx == "shannon":
                values = res.shannon_series
            elif idx == "simpson":
                values = res.simpson_series
            else:
                values = res.gini_series
            
            ax.plot(res.years, values, 'o-', label=entity, color=color, linewidth=2, markersize=4)
        
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel(idx_title, fontsize=10)
        ax.set_title(idx_title if index == "all" else (title or f"Temporal Diversity: {idx_title}"), 
                    fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(False)
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    if title and index == "all":
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, axes[0] if len(axes) == 1 else axes


def plot_temporal_diversity_heatmap(
    result: TemporalDiversityAnalysisResult,
    index: str = "shannon",
    figsize: Tuple[float, float] = (14, 8),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Create heatmap of diversity over time for multiple entities.
    
    Parameters
    ----------
    result : TemporalDiversityAnalysisResult
        Multi-entity temporal diversity result.
    index : str
        Which index: 'shannon', 'simpson', 'gini'
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
    
    # Build matrix
    entities = list(result.results.keys())
    all_years = set()
    for res in result.results.values():
        all_years.update(res.years)
    years = sorted(all_years)
    
    matrix = np.full((len(entities), len(years)), np.nan)
    
    for i, entity in enumerate(entities):
        res = result.results[entity]
        for year, val in zip(res.years, 
                            res.shannon_series if index == "shannon" 
                            else res.simpson_series if index == "simpson" 
                            else res.gini_series):
            if year in years:
                j = years.index(year)
                matrix[i, j] = val
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(entities)))
    ax.set_yticklabels(entities, fontsize=10)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    index_labels = {"shannon": "Shannon (H')", "simpson": "Simpson (1-D)", "gini": "Gini (G)"}
    cbar.set_label(index_labels.get(index, index), fontsize=10)
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Entity", fontsize=11)
    ax.set_title(title or f"Temporal Diversity Heatmap: {index_labels.get(index, index)}", 
                fontsize=12, fontweight="bold")
    
    # Remove any gridlines
    ax.grid(False)
    ax.tick_params(length=0)  # Remove tick marks
    
    # Use constrained_layout or add padding for labels
    fig.tight_layout(pad=1.5, rect=[0.1, 0.1, 0.95, 0.95])
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


# =============================================================================
# GROUP DIVERSITY ANALYSIS
# =============================================================================

@dataclass
class GroupDiversityResult:
    """Diversity analysis result for a single group."""
    group_name: str
    entity_results: Dict[str, DiversityResult]
    n_documents: int
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        rows = []
        for entity, res in self.entity_results.items():
            row = res.to_dict()
            row["Group"] = self.group_name
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class GroupDiversityAnalysisResult:
    """Complete group diversity comparison."""
    group_results: Dict[str, GroupDiversityResult]
    entities: List[str]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to DataFrame."""
        dfs = [gr.to_dataframe() for gr in self.group_results.values()]
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def to_comparison_dataframe(self, index: str = "shannon") -> pd.DataFrame:
        """
        Create comparison DataFrame across groups.
        
        Parameters
        ----------
        index : str
            Which index: 'shannon', 'simpson', 'gini'
            
        Returns
        -------
        pd.DataFrame
            Entities as rows, groups as columns.
        """
        groups = list(self.group_results.keys())
        
        data = {"Entity": self.entities}
        for group in groups:
            gr = self.group_results[group]
            values = []
            for entity in self.entities:
                if entity in gr.entity_results:
                    res = gr.entity_results[entity]
                    if index == "shannon":
                        values.append(res.shannon_normalized)
                    elif index == "simpson":
                        values.append(res.simpson)
                    elif index == "gini":
                        values.append(res.gini)
                    else:
                        values.append(np.nan)
                else:
                    values.append(np.nan)
            data[group] = values
        
        return pd.DataFrame(data)
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "GROUP DIVERSITY COMPARISON",
            "=" * 50,
            f"Groups: {len(self.group_results)}",
            f"Entities: {len(self.entities)}",
            "",
        ]
        
        for group, gr in self.group_results.items():
            lines.append(f"\n{group} (n={gr.n_documents})")
            lines.append("-" * 30)
            for entity, res in gr.entity_results.items():
                lines.append(f"  {entity}: H'={res.shannon_normalized:.3f}, 1-D={res.simpson:.3f}, G={res.gini:.3f}")
        
        return "\n".join(lines)


def compute_group_diversity(
    groups: Dict[str, pd.DataFrame],
    entities: List[str] = None,
    separator: str = "; ",
) -> GroupDiversityAnalysisResult:
    """
    Compute diversity indices for multiple groups.
    
    Parameters
    ----------
    groups : dict
        Dictionary of {group_name: DataFrame}
    entities : list, optional
        Entities to analyze. If None, auto-detects from first group.
    separator : str
        Separator for multi-valued fields.
        
    Returns
    -------
    GroupDiversityAnalysisResult
        Comparison of diversity across groups.
        
    Example
    -------
    >>> groups = {
    ...     "2010-2015": df[df["Year"].between(2010, 2015)],
    ...     "2016-2020": df[df["Year"].between(2016, 2020)],
    ... }
    >>> result = compute_group_diversity(groups)
    >>> result.to_comparison_dataframe("shannon")
    """
    if not groups:
        raise ValueError("No groups provided")
    
    # Auto-detect entities from first group
    if entities is None:
        first_df = list(groups.values())[0]
        entities = list_available_entities(first_df)
        entities = [e for e in entities if e != "Years"]
    
    group_results = {}
    
    for group_name, group_df in groups.items():
        entity_results = {}
        
        for entity in entities:
            try:
                counts, categories = extract_entity_counts(group_df, entity, separator=separator)
                
                if len(counts) == 0:
                    continue
                
                indices = compute_diversity_indices(counts, categories)
                
                entity_results[entity] = DiversityResult(
                    entity=entity,
                    shannon=indices["shannon"],
                    shannon_normalized=indices["shannon_normalized"],
                    simpson=indices["simpson"],
                    simpson_reciprocal=indices["simpson_reciprocal"],
                    gini=indices["gini"],
                    n_categories=indices["n_categories"],
                    n_items=indices["n_items"],
                    dominant_category=indices["dominant_category"],
                    dominant_share=indices["dominant_share"],
                )
            except Exception as e:
                print(f"Warning: Could not compute diversity for '{entity}' in group '{group_name}': {e}")
        
        group_results[group_name] = GroupDiversityResult(
            group_name=group_name,
            entity_results=entity_results,
            n_documents=len(group_df),
        )
    
    return GroupDiversityAnalysisResult(
        group_results=group_results,
        entities=entities,
    )


def compute_group_diversity_from_bib_group(
    bib_group,
    entities: List[str] = None,
    separator: str = "; ",
) -> GroupDiversityAnalysisResult:
    """
    Compute diversity from BiblioGroupAnalysis object.
    
    Parameters
    ----------
    bib_group : BiblioGroupAnalysis
        Group analysis object with .groups attribute.
    entities : list, optional
        Entities to analyze.
    separator : str
        Separator for multi-valued fields.
        
    Returns
    -------
    GroupDiversityAnalysisResult
        Group diversity comparison.
    """
    # Extract DataFrames from group object
    groups = {}
    
    if hasattr(bib_group, 'groups') and isinstance(bib_group.groups, dict):
        for name, group_obj in bib_group.groups.items():
            if hasattr(group_obj, 'df'):
                groups[name] = group_obj.df
            elif isinstance(group_obj, pd.DataFrame):
                groups[name] = group_obj
    elif hasattr(bib_group, 'group_dfs'):
        groups = bib_group.group_dfs
    else:
        raise ValueError("Could not extract group DataFrames from bib_group object")
    
    return compute_group_diversity(groups, entities=entities, separator=separator)


def plot_group_diversity_comparison(
    result: GroupDiversityAnalysisResult,
    index: str = "shannon",
    figsize: Tuple[float, float] = (12, 6),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Create grouped bar chart comparing diversity across groups.
    
    Parameters
    ----------
    result : GroupDiversityAnalysisResult
        Group diversity result.
    index : str
        Which index: 'shannon', 'simpson', 'gini'
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
    
    df = result.to_comparison_dataframe(index)
    entities = df["Entity"].tolist()
    groups = [c for c in df.columns if c != "Entity"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(entities))
    width = 0.8 / len(groups)
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    
    for i, (group, color) in enumerate(zip(groups, colors)):
        offset = (i - len(groups)/2 + 0.5) * width
        values = df[group].fillna(0).tolist()
        ax.bar(x + offset, values, width, label=group, color=color, alpha=0.9)
    
    index_labels = {"shannon": "Shannon (H')", "simpson": "Simpson (1-D)", "gini": "Gini (G)"}
    
    ax.set_xlabel("Entity", fontsize=11)
    ax.set_ylabel(index_labels.get(index, index), fontsize=11)
    ax.set_title(title or f"Group Diversity Comparison: {index_labels.get(index, index)}", 
                fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(entities, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(False)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_group_diversity_radar(
    result: GroupDiversityAnalysisResult,
    entity: str = None,
    figsize: Tuple[float, float] = (10, 8),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Create radar plot comparing groups for a specific entity or all indices.
    
    Parameters
    ----------
    result : GroupDiversityAnalysisResult
        Group diversity result.
    entity : str, optional
        Specific entity to compare. If None, compares all entities using Shannon.
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
    
    groups = list(result.group_results.keys())
    
    if entity:
        # Compare all indices for one entity across groups
        categories = ["Shannon (H')", "Simpson (1-D)", "Equality (1-G)"]
        n_cats = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        
        for group, color in zip(groups, colors):
            gr = result.group_results[group]
            if entity in gr.entity_results:
                res = gr.entity_results[entity]
                values = [res.shannon_normalized, res.simpson, 1 - res.gini]
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, color=color, label=group)
                ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.set_title(title or f"Group Comparison: {entity}", fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(False)
        # Remove radial gridlines and spines for cleaner look
        ax.set_yticklabels([])
        ax.spines['polar'].set_visible(False)
        
    else:
        # Compare all entities using Shannon across groups
        entities = result.entities
        n_entities = len(entities)
        
        if n_entities < 3:
            # Fall back to bar chart
            return plot_group_diversity_comparison(result, index="shannon", figsize=figsize, 
                                                   title=title, filename=filename, dpi=dpi)
        
        angles = np.linspace(0, 2 * np.pi, n_entities, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        
        for group, color in zip(groups, colors):
            gr = result.group_results[group]
            values = []
            for ent in entities:
                if ent in gr.entity_results:
                    values.append(gr.entity_results[ent].shannon_normalized)
                else:
                    values.append(0)
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, color=color, label=group)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(entities, size=9)
        ax.set_ylim(0, 1)
        ax.set_title(title or "Group Diversity Comparison (Shannon)", fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(False)
        # Remove radial gridlines and spines for cleaner look
        ax.set_yticklabels([])
        ax.spines['polar'].set_visible(False)
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


def plot_group_diversity_heatmap(
    result: GroupDiversityAnalysisResult,
    index: str = "shannon",
    figsize: Tuple[float, float] = (10, 8),
    title: str = None,
    filename: str = None,
    dpi: int = 300,
) -> Tuple:
    """
    Create heatmap of diversity across groups and entities.
    
    Parameters
    ----------
    result : GroupDiversityAnalysisResult
        Group diversity result.
    index : str
        Which index: 'shannon', 'simpson', 'gini'
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
    
    df = result.to_comparison_dataframe(index)
    entities = df["Entity"].tolist()
    groups = [c for c in df.columns if c != "Entity"]
    
    matrix = df[groups].values
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(entities)))
    ax.set_yticklabels(entities, fontsize=10)
    
    # Add text annotations
    for i in range(len(entities)):
        for j in range(len(groups)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                       color=text_color, fontsize=9)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    index_labels = {"shannon": "Shannon (H')", "simpson": "Simpson (1-D)", "gini": "Gini (G)"}
    cbar.set_label(index_labels.get(index, index), fontsize=10)
    
    ax.set_xlabel("Group", fontsize=11)
    ax.set_ylabel("Entity", fontsize=11)
    ax.set_title(title or f"Group Diversity Heatmap: {index_labels.get(index, index)}", 
                fontsize=12, fontweight="bold")
    
    # Remove any gridlines
    ax.grid(False)
    ax.tick_params(length=0)  # Remove tick marks
    
    # Use constrained_layout or add padding for labels
    fig.tight_layout(pad=1.5, rect=[0.1, 0.1, 0.95, 0.95])
    
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    
    return fig, ax


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    "DiversityResult",
    "DiversityAnalysisResult",
    "TemporalDiversityResult",
    "TemporalDiversityAnalysisResult",
    "GroupDiversityResult",
    "GroupDiversityAnalysisResult",
    # Core functions
    "compute_shannon_index",
    "compute_simpson_index", 
    "compute_gini_index",
    "compute_diversity_indices",
    "compute_research_diversity",
    "compute_research_diversity_with_benchmark",
    # Temporal diversity
    "compute_temporal_diversity",
    "compute_temporal_diversity_multi",
    "plot_temporal_diversity",
    "plot_temporal_diversity_heatmap",
    # Group diversity
    "compute_group_diversity",
    "compute_group_diversity_from_bib_group",
    "plot_group_diversity_comparison",
    "plot_group_diversity_radar",
    "plot_group_diversity_heatmap",
    # OpenAlex
    "fetch_openalex_diversity_benchmark",
    # Visualization
    "plot_diversity_radar",
    "plot_diversity_bars",
    # Helpers
    "interpret_diversity",
    "list_available_entities",
    "extract_entity_counts",
    "ENTITY_CONFIG",
]
