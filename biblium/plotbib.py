from __future__ import annotations

# --- Standard library ---
import os
import math
import itertools
import textwrap
import re
from collections import Counter, defaultdict
from datetime import datetime
import warnings

# --- Typing ---
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# --- Data handling ---
import numpy as np
import pandas as pd

# --- Visualization: Matplotlib & Seaborn ---
import matplotlib.pyplot as plt
from matplotlib import cm, colors, rcParams
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm

from typing import Iterable    

import seaborn as sns

# --- Other visualization libraries ---
from adjustText import adjust_text
import squarify

# UpSet plots
try:
    from upsetplot import UpSet, from_indicators
except ImportError:
    UpSet = from_indicators = None

# Venn diagrams
try:
    from venn import venn
except ImportError:
    venn = None

# Word clouds
try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

# Holoviews (optional)
try:
    import holoviews as hv
    from holoviews import opts
except ImportError:
    hv = opts = None

# Plotly (optional interactive plots)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale, get_colorscale
except ImportError:
    px = go = sample_colorscale = get_colorscale = None

# --- Clustering/Distance ---
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import kruskal, norm, pearsonr
import scipy.cluster.hierarchy as sch

import networkx as nx
from textwrap import wrap

# --- Local utilities ---
from biblium import utilsbib

# --- Sentiment plots ---
try:
    from biblium.plotting.sentiment_plots import (
        plot_sentiment_distribution,
        plot_sentiment_categories,
        plot_sentiment_temporal,
        plot_sentiment_certainty,
        plot_sentiment_by_source,
        plot_sentiment_heatmap,
    )
    HAS_SENTIMENT_PLOTS = True
except ImportError:
    HAS_SENTIMENT_PLOTS = False



def wrap_labels(labels, width=50):
    """Wrap long labels to a given width."""
    return ["\n".join(textwrap.wrap(label, width)) for label in labels]


def clean_label(label: str) -> str:
    """
    Clean a label for display by replacing underscores with spaces.
    
    Parameters
    ----------
    label : str
        The label to clean.
        
    Returns
    -------
    str
        Cleaned label with underscores replaced by spaces.
    """
    if not isinstance(label, str):
        label = str(label)
    return label.replace("_", " ")


def clean_labels(labels: List[str]) -> List[str]:
    """
    Clean a list of labels for display.
    
    Parameters
    ----------
    labels : list of str
        Labels to clean.
        
    Returns
    -------
    list of str
        Cleaned labels.
    """
    return [clean_label(str(l)) for l in labels]


# =============================================================================
# LABEL SHORTENING UTILITIES
# =============================================================================

def shorten_label(
    label: str,
    max_length: int = 40,
    mode: str = "smart",
    ellipsis: str = "...",
) -> str:
    """
    Shorten a single label intelligently.
    
    Parameters
    ----------
    label : str
        The label to shorten.
    max_length : int, default 40
        Maximum length of the output label.
    mode : str, default "smart"
        Shortening mode:
        - "truncate": Simple truncation with ellipsis
        - "middle": Keep start and end, ellipsis in middle
        - "smart": Context-aware shortening (best for references, affiliations)
        - "words": Truncate at word boundary
    ellipsis : str, default "..."
        String to use as ellipsis.
        
    Returns
    -------
    str
        Shortened label.
    """
    if not isinstance(label, str):
        label = str(label)
    
    if len(label) <= max_length:
        return label
    
    if mode == "truncate":
        return label[:max_length - len(ellipsis)] + ellipsis
    
    elif mode == "middle":
        # Keep start and end, put ellipsis in middle
        keep = max_length - len(ellipsis)
        start = keep // 2
        end = keep - start
        return label[:start] + ellipsis + label[-end:]
    
    elif mode == "words":
        # Truncate at word boundary
        truncated = label[:max_length - len(ellipsis)]
        # Find last space
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:
            return truncated[:last_space] + ellipsis
        return truncated + ellipsis
    
    elif mode == "smart":
        return _smart_shorten(label, max_length, ellipsis)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _smart_shorten(label: str, max_length: int = 40, ellipsis: str = "...") -> str:
    """
    Smart shortening that understands bibliometric label patterns.
    
    Handles:
    - References (Author et al., Year, Journal)
    - Affiliations (University, Department, Country)
    - Journal names (abbreviations)
    - Author names
    """
    if len(label) <= max_length:
        return label
    
    # Pattern 1: Reference format "Author(s) (Year) Title, Journal"
    # Try to keep Author + Year + abbreviated title
    ref_match = re.match(r'^([^(]+)\((\d{4})\)\s*(.+)$', label)
    if ref_match:
        author, year, rest = ref_match.groups()
        author = author.strip()
        # Keep first author only if multiple
        if "," in author:
            author = author.split(",")[0].strip() + " et al."
        base = f"{author} ({year}) "
        remaining = max_length - len(base) - len(ellipsis)
        if remaining > 10:
            return base + rest[:remaining] + ellipsis
    
    # Pattern 2: Affiliation with country "University, Dept, City, Country"
    # Keep institution and country
    if "," in label:
        parts = [p.strip() for p in label.split(",")]
        if len(parts) >= 2:
            # Try to keep first and last parts (institution + country)
            first = parts[0]
            last = parts[-1]
            
            # Common country patterns
            if len(last) <= 20:  # Likely a country
                combined = f"{first}, {last}"
                if len(combined) <= max_length:
                    return combined
                # Shorten the institution name
                inst_max = max_length - len(last) - 4 - len(ellipsis)
                if inst_max > 10:
                    return first[:inst_max] + ellipsis + ", " + last
    
    # Pattern 3: Journal names - apply common abbreviations
    abbreviated = _abbreviate_journal(label)
    if len(abbreviated) <= max_length:
        return abbreviated
    
    # Pattern 4: DOI format
    if label.startswith("10.") or "doi.org" in label.lower():
        # Keep just the DOI suffix
        if "/" in label:
            parts = label.split("/")
            return ellipsis + parts[-1][:max_length - len(ellipsis)]
    
    # Default: word-boundary truncation
    truncated = label[:max_length - len(ellipsis)]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 3:
        return truncated[:last_space] + ellipsis
    return truncated + ellipsis


def _abbreviate_journal(name: str) -> str:
    """Apply common journal name abbreviations."""
    abbreviations = {
        "Journal": "J.",
        "International": "Int.",
        "American": "Am.",
        "European": "Eur.",
        "British": "Br.",
        "Review": "Rev.",
        "Research": "Res.",
        "Science": "Sci.",
        "Sciences": "Sci.",
        "Technology": "Technol.",
        "Engineering": "Eng.",
        "Management": "Mgmt.",
        "Economics": "Econ.",
        "Quarterly": "Q.",
        "Annual": "Ann.",
        "Proceedings": "Proc.",
        "Transactions": "Trans.",
        "Letters": "Lett.",
        "Communications": "Commun.",
        "University": "Univ.",
        "Society": "Soc.",
        "Association": "Assoc.",
        "National": "Natl.",
        "Department": "Dept.",
        "Institute": "Inst.",
        "Information": "Inf.",
        "Environmental": "Environ.",
        "Biological": "Biol.",
        "Chemical": "Chem.",
        "Physical": "Phys.",
        "Mathematical": "Math.",
        "Statistical": "Stat.",
        "Computational": "Comput.",
        "Applied": "Appl.",
        "Theoretical": "Theor.",
        "Experimental": "Exp.",
        "Development": "Dev.",
        "Organization": "Org.",
        "Administration": "Admin.",
        "Professional": "Prof.",
        "Education": "Educ.",
        "Psychology": "Psychol.",
        "Sociology": "Sociol.",
        "Anthropology": "Anthropol.",
        "Geography": "Geogr.",
        "Medicine": "Med.",
        "Medical": "Med.",
        "Clinical": "Clin.",
        "Hospital": "Hosp.",
        "Pharmaceutical": "Pharm.",
        "Academy": "Acad.",
        "Foundation": "Found.",
        "Corporation": "Corp.",
        "Laboratory": "Lab.",
        "and": "&",
        "of the": "of",
    }
    
    result = name
    for full, abbr in abbreviations.items():
        # Case-insensitive replacement, preserving word boundaries
        result = re.sub(rf'\b{full}\b', abbr, result, flags=re.IGNORECASE)
    
    return result


def shorten_labels(
    labels: List[str],
    max_length: int = 40,
    mode: str = "smart",
    unique: bool = True,
) -> List[str]:
    """
    Shorten a list of labels.
    
    Parameters
    ----------
    labels : list of str
        Labels to shorten.
    max_length : int, default 40
        Maximum length per label.
    mode : str, default "smart"
        Shortening mode (see shorten_label).
    unique : bool, default True
        If True, ensure shortened labels are unique by adding suffixes.
        
    Returns
    -------
    list of str
        Shortened labels.
    """
    shortened = [shorten_label(str(l), max_length, mode) for l in labels]
    
    if unique:
        # Handle duplicates by adding numeric suffixes
        seen = {}
        result = []
        for i, s in enumerate(shortened):
            if s in seen:
                seen[s] += 1
                # Shorten more to make room for suffix
                suffix = f" ({seen[s]})"
                base = shorten_label(str(labels[i]), max_length - len(suffix), mode)
                result.append(base + suffix)
            else:
                seen[s] = 1
                result.append(s)
        return result
    
    return shortened


def shorten_df_labels(
    df: pd.DataFrame,
    axis: str = "both",
    max_length: int = 40,
    mode: str = "smart",
) -> pd.DataFrame:
    """
    Shorten labels in DataFrame index and/or columns for plotting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    axis : str, default "both"
        Which axis to shorten: "index", "columns", or "both".
    max_length : int, default 40
        Maximum label length.
    mode : str, default "smart"
        Shortening mode.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with shortened labels.
    """
    df = df.copy()
    
    if axis in ("index", "both"):
        df.index = shorten_labels(df.index.tolist(), max_length, mode)
    
    if axis in ("columns", "both"):
        df.columns = shorten_labels(df.columns.tolist(), max_length, mode)
    
    return df


# Convenience function for common label types
def shorten_references(labels: List[str], max_length: int = 50) -> List[str]:
    """Shorten reference labels, keeping author and year."""
    return shorten_labels(labels, max_length, mode="smart")


def shorten_affiliations(labels: List[str], max_length: int = 45) -> List[str]:
    """Shorten affiliation labels, keeping institution and country."""
    return shorten_labels(labels, max_length, mode="smart")


def shorten_journals(labels: List[str], max_length: int = 35) -> List[str]:
    """Shorten journal names using standard abbreviations."""
    return [_abbreviate_journal(str(l))[:max_length] for l in labels]

def infer_color_scheme(values):
    """Infer color scheme based on number of unique values and data type."""
    if values is None:
        return "default"
    values_series = pd.Series(values)
    if not pd.api.types.is_integer_dtype(values_series):
        return "continuous"
    unique_vals = values_series.nunique()
    return "categorical" if unique_vals <= 10 else "continuous"

def get_colors(n, color_scheme="default", values=None, cmap="viridis", default_color="lightblue", categorical_palette=None):
    """Return a list of colors based on the selected scheme."""
    if categorical_palette is None:
        categorical_palette = ["lightblue", "lightgreen", "lightcoral", "khaki", "lightgrey", "plum", "salmon", "skyblue", "palegreen", "gold"]

    if color_scheme == "default" or values is None:
        return [default_color] * n
    elif color_scheme == "categorical":
        palette = sns.color_palette(categorical_palette)
        return [palette[i % len(palette)] for i in range(n)]
    elif color_scheme == "continuous":
        norm = plt.Normalize(min(values), max(values))
        color_map = plt.cm.get_cmap(cmap)
        return [color_map(norm(val)) for val in values]
    else:
        raise ValueError(f"Unknown color_scheme: {color_scheme}")

def save_plot(filename_base, dpi=600):
    """
    Save current matplotlib figure to PNG, SVG, and PDF with tight layout.

    Parameters:
        filename_base (str): Path without file extension.
        dpi (int): Resolution of the saved figures.
    """
    for ext in ["png", "svg", "pdf"]:
        path = f"{filename_base}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Plot saved to {filename_base}.png (And svg, pdf)")

def plot_barh(
    df,
    x,
    y=None,
    *,
    label_col=None,
    color_scheme="default",
    color_by=None,
    filename="barh_plot",
    dpi=600,
    grid=False,
    wrap_width=50,
    max_label_length=0,
    label_mode="smart",
    cmap="viridis",
    default_color="lightblue",
    categorical_palette=None,
    label_fontsize=8,
    axis_labelsize=None,
    colorbar_labelsize=None,
    show=True,
    **_,
):
    """
    Create a horizontal bar plot from a DataFrame.

    Notes
    -----
    • Rows with 0 or NaN in column `x` are filtered out (not displayed).
    • Accepts "label_col" as an alias for `y`.
    • Ignores extra/unknown kwargs for robustness.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x : str
        Column name used for bar lengths (frequencies).
    y : str, optional
        Column for y-axis labels (uses label_col if None).
    label_col : str, optional
        Backward-compatible alias for y. Ignored if y is provided.
    color_scheme : {"default","categorical","continuous"}, default "default"
        How to color the bars. If "default" and "color_by" is provided, it is inferred.
    color_by : str, optional
        Column used to color bars (categorical or numeric).
    filename : str, default "barh_plot"
        Base filename used by save_plot(...).
    dpi : int, default 600
        Resolution for saving.
    grid : bool, default False
        Whether to show grid lines.
    wrap_width : int, default 50
        Max chars per line for wrapped y labels (0 disables).
    max_label_length : int, default 0
        Maximum label length before shortening (0 disables shortening).
        Useful for long labels like references or affiliations.
    label_mode : str, default "smart"
        Shortening mode: "smart", "truncate", "middle", "words".
    cmap : str, default "viridis"
        Matplotlib colormap for continuous coloring.
    default_color : str, default "lightblue"
        Single color when not categorizing/gradient coloring.
    categorical_palette : list[str] | dict, optional
        Colors for categorical values (order-mapped list or value->color dict).
    label_fontsize : int, default 8
        Font size for numeric value labels on bars.
    axis_labelsize : int, optional
        Font size for axis labels.
    colorbar_labelsize : int, optional
        Font size for the colorbar label.
    show : bool, default True
        Whether to display the figure (plt.show()).

    Returns
    -------
    None
    """
    import warnings
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if y is None and label_col is not None:
        y = label_col
    elif y is not None and label_col is not None and y != label_col:
        warnings.warn('Both "y" and "label_col" were provided; using "y".', RuntimeWarning)
    if y is None:
        raise ValueError('You must provide either "y" or "label_col".')

    # --- Filter out zero/NaN frequencies --------------------------------------
    df = df.copy()
    df = df[df[x].fillna(0) > 0]

    # --- Sort & prep -----------------------------------------------------------
    df = df.sort_values(by=x, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(df))))

    # Apply label shortening if requested
    if max_label_length > 0:
        y_labels = shorten_labels(df[y].astype(str).tolist(), max_label_length, label_mode)
    elif wrap_width > 0:
        y_labels = wrap_labels(df[y], wrap_width)
    else:
        y_labels = df[y].astype(str).tolist()

    values_for_colors = df[color_by] if color_by else None
    if color_by and color_scheme == "default":
        color_scheme = infer_color_scheme(values_for_colors)

    colors = get_colors(
        len(df),
        color_scheme,
        values=values_for_colors,
        cmap=cmap,
        default_color=default_color,
        categorical_palette=categorical_palette,
    )

    bars = ax.barh(y_labels, df[x], color=colors)

    for bar in bars:
        width = bar.get_width()
        label = f"{int(width)}" if float(width).is_integer() else f"{width:.2f}"
        ax.text(width, bar.get_y() + bar.get_height() / 2, label, va="center", ha="left", fontsize=label_fontsize)

    if axis_labelsize:
        ax.set_xlabel(x, fontsize=axis_labelsize)
        ax.set_ylabel(y, fontsize=axis_labelsize)
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    if color_scheme == "continuous" and values_for_colors is not None:
        vmin = float(values_for_colors.min())
        vmax = float(values_for_colors.max())
        if vmin != vmax:
            norm = plt.Normalize(vmin, vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            label_txt = color_by if color_by else ""
            if colorbar_labelsize:
                cbar.set_label(label_txt, fontsize=colorbar_labelsize)
            else:
                cbar.set_label(label_txt)
    elif color_scheme == "categorical" and values_for_colors is not None:
        cats = list(values_for_colors.astype("category").cat.categories)
        if hasattr(categorical_palette, "keys"):
            cat2color = {cat: categorical_palette.get(cat, default_color) for cat in cats}
        elif isinstance(categorical_palette, (list, tuple)) and len(categorical_palette) >= len(cats):
            cat2color = {cat: categorical_palette[i] for i, cat in enumerate(cats)}
        else:
            cat2color = {}
            for cat, col in zip(values_for_colors, colors):
                cat2color.setdefault(cat, col)
        handles = [mpatches.Patch(color=cat2color[c], label=str(c)) for c in cats]
        ax.legend(handles=handles, title=color_by or "Category", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.grid(grid)
    try:
        import seaborn as sns
        sns.despine()
    except Exception:
        pass

    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_lollipop(
    df,
    x,
    y=None,
    *,
    label_col=None,
    color_scheme="default",
    color_by=None,
    filename="lollipop_plot",
    dpi=600,
    grid=False,
    wrap_width=50,
    max_label_length=0,
    label_mode="smart",
    cmap="viridis",
    default_color="lightblue",
    categorical_palette=None,
    label_fontsize=8,
    axis_labelsize=None,
    colorbar_labelsize=None,
    show=True,
    **_,
):
    """
    Create a horizontal lollipop plot from a DataFrame.

    Notes
    -----
    • Rows with 0 or NaN in column `x` are filtered out (not displayed).
    • Accepts "label_col" as an alias for `y`.
    • Ignores extra/unknown kwargs for robustness.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x : str
        Column name used for the lollipop values (x-axis).
    y : str, optional
        Column name used for y-axis labels. If None, "label_col" is used.
    label_col : str, optional
        Backward-compatible alias for y. Ignored if y is provided.
    color_scheme : {"default","categorical","continuous"}, default "default"
        Coloring mode. If "default" and "color_by" is given, scheme is inferred.
    color_by : str, optional
        Column used to color items (categorical or numeric).
    filename : str, default "lollipop_plot"
        Base filename used by save_plot(...).
    dpi : int, default 600
        Resolution for saving.
    grid : bool, default False
        Whether to show grid lines.
    wrap_width : int, default 50
        Max chars per line for wrapped y labels (0 disables).
    max_label_length : int, default 0
        Maximum label length before shortening (0 disables shortening).
    label_mode : str, default "smart"
        Shortening mode: "smart", "truncate", "middle", "words".
    cmap : str, default "viridis"
        Matplotlib colormap for continuous coloring.
    default_color : str, default "lightblue"
        Single color when not categorizing/gradient coloring.
    categorical_palette : list[str] | dict, optional
        Colors for categories (ordered list or value->color dict).
    label_fontsize : int, default 8
        Font size for numeric value labels.
    axis_labelsize : int, optional
        Font size for axis labels.
    colorbar_labelsize : int, optional
        Font size for the colorbar label.
    show : bool, default True
        Whether to display the figure (plt.show()).

    Returns
    -------
    None
    """
    import warnings
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if y is None and label_col is not None:
        y = label_col
    elif y is not None and label_col is not None and y != label_col:
        warnings.warn('Both "y" and "label_col" were provided; using "y".', RuntimeWarning)
    if y is None:
        raise ValueError('You must provide either "y" or "label_col".')

    # --- Filter out zero/NaN frequencies --------------------------------------
    df = df.copy()
    df = df[df[x].fillna(0) > 0]

    # --- Sort & prep -----------------------------------------------------------
    df = df.sort_values(by=x, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(df))))

    # Apply label shortening if requested
    if max_label_length > 0:
        y_labels = shorten_labels(df[y].astype(str).tolist(), max_label_length, label_mode)
    elif wrap_width > 0:
        y_labels = wrap_labels(df[y], wrap_width)
    else:
        y_labels = df[y].astype(str).tolist()
    idx = range(len(df))

    values_for_colors = df[color_by] if color_by else None
    if color_by and color_scheme == "default":
        color_scheme = infer_color_scheme(values_for_colors)

    colors = get_colors(
        len(df),
        color_scheme,
        values=values_for_colors,
        cmap=cmap,
        default_color=default_color,
        categorical_palette=categorical_palette,
    )

    ax.hlines(y=list(idx), xmin=0, xmax=df[x].to_numpy(), color=colors, linewidth=1.5)

    max_val = float(df[x].max()) if len(df) else 0.0
    sizes = (300 * (df[x] / max_val)).to_numpy() if max_val > 0 else [50] * len(df)

    ax.scatter(df[x], list(idx), color=colors, s=sizes, zorder=3)

    for i, val in enumerate(df[x]):
        label = f"{int(val)}" if float(val).is_integer() else f"{val:.2f}"
        ax.text(float(val), i, label, va="center", ha="left", fontsize=label_fontsize)

    ax.set_yticks(list(idx))
    ax.set_yticklabels(y_labels)

    if axis_labelsize:
        ax.set_xlabel(x, fontsize=axis_labelsize)
        ax.set_ylabel(y, fontsize=axis_labelsize)
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    if color_scheme == "continuous" and values_for_colors is not None:
        vmin = float(values_for_colors.min())
        vmax = float(values_for_colors.max())
        if vmin != vmax:
            norm = plt.Normalize(vmin, vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            label_txt = color_by if color_by else ""
            if colorbar_labelsize:
                cbar.set_label(label_txt, fontsize=colorbar_labelsize)
            else:
                cbar.set_label(label_txt)
    elif color_scheme == "categorical" and values_for_colors is not None:
        cats = list(values_for_colors.astype("category").cat.categories)
        if hasattr(categorical_palette, "keys"):
            cat2color = {cat: categorical_palette.get(cat, default_color) for cat in cats}
        elif isinstance(categorical_palette, (list, tuple)) and len(categorical_palette) >= len(cats):
            cat2color = {cat: categorical_palette[i] for i, cat in enumerate(cats)}
        else:
            cat2color = {}
            for cat, col in zip(values_for_colors, colors):
                cat2color.setdefault(cat, col)
        handles = [mpatches.Patch(color=cat2color[c], label=str(c)) for c in cats]
        ax.legend(handles=handles, title=color_by or "Category", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.grid(grid)
    try:
        import seaborn as sns
        sns.despine()
    except Exception:
        pass

    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_timeseries(
    df,
    x="Year",
    bar_y="Number of Documents",
    line_y="Cumulative Citations",
    cut_year=None,
    filename="timeseries_plot",
    dpi=600,
    axis_labelsize=None,
    show=True,
    bar_color=None,
    line_color=None,
    xrotation=90,
    bar_labels=False,
):
    """
    Plot a time series combining bar and line plots for documents and citations.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the time column `x` and columns specified by `bar_y` and `line_y`.
    x : str, default "Year"
        Time axis column.
    bar_y : str or None, default "Number of Documents"
        Column to plot as bars. Use None to disable bars.
    line_y : str or None, default "Cumulative Citations"
        Column to plot as a line. Use None to disable line.
    cut_year : int or None, default None
        If provided, groups all rows with x < cut_year into a single category.
    filename : str, default "timeseries_plot"
        Base filename for saving.
    dpi : int, default 600
        Resolution for saved images.
    axis_labelsize : int or None
        Font size for axis labels.
    show : bool, default True
        Whether to display the plot with plt.show().
    bar_color : str or None
        Color for the bar plot. Defaults to light blue.
    line_color : str or None
        Color for the line plot. Defaults to black.
    xrotation : int, default 90
        Rotation angle for x-axis labels.
    bar_labels : bool, default False
        If True, displays values above the bars.
    """
    df_copy = df.copy()

    # ---- Handle cut_year grouping
    if cut_year is not None:
        before_df = df_copy[df_copy[x] < cut_year].copy()
        after_df = df_copy[df_copy[x] >= cut_year].copy().sort_values(by=x)

        if not before_df.empty:
            combined = {x: f"before {cut_year}"}
            for col in df.columns:
                if col == x:
                    continue
                if "Cumulative" in col:
                    combined[col] = before_df[col].max(skipna=True)
                else:
                    if pd.api.types.is_numeric_dtype(before_df[col]):
                        combined[col] = before_df[col].sum(skipna=True)
                    else:
                        combined[col] = pd.NA
            before_df = pd.DataFrame([combined])
            df_plot = pd.concat([before_df, after_df], ignore_index=True)
        else:
            df_plot = after_df
    else:
        df_plot = df_copy.sort_values(by=x)

    # ---- Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = None

    # Case 1: both bar and line
    if bar_y is not None and line_y is not None:
        bar_color = bar_color or "lightblue"
        bars = ax1.bar(df_plot[x].astype(str), df_plot[bar_y], color=bar_color, label=bar_y)
        if bar_labels:
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{int(round(height))}" if float(height).is_integer() else f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax1.set_ylabel(bar_y, fontsize=axis_labelsize)
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

        line_color = line_color or "black"
        ax2 = ax1.twinx()
        ax2.plot(df_plot[x].astype(str), df_plot[line_y], color=line_color, marker="o", linewidth=2, label=line_y)
        ax2.set_ylabel(line_y, fontsize=axis_labelsize)
        ax2.ticklabel_format(style="plain", axis="y")

    # Case 2: only bars
    elif bar_y is not None:
        bar_color = bar_color or "lightblue"
        bars = ax1.bar(df_plot[x].astype(str), df_plot[bar_y], color=bar_color, label=bar_y)
        if bar_labels:
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{int(round(height))}" if float(height).is_integer() else f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax1.set_ylabel(bar_y, fontsize=axis_labelsize)
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Case 3: only line
    elif line_y is not None:
        line_color = line_color or "black"
        ax1.plot(df_plot[x].astype(str), df_plot[line_y], color=line_color, marker="o", linewidth=2, label=line_y)
        ax1.set_ylabel(line_y, fontsize=axis_labelsize)
        ax1.ticklabel_format(style="plain", axis="y")

    ax1.set_xlabel(x, fontsize=axis_labelsize)
    ax1.set_xticks(range(len(df_plot)))
    ax1.set_xticklabels(df_plot[x].astype(str), rotation=xrotation, ha="right")

    ax1.grid(False)
    if ax2 is not None:
        ax2.grid(False)

    sns.despine()
    plt.tight_layout()
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_growth_model(
    result: dict,
    filename: str = "growth_model",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (12, 5),
    colors: dict = None,
    title: str = None,
    show_residuals: bool = False,
):
    """
    Plot growth model results with observed data, fitted curve, and forecast.
    
    Parameters
    ----------
    result : dict
        Result dictionary from fit_growth_model().
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size (width, height).
    colors : dict
        Custom colors: {"observed": ..., "fitted": ..., "forecast": ...}
    title : str
        Custom title. If None, auto-generated.
    show_residuals : bool
        If True, adds a third subplot showing residuals analysis.
    """
    pred_df = result['prediction_df']
    
    # Default colors
    if colors is None:
        colors = {
            "observed": "#2196F3",  # blue
            "fitted": "#4CAF50",    # green
            "forecast": "#FF9800",  # orange
            "residuals": "#9C27B0", # purple
        }
    
    # Determine layout based on residuals option
    if show_residuals:
        fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[1] * 1.6))
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # --- Left plot: Main growth chart ---
    hist_data = pred_df[~pred_df['Is Forecast']]
    forecast_data = pred_df[pred_df['Is Forecast']]
    
    # Observed as bars
    ax1.bar(
        hist_data['Year'], hist_data['Observed'],
        color=colors['observed'], alpha=0.7, label='Observed', width=0.8
    )
    
    # Fitted line
    ax1.plot(
        hist_data['Year'], hist_data['Fitted'],
        color=colors['fitted'], linewidth=2.5, label='Fitted', marker='', linestyle='-'
    )
    
    # Forecast
    if len(forecast_data) > 0:
        ax1.plot(
            forecast_data['Year'], forecast_data['Fitted'],
            color=colors['forecast'], linewidth=2.5, linestyle='--', 
            marker='o', markersize=5, label='Forecast'
        )
        
        # Shade forecast area
        ax1.fill_between(
            forecast_data['Year'], 0, forecast_data['Fitted'],
            color=colors['forecast'], alpha=0.2
        )
    
    # Styling
    model_name = result['model_type'].title()
    r2 = result['r_squared']
    if title is None:
        title = f'{model_name} Growth Model (R² = {r2:.3f})'
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=10)
    ax1.set_ylabel('Publications', fontsize=10)
    ax1.legend(loc='upper left', frameon=False)
    
    # Remove gridlines
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Rotate x-labels if many years
    if len(pred_df) > 15:
        ax1.tick_params(axis='x', rotation=45)
    
    # --- Right plot: Model comparison ---
    comp_df = result['comparison_df'].copy()
    
    # Highlight selected model
    bar_colors = [colors['fitted'] if m == result['model_type'] else '#BDBDBD' 
                  for m in comp_df['Model']]
    
    bars = ax2.barh(comp_df['Model'], comp_df['R²'], color=bar_colors, edgecolor='white')
    
    # Add R² labels
    for bar, r2_val in zip(bars, comp_df['R²']):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{r2_val:.3f}', va='center', fontsize=9)
    
    ax2.set_xlabel('R² (Coefficient of Determination)', fontsize=10)
    ax2.set_title('Model Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1.15)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # --- Residuals plots (if enabled) ---
    if show_residuals:
        residuals = result.get('residuals')
        years = result.get('years')
        
        if residuals is not None and years is not None:
            from matplotlib.patches import Patch
            
            # Bottom left: Residuals over time - colored by sign
            res_colors = ['#26A69A' if r >= 0 else '#EF5350' for r in residuals]
            ax3.bar(years, residuals, color=res_colors, alpha=0.8, width=0.8, 
                   edgecolor='white', linewidth=0.5)
            ax3.axhline(y=0, color='#424242', linestyle='-', linewidth=1.2)
            ax3.set_title('Residuals Over Time', fontsize=13, fontweight='bold', pad=12)
            ax3.set_xlabel('Year', fontsize=10)
            ax3.set_ylabel('Residual (Observed - Fitted)', fontsize=10)
            ax3.grid(False)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            legend_elements = [
                Patch(facecolor='#26A69A', alpha=0.8, label='Positive'),
                Patch(facecolor='#EF5350', alpha=0.8, label='Negative')
            ]
            ax3.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
            if len(years) > 15:
                ax3.tick_params(axis='x', rotation=45)
            
            # Bottom right: Residuals histogram with gradient
            n, bins, patches = ax4.hist(residuals, bins=min(20, len(residuals)//2 + 1), 
                                        alpha=0.85, edgecolor='white', linewidth=0.8)
            # Color gradient
            cm = plt.cm.get_cmap('RdYlGn_r')
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min() + 0.001)
            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm(c))
            
            ax4.axvline(x=0, color='#424242', linestyle='--', linewidth=1.5, label='Zero')
            ax4.set_title('Residuals Distribution', fontsize=13, fontweight='bold', pad=12)
            ax4.set_xlabel('Residual Value', fontsize=10)
            ax4.set_ylabel('Frequency', fontsize=10)
            ax4.grid(False)
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.legend(loc='upper right', frameon=False, fontsize=8)
            
            # Add stats annotation (without mean since it's always ~0)
            std_res = np.std(residuals)
            min_res = np.min(residuals)
            max_res = np.max(residuals)
            ax4.annotate(f'Std: {std_res:.2f}\nMin: {min_res:.2f}\nMax: {max_res:.2f}', 
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        ha='left', va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', 
                                 alpha=0.9, edgecolor='#1565C0'))
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_life_cycle(
    result: dict,
    filename: str = "life_cycle",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (14, 5),
    colors: dict = None,
    title: str = None,
):
    """
    Plot life cycle analysis results showing cumulative growth and phase indicator.
    
    Parameters
    ----------
    result : dict
        Result dictionary from fit_life_cycle_model().
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size.
    colors : dict
        Custom colors for phases.
    title : str
        Custom title.
    """
    pred_df = result['prediction_df']
    
    # Phase colors
    phase_colors = {
        "emerging": "#4CAF50",   # green
        "growth": "#2196F3",     # blue  
        "maturity": "#FF9800",   # orange
        "saturation": "#F44336", # red
    }
    
    if colors is None:
        colors = {
            "observed": "#2196F3",
            "fitted": "#4CAF50",
            "forecast": "#FF9800",
            "saturation": "#E91E63",
        }
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax1, ax2, ax3 = axes
    
    # --- Left: Cumulative growth curve ---
    hist_data = pred_df[~pred_df['Is Forecast']]
    forecast_data = pred_df[pred_df['Is Forecast']]
    
    # Observed cumulative
    ax1.scatter(
        hist_data['Year'], hist_data['Observed Cumulative'],
        color=colors['observed'], s=40, alpha=0.7, label='Observed', zorder=3
    )
    
    # Fitted S-curve
    all_years = pred_df['Year'].values
    all_fitted = pred_df['Fitted Cumulative'].values
    ax1.plot(all_years, all_fitted, color=colors['fitted'], linewidth=2.5, label='Logistic Fit')
    
    # Saturation line
    K = result['saturation_k']
    ax1.axhline(y=K, color=colors['saturation'], linestyle=':', linewidth=2, label=f'Saturation (K={K:,.0f})')
    
    # Mark inflection point
    Tm = result['peak_year']
    if hist_data['Year'].min() <= Tm <= forecast_data['Year'].max() if len(forecast_data) > 0 else hist_data['Year'].max():
        ax1.axvline(x=Tm, color='gray', linestyle='--', alpha=0.5)
        ax1.annotate(f'Peak Year\n({Tm:.0f})', xy=(Tm, K/2), fontsize=8, ha='center')
    
    # Shade forecast
    if len(forecast_data) > 0:
        ax1.fill_between(
            forecast_data['Year'], 0, forecast_data['Fitted Cumulative'],
            color=colors['forecast'], alpha=0.2
        )
    
    ax1.set_title('Cumulative Growth (S-Curve)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=10)
    ax1.set_ylabel('Cumulative Publications', fontsize=10)
    ax1.legend(loc='upper left', frameon=False, fontsize=9)
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Middle: Annual publications ---
    ax2.bar(
        hist_data['Year'], hist_data['Observed'],
        color=colors['observed'], alpha=0.7, label='Observed', width=0.8
    )
    ax2.plot(
        pred_df['Year'], pred_df['Fitted Annual'],
        color=colors['fitted'], linewidth=2, label='Fitted'
    )
    
    ax2.set_title('Annual Publications', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('Publications per Year', fontsize=10)
    ax2.legend(loc='upper left', frameon=False, fontsize=9)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    if len(pred_df) > 15:
        ax2.tick_params(axis='x', rotation=45)
    
    # --- Right: Phase indicator ---
    phase = result['current_phase']
    progress = result['progress']
    
    # Create gauge-like visualization
    phases = ['emerging', 'growth', 'maturity', 'saturation']
    phase_idx = phases.index(phase)
    
    # Stacked bar for phases
    widths = [0.1, 0.4, 0.4, 0.1]  # 10%, 40%, 40%, 10%
    starts = [0, 0.1, 0.5, 0.9]
    
    for i, (p, w, s) in enumerate(zip(phases, widths, starts)):
        color = phase_colors[p]
        alpha = 1.0 if i == phase_idx else 0.3
        ax3.barh(['Progress'], [w], left=[s], color=color, alpha=alpha, height=0.4)
        # Label
        ax3.text(s + w/2, 0, p.title(), ha='center', va='center', fontsize=8,
                color='white' if i == phase_idx else 'gray', fontweight='bold')
    
    # Progress marker
    ax3.axvline(x=progress, color='black', linewidth=3, ymin=0.25, ymax=0.75)
    ax3.scatter([progress], [0], marker='v', s=200, color='black', zorder=5)
    
    # Info text
    info_text = (
        f"Phase: {phase.upper()}\n"
        f"Progress: {progress*100:.1f}%\n"
        f"R²: {result['r_squared']:.3f}\n\n"
        f"Saturation: {K:,.0f}\n"
        f"Peak Year: {Tm:.0f}\n"
        f"Growth Duration: {result['growth_duration']:.1f} yr"
    )
    ax3.text(0.5, -0.8, info_text, ha='center', va='top', fontsize=9,
            transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_title('Life Cycle Phase', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Progress to Saturation', fontsize=10)
    ax3.set_yticks([])
    ax3.grid(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_residuals_time(
    result: dict,
    filename: str = "residuals_time",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (11, 5),
    title: str = None,
):
    """
    Plot residuals over time from growth model results.
    
    Parameters
    ----------
    result : dict
        Result dictionary from fit_growth_model() containing 'residuals' and 'years'.
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size (width, height).
    title : str
        Custom title. If None, uses default.
    """
    from matplotlib.patches import Patch
    
    residuals = result.get('residuals')
    years = result.get('years')
    
    if residuals is None or years is None:
        print("No residuals data available in result.")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color bars based on positive/negative
    colors = ['#26A69A' if r >= 0 else '#EF5350' for r in residuals]
    
    ax.bar(years, residuals, color=colors, alpha=0.8, width=0.8, edgecolor='white', linewidth=0.5)
    ax.axhline(y=0, color='#424242', linestyle='-', linewidth=1.2)
    
    if title is None:
        title = 'Residuals Over Time'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Residual (Observed - Fitted)', fontsize=10)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#26A69A', alpha=0.8, label='Positive'),
        Patch(facecolor='#EF5350', alpha=0.8, label='Negative')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    
    if len(years) > 20:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_residuals_distribution(
    result: dict,
    filename: str = "residuals_distribution",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (11, 5),
    title: str = None,
):
    """
    Plot residuals distribution histogram from growth model results.
    
    Parameters
    ----------
    result : dict
        Result dictionary from fit_growth_model() containing 'residuals'.
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size (width, height).
    title : str
        Custom title. If None, uses default.
    """
    residuals = result.get('residuals')
    
    if residuals is None:
        print("No residuals data available in result.")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram with gradient color
    n, bins, patches = ax.hist(residuals, bins=min(20, len(residuals)//2 + 1), 
                               alpha=0.85, edgecolor='white', linewidth=0.8)
    
    # Color gradient based on position
    cm = plt.cm.get_cmap('RdYlGn_r')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min() + 0.001)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    ax.axvline(x=0, color='#424242', linestyle='--', linewidth=1.5, label='Zero')
    
    if title is None:
        title = 'Residuals Distribution'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Residual Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    
    # Add stats box (without mean since it's always ~0)
    std_res = np.std(residuals)
    min_res = np.min(residuals)
    max_res = np.max(residuals)
    stats_text = f'Std: {std_res:.2f}\nMin: {min_res:.2f}\nMax: {max_res:.2f}'
    ax.annotate(stats_text, 
                xy=(0.02, 0.98), xycoords='axes fraction',
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', 
                         alpha=0.9, edgecolor='#1565C0'))
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_citation_distribution(
    result: dict,
    filename: str = "citation_distribution",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (14, 10),
    title: str = None,
):
    """
    Plot comprehensive citation distribution analysis.
    
    Creates a 2x2 panel visualization showing histogram, log-scale histogram,
    citation classes, and key metrics.
    
    Parameters
    ----------
    result : dict
        Result dictionary from analyze_citation_distribution().
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size (width, height).
    title : str
        Overall figure title.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    
    citations = result['citations']
    
    # --- Top Left: Citation Histogram ---
    # Clip to 99th percentile for better visualization
    p99 = result['percentiles'][99]
    clipped = citations[citations <= p99]
    
    ax1.hist(clipped, bins=50, color='#2196F3', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax1.axvline(x=result['basic_stats']['mean'], color='#E91E63', linestyle='--', 
                linewidth=2, label=f"Mean ({result['basic_stats']['mean']:.1f})")
    ax1.axvline(x=result['basic_stats']['median'], color='#4CAF50', linestyle='-', 
                linewidth=2, label=f"Median ({result['basic_stats']['median']:.0f})")
    
    ax1.set_xlabel('Citations', fontsize=10)
    ax1.set_ylabel('Number of Papers', fontsize=10)
    ax1.set_title('Citation Distribution (clipped at 99th percentile)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', frameon=False, fontsize=8)
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Top Right: Log-scale Histogram ---
    positive_cit = citations[citations > 0]
    if len(positive_cit) > 0:
        log_bins = np.logspace(0, np.log10(max(positive_cit) + 1), 40)
        ax2.hist(positive_cit, bins=log_bins, color='#2196F3', alpha=0.8, 
                edgecolor='white', linewidth=0.5)
        ax2.set_xscale('log')
        ax2.set_xlabel('Citations (log scale)', fontsize=10)
        ax2.set_ylabel('Number of Papers', fontsize=10)
        ax2.set_title('Citation Distribution (Log Scale, excluding zeros)', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No cited papers', ha='center', va='center', fontsize=12)
        ax2.set_title('Citation Distribution (Log Scale)', fontsize=12, fontweight='bold')
    
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # --- Bottom Left: Citation Classes ---
    class_df = result['citation_classes']
    
    # Single color for all bars
    bars = ax3.barh(class_df['Class'], class_df['Count'], color='#2196F3', 
                   alpha=0.8, edgecolor='white', height=0.6)
    
    # Add percentage labels
    for bar, pct in zip(bars, class_df['Percentage']):
        width = bar.get_width()
        ax3.text(width + max(class_df['Count']) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)
    
    ax3.set_xlabel('Number of Papers', fontsize=10)
    ax3.set_title('Citation Classes', fontsize=12, fontweight='bold')
    ax3.grid(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # --- Bottom Right: Key Metrics ---
    ax4.axis('off')
    
    metrics_text = [
        f"📊 Key Metrics",
        f"",
        f"Papers: {result['n_papers']:,}",
        f"Total Citations: {result['basic_stats']['sum']:,}",
        f"",
        f"Mean: {result['basic_stats']['mean']:.2f}",
        f"Median: {result['basic_stats']['median']:.1f}",
        f"Std Dev: {result['basic_stats']['std']:.2f}",
        f"Max: {result['basic_stats']['max']:,}",
        f"",
        f"📈 Impact Indices",
        f"H-index: {result['h_index']}",
        f"G-index: {result['g_index']}",
        f"Gini: {result['gini_coefficient']:.3f}",
        f"",
        f"📉 Distribution Shape",
        f"Skewness: {result['skewness']:.2f}",
        f"Kurtosis: {result['kurtosis']:.2f}",
        f"",
        f"📋 Percentiles",
    ]
    for p, v in result['percentiles'].items():
        metrics_text.append(f"  {p}th: {v:.0f}")
    
    ax4.text(0.1, 0.95, '\n'.join(metrics_text), transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', alpha=0.8))
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_citation_histogram(
    result: dict,
    filename: str = "citation_histogram",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
    log_scale: bool = False,
    title: str = None,
):
    """
    Plot citation histogram.
    
    Parameters
    ----------
    result : dict
        Result dictionary from analyze_citation_distribution().
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size.
    log_scale : bool
        Use log scale for x-axis.
    title : str
        Custom title.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    citations = result['citations']
    
    if log_scale:
        positive_cit = citations[citations > 0]
        if len(positive_cit) > 0:
            log_bins = np.logspace(0, np.log10(max(positive_cit) + 1), 40)
            ax.hist(positive_cit, bins=log_bins, color='#2196F3', alpha=0.8, 
                   edgecolor='white', linewidth=0.5)
            ax.set_xscale('log')
            xlabel = 'Citations (log scale)'
            default_title = 'Citation Distribution (Log Scale)'
        else:
            ax.text(0.5, 0.5, 'No cited papers', ha='center', va='center')
            xlabel = 'Citations'
            default_title = 'Citation Distribution'
    else:
        p99 = result['percentiles'][99]
        clipped = citations[citations <= p99]
        ax.hist(clipped, bins=50, color='#2196F3', alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.axvline(x=result['basic_stats']['mean'], color='#E91E63', linestyle='--', 
                  linewidth=2, label=f"Mean ({result['basic_stats']['mean']:.1f})")
        ax.axvline(x=result['basic_stats']['median'], color='#4CAF50', linestyle='-', 
                  linewidth=2, label=f"Median ({result['basic_stats']['median']:.0f})")
        ax.legend(loc='upper right', frameon=False, fontsize=9)
        xlabel = 'Citations'
        default_title = 'Citation Distribution'
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Number of Papers', fontsize=11)
    ax.set_title(title or default_title, fontsize=13, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_citation_classes(
    result: dict,
    filename: str = "citation_classes",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
    title: str = None,
):
    """
    Plot citation class distribution as horizontal bar chart.
    
    Parameters
    ----------
    result : dict
        Result dictionary from analyze_citation_distribution().
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size.
    title : str
        Custom title.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    class_df = result['citation_classes']
    
    # Single color for all bars
    bars = ax.barh(class_df['Class'], class_df['Count'], color='#2196F3', 
                   alpha=0.8, edgecolor='white', height=0.6)
    
    # Add count and percentage labels
    for bar, (count, pct) in zip(bars, zip(class_df['Count'], class_df['Percentage'])):
        width = bar.get_width()
        ax.text(width + max(class_df['Count']) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{count:,} ({pct:.1f}%)', va='center', fontsize=10)
    
    ax.set_xlabel('Number of Papers', fontsize=11)
    ax.set_title(title or 'Citation Classes', fontsize=13, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Extend x limit to fit labels
    ax.set_xlim(0, max(class_df['Count']) * 1.35)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)  # More space for y-axis labels
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_collaboration(
    result: dict,
    filename: str = "collaboration",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (14, 10),
    title: str = None,
):
    """
    Plot comprehensive collaboration analysis.
    
    Creates a 2x2 panel visualization showing author distribution,
    collaboration metrics, temporal trend, and team sizes.
    
    Parameters
    ----------
    result : dict
        Result dictionary from analyze_collaboration().
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size (width, height).
    title : str
        Overall figure title.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    
    # --- Top Left: Author Count Distribution ---
    dist = result['author_distribution']
    # Limit to first 15 for readability, but ensure we have data
    dist_plot = dist[dist['n_authors'] <= 15].copy()
    
    if len(dist_plot) > 0:
        bars = ax1.bar(dist_plot['n_authors'].astype(str), dist_plot['n_papers'], 
                      color='#2196F3', alpha=0.8, edgecolor='white')
        
        # Add percentage labels on bars with significant values
        for bar, pct in zip(bars, dist_plot['percentage']):
            if pct >= 5:  # Only label if >= 5%
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{pct:.0f}%', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Number of Authors', fontsize=10)
        ax1.set_ylabel('Number of Papers', fontsize=10)
    else:
        ax1.text(0.5, 0.5, 'No distribution data', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=11)
    
    ax1.set_title('Author Count Distribution', fontsize=12, fontweight='bold')
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Top Right: Collaboration Metrics ---
    metrics = ['CI\n(Mean Authors)', 'DC\n(Multi-Author %)', 'CC\n(Coefficient)']
    # Convert DC to percentage for better visualization
    values = [result['collaboration_index'], 
              result['degree_of_collaboration'] * 100,  # Show as percentage
              result['collaboration_coefficient'] * 100]  # Show as percentage
    colors = ['#4CAF50', '#FF9800', '#9C27B0']
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8, edgecolor='white')
    ax2.set_ylabel('Value', fontsize=10)
    ax2.set_title('Collaboration Metrics', fontsize=12, fontweight='bold')
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, val, orig in zip(bars, values, [result['collaboration_index'], 
                                              result['degree_of_collaboration'],
                                              result['collaboration_coefficient']]):
        if bar.get_height() > 0:
            # Show original value for CI, percentage for others
            if orig == result['collaboration_index']:
                label = f'{orig:.2f}'
            else:
                label = f'{orig*100:.1f}%'
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # --- Bottom Left: Temporal Trend ---
    trend = result['temporal_trend']
    if len(trend) > 1:  # Need at least 2 points for a meaningful trend
        ax3.plot(trend['year'], trend['mean_authors'], marker='o', linewidth=2, 
                color='#2196F3', label='Mean Authors', markersize=5)
        ax3.fill_between(trend['year'], trend['mean_authors'], alpha=0.2, color='#2196F3')
        
        # Set reasonable y-axis limits
        y_min = max(0, trend['mean_authors'].min() - 0.5)
        y_max = trend['mean_authors'].max() + 0.5
        ax3.set_ylim(y_min, y_max)
        
        ax3.set_xlabel('Year', fontsize=10)
        ax3.set_ylabel('Mean Authors per Paper', fontsize=10)
        ax3.legend(loc='best', frameon=False, fontsize=8)
    elif len(trend) == 1:
        # Single year - show as a bar
        ax3.bar(str(trend['year'].iloc[0]), trend['mean_authors'].iloc[0], 
               color='#2196F3', alpha=0.8, width=0.5)
        ax3.set_ylabel('Mean Authors per Paper', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'Insufficient temporal data\n(need year column)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=11)
    
    ax3.set_title('Collaboration Trend Over Time', fontsize=12, fontweight='bold')
    ax3.grid(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # --- Bottom Right: Collaboration Types (Horizontal Bar Chart) ---
    types_df = result['collaboration_types']
    # Filter out zero values
    types_nonzero = types_df[types_df['Count'] > 0]
    
    if len(types_nonzero) > 0:
        bars = ax4.barh(types_nonzero['Type'], types_nonzero['Count'], 
                       color='#2196F3', alpha=0.8, edgecolor='white', height=0.6)
        max_count = types_nonzero['Count'].max()
        for bar, (count, pct) in zip(bars, zip(types_nonzero['Count'], types_nonzero['Percentage'])):
            ax4.text(bar.get_width() + max_count * 0.02, bar.get_y() + bar.get_height()/2,
                    f'{count:,} ({pct:.1f}%)', va='center', fontsize=9)
        ax4.set_xlim(0, max_count * 1.35)
        ax4.set_xlabel('Number of Papers', fontsize=10)
        ax4.set_title('Collaboration Types', fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No collaboration type data', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=11)
        ax4.set_title('Collaboration Types', fontsize=12, fontweight='bold')
    
    ax4.grid(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_author_distribution(
    result: dict,
    filename: str = "author_distribution",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
    max_authors: int = 20,
    title: str = None,
):
    """
    Plot author count distribution.
    
    Parameters
    ----------
    result : dict
        Result dictionary from analyze_collaboration().
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size.
    max_authors : int
        Maximum number of authors to show on x-axis.
    title : str
        Custom title.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    dist = result['author_distribution']
    dist_plot = dist[dist['n_authors'] <= max_authors].copy()
    
    if len(dist_plot) == 0:
        ax.text(0.5, 0.5, 'No distribution data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title or 'Author Count Distribution', fontsize=13, fontweight='bold')
    else:
        # Use string labels for x-axis to ensure proper spacing
        x_labels = dist_plot['n_authors'].astype(str).tolist()
        bars = ax.bar(x_labels, dist_plot['n_papers'], color='#2196F3', 
                     alpha=0.8, edgecolor='white')
        
        # Add percentage labels on top of bars
        max_height = dist_plot['n_papers'].max()
        for bar, pct in zip(bars, dist_plot['percentage']):
            if pct >= 3:  # Only show if >= 3%
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_height * 0.02,
                       f"{pct:.1f}%", ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Number of Authors', fontsize=11)
        ax.set_ylabel('Number of Papers', fontsize=11)
        ax.set_title(title or 'Author Count Distribution', fontsize=13, fontweight='bold')
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_collaboration_trend(
    result: dict,
    filename: str = "collaboration_trend",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
    title: str = None,
):
    """
    Plot collaboration trend over time.
    
    Parameters
    ----------
    result : dict
        Result dictionary from analyze_collaboration().
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size.
    title : str
        Custom title.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    trend = result['temporal_trend']
    
    if len(trend) < 2:
        ax.text(0.5, 0.5, 'Insufficient temporal data\n(need at least 2 years)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title or 'Collaboration Trend Over Time', fontsize=13, fontweight='bold')
    else:
        # Main line: Mean authors
        line1 = ax.plot(trend['year'], trend['mean_authors'], marker='o', linewidth=2, 
                       color='#2196F3', label='Mean Authors', markersize=6)
        ax.fill_between(trend['year'], trend['mean_authors'], alpha=0.2, color='#2196F3')
        
        # Set reasonable y-axis limits for mean authors
        y_min = max(0, trend['mean_authors'].min() * 0.9)
        y_max = trend['mean_authors'].max() * 1.1
        ax.set_ylim(y_min, y_max)
        
        # Add secondary axis for multi-author percentage if there's variation
        if trend['multi_author_pct'].std() > 0.1:  # Only if there's meaningful variation
            ax2 = ax.twinx()
            line2 = ax2.plot(trend['year'], trend['multi_author_pct'], marker='s', linewidth=2,
                            color='#E91E63', label='Multi-Author %', linestyle='--', markersize=5)
            ax2.set_ylabel('Multi-Author Papers (%)', fontsize=10, color='#E91E63')
            ax2.tick_params(axis='y', labelcolor='#E91E63')
            ax2.spines['top'].set_visible(False)
            
            # Set reasonable limits for percentage axis
            pct_min = max(0, trend['multi_author_pct'].min() - 5)
            pct_max = min(100, trend['multi_author_pct'].max() + 5)
            ax2.set_ylim(pct_min, pct_max)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='best', frameon=False, fontsize=9)
        else:
            ax.legend(loc='best', frameon=False, fontsize=9)
        
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Mean Authors per Paper', fontsize=11)
        ax.set_title(title or 'Collaboration Trend Over Time', fontsize=13, fontweight='bold')
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_collaboration_types(
    result: dict,
    filename: str = "collaboration_types",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
    title: str = None,
):
    """
    Plot collaboration types (team sizes) distribution.
    
    Parameters
    ----------
    result : dict
        Result dictionary from analyze_collaboration().
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size.
    title : str
        Custom title.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    types_df = result['collaboration_types']
    
    # Filter to only show types with data
    types_with_data = types_df[types_df['Count'] > 0]
    
    if len(types_with_data) == 0:
        ax.text(0.5, 0.5, 'No collaboration type data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        bars = ax.barh(types_with_data['Type'], types_with_data['Count'], color='#2196F3',
                      alpha=0.8, edgecolor='white', height=0.6)
        
        # Add count and percentage labels
        max_count = types_with_data['Count'].max()
        for bar, (count, pct) in zip(bars, zip(types_with_data['Count'], types_with_data['Percentage'])):
            width = bar.get_width()
            ax.text(width + max_count * 0.02, bar.get_y() + bar.get_height()/2,
                   f'{count:,} ({pct:.1f}%)', va='center', fontsize=10)
        
        ax.set_xlim(0, max_count * 1.3)
    
    ax.set_xlabel('Number of Papers', fontsize=11)
    ax.set_title(title or 'Collaboration Types (Team Sizes)', fontsize=13, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if len(types_with_data) > 0:
        plt.subplots_adjust(left=0.22)  # Space for labels
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_heatmap(df, filename="heatmap", dpi=600, show=True, normalized=False, cmap="viridis",
                 axis_labelsize=None, cbar_label=None, wrap_width=50, square_cells=None,
                 symmetric_option=None, label_fontsize=10, tick_labelsize=10, cbar_labelsize=10,
                 xlabel=None, ylabel=None):
    """
    Plot a heatmap showing the relationship between two categories, with optional colorbar label. Optionally enforce square aspect ratio.

    Parameters:
    - df: pandas.DataFrame with counts or normalized values
    - filename: base filename for saving (default: 'heatmap')
    - dpi: resolution of saved images (default: 600)
    - show: whether to display the plot
    - normalized: if True, format numbers with 2 decimals; if False, use integers
    - cmap: color map to use (default: 'viridis')
    - axis_labelsize: font size for axis labels (optional)
    - cbar_label: label for the colorbar (default: None)
    - wrap_width: max characters per line for wrapped labels (0 disables wrapping, default: 50)
    - square_cells: if True, enforce square aspect; if None, auto-detect for square matrices (default: None)
    - symmetric_option: if 'mask', mask upper triangle; if 'highlight', draw diagonal; if None, do nothing
    - label_fontsize: font size for heatmap cell labels (default: 10)
    - tick_labelsize: font size for axis tick labels (default: 10)
    - cbar_labelsize: font size for colorbar label (default: 10)
    - xlabel: custom label for x-axis (default: None)
    - ylabel: custom label for y-axis (default: None)
    """


    fig, ax = plt.subplots(figsize=(10, 8))
    

    fmt = ".2f" if normalized else ".0f"

    if wrap_width > 0:
        df.columns = ["\n".join(wrap(str(col), wrap_width)) for col in df.columns]
        df.index = ["\n".join(wrap(str(idx), wrap_width)) for idx in df.index]

    auto_square = square_cells if square_cells is not None else df.shape[0] == df.shape[1]
    mask = None
    if symmetric_option == "mask" and df.shape[0] == df.shape[1] and (df.columns == df.index).all():
        mask = np.triu(np.ones_like(df.values, dtype=bool))

    sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, cbar=True, ax=ax,
                annot_kws={"fontsize": label_fontsize},
                cbar_kws={"label": cbar_label, "format": None} if cbar_label else {},
                square=auto_square, mask=mask)

    if symmetric_option == "highlight" and df.shape[0] == df.shape[1] and (df.columns == df.index).all():
        for i in range(len(df)):
            for j in range(i + 1):
                if i == j:
                    edgecolor = "red"
                    linewidth = 1.5
                else:
                    continue
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor=edgecolor, lw=linewidth))
    ax.set_xlabel(xlabel if xlabel is not None else df.columns.name or "", fontsize=axis_labelsize)
    ax.set_ylabel(ylabel if ylabel is not None else df.index.name or "", fontsize=axis_labelsize)
    ax.tick_params(axis='both', labelsize=tick_labelsize)
    if ax.collections and hasattr(ax.collections[0], 'colorbar') and ax.collections[0].colorbar:
        ax.collections[0].colorbar.ax.tick_params(labelsize=cbar_labelsize)
        if cbar_label:
            ax.collections[0].colorbar.set_label(cbar_label, fontsize=cbar_labelsize)
    ax.set_ylabel(df.index.name or "", fontsize=axis_labelsize)
    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_clustermap(df, filename="clustermap", dpi=600, normalized=False, cmap="viridis",
                    wrap_width=50, square_cells=None, symmetric_option=None,
                    axis_labelsize=None, label_fontsize=10, tick_labelsize=10, cbar_labelsize=10,
                    xlabel=None, ylabel=None, cbar_label=None, figsize=(10, 10),
                    method="average", metric="euclidean", show=True):
    """
    Plot a clustermap from a DataFrame with optional symmetric masking, axis label wrapping,
    colorbar styling, and clustering customization.

    Parameters:
    - df: pandas DataFrame (for full functionality square and symmetric)
    - filename: base filename for saving the plots (default: 'clustermap')
    - dpi: resolution of saved plots (default: 600)
    - normalized: if True, uses float formatting; otherwise, integers (default: False)
    - cmap: colormap for heatmap (default: 'viridis')
    - wrap_width: character limit before wrapping axis labels (0 disables wrapping)
    - square_cells: placeholder for future cell shape control (currently unused)
    - symmetric_option: 'mask' to show lower triangle only, 'highlight' to outline diagonal, or None
    - axis_labelsize: font size for axis labels
    - label_fontsize: font size for annotations in cells
    - tick_labelsize: font size for axis tick labels
    - cbar_labelsize: font size for colorbar label and ticks
    - xlabel, ylabel: axis labels (overrides index/column name if set)
    - cbar_label: label for the colorbar (default: None)
    - figsize: figure size (default: (10, 10))
    - method: linkage algorithm for clustering (e.g. 'average', 'single', etc.)
    - metric: distance function for clustering (e.g. 'euclidean', 'cityblock')
    - show: whether to display the plot after saving (default: True)
    """

    # Optionally wrap long axis labels
    if wrap_width > 0:
        df.columns = ["\n".join(wrap(str(col), wrap_width)) for col in df.columns]
        df.index = ["\n".join(wrap(str(idx), wrap_width)) for idx in df.index]

    fmt = ".2f" if normalized else ".0f"

    # Create the clustermap with clustering result to reorder matrix
    mask = None

    if df.shape[0] == df.shape[1] and (df.columns == df.index).all():
        # Compute linkage on symmetric matrix
        from scipy.spatial.distance import pdist
        dist = pdist(df.values)
        linkage_matrix = linkage(dist, method=method, metric=metric)
        order = leaves_list(linkage_matrix)
        df = df.iloc[order, :].iloc[:, order]

    if symmetric_option == "mask" and df.shape[0] == df.shape[1] and (df.columns == df.index).all():
        df = df.copy()
        for i in range(df.shape[0]):
            for j in range(i + 1, df.shape[1]):
                df.iloc[j, i] = df.iloc[i, j]
        mask = np.triu(np.ones(df.shape), k=1).astype(bool)

    g = sns.clustermap(df, method=method, metric=metric, mask=mask, cmap=cmap,
                    annot=True, fmt=fmt, annot_kws={"fontsize": label_fontsize},
                    figsize=figsize, cbar_pos=(0.84, 0.33, 0.015, 0.32))

    # Axis labels and ticks
    g.ax_heatmap.set_xlabel(xlabel if xlabel is not None else df.columns.name or "", fontsize=axis_labelsize)
    g.ax_heatmap.set_ylabel(ylabel if ylabel is not None else df.index.name or "", fontsize=axis_labelsize)
    g.ax_heatmap.tick_params(axis="both", labelsize=tick_labelsize)

    # Colorbar label and ticks
    if g.cax and cbar_label:
        g.cax.set_ylabel(cbar_label, fontsize=cbar_labelsize)
        g.cax.tick_params(labelsize=cbar_labelsize)

    # Highlight diagonal if requested
    if symmetric_option == "highlight" and df.shape[0] == df.shape[1] and (df.columns == df.index).all():
        for i in range(len(df)):
            g.ax_heatmap.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="red", lw=1.5))

    # Save and optionally display
    plt.tight_layout()
    g.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")
    g.savefig(f"{filename}.svg", dpi=dpi, bbox_inches="tight")
    g.savefig(f"{filename}.pdf", dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_histogram(df, column, filename="histogram", bins=30, color="lightblue", dpi=600, show=True,
                   log_scale=False, log_y=False,
                   xlabel=None, ylabel="Frequency", title=None, fontsize=10, figsize=(8, 6),
                   fit_curve=False, curve_color="darkred", fit_normal=False):
    """
    Plot a histogram from a specified column in a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data
    - column: name of the column to plot
    - filename: base filename for saving the plot (default: 'histogram')
    - bins: number of bins in the histogram (default: 30)
    - color: fill color of the bars (default: 'lightblue')
    - dpi: resolution of saved plots (default: 600)
    - show: whether to display the plot after saving (default: True)
    - xlabel: label for the x-axis (default: column name)
    - ylabel: label for the y-axis (default: 'Frequency')
    - title: title of the plot (default: None)
    - fontsize: font size for labels and title (default: 10)
    - figsize: figure size (default: (8, 6))
    - fit_curve: if True, overlays a fitted normal distribution (default: False)
    - curve_color: color of the fitted curve (default: 'darkred')
    - fit_normal: if True, overlay a normal distribution curve (default: False)
    - log_scale: if True, use log scale on the x-axis (default: False)
    - log_y: if True, use log scale on the y-axis (default: False)
    """


    plt.figure(figsize=figsize)
    data = df[column].dropna()
    plt.hist(data, bins=bins, color=color, edgecolor="black", density=fit_curve, log=log_y)

    if fit_curve:
        sns.kdeplot(data, color=curve_color, linewidth=2, clip=(data.min(), data.max()))

    if fit_normal:
        
        mu, std = norm.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(data.min(), data.max(), 200)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, color="black", linestyle="--", linewidth=1.5)
    if log_scale:
        plt.xscale("log")
    plt.xlabel(xlabel if xlabel else column, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    
    if show:
        plt.show()
    plt.close()


def plot_pairplot(df, columns=None, hue=None, filename="pairplot", dpi=600, show=True,
                  diag_kind="auto", palette="Set2", plot_kws=None, height=2.5):
    """
    Plot a pairplot (scatterplot matrix) from selected columns of a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to include (default: all numeric columns)
    - hue: optional column name for color encoding (categorical)
    - filename: base filename to save the plot (default: 'pairplot')
    - dpi: resolution of saved plots (default: 600)
    - show: whether to display the plot (default: True)
    - diag_kind: 'kde', 'hist', or 'auto' for diagonal plots
    - palette: color palette to use (default: 'Set2')
    - plot_kws: dictionary of keyword arguments for scatter plots
    - height: size of each subplot (default: 2.5)
    """

    data = df[columns] if columns else df.select_dtypes(include=["number"])
    g = sns.pairplot(data, hue=hue, diag_kind=diag_kind, palette=palette,
                     plot_kws=plot_kws or {}, height=height)
    g.fig.set_dpi(dpi)
    g.fig.tight_layout()
    g.savefig(f"{filename}.png", dpi=dpi)
    g.savefig(f"{filename}.svg", dpi=dpi)
    g.savefig(f"{filename}.pdf", dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    
def plot_wordcloud(
    df,
    filename="wordcloud",
    dpi=600,
    show=True,
    background_color="white",
    cbar_location="bottom",
    cbar_labelsize=10,
    cbar_ticksize=10,
    item_col=None,
    size_col="Number of documents",
    color_by=None,
    colormap="viridis",
    figsize=(10, 8),
    mask_image=None,
    prefer_horizontal=1.0,
    top_n=None,
    top_n_by=None,
    scale_func=None,
    layout_mode="archimedean",  # kept for API compatibility; not used by wordcloud
):
    """
    Generate a word cloud visualization from frequency data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with items and their frequencies.
    filename : str
        Base filename for saving (without extension).
    dpi : int
        Resolution for saved images.
    show : bool
        Whether to display the plot.
    background_color : str
        Background color of the word cloud.
    cbar_location : str
        Location of colorbar ('bottom', 'right', etc.).
    cbar_labelsize : int
        Font size for colorbar label.
    cbar_ticksize : int
        Font size for colorbar ticks.
    item_col : str, optional
        Column containing words/items. Defaults to first column.
    size_col : str
        Column containing frequencies/sizes.
    color_by : str, optional
        Column for coloring words.
    colormap : str
        Matplotlib colormap name.
    figsize : tuple
        Figure size (width, height).
    mask_image : str, optional
        Path to mask image for cloud shape.
    prefer_horizontal : float
        Fraction of horizontal words (0-1).
    top_n : int, optional
        Limit to top N items.
    top_n_by : str, optional
        Column to use for selecting top N.
    scale_func : callable, optional
        Function to transform sizes.
    layout_mode : str
        Layout algorithm (for API compatibility).
    """
    # Import locally to avoid shadowing issues.
    try:
        from wordcloud import WordCloud as WC
    except Exception as e:
        raise ImportError("Install the 'wordcloud' package: pip install wordcloud") from e
    if not callable(WC):
        raise TypeError("wordcloud.WordCloud is not callable (name shadowed?).")

    # ---- Basic validation ----
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty pandas DataFrame.")

    item_col = item_col or df.columns[0]
    top_n_by = top_n_by or size_col

    if item_col not in df.columns:
        raise KeyError(f"item_col '{item_col}' not found in DataFrame.")
    if size_col not in df.columns:
        raise KeyError(f"size_col '{size_col}' not found in DataFrame.")
    if color_by is not None and color_by not in df.columns:
        raise KeyError(f"color_by '{color_by}' not found in DataFrame.")
    if top_n is not None and not isinstance(top_n, (int, np.integer)):
        raise TypeError("top_n must be an int or None.")
    if scale_func is not None and not callable(scale_func):
        raise TypeError("scale_func must be callable or None.")

    # ---- Subset / ranking ----
    work = df.copy()
    if top_n is not None:
        if top_n_by not in work.columns:
            raise KeyError(f"top_n_by '{top_n_by}' not found in DataFrame.")
        work = work.sort_values(by=top_n_by, ascending=False).head(int(top_n))

    # ---- Items & sizes ----
    items = work[item_col].astype(str)
    sizes = pd.to_numeric(work[size_col], errors="coerce").fillna(0.0).astype(float)
    if (sizes <= 0).all():
        raise ValueError(f"All values in '{size_col}' are non-positive or NaN after coercion.")
    if scale_func is not None:
        sizes = scale_func(sizes)

    # ---- Color mapping ----
    colorbar_type = None
    legend_labels = None
    norm = None
    cmap_obj = None

    if color_by is None:
        color_map = {item: "#808080" for item in items}  # gray
    else:
        color_values = work[color_by]
        if pd.api.types.is_numeric_dtype(color_values):
            vmin = float(np.nanmin(color_values))
            vmax = float(np.nanmax(color_values))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                raise ValueError(f"'{color_by}' has no finite numeric values.")
            if vmin == vmax:
                vmax = vmin + 1e-12  # avoid zero-range
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap_obj = cm.get_cmap(colormap)
            color_map = {
                item: mcolors.to_hex(cmap_obj(norm(val if np.isfinite(val) else vmin)))
                for item, val in zip(items, color_values)
            }
            colorbar_type = "continuous"
        else:
            categories = pd.Categorical(color_values.astype(str))
            palette = plt.get_cmap("tab10")
            colors = [mcolors.to_hex(palette(i % palette.N)) for i in range(len(categories.categories))]
            color_map = {item: colors[i] for item, i in zip(items, categories.codes)}
            legend_labels = dict(zip(categories.categories, colors))
            colorbar_type = "categorical"

    # ---- Mask handling ----
    mask_arr = None
    if mask_image is not None:
        if isinstance(mask_image, (str, os.PathLike)):
            from PIL import Image
            mask_arr = np.array(Image.open(mask_image))
        elif isinstance(mask_image, np.ndarray):
            mask_arr = mask_image
        else:
            raise TypeError("mask_image must be a numpy array, image path, or None.")

    # ---- Frequencies ----
    frequencies = dict(zip(items, sizes))

    # ---- Color function for WordCloud ----
    def color_func(word, **_):
        return color_map.get(word, "#808080")

    # ---- WordCloud generation ----
    wc = WC(
        width=1000,
        height=800,
        background_color=background_color,
        prefer_horizontal=float(prefer_horizontal),
        random_state=42,
        mask=mask_arr,
        color_func=color_func,
        collocations=False,
        relative_scaling=0,   # size purely from frequencies
        contour_width=0,
        regexp=None,
        max_words=len(frequencies),
    ).generate_from_frequencies(frequencies)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    # ---- Legend / Colorbar ----
    if (color_by is not None) and (colorbar_type == "categorical") and legend_labels:
        from matplotlib.patches import Patch
        handles = [Patch(color=color, label=str(label)) for label, color in legend_labels.items()]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(5, len(handles)),
            frameon=False,
        )
    elif (color_by is not None) and (colorbar_type == "continuous") and (cbar_location != "none"):
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        orientation = "horizontal" if cbar_location == "bottom" else "vertical"
        cb = fig.colorbar(sm, ax=ax, orientation=orientation, fraction=0.046, pad=0.04)
        cb.set_label(str(color_by), fontsize=cbar_labelsize)
        cb.ax.tick_params(labelsize=cbar_ticksize)

    # ---- Save ----
    plt.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{filename}.{ext}", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return fig
    
    
def plot_treemap(df, filename="treemap", item_col=None, size_col="Number of documents", color_by=None,
                 cmap="viridis", figsize=(10, 6), dpi=600, show=True,
                 sort=True, ascending=False, label_fontsize=10, min_label_size_ratio=0.01,
                 wrap_width=15, show_frame=False, min_fontsize=6, max_fontsize=28, log_scale=False):
    """
    Plot a treemap based on a DataFrame with item sizes and optional color mapping.

    Parameters:
    - df: pandas DataFrame containing the data
    - filename: base filename for saved plots (default: 'treemap')
    - item_col: name of the column with item labels (default: first column)
    - size_col: name of the column that controls the area size of each item
    - color_by: optional column for color mapping (categorical or numerical)
    - cmap: colormap name used for continuous color scale (default: 'viridis')
    - figsize: figure size (default: (10, 6))
    - dpi: resolution of output image (default: 600)
    - show: whether to display the plot after saving
    - sort: whether to sort items by size (default: True)
    - ascending: sorting order (default: False for descending)
    - label_fontsize: font size of item labels inside treemap (default: 10)
    - min_label_size_ratio: minimum size proportion to draw label (default: 0.01)
    - wrap_width: character width for wrapped labels (default: 15)
    - show_frame: whether to show frames around boxes (default: False)
    - min_fontsize: minimum font size for labels (default: 6)
    - max_fontsize: maximum font size for labels (default: 20)
    - log_scale: whether to apply log scaling to size values for font size and box area (default: False)
    """


    item_col = item_col or df.columns[0]
    if sort:
        df = df.sort_values(by=size_col, ascending=ascending)
    sizes = df[size_col]
    labels = ["\n".join(textwrap.wrap(str(label), width=wrap_width)) if size >= sizes.sum() * min_label_size_ratio else ""
              for label, size in zip(df[item_col], sizes)]

    if color_by is None:
        color_scheme = "default"
        colors = ["lightblue"] * len(df)
        colorbar_type = None

    color_scheme = infer_color_scheme(df[color_by]) if color_by else "default"

    if color_scheme == "default":
        colors = ["lightblue"] * len(df)
        colorbar_type = None
    else:
        values = df[color_by]
        if pd.api.types.is_numeric_dtype(values):
            norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
            cmap = cm.get_cmap(cmap)
            colors = [cmap(norm(val)) for val in values]
            colorbar_type = "continuous"
        else:
            categories = pd.Categorical(values)
            palette = plt.get_cmap("tab10")
            colormap = [palette(i) for i in range(len(categories.categories))]
            legend_labels = dict(zip(categories.categories, colormap))
            colorbar_type = "categorical"

    fig, ax = plt.subplots(figsize=figsize)
    if log_scale:
        scaled_sizes = np.log1p(sizes)
    else:
        scaled_sizes = sizes
    scaled_fonts = [max(min_fontsize, min(max_fontsize, label_fontsize * (s / max(scaled_sizes)))) for s in scaled_sizes]

    normed_sizes = squarify.normalize_sizes(scaled_sizes, 100, 100)
    boxes = squarify.squarify(normed_sizes, 0, 0, 100, 100)

    # Draw the treemap boxes
    for i, (box, color) in enumerate(zip(boxes, colors)):
        rect = plt.Rectangle(
            (box['x'], box['y']), box['dx'], box['dy'],
            facecolor=color,
            edgecolor='white' if show_frame else color,
            linewidth=1 if show_frame else 0
        )
        ax.add_patch(rect)
    
    # Add labels on top of boxes
    for label, box, fontsize in zip(labels, boxes, scaled_fonts):
        if label:
            x = box['x'] + box['dx'] / 2
            y = box['y'] + box['dy'] / 2
            ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, clip_on=True,
                   color='black', weight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    if color_by and colorbar_type == "categorical":       
        handles = [Patch(color=color, label=str(label)) for label, color in legend_labels.items()]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(handles)))
    elif color_by and colorbar_type == "continuous":
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, label=color_by)

    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_boxplot(
    df, value_column, group_by=None, group_matrix=None, min_group_size=5, max_groups=None,
    figsize=(10, 6), title=None, x_label_size=14, y_label_size=14, tick_label_size=12,
    value_label=None, filename_base=None, dpi=600, group_order_user=None, order_by_size=False,
    show_counts=False, return_summary=False, label_angle=90, stat_test=False, wrap_width=30,
    show=True, group_colors=None, clean_underscore=True
):
    """
    Plot a boxplot of a numerical column grouped by either a column in df or a binary group matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    value_column : str
        Column name for the values to plot.
    group_by : str, optional
        Column name for grouping.
    group_matrix : pd.DataFrame, optional
        Binary matrix for grouping.
    min_group_size : int, default 5
        Minimum group size to include.
    max_groups : int, optional
        Maximum number of groups to show.
    figsize : tuple, default (10, 6)
        Figure size.
    title : str, optional
        Plot title.
    x_label_size : int, default 14
        X-axis label font size.
    y_label_size : int, default 14
        Y-axis label font size.
    tick_label_size : int, default 12
        Tick label font size.
    value_label : str, optional
        Custom label for value axis.
    filename_base : str, optional
        Base filename for saving.
    dpi : int, default 600
        Resolution for saved images.
    group_order_user : list, optional
        Custom group order.
    order_by_size : bool, default False
        Order groups by size.
    show_counts : bool, default False
        Show group counts in labels.
    return_summary : bool, default False
        Return summary statistics.
    label_angle : int, default 90
        Rotation angle for x-tick labels.
    stat_test : bool, default False
        Perform Kruskal-Wallis test.
    wrap_width : int, default 30
        Width for label wrapping.
    show : bool, default True
        Display the plot.
    group_colors : dict, optional
        Dictionary mapping group names to specific colors.
    clean_underscore : bool, default True
        Replace underscores with spaces in labels.
    """

    if group_by is not None and group_matrix is not None:
        raise ValueError("Specify only one of group_by or group_matrix.")

    # Prepare data
    if group_by is not None:
        group_sizes = df[group_by].value_counts()
        valid_groups = group_sizes[group_sizes >= min_group_size].index.tolist()
        if max_groups is not None:
            valid_groups = valid_groups[:max_groups]
        plot_data = df[df[group_by].isin(valid_groups)]
        group_order = group_order_user if group_order_user is not None else plot_data[group_by].value_counts().index.tolist()
        counts = plot_data[group_by].value_counts() if show_counts else None
        x = group_by
    elif group_matrix is not None:
        melted = []
        for group_name in group_matrix.columns:
            indices = group_matrix.index[group_matrix[group_name] == 1]
            values = df.loc[indices, value_column]
            if values.isna().all():
                continue
            values = values.dropna()
            if len(values) >= min_group_size:
                melted.append(pd.DataFrame({value_column: values, 'Group': group_name}))
        if not melted:
            raise ValueError("No groups met the minimum size requirement.")
        plot_data = pd.concat(melted)
        if max_groups is not None:
            group_counts = plot_data['Group'].value_counts()
            valid_groups = group_counts.head(max_groups).index
            plot_data = plot_data[plot_data['Group'].isin(valid_groups)]
        if group_order_user is not None:
            group_order = [g for g in group_order_user if g in plot_data['Group'].unique()]
        elif order_by_size:
            group_order = plot_data['Group'].value_counts().index.tolist()
        else:
            group_order = [col for col in group_matrix.columns if col in plot_data['Group'].unique()]
        counts = plot_data['Group'].value_counts() if show_counts else None
        x = 'Group'
    else:
        raise ValueError("One of group_by or group_matrix must be specified.")

    # Set colors
    if group_colors:
        palette = {g: group_colors.get(g, '#cccccc') for g in group_order}
    else:
        palette = get_colors(len(group_order), color_scheme="categorical")

    # Plotting
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=plot_data,
        x=x,
        y=value_column,
        hue=x,            # Fix for deprecation
        order=group_order,
        palette=palette,
        legend=False      # Fix for deprecation
    )

    # Clean labels if requested
    if clean_underscore:
        x_label = clean_label(group_by) if group_by is not None else "Group"
        y_label = clean_label(value_label) if value_label is not None else clean_label(value_column)
        group_order_display = clean_labels(group_order)
    else:
        x_label = group_by if group_by is not None else "Group"
        y_label = value_label if value_label is not None else value_column
        group_order_display = list(group_order)

    plt.xlabel(x_label, fontsize=x_label_size)
    plt.ylabel(y_label, fontsize=y_label_size)
    
    if show_counts and counts is not None:
        group_order_labels = [f"{g} (n={counts[group_order[i]]})" for i, g in enumerate(group_order_display)]
    else:
        group_order_labels = group_order_display
    plt.xticks(
        ticks=range(len(group_order)),
        labels=wrap_labels(group_order_labels, width=wrap_width),
        rotation=label_angle,
        ha='right',
        fontsize=tick_label_size
    )
    plt.yticks(fontsize=tick_label_size)

    if title:
        plt.title(clean_label(title) if clean_underscore else title, fontsize=16)
    else:
        plt.title("")

    plt.grid(False)
    plt.tight_layout()

    # Statistical test (Kruskal-Wallis)
    if stat_test and len(group_order) > 1:
        groups = [plot_data[plot_data[x] == grp][value_column].dropna() for grp in group_order]
        stat, pval = kruskal(*groups)
        plt.figtext(0.99, 0.01, f"Kruskal-Wallis p = {pval:.3g}", horizontalalignment='right', fontsize=12)

    if filename_base:
        save_plot(filename_base, dpi=dpi)

    if show:
        plt.show()

    if return_summary:
        return plot_data.groupby(x)[value_column].describe()


def plot_violinplot(
    df, value_column, group_by=None, group_matrix=None, min_group_size=5, max_groups=None,
    figsize=(10, 6), title=None, x_label_size=14, y_label_size=14, tick_label_size=12,
    value_label=None, filename_base=None, dpi=600, group_order_user=None, order_by_size=False,
    show_counts=False, return_summary=False, label_angle=90, stat_test=False, wrap_width=30,
    show=True, group_colors=None, clean_underscore=True
):
    """
    Plot a violin plot of a numerical column grouped by either a column in df or a binary group matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    value_column : str
        Column name for the values to plot.
    group_by : str, optional
        Column name for grouping.
    group_matrix : pd.DataFrame, optional
        Binary matrix for grouping.
    min_group_size : int, default 5
        Minimum group size to include.
    max_groups : int, optional
        Maximum number of groups to show.
    figsize : tuple, default (10, 6)
        Figure size.
    title : str, optional
        Plot title.
    x_label_size : int, default 14
        X-axis label font size.
    y_label_size : int, default 14
        Y-axis label font size.
    tick_label_size : int, default 12
        Tick label font size.
    value_label : str, optional
        Custom label for value axis.
    filename_base : str, optional
        Base filename for saving.
    dpi : int, default 600
        Resolution for saved images.
    group_order_user : list, optional
        Custom group order.
    order_by_size : bool, default False
        Order groups by size.
    show_counts : bool, default False
        Show group counts in labels.
    return_summary : bool, default False
        Return summary statistics.
    label_angle : int, default 90
        Rotation angle for x-tick labels.
    stat_test : bool, default False
        Perform Kruskal-Wallis test.
    wrap_width : int, default 30
        Width for label wrapping.
    show : bool, default True
        Display the plot.
    group_colors : dict, optional
        Dictionary mapping group names to specific colors.
    clean_underscore : bool, default True
        Replace underscores with spaces in labels.
    """
    if group_by is not None and group_matrix is not None:
        raise ValueError("Specify only one of group_by or group_matrix.")

    if group_by is not None:
        group_sizes = df[group_by].value_counts()
        valid_groups = group_sizes[group_sizes >= min_group_size].index.tolist()
        if max_groups is not None:
            valid_groups = valid_groups[:max_groups]
        plot_data = df[df[group_by].isin(valid_groups)]
        group_order = group_order_user if group_order_user is not None else plot_data[group_by].value_counts().index.tolist()
        counts = plot_data[group_by].value_counts() if show_counts else None
        x = group_by
    elif group_matrix is not None:
        melted = []
        for group_name in group_matrix.columns:
            indices = group_matrix.index[group_matrix[group_name] == 1]
            values = df.loc[indices, value_column]
            if values.isna().all():
                continue
            values = values.dropna()
            if len(values) >= min_group_size:
                melted.append(pd.DataFrame({value_column: values, 'Group': group_name}))
        if not melted:
            raise ValueError("No groups met the minimum size requirement.")
        plot_data = pd.concat(melted)
        if max_groups is not None:
            group_counts = plot_data['Group'].value_counts()
            valid_groups = group_counts.head(max_groups).index
            plot_data = plot_data[plot_data['Group'].isin(valid_groups)]
        if group_order_user is not None:
            group_order = [g for g in group_order_user if g in plot_data['Group'].unique()]
        elif order_by_size:
            group_order = plot_data['Group'].value_counts().index.tolist()
        else:
            group_order = [col for col in group_matrix.columns if col in plot_data['Group'].unique()]
        counts = plot_data['Group'].value_counts() if show_counts else None
        x = 'Group'
    else:
        raise ValueError("One of group_by or group_matrix must be specified.")

    # Set colors
    if group_colors:
        palette = {g: group_colors.get(g, '#cccccc') for g in group_order}
    else:
        palette = get_colors(len(group_order), color_scheme="categorical")

    plt.figure(figsize=figsize)
    sns.violinplot(
        data=plot_data,
        x=x,
        y=value_column,
        hue=x,              # <- Fix for deprecation
        order=group_order,
        palette=palette,
        legend=False        # <- Fix for deprecation
    )

    # Clean labels if requested
    if clean_underscore:
        x_label = clean_label(group_by) if group_by is not None else "Group"
        y_label = clean_label(value_label) if value_label is not None else clean_label(value_column)
        group_order_display = clean_labels(group_order)
    else:
        x_label = group_by if group_by is not None else "Group"
        y_label = value_label if value_label is not None else value_column
        group_order_display = list(group_order)

    plt.xlabel(x_label, fontsize=x_label_size)
    plt.ylabel(y_label, fontsize=y_label_size)
    
    if show_counts and counts is not None:
        group_order_labels = [f"{g} (n={counts[group_order[i]]})" for i, g in enumerate(group_order_display)]
    else:
        group_order_labels = group_order_display
    plt.xticks(
        ticks=range(len(group_order)),
        labels=wrap_labels(group_order_labels, width=wrap_width),
        rotation=label_angle,
        ha='right',
        fontsize=tick_label_size
    )
    plt.yticks(fontsize=tick_label_size)

    if title:
        plt.title(clean_label(title) if clean_underscore else title, fontsize=16)
    else:
        plt.title("")

    plt.grid(False)
    plt.tight_layout()

    if stat_test and len(group_order) > 1:
        groups = [plot_data[plot_data[x] == grp][value_column].dropna() for grp in group_order]
        stat, pval = kruskal(*groups)
        plt.figtext(0.99, 0.01, f"Kruskal-Wallis p = {pval:.3g}", horizontalalignment='right', fontsize=12)

    if filename_base:
        save_plot(filename_base, dpi=dpi)

    if show:
        plt.show()

    if return_summary:
        return plot_data.groupby(x)[value_column].describe()


def plot_box_and_violin_by_groups(
    df,
    value_column: str,
    group_matrix,
    *,
    # saving
    out_dir: str | None = None,        # e.g., os.path.join(self.res_folder, "plots")
    file_prefix: str | None = None,    # optional prefix for filenames
    dpi: int = 600,
    # common options (forwarded to your functions)
    min_group_size: int = 5,
    max_groups: int | None = None,
    order_by_size: bool = True,
    show_counts: bool = True,
    group_order_user: list[str] | None = None,
    value_label: str | None = None,
    title_prefix: str | None = None,
    group_colors: dict | None = None,
    # return summary (from your funcs)
    return_summary: bool = False,
    # any other kwargs get passed through to BOTH functions (kept minimal)
    **kwargs,
):
    """
    Draws boxplot AND violin plot for df[value_column] across the provided group_matrix.
    Uses existing plot_boxplot and plot_violinplot; saves as ..._box and ..._violin.
    """

    # ----- filenames -----
    safe_val = re.sub(r"[^A-Za-z0-9]+", "-", str(value_column)).strip("-").lower()
    base = f"{file_prefix+'_' if file_prefix else ''}{safe_val}-by-groups"
    filename_base_box    = os.path.join(out_dir, base + "_box")    if out_dir else None
    filename_base_violin = os.path.join(out_dir, base + "_violin") if out_dir else None

    # ----- titles -----
    tprefix = (title_prefix + " — ") if title_prefix else ""
    title_box    = f"{tprefix}{value_column} by group (Boxplot)"
    title_violin = f"{tprefix}{value_column} by group (Violin)"

    # ----- call your existing functions -----
    box_summary = plot_boxplot(
        df=df,
        value_column=value_column,
        group_by=None,
        group_matrix=group_matrix,
        min_group_size=min_group_size,
        max_groups=max_groups,
        title=title_box,
        value_label=value_label,
        filename_base=filename_base_box,
        dpi=dpi,
        group_order_user=group_order_user,
        order_by_size=order_by_size,
        show_counts=show_counts,
        return_summary=return_summary,
        group_colors=group_colors,
        **kwargs,
    )

    violin_summary = plot_violinplot(
        df=df,
        value_column=value_column,
        group_by=None,
        group_matrix=group_matrix,
        min_group_size=min_group_size,
        max_groups=max_groups,
        title=title_violin,
        value_label=value_label,
        filename_base=filename_base_violin,
        dpi=dpi,
        group_order_user=group_order_user,
        order_by_size=order_by_size,
        show_counts=show_counts,
        return_summary=return_summary,
        group_colors=group_colors,
        **kwargs,
    )

    if return_summary:
        return {"boxplot": box_summary, "violinplot": violin_summary}


def plot_group_distributions_aligned(df, numerical_cols, group_matrix, bins=30, alpha=0.7, 
                                     save=False, filename_prefix="group_dist", dpi=600, 
                                     show_grid=False, group_colors={}):
    """
    Plot histograms of numerical variables for each group defined in a binary matrix,
    with one subplot per group, aligned on a shared x-axis.

    Parameters:
    df (pd.DataFrame): DataFrame containing the numerical data.
    numerical_cols (list of str): List of numerical column names to plot.
    group_matrix (pd.DataFrame): Binary matrix where each column is a group with 0/1 values.
    bins (int): Number of bins for histograms (default: 30).
    alpha (float): Transparency for histogram fill (default: 0.7).
    save (bool): Whether to save the plot to file.
    filename_prefix (str): Prefix for saved plot filenames (default: 'group_dist').
    dpi (int): DPI resolution for saving plots (default: 600).
    show_grid (bool): Whether to show gridlines (default: False).
    group_colors (dict): Optional dict mapping group names to colors (default: {}).
    """
    default_color = "skyblue"

    for col in numerical_cols:
        num_groups = len(group_matrix.columns)
        fig, axes = plt.subplots(num_groups, 1, figsize=(8, 3 * num_groups), sharex=True)
        if num_groups == 1:
            axes = [axes]

        global_min = df[col][np.isfinite(df[col])].min()
        global_max = df[col][np.isfinite(df[col])].max()
        bin_edges = np.linspace(global_min, global_max, bins + 1)

        for ax, group in zip(axes, group_matrix.columns):
            mask = group_matrix[group] == 1
            data = df.loc[mask, col]
            data = data[np.isfinite(data)]

            color = group_colors.get(group, default_color)

            ax.hist(data, bins=bin_edges, alpha=alpha, color=color, edgecolor="black", density=True)
            ax.set_title(f"{col} - {group}")
            ax.set_ylabel("Density")
            if show_grid:
                ax.grid(True, linestyle="--", alpha=0.6)

        axes[-1].set_xlabel(col)
        plt.tight_layout()

        if save:
            save_plot(f"{filename_prefix}_{col}", dpi=dpi)

        plt.show()


def plot_group_venn(group_matrix: pd.DataFrame, title: str = None, filename: str = None, dpi: int = 600, include_totals: bool = True, show: bool = True, save_results: bool = True, group_color: dict = None, alpha: float = 0.5, **kwargs):
    """
    Plots a Venn diagram for 2–6 groups using the 'venn' package.
    Adds a compact legend in the top-right corner with matched colors, transparency, and outlines.

    Parameters:
    - group_matrix: pd.DataFrame
        Binary matrix (0/1), rows = items, columns = groups.
    - title: str
        Title of the plot.
    - filename: str or None
        If given, saves plot using save_plot(filename, dpi).
    - dpi: int
        Resolution of saved figures.
    - include_totals: bool
        If True, appends (n=...) to group labels.
    - show: bool
        If True, displays the plot.
    - save_results: bool
        If True, calls save_plot().
    - group_color: dict or None
        Optional dictionary mapping group names to colors.
    - alpha: float
        Transparency level for region fills.
    - **kwargs: dict
        Passed directly to venn(), e.g., cmap.

    Requirements:
    - venn (https://pypi.org/project/venn/)
    """


    sets_raw = {col: set(group_matrix.index[group_matrix[col] == 1]) for col in group_matrix.columns}

    if include_totals:
        label_map = {col: f"{col} (n={len(val)})" for col, val in sets_raw.items()}
        labeled_sets = {label_map[col]: sets_raw[col] for col in group_matrix.columns}
    else:
        label_map = {col: col for col in sets_raw}
        labeled_sets = sets_raw

    # Group names and final labels
    group_names = list(group_matrix.columns)
    label_names = [label_map[g] for g in group_names]

    # Assign colors
    if group_color:
        color_cycle = [group_color.get(g, "gray") for g in group_names]
    else:
        color_iter = itertools.cycle(rcParams["axes.prop_cycle"].by_key()["color"])
        color_cycle = [next(color_iter) for _ in group_names]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    venn(labeled_sets, ax=ax, **kwargs)

    # Apply color, transparency, and edge styling
    patch_handles = []
    for label, color, patch in zip(label_names, color_cycle, ax.patches):
        if patch is not None:
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)
            patch_handles.append(Patch(facecolor=color, edgecolor="black", label=label, alpha=alpha))

    ax.legend(
        handles=patch_handles,
        title="Groups",
        loc="upper right",
        fontsize="small",
        title_fontsize="small",
        frameon=True,
        borderpad=0.5,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.5
    )

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if filename and save_results:
        save_plot(filename, dpi=dpi)

    if show:
        plt.show()


def plot_group_upset(group_matrix: pd.DataFrame, title: str = None, filename: str = None, dpi: int = 600, show: bool = True, save_results: bool = True, group_color: dict = None, **kwargs):
    """
    Plots an UpSet diagram showing set intersections based on a binary group membership matrix.

    Parameters:
    - group_matrix: pd.DataFrame
        Binary matrix (0/1), rows = items (e.g., documents), columns = group names.
    - title: str
        Title for the figure.
    - filename: str or None
        Base name for saving PNG, SVG, and PDF via save_plot.
    - dpi: int
        Resolution of saved figures.
    - show: bool
        Whether to display the figure.
    - save_results: bool
        Whether to save the figure using save_plot.
    - group_color: dict or None
        Optional dictionary mapping group names to specific colors.
    - **kwargs: dict
        Additional keyword arguments passed to the UpSet constructor.

    Requirements:
    - upsetplot (https://github.com/jnothman/UpSetPlot)
    """
    try:
        import upsetplot
        import itertools
    except ImportError:
        print("The 'upsetplot' package is required for this function. Install it via 'pip install upsetplot'.")
        return

    # Prepare the UpSet data
    data = from_indicators(group_matrix.columns.tolist(), group_matrix.astype(bool))

    # Color logic
    group_names = list(group_matrix.columns)
    if group_color:
        color_cycle = [group_color.get(g, "gray") for g in group_names]
    else:
        color_iter = itertools.cycle(rcParams["axes.prop_cycle"].by_key()["color"])
        color_cycle = [next(color_iter) for _ in group_names]

    # Default UpSet kwargs (overridable by user)
    default_kwargs = {
        "show_counts": True,
        "show_percentages": False,
        "sort_categories_by": "cardinality",
        "intersection_plot_elements": 20
    }
    default_kwargs.update(kwargs)

    # Plot
    fig = plt.figure(figsize=(9, 6))
    upset = UpSet(data, **default_kwargs)
    upset.plot(fig=fig)

    # Apply colors to group totals bar chart (first subplot)
    ax_cat_totals = fig.axes[0]
    for bar, color in zip(ax_cat_totals.patches, color_cycle):
        bar.set_facecolor(color)

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if filename and save_results:
        save_plot(filename, dpi=dpi)

    if show:
        plt.show()


def plot_group_heatmap(group_matrix: pd.DataFrame,
                       methods: list = ["jaccard"],
                       title: str = None,
                       filename: str = "group_heatmap",
                       dpi: int = 600,
                       group_color: dict = None,
                       color_ticks: bool = False,
                       show: bool = True,
                       save_results: bool = True,
                       save_csv: bool = False,
                       **kwargs):
    """
    Computes and plots group × group heatmaps using various similarity/distance measures.

    Parameters:
    - group_matrix: pd.DataFrame
        Binary matrix (0/1) with rows = items and columns = group names.
    - methods: list of str
        Measures to compute. Supported: 'jaccard', 'count', 'sokal-michener', 'simple-matching', 'rogers-tanimoto'.
    - title: str
        Title prefix for the plots.
    - filename: str
        Base filename for saving output images and CSVs.
    - dpi: int
        Resolution for saved images.
    - group_color: dict
        Optional group-to-color mapping.
    - color_ticks: bool
        If True, apply group_color to tick labels.
    - show: bool
        Whether to display the plot.
    - save_results: bool
        Whether to save output image files.
    - save_csv: bool
        Whether to save the computed matrix to a CSV file.
    - **kwargs:
        Passed through to plot_heatmap, e.g. cmap, label_fontsize, wrap_width, symmetric_option, etc.
    """
    matrices = utilsbib.compute_group_similarity_matrices(group_matrix, methods)
    labels = {
        "jaccard": "Jaccard Index",
        "count": "Shared Items",
        "sokal-michener": "Sokal-Michener",
        "simple-matching": "Simple Matching",
        "rogers-tanimoto": "Rogers-Tanimoto"
    }

    for method, mat in matrices.items():
        normalized = method != "count"
        cbar_label = labels[method]
        fname = f"{filename}_{method}"

        plot_heatmap(df=mat,
                     filename=fname,
                     dpi=dpi,
                     show=show,
                     normalized=normalized,
                     cbar_label=cbar_label,
                     xlabel="",
                     ylabel="",
                     **kwargs)

        if save_csv:
            csv_filename = f"{fname}.csv"
            mat.to_csv(csv_filename)

        if color_ticks and group_color:
            ax = plt.gca()
            for label in ax.get_xticklabels():
                label.set_color(group_color.get(label.get_text(), "black"))
            for label in ax.get_yticklabels():
                label.set_color(group_color.get(label.get_text(), "black"))
            plt.draw()


def plot_group_intersection_network(
    group_matrix: pd.DataFrame,
    method: str = "jaccard",
    threshold: float = 0.1,
    node_scale: float = 500,
    edge_scale: float = 3.0,
    group_color: dict = None,
    title: str = None,
    filename: str = None,
    dpi: int = 600,
    show: bool = True,
    save_results: bool = True,
    figsize: tuple = (10, 8),
    layout: str = "spring",
    show_edge_labels: bool = True,
    font_size: int = 10,
    **kwargs
):
    """
    Plots a network visualization of group intersections/similarities.
    
    Nodes represent groups (sized by number of documents).
    Edges represent overlap/similarity between groups (weighted by similarity measure).
    
    Parameters:
    -----------
    group_matrix : pd.DataFrame
        Binary matrix (0/1), rows = items, columns = groups.
    method : str
        Similarity method: 'jaccard', 'count', 'dice', 'overlap'.
        - 'jaccard': Jaccard index (intersection / union)
        - 'count': Raw count of shared items
        - 'dice': Dice coefficient (2*intersection / (size_a + size_b))
        - 'overlap': Overlap coefficient (intersection / min(size_a, size_b))
    threshold : float
        Minimum similarity to show an edge (default: 0.1).
    node_scale : float
        Scaling factor for node sizes (default: 500).
    edge_scale : float
        Scaling factor for edge widths (default: 3.0).
    group_color : dict
        Optional dictionary mapping group names to colors.
    title : str
        Title for the plot.
    filename : str
        Base filename for saving.
    dpi : int
        Resolution for saved figures.
    show : bool
        Whether to display the plot.
    save_results : bool
        Whether to save the figure.
    figsize : tuple
        Figure size (width, height).
    layout : str
        Network layout: 'spring', 'circular', 'kamada_kawai', 'shell'.
    show_edge_labels : bool
        Whether to show similarity values on edges.
    font_size : int
        Font size for labels.
    **kwargs
        Additional arguments passed to networkx drawing functions.
    
    Returns:
    --------
    tuple
        (fig, ax, G) - figure, axes, and networkx graph object.
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required for this function. Install via 'pip install networkx'.")
        return None, None, None
    
    groups = group_matrix.columns.tolist()
    n_groups = len(groups)
    
    # Compute group sizes
    group_sizes = {g: group_matrix[g].sum() for g in groups}
    
    # Compute pairwise similarities
    similarity_matrix = pd.DataFrame(index=groups, columns=groups, dtype=float)
    
    for i, g1 in enumerate(groups):
        set1 = set(group_matrix.index[group_matrix[g1] == 1])
        for j, g2 in enumerate(groups):
            if i == j:
                similarity_matrix.loc[g1, g2] = 1.0
                continue
            
            set2 = set(group_matrix.index[group_matrix[g2] == 1])
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            if method == "jaccard":
                sim = intersection / union if union > 0 else 0
            elif method == "count":
                sim = intersection
            elif method == "dice":
                sim = 2 * intersection / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0
            elif method == "overlap":
                sim = intersection / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
            else:
                sim = intersection / union if union > 0 else 0  # default to jaccard
            
            similarity_matrix.loc[g1, g2] = sim
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes with sizes
    for g in groups:
        G.add_node(g, size=group_sizes[g])
    
    # Add edges (only above threshold)
    # Normalize threshold for count method
    if method == "count":
        actual_threshold = threshold
    else:
        actual_threshold = threshold
    
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i < j:  # Only upper triangle
                sim = similarity_matrix.loc[g1, g2]
                if sim > actual_threshold:
                    G.add_edge(g1, g2, weight=sim)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=2/np.sqrt(n_groups), iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Node colors
    if group_color:
        node_colors = [group_color.get(g, "#1f77b4") for g in G.nodes()]
    else:
        cmap = plt.cm.tab10
        node_colors = [cmap(i % 10) for i in range(len(G.nodes()))]
    
    # Node sizes based on group size
    max_size = max(group_sizes.values()) if group_sizes else 1
    node_sizes = [node_scale * (group_sizes[g] / max_size + 0.3) for g in G.nodes()]
    
    # Edge widths based on similarity
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        if method == "count":
            max_weight = max(edge_weights) if edge_weights else 1
            edge_widths = [edge_scale * (w / max_weight) for w in edge_weights]
        else:
            edge_widths = [edge_scale * w for w in edge_weights]
    else:
        edge_widths = []
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                           node_size=node_sizes, alpha=0.8)
    
    if G.edges():
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                               alpha=0.6, edge_color="gray")
    
    # Node labels with size info
    labels = {g: f"{g}\n(n={group_sizes[g]})" for g in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=font_size)
    
    # Edge labels (similarity values)
    if show_edge_labels and G.edges():
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, 
                                     font_size=font_size-2, alpha=0.7)
    
    # Title
    method_labels = {
        "jaccard": "Jaccard Similarity",
        "count": "Shared Documents",
        "dice": "Dice Coefficient",
        "overlap": "Overlap Coefficient"
    }
    if title:
        ax.set_title(title, fontsize=font_size + 4)
    else:
        ax.set_title(f"Group Intersection Network ({method_labels.get(method, method)})", 
                     fontsize=font_size + 4)
    
    ax.axis("off")
    plt.tight_layout()
    
    if filename and save_results:
        save_plot(filename, dpi=dpi)
    
    if show:
        plt.show()
    
    return fig, ax, G


def plot_group_chord(matrix: pd.DataFrame, threshold: float = 0.0, group_color: dict = None, title: str = None, filename: str = None, dpi: int = 600, show: bool = True):
    """
    Plots a chord diagram from a group × group similarity matrix.

    Parameters:
    - matrix: pd.DataFrame
        A symmetric similarity matrix (e.g., from compute_group_similarity_matrices).
    - threshold: float
        Minimum value to include a connection (default: 0.0).
    - group_color: dict
        Optional color dictionary for each group.
    - title: str
        Optional title for the figure.
    - filename: str
        Base filename to save the diagram.
    - dpi: int
        Resolution for saving the figure.
    - show: bool
        Whether to show the plot.
    """


    hv.extension("matplotlib")

    # Ensure all labels are strings
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    if group_color:
        group_color = {str(k): v for k, v in group_color.items()}

    # Extract upper triangle (no self or duplicate edges)
    links = []
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            if j <= i:
                continue
            weight = matrix.iloc[i, j]
            if weight >= threshold:
                links.append((row, col, weight))

    if not links:
        print("No links to plot above threshold.")
        return None

    # Create dataset for chord with explicit kdims and vdims
    chord_data = hv.Dataset(pd.DataFrame(links, columns=["source", "target", "value"]),
                            kdims=["source", "target"],
                            vdims=["value"])
    chord = hv.Chord(chord_data)

    # Configure options
    chord_opts = dict(
        edge_color="source",
        node_color="index",
        show_legend=False
    )
    if group_color:
        chord_opts["cmap"] = list(group_color.values())

    chord = chord.select(value=(threshold, None)).opts(opts.Chord(**chord_opts))

    # Plot using Holoviews and capture the figure
    fig = hv.render(chord, backend="matplotlib")
    fig.set_size_inches(8, 8)

    # Add manual labels
    ax = fig.axes[0]
    labels = matrix.columns.tolist()
    num_labels = len(labels)
    radius = 1.2
    angle_step = 2 * np.pi / num_labels
    for i, label in enumerate(labels):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        ha = "left" if np.cos(angle) > 0 else "right"
        va = "bottom" if np.sin(angle) > 0 else "top"
        color = group_color.get(label, "black") if group_color else "black"
        ax.text(x, y, label, ha=ha, va=va, fontsize=9, color=color, rotation=np.degrees(angle), rotation_mode="anchor")

    if title:
        ax.set_title(title)

    if filename:
        fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(f"{filename}.svg", bbox_inches="tight")
        fig.savefig(f"{filename}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return chord


# dodaj MDS plot za skupine


"""
def compute_group_similarity_matrices(group_matrix: pd.DataFrame, methods: list = ["jaccard"]) -> dict:
    ...


def plot_group_heatmap(group_matrix: pd.DataFrame,
                       methods: list = ["jaccard"],
                       title: str = None,
                       filename: str = "group_heatmap",
                       dpi: int = 600,
                       group_color: dict = None,
                       color_ticks: bool = False,
                       show: bool = True,
                       save_results: bool = True,
                       save_csv: bool = False,
                       **kwargs):
    ...
"""

def plot_group_dendrogram(group_matrix: pd.DataFrame, method: str = "average", metric: str = "euclidean", title: str = None, filename: str = None, dpi: int = 600, show: bool = True):
    """
    Plots a dendrogram (hierarchical clustering) from a binary group membership matrix.

    Parameters:
    - group_matrix: pd.DataFrame
        A binary indicator matrix (0/1), rows = items, columns = group names.
    - method: str
        Linkage method (e.g., "average", "complete", "single").
    - metric: str
        Distance metric (e.g., "euclidean", "jaccard", etc.).
    - title: str
        Plot title.
    - filename: str
        If given, saves the dendrogram to this base filename (png, svg, pdf).
    - dpi: int
        Resolution for saved image.
    - show: bool
        Whether to display the plot.
    """


    # Ensure labels are strings
    group_matrix.columns = group_matrix.columns.astype(str)

    # Compute pairwise distances between columns (groups)
    from scipy.spatial.distance import pdist
    dist_matrix = pdist(group_matrix.T, metric=metric)
    linkage = sch.linkage(dist_matrix, method=method)

    fig, ax = plt.subplots(figsize=(10, 6))
    sch.dendrogram(linkage, labels=group_matrix.columns.tolist(), leaf_rotation=90, leaf_font_size=10, ax=ax)

    ax.set_ylabel("Distance")
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if filename:
        save_plot(filename, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)


def plot_top_items_by_group(df: pd.DataFrame,
                             top_n: int = 5,
                             value_column_pattern: str = "Number of documents",
                             title: str = None,
                             filename: str = None,
                             dpi: int = 600,
                             group_color: dict = None,
                             show_values: bool = True,
                             reverse_order: bool = False,
                             show: bool = True):
    """
    Plots a horizontal bar chart showing top N items per group.

    Parameters:
    - df: pd.DataFrame
        DataFrame with item names in first column and one or more group-specific value columns.
    - top_n: int
        Number of top items to show per group (default: 5).
    - value_column_pattern: str
        Pattern prefix of value columns to plot (default: "Number of documents").
    - title: str
        Title of the plot.
    - filename: str
        Base filename to save the figure (PNG, SVG, PDF).
    - dpi: int
        Save resolution.
    - group_color: dict
        Optional color mapping for groups.
    - show_values: bool
        If True, annotate bars with their values.
    - reverse_order: bool
        If True, reverse the sort order of bars.
    - show: bool
        Whether to display the figure.
    """


    item_col = df.columns[0]
    value_cols = [col for col in df.columns if col.startswith(value_column_pattern)]

    records = []
    for col in value_cols:
        group = col.split("(")[-1].rstrip(")").strip()
        temp_df = df[[item_col, col]].copy()
        temp_df.columns = ["Item", "Value"]
        temp_df["Group"] = group
        temp_df = temp_df[temp_df["Value"] > 0]
        top_items = temp_df.sort_values("Value", ascending=False).head(top_n)

        # Handle possible ties
        min_val = top_items["Value"].min()
        all_top = temp_df[temp_df["Value"] >= min_val]
        all_top = all_top.copy()
        all_top["RankGroup"] = group
        records.append(all_top)

    plot_df = pd.concat(records)
    plot_df = plot_df.sort_values(by=["RankGroup", "Value"], ascending=[True, not reverse_order])

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(plot_df) + 1))

    # Assign default group colors if none provided
    if group_color is None:
        palette = itertools.cycle(cm.tab10.colors)
        unique_groups = plot_df["Group"].unique()
        group_color = {group: next(palette) for group in unique_groups}

    colors = plot_df["Group"].map(group_color)

    counts = Counter(plot_df["Item"])
    labels = []
    seen = Counter()
    for item in plot_df["Item"]:
        label = item
        if counts[item] > 1:
            label += " " * seen[item]  # add space padding for repeated items
            seen[item] += 1
        labels.append(label)
    bars = ax.barh(labels, plot_df["Value"], color=colors)

    if show_values:
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f"{width:.0f}", va="center", ha="left", fontsize=9)

    ax.set_xlabel(value_column_pattern)
    if title:
        ax.set_title(title)

    # Add legend
    handles = [plt.Line2D([0], [0], color=group_color[g], lw=6) for g in group_color]
    ax.legend(handles, group_color.keys(), title="Groups")

    plt.tight_layout()

    if filename:
        save_plot(filename, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)
    
code_to_coords = utilsbib.code_to_coords

def save_plotly_choropleth_map(
    df,
    value_col,
    filename_prefix=None,
    iso_col="ISO-3",
    width=1000,
    height=600,
    scale=2,
    title=None,
    projection="natural earth",
    scope="world",
    font_size=12,
    title_font_size=16,
    dark_mode=False,
    hover_name=None,
    hover_data=None,
    colormap="Viridis",
    links_df=None,
    link_weight_col="weight",
    link_color="red",
    link_opacity=0.5,
    timeout_seconds=30,
    continent_col="Continent",
    **kwargs
):
    """
    Generate and optionally save a choropleth map in PDF, PNG, SVG, and HTML formats using Plotly,
    with optional inter-country collaboration lines, using fig.to_image() to avoid blocking.
    If a column specifying continent exists and a scope is provided (other than "world"),
    filter the DataFrame to that continent before plotting.

    Args:
        df (pd.DataFrame): DataFrame containing country data.
        value_col (str): Column name with values to visualize.
        filename_prefix (str or None): Prefix for saved file names. If None, the plot is shown but not saved.
        iso_col (str): Column name with ISO-3 country codes.
        width (int): Width of the saved image.
        height (int): Height of the saved image.
        scale (int): Scale factor for image resolution.
        title (str): Optional title for the map.
        projection (str): Map projection (e.g., "natural earth", "equirectangular").
        scope (str): Geographic scope to focus the map (e.g., "world", "europe", "asia").
        font_size (int): Font size for general map text.
        title_font_size (int): Font size for the map title.
        dark_mode (bool): Use dark template if True.
        hover_name (str): Column to use as hover label.
        hover_data (list): List of columns to display on hover.
        colormap (str): Colormap name for continuous data.
        links_df (pd.DataFrame): DataFrame with "source", "target", and weight column for collaboration links.
        link_weight_col (str): Column name representing collaboration strength.
        link_color (str): Color of the collaboration lines.
        link_opacity (float): Opacity of the collaboration lines.
        timeout_seconds (int): Max seconds to wait per image format.
        continent_col (str): Name of the continent column (default: "continent").
    **kwargs: Additional keyword arguments for px.choropleth.
    """


    template = "plotly_dark" if dark_mode else "plotly"

    # Check for continent_col and filter
    col_match = None
    for col in df.columns:
        if col.lower() == continent_col.lower():
            col_match = col
            break

    if col_match and scope.lower() != "world":
        matching = df[col_match].astype(str).str.lower() == scope.lower()
        if matching.any():
            df = df[matching]
        else:
            warnings.warn(
                f"Scope '{scope}' provided but no matching values found in '{col_match}' column. Proceeding without filtering."
            )

    fig = px.choropleth(
        df,
        locations=iso_col,
        color=value_col,
        locationmode="ISO-3",
        color_continuous_scale=colormap,
        title=title,
        hover_name=hover_name,
        hover_data=hover_data,
        projection=projection,
        scope=scope,
        template=template,
        **kwargs
    )

    # Collaboration links (assumes code_to_coords dict is in global scope)
    if links_df is not None:
        for _, row in links_df.iterrows():
            src, tgt = row["source"], row["target"]
            if src not in code_to_coords or tgt not in code_to_coords:
                continue
            lat0, lon0 = code_to_coords[src]["latitude"], code_to_coords[src]["longitude"]
            lat1, lon1 = code_to_coords[tgt]["latitude"], code_to_coords[tgt]["longitude"]
            fig.add_trace(
                go.Scattergeo(
                    lon=[lon0, lon1],
                    lat=[lat0, lat1],
                    mode="lines",
                    line=dict(width=row[link_weight_col], color=link_color),
                    opacity=link_opacity,
                    showlegend=False
                )
            )

    fig.update_layout(
        font=dict(size=font_size),
        title_font=dict(size=title_font_size),
        coloraxis_colorbar=dict(
            title=dict(text=value_col, font=dict(size=font_size + 2)),
            ticks="outside",
            ticklen=5,
            tickcolor="#000",
            tickfont=dict(size=font_size),
        )
    )

    # Save or show
    if filename_prefix:
        for fmt in ("pdf", "png", "svg"):
            try:
                img_bytes = fig.to_image(
                    format=fmt,
                    width=width,
                    height=height,
                    scale=scale,
                    engine="kaleido"
                )
                with open(f"{filename_prefix}.{fmt}", "wb") as f:
                    f.write(img_bytes)
            except Exception as e:
                print(f"Error saving {fmt.upper()}: {e}")

        try:
            fig.write_html(f"{filename_prefix}.html")
        except Exception as e:
            print(f"Error saving HTML: {e}")
    else:
        fig.show()

    return fig

# Country collaboration plots

def plot_top_country_pairs(matrix_df, top_n=20, figsize=(10, 6), filename_base=None):
    """
    Plots a horizontal barplot of the top N collaborating country pairs, including ties at the cutoff.

    Parameters:
    matrix_df (pd.DataFrame): Symmetric collaboration matrix.
    top_n (int): Minimum number of top collaborating pairs to plot (ties at the cutoff are included).
    figsize (tuple): Size of the figure in inches.
    filename_base (str or None): If provided, saves the plot as PNG, SVG, and PDF using this base name.
    """
    if matrix_df.empty:
        print("Empty matrix: barplot not generated.")
        return

    pair_data = []
    for i in matrix_df.index:
        for j in matrix_df.columns:
            if i < j:
                count = matrix_df.loc[i, j]
                if count > 0:
                    pair_data.append((f"{i} – {j}", count))

    if not pair_data:
        print("No collaboration pairs found: barplot not generated.")
        return

    # Sort and find cutoff for ties
    pair_data_sorted = sorted(pair_data, key=lambda x: x[1], reverse=True)
    if len(pair_data_sorted) > top_n:
        cutoff_value = pair_data_sorted[top_n - 1][1]
        top_pairs = [pair for pair in pair_data_sorted if pair[1] >= cutoff_value]
    else:
        top_pairs = pair_data_sorted

    labels, values = zip(*top_pairs)

    plt.figure(figsize=figsize)
    plt.barh(labels, values)
    plt.xlabel("Collaboration Count")
    plt.title(f"Top Country Collaborations (≥ Top {top_n} with ties)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base)

    plt.show()


# production over time

def plot_group_dot_grid(df,
                        result_type="wide",
                        value_column="Number of documents",
                        color_column=None,
                        top_n=10,
                        year_column="Year",
                        group_column="Group",
                        figsize=(12, 8),
                        cmap="viridis",
                        output_path=None,
                        dpi=300,
                        max_dot_size=600,
                        font_size=14,
                        wrap_labels=True,
                        ylabel_label="Group",
                        year_span=None):
    """
    Plot a dot grid showing group activity over time with dot size = value and color = additional metric.

    Parameters
    ----------
    df : pd.DataFrame
        Output of aggregate_bibliometrics_by_group_and_year (wide or long format).
    result_type : str, default "wide"
        Type of df provided: "wide" or "long".
    value_column : str
        Column to determine dot size (usually "Number of documents").
    color_column : str or None
        Optional column to control dot color (e.g., citations, funding).
    top_n : int
        Show top N groups based on total of value_column.
    year_column : str
        Column indicating time axis.
    group_column : str
        Column indicating grouping (one row per group).
    figsize : tuple
        Size of the figure.
    cmap : str
        Matplotlib colormap name.
    output_path : str or None
        If provided, save the figure to this path.
    dpi : int
        Resolution for saved figure.
    max_dot_size : int
        Maximum dot area size in points^2.
    font_size : int
        Font size for labels.
    wrap_labels : bool
        If True, wrap long y-axis labels.
    ylabel_label : str
        Label for the y-axis.
    year_span : tuple or None
        Optional (start_year, end_year) to define time span. If None, use actual range based on non-zero values.

    Returns
    -------
    None
    """
    if result_type == "long":
        if not {"Metric", "Value"}.issubset(df.columns):
            raise ValueError("Expected columns 'Metric' and 'Value' in long format.")
        df = df.pivot_table(index=[year_column, group_column], columns="Metric", values="Value").reset_index()

    # Determine year span from non-zero values if not provided
    if year_span is None:
        nonzero_df = df[df[value_column] > 0]
        min_year = int(nonzero_df[year_column].min())
        max_year = int(nonzero_df[year_column].max())
        year_span = (min_year, max_year)

    # Apply year filtering early
    df = df[(df[year_column] >= year_span[0]) & (df[year_column] <= year_span[1])]

    # Filter top N groups by value_column
    group_totals = df.groupby(group_column)[value_column].sum().nlargest(top_n)
    df = df[df[group_column].isin(group_totals.index)]

    # Map groups to vertical positions (reverse order)
    group_order = group_totals.index.tolist()[::-1]
    group_pos = {name: i for i, name in enumerate(group_order)}
    df["_y"] = df[group_column].map(group_pos)

    # Scale dot sizes
    max_value = df[value_column].max()
    df["_dot_size"] = df[value_column] / max_value * max_dot_size

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.set_facecolor("white")

    # Normalize color if needed
    norm = None
    sm = None
    if color_column:
        norm = plt.Normalize(df[color_column].min(), df[color_column].max())
        cmap_instance = cm.get_cmap(cmap)
        sm = plt.cm.ScalarMappable(cmap=cmap_instance, norm=norm)
        sm.set_array([])
        color_values = df[color_column]
    else:
        color_values = "tab:blue"

    # Plot lines and points
    for group, group_df in df.groupby(group_column):
        x = group_df[year_column]
        y = group_df["_y"]
        size = group_df["_dot_size"]
        color = (group_df[color_column] if color_column else "tab:blue")
        ax.plot(x, y, color="black", linewidth=0.5, zorder=1)
        ax.scatter(x, y, s=size, c=(cmap_instance(norm(color)) if color_column else color), cmap=cmap,
                   norm=norm, edgecolors="black", zorder=2)

    # Y ticks and labels
    y_labels = group_order
    if wrap_labels:
        y_labels = ["\n".join(textwrap.wrap(label, 20)) for label in y_labels]
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(y_labels, fontsize=font_size)
    ax.set_ylabel(ylabel_label, fontsize=font_size)

    # Set x ticks and labels from year_span
    years = list(range(year_span[0], year_span[1] + 1))
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=font_size, rotation=90)
    ax.set_xlabel("Year", fontsize=font_size)

    # Optional colorbar
    if sm is not None:
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(color_column, fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi)
    plt.show()


def dplot_item_timelines(
    df: pd.DataFrame,
    min_docs: int = 3,
    regex_filter: str = None,
    top_n_year: int = 3,
    color_by: str = None,
    item_col: str = "Item",
    figsize: tuple = (10, 6),
    dpi: int = 600,
    filename: str = None,
    title: str = None,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    median_rounding: str = None,  # Options: None, "floor", "ceil"
    color_scheme: str = "auto",   # Options: "auto", "viridis", "lightblue"
    log_size: bool = True         # If True, dot sizes are on logarithmic scale
):
    """
    Plot item timelines based on their Q1-median-Q3 year range and document count.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least the following columns:
        item_col (default: "Item"), "Q1 year", "Median year", "Q3 year", "Number of documents".
        Optionally, a `color_by` column (e.g., "Cited by") can be used for coloring.
    min_docs : int, default=5
        Minimum number of documents for an item to be included.
    regex_filter : str, optional
        Regular expression to filter items.
    top_n_year : int, default=3
        Number of top items (by document count) to display per median year.
    color_by : str, optional
        Column name for coloring (e.g., "Cited by").
    item_col : str, default="Item"
        Column name for item/category labels.
    figsize : tuple, default=(10, 6)
        Figure size in inches.
    dpi : int, default=600
        Resolution of the saved plot.
    filename : str, optional
        Base name for saving the plot (PNG, SVG, PDF).
    title : str, optional
        Title of the plot.
    title_fontsize, label_fontsize, tick_fontsize : int
        Font sizes for title, labels, and ticks.
    median_rounding : str, optional
        How to round "Median year" ("floor" or "ceil").
    color_scheme : str, default="auto"
        "lightblue" = uniform color, "viridis" = gradient by `color_by`, "auto" decides automatically.
    log_size : bool, default=True
        If True, dot sizes are scaled logarithmically by document count.

    Returns
    -------
    None
    """
    # Filtering
    filtered = df[df["Number of documents"] >= min_docs].copy()
    if regex_filter:
        filtered = filtered[filtered[item_col].str.contains(regex_filter, flags=re.IGNORECASE, regex=True)]

    # Optional rounding of median year
    if median_rounding == "floor":
        filtered["Median year"] = np.floor(filtered["Median year"]).astype(int)
    elif median_rounding == "ceil":
        filtered["Median year"] = np.ceil(filtered["Median year"]).astype(int)

    # Group by median year and select top N
    grouped = (
        filtered.sort_values(["Median year", "Number of documents"], ascending=[True, False])
        .groupby("Median year")
        .head(top_n_year)
    ).reset_index(drop=True)

    grouped["y_pos"] = range(len(grouped))

    # Determine colors
    use_colorbar = False
    if color_scheme == "lightblue" or (color_scheme == "auto" and not color_by):
        colors = ["lightblue"] * len(grouped)
    elif color_scheme == "viridis" or (color_scheme == "auto" and color_by and color_by in grouped.columns):
        norm = mcolors.Normalize(vmin=grouped[color_by].min(), vmax=grouped[color_by].max())
        cmap = cm.viridis
        colors = cmap(norm(grouped[color_by].values))
        use_colorbar = True
    else:
        colors = ["lightblue"] * len(grouped)

    # Compute sizes (logarithmic or linear)
    if log_size:
        # avoid log(0); add 1
        sizes = np.log1p(grouped["Number of documents"]) * 50
    else:
        sizes = grouped["Number of documents"] * 10

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.set_facecolor("white")

    for i, row in grouped.iterrows():
        if pd.notnull(row.get("Q1 year")) and pd.notnull(row.get("Q3 year")):
            ax.plot([row["Q1 year"], row["Q3 year"]], [row["y_pos"]] * 2, color="gray", linewidth=1)
        ax.scatter(row["Median year"], row["y_pos"],
                   s=sizes[i],
                   color=colors[i],
                   edgecolor="black")

    ax.set_yticks(grouped["y_pos"])
    ax.set_yticklabels(grouped[item_col], fontsize=tick_fontsize)
    ax.set_xlabel("Year", fontsize=label_fontsize)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    if use_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(color_by, fontsize=label_fontsize)

    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.grid(False)
    plt.tight_layout()

    if log_size and fig is not None:
        fig.text(0.99, 0.01, "Dot sizes represent the logarithmic scale\n of the number of documents.",
                 ha="right", va="bottom", fontsize=10, color="black")

    if filename:
        save_plot(filename, dpi=dpi)
    plt.show()


def plot_topic_visualization(
    *,
    kind: str,
    df_out: Optional[pd.DataFrame] = None,
    topics_df: Optional[pd.DataFrame] = None,
    # Column mapping (auto-detected if None)
    doc_col: Optional[str] = None,
    topic_col: Optional[str] = None,
    doc_weight_col: Optional[str] = None,  # used only if no per-topic weight cols exist
    time_col: Optional[str] = None,
    term_col: Optional[str] = None,
    term_weight_col: Optional[str] = None,
    # Options
    topic_id: Optional[Any] = None,
    top_n_terms: int = 15,
    top_n_topics: int = 10,
    max_docs: int = 200,
    normalize: bool = True,
    figsize: Tuple[float, float] = (9, 6),
    cmap: str = "viridis",
    title: Optional[str] = None,
    grid: bool = False,
    filename_base: Optional[str] = None,
    dpi: int = 600,
    # Never renumber topics; optional display offset (visual only)
    topic_label_offset: int = 0,
    # Extra options for "topic_words_bar"
    col_wrap: int = 3,
    show_labels: bool = True,
    color_column: Optional[str] = None,
    palette_discrete: Union[str, Dict[Any, Any]] = "lightblue",
    palette_continuous: str = "viridis",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Topic-model visualizations **without renumbering topics** and compatible with the simplified pipeline.

    Kinds
    -----
    kind : {"topic_term_barchart","topic_term_heatmap","doc_topic_heatmap",
            "topic_prevalence_over_time","intertopic_map","topic_words_bar"}

    Assumptions
    -----------
    - `df_out` is what `get_topics(...)` returns (all original columns + "Topic",
      and optionally per-topic columns like "Topic 1 Weight", ..., plus any "Doc Weight").
    - `topics_df` is the long table with ["Topic", "Term"/"Word", "Weight"].

    Parameters
    ----------
    df_out, topics_df : pd.DataFrame | None
        Provide whichever the plot needs (see kinds above).
    doc_col, topic_col, doc_weight_col, time_col, term_col, term_weight_col : str | None
        Column overrides; if None, they are auto-detected.
        Note: `doc_weight_col` is only used when no per-topic weight columns exist in `df_out`.
    topic_id : Any | None
        For "topic_term_barchart": which topic to display; if None, the strongest is auto-picked.
    top_n_terms, top_n_topics, max_docs : int
        Limits for plots.
    normalize : bool
        For "topic_prevalence_over_time": normalize each time slice to sum to 1 (stacked shares).
    figsize : tuple, cmap : str, title : str | None, grid : bool, filename_base : str | None, dpi : int
        Standard plotting options.
    topic_label_offset : int
        Purely visual offset for tick labels (e.g., 1 to display 1-based labels) — data is untouched.
    col_wrap, show_labels, color_column, palette_discrete, palette_continuous
        Extra options used only by "topic_words_bar".

    Returns
    -------
    (fig, ax)
    """

    from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    def _pick_col(df: pd.DataFrame, explicit: Optional[str], candidates: List[str], label: str) -> str:
        if explicit is not None:
            if explicit in df.columns:
                return explicit
            raise ValueError(f'{label} "{explicit}" not in columns: {list(df.columns)}')
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            f"`{df.__class__.__name__}` is missing required column for {label}. "
            f"Tried: {candidates}. Available: {list(df.columns)}"
        )

    def _fmt_topic_labels(labels: Sequence[Any]) -> List[str]:
        out: List[str] = []
        for l in labels:
            if isinstance(l, (int, np.integer)) and topic_label_offset != 0:
                out.append(str(int(l) + topic_label_offset))
            else:
                out.append(str(l))
        return out

    def _melt_doc_topic(
        df: pd.DataFrame,
        _doc_col: str,
        _topic_col: str,
        _doc_weight_col: Optional[str],
        _time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create a long doc–topic table with columns:
            [_doc_col, (time_col if provided), "Topic", "Value"].

        Preference:
        - If per-topic columns like "Topic X Weight" exist: melt them (uses those weights).
        - Else: use hard assignment from `_topic_col`, with values from `_doc_weight_col` (or 1.0).
        """
        pat = re.compile(r"^Topic\s+(.+?)\s+Weight$")
        wcols = [c for c in df.columns if pat.match(str(c))]

        id_vars = [_doc_col]
        if _time_col is not None and _time_col in df.columns:
            id_vars.append(_time_col)

        if wcols:
            m = df[id_vars + wcols].melt(id_vars=id_vars, var_name="_col", value_name="Value")
            m["Topic"] = m["_col"].str.replace(r"^Topic\s+(.+?)\s+Weight$", r"\\1", regex=True)
            m["Topic"] = m["Topic"].apply(lambda s: int(s) if str(s).isdigit() else s)
            m = m.drop(columns="_col")
            return m
        else:
            if _topic_col not in df.columns:
                raise ValueError(f'"{_topic_col}" not found; cannot build doc–topic table.')
            if _doc_weight_col is not None and _doc_weight_col in df.columns:
                val = pd.to_numeric(df[_doc_weight_col], errors="coerce").fillna(1.0).to_numpy()
            else:
                val = np.ones(len(df), dtype=float)
            out = {
                _doc_col: df[_doc_col].to_numpy(),
                "Topic": df[_topic_col].to_numpy(),
                "Value": val,
            }
            if _time_col is not None and _time_col in df.columns:
                out[_time_col] = df[_time_col].to_numpy()
            return pd.DataFrame(out)

    def _resolve_cols_from_inputs() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        nonlocal doc_col, topic_col, time_col, term_col, term_weight_col
        if df_out is not None and not df_out.empty:
            doc_col_ = _pick_col(df_out, doc_col, ["Doc ID", "DocID", "ID", "Document", "doc_id", "doc"], "doc_col")
            topic_col_ = _pick_col(df_out, topic_col, ["Topic", "topic", "Topic ID", "topic_id", "Label"], "topic_col")
            if time_col is None:
                time_col_ = next((c for c in ["Year", "year", "Time", "time", "Date", "date"] if c in df_out.columns), None)
            else:
                time_col_ = time_col
        else:
            doc_col_, topic_col_, time_col_ = doc_col, topic_col, time_col

        if topics_df is not None and not topics_df.empty:
            topic_col_topics = _pick_col(topics_df, topic_col, ["Topic", "topic", "Topic ID", "topic_id", "Label"], "topic_col (topics_df)")
            term_col_ = _pick_col(topics_df, term_col, ["Term", "term", "Word", "word", "Token", "token"], "term_col")
            term_weight_col_ = _pick_col(
                topics_df,
                term_weight_col,
                ["Weight", "weight", "Beta", "beta", "Probability", "probability", "Relevance", "relevance", "Score", "score"],
                "term_weight_col",
            )
            topic_col_ = topic_col_ or topic_col_topics
        else:
            term_col_, term_weight_col_ = term_col, term_weight_col

        return doc_col_, topic_col_, time_col_, term_col_, term_weight_col_, topic_col_

    # Resolve columns
    doc_col, topic_col, time_col, term_col, term_weight_col, _ = _resolve_cols_from_inputs()

    # --- New kind: topic_words_bar (faceted horizontal bars) ---
    if kind == "topic_words_bar":
        if topics_df is None or topics_df.empty:
            raise ValueError("`topics_df` is required for 'topic_words_bar'.")
        df = topics_df.copy()

        # Robust topic ordering: integers, "Topic 7", etc.
        def _tkey(x):
            s = str(x)
            num = "".join(ch for ch in s if ch.isdigit())
            return (0, int(num)) if num.isdigit() else (1, s.lower())

        uniq_topics = sorted(df[topic_col].unique(), key=_tkey)
        df[topic_col] = pd.Categorical(df[topic_col], categories=uniq_topics, ordered=True)

        # Top-N per topic
        top_words = (
            df.sort_values([topic_col, term_weight_col], ascending=[True, False])
              .groupby(topic_col, group_keys=False)
              .head(top_n_terms)
        )

        # Build facet grid
        g = sns.catplot(
            data=top_words,
            x=term_weight_col,
            y=term_col,
            col=topic_col,
            kind="bar",
            col_wrap=col_wrap,
            height=3.0,  # facet height (auto total size below)
            aspect=1.0,
            sharex=False,
            sharey=False,
            color=palette_discrete if color_column is None else None,
            errwidth=0,
        )

        # Optional coloring
        if color_column and color_column in top_words.columns:
            sample = top_words[color_column].dropna()
            is_num = pd.api.types.is_numeric_dtype(sample)
            if is_num:
                norm = mcolors.Normalize(vmin=float(sample.min()), vmax=float(sample.max()))
                cmap_obj = cm.get_cmap(palette_continuous)
                # color bars inside each facet using that facet's data
                for ax, t in zip(g.axes.flatten(), g.col_names):
                    fdf = top_words[top_words[topic_col] == t]
                    lab2val = dict(zip(fdf[term_col], fdf[color_column]))
                    ylabels = [lab.get_text() for lab in ax.get_yticklabels()]
                    for p in ax.patches:
                        # find label index by bar y-center
                        idx = int(round(p.get_y() + p.get_height() / 2))
                        if 0 <= idx < len(ylabels):
                            val = lab2val.get(ylabels[idx])
                            if val is not None:
                                p.set_facecolor(cmap_obj(norm(val)))
                # add colorbar
                sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
                sm.set_array([])
                cbar_ax = g.fig.add_axes([0.92, 0.3, 0.02, 0.4])
                g.fig.colorbar(sm, cax=cbar_ax, label=color_column)
            else:
                # discrete palette (string name or explicit dict)
                if isinstance(palette_discrete, dict):
                    color_map = palette_discrete
                else:
                    cats = top_words[color_column].dropna().unique()
                    pal = sns.color_palette(palette_discrete, n_colors=len(cats))
                    color_map = dict(zip(cats, pal))
                for ax, t in zip(g.axes.flatten(), g.col_names):
                    fdf = top_words[top_words[topic_col] == t]
                    lab2cat = dict(zip(fdf[term_col], fdf[color_column]))
                    ylabels = [lab.get_text() for lab in ax.get_yticklabels()]
                    for p in ax.patches:
                        idx = int(round(p.get_y() + p.get_height() / 2))
                        if 0 <= idx < len(ylabels):
                            cat = lab2cat.get(ylabels[idx])
                            if cat is not None:
                                p.set_facecolor(color_map.get(cat, palette_discrete))

        # Titles, labels, size
        g.set_titles("{col_name}")
        g.set_axis_labels("Weight", "Word")

        # Size: derive from number of topics and col_wrap; allow figsize override
        n_topics = len(uniq_topics)
        used_cols = max(1, min(col_wrap, n_topics))
        n_rows = math.ceil(n_topics / used_cols)
        if figsize is not None:
            g.fig.set_size_inches(figsize)
        else:
            g.fig.set_size_inches(max(6.0, used_cols * 3.6), max(2.8, n_rows * 3.2))

        # Optional value labels
        if show_labels:
            for ax in g.axes.flatten():
                x0, x1 = ax.get_xlim()
                span = max(1e-12, x1 - x0)
                for p in ax.patches:
                    w = p.get_width()
                    ax.text(
                        w + 0.01 * span,
                        p.get_y() + p.get_height() / 2,
                        f"{w:.2f}",
                        va="center",
                    )

        g.tight_layout(pad=1.0, rect=[0, 0, 0.9, 1])
        if title:
            g.fig.suptitle(title, y=1.02)

        if filename_base:
            save_plot(filename_base, dpi=dpi)

        return g.fig, g.axes.flatten()[0]

    # ---------------------------------------------------------------------
    # Other kinds (matplotlib-based)
    fig, ax = plt.subplots(figsize=figsize)

    if kind == "topic_term_barchart":
        if topics_df is None or topics_df.empty:
            raise ValueError("`topics_df` is required for \"topic_term_barchart\".")
        use_topic = topic_id
        if use_topic is None:
            use_topic = (
                topics_df.groupby(topic_col)[term_weight_col]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
        sub = (
            topics_df.loc[topics_df[topic_col] == use_topic, [term_col, term_weight_col]]
            .sort_values(term_weight_col, ascending=False)
            .head(top_n_terms)
            .iloc[::-1]
        )
        ax.barh(sub[term_col], sub[term_weight_col], alpha=0.9)
        ax.set_xlabel(term_weight_col)
        ax.set_ylabel("Terms")
        title_topic = _fmt_topic_labels([use_topic])[0]
        ax.set_title(title or f"Top {len(sub)} Terms for Topic {title_topic}")
        if grid:
            ax.grid(axis="x", alpha=0.3)

    elif kind == "topic_term_heatmap":
        if topics_df is None or topics_df.empty:
            raise ValueError("`topics_df` is required for \"topic_term_heatmap\".")
        top_terms = (
            topics_df.sort_values([topic_col, term_weight_col], ascending=[True, False])
            .groupby(topic_col)
            .head(top_n_terms)
        )
        top_topics = (
            topics_df.groupby(topic_col)[term_weight_col].sum().sort_values(ascending=False).head(top_n_topics).index
        )
        top_terms = top_terms[top_terms[topic_col].isin(top_topics)]
        piv = top_terms.pivot(index=topic_col, columns=term_col, values=term_weight_col).fillna(0.0)
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap)
        ax.set_yticks(np.arange(piv.shape[0]), labels=_fmt_topic_labels(list(piv.index)))
        xticks = np.arange(piv.shape[1])
        step = max(1, math.ceil(piv.shape[1] / 20))
        ax.set_xticks(xticks[::step], labels=[str(c) for c in piv.columns[::step]], rotation=60, ha="right")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Topics")
        ax.set_title(title or "Topic–Term Heatmap")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(term_weight_col)
        if grid:
            ax.grid(False)

    elif kind == "doc_topic_heatmap":
        if df_out is None or df_out.empty:
            raise ValueError("`df_out` is required for \"doc_topic_heatmap\".")
        DT = _melt_doc_topic(df_out, doc_col, topic_col, doc_weight_col, None)
        top_topics = DT.groupby("Topic")["Value"].sum().sort_values(ascending=False).head(top_n_topics).index
        sub = DT[DT["Topic"].isin(top_topics)].copy()
        top_docs = sub.groupby(doc_col)["Value"].sum().sort_values(ascending=False).head(max_docs).index
        sub = sub[sub[doc_col].isin(top_docs)]
        piv = sub.pivot(index=doc_col, columns="Topic", values="Value").fillna(0.0)
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap)
        yticks = np.arange(piv.shape[0])
        ystep = max(1, math.ceil(piv.shape[0] / 30))
        ax.set_yticks(yticks[::ystep], labels=[str(d) for d in piv.index[::ystep]])
        ax.set_xticks(np.arange(piv.shape[1]), labels=_fmt_topic_labels(list(piv.columns)), rotation=60, ha="right")
        ax.set_xlabel("Topics")
        ax.set_ylabel("Documents")
        ax.set_title(title or "Document–Topic Heatmap")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Value")
        if grid:
            ax.grid(False)

    elif kind == "topic_prevalence_over_time":
        if df_out is None or df_out.empty:
            raise ValueError("`df_out` is required for \"topic_prevalence_over_time\".")
        if time_col is None or time_col not in df_out.columns:
            raise ValueError("`df_out` must contain a time column (e.g., \"Year\") or pass `time_col` explicitly.")
        DT = _melt_doc_topic(df_out, doc_col, topic_col, doc_weight_col, time_col)
        g = DT.groupby([time_col, "Topic"])["Value"].sum().reset_index()
        top_topics = g.groupby("Topic")["Value"].sum().sort_values(ascending=False).head(top_n_topics).index
        g["TopicFiltered"] = np.where(g["Topic"].isin(top_topics), g["Topic"], "Other")
        agg = g.groupby([time_col, "TopicFiltered"])["Value"].sum().reset_index()
        years = sorted(agg[time_col].unique())
        topics = list(agg["TopicFiltered"].unique())
        mat = np.zeros((len(years), len(topics)), dtype=float)
        for i, y in enumerate(years):
            row = agg[agg[time_col] == y]
            for j, t in enumerate(topics):
                val = row.loc[row["TopicFiltered"] == t, "Value"]
                mat[i, j] = float(val.iloc[0]) if not val.empty else 0.0
        if normalize:
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            mat = mat / row_sums
        order = np.argsort(-mat.sum(axis=0))
        topics = [topics[k] for k in order]
        mat = mat[:, order]
        ax.stackplot(years, mat.T, labels=_fmt_topic_labels(topics))
        ax.set_xlim(min(years), max(years))
        ax.set_xlabel(time_col)
        ax.set_ylabel("Share" if normalize else "Total weight")
        ax.set_title(title or "Topic Prevalence Over Time")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        if grid:
            ax.grid(axis="y", alpha=0.3)

    elif kind == "intertopic_map":
        if topics_df is None or topics_df.empty:
            raise ValueError("`topics_df` is required for \"intertopic_map\".")
        piv = topics_df.pivot(index=topic_col, columns=term_col, values=term_weight_col).fillna(0.0)
        mat = piv.values
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat_norm = mat / row_sums
        try:
            from sklearn.decomposition import PCA  # type: ignore
            coords = PCA(n_components=2, random_state=0).fit_transform(mat_norm)
        except Exception:
            X = mat_norm - mat_norm.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            coords = U[:, :2] * S[:2]
        ax.scatter(coords[:, 0], coords[:, 1], s=60, alpha=0.8)
        disp_labels = _fmt_topic_labels(list(piv.index))
        for i, lbl in enumerate(disp_labels):
            ax.text(coords[i, 0], coords[i, 1], lbl, ha="center", va="center", fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title or "Intertopic Distance Map (PCA on β)")
        if grid:
            ax.grid(alpha=0.3)

    else:
        raise ValueError(
            "Unknown kind. Choose one of: "
            "topic_term_barchart, topic_term_heatmap, doc_topic_heatmap, "
            "topic_prevalence_over_time, intertopic_map, topic_words_bar."
        )

    plt.tight_layout()
    if filename_base:
        save_plot(filename_base, dpi=dpi)
    return fig, ax


def plot_topic_words_bar(topic_df, top_n=10, col_wrap=3, width=12, height=8, show_labels=True, color_column=None, palette_discrete="lightblue", palette_continuous="viridis", save_path=None, dpi=600):
    """
    Plot bar charts of top words per topic based on their weights.

    Args:
        topic_df (pd.DataFrame): DataFrame with columns ["Topic", "Word"/"Term", "Weight"] and optionally a color_column.
        top_n (int): Number of top words to show per topic.
        col_wrap (int): Number of columns per row in the FacetGrid.
        width (int): Width of the entire figure.
        height (int): Height of each subplot.
        show_labels (bool): Whether to display weight labels on bars (default: True).
        color_column (str, optional): Column in topic_df used to color bars.
        palette_discrete (str or dict): Color or palette for discrete values (default: "lightblue").
        palette_continuous (str): Colormap for continuous values (default: "viridis").
        save_path (str, optional): If provided, the base path to save the figure in PNG, SVG, and PDF.
        dpi (int): Dots per inch for saving the figure (default: 600).

    Returns:
        None: Displays and optionally saves the plot.
    """
    # Make a copy to avoid modifying original
    topic_df = topic_df.copy()
    
    # Handle column name variations: "Term" -> "Word"
    if "Term" in topic_df.columns and "Word" not in topic_df.columns:
        topic_df = topic_df.rename(columns={"Term": "Word"})
    
    # Check required columns exist
    required_cols = ["Topic", "Word", "Weight"]
    missing = [c for c in required_cols if c not in topic_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(topic_df.columns)}")

    # Ensure topic sorting is consistent
    topic_df["Topic"] = pd.Categorical(
        topic_df["Topic"], 
        categories=sorted(topic_df["Topic"].unique(), key=lambda x: int(str(x).split()[-1]) if str(x).split()[-1].isdigit() else 0)
    )

    # Select top N words for each topic - use a simple approach to avoid pandas warnings
    top_words_list = []
    for topic in topic_df["Topic"].unique():
        topic_data = topic_df[topic_df["Topic"] == topic].nlargest(top_n, "Weight")
        top_words_list.append(topic_data)
    top_words = pd.concat(top_words_list, ignore_index=True)

    # Determine grid layout
    n_topics = topic_df["Topic"].nunique()
    used_cols = min(col_wrap, n_topics)
    n_rows = math.ceil(n_topics / used_cols)

    # Create barplot without hue, add colors manually
    g = sns.catplot(
        data=top_words,
        x="Weight",
        y="Word",
        col="Topic",
        kind="bar",
        col_wrap=col_wrap,
        height=height / col_wrap,
        aspect=1.0,
        sharex=False,
        sharey=False,
        color=palette_discrete if color_column is None else None
    )

    # Apply custom coloring
    if color_column and color_column in top_words.columns:
        sample_values = top_words[color_column].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(sample_values)

        if is_numeric:
            norm = mcolors.Normalize(vmin=sample_values.min(), vmax=sample_values.max())
            cmap = cm.get_cmap(palette_continuous)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            for ax in g.axes.flatten():
                for bar in ax.patches:
                    label = ax.get_yticklabels()[int(bar.get_y() + bar.get_height() / 2)].get_text()
                    value = top_words[top_words["Word"] == label][color_column].values[0]
                    bar.set_facecolor(cmap(norm(value)))

            # Move colorbar to the side without overlapping
            cbar_ax = g.fig.add_axes([0.92, 0.3, 0.02, 0.4])
            g.fig.colorbar(sm, cax=cbar_ax, label=color_column)

        else:
            color_dict = sns.color_palette(palette_discrete, n_colors=top_words[color_column].nunique())
            color_map = dict(zip(top_words[color_column].unique(), color_dict))

            for ax in g.axes.flatten():
                for bar in ax.patches:
                    label = ax.get_yticklabels()[int(bar.get_y() + bar.get_height() / 2)].get_text()
                    value = top_words[top_words["Word"] == label][color_column].values[0]
                    bar.set_facecolor(color_map.get(value, palette_discrete))

    g.set_titles("{col_name}")
    g.set_axis_labels("Weight", "Word")

    # Add padding for the colorbar/legend
    g.fig.set_size_inches(used_cols * 3.5, n_rows * 3.2)

    # Add labels if requested
    if show_labels:
        for ax in g.axes.flatten():
            for p in ax.patches:
                width = p.get_width()
                ax.text(width + 0.01, p.get_y() + p.get_height() / 2,
                        f"{width:.2f}", va="center")

    g.tight_layout(pad=1.0, rect=[0, 0, 0.9, 1])

    # Save if requested
    if save_path:
        save_plot(save_path, dpi=dpi)

    plt.show()


def plot_topic_distribution(
    df,
    value_column=None,
    palette="viridis",
    save_path=None,
    dpi=600,
    title=None,
    xlabel="Topic",
    ylabel="Number of Documents",
    show_labels=False,
    fontdict_title=None,
    fontdict_labels=None
):
    """
    Plot distribution of documents across topics.

    Args:
        df (pd.DataFrame): DataFrame with a 'Topic' column and optionally a value_column.
        value_column (str, optional): If provided, colors bars by average of this column per topic.
        palette (str): Name of the colormap to use if value_column is given (default: 'viridis').
        save_path (str, optional): If provided, saves the figure to this base path.
        dpi (int): Dots per inch for saving the figure (default: 600).
        title (str, optional): Title of the plot.
        xlabel (str): X-axis label (default: 'Topic').
        ylabel (str): Y-axis label (default: 'Number of Documents').
        show_labels (bool): Whether to show value labels above bars.
        fontdict_title (dict, optional): Font properties for title.
        fontdict_labels (dict, optional): Font properties for axis labels.

    Returns:
        None
    """


    topic_counts = df['Topic'].value_counts().sort_index()
    topic_order = topic_counts.index
    colors = "lightblue"

    fig, ax = plt.subplots(figsize=(8, 6))

    if value_column and value_column in df.columns:
        topic_means = df.groupby("Topic")[value_column].mean().reindex(topic_order)
        norm = mcolors.Normalize(vmin=topic_means.min(), vmax=topic_means.max())
        cmap = cm.get_cmap(palette)
        colors = [cmap(norm(val)) for val in topic_means]

        bars = ax.bar(topic_order, topic_counts.values, color=colors)

        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(f"Average {value_column}")
    else:
        bars = ax.bar(topic_order, topic_counts.values, color=colors)

    ax.set_xlabel(xlabel, fontdict=fontdict_labels)
    ax.set_ylabel(ylabel, fontdict=fontdict_labels)
    if title is not None:
        ax.set_title(title, fontdict=fontdict_title)

    if show_labels:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{int(height)}", ha="center", va="bottom")

    fig.tight_layout()

    if save_path:
        for ext in ["png", "svg", "pdf"]:
            path = f"{save_path}.{ext}"
            plt.savefig(path, bbox_inches="tight", dpi=dpi)

    plt.show()
    
def plot_topic_word_heatmap(topic_df, top_n=10, cmap="viridis", figsize=(12, 8), title="Topic-Word Weights Heatmap", save_path=None, dpi=600, colorbar_label="Weight", fontsize=10, title_fontsize=12, rotate=True, square=True):
    """
    Plot a heatmap of topic-word weights.

    Args:
        topic_df (pd.DataFrame): DataFrame with columns ["Topic", "Word", "Weight"].
        top_n (int): Number of top words to show per topic.
        cmap (str): Matplotlib colormap name (default: 'viridis').
        figsize (tuple): Figure size in inches (default: (12, 8)).
        title (str): Title of the plot.
        save_path (str, optional): If provided, saves the figure to this base path.
        dpi (int): Dots per inch for saving the figure (default: 600).
        colorbar_label (str): Label for the colorbar (default: 'Weight').
        fontsize (int): Font size for tick labels.
        title_fontsize (int): Font size for the plot title.
        rotate (bool): If True, rotate heatmap 90 degrees (topics on y-axis). Default is True.
        square (bool): Whether to draw square cells (default: True).

    Returns:
        None
    """


    # Select top N words per topic
    top_words = topic_df.groupby("Topic", group_keys=False).apply(
        lambda x: x.nlargest(top_n, "Weight")
    ).reset_index(drop=True)

    # Pivot table (rows: topics, columns: words, values: weights)
    heatmap_data = top_words.pivot(index="Topic", columns="Word", values="Weight").fillna(0)
    if not rotate:
        heatmap_data = heatmap_data.transpose()

    # Plot heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(heatmap_data, cmap=cmap, linewidths=0.5, linecolor="gray", cbar_kws={"label": colorbar_label}, square=square)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Words" if rotate else "Topics", fontsize=fontsize)
    ax.set_ylabel("Topics" if rotate else "Words", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()

    if save_path:
        for ext in ["png", "svg", "pdf"]:
            path = f"{save_path}.{ext}"
            plt.savefig(path, bbox_inches="tight", dpi=dpi)

    plt.show()


def plot_topic_coherence_scores(
    coherence_scores: Dict[int, float],
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Topic Coherence by Number of Topics",
    highlight_optimal: bool = True,
    color: str = "#3b82f6",
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot coherence scores for different numbers of topics.
    
    Parameters
    ----------
    coherence_scores : dict
        Mapping from n_topics to coherence score.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    highlight_optimal : bool
        Whether to highlight the optimal k.
    color : str
        Line/marker color.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ks = sorted(coherence_scores.keys())
    scores = [coherence_scores[k] for k in ks]
    
    ax.plot(ks, scores, marker='o', color=color, linewidth=2, markersize=8)
    
    if highlight_optimal and scores:
        optimal_k = ks[np.argmax(scores)]
        optimal_score = max(scores)
        ax.scatter([optimal_k], [optimal_score], color='red', s=150, zorder=5, 
                   label=f'Optimal: k={optimal_k}')
        ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.5)
        ax.legend()
    
    ax.set_xlabel("Number of Topics")
    ax.set_ylabel("Coherence Score")
    ax.set_title(title)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig, ax


def plot_topic_similarity_heatmap(
    similarity_df: pd.DataFrame,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = "RdYlBu_r",
    title: str = "Topic Similarity Matrix",
    annot: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot topic similarity matrix as heatmap.
    
    Parameters
    ----------
    similarity_df : pd.DataFrame
        Similarity matrix (topics x topics).
    figsize : tuple
        Figure size.
    cmap : str
        Colormap.
    title : str
        Plot title.
    annot : bool
        Whether to annotate cells with values.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        similarity_df, 
        cmap=cmap, 
        annot=annot, 
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Similarity"},
        ax=ax
    )
    
    ax.set_title(title)
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig, ax


def plot_topic_trends_area(
    trends_df: pd.DataFrame,
    figsize: Tuple[float, float] = (12, 6),
    cmap: str = "tab10",
    title: str = "Topic Prevalence Over Time",
    stacked: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot topic trends over time as stacked area chart.
    
    Parameters
    ----------
    trends_df : pd.DataFrame
        DataFrame with years as index and topics as columns.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for topics.
    title : str
        Plot title.
    stacked : bool
        Whether to stack areas.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(trends_df.columns)))
    
    if stacked:
        ax.stackplot(
            trends_df.index, 
            trends_df.T.values,
            labels=trends_df.columns,
            colors=colors,
            alpha=0.8
        )
    else:
        for i, col in enumerate(trends_df.columns):
            ax.plot(trends_df.index, trends_df[col], label=col, color=colors[i], linewidth=2)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion" if trends_df.max().max() <= 1 else "Count")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig, ax


def plot_topic_document_scatter(
    doc_topic_matrix: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'tsne',
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'tab10',
    title: str = 'Document-Topic Space',
    alpha: float = 0.6,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot documents in 2D topic space using dimensionality reduction.
    
    Parameters
    ----------
    doc_topic_matrix : np.ndarray
        Document-topic distribution (n_docs x n_topics).
    labels : np.ndarray, optional
        Topic labels for coloring.
    method : str, default='tsne'
        Dimensionality reduction: 'tsne', 'pca', or 'umap'.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap.
    title : str
        Plot title.
    alpha : float
        Point transparency.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    (fig, ax)
    """
    from sklearn.decomposition import PCA
    
    # Dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(doc_topic_matrix)
    elif method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        n_samples = doc_topic_matrix.shape[0]
        perplexity = min(30, max(5, n_samples // 4))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = reducer.fit_transform(doc_topic_matrix)
    elif method.lower() == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(doc_topic_matrix)
        except ImportError:
            # Fallback to PCA
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(doc_topic_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = doc_topic_matrix.argmax(axis=1) + 1
    
    unique_labels = np.unique(labels[~np.isnan(labels.astype(float))])
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            coords[mask, 0], 
            coords[mask, 1], 
            c=[colors[i]], 
            label=f'Topic {int(label)}',
            alpha=alpha,
            s=30
        )
    
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig, ax


def plot_topic_wordclouds(
    topics_df: pd.DataFrame,
    n_cols: int = 3,
    figsize_per_topic: Tuple[float, float] = (4, 3),
    colormap: str = 'viridis',
    background_color: str = 'white',
    max_words: int = 30,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    Plot word clouds for each topic.
    
    Parameters
    ----------
    topics_df : pd.DataFrame
        DataFrame with columns ["Topic", "Term"/"Word", "Weight"].
    n_cols : int
        Number of columns in grid.
    figsize_per_topic : tuple
        Size per topic subplot.
    colormap : str
        Colormap for word colors.
    background_color : str
        Background color.
    max_words : int
        Maximum words per cloud.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    fig
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        raise ImportError("Install wordcloud: pip install wordcloud")
    
    # Handle column name variations
    term_col = "Term" if "Term" in topics_df.columns else "Word"
    
    topics = topics_df["Topic"].unique()
    n_topics = len(topics)
    n_rows = int(np.ceil(n_topics / n_cols))
    
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(figsize_per_topic[0] * n_cols, figsize_per_topic[1] * n_rows)
    )
    
    if n_topics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, topic in enumerate(sorted(topics)):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        topic_data = topics_df[topics_df["Topic"] == topic]
        word_weights = dict(zip(topic_data[term_col], topic_data["Weight"]))
        
        if word_weights:
            wc = WordCloud(
                width=400, height=300,
                background_color=background_color,
                colormap=colormap,
                max_words=max_words,
            ).generate_from_frequencies(word_weights)
            
            ax.imshow(wc, interpolation='bilinear')
        
        ax.set_title(str(topic))
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_topics, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig


def plot_comparative_topic_wordclouds(
    topics_df: pd.DataFrame,
    topic_term_matrix: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    reference_topic: Optional[str] = None,
    n_cols: int = 3,
    figsize_per_topic: Tuple[float, float] = (4, 3),
    max_words: int = 30,
    background_color: str = 'white',
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    Plot comparative word clouds showing word associations between topics.
    
    For each comparison topic, words are colored by their association strength:
    - Blue: more associated with reference topic
    - Red: more associated with comparison topic
    
    Parameters
    ----------
    topics_df : pd.DataFrame
        DataFrame with columns ["Topic", "Term"/"Word", "Weight"].
    topic_term_matrix : np.ndarray, optional
        Topic-term distribution matrix (n_topics x n_terms).
        If None, will be built from topics_df.
    feature_names : list of str, optional
        Vocabulary corresponding to topic_term_matrix columns.
    reference_topic : str, optional
        Reference topic for comparison. If None, uses first topic.
    n_cols : int
        Number of columns in grid.
    figsize_per_topic : tuple
        Size per topic subplot.
    max_words : int
        Maximum words per cloud.
    background_color : str
        Background color.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    fig
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        raise ImportError("Install wordcloud: pip install wordcloud")
    
    from matplotlib.patches import Patch
    
    # Handle column name variations
    term_col = "Term" if "Term" in topics_df.columns else "Word"
    
    topics = sorted(topics_df["Topic"].unique())
    n_topics = len(topics)
    
    if n_topics < 2:
        raise ValueError("Need at least 2 topics for comparative word clouds")
    
    # Set reference topic
    if reference_topic is None:
        reference_topic = topics[0]
    
    ref_topic = str(reference_topic)
    other_topics = [t for t in topics if str(t) != ref_topic]
    
    # Build topic-term matrix if not provided
    if topic_term_matrix is None or feature_names is None:
        all_terms = sorted(topics_df[term_col].unique())
        feature_names = all_terms
        n_terms = len(all_terms)
        topic_term_matrix = np.zeros((n_topics, n_terms))
        
        term_to_idx = {term: i for i, term in enumerate(all_terms)}
        
        for i, topic in enumerate(topics):
            topic_data = topics_df[topics_df["Topic"] == topic]
            for _, row in topic_data.iterrows():
                term = row[term_col]
                weight = row["Weight"]
                if term in term_to_idx:
                    topic_term_matrix[i, term_to_idx[term]] = weight
    
    feature_names_list = list(feature_names)
    
    # Find reference topic index
    ref_idx = None
    for i, t in enumerate(topics):
        if str(t) == ref_topic:
            ref_idx = i
            break
    
    if ref_idx is None:
        raise ValueError(f"Reference topic '{ref_topic}' not found in topics")
    
    # Setup figure
    n_other = len(other_topics)
    n_rows = int(np.ceil(n_other / n_cols))
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_topic[0] * n_cols, figsize_per_topic[1] * n_rows)
    )
    
    if n_other == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Diverging colormap: blue (ref) to red (comparison)
    cmap_div = plt.cm.get_cmap('RdYlBu_r')
    
    for i, comp_topic in enumerate(other_topics):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        # Find comparison topic index
        comp_idx = None
        for j, t in enumerate(topics):
            if str(t) == str(comp_topic):
                comp_idx = j
                break
        
        if comp_idx is None:
            ax.axis('off')
            continue
        
        # Get weights for both topics
        ref_weights = topic_term_matrix[ref_idx]
        comp_weights = topic_term_matrix[comp_idx]
        
        # Get words in either topic from topics_df
        topic_data_ref = topics_df[topics_df["Topic"] == ref_topic]
        topic_data_comp = topics_df[topics_df["Topic"] == comp_topic]
        
        ref_words = set(topic_data_ref[term_col].values)
        comp_words = set(topic_data_comp[term_col].values)
        all_words = ref_words | comp_words
        
        # Build word frequencies and association scores
        word_weights = {}
        word_colors = {}
        
        for word in all_words:
            if word in feature_names_list:
                word_idx = feature_names_list.index(word)
                w_ref = float(ref_weights[word_idx])
                w_comp = float(comp_weights[word_idx])
            else:
                w_ref_arr = topic_data_ref[topic_data_ref[term_col] == word]["Weight"].values
                w_ref = float(w_ref_arr[0]) if len(w_ref_arr) > 0 else 0.0
                w_comp_arr = topic_data_comp[topic_data_comp[term_col] == word]["Weight"].values
                w_comp = float(w_comp_arr[0]) if len(w_comp_arr) > 0 else 0.0
            
            # Frequency is max of both
            freq = max(w_ref, w_comp, 0.001)
            word_weights[word] = freq
            
            # Association score: -1 (fully ref) to +1 (fully comp)
            total = w_ref + w_comp
            assoc = (w_comp - w_ref) / total if total > 0 else 0.0
            word_colors[word] = assoc
        
        if not word_weights:
            ax.axis('off')
            continue
        
        # Create color function with captured dict
        def make_color_func(colors_dict, colormap):
            def color_func(word, **kwargs):
                assoc = colors_dict.get(word, 0)
                norm_val = (assoc + 1) / 2  # Map -1,1 to 0,1
                rgba = colormap(norm_val)
                return tuple(int(c * 255) for c in rgba[:3])
            return color_func
        
        color_func = make_color_func(word_colors.copy(), cmap_div)
        
        wc = WordCloud(
            width=400, height=300,
            background_color=background_color,
            max_words=max_words,
            color_func=color_func,
        ).generate_from_frequencies(word_weights)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f"{ref_topic} vs {comp_topic}", fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_other, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#2166ac', label=f'→ {ref_topic}'),
        Patch(facecolor='#b2182b', label='→ Other topic'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig


# =============================================================================
# SEQUENTIAL AND DYNAMIC TOPIC MODEL PLOTS
# =============================================================================

def plot_topic_prevalence_evolution(
    prevalence_df: pd.DataFrame,
    time_col: str = 'Time_Slice',
    topic_col: str = 'Topic',
    value_col: str = 'Prevalence',
    figsize: Tuple[float, float] = (12, 6),
    cmap: str = 'tab10',
    title: str = 'Topic Prevalence Over Time',
    stacked: bool = True,
    show_legend: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot topic prevalence evolution over time.
    
    Parameters
    ----------
    prevalence_df : pd.DataFrame
        DataFrame with time, topic, and prevalence columns.
    time_col : str
        Column with time information.
    topic_col : str
        Column with topic labels.
    value_col : str
        Column with prevalence values.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for topics.
    title : str
        Plot title.
    stacked : bool
        Whether to create stacked area plot.
    show_legend : bool
        Whether to show legend.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pivot data for plotting
    pivot_df = prevalence_df.pivot(index=time_col, columns=topic_col, values=value_col).fillna(0)
    
    topics = pivot_df.columns.tolist()
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(topics)))
    
    if stacked:
        ax.stackplot(
            pivot_df.index,
            pivot_df.T.values,
            labels=topics,
            colors=colors,
            alpha=0.8
        )
    else:
        for i, topic in enumerate(topics):
            ax.plot(pivot_df.index, pivot_df[topic], label=topic, color=colors[i], linewidth=2, marker='o')
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Prevalence")
    ax.set_title(title)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig, ax


def plot_topic_word_evolution(
    topic_word_evolution: Dict[str, pd.DataFrame],
    topic_id: str = 'Topic 1',
    top_n: int = 10,
    figsize: Tuple[float, float] = (14, 8),
    cmap: str = 'viridis',
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot word weight evolution for a specific topic over time.
    
    Parameters
    ----------
    topic_word_evolution : dict
        Dict of {topic_id: DataFrame} with time, term, weight columns.
    topic_id : str
        Which topic to plot.
    top_n : int
        Number of top words to show.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for heatmap.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    (fig, ax)
    """
    if topic_id not in topic_word_evolution:
        raise ValueError(f"Topic '{topic_id}' not found.")
    
    df = topic_word_evolution[topic_id]
    
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig, ax
    
    # Get top words overall
    word_weights = df.groupby('Term')['Weight'].mean().nlargest(top_n)
    top_words = word_weights.index.tolist()
    
    # Filter to top words
    df_filtered = df[df['Term'].isin(top_words)]
    
    # Pivot for heatmap
    time_col = 'Time_Label' if 'Time_Label' in df.columns else 'Time_Slice'
    pivot = df_filtered.pivot(index='Term', columns=time_col, values='Weight').fillna(0)
    
    # Reorder rows by overall weight
    pivot = pivot.reindex(top_words)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        pivot,
        cmap=cmap,
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Weight', 'orientation': 'horizontal', 'pad': 0.15},
        ax=ax
    )
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Term")
    ax.set_title(title or f"{topic_id} - Word Evolution Over Time")
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig, ax


def plot_topic_evolution_alluvial(
    evolution_df: pd.DataFrame,
    figsize: Tuple[float, float] = (14, 8),
    cmap: str = 'tab10',
    title: str = 'Topic Evolution Across Time Periods',
    min_similarity: float = 0.3,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot topic evolution as an alluvial/Sankey-style diagram.
    
    Parameters
    ----------
    evolution_df : pd.DataFrame
        DataFrame with From_Period, To_Period, From_Topic, To_Topic, Similarity.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for topics.
    title : str
        Plot title.
    min_similarity : float
        Minimum similarity to show connection.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if evolution_df.empty:
        ax.text(0.5, 0.5, "No evolution data available", ha='center', va='center')
        return fig, ax
    
    # Filter by minimum similarity
    df = evolution_df[evolution_df['Similarity'] >= min_similarity].copy()
    
    if df.empty:
        ax.text(0.5, 0.5, f"No connections with similarity >= {min_similarity}", ha='center', va='center')
        return fig, ax
    
    # Get unique periods and topics
    periods = sorted(set(df['From_Period'].tolist() + df['To_Period'].tolist()))
    all_topics = sorted(set(df['From_Topic'].tolist() + df['To_Topic'].tolist()))
    
    n_topics = len(all_topics)
    n_periods = len(periods)
    
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_topics))
    topic_colors = {topic: colors[i] for i, topic in enumerate(all_topics)}
    
    # Position topics vertically
    topic_y = {topic: i for i, topic in enumerate(all_topics)}
    
    # Draw connections
    for _, row in df.iterrows():
        from_period = row['From_Period']
        to_period = row['To_Period']
        from_topic = row['From_Topic']
        to_topic = row['To_Topic']
        similarity = row['Similarity']
        
        x1 = periods.index(from_period)
        x2 = periods.index(to_period)
        y1 = topic_y[from_topic]
        y2 = topic_y[to_topic]
        
        # Draw curved line
        alpha = min(1.0, similarity)
        linewidth = similarity * 5
        
        ax.plot([x1, x2], [y1, y2], 
                color=topic_colors[from_topic], 
                alpha=alpha, 
                linewidth=linewidth)
    
    # Draw topic nodes
    for period_idx, period in enumerate(periods):
        for topic in all_topics:
            ax.scatter(period_idx, topic_y[topic], 
                      s=200, color=topic_colors[topic], 
                      edgecolor='black', linewidth=1, zorder=5)
    
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods)
    ax.set_yticks(range(len(all_topics)))
    ax.set_yticklabels(all_topics)
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Topic")
    ax.set_title(title)
    ax.grid(False)
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig, ax


def plot_dtm_topic_streams(
    prevalence_df: pd.DataFrame,
    figsize: Tuple[float, float] = (14, 6),
    cmap: str = 'tab10',
    title: str = 'Topic Streams Over Time',
    smooth: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot topic prevalence as stream graph (ThemeRiver style).
    
    Parameters
    ----------
    prevalence_df : pd.DataFrame
        DataFrame with Time_Slice, Topic, Prevalence columns.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for topics.
    title : str
        Plot title.
    smooth : bool
        Whether to apply smoothing.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if prevalence_df.empty:
        ax.text(0.5, 0.5, "No prevalence data", ha='center', va='center')
        return fig, ax
    
    # Determine column names
    time_col = 'Time_Slice' if 'Time_Slice' in prevalence_df.columns else 'Period'
    topic_col = 'Topic'
    value_col = 'Prevalence'
    
    # Pivot data
    pivot = prevalence_df.pivot(index=time_col, columns=topic_col, values=value_col).fillna(0)
    
    # Normalize
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    
    topics = pivot.columns.tolist()
    x = pivot.index.values
    
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(topics)))
    
    # Create baseline (centered)
    y_stack = np.row_stack([pivot[t].values for t in topics])
    
    # Center the streams
    baseline = -y_stack.sum(axis=0) / 2
    
    y_bottom = baseline.copy()
    
    for i, topic in enumerate(topics):
        y_top = y_bottom + pivot[topic].values
        ax.fill_between(x, y_bottom, y_top, label=topic, color=colors[i], alpha=0.8)
        y_bottom = y_top
    
    ax.set_xlabel("Time Slice")
    ax.set_ylabel("Relative Prevalence")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig, ax


def plot_term_evolution(
    term_evolution_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[float, float] = (14, 8),
    cmap: str = 'viridis',
    title: str = 'Term Frequency Evolution Over Time',
    plot_type: str = 'heatmap',
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot term frequency evolution over time.
    
    Parameters
    ----------
    term_evolution_df : pd.DataFrame
        DataFrame with Period, Term, Frequency columns.
    top_n : int, default=15
        Number of top terms to display.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for heatmap.
    title : str
        Plot title.
    plot_type : str, default='heatmap'
        Type of plot: 'heatmap', 'line', or 'area'.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if term_evolution_df.empty:
        ax.text(0.5, 0.5, "No term evolution data", ha='center', va='center')
        return fig, ax
    
    # Get top terms by average frequency
    term_avg = term_evolution_df.groupby('Term')['Frequency'].mean()
    top_terms = term_avg.nlargest(top_n).index.tolist()
    
    # Filter to top terms
    df_filtered = term_evolution_df[term_evolution_df['Term'].isin(top_terms)]
    
    # Pivot for plotting
    pivot = df_filtered.pivot(index='Term', columns='Period', values='Frequency').fillna(0)
    
    # Reorder by average frequency
    pivot = pivot.reindex(top_terms)
    
    if plot_type == 'heatmap':
        sns.heatmap(
            pivot,
            cmap=cmap,
            annot=True if len(pivot.columns) <= 10 else False,
            fmt='.4f' if pivot.values.max() < 1 else '.0f',
            cbar_kws={'label': 'Relative Frequency', 'orientation': 'horizontal', 'pad': 0.15},
            ax=ax
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Term")
        ax.set_title(title)
        
    elif plot_type == 'line':
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(top_terms)))
        
        for i, term in enumerate(top_terms):
            if term in pivot.index:
                ax.plot(pivot.columns, pivot.loc[term], label=term, 
                       color=colors[i], linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel("Year")
        ax.set_ylabel("Relative Frequency")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    elif plot_type == 'area':
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(top_terms)))
        
        ax.stackplot(
            pivot.columns,
            pivot.values,
            labels=pivot.index.tolist(),
            colors=colors,
            alpha=0.8
        )
        
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative Relative Frequency")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig, ax


def plot_term_trends(
    term_trends_df: pd.DataFrame,
    figsize: Tuple[float, float] = (12, 6),
    cmap: str = 'tab10',
    title: str = 'Term Trends Over Time',
    show_markers: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot term trends over time as line chart.
    
    Parameters
    ----------
    term_trends_df : pd.DataFrame
        DataFrame with periods as index, terms as columns.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for lines.
    title : str
        Plot title.
    show_markers : bool, default=True
        Whether to show markers on lines.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if term_trends_df.empty:
        ax.text(0.5, 0.5, "No term trends data", ha='center', va='center')
        return fig, ax
    
    terms = term_trends_df.columns.tolist()
    periods = term_trends_df.index.tolist()
    
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(terms)))
    
    for i, term in enumerate(terms):
        marker = 'o' if show_markers else None
        ax.plot(periods, term_trends_df[term], label=term, 
               color=colors[i], linewidth=2, marker=marker, markersize=5)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Relative Frequency")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig, ax


def plot_term_bump_chart(
    term_evolution_df: pd.DataFrame,
    top_n: int = 10,
    figsize: Tuple[float, float] = (14, 8),
    cmap: str = 'tab10',
    title: str = 'Term Rank Evolution (Bump Chart)',
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot term rank evolution as a bump chart.
    
    Parameters
    ----------
    term_evolution_df : pd.DataFrame
        DataFrame with Period, Term, Frequency columns.
    top_n : int, default=10
        Number of top terms to track.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for lines.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saving.
    
    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if term_evolution_df.empty:
        ax.text(0.5, 0.5, "No data for bump chart", ha='center', va='center')
        return fig, ax
    
    # Get top terms overall
    term_avg = term_evolution_df.groupby('Term')['Frequency'].mean()
    top_terms = term_avg.nlargest(top_n).index.tolist()
    
    # Filter and compute ranks per period
    df_filtered = term_evolution_df[term_evolution_df['Term'].isin(top_terms)].copy()
    
    # Compute rank within each period
    df_filtered['Rank'] = df_filtered.groupby('Period')['Frequency'].rank(
        ascending=False, method='min'
    )
    
    # Pivot to get ranks
    pivot = df_filtered.pivot(index='Period', columns='Term', values='Rank')
    
    periods = pivot.index.tolist()
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(top_terms)))
    
    for i, term in enumerate(top_terms):
        if term in pivot.columns:
            ranks = pivot[term].values
            ax.plot(periods, ranks, label=term, color=colors[i], 
                   linewidth=3, marker='o', markersize=8)
            
            # Add term labels at the end
            if not np.isnan(ranks[-1]):
                ax.annotate(term, (periods[-1], ranks[-1]), 
                           xytext=(5, 0), textcoords='offset points',
                           fontsize=8, va='center')
    
    # Invert y-axis so rank 1 is at top
    ax.invert_yaxis()
    ax.set_xlabel("Year")
    ax.set_ylabel("Rank")
    ax.set_title(title)
    ax.set_yticks(range(1, top_n + 1))
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig, ax

    
# Plotting of bibliometric laws

def plot_lotka_distribution(lotka_df, title="Lotka's Law - Author Productivity", filename_base=None, dpi=600, 
                             observed_color="blue", expected_color="orange", show_grid=False):
    """
    Plot observed vs expected author productivity under Lotka's Law.

    Parameters:
        lotka_df (pd.DataFrame): Output of compute_lotka_distribution.
        title (str): Title for the plot.
        filename_base (str, optional): Base filename to save the plot without extension.
        dpi (int): Dots per inch for saving the plot.
        observed_color (str): Color for the observed data points and line.
        expected_color (str): Color for the expected (Lotka) line.
        show_grid (bool): Whether to show grid lines (default: False).
    """
    plt.figure(figsize=(8, 6))
    plt.loglog(lotka_df["n_pubs"], lotka_df["n_authors"], "o-", label="Observed", color=observed_color)
    plt.loglog(lotka_df["n_pubs"], lotka_df["expected_n_authors"], "s--", label="Expected (Lotka)", color=expected_color)

    plt.xlabel("Number of Documents (n)", fontsize=12)
    plt.ylabel("Number of Authors", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    if show_grid:
        plt.grid(True, which="both", ls="--", lw=0.5)
    else:
        plt.grid(False)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()

def plot_bradford_distribution(source_counts, title="Bradford's Law - Source Scattering", filename_base=None, dpi=600, color="blue", show_grid=False):
    """
    Plot Bradford's Law cumulative distribution.
    
    Parameters
    ----------
    source_counts : pd.DataFrame
        Source counts with 'Cumulative_Percentage' column.
    title : str
        Plot title.
    filename_base : str, optional
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    color : str
        Line color.
    show_grid : bool
        Whether to show grid lines.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(source_counts) + 1), source_counts["Cumulative_Percentage"], marker="o", color=color)
    plt.xlabel("Ranked Sources", fontsize=12)
    plt.ylabel("Cumulative % of Documents", fontsize=12)
    plt.title(title, fontsize=14)
    if show_grid:
        plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()


def plot_bradford_zones(source_counts, title="Bradford's Law - Zones", filename_base=None, dpi=600, colors=None, annotate_core=True, show_labels="zone1", label_rotation=90, alt_label_col="Abbreviated Source Title", max_label_length=30, show_grid=False):
    """
    Plot Bradford's Law zone visualization.
    
    Parameters
    ----------
    source_counts : pd.DataFrame
        Source counts with 'Zone' and 'Document_Count' columns.
    title : str
        Plot title.
    filename_base : str, optional
        Base filename for saving.
    dpi : int
        Resolution for saved images.
    colors : list, optional
        Colors for each zone.
    annotate_core : bool
        Whether to annotate core zone.
    show_labels : str
        'zone1', 'all', or None for label display.
    label_rotation : int
        Rotation angle for labels.
    alt_label_col : str
        Alternative column for labels.
    max_label_length : int
        Maximum label length before truncation.
    show_grid : bool
        Whether to show grid lines.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    zone_count = source_counts["Zone"].max()
    if colors is None:
        colors = ["#c6dbef", "#9ecae1", "#6baed6"][:zone_count]

    start = 0
    tick_labels = []
    tick_positions = []

    for z in range(1, zone_count + 1):
        group = source_counts[source_counts["Zone"] == z].copy()
        x = list(range(start + 1, start + len(group) + 1))
        ax.fill_between(x, group["Document_Count"], color=colors[z - 1], step="mid", label=f"Zone {z}")

        if show_labels == "all" or (show_labels == "zone1" and z == 1):
            if alt_label_col and alt_label_col in group.columns:
                labels = group[alt_label_col].fillna(group["Source"])
            else:
                labels = group["Source"].apply(lambda s: s if len(s) <= max_label_length else s[:max_label_length-3] + "...")

            tick_labels.extend(labels.tolist())
            tick_positions.extend(x)

        start += len(group)

    ax.set_xlabel("Source Rank", fontsize=12)
    ax.set_ylabel("Documents", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yscale("linear")
    ax.set_xscale("log")
    if show_grid:
        ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    if tick_labels:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=label_rotation, ha="right", fontsize=9)

    if annotate_core:
        ax.text(2, max(source_counts["Document_Count"]) * 0.9, "Core Sources", fontsize=12, alpha=0.6)

    plt.tight_layout()
    if filename_base:
        save_plot(filename_base, dpi)
    plt.show()

def compute_zipf_distribution_from_counts(df, word_col=0, count_col=1):
    """
    Compute Zipf's Law distribution given a DataFrame with word/item counts.
    
    Parameters:
        df (pd.DataFrame): DataFrame where one column is words/items and another is counts.
        word_col (int or str): Column name or index for words/items.
        count_col (int or str): Column name or index for counts.
        
    Returns:
        pd.DataFrame: DataFrame with 'Word', 'Frequency', and 'Rank'.
    """
    # Convert column indices to names if integers
    if isinstance(word_col, int):
        word_col = df.columns[word_col]
    if isinstance(count_col, int):
        count_col = df.columns[count_col]
    
    zipf_df = df[[word_col, count_col]].copy()
    zipf_df.columns = ["Word", "Frequency"]
    zipf_df = zipf_df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
    zipf_df["Rank"] = np.arange(1, len(zipf_df) + 1)
    return zipf_df


def plot_zipf_distribution(zipf_df, title="Zipf's Law - Word Frequencies", filename_base=None, dpi=600, color="blue", show_grid=False, top_n_labels=10):
    """
    Plot Zipf's Law distribution: frequency vs rank on a log-log scale.

    Parameters:
        zipf_df (pd.DataFrame): Output of compute_zipf_distribution_from_counts.
        title (str): Title of the plot.
        filename_base (str, optional): Base filename to save the plot.
        dpi (int): Dots per inch for saving.
        color (str): Color of the curve.
        show_grid (bool): Whether to show grid.
        top_n_labels (int): Number of top labels to display.
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(zipf_df["Rank"], zipf_df["Frequency"], marker="o", linestyle="-", color=color)
    plt.xlabel("Rank", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(title, fontsize=14)

    if top_n_labels > 0:
        for i in range(min(top_n_labels, len(zipf_df))):
            x = zipf_df.loc[i, "Rank"]
            y = zipf_df.loc[i, "Frequency"]
            word = zipf_df.loc[i, "Word"]
            plt.text(x, y, word, fontsize=8, ha="left", va="bottom")

    if show_grid:
        plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()


def plot_prices_law(author_counts, title="Price's Law - Core Author Contribution", filename_base=None, dpi=600, color_core="red", color_tail="gray", show_grid=False):
    """
    Plot cumulative document contribution by authors, highlighting Price's core group.

    Parameters:
        author_counts (pd.Series): Series with authors as index and document counts as values.
        title (str): Plot title.
        filename_base (str): Base path to save the plot.
        dpi (int): Dots per inch for saving.
        color_core (str): Color for core authors.
        color_tail (str): Color for remaining authors.
        show_grid (bool): Whether to show grid.
    """
    sorted_counts = author_counts.sort_values(ascending=False).reset_index(drop=True)
    cumulative_docs = sorted_counts.cumsum() / sorted_counts.sum() * 100
    core_size = int(np.sqrt(len(sorted_counts)))

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_docs.index[:core_size], cumulative_docs.iloc[:core_size], color=color_core, label="Core Authors")
    plt.plot(cumulative_docs.index[core_size:], cumulative_docs.iloc[core_size:], color=color_tail, label="Other Authors")

    plt.axvline(core_size, color="black", linestyle="--", linewidth=1, label=f"sqrt(N) = {core_size}")
    plt.xlabel("Author Rank", fontsize=12)
    plt.ylabel("Cumulative % of Documents", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    if show_grid:
        plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()
    
def plot_pareto_principle(counts, top_percentage=20, title="Pareto Principle Analysis", filename_base=None, dpi=600, color_curve="blue", color_threshold="red", show_grid=False):
    """
    Plot cumulative contribution curve highlighting the Pareto threshold.

    Parameters:
        counts (pd.Series): Series of items and their counts.
        top_percentage (float): Percentage of top items (default 20).
        title (str): Title for the plot.
        filename_base (str): Base filename to save the plot.
        dpi (int): Resolution for saved plots.
        color_curve (str): Line color.
        color_threshold (str): Threshold line color.
        show_grid (bool): Whether to show grid.
    """
    sorted_counts = counts.sort_values(ascending=False).reset_index(drop=True)
    cumulative_contribution = sorted_counts.cumsum() / sorted_counts.sum() * 100

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_contribution.index, cumulative_contribution.values, color=color_curve)

    threshold_index = int(np.ceil(top_percentage / 100 * len(cumulative_contribution)))
    plt.axvline(threshold_index, color=color_threshold, linestyle="--", label=f"Top {top_percentage}%")

    plt.xlabel("Item Rank", fontsize=12)
    plt.ylabel("Cumulative % of Contribution", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()

    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()


# =============================================================================
# DISTRIBUTION FITTING PLOTS
# =============================================================================

def plot_distribution_fit(
    data,
    distribution="lognormal",
    title=None,
    xlabel="Value",
    color_hist="lightblue",
    color_fit="red",
    bins=50,
    figsize=(10, 6),
    filename_base=None,
    dpi=600,
    show=True,
):
    """
    Plot histogram with fitted distribution overlay.
    
    Parameters
    ----------
    data : array-like
        Data to fit and plot.
    distribution : str
        Distribution to fit: "lognormal", "exponential", "power_law", 
        "gamma", "weibull", "normal".
    title : str
        Plot title. Auto-generated if None.
    xlabel : str
        X-axis label.
    color_hist : str
        Histogram color.
    color_fit : str
        Fitted line color.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size.
    filename_base : str
        Base filename for saving.
    dpi : int
        Resolution.
    show : bool
        Whether to display plot.
    
    Returns
    -------
    dict
        Fitted parameters.
    """
    from scipy import stats
    
    data = np.array(data)
    data = data[~np.isnan(data)]
    data = data[data > 0]  # Most distributions need positive values
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    # Plot histogram (density)
    counts, bin_edges, _ = ax.hist(data, bins=bins, density=True, 
                                    color=color_hist, alpha=0.7, 
                                    edgecolor="white", label="Observed")
    
    # Fit distribution
    x = np.linspace(data.min(), data.max(), 200)
    params = {}
    
    if distribution == "lognormal":
        shape, loc, scale = stats.lognorm.fit(data, floc=0)
        y = stats.lognorm.pdf(x, shape, loc, scale)
        params = {"shape": shape, "loc": loc, "scale": scale}
        label = f"Log-normal (σ={shape:.2f})"
        
    elif distribution == "exponential":
        loc, scale = stats.expon.fit(data)
        y = stats.expon.pdf(x, loc, scale)
        params = {"loc": loc, "scale": scale}
        label = f"Exponential (λ={1/scale:.3f})"
        
    elif distribution == "power_law":
        # Pareto distribution
        b, loc, scale = stats.pareto.fit(data, floc=0)
        y = stats.pareto.pdf(x, b, loc, scale)
        params = {"b": b, "loc": loc, "scale": scale}
        label = f"Power law (α={b:.2f})"
        
    elif distribution == "gamma":
        a, loc, scale = stats.gamma.fit(data, floc=0)
        y = stats.gamma.pdf(x, a, loc, scale)
        params = {"a": a, "loc": loc, "scale": scale}
        label = f"Gamma (k={a:.2f})"
        
    elif distribution == "weibull":
        c, loc, scale = stats.weibull_min.fit(data, floc=0)
        y = stats.weibull_min.pdf(x, c, loc, scale)
        params = {"c": c, "loc": loc, "scale": scale}
        label = f"Weibull (k={c:.2f})"
        
    elif distribution == "normal":
        loc, scale = stats.norm.fit(data)
        y = stats.norm.pdf(x, loc, scale)
        params = {"loc": loc, "scale": scale}
        label = f"Normal (μ={loc:.2f}, σ={scale:.2f})"
        
    elif distribution == "negative_binomial":
        mean_val = data.mean()
        var_val = data.var()
        if var_val > mean_val:
            p = mean_val / var_val
            r = mean_val * p / (1 - p)
        else:
            r, p = mean_val, 0.5
        x_int = np.arange(0, int(data.max()) + 1)
        y = stats.nbinom.pmf(x_int, r, p)
        ax.plot(x_int, y, color=color_fit, linewidth=2, label=f"Neg. Binomial (r={r:.1f})")
        params = {"r": r, "p": p}
        label = None  # Already plotted
        
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Plot fitted curve
    if label:
        ax.plot(x, y, color=color_fit, linewidth=2, label=label)
    
    if title is None:
        title = f"Distribution Fit: {distribution.replace('_', ' ').title()}"
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    
    # Add statistics annotation
    stats_text = f"n={len(data)}\nmean={data.mean():.2f}\nmedian={np.median(data):.2f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment="top", horizontalalignment="right",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if filename_base:
        save_plot(filename_base, dpi)
    
    if show:
        plt.show()
    
    return params


def plot_distribution_comparison(
    data,
    distributions=None,
    title="Distribution Comparison",
    xlabel="Value",
    bins=50,
    figsize=(12, 6),
    filename_base=None,
    dpi=600,
    show=True,
):
    """
    Plot histogram with multiple fitted distributions overlaid.
    
    Parameters
    ----------
    data : array-like
        Data to fit and plot.
    distributions : list
        Distributions to compare. Default: ["lognormal", "exponential", "gamma", "weibull"].
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size.
    filename_base : str
        Base filename for saving.
    dpi : int
        Resolution.
    show : bool
        Whether to display plot.
    
    Returns
    -------
    pd.DataFrame
        Comparison of fit quality.
    """
    from scipy import stats
    
    if distributions is None:
        distributions = ["lognormal", "exponential", "gamma", "weibull"]
    
    data = np.array(data)
    data = data[~np.isnan(data)]
    data = data[data > 0]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    # Plot histogram
    ax.hist(data, bins=bins, density=True, color="lightgray", 
            alpha=0.7, edgecolor="white", label="Observed")
    
    x = np.linspace(data.min(), data.max(), 200)
    colors = plt.cm.Set1(np.linspace(0, 1, len(distributions)))
    
    results = []
    
    for dist_name, color in zip(distributions, colors):
        try:
            if dist_name == "lognormal":
                shape, loc, scale = stats.lognorm.fit(data, floc=0)
                y = stats.lognorm.pdf(x, shape, loc, scale)
                ks_stat, ks_pval = stats.kstest(data, "lognorm", args=(shape, loc, scale))
                label = f"Log-normal"
                
            elif dist_name == "exponential":
                loc, scale = stats.expon.fit(data)
                y = stats.expon.pdf(x, loc, scale)
                ks_stat, ks_pval = stats.kstest(data, "expon", args=(loc, scale))
                label = f"Exponential"
                
            elif dist_name == "gamma":
                a, loc, scale = stats.gamma.fit(data, floc=0)
                y = stats.gamma.pdf(x, a, loc, scale)
                ks_stat, ks_pval = stats.kstest(data, "gamma", args=(a, loc, scale))
                label = f"Gamma"
                
            elif dist_name == "weibull":
                c, loc, scale = stats.weibull_min.fit(data, floc=0)
                y = stats.weibull_min.pdf(x, c, loc, scale)
                ks_stat, ks_pval = stats.kstest(data, "weibull_min", args=(c, loc, scale))
                label = f"Weibull"
                
            elif dist_name == "power_law":
                b, loc, scale = stats.pareto.fit(data, floc=0)
                y = stats.pareto.pdf(x, b, loc, scale)
                ks_stat, ks_pval = stats.kstest(data, "pareto", args=(b, loc, scale))
                label = f"Power law"
                
            elif dist_name == "normal":
                loc, scale = stats.norm.fit(data)
                y = stats.norm.pdf(x, loc, scale)
                ks_stat, ks_pval = stats.kstest(data, "norm", args=(loc, scale))
                label = f"Normal"
                
            else:
                continue
            
            ax.plot(x, y, color=color, linewidth=2, label=f"{label} (KS p={ks_pval:.3f})")
            results.append({
                "Distribution": dist_name,
                "KS Statistic": ks_stat,
                "KS P-value": ks_pval,
            })
            
        except Exception as e:
            print(f"Could not fit {dist_name}: {e}")
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    
    plt.tight_layout()
    
    if filename_base:
        save_plot(filename_base, dpi)
    
    if show:
        plt.show()
    
    return pd.DataFrame(results).sort_values("KS P-value", ascending=False)


def plot_qq(
    data,
    distribution="norm",
    title=None,
    figsize=(8, 8),
    color="steelblue",
    line_color="red",
    filename_base=None,
    dpi=600,
    show=True,
):
    """
    Plot Q-Q (quantile-quantile) plot to assess distribution fit.
    
    Parameters
    ----------
    data : array-like
        Data to plot.
    distribution : str
        Theoretical distribution: "norm", "lognorm", "expon", etc.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    color : str
        Point color.
    line_color : str
        Reference line color.
    filename_base : str
        Base filename for saving.
    dpi : int
        Resolution.
    show : bool
        Whether to display plot.
    """
    from scipy import stats
    
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    if distribution == "norm":
        stats.probplot(data, dist="norm", plot=ax)
        dist_name = "Normal"
    elif distribution == "lognorm":
        # For log-normal, plot log of data against normal
        data_log = np.log(data[data > 0])
        stats.probplot(data_log, dist="norm", plot=ax)
        dist_name = "Log-normal"
        ax.set_xlabel("Theoretical Quantiles (log scale)")
    elif distribution == "expon":
        stats.probplot(data, dist="expon", plot=ax)
        dist_name = "Exponential"
    else:
        stats.probplot(data, dist=distribution, plot=ax)
        dist_name = distribution.title()
    
    # Customize appearance
    ax.get_lines()[0].set_markerfacecolor(color)
    ax.get_lines()[0].set_markeredgecolor(color)
    ax.get_lines()[1].set_color(line_color)
    
    if title is None:
        title = f"Q-Q Plot: {dist_name} Distribution"
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if filename_base:
        save_plot(filename_base, dpi)
    
    if show:
        plt.show()


def plot_distribution_temporal(
    results_df,
    param_column,
    title=None,
    ylabel=None,
    color="steelblue",
    figsize=(12, 6),
    filename_base=None,
    dpi=600,
    show=True,
):
    """
    Plot distribution parameter evolution over time.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output from analyze_distribution_over_time().
    param_column : str
        Parameter column to plot (e.g., "Shape", "Alpha", "Rate").
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    color : str
        Line/bar color.
    figsize : tuple
        Figure size.
    filename_base : str
        Base filename for saving.
    dpi : int
        Resolution.
    show : bool
        Whether to display plot.
    """
    if param_column not in results_df.columns:
        raise ValueError(f"Column '{param_column}' not found in results")
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    x = range(len(results_df))
    y = results_df[param_column].values
    
    ax.bar(x, y, color=color, alpha=0.7)
    ax.plot(x, y, color="black", linewidth=2, marker="o", markersize=6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Period"].values, rotation=45, ha="right")
    
    if title is None:
        title = f"Distribution Parameter Over Time: {param_column}"
    if ylabel is None:
        ylabel = param_column
    
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if filename_base:
        save_plot(filename_base, dpi)
    
    if show:
        plt.show()


def plot_ccdf(
    data,
    title="Complementary Cumulative Distribution",
    xlabel="Value",
    log_scale=True,
    color="steelblue",
    figsize=(10, 6),
    filename_base=None,
    dpi=600,
    show=True,
):
    """
    Plot complementary cumulative distribution function (CCDF).
    
    CCDF plots are useful for visualizing heavy-tailed distributions
    like citation counts, where a power law appears as a straight line
    on log-log axes.
    
    Parameters
    ----------
    data : array-like
        Data to plot.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    log_scale : bool
        Whether to use log-log scale.
    color : str
        Line color.
    figsize : tuple
        Figure size.
    filename_base : str
        Base filename for saving.
    dpi : int
        Resolution.
    show : bool
        Whether to display plot.
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    data = np.sort(data)
    
    # Compute CCDF: P(X > x)
    n = len(data)
    ccdf = 1 - np.arange(1, n + 1) / n
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    ax.plot(data, ccdf, color=color, linewidth=2)
    
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("P(X > x) [log scale]", fontsize=12)
        ax.set_xlabel(f"{xlabel} [log scale]", fontsize=12)
    else:
        ax.set_ylabel("P(X > x)", fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Add guide for power law
    if log_scale:
        ax.text(0.05, 0.05, "Straight line = power law", transform=ax.transAxes,
                fontsize=10, style="italic", alpha=0.7)
    
    plt.tight_layout()
    
    if filename_base:
        save_plot(filename_base, dpi)
    
    if show:
        plt.show()
    
# Citations per year (prestavi nekoliko višje)

def plot_average_citations_per_year(
    df,
    year_col="Year",
    avg_col="Average Citations per Document",
    doc_count_col=None,  # optional column for secondary axis
    plot_type="line",
    color="black",
    secondary_color="lightblue",
    title="Average Citations per Document by Year",
    xlabel="Year",
    ylabel="Average Citations per Document",
    ylabel_secondary="Number of Documents",
    fontsize_title=14,
    fontsize_labels=12,
    marker="o",
    linewidth=2,
    xtick_rotation=90,
    wrap_xticks=False,
    wrap_width=10,
    filename_base=None,
    show=True
):
    """
    Plot the average citations per document by year as a line or bar chart,
    with optional secondary y-axis and x-tick label wrapping.

    Parameters:
        df (pd.DataFrame): DataFrame containing the year and citation data.
        year_col (str): Column name for year.
        avg_col (str): Column name for average citations.
        doc_count_col (str or None): Optional column name for document counts (for secondary y-axis).
        plot_type (str): "line" (default) or "bar".
        color (str): Color for main plot.
        secondary_color (str): Color for secondary axis plot (if enabled).
        title (str): Title of the plot.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        ylabel_secondary (str): Y-axis label for secondary axis.
        fontsize_title (int): Title font size.
        fontsize_labels (int): Axis labels font size.
        marker (str): Marker for line plot.
        linewidth (int or float): Line width for line plot.
        xtick_rotation (int): Rotation angle for x-tick labels.
        wrap_xticks (bool): Whether to wrap long tick labels.
        wrap_width (int): Max width of each wrapped line.
        filename_base (str or None): Base filename for saving.
        show (bool): Whether to display the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = df[year_col]
    y = df[avg_col]

    # Plot main data
    if plot_type == "bar":
        ax1.bar(x, y, color=color)
    elif plot_type == "line":
        ax1.plot(x, y, marker=marker, color=color, linewidth=linewidth)
    else:
        raise ValueError("plot_type must be either 'line' or 'bar'.")

    ax1.set_title(title, fontsize=fontsize_title)
    ax1.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax1.set_ylabel(ylabel, fontsize=fontsize_labels)
    ax1.tick_params(axis="y", labelcolor=color)

    # Handle optional wrapping of x-tick labels
    if wrap_xticks:
        labels = [textwrap.fill(str(label), wrap_width) for label in x]
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=xtick_rotation)
    else:
        plt.xticks(rotation=xtick_rotation)

    # Optional secondary axis
    if doc_count_col is not None and doc_count_col in df.columns:
        y2 = df[doc_count_col]
        ax2 = ax1.twinx()
        ax2.plot(x, y2, color=secondary_color, linestyle="--", marker="s")
        ax2.set_ylabel(ylabel_secondary, fontsize=fontsize_labels)
        ax2.tick_params(axis="y", labelcolor=secondary_color)

    if filename_base is not None:
        save_plot(filename_base)

    if show:
        plt.show()
    else:
        plt.close()
        
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.axes import Axes

def plot_average_citations_per_year_by_group(
    stats: pd.DataFrame,
    year_col: str = "Year",
    doc_prefix: str = "Number of documents ",
    cumcit_prefix: str = "Cumulative Citations ",
    *,
    plot_type: str = "line",                 # "line" or "bar"
    group_colors: dict[str, str] | None = None,
    cut_year: int | None = None,             # aggregate years < cut_year into "Before <cut_year>"
    plot_overall: bool = True,               # black line: overall average across all groups
    overall_label: str = "Overall average",
    legend_title: str = "Group",
    title: str = "Average Citations per Document by Year and Group",
    xlabel: str = "Year",
    ylabel: str = "Average Citations per Document",
    fontsize_title: int = 14,
    fontsize_labels: int = 12,
    marker: str = "o",
    linewidth: float = 2,
    xtick_rotation: int = 90,
    wrap_xticks: bool = False,
    wrap_width: int = 10,
    bar_width: float = 0.8,
    figsize: tuple[int, int] = (10, 6),
    filename_base: str | None = None,
    save_dpi: int | None = None,
    show: bool = True,
) -> "Axes":
    """
    Plot per-year average citations for multiple groups with a continuous time axis.

    This helper *derives* per-year averages from:
        "{doc_prefix}<GROUP>"    -> documents published that year
        "{cumcit_prefix}<GROUP>" -> cumulative citations up to that year

    Average(year, group) = Δcitations(year) / docs(year), where Δcitations is obtained
    by differencing the cumulative series on a **continuous** integer year grid from
    min(stats[year_col]) to max(stats[year_col]). Missing years are inserted so the
    x-axis is continuous; for line plots, NaNs from zero-doc years are forward-filled
    for display continuity only.

    If cut_year is provided, all years strictly below cut_year are aggregated into a single
    "Before <cut_year>" bin using weighted averaging:
        avg_before(group) = sum_citations_before / sum_docs_before

    Parameters
    ----------
    stats : pd.DataFrame
        Wide table with a numeric "Year" column and per-group doc/cumulative columns.
    year_col : str
        Name of the year column in `stats`.
    doc_prefix, cumcit_prefix : str
        Prefixes for per-group columns.
    plot_type : {"line","bar"}
        Multiple lines (one per group) or grouped bars per year.
    group_colors : dict[str,str] | None
        Optional mapping group->color; fallback palette is used where missing.
    cut_year : int | None
        Aggregate all years < cut_year into "Before <cut_year>".
    plot_overall : bool
        Plot overall average (weighted by docs) as a black line.
    overall_label : str
        Legend label for the overall line.
    legend_title, title, xlabel, ylabel : str
        Text for legend title, plot title, and axis labels.
    fontsize_title, fontsize_labels : int
        Font sizes.
    marker : str
        Marker for line plots.
    linewidth : float
        Line width for line plots.
    xtick_rotation : int
        Rotation of x tick labels.
    wrap_xticks : bool
        Wrap long tick labels using `wrap_width`.
    wrap_width : int
        Maximum characters per wrapped line.
    bar_width : float
        Total width allocated per year for grouped bars.
    figsize : (int,int)
        Figure size in inches.
    filename_base : str | None
        If provided, saved via plotbib.save_plot(filename_base[, dpi=...]).
    save_dpi : int | None
        If provided, forwarded to save_plot as the dpi argument.
    show : bool
        If True, display; else close the figure.

    Returns
    -------
    matplotlib.axes.Axes
        Primary axes.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import textwrap
    # save_plot is defined in this module, no need to import

    if year_col not in stats.columns:
        raise ValueError("The stats DataFrame must contain a \"Year\" column.")

    df = stats.copy()

    # Detect groups that have BOTH docs and cumulative columns
    doc_cols = [c for c in df.columns if c.startswith(doc_prefix)]
    cum_cols = [c for c in df.columns if c.startswith(cumcit_prefix)]
    groups = sorted(
        set([c.replace(doc_prefix, "") for c in doc_cols]) &
        set([c.replace(cumcit_prefix, "") for c in cum_cols])
    )
    if not groups:
        raise ValueError("No matching groups found. Expect both doc and cumulative columns for each group.")

    # Colors (fallback palette)
    default_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    colors: dict[str, str] = {}
    for i, g in enumerate(groups):
        if isinstance(group_colors, dict) and g in group_colors:
            colors[g] = group_colors[g]
        else:
            colors[g] = default_palette[i % len(default_palette)]

    # Numeric years present in stats
    years_num = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    if years_num.empty:
        raise ValueError("No numeric years found in the stats table.")
    y_min, y_max = years_num.min(), years_num.max()

    # Continuous integer range across min..max
    full_range = np.arange(y_min, y_max + 1, dtype=int)

    # Prepare per-group *continuous* series: docs(year) and Δcitations(year)
    per_group_docs_full: dict[str, pd.Series] = {}
    per_group_cites_full: dict[str, pd.Series] = {}
    for g in groups:
        doc_col = f"{doc_prefix}{g}"
        cum_col = f"{cumcit_prefix}{g}"

        # Sum docs by year; keep largest cumulative per year (in case of duplicate rows)
        docs = pd.to_numeric(df.set_index(year_col)[doc_col], errors="coerce").groupby(level=0).sum()
        cum = pd.to_numeric(df.set_index(year_col)[cum_col], errors="coerce").groupby(level=0).max()

        docs_full = docs.reindex(full_range).fillna(0.0)
        cum_full = cum.reindex(full_range).ffill().fillna(0.0)

        per_year_cites = cum_full.diff().fillna(cum_full)  # first year uses level itself

        per_group_docs_full[g] = docs_full
        per_group_cites_full[g] = per_year_cites

    # Build continuous x-axis labels and positions (with optional "Before <cut_year>")
    if cut_year is None:
        x_years = list(full_range)
        x_labels = [str(y) for y in x_years]
    else:
        x_years = [f"Before {cut_year}"] + list(range(max(cut_year, y_min), y_max + 1))
        x_labels = [str(x) for x in x_years]
    x_pos = np.arange(len(x_years))

    # Compute per-group average series aligned to the continuous axis
    def _group_avg_values(g: str) -> np.ndarray:
        docs_full = per_group_docs_full[g]
        cites_full = per_group_cites_full[g]
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_full = (cites_full / docs_full.replace(0, np.nan))  # Series over full_range

        if cut_year is None:
            vals = avg_full.reindex(full_range).to_numpy()
            # Forward-fill for plotting continuity (do not alter underlying stats)
            return pd.Series(vals).ffill().to_numpy()
        else:
            before_mask = full_range < cut_year
            before_docs = docs_full.loc[before_mask].sum()
            before_cites = cites_full.loc[before_mask].sum()
            before_avg = np.nan if before_docs == 0 else before_cites / before_docs

            after_years = list(range(max(cut_year, y_min), y_max + 1))
            after_vals = avg_full.reindex(after_years).to_numpy()
            after_vals = pd.Series(after_vals).ffill().to_numpy()
            return np.concatenate(([before_avg], after_vals))

    avg_mat = {g: _group_avg_values(g) for g in groups}

    # Overall average (weighted) if requested, aligned to the continuous axis
    overall_vals = None
    if plot_overall:
        total_docs_full = None
        total_cites_full = None
        for g in groups:
            total_docs_full = per_group_docs_full[g] if total_docs_full is None else (total_docs_full + per_group_docs_full[g])
            total_cites_full = per_group_cites_full[g] if total_cites_full is None else (total_cites_full + per_group_cites_full[g])

        with np.errstate(divide="ignore", invalid="ignore"):
            overall_full = (total_cites_full / total_docs_full.replace(0, np.nan))

        if cut_year is None:
            overall_vals = overall_full.reindex(full_range).to_numpy()
            overall_vals = pd.Series(overall_vals).ffill().to_numpy()
        else:
            before_mask = full_range < cut_year
            before_docs = total_docs_full.loc[before_mask].sum()
            before_cites = total_cites_full.loc[before_mask].sum()
            before_avg = np.nan if before_docs == 0 else before_cites / before_docs

            after_years = list(range(max(cut_year, y_min), y_max + 1))
            after_vals = overall_full.reindex(after_years).to_numpy()
            overall_vals = np.concatenate(([before_avg], pd.Series(after_vals).ffill().to_numpy()))

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=figsize)

    if plot_type == "line":
        for g in groups:
            ax1.plot(x_pos, avg_mat[g], marker=marker, linewidth=linewidth, color=colors[g], label=g)
    elif plot_type == "bar":
        n = max(1, len(groups))
        width_each = bar_width / n
        offset_start = -bar_width / 2 + width_each / 2
        for i, g in enumerate(groups):
            # Bars: keep NaN as NaN so missing years show as gaps (x remains continuous)
            ax1.bar(x_pos + offset_start + i * width_each, avg_mat[g], width=width_each, color=colors[g], label=g)
    else:
        raise ValueError("plot_type must be either \"line\" or \"bar\".")

    # Overall line overlay
    if plot_overall and overall_vals is not None:
        ax1.plot(x_pos, overall_vals, color="black", linewidth=linewidth, marker="o", label=overall_label, zorder=10)

    # Titles/labels
    ax1.set_title(title, fontsize=fontsize_title)
    ax1.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax1.set_ylabel(ylabel, fontsize=fontsize_labels)

    # X ticks & labels
    if wrap_xticks:
        labels = [textwrap.fill(str(lbl), wrap_width) for lbl in x_labels]
    else:
        labels = x_labels
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=xtick_rotation)

    ax1.legend(title=legend_title, loc="best")

    if filename_base:
        if save_dpi is None:
            save_plot(filename_base)
        else:
            save_plot(filename_base, dpi=save_dpi)

    if show:
        plt.show()
    else:
        plt.close()

    return ax1


# factor analysis plotting functions
# use conceptual_structure_analysis function from utilsbib

def plot_word_map(
    embeddings: np.ndarray,
    terms: list,
    labels: np.ndarray,
    figsize: tuple = (10, 8),
    title: str = "Word Map",
    filename_base: str = None,
    dpi: int = 600,
    cmap: str = "tab10",
    marker_size: int = 50,
    term_fontsize: int = 8,
    title_fontsize: int = 12,
    axis_label_fontsize: int = 10,
    tick_label_fontsize: int = 8,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    show_legend: bool = True,
) -> None:
    """
    Scatter plot of term embeddings colored by cluster labels.

    Colors correspond to cluster assignments; a legend (if enabled) maps
    colors to cluster IDs. Any terms whose labels overlap exactly will still
    only show one annotation—use different DR or jitter to separate.

    Term labels are normalised with ``utilsbib._balance_closing_parenthesis``,
    so unmatched "(" are balanced with a closing ")" (same behaviour as in
    ``utilsbib.count_occurrences``).

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (n_terms, n_components) with term embeddings.
    terms : list of str
        Term labels corresponding to each embedding.
    labels : np.ndarray
        Cluster labels for each term.
    show_legend : bool, default True
        Whether to display a legend for cluster colors.
    cmap : str, default "tab10"
        Matplotlib colormap for cluster coloring.
    marker_size : int, default 50
        Size of scatter markers.
    term_fontsize : int, default 8
        Font size for term annotations.
    title_fontsize : int, default 12
        Font size for the title.
    axis_label_fontsize : int, default 10
        Font size for axis labels.
    tick_label_fontsize : int, default 8
        Font size for tick labels.
    xlabel : str, default "Dim 1"
        Label for the x-axis.
    ylabel : str, default "Dim 2"
        Label for the y-axis.

    Examples
    --------
    >>> result = conceptual_structure_analysis(df)
    >>> plot_word_map(
    ...     result["term_embeddings"],
    ...     result["terms"],
    ...     result["term_labels"],
    ...     marker_size=100,
    ...     cmap="viridis",
    ...     show_legend=True,
    ... )
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    n_pts = embeddings.shape[0]

    # Work on a local copy of terms
    terms = list(terms)

    # Auto-fill missing term labels if terms list is too short
    if len(terms) < n_pts:
        fallback = [f"cluster_{lbl}" for lbl in labels[len(terms):]]
        terms.extend(fallback)

    # Balance parentheses in all labels (same logic as in count_occurrences)
    terms = [utilsbib._balance_closing_parenthesis(str(t)) for t in terms]

    # Validate lengths
    if len(terms) != n_pts or len(labels) != n_pts:
        raise ValueError(
            f"plot_word_map: embeddings ({n_pts}) must match len(terms) "
            f"({len(terms)}) and len(labels) ({len(labels)})"
        )

    # Ensure 2D for scatter
    if embeddings.ndim == 1:
        x = embeddings
        y = np.zeros_like(x)
        embeddings = np.vstack((x, y)).T
    elif embeddings.ndim == 2 and embeddings.shape[1] == 1:
        x = embeddings[:, 0]
        y = np.zeros_like(x)
        embeddings = np.vstack((x, y)).T

    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap=cmap,
        s=marker_size,
    )

    for i, term in enumerate(terms):
        plt.text(
            embeddings[i, 0],
            embeddings[i, 1],
            term,
            fontsize=term_fontsize,
        )

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=axis_label_fontsize)
    plt.ylabel(ylabel, fontsize=axis_label_fontsize)
    plt.xticks(fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.grid(False)

    if show_legend:
        handles, label_values = scatter.legend_elements()
        legend = plt.legend(
            handles,
            label_values,
            title="Cluster",
            fontsize=tick_label_fontsize,
        )
        legend.get_title().set_fontsize(axis_label_fontsize)

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()



def plot_topic_dendrogram(
    embeddings: np.ndarray,
    terms: list,
    method: str = "ward",
    metric: str = "euclidean",
    figsize: tuple = (10, 8),
    title: str = "Topic Dendrogram",
    filename_base: str | None = None,
    dpi: int = 600,
    xlabel: str = "Terms",
    ylabel: str = "Distance",
    title_fontsize: int = 12,
    axis_label_fontsize: int = 10,
    tick_label_fontsize: int = 8,
    leaf_label_fontsize: int = 8,
) -> None:
    """
    Plot a hierarchical clustering dendrogram of term embeddings.

    The function performs agglomerative clustering on the provided
    ``embeddings`` using :func:`scipy.cluster.hierarchy.linkage` and
    visualises the resulting tree with :func:`scipy.cluster.hierarchy.dendrogram`.

    Term labels are normalised in two ways:

    * If fewer labels than points are supplied, generic labels are added.
    * Each label is passed through ``utilsbib._balance_closing_parenthesis``
      so that unmatched ``"("`` are balanced with a closing ``")`` (same
      behaviour as in :func:`utilsbib.count_occurrences`).

    If ``filename_base`` is provided, the plot is saved (PNG, SVG, PDF)
    via :func:`save_plot`.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (n_terms, n_components) with term embeddings.
    terms : list of str
        Term labels corresponding to each embedding.
    method : str, default "ward"
        Linkage method used for hierarchical clustering.
    metric : str, default "euclidean"
        Distance metric passed to :func:`scipy.cluster.hierarchy.linkage`.
    figsize : tuple of float, default (10, 8)
        Figure size in inches.
    title : str, default "Topic Dendrogram"
        Plot title.
    filename_base : str or None, default None
        Base filename (without extension) for saving the figure. If None,
        the figure is not saved.
    dpi : int, default 600
        Resolution used when saving the figure.
    xlabel : str, default "Terms"
        Label for the x-axis.
    ylabel : str, default "Distance"
        Label for the y-axis.
    title_fontsize : int, default 12
        Font size for the title.
    axis_label_fontsize : int, default 10
        Font size for axis labels.
    tick_label_fontsize : int, default 8
        Font size for axis tick labels.
    leaf_label_fontsize : int, default 8
        Font size for the leaf labels (term names).

    Examples
    --------
    >>> result = conceptual_structure_analysis(df)
    >>> plot_topic_dendrogram(
    ...     result["term_embeddings"],
    ...     result["terms"],
    ...     xlabel="Terms",
    ...     ylabel="Distance",
    ...     title_fontsize=14,
    ...     axis_label_fontsize=12,
    ...     tick_label_fontsize=10,
    ...     leaf_label_fontsize=9,
    ... )
    """
    # Ensure 2D array
    emb = np.atleast_2d(embeddings)
    n_pts = emb.shape[0]

    # Work on a local copy of labels
    terms = list(terms)

    # Auto-fill missing term labels
    if len(terms) < n_pts:
        fallback = [f"item_{i}" for i in range(len(terms), n_pts)]
        terms.extend(fallback)

    # Balance parentheses in labels using the same logic as in count_occurrences
    terms = [utilsbib._balance_closing_parenthesis(str(t)) for t in terms]

    # Validate
    if len(terms) != n_pts:
        raise ValueError(
            f"plot_topic_dendrogram: need one label per point, "
            f"got {len(terms)} labels for {n_pts} points"
        )

    # Hierarchical clustering
    Z = linkage(emb, method=method, metric=metric)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(
        Z,
        labels=terms,
        leaf_rotation=90,
        leaf_font_size=leaf_label_fontsize,
        ax=ax,
    )
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.tick_params(axis='x', labelsize=tick_label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_label_fontsize)
    
    # Remove gridlines
    ax.grid(False)
    ax.set_facecolor("white")
    
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()



# timeseries item plots


def plot_item_time_stats(
    stats_df: pd.DataFrame,
    *,
    item_order: Optional[Iterable[str]] = None,
    item_col: Optional[str] = None,
    year_col: Optional[str] = None,                  # e.g., "Median Year"
    size_col: str = "Number of documents",
    color_col: str = "Citations per document",
    cmap: str = "viridis",
    size_min: float = 30.0,
    size_max: float = 300.0,
    line_alpha: float = 0.9,
    line_width: float = 1.0,
    y_label: Optional[str] = None,
    wrap_width: int = 50,
    savepath: Optional[str] = None,
    dpi: int = 600,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    label_fontsize: Optional[int] = None,
    tick_fontsize: Optional[int] = None,
    log_size: bool = False,
    # size legend (bottom, figure-level)
    size_legend: bool = True,
    size_legend_values: Optional[Sequence[float]] = None,
    size_legend_force_int: bool = True,
    size_legend_title: Optional[str] = None,         # if None, auto → "Number of documents"
    size_legend_ncol: Optional[int] = None,
    size_legend_frame: bool = True,
    size_legend_facecolor: str = "black",
    size_legend_edgecolor: str = "black",
    size_legend_bottom: float = 0.16,                # smaller default gap
    **kwargs,
):
    """
    Plot per-item timelines with bubble sizes (documents) and color (e.g., citations).

    Tiny change requested
    ---------------------
    The colorbar label is force-displayed as "Citations per document" whenever the
    color column is "citations_per_article" (or that phrase with any case/underscores).

    Other notes (unchanged)
    -----------------------
    • Accepts `year_col`; otherwise derives spans from ("first_year","last_year") or ("Q1","Q3").
    • Aliases: "n_docs" ⇄ "Number of documents", "citations_per_article" ⇄ "Citations per document".
    • Bottom size legend never touches the axes; exactly two unique sizes → two entries.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from textwrap import wrap

    # ---- legacy/aliases ----------------------------------------------------------------
    if "color_by" in kwargs:
        color_col = kwargs.pop("color_by")
    if "color_scheme" in kwargs:
        cmap = kwargs.pop("color_scheme")
    if "filename" in kwargs and not savepath:
        savepath = kwargs.pop("filename")
    else:
        kwargs.pop("filename", None)
    for _k in ("min_docs", "min_docs_per_item", "regex_filter", "top_n_year", "median_rounding"):
        kwargs.pop(_k, None)

    if stats_df.empty:
        raise ValueError("stats_df is empty; nothing to plot.")

    df = stats_df.copy()
    lower = {c.lower(): c for c in df.columns}

    def _resolve(name: str, aliases: list[str]) -> str:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
        for a in aliases:
            if a in df.columns:
                return a
            if a.lower() in lower:
                return lower[a.lower()]
        return name

    # ---- item column -------------------------------------------------------------------
    if item_col:
        item_col = _resolve(item_col, [])
        if item_col not in df.columns:
            raise KeyError(f'Item column "{item_col}" not found.')
    else:
        for cand in ("Item", "Topic", "Keyword", "Label", "Name"):
            if cand in df.columns or cand.lower() in lower:
                item_col = lower.get(cand.lower(), cand)
                break
        else:
            nonnum = [c for c in df.columns if not np.issubdtype(np.array(df[c].dropna()).dtype, np.number)]
            nonnum = [c for c in nonnum if c.lower() not in {"year","publication year","Year",
                                                             "median year","median_year","median"}]
            if not nonnum:
                raise KeyError('Could not infer the item label column; pass "item_col".')
            item_col = nonnum[0]

    # ---- aliases & presence -------------------------------------------------------------
    size_col  = _resolve(size_col, ["n_docs"])
    color_col = _resolve(color_col, ["citations_per_article"])

    alias_pairs = [
        ("Number of documents", "n_docs"),
        ("Citations per document", "citations_per_article"),
        ("H-index", "item_h_index"),
        ("Total citations", "total_citations"),
    ]
    for pref, alt in alias_pairs:
        if pref in df.columns and alt not in df.columns:
            df[alt] = df[pref]
        if alt in df.columns and pref not in df.columns:
            df[pref] = df[alt]

    if size_col not in df.columns:
        raise KeyError(f'Size column "{size_col}" not found.')
    if color_col not in df.columns:
        raise KeyError(f'Color column "{color_col}" not found.')

    # ---- year handling ------------------------------------------------------------------
    if year_col:
        year_col = _resolve(year_col, [])
        if year_col not in df.columns:
            raise KeyError(f'Year column "{year_col}" not found.')
        have_year = True
    else:
        yc = ["Year","year","Publication Year","publication_year","Median Year","median_year","Median","median"]
        year_col = next((lower.get(c.lower(), c) for c in yc if (c in df.columns or c.lower() in lower)), None)
        have_year = year_col is not None

    fy = next((c for c in df.columns if c.lower() == "first_year"), None)
    ly = next((c for c in df.columns if c.lower() == "last_year"), None)
    q1 = next((c for c in df.columns if c.lower() in {"q1","q_1"}), None)
    q3 = next((c for c in df.columns if c.lower() in {"q3","q_3"}), None)

    # ---- order -------------------------------------------------------------------------
    if item_order is None:
        if "item_total_docs" in df.columns:
            order = df.groupby(item_col)["item_total_docs"].max().sort_values(ascending=False).index.tolist()
        else:
            order = df.groupby(item_col)[size_col].sum(min_count=1).sort_values(ascending=False).index.tolist()
    else:
        order = list(item_order)

    ymap = {itm: i for i, itm in enumerate(order[::-1], start=1)}
    df = df[df[item_col].isin(ymap)].copy()
    if df.empty:
        raise ValueError("No rows left to plot after aligning with item_order.")
    df["y"] = df[item_col].map(ymap)

    # ---- spans -------------------------------------------------------------------------
    if fy and ly:
        span = df.groupby(item_col)[[fy, ly]].agg(first_year=(fy, "min"), last_year=(ly, "max")).reset_index()
    elif q1 and q3:
        span = df.groupby(item_col)[[q1, q3]].agg(first_year=(q1, "min"), last_year=(q3, "max")).reset_index()
    elif have_year:
        span = df.groupby(item_col)[year_col].agg(first_year="min", last_year="max").reset_index()
    else:
        raise KeyError('Provide "year_col" or include "first_year"/"last_year" or "Q1"/"Q3".')

    span = span.merge(df[[item_col, "y"]].drop_duplicates(item_col), on=item_col, how="left")

    # ---- sizes -------------------------------------------------------------------------
    raw_sizes = np.asarray(df[size_col].astype(float))
    ts = np.log1p(raw_sizes) if log_size else raw_sizes
    s_min, s_max = float(np.nanmin(ts)), float(np.nanmax(ts))
    if np.isfinite(s_max) and s_max > s_min:
        area = size_min + ((ts - s_min) / (s_max - s_min)) * (size_max - size_min)
    else:
        area = np.full_like(ts, size_min, dtype=float)

    # ---- figure ------------------------------------------------------------------------
    fig_h = max(4, 0.4 * len(order))
    if figsize is None:
        figsize = (10, fig_h)
    fig, ax = plt.subplots(figsize=figsize)

    ax.hlines(span["y"].to_numpy(),
              xmin=span["first_year"].to_numpy(),
              xmax=span["last_year"].to_numpy(),
              colors="black", linewidth=line_width, alpha=line_alpha, zorder=1)

    if have_year:
        sc = ax.scatter(df[year_col].astype(float).to_numpy(),
                        df["y"].to_numpy(),
                        s=area,
                        c=df[color_col].astype(float).to_numpy(),
                        cmap=cmap,
                        zorder=2)
    else:
        mid = (span["first_year"].astype(float) + span["last_year"].astype(float)) / 2.0
        merged = df[[item_col, "y", size_col, color_col]].drop_duplicates(item_col).merge(
            span[[item_col, "y"]].drop_duplicates(item_col), on=[item_col, "y"], how="left"
        )
        sc = ax.scatter(mid.to_numpy(),
                        merged["y"].to_numpy(),
                        s=(merged[size_col].astype(float) if size_col in merged else np.full(len(merged), size_min)),
                        c=(merged[color_col].astype(float) if color_col in merged else np.zeros(len(merged))),
                        cmap=cmap,
                        zorder=2)

    # ---- axes cosmetics ----------------------------------------------------------------
    yticks = list(ymap.values())
    ytexts = list(ymap.keys())
    if wrap_width and wrap_width > 0:
        ytexts = ["\n".join(wrap(str(t), width=wrap_width)) for t in ytexts]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytexts, fontsize=tick_fontsize)
    ax.set_xlabel("Year", fontsize=label_fontsize)
    ax.set_ylabel(y_label or item_col, fontsize=label_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(axis="x", labelsize=tick_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    ax.grid(False)

    # ---- colorbar with proper display label --------------------------------------------
    cbar = plt.colorbar(sc, ax=ax)
    _cc_norm = color_col.lower().replace("_", " ")
    cbar.set_label("Citations per document" if _cc_norm in {"citations per article", "citations per document"} else color_col)

    # ---- size legend (figure-level; smaller gap) ---------------------------------------
    if size_legend:
        if size_legend_values:
            legend_vals = [float(v) for v in size_legend_values if np.isfinite(v)]
        else:
            finite = raw_sizes[np.isfinite(raw_sizes)]
            if finite.size == 0:
                legend_vals = [0.0]
            else:
                uniq = np.unique(finite)
                if uniq.size == 1:
                    legend_vals = [float(uniq.min())]
                elif uniq.size == 2:
                    legend_vals = [float(uniq.min()), float(uniq.max())]
                else:
                    vmin, vmax = float(finite.min()), float(finite.max())
                    legend_vals = [vmin, (vmin + vmax) / 2.0, vmax]
        if size_legend_force_int:
            legend_vals = [int(round(v)) for v in legend_vals]

        def _to_area(v: float) -> float:
            tv = np.log1p(v) if log_size else float(v)
            if not (np.isfinite(s_max) and s_max > s_min):
                return size_min
            return size_min + ((tv - s_min) / (s_max - s_min)) * (size_max - size_min)

        handles = [
            ax.scatter([], [], s=_to_area(v),
                       facecolors=size_legend_facecolor, edgecolors=size_legend_edgecolor)
            for v in legend_vals
        ]
        labels = [f"{v:d}" if isinstance(v, int) else f"{v:g}" for v in legend_vals]
        ncol = size_legend_ncol or max(1, len(handles))

        # auto title → human-friendly for n_docs
        if size_legend_title is None:
            if {"n_docs", "Number of documents"} & set(map(str, df.columns)) or size_col.lower() in {"n_docs", "number of documents"}:
                legend_title = "Number of documents"
            else:
                legend_title = size_col
        else:
            legend_title = size_legend_title

        bottom_reserved = float(np.clip(size_legend_bottom, 0.10, 0.22))
        anchor_y = bottom_reserved * 0.55
        fig.tight_layout(rect=[0.04, bottom_reserved, 0.98, 0.98])

        fig.legend(
            handles, labels,
            title=legend_title,
            loc="lower center",
            bbox_to_anchor=(0.5, anchor_y),
            bbox_transform=fig.transFigure,
            ncol=ncol,
            frameon=size_legend_frame,
            framealpha=1.0,
            borderpad=0.5,
            handletextpad=0.8,
            columnspacing=1.0,
        )
    else:
        plt.tight_layout()

    if savepath:
        save_plot(savepath, dpi=dpi)

    return fig

# spectrogram

def plot_reference_spectrogram(spectrogram_df, title="Spectroscopy of Science", save_path=None, group_by_decade=False, show_grid=False, fontsize=12):
    """
    Plots the spectrogram of cited years or decades.

    Args:
        spectrogram_df (pd.DataFrame): DataFrame with index as years and citation counts.
        title (str): Plot title.
        save_path (str or None): If provided, saves the plot using save_plot().
        group_by_decade (bool): If True, groups citations by decade.
    """
    df = spectrogram_df.copy()
    if group_by_decade:
        df["Decade"] = (df.index // 10) * 10
        df = df.groupby("Decade")["Citations"].sum().reset_index()
        x = df["Decade"]
        y = df["Citations"]
        xlabel = "Cited Decade"
    else:
        x = df.index
        y = df["Citations"]
        xlabel = "Cited Year"

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=2)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel("Number of Citations", fontsize=fontsize)
    plt.grid(show_grid)
    if save_path:
        save_plot(save_path)
    plt.show()
    
def plot_reference_correlation(plot_df, xlabel=None, ylabel=None, title="Reference Correlation", show_corr=True, save_path=None, show_grid=False, fontsize=12):
    """
    Plots a scatterplot using two columns from a DataFrame with optional correlation display.

    Args:
        df (pd.DataFrame): DataFrame with exactly two columns: x and y.
        xlabel (str or None): Label for x-axis (default is column name).
        ylabel (str or None): Label for y-axis (default is column name).
        title (str): Plot title.
        show_corr (bool): Whether to display Pearson correlation in title.
        save_path (str or None): If provided, saves the plot.
    """
    x = plot_df.iloc[:, 0]
    y = plot_df.iloc[:, 1]
    xlabel = xlabel or plot_df.columns[0]
    ylabel = ylabel or plot_df.columns[1]

    corr_text = ""
    if show_corr:
        r, _ = pearsonr(x, y)
        corr_text = f" (r = {r:.2f})"

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5)
    min_val, max_val = min(min(x), min(y)), max(max(x), max(y))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(f"{title}{corr_text}", fontsize=fontsize)
    plt.grid(show_grid)
    if save_path:
        save_plot(save_path)
    plt.show()
    
# Scientific production by group

def plot_stacked_production_by_group(
    stats: pd.DataFrame,
    group_colors: dict[str, str] | None = None,
    filename_base: str | None = None,
    figsize: tuple = (10, 6),
    cut_year: int | None = None,
    year_span: tuple[int, int] | None = None,
    citation_mode: str = "group",
    font_size: int = 12,
    xlabel: str = "Year",
    ylabel: str = "Number of documents",
    citation_label: str = "Cumulative Citations",
    legend_title: str = "Group",
    grid: bool = False,
) -> plt.Axes:
    """
    Plot a stacked bar chart of document counts by group per year, with optional
    pre-cut aggregation and cumulative citation lines.

    Key fixes
    ---------
    - "together" citations: build per-group cumulative series aligned to the full
      x-axis, forward-fill them (starting from 0), *then* sum across groups, and
      apply `cummax()` to guarantee nondecreasing totals.
    - "group" citations: forward-fill (starting from 0) and `cummax()` each group's
      cumulative series, avoiding drops on years with no row.

    Parameters
    ----------
    stats : pd.DataFrame
        Wide-format statistics from get_scientific_production_by_group. Must contain
        "Year", "Number of documents <GROUP>", and optionally
        "Cumulative Citations <GROUP>" columns.
    group_colors : dict[str, str], optional
        Mapping from group name to hex color code.
    filename_base : str, optional
        Base filename (without extension) to save via save_plot(...).
    figsize : tuple, default (10, 6)
        Figure size in inches.
    cut_year : int, optional
        Aggregate all years strictly before this into a single "Before {cut_year}" bin.
    year_span : (int, int), optional
        Inclusive span (start_year, end_year) to keep on the x-axis.
    citation_mode : {"group", "together", None}, default "group"
        Plot per-group cumulative lines, a single aggregated line, or none.
    font_size : int, default 12
        Base font size for labels and ticks.
    xlabel : str, default "Year"
        X-axis label.
    ylabel : str, default "Number of documents"
        Y-axis (bars) label.
    citation_label : str, default "Cumulative Citations"
        Y-axis (right) label for citations.
    legend_title : str, default "Group"
        Legend title.
    grid : bool, default False
        Whether to show grid lines.

    Returns
    -------
    matplotlib.axes.Axes
        Primary axes of the stacked bar chart.
    """
    if "Year" not in stats.columns:
        raise ValueError("The stats DataFrame must contain a 'Year' column.")

    data = stats.copy()

    # Identify columns and group names
    doc_cols = [c for c in data.columns if c.startswith("Number of documents ")]
    cit_cols = [c for c in data.columns if c.startswith("Cumulative Citations ")]

    doc_map = {c.replace("Number of documents ", ""): c for c in doc_cols}
    cit_map = {c.replace("Cumulative Citations ", ""): c for c in cit_cols}
    groups = list(doc_map.keys())

    # Colors
    if group_colors is None:
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    # Year processing & optional pre-cut aggregation
    data["Year"] = data["Year"].astype(object)
    if cut_year is not None:
        label = f"Before {cut_year}"
        mask_pre = pd.to_numeric(data["Year"], errors="coerce") < cut_year
        data.loc[mask_pre, "Year"] = label
        agg_dict = {c: "sum" for c in doc_cols}
        agg_dict.update({c: "max" for c in cit_cols})  # last level within the bin
        data = data.groupby("Year", as_index=False).agg(agg_dict)

    # Optional year span filtering (keep "Before ..." if present)
    if year_span is not None:
        numeric_year = pd.to_numeric(data["Year"], errors="coerce")
        keep = numeric_year.between(year_span[0], year_span[1])
        data = data[keep | data["Year"].astype(str).str.startswith("Before")]

    # X order: "Before ..." first, then numeric ascending
    data["Year"] = data["Year"].astype(str)
    seen = list(dict.fromkeys(data["Year"]))  # preserve appearance order
    before = [y for y in seen if y.startswith("Before")]
    nums = sorted([y for y in seen if not y.startswith("Before")], key=lambda x: int(x))
    x_order = before + nums

    # Plot bars
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.65)

    bottom = np.zeros(len(x_order), dtype=float)
    for g in groups:
        col = doc_map[g]
        s = (
            data.set_index("Year")[col]
            .reindex(x_order)
            .astype(float)
            .fillna(0.0)
        )
        vals = s.values
        ax.bar(x_order, vals, bottom=bottom, label=g, color=group_colors.get(g))
        bottom += vals

    # Helper to build a *monotone* cumulative series per group on the unified x-axis
    def _aligned_cumulative_series(group_name: str) -> pd.Series:
        """
        Build a per-group cumulative citations series aligned to x_order,
        forward-filled from 0 and forced to be nondecreasing.
        """
        if group_name not in cit_map:
            # No cumulative column for this group -> all zeros
            return pd.Series(0.0, index=x_order, dtype=float)
        s = data.set_index("Year")[cit_map[group_name]].reindex(x_order)
        s = pd.to_numeric(s, errors="coerce").ffill().fillna(0.0)
        # Safety: enforce nondecreasing (handles any upstream inconsistencies)
        return s.cummax()

    # Plot cumulative citation lines
    if citation_mode in ("group", "together") and len(groups) > 0:
        ax2 = ax.twinx()

        if citation_mode == "group":
            for g in groups:
                s_g = _aligned_cumulative_series(g)
                ax2.plot(x_order, s_g.values, linestyle="--", marker="o", color=group_colors.get(g))
        elif citation_mode == "together":
            # Build aligned, forward-filled series for each group, then sum by year
            aligned = {g: _aligned_cumulative_series(g) for g in groups}
            if aligned:
                mat = pd.DataFrame(aligned)  # index = x_order
                s_total = mat.sum(axis=1).cummax()
                ax2.plot(x_order, s_total.values, linestyle="--", marker="o", color="black")

        ax2.set_ylabel(citation_label, fontsize=font_size)
        ax2.tick_params(axis="y", labelsize=font_size)

    # Cosmetics
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size, rotation=90)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(grid)  # Control grid visibility
    ax.legend(
        title=legend_title,
        fontsize=font_size,
        title_fontsize=font_size,
        loc="upper left",
        bbox_to_anchor=(1.15, 1),
        borderaxespad=0,
    )

    plt.tight_layout()

    if filename_base:
        save_plot(filename_base)

    return ax


# Plotting of differences between two frequency distribvutions

def plot_count_differences(df: pd.DataFrame,
                            orientation: str = "auto",
                            title: str = "Relative Differences in Item Counts",
                            xlabel: str = "Percentage Point Difference (pp)",
                            ylabel: str = "Item",
                            color_pos: str = "skyblue",
                            color_neg: str = "lightcoral",
                            alpha_cap: float = 0.8,
                            grid: bool = False,
                            show_zero_line: bool = True,
                            annotate: bool = True,
                            rotation: int = 45,
                            label_offset: float = 0.1,
                            margin_ratio: float = 0.1,
                            figsize: tuple = (10, 6)) -> None:
    """
    Plot percentage point differences using horizontal or vertical bars.
    """
    df_plot = df.copy()

    if orientation == "auto":
        try:
            years = pd.to_numeric(df_plot.index, errors="coerce")
            is_time_series = years.notna().all() and years.is_monotonic_increasing
            orientation = "vertical" if is_time_series else "horizontal"
        except:
            orientation = "horizontal"

    df_plot["Color"] = np.where(df_plot["PP_Diff"] >= 0, color_pos, color_neg)

    if orientation == "horizontal":
        df_plot = df_plot.sort_values("PP_Diff", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    if orientation == "horizontal":
        bars = ax.barh(df_plot.index.astype(str), df_plot["PP_Diff"],
                       color=df_plot["Color"], alpha=alpha_cap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        x_vals = df_plot["PP_Diff"]
        x_margin = (x_vals.max() - x_vals.min()) * margin_ratio
        ax.set_xlim(x_vals.min() - x_margin, x_vals.max() + x_margin)

        if annotate:
            for bar in bars:
                value = bar.get_width()
                offset = label_offset if value > 0 else -label_offset
                ax.text(value + offset, bar.get_y() + bar.get_height() / 2,
                        f"{value:.1f} pp", va="center",
                        ha="left" if value > 0 else "right")

    else:
        bars = ax.bar(df_plot.index.astype(str), df_plot["PP_Diff"],
                      color=df_plot["Color"], alpha=alpha_cap)
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
        plt.xticks(rotation=rotation, ha="right")
        y_vals = df_plot["PP_Diff"]
        y_margin = (y_vals.max() - y_vals.min()) * margin_ratio
        ax.set_ylim(y_vals.min() - y_margin, y_vals.max() + y_margin)

        if annotate:
            for bar in bars:
                value = bar.get_height()
                offset = label_offset if value > 0 else -label_offset
                ax.text(bar.get_x() + bar.get_width() / 2, value + offset,
                        f"{value:.1f} pp", ha="center",
                        va="bottom" if value > 0 else "top")

    if show_zero_line:
        if orientation == "vertical":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        else:
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    if grid:
        ax.grid(True, linestyle=":", linewidth=0.5)
    else:
        ax.grid(False)

    ax.set_title(title)
    plt.tight_layout()
    


from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.collections import PathCollection


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    size_col: Optional[str] = None,
    color_col: Optional[str] = None,
    marker_col: Optional[str] = None,
    marker_sequence: Optional[List[str]] = None,
    label_col: Optional[str] = None,
    error_x: Optional[Union[str, Sequence[float]]] = None,
    error_y: Optional[Union[str, Sequence[float]]] = None,
    dropna: bool = True,
    fig_size: Tuple[float, float] = (8, 6),
    style_sheet: Optional[str] = None,
    grid: Union[bool, Dict[str, Any]] = False,
    alpha: float = 1.0,
    edge_color: Optional[str] = None,
    edge_width: float = 0.0,
    zorder: int = 2,
    equal_aspect: bool = False,
    tick_params: Optional[Dict[str, Any]] = None,
    formatter: Optional[Callable] = None,
    x_scale: str = "log",
    y_scale: str = "log",
    log_base: Tuple[int, int] = (10, 10),
    size_scale: str = "linear",
    max_size: float = 300,
    colormap: str = "viridis",
    lines: Optional[List[Tuple[float, float]]] = None,
    symmetric_lines: bool = True,
    mean_line: bool = False,
    median_line: bool = False,
    identity_line: bool = False,
    mean_marker: bool = False,
    adjust_kwargs: Optional[Dict[str, Any]] = None,
    highlight_points: Optional[Union[Sequence[int], Sequence[bool]]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    legend: bool = True,
    legend_kwargs: Optional[Dict[str, Any]] = None,
    filename: str = "scatter",
    dpi: int = 600,
    show: bool = True,
    *,
    force_integer_size_legend: bool = True,
    **kwargs,
) -> None:
    """
    Create a scatter plot with optional size/color/marker encodings, helper lines, and smart legends.
    Uses adjustText to de-overlap point labels when label_col is provided.

    Parameters
    ----------
    force_integer_size_legend : bool, default True
        Force integer labels in the size legend (rounded). Set False to allow decimal labels.
    formatter : Callable, optional
        Custom tick formatter. If None, and the data max on a *linear* axis is < 1000,
        that axis is formatted with plain numbers (no scientific notation).
    adjust_kwargs : dict, optional
        Extra kwargs for adjustText. Reasonable defaults are provided:
        {"only_move": {"points": "y", "text": "xy"}, "expand_points": (1.2,1.2), "expand_text": (1.2,1.2),
         "force_points": 0.1, "force_text": 0.2, "lim": 300}.
    """
    # If adjustText isn't globally imported, try to import it here.
    try:
        from adjustText import adjust_text  # type: ignore
    except Exception:  # pragma: no cover
        adjust_text = None  # Fallback: labels won't be adjusted if unavailable

    df_plot = df.copy()

    # Only include string-like columns for dropna subset
    cols: List[str] = [c for c in [x, y, size_col, color_col, marker_col, label_col] if isinstance(c, str)]
    if isinstance(error_x, str):
        cols.append(error_x)
    if isinstance(error_y, str):
        cols.append(error_y)

    if dropna and cols:
        df_plot.dropna(subset=cols, inplace=True)

    if style_sheet:
        plt.style.use(style_sheet)

    fig, ax = plt.subplots(figsize=fig_size)
    
    # Explicitly disable grid unless requested
    if grid:
        ax.grid(**grid) if isinstance(grid, dict) else ax.grid(True)
    else:
        ax.grid(False)
        ax.set_facecolor("white")

    if x_scale == "log":
        ax.set_xscale("log", base=log_base[0])
    else:
        ax.set_xscale(x_scale)

    if y_scale == "log":
        ax.set_yscale("log", base=log_base[1])
    else:
        ax.set_yscale(y_scale)

    n = len(df_plot)

    # Sizes
    if size_col:
        raw = df_plot[size_col].astype(float)
        if size_scale == "log":
            raw = np.log(raw.clip(lower=np.nextafter(0, 1)))
        max_raw = float(raw.max()) if len(raw) else 0.0
        sizes = (raw / max_raw) * max_size if max_raw > 0 else np.full(len(raw), max_size)
    else:
        sizes = np.full(n, max_size)

    # Colors
    norm_c: Optional[Normalize] = None
    use_colorbar = False
    if color_col and pd.api.types.is_numeric_dtype(df_plot[color_col]):
        norm_c = Normalize(vmin=float(df_plot[color_col].min()), vmax=float(df_plot[color_col].max()))
        color_vals = norm_c(df_plot[color_col].to_numpy(dtype=float))
        use_colorbar = True
    elif color_col:
        color_vals = df_plot[color_col]
    else:
        color_vals = None

    base_kwargs: Dict[str, Any] = {
        "s": sizes,
        "cmap": colormap,
        "alpha": alpha,
        "edgecolors": edge_color,
        "linewidths": edge_width,
        "zorder": zorder,
    }
    if color_vals is not None:
        base_kwargs["c"] = color_vals

    # Collect scatter artists so adjustText can avoid the points
    scatter_artists: List[PathCollection] = []
    marker_handles: List[Tuple[str, Any]] = []

    if marker_col:
        default_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
        markers = marker_sequence or default_markers
        for i, grp in enumerate(df_plot[marker_col].unique()):
            mask = df_plot[marker_col] == grp
            kwargs_grp = base_kwargs.copy()
            if color_vals is not None:
                kwargs_grp["c"] = np.asarray(color_vals)[mask.to_numpy()]
            kwargs_grp["s"] = np.asarray(sizes)[mask.to_numpy()]
            m = markers[i % len(markers)]
            sc = ax.scatter(df_plot.loc[mask, x], df_plot.loc[mask, y], marker=m, **kwargs_grp, **kwargs)
            scatter_artists.append(sc)
            marker_handles.append((m, grp))
    else:
        kw = dict(kwargs)
        kw.pop("cmap", None)
        sc = ax.scatter(df_plot[x], df_plot[y], **base_kwargs, **kw)
        scatter_artists.append(sc)

    # Error bars
    if error_x is not None or error_y is not None:
        xerr = df_plot[error_x] if isinstance(error_x, str) else error_x
        yerr = df_plot[error_y] if isinstance(error_y, str) else error_y
        ax.errorbar(df_plot[x], df_plot[y], xerr=xerr, yerr=yerr, fmt="none", alpha=alpha * 0.5)

    # Set axis limits based on data min/max
    if not df_plot.empty:
        data_xmin = float(df_plot[x].min())
        data_xmax = float(df_plot[x].max())
        data_ymin = float(df_plot[y].min())
        data_ymax = float(df_plot[y].max())

        # Add some padding
        if x_scale == "log":
            # Use multiplicative padding for log scale
            pad_x_min = data_xmin / 1.1
            pad_x_max = data_xmax * 1.1
        else:
            # Use additive padding for linear scale
            pad_x = (data_xmax - data_xmin) * 0.05  # 5% padding
            pad_x_min = data_xmin - (pad_x or 0.1) # Add fallback for 0 range
            pad_x_max = data_xmax + (pad_x or 0.1)

        if y_scale == "log":
            pad_y_min = data_ymin / 1.1
            pad_y_max = data_ymax * 1.1
        else:
            pad_y = (data_ymax - data_ymin) * 0.05  # 5% padding
            pad_y_min = data_ymin - (pad_y or 0.1)
            pad_y_max = data_ymax + (pad_y or 0.1)
        
        # Handle cases where min/max are the same
        if pad_x_min == pad_x_max:
            pad_x_min *= 0.9
            pad_x_max *= 1.1
        if pad_y_min == pad_y_max:
            pad_y_min *= 0.9
            pad_y_max *= 1.1

        ax.set_xlim(pad_x_min, pad_x_max)
        ax.set_ylim(pad_y_min, pad_y_max)

    # Helper for size-legend label formatting
    def _format_size_labels(vals: Sequence[float]) -> List[str]:
        if force_integer_size_legend:
            return [str(int(round(float(v)))) for v in vals]
        return [str(int(v)) if float(v).is_integer() else f"{float(v):.2f}" for v in vals]

    # Legends
    if legend:
        size_flag = size_col is not None
        marker_flag = marker_col is not None

        if size_flag and not marker_flag:
            vals = [float(df_plot[size_col].min()), float(df_plot[size_col].median()), float(df_plot[size_col].max())]
            base_max = float((np.log(df_plot[size_col]) if size_scale == "log" else df_plot[size_col]).max())
            sizes_leg = [
                ((np.log(v) if size_scale == "log" else v) / base_max) * max_size if base_max > 0 else max_size
                for v in vals
            ]
            handles = [Line2D([], [], linestyle="", marker="o", markersize=np.sqrt(s), color="black", alpha=alpha)
                       for s in sizes_leg]
            labs = _format_size_labels(vals)
            ax.legend(handles, labs, title=size_col, loc="lower right", frameon=True, edgecolor="black",
                      **(legend_kwargs or {}))

        elif marker_flag and not size_flag:
            handles = [Line2D([], [], linestyle="", marker=mh[0], color="black", markersize=6)
                       for mh in marker_handles]
            labs = [grp for _, grp in marker_handles]
            ax.legend(handles, labs, title=marker_col, loc="lower right", frameon=True, edgecolor="black",
                      **(legend_kwargs or {}))

        else:
            if size_flag:
                vals = [float(df_plot[size_col].min()), float(df_plot[size_col].median()), float(df_plot[size_col].max())]
                base_max = float((np.log(df_plot[size_col]) if size_scale == "log" else df_plot[size_col]).max())
                sizes_leg = [
                    ((np.log(v) if size_scale == "log" else v) / base_max) * max_size if base_max > 0 else max_size
                    for v in vals
                ]
                handles = [Line2D([], [], linestyle="", marker="o", markersize=np.sqrt(s), color="black", alpha=alpha)
                           for s in sizes_leg]
                labs = _format_size_labels(vals)
                fig.legend(
                    handles, labs, title=size_col, loc="lower center", bbox_to_anchor=(0.25, -0.02),
                    ncol=len(handles), frameon=True, edgecolor="black"
                )
            if marker_flag:
                handles = [Line2D([], [], linestyle="", marker=m, color="black", markersize=6)
                           for m, _ in marker_handles]
                labs = [grp for _, grp in marker_handles]
                fig.legend(
                    handles, labs, title=marker_col, loc="lower center", bbox_to_anchor=(0.75, -0.02),
                    ncol=len(handles), frameon=True, edgecolor="black"
                )

    # Colorbar
    if color_col and use_colorbar and norm_c is not None:
        sm = ScalarMappable(norm=norm_c, cmap=colormap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(color_col)

    # Helper lines
    xmin, xmax = ax.get_xlim()
    xs = np.linspace(xmin, xmax, 200)

    if mean_line:
        mx, my = float(df_plot[x].mean()), float(df_plot[y].mean())
        ax.axvline(mx, linestyle="--", color="gray", zorder=1)
        ax.axhline(my, linestyle="--", color="gray", zorder=1)

    if median_line:
        mdx, mdy = float(df_plot[x].median()), float(df_plot[y].median())
        ax.axvline(mdx, linestyle=":", color="gray", zorder=1)
        ax.axhline(mdy, linestyle=":", color="gray", zorder=1)

    if identity_line:
        ax.plot(xs, xs, linestyle="-", color="black", zorder=1)

    if lines:
        for slope, intercept in lines:
            ax.plot(xs, slope * xs + intercept, linestyle="-", color="gray", zorder=1)
            if symmetric_lines and slope != 0:
                ax.plot(xs, (xs - intercept) / slope, linestyle="--", color="gray", zorder=1)

    if mean_marker and mean_line:
        mx, my = float(df_plot[x].mean()), float(df_plot[y].mean())
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        y_offset = (ymax - ymin) * 0.02
        ax.text(
            mx, ymin, f"$\\mu_x={mx:.2f}$", color="black", fontsize=kwargs.get("label_font_size", 6),
            ha="center", va="bottom", zorder=3,
        )
        ax.text(
            xmin, my + y_offset, f"$\\mu_y={my:.2f}$", color="black", fontsize=kwargs.get("label_font_size", 6),
            ha="left", va="bottom", zorder=3,
        )

    # Highlight points (by mask or index list)
    if highlight_points is not None:
        hp = df_plot.loc[highlight_points]
        ax.scatter(hp[x], hp[y], s=kwargs.get("mean_marker_size", 6), color="red", zorder=4)

    # Point labels + robust adjustText to prevent overlaps
    if label_col:
        texts = []
        lfsize = kwargs.get("label_font_size", 6)
        for _, row in df_plot.iterrows():
            txt = ax.text(row[x], row[y], str(row[label_col]), fontsize=lfsize, zorder=5)
            texts.append(txt)

        # Sensible, stronger defaults; user may override via adjust_kwargs
        adj_defaults = {
            "only_move": {"points": "y", "text": "xy"},
            "expand_points": (1.2, 1.2),
            "expand_text": (1.2, 1.2),
            "force_points": 0.1,
            "force_text": 0.2,
            "lim": 300,
            "precision": 0.01,
        }
        merged_adjust = {**adj_defaults, **(adjust_kwargs or {})}

        if adjust_text is not None:
            # Pass plotted scatter artists so labels avoid points, too.
            adjust_text(texts, ax=ax, add_objects=scatter_artists, **merged_adjust)

    if title:
        ax.set_title(title)
    if subtitle:
        ax.set_title(subtitle, fontsize=kwargs.get("label_font_size", 6), style="italic")

    if caption:
        fig.text(0.5, -0.3, caption, ha="center", fontsize=kwargs.get("label_font_size", 6))

    ax.set_xlabel(x_label or x)
    ax.set_ylabel(y_label or y)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    if tick_params:
        ax.tick_params(**tick_params)

    # Axis formatting
    if formatter is not None:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    else:
        xdata_max = float(np.nanmax(df_plot[x].to_numpy())) if len(df_plot) else 0.0
        ydata_max = float(np.nanmax(df_plot[y].to_numpy())) if len(df_plot) else 0.0

        def _plain_number(v, pos):
            if not np.isfinite(v) or v == 0:
                return ""
            # **CHANGED**: Threshold from 10000 to 1000
            if abs(v) < 1000:
                iv = int(round(v))
                return str(iv) if abs(v - iv) < 1e-9 else f"{v:.6g}"
            return f"{v:.6g}"

        # **CHANGED**: Only apply plain formatter to NON-log axes
        if x_scale != "log" and xdata_max < 1000:
            ax.xaxis.set_major_formatter(FuncFormatter(_plain_number))
            ax.xaxis.set_minor_formatter(NullFormatter())
        # **CHANGED**: Only apply plain formatter to NON-log axes
        if y_scale != "log" and ydata_max < 1000:
            ax.yaxis.set_major_formatter(FuncFormatter(_plain_number))
            ax.yaxis.set_minor_formatter(NullFormatter())

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.18, right=0.85)

    if filename is not None:
        save_plot(filename)

    if show:
        plt.show()
    plt.close(fig)


import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale


def _select_top_binary(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Return the top-n *columns* of a binary indicator DataFrame by column sum
    (descending). If n >= number of columns, returns df unchanged.
    """
    if n is None or n >= df.shape[1]:
        return df
    # Robust to non-bool dtypes and preserves original order on ties.
    sums = df.astype("uint8").sum(axis=0)
    top_cols = sums.sort_values(ascending=False).index[:n]
    return df.loc[:, top_cols]


def prepare_for_sankey(
    dataframes: List[pd.DataFrame],
    top_n: Union[int, Sequence[int]] = 10,
    label_maps: Optional[Dict[str, str]] = None,
    color_series: Optional[pd.Series] = None,
    color_func: Callable[[pd.Series], float] = np.mean,
    all_pairs: bool = False
) -> Tuple[pd.DataFrame, List[str], List[float], List[int], List[int]]:
    """
    Prepare link and node data for a Sankey diagram from multiple binary indicator DataFrames.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        Binary indicator matrices (docs × concepts). Each DataFrame represents a field.
    top_n : int or sequence of int, default 10
        Number of top columns to keep per DataFrame (by column sum). If a sequence,
        it is applied per field in order.
    label_maps : dict, optional
        Mapping for *display* labels (does not affect node identity).
    color_series : pandas.Series, optional
        Series indexed like the rows of the input DataFrames, used to compute node colors.
    color_func : callable, default numpy.mean
        Aggregation over the values of `color_series` for rows where a node==1.
        Vectorized mean over non-NaN values is used when possible; otherwise falls back
        to a safe per-column aggregation.
    all_pairs : bool, default False
        If True, compute links for all unique field pairs (i < j). If False, only links
        between consecutive fields are computed.

    Returns
    -------
    links_df : pandas.DataFrame
        Columns: ["source", "target", "value"] with node indices and link weights.
    labels : list of str
        Display labels, optionally renamed by `label_maps`.
    color_values : list of float
        Aggregated color values per node (empty list if `color_series` is None).
    selected_counts : list of int
        Number of columns kept per field after `top_n` selection.
    group_ids : list of int
        Field index (0..k-1) for each node, aligned with `labels`.
    """
    # --- Normalize and select top-n columns per field --------------------------
    if top_n is None:
        selected_counts = [df.shape[1] for df in dataframes]
    elif isinstance(top_n, int):
        selected_counts = [min(top_n, df.shape[1]) for df in dataframes]
    else:
        selected_counts = [min(n, df.shape[1]) for df, n in zip(dataframes, list(top_n))]

    selected_fields: List[pd.DataFrame] = []
    for df, n in zip(dataframes, selected_counts):
        # Ensure binary uint8 to shrink memory and speed dot-products.
        d = df.astype("uint8", copy=False)
        d = _select_top_binary(d, n)
        selected_fields.append(d)

    # --- Build node identity with (group_id, label) to avoid collisions --------
    raw_labels: List[str] = []
    group_ids: List[int] = []
    node_keys: List[Tuple[int, str]] = []
    for gid, df in enumerate(selected_fields):
        cols = list(df.columns)
        raw_labels.extend(cols)
        group_ids.extend([gid] * len(cols))
        node_keys.extend((gid, c) for c in cols)

    label_to_idx = {key: idx for idx, key in enumerate(node_keys)}

    # --- Compute color values per node (vectorized when possible) --------------
    color_values: List[float] = []
    if color_series is not None and len(raw_labels) > 0:
        # Align index once
        s = color_series.reindex(selected_fields[0].index)
        s_values = s.to_numpy()
        s_isfinite = np.isfinite(s_values)

        def _vectorized_mean(field_df: pd.DataFrame) -> np.ndarray:
            # counts of non-NaN contributing rows
            counts = field_df.astype("float32").T.dot(s_isfinite.astype("float32"))
            # sums over non-NaN (replace NaN with 0 so they don't contribute)
            sums = field_df.astype("float32").T.dot(np.nan_to_num(s_values, nan=0.0).astype("float32"))
            with np.errstate(invalid="ignore", divide="ignore"):
                means = sums / counts
            return means

        can_vectorize = color_func in (np.mean, np.nanmean) or getattr(color_func, "__name__", "") in {"mean", "nanmean"}

        if can_vectorize:
            for df in selected_fields:
                means = _vectorized_mean(df)
                color_values.extend(means.tolist())
        else:
            # Safe fallback for arbitrary aggregators
            for df in selected_fields:
                for col in df.columns:
                    mask = df[col].astype(bool).to_numpy()
                    vals = s_values[mask]
                    vals = vals[np.isfinite(vals)]
                    color_values.append(float(color_func(pd.Series(vals))) if vals.size else np.nan)

    # --- Compute links via pairwise column-wise dot-products -------------------
    link_frames = []
    k = len(selected_fields)
    pairs = itertools.combinations(range(k), 2) if all_pairs else zip(range(k - 1), range(1, k))
    for i, j in pairs:
        # counts_{i->j}[ci, cj] = co-occurrence count
        counts = selected_fields[i].T.dot(selected_fields[j])  # shape: (cols_i × cols_j)
        if isinstance(counts, pd.DataFrame):
            counts = counts.astype("int64")
            stacked = counts.stack()
            if stacked.empty:
                continue
            stacked = stacked[stacked > 0]  # drop zero links early
            if stacked.empty:
                continue
            src_labels = stacked.index.get_level_values(0)
            tgt_labels = stacked.index.get_level_values(1)
            links = pd.DataFrame(
                {
                    "source": [label_to_idx[(i, s)] for s in src_labels],
                    "target": [label_to_idx[(j, t)] for t in tgt_labels],
                    "value": stacked.to_numpy(),
                }
            )
            link_frames.append(links)

    if link_frames:
        links_df = pd.concat(link_frames, ignore_index=True)
    else:
        links_df = pd.DataFrame({"source": [], "target": [], "value": []}, dtype=int)

    # --- Final display labels (mapped at the end so indices remain stable) -----
    rename = (lambda x: label_maps.get(x, x)) if label_maps else (lambda x: x)
    final_labels = [rename(lbl) for lbl in raw_labels]

    return links_df, final_labels, color_values, selected_counts, group_ids


def plot_sankey(
    links_df: pd.DataFrame,
    labels: list[str],
    color_values: list[float] | None = None,
    group_ids: list[int] | None = None,
    field_names: list[str] | None = None,
    save_png: str | None = None,
    save_html: str | None = None,
    colorscale: str | list = "Viridis",
    colorbar_title: str = "",
    side_margin: int = 24,
    top_margin: int = 56,
    shorten_labels: bool = False,
    max_label_chars: int = 22,
    domain_pad_x: float = 0.02,
    domain_pad_y_top: float = 0.05,
    domain_pad_y_bottom: float = 0.00,
) -> Any:
    """
    Build a tidy Sankey with:
    - locked columns by field (arrangement='fixed'),
    - field labels under the plot (above the colorbar),
    - horizontal colorbar at the bottom,
    - visible top/side margins, and
    - internal headroom using Sankey domain padding so the top gap is guaranteed.

    Parameters
    ----------
    domain_pad_x : float, default 0.02
        Left/right padding inside the trace domain (0..1).
    domain_pad_y_top : float, default 0.05
        Top padding inside the trace domain (ensures visible top gap).
    domain_pad_y_bottom : float, default 0.00
        Bottom padding inside the trace domain.
    shorten_labels : bool, default False
        If True, node labels are truncated to `max_label_chars` with an ellipsis.
    """
    # --- links
    src = links_df["source"].astype(int).to_numpy()
    tgt = links_df["target"].astype(int).to_numpy()
    val = links_df["value"].astype(float).to_numpy()

    # --- optional label shortening
    def _trunc(s: str) -> str:
        if not shorten_labels or max_label_chars <= 0 or not isinstance(s, str):
            return s
        return s if len(s) <= max_label_chars else s[: max_label_chars - 1] + "…"
    disp_labels = [_trunc(s) for s in labels]

    # --- node/colors
    node = {
        "label": disp_labels,
        "pad": 18,
        "thickness": 16,
        "line": {"color": "rgba(0,0,0,0.15)", "width": 0.5},
        "hovertemplate": "%{label}<extra></extra>",
    }
    vmin = vmax = None
    if color_values is not None and len(color_values) == len(labels):
        vals = np.asarray(color_values, float)
        if np.isfinite(vals).any():
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
            norm = np.where(np.isfinite(norm), norm, 0.5)
            node["color"] = sample_colorscale(colorscale, norm.tolist())

    # --- column locking (x/y)
    if group_ids is not None and field_names:
        k = len(field_names); denom = max(1, k - 1)
        node["x"] = [gid / denom for gid in group_ids]
        y = np.zeros(len(labels), float)
        y_lo, y_hi = 0.08, 0.92
        for gid in range(k):
            idxs = [i for i, g in enumerate(group_ids) if g == gid]
            if idxs:
                ys = np.linspace(y_lo, y_hi, len(idxs))
                for p, idx in enumerate(idxs):
                    y[idx] = ys[p]
        node["y"] = y.tolist()

    # --- build figure with a padded domain (guarantees visible top gap)
    sankey = go.Sankey(
        link={"source": src.tolist(), "target": tgt.tolist(), "value": val.tolist()},
        node=node,
        arrangement="fixed",
        domain={
            "x": [max(0.0, domain_pad_x), 1.0 - max(0.0, domain_pad_x)],
            "y": [max(0.0, domain_pad_y_bottom), 1.0 - max(0.0, domain_pad_y_top)],
        },
    )
    fig = go.Figure(sankey)

    # field labels under the plot
    if field_names:
        k = len(field_names); denom = max(1, k - 1)
        fig.update_layout(annotations=[
            dict(x=i/denom, y=-0.06, xref="paper", yref="paper",
                 text=name, showarrow=False, font=dict(size=14))
            for i, name in enumerate(field_names)
        ])

    # bottom colorbar
    if color_values is not None and np.isfinite(np.asarray(color_values, float)).any():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(
                    colorscale=colorscale, cmin=vmin, cmax=vmax, color=[vmin],
                    showscale=True,
                    colorbar=dict(
                        title=colorbar_title, orientation="h",
                        x=0.5, xanchor="center",
                        y=-0.14, yanchor="top", len=0.65, thickness=12,
                    ),
                ),
                showlegend=False,
            )
        )

    # layout & export
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(
        margin=dict(l=int(side_margin), r=int(side_margin), t=int(top_margin), b=120),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    if save_html:
        fig.write_html(save_html)
    if save_png:
        fig.write_image(save_png)
    return fig


def k_fields_plot(
    field_dfs: Dict[str, pd.DataFrame],
    df_main: pd.DataFrame,
    fields: List[str] = ("keywords", "sources"),
    customs: Optional[Dict[str, pd.DataFrame]] = None,
    top_n: Union[int, List[int]] = 10,
    color_option: str = "Average year",
    save_png: Optional[str] = "sankey.png",
    save_html: Optional[str] = None,
    label_maps: Optional[Dict[str, str]] = None
) -> Any:
    """
    Build and render a k-field Sankey diagram.

    Notes
    -----
    - By default saves a PNG to "sankey.png". Set `save_png=None` to disable saving.
    - HTML export is available via `save_html="file.html"` but is disabled by default (None).
    - Requires kaleido for PNG export.

    Parameters
    ----------
    field_dfs : dict[str, pd.DataFrame]
        {field_name -> binary indicator DataFrame} (docs × concepts).
    df_main : pd.DataFrame
        Document table; used for color metrics ("Year", "Cited by").
    fields : list[str], default ("keywords", "sources")
        Ordered fields to include.
    customs : dict[str, pd.DataFrame], optional
        Per-field overrides (take precedence over `field_dfs`).
    top_n : int or list[int], default 10
        Top columns (by column sum) kept per field.
    color_option : {"Average year", "Citations per document"} or str
        Node coloring rule. If unknown or column missing, no colors applied.
    save_png : str or None, default "sankey.png"
        Path to save PNG; use None to skip.
    save_html : str or None, default None
        Path to save HTML; None disables HTML export.
    label_maps : dict[str, str], optional
        Display label remapping.

    Returns
    -------
    plotly.graph_objects.Figure
        The Sankey figure.
    """
    customs = {} if customs is None else customs

    # collect and align
    dfs: List[pd.DataFrame] = []
    for fld in fields:
        if fld in customs:
            dfs.append(customs[fld])
        elif fld in field_dfs:
            dfs.append(field_dfs[fld])
        else:
            raise KeyError(f"No data for field \"{fld}\".")

    doc_index = df_main.index
    aligned = [(d.reindex(doc_index).fillna(0).astype(bool).astype("uint8")) for d in dfs]

    # normalize top_n
    top_n_list = [top_n] * len(aligned) if isinstance(top_n, int) else list(top_n)[:len(aligned)]
    if len(top_n_list) < len(aligned):
        top_n_list += [top_n_list[-1]] * (len(aligned) - len(top_n_list))

    # colors
    color_series: Optional[pd.Series]
    color_func: Callable[[pd.Series], float]
    cmap = {"Average year": ("Year", np.mean), "Citations per document": ("Cited by", np.mean)}
    if color_option in cmap and cmap[color_option][0] in df_main.columns:
        col, color_func = cmap[color_option]
        color_series = df_main[col]
    else:
        color_series, color_func = None, np.mean

    # prepare + plot
    links_df, labels, color_values, _, group_ids = prepare_for_sankey(
        dataframes=aligned,
        top_n=top_n_list,
        label_maps=label_maps,
        color_series=color_series,
        color_func=color_func,
        all_pairs=False,
    )
    fig = plot_sankey(
        links_df=links_df,
        labels=labels,
        color_values=(color_values if color_series is not None else None),
        group_ids=group_ids,
        field_names=list(fields),
        save_png=save_png,          # PNG enabled by default
        save_html=save_html,        # HTML available but off by default
        colorscale="Viridis",
        colorbar_title=(color_option if color_series is not None else ""),
    )
    return fig


"""
# Sample usage
df_authors = pd.DataFrame({"A": [1, 0, 1], "B": [0, 1, 1]})
df_keywords = pd.DataFrame({"X": [1, 1, 0], "Y": [0, 1, 1]})
df_countries = pd.DataFrame({"US": [1, 0, 1], "UK": [0, 1, 0]})
df_main = pd.DataFrame({"Year": [2019, 2020, 2021], "Cited by": [5, 10, 3]})
field_dfs = {"authors": df_authors, "keywords": df_keywords, "countries": df_countries}
fig = k_fields_plot(
    field_dfs=field_dfs,
    df_main=df_main,
    fields=["authors", "keywords", "countries"],
    top_n=2,
    color_option=None,
    save_png="sankey.png",
    save_html="sankey.html",
    label_maps={"A": "Alice", "X": "Xterm"}
)
print("Plots saved: sankey.png, sankey.html")
"""

# Plotting of networks

def plot_network(
    G,
    # --- coloring by partition / attribute ---
    partition_attr: "str | None" = None,
    color_attr: "str | None" = None,
    cluster_labels: "dict | None" = None,  # {cluster_id: "custom name"}; used in partition legend
    # --- sizing ---
    size_attr: "str | None" = None,
    size_scale: float = 350.0,
    fix_max_size: bool = True,
    log_scale: bool = True,  # kept for backward-compat if size_transform is None
    size_transform: "str | Callable[[float], float] | None" = None,  # "log" | "sqrt" | callable
    size_vmin: "float | None" = 0.0,
    size_vmax: "float | None" = None,
    default_node_size: float = 300.0,
    # --- layout / axes ---
    layout: str = "spring",
    pos: "dict | None" = None,
    layout_kwargs: "dict | None" = None,
    ax=None,
    figsize: "tuple[float, float]" = (8, 6),
    dpi: int = 300,
    background: str = "#fcfcfc",
    show_frame: bool = False,
    tight_layout: bool = True,
    largest_component: bool = True,
    random_state: int = 0,
    # --- node style ---
    node_shape: str = "o",
    node_alpha: float = 0.9,
    node_outline_color: str = "#ffffff",
    node_outline_width: float = 1.2,
    default_node_color: str = "tab:blue",
    missing_color: str = "#bdbdbd",
    # --- node colormaps ---
    cmap_name_continuous: str = "viridis",
    cmap_name_discrete: str = "tab10",
    vmin: "float | None" = None,
    vmax: "float | None" = None,
    max_cat_legend: int = 30,
    # --- edges ---
    edge_alpha: float = 0.6,
    edge_width: float = 1.0,
    min_edge_width: float = 0.4,
    max_edge_width: float = 4.0,
    edge_curve_rad: float = 0.15,
    curved_edges: bool = True,
    arrows: "bool | None" = None,  # None → G.is_directed()
    edge_min_weight: "float | None" = None,
    edge_keep_fraction: float = 1.0,
    edge_width_percentiles: "tuple[int, int]" = (5, 95),
    # --- edge colors (optional continuous mapping) ---
    edge_color_by: "str | None" = None,  # e.g., "weight"; None → blend node colors
    edge_cmap_name: str = "viridis",
    edge_vmin: "float | None" = None,
    edge_vmax: "float | None" = None,
    # --- labels ---
    label_attr: "str | None" = None,  # node attribute to label; default node id
    label_formatter: "Callable[[str, object, dict], str] | None" = None,
    label_fontsize: float = 12.0,
    label_fontsize_min: float = 7.0,
    label_fontsize_max: float = 18.0,
    label_scale_with_size: bool = True,
    label_min_fraction: float = 0.0,
    label_top_k: "int | None" = None,
    adjust_labels: bool = True,
    label_halo_color: str = "#ffffff",
    label_halo_width: float = 2.5,
    auto_label_contrast: bool = False,
    # --- legend & colorbar ---
    legend: bool = True,
    legend_loc: str = "upper right",
    show_colorbar: bool = True,
    colorbar_kwargs: "dict | None" = None,
    # --- export ---
    filename: "str | None" = None,
    export_format: str = "png",  # "png" | "pdf" | "svg"
    transparent: bool = False,
    **kwargs,
) -> "tuple":
    """
    "Plot a NetworkX graph with clean defaults, discrete/continuous coloring, robust size transforms,
    better legends, optional adjustText label nudging, and a partition legend titled
    'Clusters by <partition_attr>' (supports custom `cluster_labels`)."

    Key behaviors:
    - If `partition_attr` is given (and `color_attr` is None), nodes are colored by cluster and a legend is shown
      with title "Clusters by <partition_attr>". Labels in the legend come from `cluster_labels` when provided,
      otherwise raw ids 0, 1, 2, ...
    - If `color_attr` is given, auto-detects discrete vs continuous:
        * Discrete legend for strings or integer-like categories with count ≤ `max_cat_legend`.
        * Continuous colorbar otherwise.
    - Node sizes come from `size_attr`, with optional clamp and transform ("log"/"sqrt"/callable).
      Labels can be filtered via `label_min_fraction` and/or `label_top_k`.
    - Returns `(fig, ax, pos)`. Saves to `<filename>.<export_format>` if `filename` is provided.

    Notes:
    - Uses string type annotations to avoid import-time typing errors; no need to import `typing.Callable`.
    - Internally imports numpy/matplotlib/networkx; ensure those libraries are installed.
    """
    
    # Check for None or empty graph
    if G is None:
        raise ValueError(
            "Graph G is None. You need to build the co-occurrence network first. "
            "Try calling the appropriate method like `get_keyword_coocurrence()` or "
            "`build_keyword_network()` before plotting."
        )

    # --- local imports to keep this function paste-and-run ---
    import inspect

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe

    # optional label adjustment
    try:
        from adjustText import adjust_text
        _HAS_ADJUST_TEXT = True
    except Exception:
        _HAS_ADJUST_TEXT = False

    # -------------------- helpers --------------------
    def _safe_numeric(v, default=0.0) -> float:
        try:
            if v is None:
                return default
            x = float(v)
            if not math.isfinite(x):
                return default
            return x
        except Exception:
            return default

    def _normalize_edge_widths(w, lo=min_edge_width, hi=max_edge_width, pct=edge_width_percentiles):
        if len(w) == 0:
            return []
        w = np.asarray(w, dtype=float)
        if pct is not None and len(w) > 1:
            p_lo, p_hi = np.clip(np.asarray(pct, dtype=float), 0, 100)
            a, b = np.percentile(w, [p_lo, p_hi])
        else:
            a, b = float(np.min(w)), float(np.max(w))
        if not np.isfinite(a) or not np.isfinite(b) or a == b:
            return [0.5 * (lo + hi)] * len(w)
        ww = (w - a) / (b - a)
        return list(lo + ww * (hi - lo))

    def _blend_rgba(c1, c2):
        a, b = mcolors.to_rgba(c1), mcolors.to_rgba(c2)
        return tuple((x + y) * 0.5 for x, y in zip(a, b))

    def _call_safe(func, **k):
        sig = inspect.signature(func)
        return func(**{kk: vv for kk, vv in k.items() if kk in sig.parameters})

    def _is_integer_like(vals, tol=1e-9):
        for v in vals:
            try:
                x = float(v)
            except Exception:
                return False
            if not math.isfinite(x):
                return False
            if abs(x - round(x)) > tol:
                return False
        return True

    def _is_discrete(values, max_cats=max_cat_legend):
        vals = [v for v in values if v is not None]
        if not vals:
            return False, []
        # strings/categories → discrete
        if any(isinstance(v, str) for v in vals):
            cats = sorted({str(v) for v in vals})
            return (len(cats) <= max_cats), cats
        # numeric
        if _is_integer_like(vals):
            cats = sorted({int(round(float(v))) for v in vals})
            return (len(cats) <= max_cats), cats
        return False, []

    # -------------------- largest component --------------------
    if largest_component and G.number_of_nodes() > 0:
        comps = list(nx.weakly_connected_components(G) if G.is_directed() else nx.connected_components(G))
        if comps:
            G = G.subgraph(max(comps, key=len)).copy()

    # -------------------- layout --------------------
    if pos is None:
        layout_kwargs = dict(layout_kwargs or {})
        if "seed" not in layout_kwargs:
            layout_kwargs["seed"] = random_state
        layouts = {
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "shell": nx.shell_layout,
        }
        layout_fn = layouts.get(layout, nx.spring_layout)
        pos = layout_fn(G, **layout_kwargs)

    nodes = list(G.nodes())
    if not nodes:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure
        ax.set_facecolor(background)
        if not show_frame:
            ax.axis("off")
        if tight_layout:
            plt.tight_layout()
        if filename:
            fig.savefig(f"{filename}.{export_format}", dpi=dpi, bbox_inches="tight", transparent=transparent)
        return fig, ax, {}

    # -------------------- figure --------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    ax.set_facecolor(background)
    if not show_frame:
        ax.axis("off")

    # -------------------- node colors --------------------
    node_colors = {}
    colorbar_mappable = None
    legend_handles = None
    legend_title = None

    # (A) explicit partition (color by communities)
    if partition_attr and not color_attr:
        # Collect communities
        comms = {}
        for n, d in G.nodes(data=True):
            cid = d.get(partition_attr, d.get(f"partition_{partition_attr}", 0))
            comms.setdefault(cid, []).append(n)
        # palette
        base = "tab20" if (len(comms) > 10 and cmap_name_discrete == "tab10") else cmap_name_discrete
        cmap = cm.get_cmap(base, max(len(comms), 1))
        handles = []
        for i, (cid, members) in enumerate(sorted(comms.items(), key=lambda x: x[0])):
            col = cmap(i)
            for n in members:
                node_colors[n] = col
            # use custom labels if provided (manual); else raw id
            lab = (cluster_labels or {}).get(cid, str(cid))
            handles.append(
                Line2D([0], [0], marker="o", linestyle="", markersize=9,
                       markerfacecolor=col, markeredgecolor=node_outline_color,
                       markeredgewidth=node_outline_width, label=lab)
            )
        legend_handles = handles
        legend_title = f"Clusters by {partition_attr}"

    # (B) color_attr set → discrete vs continuous
    else:
        if color_attr:
            raw_vals = [G.nodes[n].get(color_attr, None) for n in nodes]
            present = [v for v in raw_vals if v is not None]
            is_disc, cats = _is_discrete(present, max_cat_legend)

            if is_disc:
                base = "tab20" if (len(cats) > 10 and cmap_name_discrete == "tab10") else cmap_name_discrete
                cmap = cm.get_cmap(base, max(len(cats), 1))
                cat2col = {c: cmap(i) for i, c in enumerate(cats)}
                for n, v in zip(nodes, raw_vals):
                    if v is None:
                        node_colors[n] = missing_color
                    else:
                        key = str(v) if isinstance(v, str) else int(round(float(v)))
                        node_colors[n] = cat2col.get(key, missing_color)
                legend_handles = [
                    Line2D([0], [0], marker="o", linestyle="", markersize=9,
                           markerfacecolor=cat2col[c], markeredgecolor=node_outline_color,
                           markeredgewidth=node_outline_width, label=str(c))
                    for c in cats
                ]
                legend_title = str(color_attr)
            else:
                # continuous mapping
                clean = []
                for v in raw_vals:
                    try:
                        clean.append(float(v) if v is not None else None)
                    except Exception:
                        clean.append(None)
                vals = np.array([x for x in clean if x is not None], dtype=float)
                _vmin = vmin if vmin is not None else (float(np.min(vals)) if len(vals) else 0.0)
                _vmax = vmax if vmax is not None else (float(np.max(vals)) if len(vals) else 1.0)
                if _vmin == _vmax:
                    _vmin, _vmax = _vmin - 0.5, _vmax + 0.5
                norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)
                cmap = cm.get_cmap(cmap_name_continuous)
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                colorbar_mappable = sm
                for n, v in zip(nodes, clean):
                    node_colors[n] = (cmap(norm(v)) if v is not None else missing_color)
        else:
            for n in nodes:
                node_colors[n] = default_node_color

    # -------------------- node sizes --------------------
    raw_sizes = np.array(
        [
            _safe_numeric(G.nodes[n].get(size_attr, default_node_size), default_node_size) if size_attr else default_node_size
            for n in nodes
        ],
        dtype=float,
    )
    # clamp
    if size_vmin is not None:
        raw_sizes = np.maximum(raw_sizes, float(size_vmin))
    if size_vmax is not None:
        raw_sizes = np.minimum(raw_sizes, float(size_vmax))

    # transform
    def _apply_size_transform(arr: "np.ndarray") -> "np.ndarray":
        tr = size_transform
        if tr is None and log_scale:
            tr = "log"
        if callable(tr):
            out = np.array([max(tr(float(x)), 0.0) for x in arr], dtype=float)
        elif tr == "log":
            out = np.log1p(np.maximum(arr, 0.0))
        elif tr == "sqrt":
            out = np.sqrt(np.maximum(arr, 0.0))
        else:
            out = np.maximum(arr, 0.0)
        return out

    svals = _apply_size_transform(raw_sizes)
    if fix_max_size:
        smax = float(np.max(svals)) if len(svals) else 1.0
        denom = max(smax, 1e-12)
        sizes = (svals / denom) * size_scale
    else:
        sizes = svals * size_scale
    sizes = sizes.tolist()

    # -------------------- edges --------------------
    def _edge_weight(u, v, d):
        return _safe_numeric(d.get("weight", edge_width), edge_width)

    edges_raw = [(u, v, _edge_weight(u, v, d)) for u, v, d in G.edges(data=True) if u != v]

    # min weight filter
    if edge_min_weight is not None:
        edges_raw = [(u, v, w) for (u, v, w) in edges_raw if w >= edge_min_weight]

    # keep fraction
    if not (0.0 < edge_keep_fraction <= 1.0):
        raise ValueError("edge_keep_fraction must be in (0, 1].")
    if edges_raw and edge_keep_fraction < 1.0:
        weights_sorted = sorted([w for _, _, w in edges_raw])
        k = max(1, int(round(len(weights_sorted) * edge_keep_fraction)))
        thresh = weights_sorted[-k]
        edges_raw = [(u, v, w) for (u, v, w) in edges_raw if w >= thresh]

    edgelist = [(u, v) for (u, v, _) in edges_raw]
    w_all = [w for _, _, w in edges_raw]
    edge_widths = _normalize_edge_widths(w_all)

    # edge colors
    if edge_color_by:
        # continuous mapping on edges
        vals = []
        for u, v in edgelist:
            d = G[u][v]
            val = d.get(edge_color_by, d.get("weight", None))
            vals.append(_safe_numeric(val, None))
        _vals = np.array([v for v in vals if v is not None], dtype=float)
        _emin = edge_vmin if edge_vmin is not None else (float(np.min(_vals)) if len(_vals) else 0.0)
        _emax = edge_vmax if edge_vmax is not None else (float(np.max(_vals)) if len(_vals) else 1.0)
        if _emin == _emax:
            _emin, _emax = _emin - 0.5, _emax + 0.5
        enorm = mcolors.Normalize(vmin=_emin, vmax=_emax)
        ecmap = cm.get_cmap(edge_cmap_name)
        edge_colors = [ecmap(enorm(v)) if v is not None else (0, 0, 0, edge_alpha) for v in vals]
    else:
        # blend incident node colors
        edge_colors = [_blend_rgba(node_colors.get(u, default_node_color), node_colors.get(v, default_node_color))
                       for (u, v) in edgelist]

    # -------------------- draw edges --------------------
    _arrows = G.is_directed() if arrows is None else bool(arrows)
    if curved_edges and not _arrows:
        _arrows = True
    ek = dict(
        G=G,
        pos=pos,
        edgelist=edgelist,
        edge_color=edge_colors or None,
        width=edge_widths or edge_width,
        alpha=edge_alpha,
        ax=ax,
        arrows=_arrows,
    )
    if curved_edges:
        ek.update({"connectionstyle": f"arc3,rad={edge_curve_rad}", "arrowstyle": "-", "arrowsize": 1, "node_size": 0})
    coll = _call_safe(nx.draw_networkx_edges, **ek)
    if coll is not None and hasattr(coll, "set_rasterized"):
        coll.set_rasterized(True)

    # -------------------- draw nodes --------------------
    nk = dict(
        G=G,
        pos=pos,
        nodelist=nodes,
        node_shape=node_shape,
        node_color=[node_colors[n] for n in nodes],
        node_size=sizes,
        alpha=node_alpha,
        linewidths=node_outline_width,
        edgecolors=node_outline_color,
        ax=ax,
    )
    _call_safe(nx.draw_networkx_nodes, **nk)

    # -------------------- labels --------------------
    # label text source
    if label_attr:
        raw_labels = [G.nodes[n].get(label_attr, str(n)) for n in nodes]
    else:
        raw_labels = [str(n) for n in nodes]
    # optional formatting
    if label_formatter:
        raw_labels = [label_formatter(lbl, n, G.nodes[n]) for lbl, n in zip(raw_labels, nodes)]

    # visibility mask
    import numpy as _np  # avoid shadowing earlier np in closures
    show_mask = _np.ones(len(nodes), dtype=bool)
    if label_min_fraction > 0.0 and len(sizes) > 0:
        smax = max(sizes)
        thr = label_min_fraction * smax
        show_mask &= (_np.asarray(sizes) >= thr)
    if label_top_k is not None and label_top_k >= 0 and len(sizes) > 0:
        order = _np.argsort(-_np.asarray(sizes))
        keep_idx = set(order[: int(label_top_k)])
        show_mask &= _np.array([i in keep_idx for i in range(len(nodes))], dtype=bool)

    texts = []
    if _np.any(show_mask):
        s_arr = _np.asarray(sizes)
        s_min, s_max = (float(_np.min(s_arr)), float(_np.max(s_arr))) if len(s_arr) else (1.0, 1.0)
        eps = 1e-9
        for i, n in enumerate(nodes):
            if not show_mask[i]:
                continue
            x, y = pos[n]
            if label_scale_with_size and s_max > s_min + eps:
                fs = label_fontsize_min + ((s_arr[i] - s_min) / (s_max - s_min)) * (label_fontsize_max - label_fontsize_min)
            else:
                fs = label_fontsize
            # pick text color
            tcolor = "black"
            if auto_label_contrast:
                rgba = mcolors.to_rgba(node_colors.get(n, default_node_color))
                r, g, b = rgba[:3]
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                tcolor = "black" if luminance > 0.6 else "white"
            t = ax.text(x, y, raw_labels[i], fontsize=fs, ha="center", va="center", color=tcolor)
            t.set_path_effects([pe.withStroke(linewidth=label_halo_width, foreground=label_halo_color)])
            texts.append(t)

    if adjust_labels and texts and _HAS_ADJUST_TEXT:
        adjust_text(texts, only_move={"points": "y", "text": "xy"}, autoalign="xy", ax=ax)

    # -------------------- legend / colorbar --------------------
    # Discrete legend when `legend_handles` is built (partition or discrete color_attr)
    if legend and legend_handles:
        leg = ax.legend(
            handles=legend_handles,
            loc=legend_loc,
            frameon=False,
            title=legend_title,
            handlelength=1.2,
            handletextpad=0.5,
            borderaxespad=0.4,
        )
        ax.add_artist(leg)

    # Colorbar for continuous color_attr
    if color_attr and show_colorbar and (legend_handles is None) and (colorbar_mappable is not None):
        cb_kwargs = dict()
        if colorbar_kwargs:
            cb_kwargs.update(colorbar_kwargs)
        cbar = plt.colorbar(colorbar_mappable, ax=ax, **cb_kwargs)
        cbar.set_label(str(color_attr))

    # -------------------- finalize --------------------
    ax.set_aspect("equal")
    if tight_layout:
        plt.tight_layout()
    if filename:
        fig.savefig(f"{filename}.{export_format}", dpi=dpi, bbox_inches="tight", transparent=transparent)

    return fig, ax, pos


def plot_degree_distribution(G, log_log=False, ax=None, **kwargs):
    """
    Plot the degree distribution histogram.

    Parameters
    ----------
    G : networkx.Graph
    log_log : bool, optional
        If True, use log-log scale.
    ax : matplotlib.axes.Axes, optional
    kwargs : passed to plt.hist

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()
    degrees = [d for _, d in G.degree()]
    ax.hist(degrees, **kwargs)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    if log_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    return ax

# Plotting of citation network and main path


def plot_citation_network(
    G: nx.DiGraph,
    size_dict: Optional[Dict[str, float]] = None,
    color_dict: Optional[Dict[str, float]] = None,
    label_dict: Optional[Dict[str, str]] = None,
    cmap: str = 'viridis',
    arrow_size: int = 10,
    font_size: int = 8,
    node_size_factor: float = 100,
    sqrt_sizes: bool = False,
    edge_width: float = 0.5,
    layout: str = 'kamada_kawai',
    highlight_main_path: bool = False,
    main_path: Optional[List[str]] = None,
    main_path_color: str = 'crimson',
    main_path_width: float = 2.5,
    main_path_style: str = 'solid',
    filename: Optional[str] = None
) -> None:
    """
    Visualize a directed citation network with optional node sizing, coloring,
    labeling, layout choices, and main path highlighting.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph representing the citation network.
    size_dict : dict[str, float], optional
        Dictionary mapping node IDs to size values. If None, in-degree is used.
    color_dict : dict[str, float], optional
        Dictionary mapping node IDs to scalar values for coloring.
    label_dict : dict[str, str], optional
        Dictionary mapping node IDs to label strings.
    cmap : str, default='viridis'
        Matplotlib colormap name used for node colors.
    arrow_size : int, default=10
        Size of the arrowheads on directed edges.
    font_size : int, default=8
        Font size for node labels.
    node_size_factor : float, default=100
        Scaling factor applied to node sizes.
    sqrt_sizes : bool, default=False
        Whether to apply square root scaling to node sizes.
    edge_width : float, default=0.5
        Width of network edges.
    layout : str, default='kamada_kawai'
        Layout algorithm for node positions. Options: 'kamada_kawai', 'spring', 'circular'.
    highlight_main_path : bool, default=False
        If True, highlights the main citation path in the network.
    main_path : list[str], optional
        Optional list of node IDs representing the main citation path. If None and
        `highlight_main_path` is True, the path is computed via `utilsbib.compute_main_path`.
    main_path_color : str, default='crimson'
        Color used to highlight edges along the main path.
    main_path_width : float, default=2.5
        Width of the highlighted main path edges.
    main_path_style : str, default='solid'
        Line style for the main path edges (e.g., 'solid', 'dashed').
    filename : str, optional
        Base filename (without extension) to save the plot. If None, the plot is shown interactively.

    Returns
    -------
    None
    """
    if layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        raise ValueError(f'Unknown layout: {layout}')

    if not isinstance(size_dict, dict):
        size_dict = None
    if size_dict is None:
        size_dict = dict(G.in_degree())

    node_sizes = [size_dict.get(node, 1) * node_size_factor for node in G.nodes()]
    if sqrt_sizes:
        node_sizes = [np.sqrt(s) for s in node_sizes]

    if color_dict is not None:
        node_colors = [color_dict.get(node, 0.5) for node in G.nodes()]
    else:
        node_colors = 'lightblue'

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.get_cmap(cmap))
    nx.draw_networkx_edges(G, pos, width=edge_width, arrows=True, arrowstyle='-|>', arrowsize=arrow_size)

    if label_dict is None:
        label_dict = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=font_size)

    if highlight_main_path:
        if main_path is None:

            main_path = utilsbib.compute_main_path(G)
        main_path_edges = list(zip(main_path[:-1], main_path[1:]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=main_path_edges,
            width=main_path_width,
            edge_color=main_path_color,
            style=main_path_style,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=arrow_size
        )

    if color_dict is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(color_dict.values()), vmax=max(color_dict.values())))
        sm.set_array([])
        plt.colorbar(sm)

    plt.axis('off')
    plt.tight_layout()
    if filename:
        save_plot(filename)
    else:
        plt.show()


def plot_main_path(
    G: nx.DiGraph,
    path: list[str] | None = None,
    size_dict: dict[str, float] | None = None,
    color_dict: dict[str, float] | None = None,
    label_map: dict[str, str] | None = None,
    cmap = plt.cm.viridis,
    edge_color: str = "red",
    edge_width: float = 2.0,
    arrow_size: int = 10,
    font_size: int = 10,
    layout: str = "kamada_kawai",
    filename: Optional[str] = None
) -> None:
    """
    Plot only the main citation path from a directed citation graph,
    with adjustable visuals for nodes, edges, and labels.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph representing the full citation network.
    path : list[str], optional
        List of node IDs forming the main citation path. If None, computed using
        `utilsbib.compute_main_path`.
    size_dict : dict[str, float], optional
        Dictionary mapping node IDs to node sizes.
    color_dict : dict[str, float], optional
        Dictionary mapping node IDs to scalar values for coloring.
    label_map : dict[str, str], optional
        Dictionary mapping node IDs to label strings.
    cmap : matplotlib colormap, default=plt.cm.viridis
        Colormap to use when `color_dict` is provided.
    edge_color : str, default="red"
        Color of the edges along the main path.
    edge_width : float, default=2.0
        Width of the edges along the main path.
    arrow_size : int, default=10
        Size of the arrowheads on the main path.
    font_size : int, default=10
        Font size for node labels.
    layout : str, default="kamada_kawai"
        Layout algorithm to position nodes. Options include: "kamada_kawai", "spring", "circular", "shell".
    filename : str, optional
        Base filename (without extension) to save the plot. If None, the plot is shown interactively.

    Returns
    -------
    None
    """
    if path is None:
        path = utilsbib.compute_main_path(G)
    subG = nx.DiGraph()
    subG.add_nodes_from(path)
    subG.add_edges_from(zip(path, path[1:]))

    layout_funcs = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
        "shell": nx.shell_layout
    }
    pos = layout_funcs.get(layout, nx.kamada_kawai_layout)(subG)

    sizes = ([size_dict.get(n, 300) for n in subG.nodes()] if size_dict else
             [300 + 200 * G.in_degree(n) for n in subG.nodes()])

    if color_dict:
        vals = [color_dict.get(n, 0) for n in subG.nodes()]
        norm = plt.Normalize(vmin=min(vals), vmax=max(vals))
        colors = [cmap(norm(v)) for v in vals]
        numeric = True
    else:
        colors = "orange"
        numeric = False

    labels = ({n: label_map.get(n, n) for n in subG.nodes()} if label_map else
              {n: n for n in subG.nodes()})

    plt.figure(figsize=(8, 6))
    nx.draw(
        subG, pos,
        labels=labels,
        node_size=sizes,
        node_color=colors,
        font_size=font_size,
        arrowsize=arrow_size,
        edge_color=edge_color,
        width=edge_width
    )
    if numeric:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label="Color Value")

    plt.title("Main Citation Path")
    plt.tight_layout()
    if filename:
        save_plot(filename)
    else:
        plt.show()


# historiograph

def layout_historiograph(G):
    """Compute a chronological layout: x = year, y = randomly jittered within each year."""
    

    year_nodes = defaultdict(list)
    for node, attrs in G.nodes(data=True):
        year = attrs.get("year")
        if year is not None:
            year_nodes[year].append(node)

    pos = {}
    for year, nodes in year_nodes.items():
        for i, node in enumerate(sorted(nodes)):
            jitter = np.random.uniform(-0.5, 0.5)
            pos[node] = (year, jitter)

    return pos


def plot_historiograph(
    G,
    pos,
    figsize=(12, 8),
    size_attr=None,
    min_indegree=None,
    min_citations=100,
    min_year=None,
    max_year=None,
    save_as=None,
    dpi=600,
):
    """Draw the historiograph using matplotlib, excluding isolated nodes, loops, and applying filters."""
    plt.figure(figsize=figsize)

    def node_passes_filters(n, d):
        if min_year and d.get("year") < min_year:
            return False
        if max_year and d.get("year") > max_year:
            return False
        if min_citations and d.get("Cited by", 0) < min_citations:
            return False
        if min_indegree and G.in_degree(n) < min_indegree:
            return False
        return True

    filtered_nodes = [n for n, d in G.nodes(data=True) if node_passes_filters(n, d)]
    filtered_graph = G.subgraph(filtered_nodes).copy()
    filtered_graph.remove_edges_from(nx.selfloop_edges(filtered_graph))
    connected_nodes = [n for n in filtered_graph.nodes if filtered_graph.degree(n) > 0]
    subgraph = filtered_graph.subgraph(connected_nodes).copy()

    if size_attr:
        sizes = [subgraph.nodes[n].get(size_attr, 3) * 10 for n in subgraph.nodes]
    else:
        sizes = [300 for _ in subgraph.nodes]

    sub_pos = {k: v for k, v in pos.items() if k in subgraph.nodes}

    nx.draw(subgraph, sub_pos, with_labels=False, arrows=True, node_size=sizes, node_color="lightblue")

    labels = {k: v for k, v in nx.get_node_attributes(subgraph, "title").items()}
    texts = [plt.text(sub_pos[n][0], sub_pos[n][1], labels[n], fontsize=8, ha="center", va="center") for n in labels]
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

    plt.title("Historiograph")
    plt.xlabel("Publication Year")
    plt.axis("off")
    plt.tight_layout()

    if save_as:
        save_plot(save_as, dpi=dpi)

    plt.show()

# specific networks

def plot_country_collab_network(matrix_df, threshold=1, figsize=(12, 12), layout_func="spring", filename_base=None):
    """
    Plots a network graph of country collaborations above a threshold.

    Parameters:
    matrix_df (pd.DataFrame): Symmetric collaboration matrix.
    threshold (int): Minimum collaboration count to include an edge.
    figsize (tuple): Size of the figure in inches.
    layout_func (callable): NetworkX layout function (e.g., nx.spring_layout).
    filename_base (str or None): If provided, saves the plot as PNG, SVG, and PDF using this base name.
    """
    if matrix_df.empty:
        print("Empty matrix: network not generated.")
        return

    G = nx.Graph()
    for i in matrix_df.index:
        for j in matrix_df.columns:
            weight = matrix_df.loc[i, j]
            if i != j and weight >= threshold:
                G.add_edge(i, j, weight=weight)

    layout = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
        "shell": nx.shell_layout
    }[layout_func]

    if len(G.nodes) == 0:
        print("No edges above threshold: network not generated.")
        return

    pos = layout(G)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, width=[w * 0.1 for w in edge_weights])
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Country Collaboration Network")
    plt.axis("off")
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base)

    plt.show()

# specific heatmap

def plot_country_collab_heatmap(matrix_df, top_n=50, figsize=(12, 10), cmap="Blues", annotate=False, filename_base=None):
    """
    Plots a heatmap of the country collaboration matrix, optionally limited to the top N countries by total collaboration.

    Parameters:
    matrix_df (pd.DataFrame): Symmetric collaboration matrix.
    top_n (int): Number of top countries (by total collaborations) to include.
    figsize (tuple): Size of the figure in inches.
    cmap (str): Colormap for heatmap shading.
    annotate (bool): Whether to show collaboration counts in each cell.
    filename_base (str or None): If provided, saves the plot as PNG, SVG, and PDF using this base name.
    """
    if matrix_df.empty:
        print("Empty matrix: heatmap not generated.")
        return

    # Compute total collaborations and select top N countries
    totals = matrix_df.sum(axis=1) + matrix_df.sum(axis=0)
    top_countries = totals.sort_values(ascending=False).head(top_n).index
    matrix_top = matrix_df.loc[top_countries, top_countries]

    plt.figure(figsize=figsize)

    is_integer = np.allclose(matrix_top, matrix_top.astype(int))
    fmt = "d" if is_integer else ".2f"
    sns.heatmap(matrix_top, cmap=cmap, square=True, annot=annotate, fmt=fmt,
                cbar_kws={"label": "Collaboration Count"})    

    plt.title("Country Collaboration Matrix (Top {} Countries)".format(top_n))
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base)

    plt.show()


# Thematic map and evolution

def save_sankey(diagram, filename_base, formats=("png", "svg", "pdf", "html")):
    """
    Save a Plotly Sankey diagram to multiple formats.

    Parameters
    ----------
    diagram : plotly.graph_objects.Figure
        Sankey diagram figure.
    filename_base : str
        Base filename without extension.
    formats : tuple of str, optional
        File formats to save (png, svg, pdf, html).
    """
    for ext in formats:
        path = f"{filename_base}.{ext}"
        if ext == "html":
            diagram.write_html(path)
        else:
            diagram.write_image(path)


def plot_bipartite_sankey(
    contingency_matrix: pd.DataFrame,
    left_label: str = "Source",
    right_label: str = "Target",
    title: str = None,
    max_links: int = 50,
    min_value: float = 0,
    colorscale: str = "Blues",
    node_color_left: str = "#1f77b4",
    node_color_right: str = "#ff7f0e",
    save_png: str = None,
    save_html: str = None,
    width: int = 1000,
    height: int = 700,
    font_size: int = 11,
):
    """
    Create a bipartite Sankey diagram from a contingency matrix.
    
    This is useful for visualizing relationships between two categorical variables,
    such as Authors-Keywords, Countries-Sources, etc.
    
    Parameters
    ----------
    contingency_matrix : pd.DataFrame
        A contingency/co-occurrence matrix where rows represent one entity type
        and columns represent another. Values represent the strength of connection.
    left_label : str, default "Source"
        Label for the left (row) entities.
    right_label : str, default "Target"
        Label for the right (column) entities.
    title : str, optional
        Title for the diagram. If None, uses "{left_label} → {right_label}".
    max_links : int, default 50
        Maximum number of links to show (top by value).
    min_value : float, default 0
        Minimum value threshold for including a link.
    colorscale : str, default "Blues"
        Colorscale for link colors (based on value).
    node_color_left : str, default "#1f77b4"
        Color for left-side nodes.
    node_color_right : str, default "#ff7f0e"
        Color for right-side nodes.
    save_png : str, optional
        Path to save PNG file.
    save_html : str, optional
        Path to save HTML file.
    width : int, default 1000
        Figure width in pixels.
    height : int, default 700
        Figure height in pixels.
    font_size : int, default 11
        Font size for node labels.
        
    Returns
    -------
    plotly.graph_objects.Figure
        The Sankey diagram figure.
    """
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    
    # Extract links from contingency matrix
    links = []
    for row_idx, row_name in enumerate(contingency_matrix.index):
        for col_idx, col_name in enumerate(contingency_matrix.columns):
            value = contingency_matrix.iloc[row_idx, col_idx]
            if pd.notna(value) and value > min_value:
                links.append({
                    'source': str(row_name),
                    'target': str(col_name),
                    'value': float(value)
                })
    
    if not links:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for Sankey diagram",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Sort by value and take top links
    links.sort(key=lambda x: x['value'], reverse=True)
    links = links[:max_links]
    
    # Get unique sources and targets (in order of appearance)
    sources = list(dict.fromkeys([l['source'] for l in links]))
    targets = list(dict.fromkeys([l['target'] for l in links]))
    
    # Create node list: sources first, then targets
    all_nodes = sources + targets
    node_map = {name: idx for idx, name in enumerate(all_nodes)}
    
    # Prepare link data
    source_indices = [node_map[l['source']] for l in links]
    target_indices = [node_map[l['target']] for l in links]
    values = [l['value'] for l in links]
    
    # Normalize values for color
    max_val = max(values) if values else 1
    min_val = min(values) if values else 0
    if max_val > min_val:
        normalized = [(v - min_val) / (max_val - min_val) for v in values]
    else:
        normalized = [0.5] * len(values)
    
    # Sample colors from colorscale
    link_colors = sample_colorscale(colorscale, normalized)
    # Add transparency
    link_colors = [c.replace('rgb', 'rgba').replace(')', ', 0.6)') if 'rgb' in c else c 
                   for c in link_colors]
    
    # Node colors
    node_colors = [node_color_left] * len(sources) + [node_color_right] * len(targets)
    
    # Position nodes: sources on left (x=0), targets on right (x=1)
    n_sources = len(sources)
    n_targets = len(targets)
    
    node_x = [0.001] * n_sources + [0.999] * n_targets
    
    # Distribute y positions evenly
    if n_sources > 1:
        source_y = [0.1 + 0.8 * i / (n_sources - 1) for i in range(n_sources)]
    else:
        source_y = [0.5]
    
    if n_targets > 1:
        target_y = [0.1 + 0.8 * i / (n_targets - 1) for i in range(n_targets)]
    else:
        target_y = [0.5]
    
    node_y = source_y + target_y
    
    # Create Sankey diagram
    fig = go.Figure(go.Sankey(
        arrangement='fixed',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors,
            x=node_x,
            y=node_y,
            hovertemplate='%{label}<br>Total: %{value}<extra></extra>',
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}<br>Value: %{value}<extra></extra>',
        ),
        domain=dict(x=[0.0, 1.0], y=[0.0, 0.9])
    ))
    
    # Add title
    if title is None:
        title = f"{left_label} → {right_label}"
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        font=dict(size=font_size),
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=50, b=50),
    )
    
    # Add column labels
    fig.add_annotation(
        x=0.0, y=-0.05,
        xref="paper", yref="paper",
        text=f"<b>{left_label}</b>",
        showarrow=False,
        font=dict(size=13),
    )
    fig.add_annotation(
        x=1.0, y=-0.05,
        xref="paper", yref="paper",
        text=f"<b>{right_label}</b>",
        showarrow=False,
        font=dict(size=13),
    )
    
    # Save if requested
    if save_png:
        try:
            fig.write_image(save_png, scale=2)
        except Exception as e:
            print(f"Warning: Could not save PNG: {e}")
    
    if save_html:
        try:
            fig.write_html(save_html)
        except Exception as e:
            print(f"Warning: Could not save HTML: {e}")
    
    return fig


def plot_thematic_map(
    G,
    partition_attr,
    max_dot_size=200,
    quadrant_labels=False,
    items_per_cluster=3,
    cmap_name="viridis",
    figsize=(8, 6),
    max_clusters=None,
    min_cluster_size=5,
    include_cluster_label=False,
    color_df=None,
    color_col=None,
    save_plot_base=None,
    dpi=600,
    ax=None,
    item_sep="\n"
):
    """
    Plot thematic map of clusters with axis labels, no tick values,
    optional spaced quadrant labels, and non-overlapping cluster annotations.
    Optionally save figure to files using save_plot.

    Parameters
    ----------
    G : networkx.Graph
    partition_attr : str
        Node attribute name (without or with "partition_" prefix).
    max_dot_size : float, optional
        Size of the largest cluster marker.
    quadrant_labels : dict or None, optional
        Mapping quadrant keys ("NE","NW","SW","SE") to labels;
        if None, quadrant labels are not shown.
    items_per_cluster : int, optional
        Number of top nodes (by degree centrality) to list per cluster in plot.
    cmap_name : str, optional
        Colormap for clusters.
    figsize : tuple, optional
        Figure size when creating new figure.
    max_clusters : int, optional
        Limit to this many largest clusters.
    min_cluster_size : int, optional
        Exclude clusters smaller than this size.
    include_cluster_label : bool, optional
        Whether to prefix each label with the cluster ID.
    color_df : pandas.DataFrame, optional
        DataFrame with cluster IDs and a color column.
    color_col : str, optional
        Column name in color_df for coloring clusters.
    save_plot_base : str, optional
        Base filename (without extension) for saving the plot; if None, plot is not auto-saved.
    dpi : int, optional
        Resolution in dots per inch for saving.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot onto. If None, a new figure and axis are created.
    item_sep : str, optional
        Separator string between item labels.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    key = partition_attr if partition_attr.startswith("partition_") else f"partition_{partition_attr}"
    metrics = utilsbib.compute_cluster_metrics(G, key)
    cids = [c for c in metrics if metrics[c]["size"] >= min_cluster_size]
    if max_clusters and len(cids) > max_clusters:
        cids = sorted(cids, key=lambda c: metrics[c]["size"], reverse=True)[:max_clusters]

    densities = [metrics[c]["density"] for c in cids]
    centrals = [metrics[c]["avg_degree_centrality"] for c in cids]
    sizes_raw = [metrics[c]["size"] for c in cids]
    max_raw = max(sizes_raw) if sizes_raw else 1
    sizes = [(s / max_raw) * max_dot_size for s in sizes_raw]

    if color_df is not None and color_col is not None:
        df_col = color_df.set_index(key)
        vals = [df_col.loc[c, color_col] if c in df_col.index else 0 for c in cids]
        norm = colors.Normalize(vmin=min(vals), vmax=max(vals))
        cmap = cm.get_cmap(cmap_name)
        colors_list = [cmap(norm(v)) for v in vals]
    else:
        colors_list = ["lightgrey"] * len(cids)

    deg_c = nx.degree_centrality(G)
    text_objs = []
    for i, cid in enumerate(cids):
        ax.scatter(densities[i], centrals[i], s=sizes[i], color=colors_list[i], alpha=0.7)
        nodes = [n for n, d in G.nodes(data=True) if d.get(key) == cid]
        top_nodes = sorted(nodes, key=lambda n: deg_c.get(n, 0), reverse=True)[:items_per_cluster]
        labels = [str(n) for n in top_nodes]
        if include_cluster_label:
            labels.insert(0, str(cid))
        txt = ax.text(densities[i], centrals[i], item_sep.join(labels), fontsize=8)
        text_objs.append(txt)

    x_mid = sum(densities) / len(densities) if densities else 0
    y_mid = sum(centrals) / len(centrals) if centrals else 0
    ax.axvline(x_mid, color="grey", lw=0.8, linestyle="--")
    ax.axhline(y_mid, color="grey", lw=0.8, linestyle="--")
    adjust_text(text_objs, ax=ax, only_move={"points": "y", "texts": "y"})

    if quadrant_labels:
        ql = {"NE": "Motor Themes", "NW": "Niche Themes",
                           "SW": "Emerging or Declining Themes",
                           "SE": "Basic Themes"}
    else:
        ql = {}

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    dx = (x_max - x_min) * 0.1
    dy = (y_max - y_min) * 0.1
    ax.text(x_max - dx, y_max - dy, ql.get("NE", ""), ha="right", va="top", fontsize=9, color="grey")
    ax.text(x_min + dx, y_max - dy, ql.get("NW", ""), ha="left", va="top", fontsize=9, color="grey")
    ax.text(x_min + dx, y_min + dy, ql.get("SW", ""), ha="left", va="bottom", fontsize=9, color="grey")
    ax.text(x_max - dx, y_min + dy, ql.get("SE", ""), ha="right", va="bottom", fontsize=9, color="grey")

    ax.set_xlabel("Density")
    ax.set_ylabel("Centrality")
    ax.set_xticks([])
    ax.set_yticks([])

    if save_plot_base:
        save_plot(save_plot_base, dpi=dpi)

    return ax


def plot_thematic_evolution(
    graphs,
    partition_attr,
    time_labels=None,
    map_kwargs=None,
    sankey_kwargs=None,
    save_maps_base=None,
    map_formats=("png", "svg", "pdf"),
    save_sankey_base=None,
    sankey_formats=("png", "svg", "pdf", "html"),
    top_k=2,
    item_sep="\n"
):
    """
    Plot a sequence of thematic maps and a Sankey diagram showing cluster evolution across time.

    Instead of showing cluster IDs and timepoint labels, display top_k nodes per cluster.
    Optionally save maps and Sankey diagram to files.

    Parameters
    ----------
    graphs : list of networkx.Graph
        Sequence of graphs representing different time points.
    partition_attr : str
        Node attribute name for cluster assignment.
    time_labels : list of str, optional
        Labels for each time point; not displayed when top_k is used.
    map_kwargs : dict, optional
        Keyword args passed to plot_thematic_map.
    sankey_kwargs : dict, optional
        Keyword args for plotly Sankey layout.
    save_maps_base : str, optional
        Base filename (without extension) for saving thematic maps; if None, maps are not auto-saved.
    map_formats : tuple of str, optional
        File formats to save thematic maps (png, svg, pdf).
    save_sankey_base : str, optional
        Base filename (without extension) to save Sankey diagram; if None, Sankey is not auto-saved.
    sankey_formats : tuple of str, optional
        Formats to save Sankey (png, svg, pdf, html).
    top_k : int, optional
        Number of top nodes (by degree centrality) per cluster to display.
    item_sep : str, optional
        Separator string between items in Sankey node labels.

    Returns
    -------
    tuple
        (matplotlib.figure.Figure, plotly.graph_objects.Figure)
    """
    if map_kwargs is None:
        map_kwargs = {}
    if sankey_kwargs is None:
        sankey_kwargs = {}
    n = len(graphs)
    fig_maps, axes = plt.subplots(
        1, n, figsize=(n * map_kwargs.get("figsize", (8, 6))[0], map_kwargs.get("figsize", (8, 6))[1])
    )
    if n == 1:
        axes = [axes]

    # plot thematic maps without titles
    for ax, G in zip(axes, graphs):
        plot_thematic_map(
            G,
            partition_attr,
            quadrant_labels=None,
            ax=ax,
            item_sep=item_sep,
            **map_kwargs
        )

    plt.tight_layout()

    # auto-save thematic maps
    if save_maps_base:
        for ext in map_formats:
            path = f"{save_maps_base}.{ext}"
            fig_maps.savefig(path, bbox_inches="tight")

    # prepare Sankey data with top_k labels
    clusters_list = []
    degc_list = []
    for G in graphs:
        key = partition_attr if partition_attr.startswith("partition_") else f"partition_{partition_attr}"
        clust = {}
        deg_c = nx.degree_centrality(G)
        for node, d in G.nodes(data=True):
            cid = d.get(key)
            clust.setdefault(cid, set()).add(node)
        clusters_list.append(clust)
        degc_list.append(deg_c)

    node_labels = []
    offsets = []
    cum = 0
    for clust, deg_c in zip(clusters_list, degc_list):
        offsets.append(cum)
        for cid, nodes in clust.items():
            sorted_nodes = sorted(nodes, key=lambda n: deg_c.get(n, 0), reverse=True)[:top_k]
            label = item_sep.join(str(n) for n in sorted_nodes)
            node_labels.append(label)
        cum += len(clust)

    sources, targets, values = [], [], []
    for t in range(n - 1):
        src_cids = list(clusters_list[t].keys())
        tgt_cids = list(clusters_list[t + 1].keys())
        for i, src in enumerate(src_cids):
            for j, tgt in enumerate(tgt_cids):
                val = len(clusters_list[t][src] & clusters_list[t + 1][tgt])
                if val > 0:
                    sources.append(offsets[t] + i)
                    targets.append(offsets[t + 1] + j)
                    values.append(val)

    sankey_node = dict(label=node_labels)
    sankey_link = dict(source=sources, target=targets, value=values)
    fig_sankey = go.Figure(data=[go.Sankey(node=sankey_node, link=sankey_link)], **sankey_kwargs)
    fig_sankey.update_layout(title_text="Thematic Evolution Sankey", font_size=10)

    # auto-save Sankey diagram
    if save_sankey_base:
        save_sankey(fig_sankey, save_sankey_base, formats=sankey_formats)

    return fig_maps, fig_sankey


# plotting of relationships

def plot_correspondence_analysis(
    row_coords: pd.DataFrame,
    col_coords: pd.DataFrame,
    explained_inertia: list,
    df_relation: pd.DataFrame,
    figsize=(8, 6),
    annotate=True,
    alpha=0.8,
    size_scale=300,
    use_size: bool = True,
    filename_base: str = None,
    dpi: int = 600,
    row_label_name: str = "Rows",
    col_label_name: str = "Columns",
    title: str = "Correspondence Analysis with Frequencies",
    abbreviate_labels: bool = False,
    abbreviate_kwargs: dict | None = None,
):
    """
    Plot 2D correspondence analysis with optional scaling by frequency, 
    label customization, and label abbreviation.

    Parameters
    ----------
    row_coords : pd.DataFrame
        Row coordinates in CA space.
    col_coords : pd.DataFrame
        Column coordinates in CA space.
    explained_inertia : list
        Variance explained by each axis.
    df_relation : pd.DataFrame
        Original contingency table (used for frequencies).
    figsize : tuple
        Size of the figure.
    annotate : bool
        Whether to annotate points with labels.
    alpha : float
        Point transparency.
    size_scale : float
        Scaling factor for point sizes (if used).
    use_size : bool
        Whether to scale point size by marginal frequency.
    filename_base : str
        If provided, saves plot to PNG, SVG, and PDF.
    dpi : int
        Resolution for saved figures.
    row_label_name : str
        Legend name for row group.
    col_label_name : str
        Legend name for column group.
    title : str
        Plot title. If None, no title is shown.
    abbreviate_labels : bool, default=False
        If True, applies `abbreviate_words` to row and column labels.
    abbreviate_kwargs : dict, optional
        Extra keyword arguments passed to `abbreviate_words`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.set_facecolor("white")

    row_freq = df_relation.sum(axis=1)
    col_freq = df_relation.sum(axis=0)

    row_sizes = (row_freq.loc[row_coords.index] / row_freq.max() * size_scale) if use_size else size_scale
    col_sizes = (col_freq.loc[col_coords.index] / col_freq.max() * size_scale) if use_size else size_scale

    # Plot rows
    ax.scatter(
        row_coords.iloc[:, 0], row_coords.iloc[:, 1],
        c="tab:blue", s=row_sizes, alpha=alpha, label=row_label_name
    )

    # Plot columns
    ax.scatter(
        col_coords.iloc[:, 0], col_coords.iloc[:, 1],
        c="tab:red", s=col_sizes, alpha=alpha, marker="^", label=col_label_name
    )

    # Annotate
    if annotate:
        texts = []
        for label, (x, y) in row_coords.iloc[:, :2].iterrows():
            lab = utilsbib.abbreviate_words(label, **(abbreviate_kwargs or {})) if abbreviate_labels else label
            texts.append(ax.text(x, y, lab, fontsize=8, color="tab:blue"))
        for label, (x, y) in col_coords.iloc[:, :2].iterrows():
            lab = utilsbib.abbreviate_words(label, **(abbreviate_kwargs or {})) if abbreviate_labels else label
            texts.append(ax.text(x, y, lab, fontsize=8, color="tab:red"))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Axes and layout
    ax.set_xlabel(f"Dimension 1 ({explained_inertia[0]*100:.1f}%)")
    ax.set_ylabel(f"Dimension 2 ({explained_inertia[1]*100:.1f}%)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(False)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi=dpi)

    plt.show()
    

def plot_residual_heatmap(
    residuals_df: pd.DataFrame,
    center: float = 0.0,
    cmap: str = "coolwarm",
    figsize=(10, 8),
    annotate: bool = False,
    square: bool = True,
    filename_base: str = None,
    dpi: int = 600,
    title: str = "Standardized Pearson Residuals",
    row_label: str = None,
    col_label: str = None, **kwargs
):
    """
    Plot a heatmap of Pearson residuals with optional customization.

    Parameters:
        residuals_df (pd.DataFrame): DataFrame of standardized residuals.
        center (float): Value at center of colormap. Typically 0.
        cmap (str): Seaborn/matplotlib colormap.
        figsize (tuple): Size of figure.
        annotate (bool): Whether to annotate heatmap cells with values.
        square (bool): Whether to enforce square aspect ratio for cells.
        filename_base (str): If provided, saves plot to PNG, SVG, PDF.
        dpi (int): Resolution for saved images.
        title (str): Title of the plot. Use None to omit.
        row_label (str): Label for y-axis. If None, uses DataFrame index name.
        col_label (str): Label for x-axis. If None, uses DataFrame column name.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        residuals_df,
        center=center,
        cmap=cmap,
        annot=annotate,
        fmt=".2f" if annotate else "",
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        square=square,
        cbar_kws={"label": "Residual"},
        **kwargs
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel(col_label or residuals_df.columns.name or "Columns")
    ax.set_ylabel(row_label or residuals_df.index.name or "Rows")
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi=dpi)

    plt.show()

def plot_bipartite_network(
    B: nx.Graph,
    row_nodes: list,
    col_nodes: list,
    node_size_scale: float = 200,
    edge_alpha: float = 0.3,
    same_size: bool = False,
    weight_threshold: float = 0,
    show_edge_weights: bool = False,
    edge_width_scale: float = 1.0,
    figsize=(10, 8),
    title: str = None,
    filename_base: str = None,
    dpi: int = 600,
    row_label_name: str = "Rows",
    col_label_name: str = "Columns"
):
    """
    Visualize a bipartite network with label adjustment, thresholding, and edge weight rendering.

    Parameters:
        B (nx.Graph): Bipartite graph.
        row_nodes (list): Row-type nodes.
        col_nodes (list): Column-type nodes.
        node_size_scale (float): Scaling factor for node size (by degree).
        edge_alpha (float): Edge transparency.
        same_size (bool): If True, all nodes have the same size.
        weight_threshold (float): Minimum edge weight to include.
        show_edge_weights (bool): If True, edge width is scaled by weight.
        edge_width_scale (float): Factor to scale edge width (default 1.0).
        figsize (tuple): Size of the figure.
        title (str): Optional plot title.
        filename_base (str): If provided, saves to PNG/SVG/PDF.
        dpi (int): DPI for saved files.
        row_label_name (str): Legend label for row nodes.
        col_label_name (str): Legend label for column nodes.
    """

    # Filter edges
    edges_to_plot = [
        (u, v) for u, v, d in B.edges(data=True)
        if d.get("weight", 1) >= weight_threshold
    ]
    filtered_nodes = set(u for u, v in edges_to_plot) | set(v for u, v in edges_to_plot)
    B_sub = B.subgraph(filtered_nodes).copy()

    pos = nx.spring_layout(B_sub, seed=42, k=0.15)
    degrees = dict(B_sub.degree())

    row_sizes = [node_size_scale if same_size else degrees[n] * node_size_scale for n in row_nodes if n in B_sub]
    col_sizes = [node_size_scale if same_size else degrees[n] * node_size_scale for n in col_nodes if n in B_sub]

    fig, ax = plt.subplots(figsize=figsize)

    # Nodes
    nx.draw_networkx_nodes(B_sub, pos, nodelist=[n for n in row_nodes if n in B_sub],
                           node_color="tab:blue", node_size=row_sizes, alpha=0.8, label=row_label_name)
    nx.draw_networkx_nodes(B_sub, pos, nodelist=[n for n in col_nodes if n in B_sub],
                           node_color="tab:red", node_shape="s", node_size=col_sizes, alpha=0.8, label=col_label_name)

    # Edge weights
    edge_weights = [B_sub[u][v].get("weight", 1) for u, v in edges_to_plot]
    if show_edge_weights:
        scaled_widths = [w * edge_width_scale for w in edge_weights]
    else:
        scaled_widths = 1

    nx.draw_networkx_edges(B_sub, pos, edgelist=edges_to_plot, width=scaled_widths, alpha=edge_alpha, edge_color="gray")

    # Node labels
    texts = [ax.text(pos[n][0], pos[n][1], n, fontsize=8) for n in B_sub.nodes]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=row_label_name, markerfacecolor="tab:blue", markersize=8),
        plt.Line2D([0], [0], marker="s", color="w", label=col_label_name, markerfacecolor="tab:red", markersize=8)
    ]
    ax.legend(handles=handles, fontsize=8)

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi=dpi)

    plt.show()
    
def plot_top_n_pairs(
    sorted_pairs_df,
    *,
    metric_column: str = "Residual",
    top_n: int = 20,
    size_column: str | None = "observed",  # Column for bubble sizes (e.g., "observed" for intersection count)
    size_scale: float = 100.0,             # Scale factor for bubble sizes
    min_size: float = 50.0,                # Minimum bubble size
    max_size: float = 500.0,               # Maximum bubble size
    color_map=None,
    center_color: float | None = 0.0,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    x_label: str = "Row",
    y_label: str = "Column",
    filename_base: str | None = None,
    dpi: int = 600,
    show_colorbar: bool = True,
    show_guides: bool = False,             # Grid lines off by default
    sign: str = "both",                    # "both" | "positive" | "negative"
    order: str = "freq",                   # "freq" (default), "alpha", "custom"
    row_order: list[str] | None = None,    # used when order="custom"
    col_order: list[str] | None = None,    # used when order="custom"
):
    """
    Plot top-N row/column pairs as a bubble chart.

    Visual encoding
    ---------------
    - Color encodes `metric_column`. If `center_color` is not None, uses a diverging norm.
    - Size encodes `size_column` (default: "observed" = intersection count).
      If size_column is None or not found, uses uniform size.

    Ordering
    --------
    - order="freq": axes ordered by descending marginal frequency (uses "Count" if present,
      otherwise sum of |metric|) computed after `sign` filtering; intersected with used labels.
    - order="alpha": alphabetical.
    - order="custom": respect `row_order` / `col_order`; missing labels are ignored and the
      remaining used labels are appended alphabetically.

    Returns
    -------
    (fig, ax)
    """
    # Validate
    if not isinstance(sorted_pairs_df, pd.DataFrame):
        raise TypeError("sorted_pairs_df must be a pandas DataFrame.")
    for col in ("Row", "Column", metric_column):
        if col not in sorted_pairs_df.columns:
            raise ValueError(f"sorted_pairs_df must contain a \"{col}\" column.")
    if order not in {"freq", "alpha", "custom"}:
        raise ValueError("order must be \"freq\", \"alpha\", or \"custom\".")
    if sign not in {"both", "positive", "negative"}:
        raise ValueError("sign must be \"both\", \"positive\", or \"negative\".")

    df = sorted_pairs_df.copy()

    # Filter by sign
    if sign == "positive":
        df = df[df[metric_column] > 0]
    elif sign == "negative":
        df = df[df[metric_column] < 0]
    if df.empty:
        raise ValueError("No data to plot after applying sign filter.")

    # Rank by |metric| and take top_n
    df["abs_metric"] = df[metric_column].abs()
    df_top = df.sort_values("abs_metric", ascending=False, kind="mergesort").head(top_n)
    if df_top.empty:
        raise ValueError("No rows in top-N selection. Check inputs.")

    # Used labels
    used_rows = list(pd.Index(df_top["Row"].astype(str).unique()))
    used_cols = list(pd.Index(df_top["Column"].astype(str).unique()))

    # Ordering helpers
    def _alpha(labels: list[str]) -> list[str]:
        return sorted(labels)

    def _freq_axis(axis_col: str, used_labels: list[str]) -> list[str]:
        if "Count" in df.columns and df["Count"].notna().any():
            totals = df.groupby(axis_col)["Count"].sum().sort_values(ascending=False)
        else:
            totals = df.groupby(axis_col)["abs_metric"].sum().sort_values(ascending=False)
        order_all = list(totals.index.astype(str))
        used_set = set(used_labels)
        return [lab for lab in order_all if lab in used_set]

    def _custom_axis(provided: list[str] | None, used_labels: list[str]) -> list[str]:
        if provided is None:
            return _alpha(used_labels)
        provided = [str(x) for x in provided]
        used_set = set(used_labels)
        ordered = [lab for lab in provided if lab in used_set]
        leftovers = [lab for lab in used_labels if lab not in set(ordered)]
        return ordered + sorted(leftovers)

    # Build axis orders
    if order == "freq":
        row_labels = _freq_axis("Row", used_rows)
        col_labels = _freq_axis("Column", used_cols)
    elif order == "alpha":
        row_labels = _alpha(used_rows)
        col_labels = _alpha(used_cols)
    else:
        row_labels = _custom_axis(row_order, used_rows)
        col_labels = _custom_axis(col_order, used_cols)

    # Map to grid
    x_pos = {label: i for i, label in enumerate(row_labels)}
    y_pos = {label: i for i, label in enumerate(col_labels)}
    df_top = df_top[df_top["Row"].astype(str).isin(row_labels) & df_top["Column"].astype(str).isin(col_labels)].copy()
    if df_top.empty:
        raise ValueError("No top-N pairs remain after applying axis label orders.")
    df_top["x"] = df_top["Row"].astype(str).map(x_pos)
    df_top["y"] = df_top["Column"].astype(str).map(y_pos)

    # Calculate bubble sizes - proportional to size_column if available
    if size_column and size_column in df_top.columns:
        size_values = df_top[size_column].fillna(0).astype(float).values
        if size_values.max() > size_values.min():
            # Normalize to min_size - max_size range
            normalized = (size_values - size_values.min()) / (size_values.max() - size_values.min())
            sizes = min_size + normalized * (max_size - min_size)
        else:
            sizes = np.full(len(df_top), (min_size + max_size) / 2)
    else:
        # Uniform size if no size column
        sizes = np.full(len(df_top), float(size_scale))

    # Colors & norm
    v = df_top[metric_column].astype(float)
    vmin, vmax = float(v.min()), float(v.max())
    if color_map is None:
        color_map = "coolwarm"
    if center_color is not None and vmin < center_color < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center_color, vmax=vmax)
    elif center_color is not None and vmin == vmax == center_color:
        norm = Normalize(vmin=vmin - 1.0, vmax=vmax + 1.0)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Remove any grid styling that might come from global styles (seaborn, etc.)
    ax.set_facecolor("white")
    ax.grid(False)
    
    sc = ax.scatter(
        df_top["x"].values,
        df_top["y"].values,
        s=sizes,
        c=v.values,
        cmap=color_map,
        norm=norm,
        edgecolors="none",
    )

    ax.set_xticks(np.arange(len(row_labels)))
    ax.set_xticklabels(row_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(col_labels)))
    ax.set_yticklabels(col_labels)
    ax.invert_yaxis()
    
    # Disable all grid lines
    ax.grid(False)
    ax.set_axisbelow(True)
    # Remove any spine styling that creates grid-like appearance
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.5)

    if show_guides:
        for xi in range(len(row_labels)):
            ax.axvline(x=xi, lw=0.5, color="0.9", zorder=0)
        for yi in range(len(col_labels)):
            ax.axhline(y=yi, lw=0.5, color="0.9", zorder=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    if show_colorbar:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(metric_column)

    ax.set_xlim(-0.5, len(row_labels) - 0.5)
    ax.set_ylim(len(col_labels) - 0.5, -0.5)
    fig.tight_layout()

    if filename_base:
        dirn = os.path.dirname(filename_base)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        try:
            save_plot(filename_base, dpi=dpi)
        except Exception:
            fig.savefig(f"{filename_base}.png", dpi=dpi, bbox_inches="tight")
            fig.savefig(f"{filename_base}.svg", dpi=dpi, bbox_inches="tight")
            fig.savefig(f"{filename_base}.pdf", dpi=dpi, bbox_inches="tight")

    return fig, ax


# plots for groups


def _resolve_metric(stats_df: pd.DataFrame, metric: str) -> str:
    """
    Resolve metric name with case-insensitive matching.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with metric columns.
    metric : str
        Metric name to find.
        
    Returns
    -------
    str
        Actual column name matching the metric.
        
    Raises
    ------
    ValueError
        If metric not found.
    """
    # exact
    if metric in stats_df.columns:
        return metric
    # case-insensitive
    for m in stats_df.columns:
        if str(m).lower() == str(metric).lower():
            return m
    raise ValueError(f"Metric '{metric}' not found. Available: {list(stats_df.columns)}")


# ---------- Common tidy helper ----------
def _tidy_from_wide(stats_df: pd.DataFrame):
    """Accepts a 'wide' frame (index=['group','<entity>'], columns=metrics)."""
    if isinstance(stats_df.index, pd.MultiIndex):
        gname, ename = stats_df.index.names
        df_long = (
            stats_df.reset_index()
                    .melt(id_vars=[gname, ename], var_name="metric", value_name="value")
                    .rename(columns={gname: "group", ename: "entity"})
        )
    else:
        idx_name = stats_df.index.name or "entity"
        df_long = (
            stats_df.reset_index()
                    .melt(id_vars=[idx_name, "group"], var_name="metric", value_name="value")
                    .rename(columns={idx_name: "entity"})
        )
    return df_long

def plot_group_metric_heatmap(
    stats_df: pd.DataFrame,
    metric: str = "Number of documents",
    top_k: int = 20,
    title: str | None = None,
    filename_base: str | None = None,
    dpi: int = 600,
):
    """
    Plot a heatmap of a metric by entity and group.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        DataFrame containing at least the columns:
        - "entity": item label (e.g. keyword, source, author).
        - "group": group label (e.g. field, country group).
        - ``metric``: numeric values for the chosen metric.
        Multiple rows per (entity, group) are allowed and will be summed.
    metric : str, default "Number of documents"
        Name of the column in ``stats_df`` to plot.
    top_k : int, default 20
        Maximal number of entities (rows) to show. If ``top_k <= 0``,
        all entities are shown. Entities are ranked by their maximum
        value across groups.
    title : str or None, optional
        Title for the plot. If ``None``, ``metric`` is used.
    filename_base : str or None, optional
        Base filename (without extension) for saving the figure. If
        ``None``, the figure is not saved.
    dpi : int, default 600
        Resolution used when saving the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the heatmap.
    """
    import matplotlib.pyplot as plt

    df = stats_df.copy()

    required = {"entity", "group", metric}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            f"stats_df must contain columns {sorted(required)}, "
            f"missing: {sorted(missing)}"
        )

    # Keep only what we need and drop missing metric values
    df = df[["entity", "group", metric]].dropna(subset=[metric])

    # Aggregate duplicates for (entity, group)
    df = (
        df.groupby(["entity", "group"], dropna=False)[metric]
        .sum()
        .reset_index()
    )

    if df.empty:
        raise ValueError("No data available to plot after aggregation.")

    # Pivot to entity × group matrix
    mat = df.pivot(index="entity", columns="group", values=metric).fillna(0)

    if mat.empty:
        raise ValueError("No data available to plot after pivoting.")

    # Apply top_k based on the maximum across groups
    if top_k and top_k > 0 and len(mat) > top_k:
        keep = mat.max(axis=1).nlargest(top_k).index
        mat = mat.loc[keep]

    # Sort for nicer display
    mat = mat.sort_index(axis=0)
    mat = mat.sort_index(axis=1)

    fig_width = max(6.0, 0.9 * mat.shape[1])
    fig_height = max(5.0, 0.35 * mat.shape[0])

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(mat.values, aspect="auto")

    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")

    ax.set_xlabel("Group")
    ax.set_ylabel("Entity")
    ax.set_title(title if title is not None else metric)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)

    fig.tight_layout()
    save_plot(filename_base, dpi=dpi)

    return fig



# ---------- Bubble map: size = metric ----------
def plot_group_metric_bubblemap(
    stats_df: pd.DataFrame,
    metric: str = "H-index",                 # COLOR metric
    top_k: int = 25,                         # keep entities by max Docs
    title: str | None = None,
    filename_base: str | None = None,        # base path, no extension
    dpi: int = 600,
    gap: float = 0.06,                       # gap inside each 1x1 grid cell
    xpad_extra: float = 0.20,                # EXTRA padding (cells) beyond circle radius on L/R
    ypad_extra: float = 0.20,                # EXTRA padding (cells) beyond circle radius on T/B
):
    """
    Plot a bubble map of group metrics with size and color encoding.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Statistics DataFrame with metrics per entity and group.
    metric : str
        Metric column for bubble color.
    top_k : int
        Number of top entities to show.
    title : str, optional
        Plot title.
    filename_base : str, optional
        Base path for saving (without extension).
    dpi : int
        Resolution for saved images.
    gap : float
        Gap between bubbles in grid cells.
    xpad_extra : float
        Extra horizontal padding.
    ypad_extra : float
        Extra vertical padding.
    """
    size_metric  = _resolve_metric(stats_df, "Number of Documents")  # for circle size
    color_metric = _resolve_metric(stats_df, metric)                  # for color

    tidy = _tidy_from_wide(stats_df)
    sub_size  = tidy[tidy["metric"] == size_metric]
    sub_color = tidy[tidy["metric"] == color_metric]

    mat_size  = sub_size.pivot(index="entity", columns="group", values="value")
    mat_color = sub_color.pivot(index="entity", columns="group", values="value")

    # Align matrices
    ents = sorted(set(mat_size.index) | set(mat_color.index))
    grps = sorted(set(mat_size.columns) | set(mat_color.columns))
    mat_size  = mat_size.reindex(index=ents, columns=grps).fillna(0.0)
    mat_color = mat_color.reindex(index=ents, columns=grps).fillna(0.0)
    if mat_size.empty:
        raise ValueError("No data available to plot.")

    # Keep most relevant entities by max Docs
    if top_k and len(mat_size) > top_k:
        keep = mat_size.max(axis=1).nlargest(top_k).index
        mat_size  = mat_size.loc[keep]
        mat_color = mat_color.loc[keep]
        ents = list(mat_size.index)

    vals_docs  = mat_size.values.astype(float)
    vals_color = mat_color.values.astype(float)

    vmax_docs = np.nanmax(vals_docs) if vals_docs.size else 0.0
    r_max_cell = max(0.0, 0.5 - gap)  # radius in DATA units; ensures no overlap

    if vmax_docs <= 0:
        radii = np.full_like(vals_docs, r_max_cell * 0.18)
    else:
        radii = r_max_cell * np.sqrt(np.clip(vals_docs / vmax_docs, 0, 1))

    # ----- figure -----
    fig, ax = plt.subplots(figsize=(max(6, 0.9*len(grps)), max(5, 0.35*len(ents))))

    circles, colors = [], []
    for i in range(len(ents)):
        for j in range(len(grps)):
            r = float(radii[i, j])
            if r > 0:
                circles.append(Circle((j, i), radius=r))
                colors.append(float(vals_color[i, j]))

    pc = PatchCollection(circles, edgecolor="none")
    pc.set_array(np.asarray(colors, dtype=float))
    if colors:
        pc.set_clim(vmin=float(np.nanmin(colors)), vmax=float(np.nanmax(colors)))
    ax.add_collection(pc)
    ax.set_aspect("equal")

    # ticks/labels
    gx = np.arange(len(grps)); ey = np.arange(len(ents))
    ax.set_xticks(gx); ax.set_xticklabels(grps, rotation=30, ha="right")
    ax.set_yticks(ey); ax.set_yticklabels(ents)
    ax.set_xlabel("Group"); ax.set_ylabel("Entity")
    if title: ax.set_title(title)

    # ----- dynamic padding so edge bubbles are fully visible -----
    left_r  = float(radii[:, 0].max())  if radii.size else 0.0
    right_r = float(radii[:, -1].max()) if radii.size else 0.0
    top_r   = float(radii[0, :].max())  if radii.size else 0.0
    bot_r   = float(radii[-1, :].max()) if radii.size else 0.0

    ax.set_xlim(- (left_r  + xpad_extra), (len(grps) - 1) + (right_r + xpad_extra))
    ax.set_ylim(  (len(ents) - 1) + (bot_r + ypad_extra), - (top_r + ypad_extra))  # invert y

    cbar = plt.colorbar(pc, ax=ax, pad=0.02)
    cbar.set_label(color_metric)

    fig.tight_layout()
    if filename_base:
        save_plot(filename_base, dpi=dpi)
    return fig


# ---------- Slope chart: compare two groups ----------
def plot_group_metric_slope(
    stats_df: pd.DataFrame,
    metric: str,
    group_a: str,
    group_b: str,
    top_k: int = 20,
    title: str | None = None,
    filename_base: str | None = None,  # base path without extension
    dpi: int = 600,
    connector_color: str = "0.75",      # neutral gray
    connector_alpha: float = 0.8,
    color_by_change: bool = False,      # set True to use up/down colors
    up_color: str = "#2ca02c",          # green
    down_color: str = "#d62728",        # red
):
    """
    Plot a slope chart comparing a metric between two groups.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Statistics DataFrame with metrics per entity and group.
    metric : str
        Metric to compare.
    group_a : str
        First group name.
    group_b : str
        Second group name.
    top_k : int
        Number of top entities to show.
    title : str, optional
        Plot title.
    filename_base : str, optional
        Base path for saving (without extension).
    dpi : int
        Resolution for saved images.
    connector_color : str
        Color for connecting lines.
    connector_alpha : float
        Transparency of connecting lines.
    color_by_change : bool
        If True, color lines by direction of change.
    up_color : str
        Color for increases.
    down_color : str
        Color for decreases.
    """
    metric = _resolve_metric(stats_df, metric)
    tidy = _tidy_from_wide(stats_df)

    # restrict to the two groups
    sub = tidy[(tidy["metric"] == metric) & (tidy["group"].isin([group_a, group_b]))]
    if sub.empty:
        raise ValueError("No data for the selected metric and groups.")
    mat = sub.pivot(index="entity", columns="group", values="value").dropna(how="any")
    if mat.empty:
        raise ValueError("No overlapping entities between the two groups.")

    # focus on most relevant entities
    pick = mat[[group_a, group_b]].max(axis=1).nlargest(min(top_k, len(mat))).index
    mat = mat.loc[pick].sort_values(group_b)

    y = np.arange(len(mat))
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(mat))))

    # points + lines for the two groups
    ax.plot(mat[group_a].values, y, "o-", label=group_a, zorder=3)
    ax.plot(mat[group_b].values, y, "o-", label=group_b, zorder=3)

    # neutral (or directional) connector segments behind points
    for i in range(len(mat)):
        x1 = float(mat[group_a].iloc[i]); x2 = float(mat[group_b].iloc[i])
        if color_by_change:
            col = up_color if x2 >= x1 else down_color
        else:
            col = connector_color
        ax.hlines(y=i, xmin=x1, xmax=x2, color=col, alpha=connector_alpha, linewidth=2, zorder=2)

    ax.set_yticks(y); ax.set_yticklabels(mat.index)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.legend()
    if title: ax.set_title(title)
    fig.tight_layout()
    if filename_base: save_plot(filename_base, dpi=dpi)
    return fig


# ---------- Bump chart: ranks across groups ----------

def plot_group_metric_bump(
    stats_df: pd.DataFrame,
    metric: str = "Number of documents",    # case-insensitive
    entities: list[str] | None = None,
    top_k_auto: int = 12,
    title: str | None = None,
    filename_base: str | None = None,
    dpi: int = 600,
    groups_order: list[str] | None = None,
    show_end_labels: bool = True,
    label_side: str = "right",              # "right", "left", or "both"
    min_label_gap: float = 0.28,            # vertical spacing (rank units)
    label_pad_x: float = 0.20,              # horizontal offset of labels (axis units)
    line_alpha: float = 0.9,
    linewidth: float = 2.0,
):
    """
    Plot a bump chart showing entity rank changes across groups.
    
    Visualizes how entities (authors, sources, etc.) change rank position
    across different groups (years, categories, etc.).
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Statistics DataFrame with metrics per entity and group.
    metric : str
        Metric to use for ranking (case-insensitive).
    entities : list, optional
        Specific entities to include. If None, uses top_k_auto.
    top_k_auto : int
        Number of top entities to auto-select if entities is None.
    title : str, optional
        Plot title.
    filename_base : str, optional
        Base path for saving (without extension).
    dpi : int
        Resolution for saved images.
    groups_order : list, optional
        Custom order for groups on x-axis.
    show_end_labels : bool
        Whether to show entity labels at line ends.
    label_side : str
        Where to show labels: 'right', 'left', or 'both'.
    min_label_gap : float
        Minimum vertical spacing between labels.
    label_pad_x : float
        Horizontal padding for labels.
    line_alpha : float
        Line transparency.
    linewidth : float
        Line width.
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    # ---------- helpers ----------
    def _infer_entity_col(df: pd.DataFrame) -> str:
        cands = ["entity","Keyword","Source","Source title","Author","Authors",
                 "Affiliation","Reference","Country","Field","Area","Science",
                 "Term","Processed Title","Processed Abstract","Journal"]
        for c in cands:
            if c in df.columns: return c
        low = {str(c).lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in low: return low[c.lower()]
        # fallback: anything that's not 'group' and not obviously numeric-only
        for c in df.columns:
            if c != "group" and not pd.api.types.is_numeric_dtype(df[c]):
                return c
        return df.columns[1]

    def _tidy(df: pd.DataFrame) -> pd.DataFrame:
        if "group" in df.columns:
            ec = _infer_entity_col(df)
            m = df.melt(id_vars=["group", ec], var_name="metric", value_name="value")
            return m.rename(columns={ec: "entity"})
        if isinstance(df.index, pd.MultiIndex):
            gname, ename = df.index.names
            return (df.reset_index()
                      .melt(id_vars=[gname, ename], var_name="metric", value_name="value")
                      .rename(columns={gname: "group", ename: "entity"}))
        raise ValueError("stats_df must have column 'group' (and an entity column).")

    def _resolve_metric_name(tidy: pd.DataFrame, name: str) -> str:
        for m in tidy["metric"].dropna().unique():
            if str(m).lower() == str(name).lower():
                return m
        raise ValueError(f"Metric '{name}' not found.")

    def _spread_positions(y_vals, low, high, min_gap):
        """Greedy distribute y-positions so neighboring labels are at least min_gap apart."""
        idx_y = sorted(enumerate(y_vals), key=lambda t: t[1])
        out = [None]*len(y_vals)
        cur = low
        for idx, y in idx_y:
            y = max(y, cur)
            out[idx] = y
            cur = y + min_gap
        overflow = out[idx_y[-1][0]] - high
        if overflow > 0:
            out = [p - overflow for p in out]
        return [min(max(p, low), high) for p in out]

    # ---------- tidy + ranks ----------
    tidy = _tidy(stats_df)
    metric = _resolve_metric_name(tidy, metric)
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")

    sub = tidy[tidy["metric"] == metric].copy()
    sub["rank"] = sub.groupby("group")["value"].rank(ascending=False, method="min")

    if entities is None:
        keep = sub.groupby("entity")["value"].max().nlargest(top_k_auto).index
        sub = sub[sub["entity"].isin(keep)]
    else:
        sub = sub[sub["entity"].isin(entities)]

    # group order
    if groups_order is None:
        seen = []
        for g in tidy["group"]:
            if pd.notna(g) and g not in seen:
                seen.append(g)
        groups_order = seen
    order = {g:i for i,g in enumerate(groups_order)}

    # ---------- plot ----------
    n_groups = len(groups_order)
    n_entities = sub["entity"].nunique()
    fig, ax = plt.subplots(figsize=(max(6, 0.9*n_groups), max(4, 0.5*n_entities)))

    # lines
    for ent, df_e in sub.groupby("entity"):
        df_e = df_e[df_e["group"].isin(groups_order)].copy()
        df_e["gx"] = df_e["group"].map(order)
        df_e = df_e.dropna(subset=["gx"]).sort_values("gx")
        ax.plot(df_e["gx"], df_e["rank"], marker="o",
                linewidth=linewidth, alpha=line_alpha, label=str(ent))

    # axes
    max_rank = int(np.nanmax(sub["rank"])) if len(sub) else 1
    ax.set_xlim(-0.2, n_groups-1+0.2)
    ax.set_ylim(max_rank+0.5, 0.5)  # invert
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(groups_order, rotation=30, ha="right")
    ax.set_yticks(range(1, max_rank+1))
    ax.set_xlabel("Group")
    ax.set_ylabel("Rank (1 = best)")
    if title: ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    # ---------- non-overlapping end labels ----------
    if show_end_labels and n_entities:
        left_x  = 0 - label_pad_x
        right_x = (n_groups-1) + label_pad_x

        # collect endpoints
        ends = {}
        for side in ["left","right"]:
            if (side == "left" and label_side in ("left","both")) or \
               (side == "right" and label_side in ("right","both")):
                xg = 0 if side == "left" else (n_groups-1)
                df_end = sub[sub["group"] == groups_order[xg]]
                y_raw = df_end.groupby("entity")["rank"].min().reindex(sub["entity"].unique()).dropna()
                if not y_raw.empty:
                    ends[side] = y_raw

        low, high = 0.5, max_rank + 0.5
        for side, y_series in ends.items():
            ents = list(y_series.index)
            y_vals = y_series.values.astype(float).tolist()
            y_adj = _spread_positions(y_vals, low, high, min_label_gap)

            for e, y0, y1 in zip(ents, y_vals, y_adj):
                x_point = 0 if side == "left" else (n_groups-1)
                x_label = left_x if side == "left" else right_x
                # connector
                ax.plot([x_point, x_label - (0.02 if side=="right" else -0.02)], [y0, y1],
                        color="0.6", linewidth=1, alpha=0.9, zorder=1, clip_on=False)
                # text
                ha = "right" if side == "left" else "left"
                ax.text(x_label, y1, str(e), va="center", ha=ha, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                        clip_on=False)

        # extend limits to fit labels
        ax.set_xlim(-0.6 if label_side in ("left","both") else -0.2,
                    (n_groups-1) + (0.6 if label_side in ("right","both") else 0.2))

    fig.tight_layout()
    save_plot(filename_base, dpi=dpi)
    return fig



"""
==========================================================================

This part provides visualization functions for Sleeping Beauty analysis,
including citation trajectories, overview dashboards, and comparative plots.
"""


from biblium.utilsbib import SleepingBeautyResult

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid")



def plot_citation_trajectory(
    paper_result: SleepingBeautyResult,
    figsize: Tuple[int, int] = (12, 6),
    show_expected_line: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 600
) -> plt.Figure:
    """
    Plot the citation trajectory of a single paper highlighting its SB characteristics.
    
    Args:
        paper_result: SleepingBeautyResult object
        figsize: Figure size
        show_expected_line: Whether to show the "expected" linear trajectory
        save_path: Path to save the figure (without extension)
        dpi: Resolution for saved figures
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    history = paper_result.citation_history
    years = sorted(history.keys())
    citations = [history[y] for y in years]
    
    # Plot actual citations
    ax.bar(years, citations, color="steelblue", alpha=0.7, label="Actual Citations")
    ax.plot(years, citations, "o-", color="darkblue", linewidth=2, markersize=6)
    
    # Show expected linear trajectory
    if show_expected_line and paper_result.awakening_year:
        pub_year = paper_result.publication_year
        peak_year = paper_result.max_citation_year
        peak_citations = paper_result.max_citations_in_year
        
        if peak_year > pub_year:
            expected_years = list(range(pub_year, peak_year + 1))
            expected_citations = [
                peak_citations * (y - pub_year) / (peak_year - pub_year) 
                for y in expected_years
            ]
            ax.plot(expected_years, expected_citations, "--", color="red", 
                   linewidth=2, label="Expected Linear Growth")
    
    # Mark awakening year
    if paper_result.awakening_year:
        ax.axvline(x=paper_result.awakening_year, color="green", linestyle=":", 
                  linewidth=2, label=f"Awakening Year ({paper_result.awakening_year})")
    
    # Mark publication year
    ax.axvline(x=paper_result.publication_year, color="gray", linestyle="--", 
              linewidth=1.5, alpha=0.7, label=f"Published ({paper_result.publication_year})")
    
    # Formatting
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Citations", fontsize=12)
    ax.set_title(f"Citation Trajectory: {paper_result.title[:60]}...\n"
                f"Beauty Coefficient: {paper_result.beauty_coefficient:.1f} | "
                f"Sleep Duration: {paper_result.sleep_duration} years", fontsize=11)
    ax.legend(loc="upper left")
    ax.set_xlim(min(years) - 1, max(years) + 1)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig


def plot_sleeping_beauties_overview(
    sb_df,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    dpi: int = 600
) -> plt.Figure:
    """
    Create an overview visualization of all identified Sleeping Beauties.
    
    Args:
        sb_df: DataFrame of Sleeping Beauties
        figsize: Figure size
        save_path: Path to save the figure (without extension)
        dpi: Resolution for saved figures
    
    Returns:
        matplotlib Figure object
    """
    if sb_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No Sleeping Beauties found with current criteria",
               ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Beauty Coefficient Distribution
    ax1 = axes[0, 0]
    ax1.hist(sb_df["beauty_coefficient"], bins=20, color="steelblue", edgecolor="white", alpha=0.7)
    ax1.set_xlabel("Beauty Coefficient")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Beauty Coefficients")
    ax1.axvline(sb_df["beauty_coefficient"].median(), color="red", linestyle="--", 
               label=f"Median: {sb_df['beauty_coefficient'].median():.1f}")
    ax1.legend()
    
    # 2. Sleep Duration vs Beauty Coefficient
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        sb_df["sleep_duration"], 
        sb_df["beauty_coefficient"],
        c=sb_df["total_citations"],
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="white"
    )
    ax2.set_xlabel("Sleep Duration (Years)")
    ax2.set_ylabel("Beauty Coefficient")
    ax2.set_title("Sleep Duration vs Beauty Coefficient")
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Total Citations")
    
    # 3. Publication Year Distribution
    ax3 = axes[1, 0]
    pub_years = sb_df["publication_year"].value_counts().sort_index()
    ax3.bar(pub_years.index, pub_years.values, color="coral", edgecolor="white", alpha=0.7)
    ax3.set_xlabel("Publication Year")
    ax3.set_ylabel("Number of Sleeping Beauties")
    ax3.set_title("Sleeping Beauties by Publication Year")
    
    # 4. Awakening Intensity Distribution
    ax4 = axes[1, 1]
    valid_intensity = sb_df["awakening_intensity"].dropna()
    if len(valid_intensity) > 0:
        ax4.hist(valid_intensity, bins=15, color="forestgreen", edgecolor="white", alpha=0.7)
        ax4.set_xlabel("Awakening Intensity (Citation Ratio)")
        ax4.set_ylabel("Count")
        ax4.set_title("Distribution of Awakening Intensity")
        ax4.axvline(valid_intensity.median(), color="red", linestyle="--",
                   label=f"Median: {valid_intensity.median():.1f}")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No awakening intensity data", ha="center", va="center")
        ax4.axis("off")
    
    plt.suptitle(f"Sleeping Beauties Analysis Overview (n={len(sb_df)})", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig


def plot_multi_paper_trajectories(
    papers: List[SleepingBeautyResult],
    figsize: Tuple[int, int] = (14, 8),
    max_papers: int = 10,
    save_path: Optional[str] = None,
    dpi: int = 600
) -> plt.Figure:
    """
    Plot citation trajectories for multiple papers on the same chart.
    
    Args:
        papers: List of SleepingBeautyResult objects
        figsize: Figure size
        max_papers: Maximum number of papers to plot
        save_path: Path to save the figure (without extension)
        dpi: Resolution for saved figures
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    papers = papers[:max_papers]
    colors = plt.cm.tab10(np.linspace(0, 1, len(papers)))
    
    for paper, color in zip(papers, colors):
        history = paper.citation_history
        years = sorted(history.keys())
        citations = [history[y] for y in years]
        
        # Normalize by max to compare trajectories
        max_cite = max(citations) if citations else 1
        normalized = [c / max_cite for c in citations]
        
        label = f"{paper.title[:30]}... (B={paper.beauty_coefficient:.0f})"
        ax.plot(years, normalized, "o-", color=color, linewidth=2, 
               markersize=4, label=label, alpha=0.8)
        
        # Mark awakening
        if paper.awakening_year and paper.awakening_year in history:
            ax.scatter([paper.awakening_year], 
                      [history[paper.awakening_year] / max_cite],
                      color=color, s=150, marker="*", zorder=5)
    
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Normalized Citations (0-1)", fontsize=12)
    ax.set_title("Comparative Citation Trajectories of Sleeping Beauties\n(Stars mark awakening years)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig


def plot_beauty_coefficient_ranking(
    all_papers_df,
    top_n: int = 30,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    dpi: int = 600
) -> plt.Figure:
    """
    Create a horizontal bar chart ranking papers by Beauty Coefficient.
    
    Args:
        all_papers_df: DataFrame with all papers and their metrics
        top_n: Number of top papers to show
        figsize: Figure size
        save_path: Path to save the figure (without extension)
        dpi: Resolution for saved figures
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    top_papers = all_papers_df.nlargest(top_n, "beauty_coefficient")
    
    # Truncate titles for display
    titles = [t[:50] + "..." if len(t) > 50 else t for t in top_papers["title"]]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_papers)))
    
    bars = ax.barh(range(len(top_papers)), top_papers["beauty_coefficient"], 
                   color=colors, edgecolor="white", alpha=0.8)
    
    ax.set_yticks(range(len(top_papers)))
    ax.set_yticklabels(titles, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Beauty Coefficient", fontsize=12)
    ax.set_title(f"Top {top_n} Papers by Beauty Coefficient", fontsize=14)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_papers["beauty_coefficient"])):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f"{val:.1f}", va="center", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig


def plot_awakening_timeline(
    sb_df,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    dpi: int = 600
) -> plt.Figure:
    """
    Plot a timeline showing when Sleeping Beauties were published and awakened.
    
    Args:
        sb_df: DataFrame of Sleeping Beauties
        figsize: Figure size
        save_path: Path to save the figure (without extension)
        dpi: Resolution for saved figures
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    valid_data = sb_df[sb_df["awakening_year"].notna()].copy()
    valid_data = valid_data.sort_values("Year")
    
    if len(valid_data) == 0:
        ax.text(0.5, 0.5, "No papers with valid awakening data", ha="center", va="center")
        return fig
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(valid_data)))
    
    for i, (idx, row) in enumerate(valid_data.iterrows()):
        pub = row["Year"]
        awake = row["awakening_year"]
        
        # Draw sleeping period
        ax.plot([pub, awake], [i, i], "o-", color=colors[i], linewidth=3,
               markersize=8, alpha=0.7)
        
        # Add title annotation
        title_short = row["title"][:40] + "..." if len(row["title"]) > 40 else row["title"]
        ax.annotate(title_short, xy=(awake, i), xytext=(5, 0), 
                   textcoords="offset points", fontsize=8, va="center")
    
    ax.set_yticks(range(len(valid_data)))
    ax.set_yticklabels([f"Paper {i+1}" for i in range(len(valid_data))])
    ax.set_xlabel("Year", fontsize=12)
    ax.set_title("Sleeping Beauty Timeline\n(Circle = Publication, Line End = Awakening)", fontsize=14)
    
    # Add legend for sleep duration
    ax.annotate("← Sleep Duration →", xy=(0.5, -0.1), xycoords="axes fraction",
               ha="center", fontsize=10, style="italic")
    
    plt.tight_layout()
    
    if save_path:
        save_plot(save_path, dpi=dpi)
    
    return fig

# =============================================================================
# ALTMETRICS VISUALIZATIONS
# =============================================================================

def plot_altmetric_score_distribution(
    result: dict,
    filename: str = "altmetric_score_distribution",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot altmetric score distribution.
    
    Parameters
    ----------
    result : dict
        Result from analyze_altmetrics().
    filename : str
        Base filename for saving.
    dpi : int
        Resolution.
    show : bool
        Whether to display.
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df = result["summary_df"]
    scores = df["Altmetric Score"]
    scores_nonzero = scores[scores > 0]
    
    if len(scores_nonzero) > 0:
        ax.hist(np.log1p(scores_nonzero), bins=50, color='steelblue', 
                edgecolor='white', alpha=0.8)
        ax.set_xlabel('log(Altmetric Score + 1)', fontsize=11)
        ax.set_ylabel('Number of Papers', fontsize=11)
        n_with = len(scores_nonzero)
        pct = n_with / len(scores) * 100
        ax.set_title(f'Altmetric Score Distribution\n({n_with} papers with attention, {pct:.1f}%)', 
                    fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No altmetric scores available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_altmetric_source_coverage(
    result: dict,
    filename: str = "altmetric_source_coverage",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot source coverage (% of papers with mentions from each source).
    
    Bars are ordered with highest coverage at top.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    source_coverage = result["source_coverage"]
    
    if source_coverage:
        # Sort by value descending (highest at top in horizontal bar)
        sorted_pairs = sorted(source_coverage.items(), key=lambda x: x[1])
        sources = [p[0] for p in sorted_pairs]
        coverage = [p[1] for p in sorted_pairs]
        
        bars = ax.barh(range(len(sources)), coverage, color='steelblue', 
                      alpha=0.8, edgecolor='white', height=0.6)
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources)
        ax.set_xlabel('Coverage (%)', fontsize=11)
        ax.set_xlim(0, max(coverage) * 1.15)
        ax.set_title('Source Coverage (% of papers with mentions)', 
                    fontsize=12, fontweight='bold')
        
        for bar, cov in zip(bars, coverage):
            ax.text(bar.get_width() + max(coverage) * 0.02, 
                   bar.get_y() + bar.get_height()/2,
                   f'{cov:.1f}%', va='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No coverage data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_altmetric_total_by_source(
    result: dict,
    filename: str = "altmetric_total_by_source",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot total mentions/readers by source.
    
    Bars are ordered with highest total at top.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df = result["summary_df"]
    source_cols = ["Twitter", "Mendeley", "News", "Blogs", "Reddit", 
                   "Wikipedia", "Policy", "Patents", "GitHub"]
    
    available = [c for c in source_cols if c in df.columns]
    
    if available:
        totals = {k: df[k].sum() for k in available if df[k].sum() > 0}
        
        if totals:
            # Sort ascending for horizontal bar (highest at top)
            sorted_items = sorted(totals.items(), key=lambda x: x[1])
            names = [x[0] for x in sorted_items]
            values = [x[1] for x in sorted_items]
            
            bars = ax.barh(range(len(names)), values, color='steelblue', 
                          alpha=0.8, height=0.6, edgecolor='white')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel('Total Mentions/Readers', fontsize=11)
            ax.set_title('Total by Source', fontsize=12, fontweight='bold')
            
            max_val = max(values)
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + max_val * 0.02, 
                       bar.get_y() + bar.get_height()/2,
                       f'{val:,}', va='center', fontsize=9)
            
            ax.set_xlim(0, max_val * 1.15)
        else:
            ax.text(0.5, 0.5, 'No source data', 
                   ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No source columns available', 
               ha='center', va='center', transform=ax.transAxes)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_altmetric_score_components(
    result: dict,
    filename: str = "altmetric_score_components",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot total score by component type.
    
    Bars ordered with highest at top.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df = result["summary_df"]
    component_cols = ["Social Score", "Scholarly Score", "Public Score", "Practice Score"]
    available = [c for c in component_cols if c in df.columns]
    
    if available:
        sums = {c.replace(" Score", ""): df[c].sum() for c in available}
        
        # Sort ascending for horizontal bar (highest at top)
        sorted_items = sorted(sums.items(), key=lambda x: x[1])
        names = [x[0] for x in sorted_items]
        values = [x[1] for x in sorted_items]
        
        bars = ax.barh(range(len(names)), values, color='steelblue', 
                      alpha=0.8, height=0.6, edgecolor='white')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Total Score', fontsize=11)
        ax.set_title('Score Components', fontsize=12, fontweight='bold')
        
        max_val = max(values) if values else 1
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max_val * 0.02, 
                   bar.get_y() + bar.get_height()/2,
                   f'{val:,.0f}', va='center', fontsize=9)
        
        ax.set_xlim(0, max_val * 1.15)
    else:
        ax.text(0.5, 0.5, 'No component data', 
               ha='center', va='center', transform=ax.transAxes)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_altmetric_citation_correlation(
    result: dict,
    filename: str = "altmetric_citation_correlation",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 8),
):
    """
    Plot citation vs altmetric score scatter plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df = result["summary_df"]
    
    if "Citations" in df.columns:
        nonzero = df[(df["Citations"] > 0) & (df["Altmetric Score"] > 0)]
        
        if len(nonzero) > 5:
            ax.scatter(nonzero["Citations"], nonzero["Altmetric Score"],
                      alpha=0.5, s=30, c='steelblue', edgecolor='white', linewidth=0.5)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Citations (log scale)', fontsize=11)
            ax.set_ylabel('Altmetric Score (log scale)', fontsize=11)
            ax.set_title('Citations vs Altmetric Score', fontsize=12, fontweight='bold')
            
            stats = result.get("statistics", {})
            if "citation_altmetric_correlation" in stats:
                corr = stats["citation_altmetric_correlation"]
                ax.text(0.05, 0.95, f'Spearman r = {corr:.2f}',
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Insufficient data for scatter plot', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, 'Citation data not available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_altmetric_trend_mean(
    result: dict,
    filename: str = "altmetric_trend_mean",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot mean altmetric score over time.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    temporal = result.get("temporal_trends", pd.DataFrame())
    
    if len(temporal) == 0:
        ax.text(0.5, 0.5, 'Temporal data not available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.grid(False)
        return fig
    
    df = temporal.dropna(subset=["Year"])
    df = df[(df["Year"] >= 2000) & (df["Year"] <= 2030)]
    
    if "Mean Score" in df.columns and len(df) > 1:
        ax.plot(df["Year"], df["Mean Score"], marker='o', linewidth=2, 
               color='steelblue', markersize=6)
        ax.fill_between(df["Year"], df["Mean Score"], alpha=0.2, color='steelblue')
        ax.set_xlabel('Publication Year', fontsize=11)
        ax.set_ylabel('Mean Altmetric Score', fontsize=11)
        ax.set_title('Mean Altmetric Score by Year', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Insufficient trend data', 
               ha='center', va='center', transform=ax.transAxes)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_altmetric_trend_total(
    result: dict,
    filename: str = "altmetric_trend_total",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot total altmetric score by year.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    temporal = result.get("temporal_trends", pd.DataFrame())
    
    if len(temporal) == 0:
        ax.text(0.5, 0.5, 'Temporal data not available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.grid(False)
        return fig
    
    df = temporal.dropna(subset=["Year"])
    df = df[(df["Year"] >= 2000) & (df["Year"] <= 2030)]
    
    if "Total Score" in df.columns:
        ax.bar(df["Year"], df["Total Score"], color='steelblue', alpha=0.8, edgecolor='white')
        ax.set_xlabel('Publication Year', fontsize=11)
        ax.set_ylabel('Total Altmetric Score', fontsize=11)
        ax.set_title('Total Altmetric Score by Year', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No total score data', 
               ha='center', va='center', transform=ax.transAxes)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_altmetric_correlation_matrix(
    result: dict,
    filename: str = "altmetric_correlation_matrix",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 8),
):
    """
    Plot correlation matrix between altmetric measures.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    corr_matrix = result.get("correlation_matrix", pd.DataFrame())
    
    if len(corr_matrix) > 0:
        cols = ["Altmetric Score", "Twitter", "Mendeley", "News", 
                "Social Score", "Scholarly Score", "Citations"]
        available_corr = [c for c in cols if c in corr_matrix.columns]
        
        if len(available_corr) > 1:
            corr_subset = corr_matrix.loc[available_corr, available_corr]
            
            im = ax.imshow(corr_subset.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax.set_xticks(range(len(available_corr)))
            ax.set_yticks(range(len(available_corr)))
            
            short_labels = [c.replace(" Score", "").replace("Altmetric", "Altmetric") 
                           for c in available_corr]
            ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(short_labels, fontsize=9)
            
            for i in range(len(available_corr)):
                for j in range(len(available_corr)):
                    val = corr_subset.iloc[i, j]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                           fontsize=9, color=color)
            
            ax.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for correlation matrix', 
                   ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No correlation data', 
               ha='center', va='center', transform=ax.transAxes)
    
    ax.grid(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


# Keep legacy function names for backward compatibility
def plot_altmetrics(result, filename="altmetrics", **kwargs):
    """Legacy wrapper - creates multiple separate plots."""
    figs = []
    base = filename.replace(".png", "").replace(".pdf", "").replace(".svg", "")
    figs.append(plot_altmetric_score_distribution(result, filename=f"{base}_distribution", **kwargs))
    figs.append(plot_altmetric_source_coverage(result, filename=f"{base}_coverage", **kwargs))
    figs.append(plot_altmetric_score_components(result, filename=f"{base}_components", **kwargs))
    figs.append(plot_altmetric_citation_correlation(result, filename=f"{base}_citation_corr", **kwargs))
    return figs


def plot_altmetric_sources(result, filename="altmetric_sources", **kwargs):
    """Legacy wrapper - creates source breakdown plot."""
    return plot_altmetric_total_by_source(result, filename=filename, **kwargs)


def plot_altmetric_trends(result, filename="altmetric_trends", **kwargs):
    """Legacy wrapper - creates multiple trend plots."""
    figs = []
    base = filename.replace(".png", "").replace(".pdf", "").replace(".svg", "")
    figs.append(plot_altmetric_trend_mean(result, filename=f"{base}_mean", **kwargs))
    figs.append(plot_altmetric_trend_total(result, filename=f"{base}_total", **kwargs))
    return figs


# =============================================================================
# NOVELTY / ATYPICALITY VISUALIZATIONS
# =============================================================================

def plot_novelty_distribution(
    result: dict,
    filename: str = "novelty_distribution",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot distribution of composite novelty scores.
    
    Parameters
    ----------
    result : dict
        Result from analyze_novelty().
    filename : str
        Base filename for saving.
    dpi : int
        Resolution.
    show : bool
        Whether to display.
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    novelty_df = result["novelty_df"]
    scores = novelty_df["Composite Novelty"]
    
    ax.hist(scores, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    
    # Add percentile lines
    p90 = scores.quantile(0.90)
    ax.axvline(p90, color='firebrick', linestyle='--', linewidth=2, 
               label=f'Top 10% threshold: {p90:.3f}')
    
    ax.set_xlabel('Composite Novelty Score', fontsize=11)
    ax.set_ylabel('Number of Papers', fontsize=11)
    ax.set_title('Distribution of Novelty Scores', fontsize=12, fontweight='bold')
    ax.legend(frameon=False)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_novelty_components(
    result: dict,
    filename: str = "novelty_components",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot breakdown of novelty components (keyword, subject, reference).
    
    Bars ordered with highest mean at top.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    novelty_df = result["novelty_df"]
    
    components = []
    means = []
    
    if "Keyword Novelty" in novelty_df.columns:
        kw_mean = novelty_df["Keyword Novelty"].mean()
        if kw_mean > 0:
            components.append("Keyword\nNovelty")
            means.append(kw_mean)
    
    if "Subject Bridging" in novelty_df.columns:
        sb_mean = novelty_df["Subject Bridging"].mean()
        if sb_mean > 0:
            components.append("Subject\nBridging")
            means.append(sb_mean)
    
    if "Reference Diversity" in novelty_df.columns:
        rd_mean = novelty_df["Reference Diversity"].mean()
        if rd_mean > 0:
            components.append("Reference\nDiversity")
            means.append(rd_mean)
    
    if components:
        # Sort ascending for horizontal bar (highest at top)
        sorted_pairs = sorted(zip(means, components))
        means = [p[0] for p in sorted_pairs]
        components = [p[1] for p in sorted_pairs]
        
        bars = ax.barh(range(len(components)), means, color='steelblue', 
                      alpha=0.8, height=0.6, edgecolor='white')
        ax.set_yticks(range(len(components)))
        ax.set_yticklabels(components)
        ax.set_xlabel('Mean Score', fontsize=11)
        ax.set_title('Novelty Components (Mean Scores)', fontsize=12, fontweight='bold')
        
        max_val = max(means) if means else 1
        for bar, val in zip(bars, means):
            ax.text(bar.get_width() + max_val * 0.02, 
                   bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=10)
        ax.set_xlim(0, max_val * 1.15)
    else:
        ax.text(0.5, 0.5, 'No novelty component data available', 
               ha='center', va='center', transform=ax.transAxes)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_novelty_trend(
    result: dict,
    filename: str = "novelty_trend",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot novelty trend over time.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    temporal = result.get("temporal_trends")
    
    if temporal is None or len(temporal) == 0:
        ax.text(0.5, 0.5, 'No temporal data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.grid(False)
        plt.tight_layout()
        if filename is not None:
            save_plot(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close()
        return fig
    
    df = temporal[(temporal["Year"] >= 1990) & (temporal["Year"] <= 2030)]
    
    if len(df) > 1:
        ax.plot(df["Year"], df["Mean Novelty"], marker='o', linewidth=2, 
               color='steelblue', markersize=6, label='Mean')
        ax.fill_between(df["Year"], df["Mean Novelty"], alpha=0.2, color='steelblue')
        
        if "Median Novelty" in df.columns:
            ax.plot(df["Year"], df["Median Novelty"], marker='s', linewidth=2, 
                   color='coral', markersize=5, linestyle='--', label='Median')
        
        ax.set_xlabel('Publication Year', fontsize=11)
        ax.set_ylabel('Novelty Score', fontsize=11)
        ax.set_title('Novelty Trend Over Time', fontsize=12, fontweight='bold')
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, 'Insufficient trend data', 
               ha='center', va='center', transform=ax.transAxes)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_keyword_pairs(
    result: dict,
    top_n: int = 15,
    filename: str = "keyword_pairs",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot most common and rare keyword pairs.
    
    Bars ordered with highest at top.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    keyword_novelty = result.get("keyword_novelty")
    
    if keyword_novelty is None:
        ax.text(0.5, 0.5, 'No keyword data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.grid(False)
        plt.tight_layout()
        if filename is not None:
            save_plot(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close()
        return fig
    
    common_pairs = keyword_novelty.get("most_common_pairs", [])[:top_n]
    
    if common_pairs:
        # Sort ascending for horizontal bar (highest at top)
        common_pairs = sorted(common_pairs, key=lambda x: x[1])
        
        labels = [f"{p[0][0][:20]} + {p[0][1][:20]}" for p in common_pairs]
        values = [p[1] for p in common_pairs]
        
        bars = ax.barh(range(len(labels)), values, color='steelblue', 
                      alpha=0.8, height=0.7, edgecolor='white')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Co-occurrence Count', fontsize=11)
        ax.set_title('Most Common Keyword Pairs', fontsize=12, fontweight='bold')
        
        max_val = max(values) if values else 1
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max_val * 0.02, 
                   bar.get_y() + bar.get_height()/2,
                   f'{val}', va='center', fontsize=9)
        ax.set_xlim(0, max_val * 1.12)
    else:
        ax.text(0.5, 0.5, 'No keyword pair data', 
               ha='center', va='center', transform=ax.transAxes)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_subject_distribution(
    result: dict,
    top_n: int = 15,
    filename: str = "subject_distribution",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot subject category distribution.
    
    Bars ordered with highest at top.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    subject_bridging = result.get("subject_bridging")
    
    if subject_bridging is None:
        ax.text(0.5, 0.5, 'No subject data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.grid(False)
        plt.tight_layout()
        if filename is not None:
            save_plot(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close()
        return fig
    
    subject_dist = subject_bridging.get("subject_distribution", {})
    
    if subject_dist:
        # Sort and take top N
        sorted_items = sorted(subject_dist.items(), key=lambda x: x[1])[-top_n:]
        
        labels = [s[0][:35] for s in sorted_items]
        values = [s[1] for s in sorted_items]
        
        bars = ax.barh(range(len(labels)), values, color='steelblue', 
                      alpha=0.8, height=0.7, edgecolor='white')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Number of Papers', fontsize=11)
        ax.set_title('Subject Category Distribution', fontsize=12, fontweight='bold')
        
        max_val = max(values) if values else 1
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max_val * 0.02, 
                   bar.get_y() + bar.get_height()/2,
                   f'{val}', va='center', fontsize=9)
        ax.set_xlim(0, max_val * 1.12)
    else:
        ax.text(0.5, 0.5, 'No subject distribution data', 
               ha='center', va='center', transform=ax.transAxes)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig


def plot_novelty_scatter(
    result: dict,
    x_col: str = "Keyword Novelty",
    y_col: str = "Subject Bridging",
    filename: str = "novelty_scatter",
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 8),
):
    """
    Plot scatter of two novelty components.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    novelty_df = result["novelty_df"]
    
    if x_col not in novelty_df.columns or y_col not in novelty_df.columns:
        ax.text(0.5, 0.5, f'Columns {x_col} or {y_col} not available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.grid(False)
        plt.tight_layout()
        if filename is not None:
            save_plot(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close()
        return fig
    
    x = novelty_df[x_col]
    y = novelty_df[y_col]
    
    # Filter to non-zero values
    mask = (x > 0) | (y > 0)
    x = x[mask]
    y = y[mask]
    
    if len(x) > 0:
        ax.scatter(x, y, alpha=0.5, s=30, c='steelblue', edgecolor='white', linewidth=0.5)
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        ax.set_title(f'{x_col} vs {y_col}', fontsize=12, fontweight='bold')
        
        # Add correlation
        if len(x) > 5:
            from scipy.stats import spearmanr
            corr, pval = spearmanr(x, y)
            ax.text(0.05, 0.95, f'Spearman r = {corr:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Insufficient data for scatter plot', 
               ha='center', va='center', transform=ax.transAxes)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if filename is not None:
        save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    return fig
