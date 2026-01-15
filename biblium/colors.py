# -*- coding: utf-8 -*-
"""
Unified Color Scheme for Biblium

Provides consistent colors across all visualizations.

@author: Lan.Umek
"""

from typing import Dict, List, Optional, Union
import numpy as np

# =============================================================================
# MAIN COLOR PALETTES
# =============================================================================

# Primary palette - for most visualizations
PRIMARY_PALETTE = [
    "#2E86AB",  # Steel blue
    "#A23B72",  # Raspberry
    "#F18F01",  # Orange
    "#C73E1D",  # Vermillion
    "#3B1F2B",  # Dark purple
    "#95C623",  # Lime green
    "#5C4D7D",  # Purple
    "#E84855",  # Red
    "#F9DC5C",  # Yellow
    "#3185FC",  # Bright blue
]

# Sequential palette - for continuous data
SEQUENTIAL_PALETTE = "YlOrRd"  # Yellow-Orange-Red

# Diverging palette - for data with meaningful center
DIVERGING_PALETTE = "RdYlBu"  # Red-Yellow-Blue

# Categorical palette - for distinct categories
CATEGORICAL_PALETTE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

# =============================================================================
# SPECIALIZED PALETTES
# =============================================================================

# SDG Colors (official UN colors)
SDG_COLORS = {
    1: "#E5243B",   # No Poverty - Red
    2: "#DDA63A",   # Zero Hunger - Mustard
    3: "#4C9F38",   # Good Health - Green
    4: "#C5192D",   # Quality Education - Dark Red
    5: "#FF3A21",   # Gender Equality - Red Orange
    6: "#26BDE2",   # Clean Water - Light Blue
    7: "#FCC30B",   # Affordable Energy - Yellow
    8: "#A21942",   # Decent Work - Burgundy
    9: "#FD6925",   # Industry Innovation - Orange
    10: "#DD1367",  # Reduced Inequalities - Magenta
    11: "#FD9D24",  # Sustainable Cities - Orange
    12: "#BF8B2E",  # Responsible Consumption - Brown
    13: "#3F7E44",  # Climate Action - Dark Green
    14: "#0A97D9",  # Life Below Water - Blue
    15: "#56C02B",  # Life on Land - Bright Green
    16: "#00689D",  # Peace Justice - Dark Blue
    17: "#19486A",  # Partnerships - Navy
}

# SDG Perspective Colors
SDG_PERSPECTIVE_COLORS = {
    "Life": "#4C9F38",           # Green
    "Social Development": "#E5243B",  # Red
    "Economic Growth": "#FCC30B",     # Yellow
    "Peace": "#00689D",          # Dark Blue
    "Partnership": "#19486A",    # Navy
    "Planet": "#0A97D9",         # Blue
}

# Document type colors
DOCTYPE_COLORS = {
    "Article": "#2E86AB",
    "Review": "#A23B72",
    "Conference Paper": "#F18F01",
    "Book Chapter": "#C73E1D",
    "Book": "#3B1F2B",
    "Editorial": "#95C623",
    "Letter": "#5C4D7D",
    "Note": "#7f7f7f",
    "Other": "#bcbd22",
}

# Open Access colors
OA_COLORS = {
    "Gold": "#FFD700",
    "Green": "#228B22",
    "Hybrid": "#FFA500",
    "Bronze": "#CD7F32",
    "Closed": "#808080",
    "Unknown": "#D3D3D3",
}

# Methodology paradigm colors
PARADIGM_COLORS = {
    "quantitative": "#1f77b4",   # Blue
    "qualitative": "#2ca02c",    # Green
    "mixed_methods": "#ff7f0e",  # Orange
    "unknown": "#7f7f7f",        # Gray
}

# Citation quartile colors
QUARTILE_COLORS = {
    "Q1": "#2E86AB",  # Top 25%
    "Q2": "#95C623",  # 25-50%
    "Q3": "#F18F01",  # 50-75%
    "Q4": "#C73E1D",  # Bottom 25%
}

# Collaboration type colors
COLLAB_COLORS = {
    "Single Author": "#7f7f7f",
    "Institutional": "#2E86AB",
    "National": "#95C623",
    "International": "#A23B72",
}

# =============================================================================
# COLOR UTILITY FUNCTIONS
# =============================================================================

def get_color(index: int, palette: str = "primary") -> str:
    """
    Get a color by index from the specified palette.
    
    Parameters
    ----------
    index : int
        Color index (wraps around if exceeds palette length).
    palette : str
        Palette name: "primary", "categorical".
    
    Returns
    -------
    str
        Hex color code.
    """
    if palette == "primary":
        colors = PRIMARY_PALETTE
    elif palette == "categorical":
        colors = CATEGORICAL_PALETTE
    else:
        colors = PRIMARY_PALETTE
    
    return colors[index % len(colors)]


def get_colors(n: int, palette: str = "primary") -> List[str]:
    """
    Get n colors from the specified palette.
    
    Parameters
    ----------
    n : int
        Number of colors needed.
    palette : str
        Palette name.
    
    Returns
    -------
    list
        List of hex color codes.
    """
    if palette == "primary":
        colors = PRIMARY_PALETTE
    elif palette == "categorical":
        colors = CATEGORICAL_PALETTE
    else:
        colors = PRIMARY_PALETTE
    
    if n <= len(colors):
        return colors[:n]
    else:
        # Repeat colors if needed
        return [colors[i % len(colors)] for i in range(n)]


def get_sdg_color(sdg: int) -> str:
    """Get color for an SDG number."""
    return SDG_COLORS.get(sdg, "#888888")


def get_sdg_colors(sdgs: List[int]) -> List[str]:
    """Get colors for multiple SDGs."""
    return [get_sdg_color(s) for s in sdgs]


def get_perspective_color(perspective: str) -> str:
    """Get color for an SDG perspective."""
    return SDG_PERSPECTIVE_COLORS.get(perspective, "#888888")


def get_sequential_cmap(name: str = None) -> str:
    """Get sequential colormap name."""
    return name if name else SEQUENTIAL_PALETTE


def get_diverging_cmap(name: str = None) -> str:
    """Get diverging colormap name."""
    return name if name else DIVERGING_PALETTE


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB tuple to hex color."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def lighten_color(hex_color: str, factor: float = 0.3) -> str:
    """
    Lighten a color by a factor.
    
    Parameters
    ----------
    hex_color : str
        Hex color code.
    factor : float
        Lightening factor (0-1).
    
    Returns
    -------
    str
        Lightened hex color.
    """
    rgb = hex_to_rgb(hex_color)
    new_rgb = tuple(int(c + (255 - c) * factor) for c in rgb)
    return rgb_to_hex(new_rgb)


def darken_color(hex_color: str, factor: float = 0.3) -> str:
    """
    Darken a color by a factor.
    
    Parameters
    ----------
    hex_color : str
        Hex color code.
    factor : float
        Darkening factor (0-1).
    
    Returns
    -------
    str
        Darkened hex color.
    """
    rgb = hex_to_rgb(hex_color)
    new_rgb = tuple(int(c * (1 - factor)) for c in rgb)
    return rgb_to_hex(new_rgb)


def get_gradient(color1: str, color2: str, n: int) -> List[str]:
    """
    Create a gradient between two colors.
    
    Parameters
    ----------
    color1 : str
        Start color (hex).
    color2 : str
        End color (hex).
    n : int
        Number of colors in gradient.
    
    Returns
    -------
    list
        List of hex color codes.
    """
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    
    gradient = []
    for i in range(n):
        ratio = i / (n - 1) if n > 1 else 0
        rgb = tuple(int(rgb1[j] + (rgb2[j] - rgb1[j]) * ratio) for j in range(3))
        gradient.append(rgb_to_hex(rgb))
    
    return gradient


# =============================================================================
# MATPLOTLIB STYLE CONFIGURATION
# =============================================================================

def get_plot_style() -> Dict:
    """
    Get matplotlib rcParams for consistent styling.
    
    Returns
    -------
    dict
        Dictionary of rcParams settings.
    """
    return {
        # Figure
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "figure.facecolor": "white",
        
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": f"cycler('color', {PRIMARY_PALETTE})",
        
        # Grid (off by default)
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        
        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        
        # Legend
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        
        # Font
        "font.family": "sans-serif",
        "font.size": 10,
        
        # Savefig
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
    }


def apply_plot_style():
    """Apply the unified plot style to matplotlib."""
    import matplotlib.pyplot as plt
    
    style = get_plot_style()
    for key, value in style.items():
        try:
            plt.rcParams[key] = value
        except (KeyError, ValueError):
            pass  # Skip invalid params


def reset_plot_style():
    """Reset matplotlib to default style."""
    import matplotlib.pyplot as plt
    plt.rcdefaults()


# =============================================================================
# COLORBLIND-FRIENDLY PALETTES
# =============================================================================

# Wong palette (colorblind-safe)
COLORBLIND_PALETTE = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky blue
    "#009E73",  # Bluish green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish purple
]

# Tableau colorblind palette
TABLEAU_COLORBLIND = [
    "#006BA4",
    "#FF800E",
    "#ABABAB",
    "#595959",
    "#5F9ED1",
    "#C85200",
    "#898989",
    "#A2C8EC",
    "#FFBC79",
    "#CFCFCF",
]


def get_colorblind_colors(n: int) -> List[str]:
    """Get colorblind-friendly colors."""
    colors = COLORBLIND_PALETTE
    if n <= len(colors):
        return colors[:n]
    return [colors[i % len(colors)] for i in range(n)]


# =============================================================================
# QUICK ACCESS - DEFAULT SINGLE COLORS
# =============================================================================

# Default colors for common use cases
DEFAULT_BAR_COLOR = "#2E86AB"
DEFAULT_LINE_COLOR = "#2E86AB"
DEFAULT_HIGHLIGHT_COLOR = "#A23B72"
DEFAULT_SECONDARY_COLOR = "#95C623"
DEFAULT_NEUTRAL_COLOR = "#7f7f7f"
DEFAULT_POSITIVE_COLOR = "#2ca02c"
DEFAULT_NEGATIVE_COLOR = "#d62728"
DEFAULT_WARNING_COLOR = "#F18F01"
