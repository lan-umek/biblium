# -*- coding: utf-8 -*-
"""
Plot Configuration Module.

Provides a comprehensive PlotConfig dataclass that allows full customization
of all plot parameters. Works with both Matplotlib and Bokeh backends.

Usage
-----
    from biblium.plotting import PlotConfig
    
    # Create custom config
    config = PlotConfig(
        width=800,
        height=600,
        title="My Custom Plot",
        colormap="viridis",
        show_grid=False,
    )
    
    # Use with any plot
    ba.plot_sources_bar(config=config)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULT_COLORS = [
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

DEFAULT_SEQUENTIAL_PALETTE = "YlOrRd"
DEFAULT_DIVERGING_PALETTE = "RdYlBu"


# =============================================================================
# PLOT CONFIG
# =============================================================================

@dataclass
class PlotConfig:
    """
    Comprehensive configuration for all plot types.
    
    This dataclass provides full customization of plot appearance and behavior.
    All parameters have sensible defaults following Biblium's design guidelines
    (no gridlines, consistent colors, no pie charts).
    
    Parameters are organized into categories:
    - Figure: size, margins, background
    - Title: text, font, position
    - Axes: labels, limits, ticks
    - Colors: palette, colormap, transparency
    - Grid: visibility, style
    - Legend: position, font, visibility
    - Annotations: labels, values on bars
    - Export: format, resolution
    - Interactive: tooltips, hover (Bokeh-specific)
    
    Examples
    --------
    >>> config = PlotConfig(width=1000, height=600, title="Publications by Year")
    >>> config.show_grid = False
    >>> config.colormap = "viridis"
    """
    
    # =========================================================================
    # FIGURE SETTINGS
    # =========================================================================
    
    width: int = 800
    """Figure width in pixels."""
    
    height: int = 600
    """Figure height in pixels."""
    
    dpi: int = 100
    """Resolution for raster export (matplotlib)."""
    
    background_color: str = "#ffffff"
    """Figure background color."""
    
    plot_background_color: str = "#ffffff"
    """Plot area background color."""
    
    margin_left: float = 0.1
    """Left margin as fraction of figure width."""
    
    margin_right: float = 0.1
    """Right margin as fraction of figure width."""
    
    margin_top: float = 0.1
    """Top margin as fraction of figure height."""
    
    margin_bottom: float = 0.1
    """Bottom margin as fraction of figure height."""
    
    # =========================================================================
    # TITLE SETTINGS
    # =========================================================================
    
    title: Optional[str] = None
    """Plot title text."""
    
    title_font_size: int = 14
    """Title font size in points."""
    
    title_font_weight: Literal["normal", "bold", "light"] = "bold"
    """Title font weight."""
    
    title_font_family: str = "sans-serif"
    """Title font family."""
    
    title_color: str = "#333333"
    """Title text color."""
    
    title_position: Literal["center", "left", "right"] = "center"
    """Title horizontal alignment."""
    
    subtitle: Optional[str] = None
    """Subtitle text (optional)."""
    
    subtitle_font_size: int = 11
    """Subtitle font size."""
    
    # =========================================================================
    # AXES SETTINGS
    # =========================================================================
    
    xlabel: Optional[str] = None
    """X-axis label."""
    
    ylabel: Optional[str] = None
    """Y-axis label."""
    
    xlabel_font_size: int = 12
    """X-axis label font size."""
    
    ylabel_font_size: int = 12
    """Y-axis label font size."""
    
    label_color: str = "#333333"
    """Axis label color."""
    
    xlim: Optional[Tuple[float, float]] = None
    """X-axis limits (min, max)."""
    
    ylim: Optional[Tuple[float, float]] = None
    """Y-axis limits (min, max)."""
    
    xscale: Literal["linear", "log", "symlog"] = "linear"
    """X-axis scale."""
    
    yscale: Literal["linear", "log", "symlog"] = "linear"
    """Y-axis scale."""
    
    invert_xaxis: bool = False
    """Invert X-axis direction."""
    
    invert_yaxis: bool = False
    """Invert Y-axis direction."""
    
    # =========================================================================
    # TICK SETTINGS
    # =========================================================================
    
    xtick_font_size: int = 10
    """X-axis tick label font size."""
    
    ytick_font_size: int = 10
    """Y-axis tick label font size."""
    
    tick_color: str = "#666666"
    """Tick label color."""
    
    xtick_rotation: float = 0
    """X-axis tick label rotation in degrees."""
    
    ytick_rotation: float = 0
    """Y-axis tick label rotation in degrees."""
    
    show_xticks: bool = True
    """Show X-axis ticks."""
    
    show_yticks: bool = True
    """Show Y-axis ticks."""
    
    xtick_format: Optional[str] = None
    """X-axis tick format string (e.g., '{:.1f}', '{:,.0f}')."""
    
    ytick_format: Optional[str] = None
    """Y-axis tick format string."""
    
    max_xticks: Optional[int] = None
    """Maximum number of X-axis ticks."""
    
    max_yticks: Optional[int] = None
    """Maximum number of Y-axis ticks."""
    
    # =========================================================================
    # COLOR SETTINGS
    # =========================================================================
    
    colors: Optional[List[str]] = None
    """List of colors for categorical data. Uses DEFAULT_COLORS if None."""
    
    colormap: str = "YlOrRd"
    """Colormap for continuous data (matplotlib/bokeh palette name)."""
    
    color: Optional[str] = None
    """Single color for uniform coloring."""
    
    alpha: float = 1.0
    """Global transparency (0-1)."""
    
    edge_color: str = "none"
    """Edge/border color for shapes."""
    
    edge_width: float = 0
    """Edge/border width."""
    
    # =========================================================================
    # GRID SETTINGS (Default: OFF per Biblium guidelines)
    # =========================================================================
    
    show_grid: bool = False
    """Show grid lines (default False per Biblium guidelines)."""
    
    grid_color: str = "#e0e0e0"
    """Grid line color."""
    
    grid_alpha: float = 0.5
    """Grid line transparency."""
    
    grid_style: Literal["solid", "dashed", "dotted"] = "dashed"
    """Grid line style."""
    
    grid_width: float = 0.5
    """Grid line width."""
    
    grid_axis: Literal["both", "x", "y"] = "both"
    """Which axes to show grid on."""
    
    # =========================================================================
    # SPINE SETTINGS
    # =========================================================================
    
    show_top_spine: bool = False
    """Show top spine."""
    
    show_right_spine: bool = False
    """Show right spine."""
    
    show_bottom_spine: bool = True
    """Show bottom spine."""
    
    show_left_spine: bool = True
    """Show left spine."""
    
    spine_color: str = "#333333"
    """Spine color."""
    
    spine_width: float = 1.0
    """Spine width."""
    
    # =========================================================================
    # LEGEND SETTINGS
    # =========================================================================
    
    show_legend: bool = True
    """Show legend."""
    
    legend_position: Literal[
        "best", "upper right", "upper left", "lower left", "lower right",
        "right", "center left", "center right", "lower center", "upper center",
        "center", "outside right", "outside top"
    ] = "best"
    """Legend position."""
    
    legend_font_size: int = 10
    """Legend font size."""
    
    legend_title: Optional[str] = None
    """Legend title."""
    
    legend_title_font_size: int = 11
    """Legend title font size."""
    
    legend_frameon: bool = True
    """Show legend frame/border."""
    
    legend_ncol: int = 1
    """Number of legend columns."""
    
    # =========================================================================
    # BAR CHART SETTINGS
    # =========================================================================
    
    bar_width: float = 0.8
    """Bar width (0-1 for categorical, absolute for numerical)."""
    
    bar_orientation: Literal["vertical", "horizontal"] = "vertical"
    """Bar orientation."""
    
    show_values: bool = False
    """Show values on bars."""
    
    value_format: str = "{:.0f}"
    """Format string for bar values."""
    
    value_font_size: int = 9
    """Font size for bar values."""
    
    value_position: Literal["inside", "outside", "center"] = "outside"
    """Position of value labels."""
    
    # =========================================================================
    # LINE CHART SETTINGS
    # =========================================================================
    
    line_width: float = 2.0
    """Line width."""
    
    line_style: Literal["solid", "dashed", "dotted", "dashdot"] = "solid"
    """Line style."""
    
    show_markers: bool = False
    """Show markers on line points."""
    
    marker_size: float = 6.0
    """Marker size."""
    
    marker_style: str = "o"
    """Marker style (matplotlib marker codes)."""
    
    fill_area: bool = False
    """Fill area under line."""
    
    fill_alpha: float = 0.3
    """Fill area transparency."""
    
    # =========================================================================
    # SCATTER PLOT SETTINGS
    # =========================================================================
    
    scatter_size: Union[float, str] = 50
    """Scatter point size (or column name for size mapping)."""
    
    scatter_size_range: Tuple[float, float] = (20, 200)
    """Min/max size when mapping to column."""
    
    scatter_color_by: Optional[str] = None
    """Column name for color mapping."""
    
    # =========================================================================
    # HEATMAP SETTINGS
    # =========================================================================
    
    annotate: bool = True
    """Show values in heatmap cells."""
    
    annotation_format: str = "{:.1f}"
    """Format for heatmap annotations."""
    
    annotation_font_size: int = 8
    """Heatmap annotation font size."""
    
    colorbar: bool = True
    """Show colorbar for heatmap."""
    
    colorbar_label: Optional[str] = None
    """Colorbar label."""
    
    # =========================================================================
    # NETWORK SETTINGS
    # =========================================================================
    
    node_size: float = 300
    """Network node size."""
    
    node_size_by: Optional[str] = None
    """Attribute for node size mapping."""
    
    node_color_by: Optional[str] = None
    """Attribute for node color mapping."""
    
    edge_width_by: Optional[str] = "weight"
    """Attribute for edge width mapping."""
    
    edge_alpha: float = 0.5
    """Edge transparency."""
    
    show_labels: bool = True
    """Show node labels."""
    
    label_font_size: int = 8
    """Node label font size."""
    
    layout: Literal["spring", "circular", "kamada_kawai", "random"] = "spring"
    """Network layout algorithm."""
    
    # =========================================================================
    # INTERACTIVE SETTINGS (Bokeh-specific)
    # =========================================================================
    
    interactive: bool = True
    """Enable interactive features (Bokeh)."""
    
    show_tooltips: bool = True
    """Show tooltips on hover."""
    
    tooltip_fields: Optional[List[str]] = None
    """Fields to show in tooltips."""
    
    enable_zoom: bool = True
    """Enable zoom tool."""
    
    enable_pan: bool = True
    """Enable pan tool."""
    
    enable_save: bool = True
    """Enable save tool."""
    
    enable_reset: bool = True
    """Enable reset tool."""
    
    # =========================================================================
    # EXPORT SETTINGS
    # =========================================================================
    
    save_format: List[str] = field(default_factory=lambda: ["png", "svg", "pdf"])
    """Formats to save (matplotlib)."""
    
    export_dpi: int = 600
    """DPI for export."""
    
    tight_layout: bool = True
    """Use tight layout for export."""
    
    transparent: bool = False
    """Transparent background for export."""
    
    # =========================================================================
    # METHODS
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def get_colors(self, n: int) -> List[str]:
        """Get n colors from the palette."""
        if self.colors:
            colors = self.colors
        else:
            colors = DEFAULT_COLORS
        
        # Cycle colors if needed
        return [colors[i % len(colors)] for i in range(n)]
    
    def update(self, **kwargs) -> "PlotConfig":
        """Return a new config with updated values."""
        d = self.to_dict()
        d.update(kwargs)
        return PlotConfig(**d)
    
    def merge(self, other: "PlotConfig") -> "PlotConfig":
        """Merge another config, other takes precedence for non-None values."""
        d = self.to_dict()
        for k, v in other.to_dict().items():
            if v is not None:
                d[k] = v
        return PlotConfig(**d)
    
    @classmethod
    def minimal(cls) -> "PlotConfig":
        """Create a minimal config with reduced decoration."""
        return cls(
            show_grid=False,
            show_top_spine=False,
            show_right_spine=False,
            show_legend=False,
        )
    
    @classmethod
    def presentation(cls) -> "PlotConfig":
        """Create a config optimized for presentations."""
        return cls(
            width=1200,
            height=800,
            title_font_size=20,
            xlabel_font_size=16,
            ylabel_font_size=16,
            xtick_font_size=14,
            ytick_font_size=14,
            legend_font_size=14,
            line_width=3.0,
        )
    
    @classmethod
    def publication(cls) -> "PlotConfig":
        """Create a config optimized for publication."""
        return cls(
            width=600,
            height=450,
            dpi=300,
            export_dpi=600,
            title_font_size=11,
            xlabel_font_size=10,
            ylabel_font_size=10,
            xtick_font_size=9,
            ytick_font_size=9,
            legend_font_size=9,
            line_width=1.5,
        )
    
    @classmethod
    def dark_mode(cls) -> "PlotConfig":
        """Create a dark mode config."""
        return cls(
            background_color="#1a1a2e",
            plot_background_color="#16213e",
            title_color="#ffffff",
            label_color="#ffffff",
            tick_color="#cccccc",
            spine_color="#444444",
            grid_color="#333333",
        )


# =============================================================================
# PRESET CONFIGS
# =============================================================================

# Default config following Biblium guidelines
DEFAULT_CONFIG = PlotConfig()

# Minimal config
MINIMAL_CONFIG = PlotConfig.minimal()

# Presentation config
PRESENTATION_CONFIG = PlotConfig.presentation()

# Publication config
PUBLICATION_CONFIG = PlotConfig.publication()

# Dark mode config
DARK_CONFIG = PlotConfig.dark_mode()
