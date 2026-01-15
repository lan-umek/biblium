# -*- coding: utf-8 -*-
"""
Biblium Plotting Module.

Provides a unified interface for creating plots with different backends
(Matplotlib, Bokeh) and full customization through PlotConfig.

Usage
-----
    from biblium.plotting import Plot, PlotConfig, set_backend
    
    # Set global backend
    set_backend("bokeh")  # or "matplotlib"
    
    # Create fully customized plot
    config = PlotConfig(
        width=800,
        height=600,
        title="My Plot",
        colormap="viridis",
        show_grid=False,
    )
    
    fig = Plot.bar(df, x="category", y="value", config=config)
    Plot.save(fig, "my_plot")
    Plot.show(fig)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type, Union

import pandas as pd

from biblium.plotting.config import (
    PlotConfig,
    DEFAULT_CONFIG,
    MINIMAL_CONFIG,
    PRESENTATION_CONFIG,
    PUBLICATION_CONFIG,
    DARK_CONFIG,
)
from biblium.plotting.backends.base import PlotBackend
from biblium.plotting.backends.matplotlib_backend import MatplotlibBackend
from biblium.plotting.backends.bokeh_backend import BokehBackend
from biblium.plotting.interface import PlotInterface


# =============================================================================
# GLOBAL STATE
# =============================================================================

_current_backend: PlotBackend = MatplotlibBackend()
_backend_classes: Dict[str, Type[PlotBackend]] = {
    "matplotlib": MatplotlibBackend,
    "mpl": MatplotlibBackend,
    "bokeh": BokehBackend,
}


# =============================================================================
# BACKEND MANAGEMENT
# =============================================================================

def set_backend(name: Literal["matplotlib", "mpl", "bokeh"], config: Optional[PlotConfig] = None) -> None:
    """
    Set the global plotting backend.
    
    Parameters
    ----------
    name : str
        Backend name: "matplotlib" (or "mpl") or "bokeh".
    config : PlotConfig, optional
        Default configuration for the backend.
        
    Examples
    --------
    >>> set_backend("bokeh")
    >>> set_backend("matplotlib", config=PlotConfig.presentation())
    """
    global _current_backend
    
    if name not in _backend_classes:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_backend_classes.keys())}")
    
    _current_backend = _backend_classes[name](config)
    print(f"Plotting backend set to: {_current_backend.name}")


def get_backend() -> PlotBackend:
    """Get the current plotting backend."""
    return _current_backend


def get_backend_name() -> str:
    """Get the name of the current backend."""
    return _current_backend.name


def list_backends() -> List[str]:
    """List available backends."""
    return list(_backend_classes.keys())


# =============================================================================
# PLOT CLASS (Static methods for convenience)
# =============================================================================

class Plot:
    """
    Unified plotting interface.
    
    All methods are static and use the global backend.
    For backend-specific operations, use set_backend() first.
    
    Examples
    --------
    >>> from biblium.plotting import Plot, PlotConfig, set_backend
    >>> 
    >>> # Use matplotlib (default)
    >>> fig = Plot.bar(df, x="category", y="value")
    >>> Plot.save(fig, "bar_chart")
    >>> 
    >>> # Switch to bokeh for interactive plots
    >>> set_backend("bokeh")
    >>> fig = Plot.bar(df, x="category", y="value")
    >>> Plot.show(fig)  # Opens in browser
    """
    
    @staticmethod
    def bar(
        data: pd.DataFrame,
        x: str,
        y: str,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a vertical bar chart.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str
            Column for x-axis (categories).
        y : str
            Column for y-axis (values).
        config : PlotConfig, optional
            Plot configuration.
            
        Returns
        -------
        Figure object (backend-specific)
        """
        return _current_backend.bar(data, x, y, config, **kwargs)
    
    @staticmethod
    def barh(
        data: pd.DataFrame,
        x: str,
        y: str,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a horizontal bar chart.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str
            Column for values (bar length).
        y : str
            Column for categories.
        config : PlotConfig, optional
            Plot configuration.
            
        Returns
        -------
        Figure object
        """
        return _current_backend.barh(data, x, y, config, **kwargs)
    
    @staticmethod
    def line(
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a line chart.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str
            Column for x-axis.
        y : str or list
            Column(s) for y-axis.
        config : PlotConfig, optional
            Plot configuration.
            
        Returns
        -------
        Figure object
        """
        return _current_backend.line(data, x, y, config, **kwargs)
    
    @staticmethod
    def scatter(
        data: pd.DataFrame,
        x: str,
        y: str,
        size: Optional[Union[str, float]] = None,
        color: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a scatter plot.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str
            Column for x-axis.
        y : str
            Column for y-axis.
        size : str or float, optional
            Column for size mapping or fixed size.
        color : str, optional
            Column for color mapping.
        config : PlotConfig, optional
            Plot configuration.
            
        Returns
        -------
        Figure object
        """
        return _current_backend.scatter(data, x, y, size, color, config, **kwargs)
    
    @staticmethod
    def heatmap(
        data: pd.DataFrame,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a heatmap.
        
        Parameters
        ----------
        data : pd.DataFrame
            Matrix data (index and columns are labels).
        config : PlotConfig, optional
            Plot configuration.
            
        Returns
        -------
        Figure object
        """
        return _current_backend.heatmap(data, config, **kwargs)
    
    @staticmethod
    def network(
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a network graph.
        
        Parameters
        ----------
        nodes : pd.DataFrame
            Node data with 'id' column.
        edges : pd.DataFrame
            Edge data with 'source', 'target' columns.
        config : PlotConfig, optional
            Plot configuration.
            
        Returns
        -------
        Figure object
        """
        return _current_backend.network(nodes, edges, config, **kwargs)
    
    # =========================================================================
    # PHASE 2: DISTRIBUTION PLOTS
    # =========================================================================
    
    @staticmethod
    def histogram(
        data: pd.DataFrame,
        x: str,
        bins: int = 30,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a histogram."""
        return _current_backend.histogram(data, x, bins, config, **kwargs)
    
    @staticmethod
    def boxplot(
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: str = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a box plot."""
        return _current_backend.boxplot(data, x, y, config, **kwargs)
    
    @staticmethod
    def violinplot(
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: str = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a violin plot."""
        return _current_backend.violinplot(data, x, y, config, **kwargs)
    
    # =========================================================================
    # PHASE 2: AREA PLOTS
    # =========================================================================
    
    @staticmethod
    def area(
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        stacked: bool = False,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create an area chart."""
        return _current_backend.area(data, x, y, stacked, config, **kwargs)
    
    # =========================================================================
    # PHASE 2: GROUPED/STACKED BAR PLOTS
    # =========================================================================
    
    @staticmethod
    def grouped_bar(
        data: pd.DataFrame,
        x: str,
        y: List[str],
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a grouped bar chart."""
        return _current_backend.grouped_bar(data, x, y, config, **kwargs)
    
    @staticmethod
    def stacked_bar(
        data: pd.DataFrame,
        x: str,
        y: List[str],
        horizontal: bool = False,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a stacked bar chart."""
        return _current_backend.stacked_bar(data, x, y, horizontal, config, **kwargs)
    
    # =========================================================================
    # PHASE 2: COMPARISON PLOTS
    # =========================================================================
    
    @staticmethod
    def dumbbell(
        data: pd.DataFrame,
        y: str,
        x_start: str,
        x_end: str,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a dumbbell chart."""
        return _current_backend.dumbbell(data, y, x_start, x_end, config, **kwargs)
    
    @staticmethod
    def lollipop(
        data: pd.DataFrame,
        x: str,
        y: str,
        horizontal: bool = True,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a lollipop chart."""
        return _current_backend.lollipop(data, x, y, horizontal, config, **kwargs)
    
    # =========================================================================
    # PHASE 2: BUBBLE CHART
    # =========================================================================
    
    @staticmethod
    def bubble(
        data: pd.DataFrame,
        x: str,
        y: str,
        size: str,
        color: Optional[str] = None,
        label: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a bubble chart."""
        return _current_backend.bubble(data, x, y, size, color, label, config, **kwargs)
    
    # =========================================================================
    # PHASE 2: DONUT CHART
    # =========================================================================
    
    @staticmethod
    def donut(
        data: pd.DataFrame,
        values: str,
        labels: str,
        hole_size: float = 0.4,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a donut chart."""
        return _current_backend.donut(data, values, labels, hole_size, config, **kwargs)
    
    # =========================================================================
    # PHASE 2: TREEMAP
    # =========================================================================
    
    @staticmethod
    def treemap(
        data: pd.DataFrame,
        values: str,
        labels: str,
        parents: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a treemap."""
        return _current_backend.treemap(data, values, labels, parents, config, **kwargs)
    
    # =========================================================================
    # PHASE 2: WORDCLOUD
    # =========================================================================
    
    @staticmethod
    def wordcloud(
        data: pd.DataFrame,
        text: str,
        weight: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a word cloud."""
        return _current_backend.wordcloud(data, text, weight, config, **kwargs)
    
    # =========================================================================
    # PHASE 2: RADAR CHART
    # =========================================================================
    
    @staticmethod
    def radar(
        data: pd.DataFrame,
        categories: List[str],
        values: Union[str, List[str]],
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a radar/spider chart."""
        return _current_backend.radar(data, categories, values, config, **kwargs)
    
    # =========================================================================
    # SAVE AND SHOW
    # =========================================================================
    
    @staticmethod
    def save(
        figure: Any,
        filename: str,
        config: Optional[PlotConfig] = None,
    ) -> List[str]:
        """
        Save figure to file(s).
        
        Parameters
        ----------
        figure : Any
            Figure object to save.
        filename : str
            Base filename (without extension).
        config : PlotConfig, optional
            Save configuration.
            
        Returns
        -------
        List of saved file paths.
        """
        return _current_backend.save(figure, filename, config)
    
    @staticmethod
    def show(figure: Any) -> None:
        """
        Display figure.
        
        Parameters
        ----------
        figure : Any
            Figure object to display.
        """
        _current_backend.show(figure)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    "PlotConfig",
    "DEFAULT_CONFIG",
    "MINIMAL_CONFIG", 
    "PRESENTATION_CONFIG",
    "PUBLICATION_CONFIG",
    "DARK_CONFIG",
    # Backend management
    "set_backend",
    "get_backend",
    "get_backend_name",
    "list_backends",
    # Plot class
    "Plot",
    # Interface
    "PlotInterface",
    # Backends
    "PlotBackend",
    "MatplotlibBackend",
    "BokehBackend",
]
