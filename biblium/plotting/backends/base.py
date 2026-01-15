# -*- coding: utf-8 -*-
"""
Abstract Plot Backend Base Class.

Defines the interface that all plotting backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from biblium.plotting.config import PlotConfig


class PlotBackend(ABC):
    """
    Abstract base class for plotting backends.
    
    All plotting backends (Matplotlib, Bokeh, etc.) must implement
    this interface to ensure consistent behavior.
    
    Plot Types:
    - Basic: bar, barh, line, scatter, heatmap, network
    - Distribution: histogram, boxplot, violinplot, density
    - Ranking: treemap, sunburst, waffle
    - Time: area, stacked_area, streamgraph
    - Comparison: grouped_bar, stacked_bar, dumbbell, slope
    - Relationship: bubble, parallel_coordinates
    """
    
    name: str = "base"
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize backend with optional config.
        
        Parameters
        ----------
        config : PlotConfig, optional
            Default configuration for all plots.
        """
        self.config = config or PlotConfig()
    
    # =========================================================================
    # BASIC PLOTS (Phase 1)
    # =========================================================================
    
    @abstractmethod
    def bar(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a vertical bar chart."""
        pass
    
    @abstractmethod
    def barh(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a horizontal bar chart."""
        pass
    
    @abstractmethod
    def line(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a line chart."""
        pass
    
    @abstractmethod
    def scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        size: Optional[Union[str, float]] = None,
        color: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a scatter plot."""
        pass
    
    @abstractmethod
    def heatmap(
        self,
        data: pd.DataFrame,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a heatmap."""
        pass
    
    @abstractmethod
    def network(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a network graph."""
        pass
    
    # =========================================================================
    # DISTRIBUTION PLOTS (Phase 2)
    # =========================================================================
    
    @abstractmethod
    def histogram(
        self,
        data: pd.DataFrame,
        x: str,
        bins: int = 30,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a histogram.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str
            Column for values.
        bins : int
            Number of bins.
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    @abstractmethod
    def boxplot(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: str = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a box plot.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str, optional
            Column for grouping (categorical).
        y : str
            Column for values.
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    @abstractmethod
    def violinplot(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: str = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a violin plot.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str, optional
            Column for grouping.
        y : str
            Column for values.
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    # =========================================================================
    # AREA PLOTS (Phase 2)
    # =========================================================================
    
    @abstractmethod
    def area(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        stacked: bool = False,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create an area chart.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str
            Column for x-axis.
        y : str or list
            Column(s) for y-axis.
        stacked : bool
            Whether to stack areas.
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    # =========================================================================
    # GROUPED/STACKED PLOTS (Phase 2)
    # =========================================================================
    
    @abstractmethod
    def grouped_bar(
        self,
        data: pd.DataFrame,
        x: str,
        y: List[str],
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a grouped bar chart.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str
            Column for categories.
        y : list
            Columns for values (each becomes a group).
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    @abstractmethod
    def stacked_bar(
        self,
        data: pd.DataFrame,
        x: str,
        y: List[str],
        horizontal: bool = False,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a stacked bar chart.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str
            Column for categories.
        y : list
            Columns for values (each becomes a stack segment).
        horizontal : bool
            Create horizontal stacked bars.
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    # =========================================================================
    # COMPARISON PLOTS (Phase 2)
    # =========================================================================
    
    @abstractmethod
    def dumbbell(
        self,
        data: pd.DataFrame,
        y: str,
        x_start: str,
        x_end: str,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a dumbbell (lollipop) chart.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        y : str
            Column for categories (y-axis).
        x_start : str
            Column for start values.
        x_end : str
            Column for end values.
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    @abstractmethod
    def lollipop(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        horizontal: bool = True,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a lollipop chart.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str
            Column for values.
        y : str
            Column for categories.
        horizontal : bool
            Create horizontal lollipops.
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    # =========================================================================
    # BUBBLE CHART (Phase 2)
    # =========================================================================
    
    @abstractmethod
    def bubble(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        size: str,
        color: Optional[str] = None,
        label: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a bubble chart.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        x : str
            Column for x-axis.
        y : str
            Column for y-axis.
        size : str
            Column for bubble size.
        color : str, optional
            Column for color mapping.
        label : str, optional
            Column for bubble labels.
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    # =========================================================================
    # DONUT CHART (Phase 2) - Alternative to pie
    # =========================================================================
    
    @abstractmethod
    def donut(
        self,
        data: pd.DataFrame,
        values: str,
        labels: str,
        hole_size: float = 0.4,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a donut chart.
        
        Note: While we avoid pie charts, donut charts are more readable
        and better at showing part-to-whole relationships.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        values : str
            Column for values.
        labels : str
            Column for labels.
        hole_size : float
            Size of center hole (0-1).
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    # =========================================================================
    # TREEMAP (Phase 2)
    # =========================================================================
    
    @abstractmethod
    def treemap(
        self,
        data: pd.DataFrame,
        values: str,
        labels: str,
        parents: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a treemap.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        values : str
            Column for area size.
        labels : str
            Column for labels.
        parents : str, optional
            Column for hierarchy.
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    # =========================================================================
    # WORDCLOUD (Phase 2)
    # =========================================================================
    
    @abstractmethod
    def wordcloud(
        self,
        data: pd.DataFrame,
        text: str,
        weight: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a word cloud.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with text and optional weights.
        text : str
            Column for text/words.
        weight : str, optional
            Column for word weights/frequencies.
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    # =========================================================================
    # RADAR/SPIDER CHART (Phase 2)
    # =========================================================================
    
    @abstractmethod
    def radar(
        self,
        data: pd.DataFrame,
        categories: List[str],
        values: Union[str, List[str]],
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Create a radar/spider chart.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        categories : list
            Columns for radar axes.
        values : str or list
            Row(s) or column(s) to plot.
        config : PlotConfig, optional
            Plot configuration.
        """
        pass
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    @abstractmethod
    def save(
        self,
        figure: Any,
        filename: str,
        config: Optional[PlotConfig] = None,
    ) -> List[str]:
        """Save figure to file(s)."""
        pass
    
    @abstractmethod
    def show(self, figure: Any) -> None:
        """Display figure."""
        pass
    
    def _merge_config(self, config: Optional[PlotConfig]) -> PlotConfig:
        """Merge provided config with default config."""
        if config is None:
            return self.config
        return self.config.merge(config)
    
    def _get_colors(self, n: int, config: PlotConfig) -> List[str]:
        """Get n colors from config."""
        return config.get_colors(n)
