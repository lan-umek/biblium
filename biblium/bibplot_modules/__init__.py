# -*- coding: utf-8 -*-
"""
Bibplot Modules - Modular plotting utilities for bibliometric analysis.

This package organizes plotting functions into focused modules:

- basic: Bar charts, distributions, scatter plots
- time_series: Production over time, trends
- laws: Lotka, Bradford, Zipf
- geographic: Country maps, collaboration maps
- network: Co-occurrence, citation networks
- thematic: Word clouds, topic maps, dendrograms
- race_bar: Animated race bar charts
- advanced: Streamgraphs, bump charts, animated choropleths, interactive networks

Usage
-----
These modules provide mixin classes that are combined in BiblioPlot:

    from biblium.bibplot_modules.basic import BasicPlotsMixin
    from biblium.bibplot_modules.network import NetworkPlotsMixin
    from biblium.bibplot_modules.race_bar import RaceBarMixin
    from biblium.bibplot_modules.advanced import AdvancedVisualizationsMixin
    
Or import individual functions if needed.
"""

from biblium.bibplot_modules.basic import BasicPlotsMixin
from biblium.bibplot_modules.time_series import TimeSeriesPlotsMixin
from biblium.bibplot_modules.laws import LawsPlotsMixin
from biblium.bibplot_modules.geographic import GeographicPlotsMixin
from biblium.bibplot_modules.network import NetworkPlotsMixin
from biblium.bibplot_modules.thematic import ThematicPlotsMixin
from biblium.bibplot_modules.race_bar import RaceBarMixin, create_race_bar_from_dataframe
from biblium.bibplot_modules.advanced import AdvancedVisualizationsMixin


__all__ = [
    "BasicPlotsMixin",
    "TimeSeriesPlotsMixin",
    "LawsPlotsMixin",
    "GeographicPlotsMixin",
    "NetworkPlotsMixin",
    "ThematicPlotsMixin",
    "RaceBarMixin",
    "create_race_bar_from_dataframe",
    "AdvancedVisualizationsMixin",
]
