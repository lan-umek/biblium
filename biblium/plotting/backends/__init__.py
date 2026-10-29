# -*- coding: utf-8 -*-
"""
Plotting Backends Module.

Provides different rendering backends for Biblium plots:
- matplotlib: Traditional static plots
- bokeh: Interactive web-based plots
"""

from biblium.plotting.backends.base import PlotBackend
from biblium.plotting.backends.matplotlib_backend import MatplotlibBackend
from biblium.plotting.backends.bokeh_backend import BokehBackend

__all__ = [
    "PlotBackend",
    "MatplotlibBackend", 
    "BokehBackend",
]
