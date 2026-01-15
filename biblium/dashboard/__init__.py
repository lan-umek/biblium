# -*- coding: utf-8 -*-
"""
Biblium Interactive Dashboard Module.

Generate standalone HTML dashboards with interactive Bokeh visualizations.

Usage
-----
    from biblium import BiblioAnalysis
    from biblium.dashboard import Dashboard
    
    ba = BiblioAnalysis("data.csv", db="scopus")
    
    # Create dashboard
    dashboard = Dashboard(ba)
    dashboard.create("my_analysis.html")
    
    # Or with configuration
    dashboard.create("analysis.html", title="Research Overview", theme="dark")
"""

from biblium.dashboard.generator import Dashboard, DashboardConfig

__all__ = ["Dashboard", "DashboardConfig"]
