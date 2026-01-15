# -*- coding: utf-8 -*-
"""
GUI Panels Package
==================
Analysis and data panels for the workspace.
"""

from biblium.gui.panels.base import BasePanel

# Data panels
from biblium.gui.panels.data import LoadDataPanel, ViewDataPanel, FilterPanel, APIDataPanel

# Analysis panels
from biblium.gui.panels.analysis import CountsPanel, OverviewPanel, TrendsPanel, LawsPanel

# Network panels
from biblium.gui.panels.networks import NetworkPanel

# Mapping panels
from biblium.gui.panels.mapping import (
    ConceptualMapPanel, ThematicMapPanel, TopicsPanel, ClustersPanel
)

# Advanced panels
from biblium.gui.panels.advanced import (
    SleepingBeautyPanel, DisruptionPanel, ResearchFrontsPanel
)

# Report panels
from biblium.gui.panels.reports import ReportBuilderPanel, CustomReportPanel

__all__ = [
    "BasePanel",
    # Data
    "LoadDataPanel",
    "ViewDataPanel", 
    "FilterPanel",
    "APIDataPanel",
    # Analysis
    "CountsPanel",
    "OverviewPanel",
    "TrendsPanel",
    "LawsPanel",
    # Networks
    "NetworkPanel",
    # Mapping
    "ConceptualMapPanel",
    "ThematicMapPanel",
    "TopicsPanel",
    "ClustersPanel",
    # Advanced
    "SleepingBeautyPanel",
    "DisruptionPanel",
    "ResearchFrontsPanel",
    # Reports
    "ReportBuilderPanel",
    "CustomReportPanel",
]
