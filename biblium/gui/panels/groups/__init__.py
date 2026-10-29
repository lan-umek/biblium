# -*- coding: utf-8 -*-
"""
Group Analysis Panels
=====================
Panels for creating, configuring, and analyzing document groups.

This module provides comprehensive group analysis capabilities including:
- Group Setup: Create groups by column, year periods, clustering, concepts, or random
- Group Counts: Count entities across groups
- Group Statistics: Compute performance statistics by group
- Group Comparison: Compare continuous variables
- Group Intersections: Analyze document overlap between groups
- Group Associations: Analyze entity-group relationships
- Group Visualizations: Overlap plots, distributions, production over time
- Group Classification: ML-based classification of group membership
- Group Logistic Regression: Statistical logistic regression with coefficients
"""

from biblium.gui.panels.groups.group_setup_panel import GroupSetupPanel
from biblium.gui.panels.groups.group_counts_panel import GroupCountsPanel
from biblium.gui.panels.groups.group_stats_panel import GroupStatsPanel
from biblium.gui.panels.groups.group_compare_panel import GroupComparePanel
from biblium.gui.panels.groups.group_intersections_panel import GroupIntersectionsPanel
from biblium.gui.panels.groups.group_associations_panel import GroupAssociationsPanel
from biblium.gui.panels.groups.group_visualizations_panel import GroupVisualizationsPanel
from biblium.gui.panels.groups.group_classification_panel import GroupClassificationPanel
from biblium.gui.panels.groups.group_logistic_panel import GroupLogisticPanel
from biblium.gui.panels.groups.group_diversity_panel import GroupDiversityPanel

__all__ = [
    "GroupSetupPanel",
    "GroupCountsPanel",
    "GroupStatsPanel",
    "GroupComparePanel",
    "GroupIntersectionsPanel",
    "GroupAssociationsPanel",
    "GroupVisualizationsPanel",
    "GroupClassificationPanel",
    "GroupLogisticPanel",
    "GroupDiversityPanel",
]
