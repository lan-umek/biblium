# -*- coding: utf-8 -*-
"""
BiblioGroup mixin modules.

This package contains mixin classes that provide functionality for BiblioGroup:

- counting: Group counting methods (group_count_*)
- stats: Group statistics methods (get_group_*_stats)
- associations: Group association methods (associate_*)
- analysis: Analysis and comparison methods
"""

from biblium.bibgroup_modules.counting import GroupCountingMixin
from biblium.bibgroup_modules.stats import GroupStatsMixin
from biblium.bibgroup_modules.associations import GroupAssociationsMixin
from biblium.bibgroup_modules.analysis import GroupAnalysisMixin
from biblium.bibgroup_modules.year_trend import GroupYearTrendMixin
from biblium.bibgroup_modules.content_analysis import GroupContentAnalysisMixin
from biblium.bibgroup_modules.field_dynamics import GroupFieldDynamicsMixin
from biblium.bibgroup_modules.pairs import GroupPairsMixin

__all__ = [
    "GroupCountingMixin",
    "GroupStatsMixin",
    "GroupAssociationsMixin",
    "GroupAnalysisMixin",
    "GroupYearTrendMixin",
    "GroupContentAnalysisMixin",
    "GroupFieldDynamicsMixin",
    "GroupPairsMixin",
]
