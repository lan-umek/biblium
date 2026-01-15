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

__all__ = [
    "GroupCountingMixin",
    "GroupStatsMixin",
    "GroupAssociationsMixin",
    "GroupAnalysisMixin",
]
