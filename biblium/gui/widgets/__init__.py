# -*- coding: utf-8 -*-
"""
GUI Widgets Package
===================
Custom themed widgets for the Biblium GUI.
"""

from biblium.gui.widgets.buttons import (
    ThemedButton,
    IconButton,
    ActionButton,
    ToggleButton,
)
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.forms import (
    LabeledEntry,
    LabeledCombobox,
    LabeledCheckbox,
    LabeledSpinbox,
    LabeledTextArea,
    FormSection,
    RadioGroup,
)
from biblium.gui.widgets.tables import DataTable, SortableTable, add_table_context_menu
from biblium.gui.widgets.plots import (
    PlotFrame, 
    PlotToolbar, 
    PlotNotebook, 
    ScaledImageFrame, 
    ResizablePlotCanvas,
    add_plot_context_menu,
    make_canvas_resizable,
)
from biblium.gui.widgets.progress import ProgressBar, LoadingSpinner, ProgressDialog
from biblium.gui.widgets.tooltips import ToolTip, create_tooltip
from biblium.gui.widgets.dialogs import (
    BaseDialog,
    MessageDialog,
    ConfirmDialog,
    InputDialog,
    SelectDialog,
)

__all__ = [
    # Buttons
    "ThemedButton",
    "IconButton", 
    "ActionButton",
    "ToggleButton",
    # Cards
    "Card",
    "CollapsibleCard",
    "StatsCard",
    "CardGrid",
    # Forms
    "LabeledEntry",
    "LabeledCombobox",
    "LabeledCheckbox",
    "LabeledSpinbox",
    "LabeledTextArea",
    "FormSection",
    "RadioGroup",
    # Tables
    "DataTable",
    "SortableTable",
    "add_table_context_menu",
    # Plots
    "PlotFrame",
    "PlotToolbar",
    "PlotNotebook",
    "ScaledImageFrame",
    "ResizablePlotCanvas",
    "add_plot_context_menu",
    "make_canvas_resizable",
    # Progress
    "ProgressBar",
    "LoadingSpinner",
    "ProgressDialog",
    # Tooltips
    "ToolTip",
    "create_tooltip",
    # Dialogs
    "BaseDialog",
    "MessageDialog",
    "ConfirmDialog",
    "InputDialog",
    "SelectDialog",
]
