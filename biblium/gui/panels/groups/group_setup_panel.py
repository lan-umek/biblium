# -*- coding: utf-8 -*-
"""
Group Setup Panel
=================
Panel for creating and configuring BiblioGroup instances.

Supports multiple grouping methods:
1. By Column - categorical or multi-item columns
2. By Year Periods - custom cutpoints or automatic binning
3. By Clustering - K-means or hierarchical clustering
4. By Concept DataFrame - SDG, topics, or custom concepts
5. By Dictionary/Regex - custom pattern matching
6. Random Groups - overlapping random assignment for testing
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
from typing import Dict, List, Optional, Any, Tuple

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import (
    LabeledCombobox, LabeledCheckbox, LabeledSpinbox, LabeledEntry
)
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.tooltips import ToolTip

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from biblium import BiblioGroupAnalysis
    from biblium.bibgroup import BiblioGroup
    from biblium import utilsbib
    HAS_BIBLIUM_GROUP = True
except ImportError:
    HAS_BIBLIUM_GROUP = False
    BiblioGroupAnalysis = None
    BiblioGroup = None


class GroupSetupPanel(BasePanel):
    """
    Panel for creating and configuring document groups.
    
    This panel allows users to create BiblioGroup instances using various
    grouping strategies, which can then be used for comparative analysis
    across groups.
    """
    
    title = "Group Setup"
    icon = "⚙️"
    description = "Create document groups for comparative analysis"
    requires_data = True
    
    # Grouping method options
    GROUPING_METHODS = [
        ("column", "By Column", "Group by values in a categorical column"),
        ("year_periods", "By Year Periods", "Group documents into time periods"),
        ("multiitem", "By Multi-Item Column", "Group by items in a list column (e.g., keywords)"),
        ("clustering", "By Clustering", "Group using K-means or hierarchical clustering"),
        ("concept", "By Concept DataFrame", "Group by matching concepts/terms in text"),
        ("regex", "By Dictionary/Regex", "Group by custom regex patterns"),
        ("random", "Random Groups", "Create random overlapping groups (for testing)"),
    ]
    
    # Predefined color palettes
    COLOR_PALETTES = {
        "default": [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ],
        "pastel": [
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        ],
        "bold": [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
            "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
        ],
        "earth": [
            "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
            "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
        ],
    }
    
    def _create_options(self):
        """Create the options panel with all grouping configuration options."""
        self._add_title()
        
        if not self.bib:
            self._show_no_data_message()
            return
        
        # Use self.options_content directly (already scrollable from BasePanel)
        # No need for custom scrollable frame
        self.scrollable_frame = self.options_content
        
        # Method Selection Card
        self._create_method_selection_card()
        
        # Dynamic Options Frame (changes based on method)
        self._create_dynamic_options_frame()
        
        # Advanced Options Card
        self._create_advanced_options_card()
        
        # Color Configuration Card
        self._create_color_config_card()
        
        # Action Buttons
        self._create_action_buttons()
        
        # Initialize with first method
        self._on_method_changed()
    
    # Note: Scrolling is handled by BasePanel's options_content
    # The following methods are kept for backwards compatibility but not used
    
    def _bind_mousewheel(self):
        pass  # Handled by BasePanel
    
    def _unbind_mousewheel(self):
        pass  # Handled by BasePanel
    
    def _on_mousewheel(self, event):
        pass  # Handled by BasePanel
    
    def _create_method_selection_card(self):
        """Create the grouping method selection card."""
        method_card = Card(
            self.scrollable_frame,
            title="📋 Grouping Method",
            theme=self.theme_name
        )
        method_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Method selection variable
        self.method_var = tk.StringVar(value="column")
        
        # Create radio buttons for each method
        for method_id, method_name, method_desc in self.GROUPING_METHODS:
            frame = tk.Frame(method_card.content, bg=self.theme["bg_card"])
            frame.pack(fill=tk.X, pady=2)
            
            rb = tk.Radiobutton(
                frame,
                text=method_name,
                variable=self.method_var,
                value=method_id,
                command=self._on_method_changed,
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
                activebackground=self.theme["bg_card"],
                activeforeground=self.theme["accent_primary"],
                font=FONTS.get_font("body"),
            )
            rb.pack(side=tk.LEFT)
            
            # Add tooltip with description
            ToolTip(rb, method_desc)
    
    def _create_dynamic_options_frame(self):
        """Create the frame for method-specific options."""
        self.dynamic_card = Card(
            self.scrollable_frame,
            title="⚙️ Method Options",
            theme=self.theme_name
        )
        self.dynamic_card.pack(fill=tk.X, padx=8, pady=8)
        
        # This frame will be repopulated when method changes
        self.dynamic_frame = tk.Frame(
            self.dynamic_card.content,
            bg=self.theme["bg_card"]
        )
        self.dynamic_frame.pack(fill=tk.X)
    
    def _create_advanced_options_card(self):
        """Create advanced options card."""
        adv_card = CollapsibleCard(
            self.scrollable_frame,
            title="🔧 Advanced Options",
            theme=self.theme_name,
            collapsed=True
        )
        adv_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Merge data with group
        self.merge_data_var = tk.BooleanVar(value=True)
        self.merge_check = LabeledCheckbox(
            adv_card.content,
            label="Merge group columns with main DataFrame",
            variable=self.merge_data_var,
            theme=self.theme_name,
        )
        self.merge_check.pack(fill=tk.X, pady=2)
        ToolTip(self.merge_check, "Add group membership columns to the dataset")
        
        # Results folder
        self.results_folder = LabeledEntry(
            adv_card.content,
            label="Results Folder:",
            default="results-groups",
            theme=self.theme_name,
            label_width=14
        )
        self.results_folder.pack(fill=tk.X, pady=4)
        
        # Preprocessing level
        self.preprocess_level = LabeledSpinbox(
            adv_card.content,
            label="Preprocess Level:",
            from_=0, to=3, default=0,
            theme=self.theme_name,
            label_width=14
        )
        self.preprocess_level.pack(fill=tk.X, pady=4)
        ToolTip(self.preprocess_level, "0=none, 1=basic, 2=moderate, 3=aggressive")
        
        # Separator
        self.separator_entry = LabeledEntry(
            adv_card.content,
            label="List Separator:",
            default="; ",
            theme=self.theme_name,
            label_width=14
        )
        self.separator_entry.pack(fill=tk.X, pady=4)
    
    def _create_color_config_card(self):
        """Create color configuration card."""
        color_card = CollapsibleCard(
            self.scrollable_frame,
            title="🎨 Group Colors",
            theme=self.theme_name,
            collapsed=True
        )
        color_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Palette selection
        self.palette_var = tk.StringVar(value="default")
        palette_frame = tk.Frame(color_card.content, bg=self.theme["bg_card"])
        palette_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            palette_frame,
            text="Color Palette:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        palette_combo = ttk.Combobox(
            palette_frame,
            textvariable=self.palette_var,
            values=list(self.COLOR_PALETTES.keys()),
            state="readonly",
            width=15
        )
        palette_combo.pack(side=tk.LEFT)
        palette_combo.bind("<<ComboboxSelected>>", self._on_palette_changed)
        
        # Color preview frame (palette preview)
        self.color_preview_frame = tk.Frame(color_card.content, bg=self.theme["bg_card"])
        self.color_preview_frame.pack(fill=tk.X, pady=8)
        self._update_color_preview()
        
        # Separator
        ttk.Separator(color_card.content, orient="horizontal").pack(fill=tk.X, pady=8)
        
        # Group color editor frame (appears after preview)
        self.group_color_editor_frame = tk.Frame(color_card.content, bg=self.theme["bg_card"])
        self.group_color_editor_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            self.group_color_editor_frame,
            text="Click on a color swatch to customize group colors (available after Preview)",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=300,
        ).pack(fill=tk.X)
        
        # Container for individual group color editors
        self.group_colors_container = tk.Frame(color_card.content, bg=self.theme["bg_card"])
        self.group_colors_container.pack(fill=tk.X, pady=4)
        
        # Store custom group colors
        self.custom_group_colors = {}
    
    def _update_color_preview(self):
        """Update the color preview display."""
        for widget in self.color_preview_frame.winfo_children():
            widget.destroy()
        
        palette = self.COLOR_PALETTES.get(self.palette_var.get(), self.COLOR_PALETTES["default"])
        
        for i, color in enumerate(palette[:10]):
            swatch = tk.Frame(
                self.color_preview_frame,
                bg=color,
                width=24,
                height=24,
                relief=tk.RAISED,
                bd=1
            )
            swatch.pack(side=tk.LEFT, padx=2)
            swatch.pack_propagate(False)
    
    def _on_palette_changed(self, event=None):
        """Handle palette selection change."""
        self._update_color_preview()
        # Update group colors if groups exist
        self._apply_palette_to_groups()
    
    def _apply_palette_to_groups(self):
        """Apply selected palette to all groups."""
        if not hasattr(self, 'group_colors_container'):
            return
        
        palette = self.COLOR_PALETTES.get(self.palette_var.get(), self.COLOR_PALETTES["default"])
        
        # Update custom colors dict with palette colors
        for i, (group_name, _) in enumerate(list(self.custom_group_colors.items())):
            self.custom_group_colors[group_name] = palette[i % len(palette)]
        
        # Refresh the color editor display
        self._update_group_color_editor()
    
    def _update_group_color_editor(self, group_names=None):
        """Update the group color editor with clickable swatches."""
        # Clear existing
        for widget in self.group_colors_container.winfo_children():
            widget.destroy()
        
        if group_names is None:
            group_names = list(self.custom_group_colors.keys())
        
        if not group_names:
            return
        
        # Initialize colors if not set
        palette = self.COLOR_PALETTES.get(self.palette_var.get(), self.COLOR_PALETTES["default"])
        for i, name in enumerate(group_names):
            if name not in self.custom_group_colors:
                self.custom_group_colors[name] = palette[i % len(palette)]
        
        # Create header
        tk.Label(
            self.group_colors_container,
            text="Group Colors (click to change):",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(4, 8))
        
        # Create color swatches for each group
        for group_name in group_names:
            color = self.custom_group_colors.get(group_name, palette[0])
            self._create_group_color_row(group_name, color)
    
    def _create_group_color_row(self, group_name, color):
        """Create a row with group name and clickable color swatch."""
        row_frame = tk.Frame(self.group_colors_container, bg=self.theme["bg_card"])
        row_frame.pack(fill=tk.X, pady=2)
        
        # Color swatch (clickable)
        swatch = tk.Frame(
            row_frame,
            bg=color,
            width=28,
            height=28,
            relief=tk.RAISED,
            bd=2,
            cursor="hand2"
        )
        swatch.pack(side=tk.LEFT, padx=(0, 8))
        swatch.pack_propagate(False)
        
        # Bind click to color picker
        swatch.bind("<Button-1>", lambda e, g=group_name, s=swatch: self._pick_group_color(g, s))
        
        # Group name label
        label_text = str(group_name)[:30] + "..." if len(str(group_name)) > 30 else str(group_name)
        tk.Label(
            row_frame,
            text=label_text,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Color hex label
        hex_label = tk.Label(
            row_frame,
            text=color,
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            width=8,
        )
        hex_label.pack(side=tk.RIGHT, padx=4)
        
        # Store reference to update later
        swatch.hex_label = hex_label
    
    def _pick_group_color(self, group_name, swatch_widget):
        """Open color picker dialog for a group."""
        from tkinter import colorchooser
        
        current_color = self.custom_group_colors.get(group_name, "#1f77b4")
        
        # Open color chooser
        result = colorchooser.askcolor(
            color=current_color,
            title=f"Choose color for '{group_name}'"
        )
        
        if result[1]:  # User selected a color (not cancelled)
            new_color = result[1]
            self.custom_group_colors[group_name] = new_color
            
            print(f"[DEBUG] Color changed for '{group_name}': {current_color} -> {new_color}")
            print(f"[DEBUG] All custom colors: {self.custom_group_colors}")
            
            # Update the swatch
            swatch_widget.configure(bg=new_color)
            
            # Update the hex label if it exists
            if hasattr(swatch_widget, 'hex_label'):
                swatch_widget.hex_label.configure(text=new_color)
    
    def _get_current_group_colors(self):
        """Get the current group colors dictionary."""
        if self.custom_group_colors:
            return self.custom_group_colors.copy()
        
        # Fall back to palette colors
        palette = self.COLOR_PALETTES.get(self.palette_var.get(), self.COLOR_PALETTES["default"])
        return {f"Group_{i}": palette[i % len(palette)] for i in range(10)}
    
    def _create_action_buttons(self):
        """Create action buttons."""
        btn_frame = tk.Frame(self.scrollable_frame, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        # Preview button
        ThemedButton(
            btn_frame,
            text="👁️ Preview Groups",
            command=self._preview_groups,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # Create button
        ActionButton(
            btn_frame,
            text="Create Groups",
            icon="✨",
            command=self._create_groups,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
    
    def _on_method_changed(self, event=None):
        """Handle grouping method change - update dynamic options."""
        # Clear current dynamic options
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()
        
        method = self.method_var.get()
        
        if method == "column":
            self._create_column_options()
        elif method == "year_periods":
            self._create_year_period_options()
        elif method == "multiitem":
            self._create_multiitem_options()
        elif method == "clustering":
            self._create_clustering_options()
        elif method == "concept":
            self._create_concept_options()
        elif method == "regex":
            self._create_regex_options()
        elif method == "random":
            self._create_random_options()
    
    def _get_available_columns(self) -> List[str]:
        """Get list of available columns from the dataset."""
        if self.bib and hasattr(self.bib, 'df'):
            return list(self.bib.df.columns)
        return []
    
    def _get_categorical_columns(self) -> List[str]:
        """Get columns suitable for categorical grouping."""
        if not self.bib or not hasattr(self.bib, 'df'):
            return []
        
        suitable = []
        df = self.bib.df
        
        for col in df.columns:
            # Skip columns with too many unique values
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 50:
                suitable.append(col)
        
        # Add common grouping columns even if not detected
        common = ["Year", "Document Type", "CA Country", "Language", "Source title"]
        for col in common:
            if col in df.columns and col not in suitable:
                suitable.append(col)
        
        return sorted(suitable)
    
    def _get_text_columns(self) -> List[str]:
        """Get columns suitable for text matching."""
        if not self.bib or not hasattr(self.bib, 'df'):
            return []
        
        text_cols = []
        df = self.bib.df
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it looks like text
                sample = df[col].dropna().head(10)
                avg_len = sample.str.len().mean() if len(sample) > 0 else 0
                if avg_len > 20:  # Likely text content
                    text_cols.append(col)
        
        # Add common text columns
        common = ["Title", "Abstract", "Author Keywords", "Index Keywords", 
                  "Processed Title", "Processed Abstract"]
        for col in common:
            if col in df.columns and col not in text_cols:
                text_cols.append(col)
        
        return sorted(text_cols)
    
    def _get_list_columns(self) -> List[str]:
        """Get columns that contain list-like values."""
        if not self.bib or not hasattr(self.bib, 'df'):
            return []
        
        list_cols = []
        df = self.bib.df
        sep = self.bib.default_separator if hasattr(self.bib, 'default_separator') else "; "
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if values contain separator
                sample = df[col].dropna().head(100).astype(str)
                has_sep = sample.str.contains(sep, regex=False).any()
                if has_sep:
                    list_cols.append(col)
        
        # Add common list columns
        common = ["Authors", "Author Keywords", "Index Keywords", "Affiliations",
                  "References", "Author and Index Keywords"]
        for col in common:
            if col in df.columns and col not in list_cols:
                list_cols.append(col)
        
        return sorted(list_cols)
    
    # =========================================================================
    # Method-specific option creators
    # =========================================================================
    
    def _create_column_options(self):
        """Create options for column-based grouping."""
        tk.Label(
            self.dynamic_frame,
            text="Select a column to group by. Each unique value becomes a group.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        ).pack(fill=tk.X, pady=(0, 8))
        
        # Column selection
        columns = self._get_categorical_columns()
        self.column_var = tk.StringVar(value=columns[0] if columns else "")
        
        self.column_combo = LabeledCombobox(
            self.dynamic_frame,
            label="Column:",
            values=columns,
            variable=self.column_var,
            theme=self.theme_name,
            label_width=12
        )
        self.column_combo.pack(fill=tk.X, pady=4)
        
        # Show unique values count
        self.unique_label = tk.Label(
            self.dynamic_frame,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        )
        self.unique_label.pack(fill=tk.X, pady=4)
        
        # Bind to update unique count
        self.column_combo.combobox.bind("<<ComboboxSelected>>", self._update_unique_count)
        self._update_unique_count()
    
    def _update_unique_count(self, event=None):
        """Update the unique values count display."""
        if not hasattr(self, 'column_var') or not self.bib:
            return
        
        col = self.column_var.get()
        if col and col in self.bib.df.columns:
            n_unique = self.bib.df[col].nunique()
            self.unique_label.config(text=f"Unique values: {n_unique} (will create {n_unique} groups)")
    
    def _create_year_period_options(self):
        """Create options for year period grouping."""
        tk.Label(
            self.dynamic_frame,
            text="Divide documents into time periods based on publication year.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        ).pack(fill=tk.X, pady=(0, 8))
        
        # Year column
        year_cols = ["Year", "PY", "Publication Year"]
        available_year_cols = [c for c in year_cols if c in self._get_available_columns()]
        if not available_year_cols:
            available_year_cols = self._get_available_columns()
        
        self.year_col_var = tk.StringVar(value=available_year_cols[0] if available_year_cols else "")
        
        self.year_col_combo = LabeledCombobox(
            self.dynamic_frame,
            label="Year Column:",
            values=available_year_cols,
            variable=self.year_col_var,
            theme=self.theme_name,
            label_width=14
        )
        self.year_col_combo.pack(fill=tk.X, pady=4)
        
        # Period definition method
        self.period_method_var = tk.StringVar(value="n_periods")
        
        method_frame = tk.Frame(self.dynamic_frame, bg=self.theme["bg_card"])
        method_frame.pack(fill=tk.X, pady=8)
        
        tk.Radiobutton(
            method_frame,
            text="Number of periods",
            variable=self.period_method_var,
            value="n_periods",
            command=self._on_period_method_changed,
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_secondary"],
            font=FONTS.get_font("body"),
        ).pack(anchor=tk.W)
        
        tk.Radiobutton(
            method_frame,
            text="Custom cutpoints",
            variable=self.period_method_var,
            value="cutpoints",
            command=self._on_period_method_changed,
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_secondary"],
            font=FONTS.get_font("body"),
        ).pack(anchor=tk.W)
        
        # Number of periods
        self.n_periods_frame = tk.Frame(self.dynamic_frame, bg=self.theme["bg_card"])
        self.n_periods_frame.pack(fill=tk.X, pady=4)
        
        self.n_periods_spin = LabeledSpinbox(
            self.n_periods_frame,
            label="Number of periods:",
            from_=2, to=20, default=5,
            theme=self.theme_name,
            label_width=16
        )
        self.n_periods_spin.pack(fill=tk.X)
        
        # Custom cutpoints
        self.cutpoints_frame = tk.Frame(self.dynamic_frame, bg=self.theme["bg_card"])
        
        self.cutpoints_entry = LabeledEntry(
            self.cutpoints_frame,
            label="Cutpoints:",
            default="2010, 2015, 2020",
            theme=self.theme_name,
            label_width=16
        )
        self.cutpoints_entry.pack(fill=tk.X)
        
        tk.Label(
            self.cutpoints_frame,
            text="Enter years separated by commas (e.g., 2010, 2015, 2020)",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(fill=tk.X)
        
        # Show year range info
        self._show_year_range_info()
    
    def _show_placeholder(self):
        """Show detailed placeholder with instructions."""
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
            for widget in self.results_tab.winfo_children():
                widget.destroy()
        except tk.TclError:
            return
        
        msg = (
            "⚙️ Group Setup\n\n"
            "Define comparison groups for analysis.\n\n"
            "Features:\n"
            "• Keyword-based group definition\n"
            "• Year/period-based groups\n"
            "• Field value matching\n"
            "• Boolean combinations\n"
            "\n"
            "Groups defined here are used in all Group panels.\n\n"
            "Steps:\n"
            "1. Load a dataset\n"
            "2. Click 'Add Group'\n"
            "3. Define group criteria\n"
            "4. Repeat for additional groups\n"
        )
        
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
            tk.Label(
                self.results_tab,
                text=msg,
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
                justify=tk.CENTER,
            ).pack(expand=True)
        except tk.TclError:
            pass

    def _on_period_method_changed(self):
        """Toggle between number of periods and custom cutpoints."""
        if self.period_method_var.get() == "n_periods":
            self.cutpoints_frame.pack_forget()
            self.n_periods_frame.pack(fill=tk.X, pady=4)
        else:
            self.n_periods_frame.pack_forget()
            self.cutpoints_frame.pack(fill=tk.X, pady=4)
    
    def _show_year_range_info(self):
        """Show information about the year range in the data."""
        if self.bib and hasattr(self, 'year_col_var'):
            col = self.year_col_var.get()
            if col and col in self.bib.df.columns:
                years = pd.to_numeric(self.bib.df[col], errors='coerce').dropna()
                if len(years) > 0:
                    min_year = int(years.min())
                    max_year = int(years.max())
                    
                    info_label = tk.Label(
                        self.dynamic_frame,
                        text=f"Year range in data: {min_year} - {max_year}",
                        font=FONTS.get_font("small"),
                        bg=self.theme["bg_card"],
                        fg=self.theme["text_muted"],
                    )
                    info_label.pack(fill=tk.X, pady=4)
    
    def _create_multiitem_options(self):
        """Create options for multi-item column grouping."""
        tk.Label(
            self.dynamic_frame,
            text="Group by items in a list column. Each unique item becomes a group "
                 "(documents can belong to multiple groups).",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        ).pack(fill=tk.X, pady=(0, 8))
        
        # Column selection
        columns = self._get_list_columns()
        self.multiitem_col_var = tk.StringVar(value=columns[0] if columns else "")
        
        self.multiitem_combo = LabeledCombobox(
            self.dynamic_frame,
            label="Column:",
            values=columns,
            variable=self.multiitem_col_var,
            theme=self.theme_name,
            label_width=12
        )
        self.multiitem_combo.pack(fill=tk.X, pady=4)
        
        # Top N items
        self.topn_spin = LabeledSpinbox(
            self.dynamic_frame,
            label="Top N items:",
            from_=0, to=100, default=10,
            theme=self.theme_name,
            label_width=12
        )
        self.topn_spin.pack(fill=tk.X, pady=4)
        ToolTip(self.topn_spin, "0 = keep all items")
        
        # Include items filter
        self.include_items_entry = LabeledEntry(
            self.dynamic_frame,
            label="Include only:",
            default="",
            theme=self.theme_name,
            label_width=12
        )
        self.include_items_entry.pack(fill=tk.X, pady=4)
        ToolTip(self.include_items_entry, "Comma-separated list of items to include (leave empty for all)")
        
        # Exclude items filter
        self.exclude_items_entry = LabeledEntry(
            self.dynamic_frame,
            label="Exclude:",
            default="",
            theme=self.theme_name,
            label_width=12
        )
        self.exclude_items_entry.pack(fill=tk.X, pady=4)
    
    def _create_clustering_options(self):
        """Create options for clustering-based grouping."""
        tk.Label(
            self.dynamic_frame,
            text="Group documents using clustering algorithms based on text similarity.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        ).pack(fill=tk.X, pady=(0, 8))
        
        # Clustering method
        self.cluster_method_var = tk.StringVar(value="kmeans")
        
        self.cluster_method_combo = LabeledCombobox(
            self.dynamic_frame,
            label="Method:",
            values=["kmeans", "hierarchical"],
            variable=self.cluster_method_var,
            theme=self.theme_name,
            label_width=12
        )
        self.cluster_method_combo.pack(fill=tk.X, pady=4)
        
        # Number of clusters
        self.n_clusters_spin = LabeledSpinbox(
            self.dynamic_frame,
            label="Clusters:",
            from_=2, to=20, default=5,
            theme=self.theme_name,
            label_width=12
        )
        self.n_clusters_spin.pack(fill=tk.X, pady=4)
        
        # Text column for clustering
        text_cols = self._get_text_columns()
        self.cluster_text_var = tk.StringVar(
            value="Abstract" if "Abstract" in text_cols else (text_cols[0] if text_cols else "")
        )
        
        self.cluster_text_combo = LabeledCombobox(
            self.dynamic_frame,
            label="Text Column:",
            values=text_cols,
            variable=self.cluster_text_var,
            theme=self.theme_name,
            label_width=12
        )
        self.cluster_text_combo.pack(fill=tk.X, pady=4)
        
        # Use keywords option
        self.use_keywords_var = tk.BooleanVar(value=True)
        self.use_keywords_check = LabeledCheckbox(
            self.dynamic_frame,
            label="Also use keywords for clustering",
            variable=self.use_keywords_var,
            theme=self.theme_name,
        )
        self.use_keywords_check.pack(fill=tk.X, pady=4)
    
    def _create_concept_options(self):
        """Create options for concept DataFrame grouping."""
        tk.Label(
            self.dynamic_frame,
            text="Group by concepts - either from a file, predefined SDGs, "
                 "or existing binary columns in your dataset.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        ).pack(fill=tk.X, pady=(0, 8))
        
        # Source selection
        self.concept_source_var = tk.StringVar(value="file")
        
        self.source_frame = tk.Frame(self.dynamic_frame, bg=self.theme["bg_card"])
        self.source_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            self.source_frame,
            text="Source:",
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W)
        
        # Radio buttons stacked vertically
        for src_id, src_text, src_desc in [
            ("file", "Concept File", "Load concepts from Excel/CSV file"),
            ("sdg", "SDG Concepts", "Use predefined Sustainable Development Goals"),
            ("columns", "Binary Columns from Dataset", "Use existing 0/1 columns in your data"),
        ]:
            rb_frame = tk.Frame(self.source_frame, bg=self.theme["bg_card"])
            rb_frame.pack(fill=tk.X, anchor=tk.W)
            
            rb = tk.Radiobutton(
                rb_frame,
                text=src_text,
                variable=self.concept_source_var,
                value=src_id,
                command=self._on_concept_source_changed,
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
                activebackground=self.theme["bg_card"],
                font=FONTS.get_font("body"),
            )
            rb.pack(side=tk.LEFT)
            
            # Add description
            tk.Label(
                rb_frame,
                text=f"- {src_desc}",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(side=tk.LEFT, padx=(4, 0))
        
        # === File selection frame ===
        self.concept_file_frame = tk.Frame(self.dynamic_frame, bg=self.theme["bg_card"])
        self.concept_file_frame.pack(fill=tk.X, pady=4)
        
        self.concept_file_var = tk.StringVar(value="")
        
        tk.Label(
            self.concept_file_frame,
            text="Concept File:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            width=12,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.concept_file_entry = tk.Entry(
            self.concept_file_frame,
            textvariable=self.concept_file_var,
            font=FONTS.get_font("body"),
        )
        self.concept_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        ThemedButton(
            self.concept_file_frame,
            text="Browse",
            command=self._browse_concept_file,
            theme=self.theme_name,
        ).pack(side=tk.LEFT)
        
        # === Binary columns selection frame ===
        self.concept_columns_frame = tk.Frame(self.dynamic_frame, bg=self.theme["bg_card"])
        # Will be shown when "columns" source is selected
        
        tk.Label(
            self.concept_columns_frame,
            text="Select binary columns to use as groups:",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W, pady=(0, 4))
        
        # Select all / Deselect all buttons at top
        btn_frame = tk.Frame(self.concept_columns_frame, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=(0, 4))
        
        ThemedButton(
            btn_frame,
            text="Select All",
            command=self._select_all_binary_cols,
            theme=self.theme_name,
            style="ghost",
            size="small",
        ).pack(side=tk.LEFT, padx=(0, 4))
        
        ThemedButton(
            btn_frame,
            text="Deselect All",
            command=self._deselect_all_binary_cols,
            theme=self.theme_name,
            style="ghost",
            size="small",
        ).pack(side=tk.LEFT)
        
        # Scrollable frame for checkboxes
        self.binary_cols_container = tk.Frame(self.concept_columns_frame, bg=self.theme["bg_card"])
        self.binary_cols_container.pack(fill=tk.X, pady=4)
        
        # Canvas for scrolling
        self.binary_cols_canvas = tk.Canvas(
            self.binary_cols_container,
            bg=self.theme["bg_card"],
            highlightthickness=1,
            highlightbackground=self.theme["border"],
            height=150,
        )
        self.binary_cols_scrollbar = ttk.Scrollbar(
            self.binary_cols_container,
            orient=tk.VERTICAL,
            command=self.binary_cols_canvas.yview
        )
        
        self.binary_cols_inner = tk.Frame(self.binary_cols_canvas, bg=self.theme["bg_card"])
        
        self.binary_cols_inner.bind(
            "<Configure>",
            lambda e: self.binary_cols_canvas.configure(scrollregion=self.binary_cols_canvas.bbox("all"))
        )
        
        self.binary_cols_canvas.create_window((0, 0), window=self.binary_cols_inner, anchor="nw")
        self.binary_cols_canvas.configure(yscrollcommand=self.binary_cols_scrollbar.set)
        
        self.binary_cols_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.binary_cols_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel
        self.binary_cols_canvas.bind("<Enter>", lambda e: self.binary_cols_canvas.bind_all("<MouseWheel>", self._on_binary_cols_scroll))
        self.binary_cols_canvas.bind("<Leave>", lambda e: self.binary_cols_canvas.unbind_all("<MouseWheel>"))
        
        # Store checkbox variables
        self.binary_col_vars = {}
        self._binary_col_names = []
        
        # Populate with binary columns
        self._populate_binary_columns()
        
        # Text column for matching (only for file/sdg source)
        self.concept_text_frame = tk.Frame(self.dynamic_frame, bg=self.theme["bg_card"])
        self.concept_text_frame.pack(fill=tk.X, pady=4)
        
        text_cols = self._get_text_columns()
        self.concept_text_var = tk.StringVar(
            value="Abstract" if "Abstract" in text_cols else (text_cols[0] if text_cols else "")
        )
        
        self.concept_text_combo = LabeledCombobox(
            self.concept_text_frame,
            label="Match in:",
            values=text_cols,
            variable=self.concept_text_var,
            theme=self.theme_name,
            label_width=12
        )
        self.concept_text_combo.pack(fill=tk.X)
        
        # Whole word matching (only for file/sdg source)
        self.whole_word_frame = tk.Frame(self.dynamic_frame, bg=self.theme["bg_card"])
        self.whole_word_frame.pack(fill=tk.X, pady=4)
        
        self.whole_word_var = tk.BooleanVar(value=False)
        self.whole_word_check = LabeledCheckbox(
            self.whole_word_frame,
            label="Match whole words only",
            variable=self.whole_word_var,
            theme=self.theme_name,
        )
        self.whole_word_check.pack(fill=tk.X)
        
        # Initialize visibility
        self._on_concept_source_changed()
    
    def _on_binary_cols_scroll(self, event):
        """Handle mousewheel scroll in binary columns list."""
        self.binary_cols_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _select_all_binary_cols(self):
        """Select all binary column checkboxes."""
        for var in self.binary_col_vars.values():
            var.set(True)
    
    def _deselect_all_binary_cols(self):
        """Deselect all binary column checkboxes."""
        for var in self.binary_col_vars.values():
            var.set(False)
    
    def _populate_binary_columns(self):
        """Find and populate binary (0/1) columns from the dataframe as checkboxes."""
        # Clear existing checkboxes
        for widget in self.binary_cols_inner.winfo_children():
            widget.destroy()
        
        self.binary_col_vars = {}
        self._binary_col_names = []
        
        if not self.bib or self.bib.df is None:
            tk.Label(
                self.binary_cols_inner,
                text="No data loaded",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(anchor=tk.W, padx=4, pady=4)
            return
        
        binary_cols = []
        for col in self.bib.df.columns:
            try:
                # Check if column contains only 0, 1, or NaN
                unique_vals = self.bib.df[col].dropna().unique()
                if len(unique_vals) <= 2 and all(v in [0, 1, 0.0, 1.0, True, False] for v in unique_vals):
                    # Count how many 1s
                    count_ones = int((self.bib.df[col] == 1).sum())
                    if count_ones > 0:  # Only include if at least some matches
                        binary_cols.append((col, count_ones))
            except:
                pass
        
        if not binary_cols:
            tk.Label(
                self.binary_cols_inner,
                text="No binary columns found in dataset",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(anchor=tk.W, padx=4, pady=4)
            return
        
        # Sort by count descending
        binary_cols.sort(key=lambda x: -x[1])
        
        # Create checkboxes for each binary column
        for col, count in binary_cols:
            var = tk.BooleanVar(value=False)
            self.binary_col_vars[col] = var
            self._binary_col_names.append(col)
            
            cb = tk.Checkbutton(
                self.binary_cols_inner,
                text=f"{col} ({count} docs)",
                variable=var,
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                activebackground=self.theme["bg_card"],
                selectcolor=self.theme["bg_input"],
                anchor=tk.W,
            )
            cb.pack(fill=tk.X, anchor=tk.W, padx=4, pady=1)
    
    def _on_concept_source_changed(self):
        """Handle concept source radio button change."""
        source = self.concept_source_var.get()
        
        if source == "file":
            self.concept_file_frame.pack(fill=tk.X, pady=4, after=self._get_source_frame())
            self.concept_columns_frame.pack_forget()
            self.concept_text_frame.pack(fill=tk.X, pady=4)
            self.whole_word_frame.pack(fill=tk.X, pady=4)
        elif source == "sdg":
            self.concept_file_frame.pack_forget()
            self.concept_columns_frame.pack_forget()
            self.concept_text_frame.pack(fill=tk.X, pady=4)
            self.whole_word_frame.pack(fill=tk.X, pady=4)
        else:  # columns
            self.concept_file_frame.pack_forget()
            self.concept_columns_frame.pack(fill=tk.X, pady=4, after=self._get_source_frame())
            self.concept_text_frame.pack_forget()
            self.whole_word_frame.pack_forget()
    
    def _get_source_frame(self):
        """Get the source frame after which to pack dynamic content."""
        return self.source_frame if hasattr(self, 'source_frame') else None
    
    def _browse_concept_file(self):
        """Open file browser for concept file."""
        filetypes = [
            ("Excel Files", "*.xlsx *.xls"),
            ("CSV Files", "*.csv"),
            ("All Files", "*.*"),
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Concept File",
            filetypes=filetypes,
        )
        
        if filename:
            self.concept_file_var.set(filename)
    
    def _create_regex_options(self):
        """Create options for dictionary/regex grouping."""
        tk.Label(
            self.dynamic_frame,
            text="Define groups using custom patterns. Each line is: "
                 "GroupName: pattern1, pattern2, ...",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        ).pack(fill=tk.X, pady=(0, 8))
        
        # Text area for patterns
        tk.Label(
            self.dynamic_frame,
            text="Pattern Definition:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            anchor=tk.W
        ).pack(fill=tk.X)
        
        self.patterns_text = tk.Text(
            self.dynamic_frame,
            height=8,
            font=FONTS.get_font("mono"),
            wrap=tk.WORD,
        )
        self.patterns_text.pack(fill=tk.X, pady=4)
        
        # Insert example
        example = """Machine Learning: machine learning, deep learning, neural network*
NLP: natural language, NLP, text mining
Computer Vision: image*, vision, object detection"""
        self.patterns_text.insert("1.0", example)
        
        # Text column for matching
        text_cols = self._get_text_columns()
        self.regex_text_var = tk.StringVar(
            value="Abstract" if "Abstract" in text_cols else (text_cols[0] if text_cols else "")
        )
        
        self.regex_text_combo = LabeledCombobox(
            self.dynamic_frame,
            label="Match in:",
            values=text_cols,
            variable=self.regex_text_var,
            theme=self.theme_name,
            label_width=12
        )
        self.regex_text_combo.pack(fill=tk.X, pady=4)
        
        # Case sensitive
        self.case_sensitive_var = tk.BooleanVar(value=False)
        self.case_sensitive_check = LabeledCheckbox(
            self.dynamic_frame,
            label="Case sensitive matching",
            variable=self.case_sensitive_var,
            theme=self.theme_name,
        )
        self.case_sensitive_check.pack(fill=tk.X, pady=4)
    
    def _create_random_options(self):
        """Create options for random group generation."""
        tk.Label(
            self.dynamic_frame,
            text="Create random overlapping groups for testing purposes. "
                 "Each document has ~50% chance of belonging to each group.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        ).pack(fill=tk.X, pady=(0, 8))
        
        # Number of random groups
        self.n_random_groups_spin = LabeledSpinbox(
            self.dynamic_frame,
            label="Number of groups:",
            from_=2, to=20, default=5,
            theme=self.theme_name,
            label_width=16
        )
        self.n_random_groups_spin.pack(fill=tk.X, pady=4)
        
        # Random seed (optional)
        self.random_seed_entry = LabeledEntry(
            self.dynamic_frame,
            label="Random seed:",
            default="",
            theme=self.theme_name,
            label_width=16
        )
        self.random_seed_entry.pack(fill=tk.X, pady=4)
        ToolTip(self.random_seed_entry, "Leave empty for random seed, or enter a number for reproducibility")
        
        # Info about random groups
        info_frame = tk.Frame(self.dynamic_frame, bg=self.theme["bg_card"])
        info_frame.pack(fill=tk.X, pady=8)
        
        tk.Label(
            info_frame,
            text="ℹ️ Random groups are useful for:\n"
                 "• Testing group analysis workflows\n"
                 "• Baseline comparisons\n"
                 "• Understanding group overlap effects",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.LEFT,
        ).pack(fill=tk.X)
    
    # =========================================================================
    # Results Display
    # =========================================================================
    
    def _create_results(self):
        """Create the results panel."""
        super()._create_results()
        
        # Create notebook with Info tab
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="📊 Preview")
        
        # Info tab
        self.info_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_tab, text="ℹ️ Info")
        self._create_info_content(self.info_tab)
        
        self._show_placeholder()
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading state within the results tab (not destroying notebook)."""
        from biblium.gui.widgets.progress import LoadingSpinner
        
        self._stop_active_spinners()
        
        try:
            if hasattr(self, 'results_tab') and self.results_tab.winfo_exists():
                for widget in self.results_tab.winfo_children():
                    try:
                        widget.destroy()
                    except:
                        pass
                
                loading_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
                loading_frame.pack(expand=True)
                
                spinner = LoadingSpinner(loading_frame, size=32, theme=self.theme_name)
                spinner.pack()
                spinner.start()
                
                if not hasattr(self, '_active_spinners'):
                    self._active_spinners = []
                self._active_spinners.append(spinner)
                
                tk.Label(
                    loading_frame,
                    text=message,
                    font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"],
                    fg=self.theme["text_muted"],
                ).pack(pady=(12, 0))
        except tk.TclError:
            pass
    
    def _show_error(self, message: str):
        """Show error message within the results tab."""
        self._stop_active_spinners()
        
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
                
            for widget in self.results_tab.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
            
            error_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
            error_frame.pack(expand=True)
            
            tk.Label(error_frame, text="❌", font=("Segoe UI", 32), bg=self.theme["bg_card"]).pack()
            tk.Label(error_frame, text="Error", font=FONTS.get_font("heading", bold=True),
                    bg=self.theme["bg_card"], fg=self.theme["danger"]).pack(pady=(8, 4))
            tk.Label(error_frame, text=message, font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_secondary"], wraplength=400).pack()
        except tk.TclError:
            pass
    
    def _show_initial_message(self, message: str = None):
        """Show initial message within the results tab."""
        self._stop_active_spinners()
        
        if message is None:
            message = "⚙️ Group Setup\n\nConfigure grouping options and click 'Preview' or 'Create Groups'."
        
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
                
            for widget in self.results_tab.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
            
            tk.Label(self.results_tab, text=message, font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"], justify=tk.CENTER).pack(expand=True)
        except tk.TclError:
            pass
    
    def _show_placeholder(self):
        """Show placeholder in results tab."""
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
            for widget in self.results_tab.winfo_children():
                widget.destroy()
            tk.Label(
                self.results_tab,
                text="⚙️ Group Setup\n\n"
                     "Configure grouping options and click 'Preview' or 'Create Groups'.",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
                justify=tk.CENTER,
            ).pack(expand=True)
        except tk.TclError:
            pass  # Widget was destroyed
    
    def _preview_groups(self):
        """Preview the groups without creating them."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Previewing groups...")
        
        def do_preview():
            try:
                group_matrix = self._generate_group_matrix()
                self._safe_after(lambda: self._display_preview(group_matrix))
            except Exception as e:
                self._safe_after(lambda msg=str(e): self._show_error(msg))
        
        threading.Thread(target=do_preview, daemon=True).start()
    
    def _display_preview(self, group_matrix: pd.DataFrame):
        """Display preview of group assignment."""
        # Safety check - ensure widgets still exist
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                self._create_results()
        except tk.TclError:
            self._create_results()
        
        try:
            for widget in self.results_tab.winfo_children():
                widget.destroy()
        except tk.TclError:
            return  # Widget no longer exists
        
        if group_matrix is None or group_matrix.empty:
            self._show_error("No groups could be created with current settings.")
            return
        
        # Summary cards
        grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        n_groups = len(group_matrix.columns)
        n_docs = len(group_matrix)
        
        # Calculate overlap
        docs_in_multiple = (group_matrix.sum(axis=1) > 1).sum()
        overlap_pct = docs_in_multiple / n_docs * 100 if n_docs > 0 else 0
        
        grid.add_card(StatsCard(grid, "Groups", str(n_groups), "📊", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Documents", f"{n_docs:,}", "📄", self.theme_name))
        grid.add_card(StatsCard(grid, "Overlap", f"{overlap_pct:.1f}%", "🔀", self.theme_name))
        
        # Safety check for results_content
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
        except tk.TclError:
            return
        
        # Group sizes table
        tk.Label(
            self.results_tab,
            text="📋 Preview - Group Sizes",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4))
        
        # Create summary DataFrame and initialize colors
        summary_data = []
        palette = self.COLOR_PALETTES.get(self.palette_var.get(), self.COLOR_PALETTES["default"])
        group_names = list(group_matrix.columns)
        
        # Initialize custom colors if not already set for these groups
        for i, col in enumerate(group_names):
            if col not in self.custom_group_colors:
                self.custom_group_colors[col] = palette[i % len(palette)]
        
        for i, col in enumerate(group_matrix.columns):
            count = group_matrix[col].sum()
            pct = count / n_docs * 100 if n_docs > 0 else 0
            color = self.custom_group_colors.get(col, palette[i % len(palette)])
            summary_data.append({
                "Group": col,
                "Documents": int(count),
                "Percentage": f"{pct:.1f}%",
                "Color": color,
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        table = DataTable(self.results_tab, theme=self.theme_name, height=min(15, n_groups + 2))
        table.pack(fill=tk.BOTH, expand=True, pady=8)
        table.set_data(summary_df)
        
        # Update the group color editor with actual group names
        self._update_group_color_editor(group_names)
        
        # Preview note
        tk.Label(
            self.results_tab,
            text="⚠️ This is a preview. Click 'Create Groups' to finalize. "
                 "Customize colors in the 'Group Colors' section above.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=400,
        ).pack(fill=tk.X, pady=8)
    
    def _create_groups(self):
        """Create the BiblioGroup instance."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        # Check if biblium group is available
        if not HAS_BIBLIUM_GROUP:
            messagebox.showerror("Error", "BiblioGroupAnalysis is not available. Check biblium installation.")
            return
        
        self._show_loading("Creating groups...")
        
        # Capture colors BEFORE starting thread (tkinter vars must be accessed from main thread)
        captured_custom_colors = self.custom_group_colors.copy() if self.custom_group_colors else {}
        captured_palette = self.COLOR_PALETTES.get(self.palette_var.get(), self.COLOR_PALETTES["default"])
        
        print(f"[DEBUG] Captured custom colors: {captured_custom_colors}")
        print(f"[DEBUG] Captured palette: {captured_palette[:3]}...")
        
        def do_create():
            try:
                # Get configuration
                config = self._get_group_config()
                
                # Debug print
                print(f"Creating groups with config: {config['group_desc']}")
                print(f"Matrix kwargs: {config.get('matrix_kwargs', {})}")
                
                # Build basic kwargs for BiblioGroupAnalysis
                kwargs = {
                    "df": self.bib.df.copy(),
                    "db": self.bib.db if hasattr(self.bib, 'db') else "scopus",
                    "group_desc": config["group_desc"],
                    "res_folder": self.results_folder.get() or "results-groups",
                    "preprocess_level": self.preprocess_level.get(),
                    "merge_data_with_group": self.merge_data_var.get(),
                }
                
                # Add group colors if defined
                if config.get("group_colors"):
                    kwargs["group_colors"] = config["group_colors"]
                
                # Add matrix generation kwargs that BiblioGroupAnalysis supports
                # These get passed through to generate_group_matrix
                # Note: don't pass 'sep' here - BiblioGroupAnalysis uses self.default_separator
                matrix_kwargs = config.get("matrix_kwargs", {})
                
                for key in ["force_type", "n_periods", "cutpoints", "top_n", 
                           "include_items", "exclude_items", "text_column",
                           "concept_whole_word", "regex_flags"]:
                    if key in matrix_kwargs:
                        kwargs[key] = matrix_kwargs[key]
                
                # Create BiblioGroupAnalysis
                bib_group = BiblioGroupAnalysis(**kwargs)
                
                # Store group colors - use captured custom colors if defined, otherwise palette
                bib_group.group_colors = {}
                
                for i, group_name in enumerate(bib_group.groups.keys()):
                    if captured_custom_colors and group_name in captured_custom_colors:
                        bib_group.group_colors[group_name] = captured_custom_colors[group_name]
                    else:
                        bib_group.group_colors[group_name] = captured_palette[i % len(captured_palette)]
                
                print(f"Created BiblioGroupAnalysis with {len(bib_group.groups)} groups")
                print(f"Group colors: {bib_group.group_colors}")
                
                # Use safe_after to schedule UI update
                self._safe_after(lambda: self._on_groups_created(bib_group))
                
            except Exception as e:
                import traceback
                error_msg = str(e)
                traceback.print_exc()
                self._safe_after(lambda msg=error_msg: self._show_error(f"Failed to create groups: {msg}"))
        
        threading.Thread(target=do_create, daemon=True).start()
    
    def _safe_after(self, callback, delay=0):
        """Safely schedule a callback on the main thread.
        
        This wraps self.after() in a try-except to handle cases where
        the widget has been destroyed or the main loop isn't running.
        """
        try:
            if self.winfo_exists():
                self.after(delay, callback)
        except (tk.TclError, RuntimeError) as e:
            # Widget was destroyed or main loop issue
            print(f"[WARNING] Could not schedule callback: {e}")
    
    def _generate_group_matrix(self) -> pd.DataFrame:
        """Generate group matrix based on current settings (for preview)."""
        config = self._get_group_config()
        
        # Get separator from bib
        sep = getattr(self.bib, 'default_separator', '; ')
        
        # Add sep to matrix_kwargs
        matrix_kwargs = config.get("matrix_kwargs", {})
        matrix_kwargs["sep"] = sep
        
        return utilsbib.generate_group_matrix(
            df=self.bib.df,
            group_desc=config["group_desc"],
            **matrix_kwargs
        )
    
    def _get_group_config(self) -> Dict[str, Any]:
        """Get the group configuration based on current settings."""
        method = self.method_var.get()
        config = {
            "group_desc": None,
            "matrix_kwargs": {},
            "extra_kwargs": {},
            "group_colors": None,
        }
        
        # Get color palette
        palette = self.COLOR_PALETTES.get(self.palette_var.get(), self.COLOR_PALETTES["default"])
        
        if method == "column":
            config["group_desc"] = self.column_var.get()
            config["matrix_kwargs"]["force_type"] = "column"
            
        elif method == "year_periods":
            config["group_desc"] = self.year_col_var.get()
            config["matrix_kwargs"]["force_type"] = "year"
            
            if self.period_method_var.get() == "n_periods":
                config["matrix_kwargs"]["n_periods"] = self.n_periods_spin.get()
            else:
                # Parse cutpoints
                cutpoints_str = self.cutpoints_entry.get()
                cutpoints = [int(x.strip()) for x in cutpoints_str.split(",") if x.strip()]
                config["matrix_kwargs"]["cutpoints"] = cutpoints
                
        elif method == "multiitem":
            config["group_desc"] = self.multiitem_col_var.get()
            config["matrix_kwargs"]["force_type"] = "multiitem"
            
            top_n = self.topn_spin.get()
            if top_n > 0:
                config["matrix_kwargs"]["top_n"] = top_n
            
            include = self.include_items_entry.get().strip()
            if include:
                config["matrix_kwargs"]["include_items"] = [x.strip() for x in include.split(",")]
            
            exclude = self.exclude_items_entry.get().strip()
            if exclude:
                config["matrix_kwargs"]["exclude_items"] = [x.strip() for x in exclude.split(",")]
                
        elif method == "clustering":
            # For clustering, we need to handle it differently
            # Return the number of clusters and let BiblioGroupAnalysis handle it
            if not hasattr(self, 'n_clusters_spin'):
                raise ValueError("Clustering options not initialized. Please select 'By Clustering' method first.")
            n_clusters = self.n_clusters_spin.get()
            config["group_desc"] = f"cluster_{n_clusters}"
            config["extra_kwargs"]["n_clusters"] = n_clusters
            config["extra_kwargs"]["cluster_method"] = getattr(self, 'cluster_method_var', tk.StringVar(value="kmeans")).get()
            config["extra_kwargs"]["cluster_text_col"] = getattr(self, 'cluster_text_var', tk.StringVar(value="Abstract")).get()
            
        elif method == "concept":
            source = self.concept_source_var.get()
            
            if source == "sdg":
                # Load SDG concepts
                try:
                    sdg_path = os.path.join(
                        os.path.dirname(__file__),
                        "..", "..", "..", "additional files", "Scopus_SDG_metadata.xlsx"
                    )
                    if os.path.exists(sdg_path):
                        config["group_desc"] = pd.read_excel(sdg_path)
                    else:
                        raise FileNotFoundError("SDG metadata file not found")
                except Exception as e:
                    raise ValueError(f"Could not load SDG concepts: {e}")
                
                config["matrix_kwargs"]["force_type"] = "concept"
                config["matrix_kwargs"]["text_column"] = self.concept_text_var.get()
                config["matrix_kwargs"]["concept_whole_word"] = self.whole_word_var.get()
                
            elif source == "columns":
                # Use binary columns from dataframe - get selected from checkboxes
                selected_cols = [col for col, var in self.binary_col_vars.items() if var.get()]
                if not selected_cols:
                    raise ValueError("Please select at least one binary column")
                
                # Create binary DataFrame from selected columns
                binary_df = self.bib.df[selected_cols].copy()
                # Ensure binary values (0/1)
                binary_df = binary_df.fillna(0).astype(int).astype(bool)
                
                config["group_desc"] = binary_df
                config["matrix_kwargs"]["force_type"] = "binary"
                
            else:  # file
                # Load from file
                concept_file = self.concept_file_var.get()
                if not concept_file:
                    raise ValueError("Please select a concept file")
                
                if concept_file.endswith('.csv'):
                    config["group_desc"] = pd.read_csv(concept_file)
                else:
                    config["group_desc"] = pd.read_excel(concept_file)
                
                config["matrix_kwargs"]["force_type"] = "concept"
                config["matrix_kwargs"]["text_column"] = self.concept_text_var.get()
                config["matrix_kwargs"]["concept_whole_word"] = self.whole_word_var.get()
            
        elif method == "regex":
            # Parse patterns
            patterns_text = self.patterns_text.get("1.0", tk.END).strip()
            patterns_dict = {}
            
            for line in patterns_text.split("\n"):
                line = line.strip()
                if ":" in line:
                    group_name, terms_str = line.split(":", 1)
                    terms = [t.strip() for t in terms_str.split(",") if t.strip()]
                    if terms:
                        patterns_dict[group_name.strip()] = terms
            
            if not patterns_dict:
                raise ValueError("No valid patterns defined")
            
            config["group_desc"] = patterns_dict
            config["matrix_kwargs"]["force_type"] = "regex"
            config["matrix_kwargs"]["text_column"] = self.regex_text_var.get()
            
            if not self.case_sensitive_var.get():
                import re
                config["matrix_kwargs"]["regex_flags"] = re.IGNORECASE
                
        elif method == "random":
            n_groups = self.n_random_groups_spin.get()
            config["group_desc"] = n_groups
            
            # Set random seed if provided
            seed_str = self.random_seed_entry.get().strip()
            if seed_str:
                try:
                    np.random.seed(int(seed_str))
                except ValueError:
                    pass
        
        return config
    
    def _on_groups_created(self, bib_group):
        """Handle successful group creation."""
        # Safety check - ensure widgets still exist
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                # Recreate results if needed
                self._create_results()
        except tk.TclError:
            # Widget was destroyed, recreate
            self._create_results()
        
        # Clear results
        try:
            for widget in self.results_tab.winfo_children():
                widget.destroy()
        except tk.TclError:
            return  # Widget no longer exists, abort
        
        # Summary cards
        grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        n_groups = len(bib_group.groups)
        n_docs = bib_group.n
        
        grid.add_card(StatsCard(grid, "Groups Created", str(n_groups), "✅", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Documents", f"{n_docs:,}", "📄", self.theme_name))
        
        # Calculate average group size
        avg_size = sum(len(g.df) for g in bib_group.groups.values()) / n_groups if n_groups > 0 else 0
        grid.add_card(StatsCard(grid, "Avg Size", f"{avg_size:.0f}", "📊", self.theme_name))
        
        # Safety check for results_content
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
        except tk.TclError:
            return
        
        # Success message
        tk.Label(
            self.results_tab,
            text="✨ Groups created successfully!",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_card"],
            fg=self.theme.get("accent_success", self.theme.get("success", "#22c55e")),
        ).pack(fill=tk.X, pady=8)
        
        # Group details table
        tk.Label(
            self.results_tab,
            text="📋 Group Details",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4))
        
        details_data = []
        for group_name, group_bib in bib_group.groups.items():
            details_data.append({
                "Group": group_name,
                "Documents": len(group_bib.df),
                "Color": bib_group.group_colors.get(group_name, "#808080"),
            })
        
        details_df = pd.DataFrame(details_data)
        
        table = DataTable(self.results_tab, theme=self.theme_name, height=min(12, n_groups + 2))
        table.pack(fill=tk.BOTH, expand=True, pady=8)
        table.set_data(details_df)
        
        # Emit event to notify app
        event_bus.emit("group_created", {"bib_group": bib_group})
        event_bus.emit("group_created", {"bib_group": bib_group})
        
        # Next steps
        next_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        next_frame.pack(fill=tk.X, pady=16)
        
        tk.Label(
            next_frame,
            text="🚀 Next Steps:",
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W)
        
        tk.Label(
            next_frame,
            text="• Group Counts - Count entities across groups\n"
                 "• Group Stats - Compute performance statistics\n"
                 "• Group Compare - Compare variables and overlaps\n"
                 "• Group Visualizations - Create comparison plots",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
GROUP SETUP
═══════════

Define comparison groups for analysis.

GROUP DEFINITION METHODS
────────────────────────
• Keyword matching
  Terms in titles/abstracts/keywords
  
• Field value matching
  Exact match on any field
  
• Year range
  Time period filtering
  
• Boolean logic
  Combine conditions

KEYWORD SYNTAX
──────────────
Simple: climate
Phrase: "machine learning"
OR: climate OR environment
AND: climate AND policy
NOT: energy NOT nuclear

GROUP PROPERTIES
────────────────
• Name: Descriptive label
• Definition: Criteria
• Color: For visualizations
• Size: Document count

VALIDATION
──────────
• Preview matches
• Check overlaps
• Verify sizes
• Test criteria

BEST PRACTICES
──────────────
• Use meaningful names
• Document criteria
• Check group sizes
• Save definitions
"""
        text_widget = tk.Text(
            parent, font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            wrap=tk.WORD, padx=16, pady=16, relief=tk.FLAT,
        )
        text_widget.insert("1.0", info_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Add right-click context menu for copy
        def show_context_menu(event):
            menu = tk.Menu(text_widget, tearoff=0)
            menu.add_command(label="Copy Selected", command=lambda: copy_selected(text_widget))
            menu.add_command(label="Copy All", command=lambda: copy_all(text_widget))
            menu.tk_popup(event.x_root, event.y_root)
        
        def copy_selected(widget):
            try:
                widget.config(state=tk.NORMAL)
                selected = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
                widget.config(state=tk.DISABLED)
                widget.clipboard_clear()
                widget.clipboard_append(selected)
            except tk.TclError:
                pass
        
        def copy_all(widget):
            widget.config(state=tk.NORMAL)
            content = widget.get("1.0", tk.END)
            widget.config(state=tk.DISABLED)
            widget.clipboard_clear()
            widget.clipboard_append(content.strip())
        
        text_widget.bind("<Button-3>", show_context_menu)
        text_widget.bind("<Control-c>", lambda e: copy_selected(text_widget))

    def _show_no_data_message(self):
        """Show message when no data is loaded."""
        tk.Label(
            self.options_content,
            text="📂 Please load a dataset first\n\n"
                 "Go to DATA → Load Dataset to import your bibliometric data.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)
