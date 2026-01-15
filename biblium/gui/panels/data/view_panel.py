# -*- coding: utf-8 -*-
"""
View Data Panel
===============
Panel for viewing and exploring the dataset.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, StatsCard, CardGrid, ScrollableStatsRow
from biblium.gui.widgets.buttons import ThemedButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledEntry, DualListSelector
from biblium.gui.widgets.tables import DataTable

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class ViewDataPanel(BasePanel):
    """Panel for viewing the dataset."""
    
    title = "View Data"
    icon = "📋"
    description = "Browse and explore your dataset"
    requires_data = True
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Column Selection Card - Two columns: Shown (green) / Hidden (red)
        col_card = Card(self.options_content, title="📋 Column Visibility", theme=self.theme_name)
        col_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Use DualListSelector with Shown on left (green) and Hidden on right (red)
        self.column_selector = DualListSelector(
            col_card.content,
            available_label="Shown",  # Green - columns to display
            selected_label="Hidden",   # Red - columns to hide
            height=12,
            theme=self.theme_name,
        )
        self.column_selector.pack(fill=tk.BOTH, expand=True, pady=4)
        
        # Row Options Card
        row_card = Card(self.options_content, title="📊 Rows", theme=self.theme_name)
        row_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.max_rows_combo = LabeledCombobox(
            row_card.content,
            label="Show rows:",
            values=["100", "500", "1000", "5000", "All"],
            default="500",
            theme=self.theme_name,
            label_width=12,
        )
        self.max_rows_combo.pack(fill=tk.X, pady=4)
        
        # Quick Filter
        filter_card = Card(self.options_content, title="🔍 Quick Filter", theme=self.theme_name)
        filter_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.filter_col = LabeledCombobox(
            filter_card.content,
            label="Column:",
            values=[],
            theme=self.theme_name,
            label_width=12,
        )
        self.filter_col.pack(fill=tk.X, pady=4)
        
        self.filter_value = LabeledEntry(
            filter_card.content,
            label="Contains:",
            placeholder="Enter filter text...",
            theme=self.theme_name,
            label_width=12,
        )
        self.filter_value.pack(fill=tk.X, pady=4)
        
        # Apply Button
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ThemedButton(
            btn_frame, text="▶ Apply & Refresh", style="primary",
            command=self._refresh_view, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        # Populate columns when data is available
        if self.bib is not None:
            self._populate_columns()
    
    def _create_results(self):
        """Create the results panel with data table."""
        # Results card
        self.results_card = tk.Frame(
            self.results_frame,
            bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Header with stats
        self.header_frame = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        self.header_frame.pack(fill=tk.X, padx=12, pady=8)
        
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        # Table
        self.table_frame = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        self.table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        if self.bib is not None:
            self._show_data()
        else:
            tk.Label(
                self.table_frame,
                text="Load a dataset to view data",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True)
    
    def _populate_columns(self):
        """Populate column selector - all columns shown by default."""
        if not hasattr(self, 'column_selector'):
            return
            
        self.column_selector.clear()
        
        if self.bib is not None and hasattr(self.bib, 'df'):
            columns = list(self.bib.df.columns)
            
            # All columns shown by default (in "available" which is labeled "Shown")
            self.column_selector.set_available_items(columns)
            
            # Hidden list starts empty
            self.column_selector.set_selected_items([])
            
            # Update filter column dropdown
            self.filter_col.set_values(columns)
    
    def _get_selected_columns(self):
        """Get list of columns to display (from Shown list)."""
        return self.column_selector.get_available()
    
    def _refresh_view(self):
        """Refresh the data view."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_data()
    
    def _show_data(self):
        """Display the data."""
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        for widget in self.header_frame.winfo_children():
            widget.destroy()
        
        if self.bib is None or not hasattr(self.bib, 'df'):
            return
        
        df = self.bib.df
        
        # Apply column filter - show only columns in "Shown" list
        columns = self._get_selected_columns()
        if columns:
            # Filter to only existing columns
            valid_columns = [c for c in columns if c in df.columns]
            if valid_columns:
                df = df[valid_columns]
        
        # Apply row filter
        filter_col = self.filter_col.get()
        filter_val = self.filter_value.get()
        if filter_col and filter_val and filter_col in df.columns:
            mask = df[filter_col].astype(str).str.contains(filter_val, case=False, na=False)
            df = df[mask]
        
        # Row limit
        max_rows_str = self.max_rows_combo.get()
        max_rows = None if max_rows_str == "All" else int(max_rows_str)
        
        # Stats cards - scrollable row
        grid = ScrollableStatsRow(self.header_frame, theme=self.theme_name)
        grid.pack(fill=tk.X)
        
        grid.add_card(StatsCard(grid.inner_frame, "Rows", f"{len(df):,}", "📊", self.theme_name))
        grid.add_card(StatsCard(grid.inner_frame, "Columns", f"{len(df.columns)}", "📋", self.theme_name))
        
        hidden_count = len(self.column_selector.get_selected())
        if hidden_count > 0:
            grid.add_card(StatsCard(grid.inner_frame, "Hidden Cols", f"{hidden_count}", "🙈", self.theme_name))
        
        if len(df) < len(self.bib.df):
            grid.add_card(StatsCard(grid.inner_frame, "Filtered", f"{len(self.bib.df) - len(df):,} rows", "🔍", self.theme_name))
        
        # Table
        table = DataTable(
            self.table_frame,
            theme=self.theme_name,
            max_rows=max_rows or 1000,
        )
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)
    
    def _on_dataset_loaded(self, data):
        """Handle dataset loaded event."""
        super()._on_dataset_loaded(data)
        # Schedule for main thread
        self.after(100, self._populate_columns)
        self.after(200, self._show_data)
