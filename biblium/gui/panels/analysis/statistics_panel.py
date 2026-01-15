# -*- coding: utf-8 -*-
"""
Statistics Panel
================
Statistical analysis with scatter plots and bar charts for bibliometric data.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Dict, List, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme, ENTITY_TYPES
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox, LabeledTextArea, LabeledEntry
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Statistics analysis types - maps to biblium count methods and result attributes
STATS_TYPES = {
    # Primary entities (most commonly used)
    "authors": {
        "label": "Authors",
        "count_method": "count_authors",
        "count_attr": "authors_counts_df",
        "stats_attr": "authors_stats_df",
        "stats_method": "get_authors_stats",
        "name_col": "Author",
    },
    "sources": {
        "label": "Sources",
        "count_method": "count_sources",
        "count_attr": "sources_counts_df",
        "stats_attr": "sources_stats_df",
        "stats_method": "get_sources_stats",
        "name_col": "Source",
    },
    "countries": {
        "label": "Countries",
        "count_method": "count_all_countries",  # Changed from count_ca_countries
        "count_attr": "all_countries_counts_df",  # Changed from CA_countries_counts_df
        "stats_attr": "all_countries_stats_df",  # Changed from ca_countries_stats_df
        "stats_method": "get_all_countries_stats",  # Changed from get_ca_countries_stats
        "name_col": "Country",
    },
    "affiliations": {
        "label": "Affiliations",
        "count_method": "count_affiliations",
        "count_attr": "affiliations_counts_df",
        "stats_attr": "affiliations_stats_df",
        "stats_method": "get_affiliations_stats",
        "name_col": "Affiliation",
    },
    "keywords": {
        "label": "Author Keywords",
        "count_method": "count_author_keywords",
        "count_attr": "author_keywords_counts_df",
        "stats_attr": "author_keywords_stats_df",
        "stats_method": "get_author_keywords_stats",
        "name_col": "Keyword",
    },
    # Secondary entities
    "index_keywords": {
        "label": "Index Keywords",
        "count_method": "count_index_keywords",
        "count_attr": "index_keywords_counts_df",
        "stats_attr": "index_keywords_stats_df",
        "stats_method": "get_index_keywords_stats",
        "name_col": "Keyword",
    },
    "both_keywords": {
        "label": "All Keywords (Author + Index)",
        "count_method": "count_both_keywords",
        "count_attr": "author_and_index_keywords_counts_df",
        "stats_attr": "author_and_index_keywords_stats_df",
        "stats_method": "get_author_and_index_keywords_stats",
        "name_col": "Keyword",
    },
    "references": {
        "label": "References",
        "count_method": "count_references",
        "count_attr": "references_counts_df",
        "stats_attr": "references_stats_df",
        "stats_method": "get_references_stats",
        "name_col": "Reference",
    },
    "document_types": {
        "label": "Document Types",
        "count_method": "count_document_types",
        "count_attr": "document_type_counts_df",
        "stats_attr": "document_type_stats_df",
        "stats_method": "get_document_type_stats",
        "name_col": "Document Type",
    },
    # Subject classification entities
    "fields": {
        "label": "Subject Fields",
        "count_method": "count_fields",
        "count_attr": "fields_counts_df",
        "stats_attr": "fields_stats_df",
        "stats_method": "get_fields_stats",
        "name_col": "Field",
    },
    "areas": {
        "label": "Subject Areas",
        "count_method": "count_areas",
        "count_attr": "areas_counts_df",
        "stats_attr": "areas_stats_df",
        "stats_method": "get_areas_stats",
        "name_col": "Area",
    },
    "sciences": {
        "label": "Sciences",
        "count_method": "count_sciences",
        "count_attr": "sciences_counts_df",
        "stats_attr": "sciences_stats_df",
        "stats_method": "get_sciences_stats",
        "name_col": "Science",
    },
    # N-gram entities
    "ngrams_title": {
        "label": "N-grams (Title)",
        "count_method": "count_ngrams_title",
        "count_attr": "words_tit_counts_df",
        "stats_attr": "ngrams_title_stats_df",
        "stats_method": "get_ngrams_title_stats",
        "name_col": "N-gram",
    },
    "ngrams_abstract": {
        "label": "N-grams (Abstract)",
        "count_method": "count_ngrams_abstract",
        "count_attr": "words_abs_counts_df",
        "stats_attr": "ngrams_abstract_stats_df",
        "stats_method": "get_ngrams_abstract_stats",
        "name_col": "N-gram",
    },
    "ngrams_combined": {
        "label": "N-grams (Combined Text)",
        "count_method": "count_ngrams_combined_text",
        "count_attr": "words_comb_counts_df",
        "stats_attr": "ngrams_combined_text_stats_df",
        "stats_method": "get_ngrams_combined_text_stats",
        "name_col": "N-gram",
    },
}


class StatisticsPanel(BasePanel):
    """Panel for statistical analysis with scatter plots and bar charts."""
    
    title = "Statistics"
    icon = "📉"
    description = "Statistical analysis with scatter plots and bar charts"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._stats_df = None
        self._tab_spinners = []
        
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_statistics  # Set primary action for toolbar Run button
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Entity Selection Card
        entity_card = Card(self.options_content, title="📊 Entity Type", theme=self.theme_name)
        entity_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.entity_combo = LabeledCombobox(
            entity_card.content,
            label="Analyze:",
            values=[v["label"] for v in STATS_TYPES.values()],
            default="Authors",
            theme=self.theme_name,
            label_width=12,
        )
        self.entity_combo.pack(fill=tk.X, pady=4)
        
        # Options Card
        options_card = Card(self.options_content, title="⚙️ Options", theme=self.theme_name)
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.top_n_spin = LabeledSpinbox(
            options_card.content,
            label="Top N:",
            from_=10,
            to=500,
            default=50,
            theme=self.theme_name,
            label_width=12,
        )
        self.top_n_spin.pack(fill=tk.X, pady=4)
        
        # Computation depth for indicators
        self.depth_combo = LabeledCombobox(
            options_card.content,
            label="Depth:",
            values=["Core (H-index, citations)", "Extended (+G-index, C-index)", "Full (+A,R,W,T,Pi,HG,Chi indices)"],
            default="Core (H-index, citations)",
            theme=self.theme_name,
            label_width=12,
        )
        self.depth_combo.pack(fill=tk.X, pady=4)
        
        # Entity Filtering Card (Collapsible)
        filter_card = CollapsibleCard(
            self.options_content, 
            title="🎯 Entity Filtering", 
            theme=self.theme_name,
            collapsed=True
        )
        filter_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Include specific entities - with label and load button
        include_header = tk.Frame(filter_card.content, bg=self.theme["bg_card"])
        include_header.pack(fill=tk.X, pady=(4, 0))
        
        tk.Label(
            include_header,
            text="Include only (one per line):",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        load_include_btn = tk.Button(
            include_header,
            text="📂 Load",
            font=FONTS.get_font("small"),
            command=self._load_include_list,
            relief=tk.FLAT,
            bg=self.theme["bg_card"],
            fg=self.theme["accent_primary"],
            cursor="hand2",
        )
        load_include_btn.pack(side=tk.RIGHT)
        
        self.include_entities = LabeledTextArea(
            filter_card.content,
            label="",  # Label handled above
            height=3,
            theme=self.theme_name,
        )
        self.include_entities.pack(fill=tk.X, pady=(0, 4))
        # Hide the empty label
        self.include_entities.label.pack_forget()
        
        # Exclude specific entities - with label and load button
        exclude_header = tk.Frame(filter_card.content, bg=self.theme["bg_card"])
        exclude_header.pack(fill=tk.X, pady=(4, 0))
        
        tk.Label(
            exclude_header,
            text="Exclude (one per line):",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        load_exclude_btn = tk.Button(
            exclude_header,
            text="📂 Load",
            font=FONTS.get_font("small"),
            command=self._load_exclude_list,
            relief=tk.FLAT,
            bg=self.theme["bg_card"],
            fg=self.theme["accent_primary"],
            cursor="hand2",
        )
        load_exclude_btn.pack(side=tk.RIGHT)
        
        self.exclude_entities = LabeledTextArea(
            filter_card.content,
            label="",  # Label handled above
            height=3,
            theme=self.theme_name,
        )
        self.exclude_entities.pack(fill=tk.X, pady=(0, 4))
        # Hide the empty label
        self.exclude_entities.label.pack_forget()
        
        # Regex include pattern
        self.regex_include = LabeledEntry(
            filter_card.content,
            label="Regex include:",
            placeholder="e.g., ^Smith|^Jones (regex pattern)",
            theme=self.theme_name,
            label_width=12,
        )
        self.regex_include.pack(fill=tk.X, pady=4)
        
        # Regex exclude pattern
        self.regex_exclude = LabeledEntry(
            filter_card.content,
            label="Regex exclude:",
            placeholder="e.g., Unknown|Anonymous",
            theme=self.theme_name,
            label_width=12,
        )
        self.regex_exclude.pack(fill=tk.X, pady=4)
        
        # Max items when using regex
        self.max_items_spin = LabeledSpinbox(
            filter_card.content,
            label="Max items:",
            from_=0,
            to=1000,
            default=0,
            theme=self.theme_name,
            label_width=12,
        )
        self.max_items_spin.pack(fill=tk.X, pady=4)
        
        # Skip header row option for file loading
        self.skip_header_cb = LabeledCheckbox(
            filter_card.content,
            label="Skip header row when loading files",
            default=True,
            theme=self.theme_name,
        )
        self.skip_header_cb.pack(fill=tk.X, pady=4)
        
        # Help text
        help_label = tk.Label(
            filter_card.content,
            text="Note: 'Include only' overrides Top N. Max items=0 means no limit.\nLoad from .txt, .csv, or .xlsx files.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=250,
            justify=tk.LEFT,
        )
        help_label.pack(fill=tk.X, pady=(4, 0))
        
        # Geographic Filtering Card (for Countries only)
        self.geo_filter_card = CollapsibleCard(
            self.options_content, 
            title="🌍 Geographic Filter (Countries)", 
            theme=self.theme_name,
            collapsed=True
        )
        self.geo_filter_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Continent filter
        continent_options = ["All", "Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
        self.continent_combo = LabeledCombobox(
            self.geo_filter_card.content,
            label="Continent:",
            values=continent_options,
            default="All",
            theme=self.theme_name,
            label_width=12,
        )
        self.continent_combo.pack(fill=tk.X, pady=4)
        
        # EU filter
        self.eu_filter_combo = LabeledCombobox(
            self.geo_filter_card.content,
            label="EU Status:",
            values=["All countries", "EU only", "Non-EU only"],
            default="All countries",
            theme=self.theme_name,
            label_width=12,
        )
        self.eu_filter_combo.pack(fill=tk.X, pady=4)
        
        # Note about geographic filter
        geo_note = tk.Label(
            self.geo_filter_card.content,
            text="Note: Geographic filters only apply when analyzing Countries.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=250,
            justify=tk.LEFT,
        )
        geo_note.pack(fill=tk.X, pady=(4, 0))
        
        # N-gram Options Card
        self.ngram_card = CollapsibleCard(
            self.options_content, 
            title="📝 N-gram Options", 
            theme=self.theme_name,
            collapsed=True
        )
        self.ngram_card.pack(fill=tk.X, padx=8, pady=8)
        
        # N-gram range min
        self.ngram_min_spin = LabeledSpinbox(
            self.ngram_card.content,
            label="Min n-gram:",
            from_=1,
            to=5,
            default=1,
            theme=self.theme_name,
            label_width=12,
        )
        self.ngram_min_spin.pack(fill=tk.X, pady=4)
        
        # N-gram range max
        self.ngram_max_spin = LabeledSpinbox(
            self.ngram_card.content,
            label="Max n-gram:",
            from_=1,
            to=5,
            default=2,
            theme=self.theme_name,
            label_width=12,
        )
        self.ngram_max_spin.pack(fill=tk.X, pady=4)
        
        # Note about n-gram options
        ngram_note = tk.Label(
            self.ngram_card.content,
            text="N-gram range: 1=unigrams, 2=bigrams, 3=trigrams.\nExample: (1,2) extracts both single words and two-word phrases.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=250,
            justify=tk.LEFT,
        )
        ngram_note.pack(fill=tk.X, pady=(4, 0))
        
        # Scatter Plot Options Card
        scatter_card = Card(self.options_content, title="🔵 Scatter Plot", theme=self.theme_name)
        scatter_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Metric options for scatter - use biblium column names (with variations handled by _find_column)
        # Organized by computation depth
        axis_options = [
            # Core indices
            "Number of documents", "Total citations", "H-index", "Average year",
            # Extended indices
            "G-index", "First year", "Median year", "Last year",
            # Full mode indices
            "A-index", "R-index", "H(2)-index", "W-index", "T-index", "Pi-index",
            "Gini index", "HG-index", "Chi-index", "Interdisciplinarity",
            # Additional metrics
            "Collaboration index", "Documents per active year", "Citations per active year",
            "Number of cited documents",
            # Computed/derived
            "Rank", "Percentage of documents",
        ]
        metric_options = ["None"] + axis_options
        
        self.x_axis_combo = LabeledCombobox(
            scatter_card.content,
            label="X-Axis:",
            values=axis_options,
            default="Number of documents",
            theme=self.theme_name,
            label_width=12,
        )
        self.x_axis_combo.pack(fill=tk.X, pady=4)
        
        self.y_axis_combo = LabeledCombobox(
            scatter_card.content,
            label="Y-Axis:",
            values=axis_options,
            default="Total citations",
            theme=self.theme_name,
            label_width=12,
        )
        self.y_axis_combo.pack(fill=tk.X, pady=4)
        
        self.scatter_size_combo = LabeledCombobox(
            scatter_card.content,
            label="Size By:",
            values=metric_options,
            default="H-index",
            theme=self.theme_name,
            label_width=12,
        )
        self.scatter_size_combo.pack(fill=tk.X, pady=4)
        
        self.scatter_color_combo = LabeledCombobox(
            scatter_card.content,
            label="Color By:",
            values=metric_options,
            default="Average year",
            theme=self.theme_name,
            label_width=12,
        )
        self.scatter_color_combo.pack(fill=tk.X, pady=4)
        
        self.scatter_label_combo = LabeledCombobox(
            scatter_card.content,
            label="Labels:",
            values=["None", "Name", "Name (top 10)"],
            default="Name",
            theme=self.theme_name,
            label_width=12,
        )
        self.scatter_label_combo.pack(fill=tk.X, pady=4)
        
        # Log scale checkboxes
        log_frame = tk.Frame(scatter_card.content, bg=self.theme["bg_card"])
        log_frame.pack(fill=tk.X, pady=4)
        
        self.log_x_cb = LabeledCheckbox(
            log_frame,
            label="Log X-axis",
            default=True,
            theme=self.theme_name,
        )
        self.log_x_cb.pack(side=tk.LEFT, padx=(0, 16))
        
        self.log_y_cb = LabeledCheckbox(
            log_frame,
            label="Log Y-axis",
            default=True,
            theme=self.theme_name,
        )
        self.log_y_cb.pack(side=tk.LEFT)
        
        # Bar Chart Options Card
        bar_card = Card(self.options_content, title="📊 Bar Chart", theme=self.theme_name)
        bar_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.bar_metric_combo = LabeledCombobox(
            bar_card.content,
            label="Bar Height:",
            values=["Number of documents", "Total citations", "H-index", "Average year", "Rank", "Percentage of documents"],
            default="Number of documents",
            theme=self.theme_name,
            label_width=12,
        )
        self.bar_metric_combo.pack(fill=tk.X, pady=4)
        
        self.bar_color_combo = LabeledCombobox(
            bar_card.content,
            label="Color By:",
            values=["None", "Rank", "Number of documents", "Average year", "H-index", "Total citations", "Percentage of documents"],
            default="Average year",
            theme=self.theme_name,
            label_width=12,
        )
        self.bar_color_combo.pack(fill=tk.X, pady=4)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame,
            text="Compute Statistics",
            icon="📉",
            command=self._run_statistics,
            theme=self.theme_name,
        ).pack(fill=tk.X)
        
        # Bind Enter key to run statistics
        self._bind_enter_key()
    
    def _bind_enter_key(self):
        """Bind Enter key to run statistics on input widgets."""
        # Bind to spinbox
        if hasattr(self, 'top_n_spin') and hasattr(self.top_n_spin, 'spinbox'):
            self.top_n_spin.spinbox.bind('<Return>', lambda e: self._run_statistics())
        
        # Bind to the whole panel
        self.bind_all('<Return>', self._on_enter_key)
    
    def _on_enter_key(self, event):
        """Handle Enter key press - run statistics if focus is in this panel."""
        # Check if the event widget is a child of this panel
        try:
            widget = event.widget
            # Walk up the widget tree to see if it's in our options panel
            while widget:
                if widget == self.options_content or widget == self:
                    self._run_statistics()
                    return "break"
                widget = widget.master
        except:
            pass
    
    def _create_results(self):
        """Create the results panel with tabs."""
        # Results container
        self.results_card = tk.Frame(
            self.results_frame,
            bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Header
        header = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header,
            text="Results",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            padx=12,
            pady=8,
        ).pack(side=tk.LEFT)
        
        # Export button
        ThemedButton(
            header,
            text="📥 Export",
            style="ghost",
            size="small",
            command=self._export_results,
            theme=self.theme_name,
        ).pack(side=tk.RIGHT, padx=8, pady=4)
        
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        # Notebook for tabs
        self.results_notebook = ttk.Notebook(self.results_card)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Table tab
        self.table_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.table_frame, text="  📊 Table  ")
        
        # Bar Chart tab
        self.bar_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.bar_frame, text="  📊 Bar Chart  ")
        
        # Scatter Plot tab
        self.scatter_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.scatter_frame, text="  🔵 Scatter Plot  ")

        

        # Info tab

        info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])

        self.results_notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        # Initial message
        self._show_table_message("Select an entity type and click 'Compute Statistics'")
    
    def _show_table_message(self, message: str):
        """Show message in table tab."""
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.table_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _show_bar_message(self, message: str):
        """Show message in bar chart tab."""
        for widget in self.bar_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.bar_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _show_scatter_message(self, message: str):
        """Show message in scatter tab."""
        for widget in self.scatter_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.scatter_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _get_geographic_filter_items(self, continent: str, eu_status: str) -> tuple:
        """
        Get lists of countries to include/exclude based on geographic filters.
        
        Parameters
        ----------
        continent : str
            Continent name or "All"
        eu_status : str
            "All countries", "EU only", or "Non-EU only"
        
        Returns
        -------
        tuple
            (items_to_include, items_to_exclude) - either can be None or list
        """
        try:
            from biblium import utilsbib
            
            # Load country data
            utilsbib._load_countries_data()
            continent_dct = utilsbib._continent_dct
            eu_countries = utilsbib._eu_countries
            all_countries = utilsbib._l_countries
            
            include_items = None
            exclude_items = None
            
            # Filter by continent
            if continent != "All" and continent_dct:
                continent_countries = [c for c, cont in continent_dct.items() if cont == continent]
                include_items = continent_countries
            
            # Filter by EU status
            if eu_status == "EU only" and eu_countries:
                if include_items:
                    # Intersection with continent filter
                    include_items = [c for c in include_items if c in eu_countries]
                else:
                    include_items = eu_countries
            elif eu_status == "Non-EU only" and eu_countries:
                if include_items:
                    # Remove EU countries from continent filter
                    include_items = [c for c in include_items if c not in eu_countries]
                else:
                    # All non-EU countries
                    include_items = [c for c in all_countries if c not in eu_countries]
            
            return (include_items, exclude_items)
            
        except Exception as e:
            print(f"DEBUG: Geographic filter error: {e}")
            return (None, None)
    
    def _load_include_list(self):
        """Load include list from file."""
        items = self._load_entity_list_from_file("Select Include List File")
        if items:
            # Get current text and append or replace
            current = self.include_entities.get().strip()
            if current:
                # Ask if user wants to append or replace
                from tkinter import messagebox
                result = messagebox.askyesnocancel(
                    "Load List",
                    f"Found {len(items)} items.\n\nYes = Replace current list\nNo = Append to current list\nCancel = Abort"
                )
                if result is None:  # Cancel
                    return
                elif result:  # Yes - Replace
                    self.include_entities.text.delete("1.0", tk.END)
                else:  # No - Append
                    self.include_entities.text.insert(tk.END, "\n")
            
            self.include_entities.text.insert(tk.END, "\n".join(items))
    
    def _load_exclude_list(self):
        """Load exclude list from file."""
        items = self._load_entity_list_from_file("Select Exclude List File")
        if items:
            # Get current text and append or replace
            current = self.exclude_entities.get().strip()
            if current:
                # Ask if user wants to append or replace
                from tkinter import messagebox
                result = messagebox.askyesnocancel(
                    "Load List",
                    f"Found {len(items)} items.\n\nYes = Replace current list\nNo = Append to current list\nCancel = Abort"
                )
                if result is None:  # Cancel
                    return
                elif result:  # Yes - Replace
                    self.exclude_entities.text.delete("1.0", tk.END)
                else:  # No - Append
                    self.exclude_entities.text.insert(tk.END, "\n")
            
            self.exclude_entities.text.insert(tk.END, "\n".join(items))
    
    def _load_entity_list_from_file(self, title: str) -> list:
        """Load entity list from txt, csv, or xlsx file."""
        from tkinter import filedialog
        
        filepath = filedialog.askopenfilename(
            title=title,
            filetypes=[
                ("All supported", "*.txt *.csv *.xlsx *.xls"),
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ]
        )
        
        if not filepath:
            return []
        
        # Check if we should skip header row
        skip_header = self.skip_header_cb.get() if hasattr(self, 'skip_header_cb') else True
        
        items = []
        try:
            ext = filepath.lower().split(".")[-1]
            
            if ext == "txt":
                # Plain text - one item per line
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if skip_header and len(lines) > 1:
                        items = lines[1:]
                    else:
                        items = lines
            
            elif ext == "csv":
                # CSV - use first column
                import pandas as pd
                header_param = 0 if skip_header else None
                df = pd.read_csv(filepath, header=header_param)
                items = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
                items = [item for item in items if item]
            
            elif ext in ["xlsx", "xls"]:
                # Excel - use first column of first sheet
                import pandas as pd
                header_param = 0 if skip_header else None
                df = pd.read_excel(filepath, header=header_param)
                items = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
                items = [item for item in items if item]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_items = []
            for item in items:
                if item.lower() not in seen:
                    seen.add(item.lower())
                    unique_items.append(item)
            items = unique_items
            
            if items:
                from tkinter import messagebox
                messagebox.showinfo("Loaded", f"Loaded {len(items)} unique items from file.")
            else:
                from tkinter import messagebox
                messagebox.showwarning("Empty", "No items found in file.")
                
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            return []
        
        return items
    
    def _get_entity_key(self) -> str:
        """Get selected entity key."""
        label = self.entity_combo.get()
        for key, config in STATS_TYPES.items():
            if config["label"] == label:
                return key
        return "authors"
    
    def _show_loading_in_tabs(self):
        """Show loading in all tabs."""
        self._stop_tab_spinners()
        
        for frame in [self.table_frame, self.bar_frame, self.scatter_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
            
            from biblium.gui.widgets.progress import LoadingSpinner
            spinner = LoadingSpinner(frame, theme=self.theme_name)
            spinner.pack(expand=True)
            spinner.start()
            self._tab_spinners.append(spinner)
    
    def _stop_tab_spinners(self):
        """Stop all tab spinners."""
        for spinner in self._tab_spinners:
            try:
                spinner.stop()
            except:
                pass
        self._tab_spinners.clear()
    
    def _run_statistics(self):
        """Run statistical analysis using biblium's stats_all()."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        entity_key = self._get_entity_key()
        self._show_loading_in_tabs()
        
        # Map depth selection to mode parameter
        depth_selection = self.depth_combo.get()
        if "Full" in depth_selection:
            mode = "full"
        elif "Extended" in depth_selection:
            mode = "extended"
        else:
            mode = "core"
        
        # Parse entity filtering options
        include_text = self.include_entities.get().strip()
        items_of_interest = None
        if include_text:
            items_of_interest = [line.strip() for line in include_text.split("\n") if line.strip()]
        
        exclude_text = self.exclude_entities.get().strip()
        exclude_items = None
        if exclude_text:
            exclude_items = [line.strip() for line in exclude_text.split("\n") if line.strip()]
        
        regex_include = self.regex_include.get().strip() or None
        regex_exclude = self.regex_exclude.get().strip() or None
        max_items = self.max_items_spin.get()
        
        # Get geographic filter options (for countries)
        continent_filter = self.continent_combo.get() if hasattr(self, 'continent_combo') else "All"
        eu_filter = self.eu_filter_combo.get() if hasattr(self, 'eu_filter_combo') else "All countries"
        
        # Apply geographic filters for countries - add to items_of_interest or exclude_items
        if entity_key == "countries" and (continent_filter != "All" or eu_filter != "All countries"):
            geo_items, geo_exclude = self._get_geographic_filter_items(continent_filter, eu_filter)
            
            # Merge with existing filters
            if geo_items:
                if items_of_interest:
                    # Intersection: keep only items in both lists
                    items_of_interest = [i for i in items_of_interest if i in geo_items]
                else:
                    items_of_interest = geo_items
            
            if geo_exclude:
                if exclude_items:
                    exclude_items = list(set(exclude_items + geo_exclude))
                else:
                    exclude_items = geo_exclude
        
        # Get n-gram options (for ngram entities)
        ngram_min = self.ngram_min_spin.get() if hasattr(self, 'ngram_min_spin') else 1
        ngram_max = self.ngram_max_spin.get() if hasattr(self, 'ngram_max_spin') else 2
        ngram_range = (ngram_min, ngram_max)
        
        options = {
            "top_n": self.top_n_spin.get(),
            "mode": mode,
            "items_of_interest": items_of_interest,
            "exclude_items": exclude_items,
            "regex_include": regex_include,
            "regex_exclude": regex_exclude,
            "max_items": max_items,
            "ngram_range": ngram_range,
            "x_axis": self.x_axis_combo.get(),
            "y_axis": self.y_axis_combo.get(),
            "scatter_size": self.scatter_size_combo.get(),
            "scatter_color": self.scatter_color_combo.get(),
            "scatter_label": self.scatter_label_combo.get(),
            "log_x": self.log_x_cb.get(),
            "log_y": self.log_y_cb.get(),
            "bar_metric": self.bar_metric_combo.get(),
            "bar_color": self.bar_color_combo.get(),
        }
        
        def do_stats():
            try:
                config = STATS_TYPES[entity_key]
                top_n = options["top_n"]
                mode = options.get("mode", "core")
                
                # Extract entity filtering options
                items_of_interest = options.get("items_of_interest")
                exclude_items = options.get("exclude_items")
                regex_include = options.get("regex_include")
                regex_exclude = options.get("regex_exclude")
                max_items = options.get("max_items", 0)
                
                # Get biblium's mapping for this entity type
                bib_mapping = None
                mapping_key = None
                if hasattr(self.bib, 'mapping'):
                    # Find the right mapping key
                    # Note: For countries, try 'all countries' first since 'CA countries' often fails
                    mapping_keys_map = {
                        "authors": "authors",
                        "sources": "sources", 
                        "countries": "all countries",  # Changed: use 'all countries' instead of 'CA countries'
                        "affiliations": "affiliations",
                        "keywords": "author keywords",
                    }
                    mapping_key = mapping_keys_map.get(entity_key)
                    if mapping_key and mapping_key in self.bib.mapping:
                        bib_mapping = self.bib.mapping[mapping_key]
                        print(f"DEBUG Statistics: Found biblium mapping for '{mapping_key}': {list(bib_mapping.keys())}")
                
                # Step 1: Call the counter method first (required by biblium)
                count_method = config.get("count_method")
                if bib_mapping and "counter" in bib_mapping:
                    count_method = bib_mapping["counter"]
                
                if count_method and hasattr(self.bib, count_method):
                    print(f"DEBUG Statistics: Calling counter {count_method}()")
                    try:
                        # Pass ngram_range for ngram methods
                        if entity_key.startswith("ngrams_"):
                            ngram_range = options.get("ngram_range", (1, 2))
                            print(f"DEBUG Statistics: Using ngram_range={ngram_range}")
                            getattr(self.bib, count_method)(ngram_range=ngram_range)
                        else:
                            getattr(self.bib, count_method)()
                    except Exception as e:
                        print(f"DEBUG Statistics: Counter {count_method} failed: {e}")
                
                # Step 2: Call the stats getter method
                stats_method = config.get("stats_method")
                if bib_mapping and "getter" in bib_mapping:
                    stats_method = bib_mapping["getter"]
                
                stats_result = None
                stats_generated = False
                
                if stats_method and hasattr(self.bib, stats_method):
                    try:
                        # Build kwargs for the stats method
                        stats_kwargs = {
                            "top_n": top_n,
                            "mode": mode,
                        }
                        
                        # Add filtering options if specified
                        if items_of_interest:
                            stats_kwargs["items_of_interest"] = items_of_interest
                        if exclude_items:
                            stats_kwargs["exclude_items"] = exclude_items
                        if regex_include:
                            stats_kwargs["regex_include"] = regex_include
                        if regex_exclude:
                            stats_kwargs["regex_exclude"] = regex_exclude
                        if max_items > 0:
                            stats_kwargs["max_items"] = max_items
                        
                        # For ngram entities, pass counts_df to prevent re-counting with default ngram_range
                        if entity_key.startswith("ngrams_"):
                            count_attr = config.get("count_attr")
                            if count_attr and hasattr(self.bib, count_attr):
                                counts_df = getattr(self.bib, count_attr)
                                if counts_df is not None and len(counts_df) > 0:
                                    stats_kwargs["counts_df"] = counts_df
                                    print(f"DEBUG Statistics: Passing pre-computed counts_df with {len(counts_df)} items")
                        
                        print(f"DEBUG Statistics: Calling {stats_method} with kwargs: {list(stats_kwargs.keys())}")
                        method = getattr(self.bib, stats_method)
                        stats_result = method(**stats_kwargs)
                        print(f"DEBUG Statistics: {stats_method} returned type: {type(stats_result)}")
                        if stats_result is not None and hasattr(stats_result, 'shape'):
                            print(f"DEBUG Statistics: {stats_method} returned DataFrame with shape: {stats_result.shape}")
                        stats_generated = True
                    except Exception as e:
                        import traceback
                        print(f"DEBUG Statistics: {stats_method} failed: {e}")
                        traceback.print_exc()
                        stats_generated = False
                
                if not stats_generated:
                    print(f"DEBUG Statistics: No stats generation method found, will use counts with computed stats")
                
                # Check what stats attributes were created
                stats_attrs = [a for a in dir(self.bib) if 'stats_df' in a]
                if stats_attrs:
                    print(f"DEBUG Statistics: Available stats_df attrs: {stats_attrs}")
                
                # Get the stats DataFrame - first try the returned result, then the attribute
                stats_attr = config.get("stats_attr")
                if bib_mapping and "stats_attr" in bib_mapping:
                    stats_attr = bib_mapping["stats_attr"]
                
                stats_df = None
                
                # First, check if the getter returned a valid DataFrame directly
                if stats_result is not None and hasattr(stats_result, 'shape') and len(stats_result) > 0:
                    stats_df = stats_result
                    print(f"DEBUG Statistics: Using returned DataFrame directly with shape={stats_df.shape}")
                else:
                    # Try the attribute
                    stats_df = getattr(self.bib, stats_attr, None) if stats_attr else None
                
                print(f"DEBUG Statistics: stats_attr={stats_attr}")
                if stats_df is not None and len(stats_df) > 0:
                    print(f"DEBUG Statistics: stats_df shape={stats_df.shape}")
                    print(f"DEBUG Statistics: stats_df columns={list(stats_df.columns[:10])}...")
                else:
                    print(f"DEBUG Statistics: stats_df empty or None")
                
                # If stats_df is empty or None, try to use counts_df and compute stats manually
                if stats_df is None or len(stats_df) == 0:
                    print(f"DEBUG Statistics: stats_df empty, falling back to counts_df + manual stats")
                    count_attr = config.get("count_attr")
                    count_method = config.get("count_method")
                    
                    # First try to get existing counts_df
                    count_df = getattr(self.bib, count_attr, None) if count_attr else None
                    print(f"DEBUG Statistics: Initial count_df from {count_attr}: {count_df is not None and len(count_df) if count_df is not None else 'None'}")
                    
                    # If counts_df is None or empty, call the count method
                    if (count_df is None or len(count_df) == 0) and count_method and hasattr(self.bib, count_method):
                        print(f"DEBUG Statistics: Calling {count_method}()")
                        try:
                            getattr(self.bib, count_method)()
                            count_df = getattr(self.bib, count_attr, None)
                            print(f"DEBUG Statistics: After {count_method}, count_df from {count_attr}: {count_df is not None and len(count_df) if count_df is not None else 'None'}")
                        except Exception as e:
                            print(f"DEBUG Statistics: {count_method} failed: {e}")
                    
                    # If still no count_df, try manual counting
                    if count_df is None or len(count_df) == 0:
                        count_df = self._manual_count_entities(entity_key, top_n)
                    
                    # IMPORTANT: Even if count_df has data, we need to compute full stats
                    # because biblium's stats methods returned empty for this entity type
                    if count_df is not None and len(count_df) > 0:
                        # Compute full statistics manually (H-index, citations, etc.)
                        stats_df = self._compute_entity_stats(count_df, entity_key, top_n)
                        print(f"DEBUG Statistics: Computed stats_df with shape={stats_df.shape}")
                        print(f"DEBUG Statistics: stats_df columns={list(stats_df.columns)}")
                
                if stats_df is None or len(stats_df) == 0:
                    # Provide specific guidance based on entity type
                    if entity_key == "countries":
                        raise ValueError(f"No country data available. The dataset may not have country information or the country column format may not be recognized.")
                    else:
                        raise ValueError(f"No statistics data available for {config['label']}")
                
                # Standardize column names for display
                stats_df = self._standardize_columns(stats_df, entity_key)
                
                self.after(0, lambda df=stats_df: self._on_stats_success(df, entity_key, options))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                error_msg = str(e)
                print(f"DEBUG Statistics ERROR: {error_msg}")
                print(f"DEBUG Statistics TRACEBACK:\n{tb}")
                self.after(0, lambda msg=error_msg, t=tb: self._on_stats_error(msg, t))
        
        threading.Thread(target=do_stats, daemon=True).start()
    
    def _standardize_columns(self, df: pd.DataFrame, entity_key: str) -> pd.DataFrame:
        """Standardize column names for consistent display."""
        # Create a copy
        df = df.copy()
        
        # Map common column variations to standard names
        column_map = {
            # Name columns - only rename the first matching one to avoid duplicates
            "Abbreviated Source Title": "Abbrev",
            # Stats columns - Keep "Number of documents" as standard
            # Don't rename "Number of documents" - keep original
            "Number of documents_stats": "Docs (stats)",  # Rename the duplicate to avoid conflict
            "Total citations": "Total citations",  # Keep as is
            "H-index": "H-index",  # Keep as is
            "Average year": "Avg Year",
            "G-index": "G-index",  # Keep as is
            "First year": "First Year",
            "Last year": "Last Year",
            "Median year": "Median Year",
            # Don't rename these - keep original biblium names
            # "Proportion of documents": "Prop Docs",
            # "Percentage of documents": "% Docs",
        }
        
        # Rename columns (non-Name columns first)
        rename_dict = {}
        for old_col, new_col in column_map.items():
            if old_col in df.columns:
                rename_dict[old_col] = new_col
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        # Handle Name column - only rename the first column to Name if needed
        # Avoid renaming multiple columns to Name
        name_candidates = ["Author", "Author(s) ID", "Source", "Country", "Affiliation", "Keyword"]
        name_found = False
        
        for col in df.columns:
            if col == "Name":
                name_found = True
                break
            if col in name_candidates and not name_found:
                # Rename this one to Name, drop others if they exist
                df = df.rename(columns={col: "Name"})
                name_found = True
                # Don't break - we continue to check for duplicates
            elif col in name_candidates and name_found:
                # Drop duplicate name columns
                df = df.drop(columns=[col])
        
        # Ensure Name column exists (use first column if not)
        if "Name" not in df.columns and len(df.columns) > 0:
            df = df.rename(columns={df.columns[0]: "Name"})
        
        return df
    
    def _on_stats_success(self, stats_df, entity_key: str, options: Dict):
        """Handle successful statistics computation."""
        self._stop_tab_spinners()
        self._stats_df = stats_df
        
        config = STATS_TYPES[entity_key]
        
        if stats_df is None or len(stats_df) == 0:
            self._show_table_message("No statistics could be computed.")
            self._show_bar_message("No data available.")
            self._show_scatter_message("No data available.")
            return
        
        # Show table
        self._show_stats_table(stats_df, options["top_n"])
        
        # Show bar chart
        self._show_bar_chart(
            stats_df,
            config["label"],
            options["bar_metric"],
            options["bar_color"],
            options["top_n"]
        )
        
        # Show scatter plot with all options
        self._show_scatter_plot(
            stats_df, 
            config["label"],
            options
        )
        
        # Emit event
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": f"Statistics: {config['label']}"})
    
    def _on_stats_error(self, error: str, traceback_str: str = ""):
        """Handle statistics error with user-friendly messages."""
        self._stop_tab_spinners()
        
        # Determine if this is an "unavailable" warning vs a real error
        is_unavailable = "No statistics data available" in error or "not available" in error.lower()
        
        if is_unavailable:
            # User-friendly warning for unavailable statistics
            warning_msg = f"⚠️ {error}\n\nThis statistic may not be supported for the current database or data format."
            self._show_table_warning(warning_msg)
            self._show_bar_message("Statistics not available for this entity type.")
            self._show_scatter_message("Statistics not available for this entity type.")
        else:
            # Real error
            self._show_table_message(f"Error: {error}")
            self._show_bar_message(f"Error: {error}")
            self._show_scatter_message(f"Error: {error}")
        
        print(f"Statistics Error: {error}")
        if traceback_str:
            print(traceback_str)
    
    def _show_table_warning(self, message: str):
        """Show a warning message in the table area."""
        self._safe_clear_frame(self.table_frame)
        
        warning_frame = tk.Frame(self.table_frame, bg=self.theme["bg_card"])
        warning_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Warning icon and message
        tk.Label(
            warning_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme.get("warning", "#E67E22"),
            wraplength=400,
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _show_stats_table(self, df, top_n: int):
        """Show statistics table."""
        self._safe_clear_frame(self.table_frame)
        
        if df is None or len(df) == 0:
            self._show_table_message("No statistics available.")
            return
        
        # Summary stats
        summary = tk.Frame(self.table_frame, bg=self.theme["bg_card"])
        summary.pack(fill=tk.X, pady=(0, 8))
        
        grid = CardGrid(summary, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X)
        
        grid.add_card(StatsCard(grid, "Entities", f"{len(df):,}", "📊", self.theme_name))
        
        # Find columns using flexible matching
        cit_col = self._find_column(df, "Total citations")
        if cit_col:
            total_cit = pd.to_numeric(df[cit_col], errors='coerce').sum()
            grid.add_card(StatsCard(grid, "Total Citations", f"{int(total_cit):,}", "📈", self.theme_name))
        
        h_col = self._find_column(df, "H-index")
        if h_col:
            max_h = pd.to_numeric(df[h_col], errors='coerce').max()
            grid.add_card(StatsCard(grid, "Max H-Index", f"{int(max_h)}", "🏆", self.theme_name))
        
        year_col = self._find_column(df, "Average year")
        if year_col:
            avg_year = pd.to_numeric(df[year_col], errors='coerce').mean()
            if not pd.isna(avg_year):
                grid.add_card(StatsCard(grid, "Avg Year", f"{avg_year:.1f}", "📅", self.theme_name))
        
        # Select key columns for display based on computation depth
        # Start with entity name columns and core indices
        display_cols = []
        
        # Entity name columns (always first)
        name_cols = ["Name", "Source", "Author", "Country", "Affiliation", "Keyword", "Author full names"]
        
        # Core indices (always shown)
        core_cols = ["Number of documents", "Total citations", "H-index", "Average year", "Rank"]
        
        # Extended indices
        extended_cols = ["G-index", "First year", "Median year", "Last year",
                        "C1", "C5", "C10", "C20", "C50", "C100"]
        
        # Full mode indices
        full_cols = ["A-index", "R-index", "H(2)-index", "W-index", "T-index", "Pi-index",
                    "Gini index", "HG-index", "Chi-index", "Interdisciplinarity",
                    "Number of cited documents", "Collaboration index",
                    "Documents per active year", "Citations per active year"]
        
        # Build key_cols based on what's available in the dataframe
        # This way we show all computed indices regardless of mode
        key_cols = name_cols + core_cols
        
        # Check if extended columns exist and add them
        for col in extended_cols:
            if self._find_column(df, col):
                key_cols.append(col)
        
        # Check if full columns exist and add them
        for col in full_cols:
            if self._find_column(df, col):
                key_cols.append(col)
        
        for key in key_cols:
            col = self._find_column(df, key)
            if col and col not in display_cols:
                display_cols.append(col)
        
        # Use display_cols or all columns if none matched
        if display_cols:
            display_df = df[display_cols].head(top_n)
        else:
            # Show all columns if no specific ones matched
            display_df = df.head(top_n)
        
        # Table
        table = DataTable(self.table_frame, theme=self.theme_name, max_rows=top_n)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(display_df)
    
    def _safe_clear_frame(self, frame):
        """Safely clear a frame, properly closing matplotlib figures."""
        try:
            for widget in frame.winfo_children():
                # Close matplotlib figures first
                if hasattr(widget, 'figure'):
                    try:
                        plt.close(widget.figure)
                    except:
                        pass
                # Destroy widget
                try:
                    widget.destroy()
                except:
                    pass
        except:
            pass
    
    def _show_bar_chart(self, df, entity_label: str, metric: str, color_by: str, top_n: int):
        """Show bar chart with optional color coding."""
        # Clean up existing plots safely
        self._safe_clear_frame(self.bar_frame)
        
        if df is None or len(df) == 0:
            self._show_bar_message("No data available.")
            return
        
        if not HAS_MATPLOTLIB:
            self._show_bar_message("Matplotlib required for plots.")
            return
        
        # Find metric column with fallback
        metric_col = self._find_column(df, metric)
        if not metric_col:
            # Try common fallbacks
            metric_col = self._find_column(df, "Documents") or self._find_column(df, "Number of documents")
            if not metric_col:
                # Use first numeric column
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    metric_col = numeric_cols[0]
        
        if not metric_col:
            self._show_bar_message(f"Could not find metric column: {metric}")
            return
        
        # Get data
        n_items = min(top_n, 20, len(df))
        plot_df = df.head(n_items).copy()
        
        # Get labels and values - handle potential duplicate columns
        name_col = plot_df.columns[0]
        # Use iloc to get first column as Series (avoids duplicate column issue)
        labels = plot_df.iloc[:, 0].astype(str).tolist()
        
        # Find metric column index
        metric_col_idx = list(plot_df.columns).index(metric_col) if metric_col in plot_df.columns else None
        if metric_col_idx is not None:
            values = pd.to_numeric(plot_df.iloc[:, metric_col_idx], errors='coerce').fillna(0).tolist()
        else:
            values = pd.to_numeric(plot_df[metric_col], errors='coerce').fillna(0).tolist()
        
        # Wrap labels
        import textwrap
        wrapped_labels = [textwrap.fill(str(l), width=20) for l in labels]
        
        # Reverse for proper display (highest at top)
        wrapped_labels = wrapped_labels[::-1]
        values = values[::-1]
        
        # Get colors - search in full df first, then plot_df
        color_col = None
        c_min, c_max = 0, 1
        colors = [self.theme["accent_primary"]] * len(values)
        
        if color_by != "None":
            # First try to find the exact column
            color_col = self._find_column(df, color_by)
            
            # If not found, don't use color coding (silently skip)
            if color_col and color_col in plot_df.columns:
                try:
                    color_values = pd.to_numeric(plot_df[color_col], errors='coerce').fillna(0).tolist()
                    color_values = color_values[::-1]
                    
                    # Normalize colors
                    c_min = min(color_values) if color_values else 0
                    c_max = max(color_values) if color_values else 1
                    if c_max == c_min:
                        c_max = c_min + 1
                    
                    norm_colors = [(v - c_min) / (c_max - c_min) for v in color_values]
                    colors = [cm.viridis(c) for c in norm_colors]
                except:
                    # If color processing fails, use default colors
                    color_col = None
                    colors = [self.theme["accent_primary"]] * len(values)
            else:
                color_col = None  # Mark as not found
        
        # Add AI button header
        ai_header = tk.Frame(self.bar_frame, bg=self.theme["bg_card"])
        ai_header.pack(fill=tk.X, padx=4, pady=(4, 2))
        
        plot_info = {
            "type": "horizontal bar chart",
            "title": f"Top {n_items} {entity_label}",
            "x_label": metric_col,
            "y_label": entity_label,
            "data_summary": f"{n_items} items. Top: {labels[0] if labels else 'N/A'} ({values[0] if values else 0})",
        }
        
        def on_ai_click(info=plot_info, frame=self.bar_frame):
            self._ai_describe_plot(info, frame)
        
        ai_btn = tk.Button(
            ai_header, text="🤖 AI Describe",
            font=("Segoe UI", 9),
            bg=self.theme["accent_primary"], fg="white",
            relief=tk.FLAT, cursor="hand2", padx=8, pady=2,
            command=on_ai_click,
        )
        ai_btn.pack(side=tk.RIGHT, padx=4)
        
        # Create plot (no toolbar to avoid focus issues with notebook tabs)
        fig_height = max(5, min(12, n_items * 0.5))
        plot = PlotFrame(self.bar_frame, theme=self.theme_name, figsize=(8, fig_height), show_toolbar=False)
        plot.pack(fill=tk.BOTH, expand=True)
        
        fig, ax = plot.get_figure()
        
        # Create bars
        y_pos = range(len(wrapped_labels))
        bars = ax.barh(y_pos, values, color=colors, height=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wrapped_labels, fontsize=8)
        ax.set_xlabel(metric_col)
        ax.set_title(f"Top {n_items} {entity_label}")
        ax.grid(False)
        
        # Add colorbar if color coded
        if color_by != "None" and color_col:
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=c_min, vmax=c_max))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
            cbar.set_label(color_col)
        
        # Integer ticks
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Value labels
        max_val = max(values) if values else 1
        for i, (bar, v) in enumerate(zip(bars, values)):
            ax.text(v + max_val * 0.02, i, f"{int(v)}", va='center', fontsize=8)
        
        ax.set_xlim(0, max_val * 1.15)
        
        fig.tight_layout()
        plot.refresh()
    
    def _show_scatter_plot(self, df, entity_label: str, options: Dict):
        """Show scatter plot with size, color, label and log scale options."""
        # Clean up existing plots safely
        self._safe_clear_frame(self.scatter_frame)
        
        if df is None or len(df) == 0:
            self._show_scatter_message("No data available for scatter plot.")
            return
        
        if not HAS_MATPLOTLIB:
            self._show_scatter_message("Matplotlib required for plots.")
            return
        
        try:
            self._create_scatter_plot(df, entity_label, options)
        except Exception as e:
            import traceback
            print(f"DEBUG Scatter ERROR: {e}")
            traceback.print_exc()
            self._show_scatter_message(f"Error creating scatter plot: {e}")
    
    def _create_scatter_plot(self, df, entity_label: str, options: Dict):
        """Internal method to create scatter plot."""
        # Extract options
        x_col = options["x_axis"]
        y_col = options["y_axis"]
        size_by = options.get("scatter_size", "None")
        color_by = options.get("scatter_color", "None")
        label_opt = options.get("scatter_label", "None")
        log_x = options.get("log_x", True)
        log_y = options.get("log_y", True)
        top_n = options.get("top_n", 50)
        
        print(f"DEBUG Scatter Options: x='{x_col}', y='{y_col}', size='{size_by}', color='{color_by}'")
        print(f"DEBUG Scatter: df columns = {list(df.columns[:15])}...")
        
        # Check for duplicate columns
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        if dup_cols:
            print(f"DEBUG Scatter WARNING: Duplicate columns found: {dup_cols}")
            # Remove duplicate columns, keeping first occurrence
            df = df.loc[:, ~df.columns.duplicated()]
            print(f"DEBUG Scatter: After removing duplicates, columns = {list(df.columns[:15])}...")
        
        # Find matching columns - with fallback to available columns
        x_col_match = self._find_column(df, x_col)
        y_col_match = self._find_column(df, y_col)
        
        print(f"DEBUG Scatter: x_col_match='{x_col_match}', y_col_match='{y_col_match}'")
        
        # If columns not found, try to use available numeric columns as fallback
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Preferred fallback columns in order of preference
        x_fallbacks = ["Documents", "Number of documents", "Rank"]
        y_fallbacks = ["Citations", "Total citations", "Cited by", "Rank", "Documents", "Number of documents"]
        
        if not x_col_match:
            for fallback in x_fallbacks:
                x_col_match = self._find_column(df, fallback)
                if x_col_match:
                    break
            if not x_col_match and numeric_cols:
                x_col_match = numeric_cols[0]
                
        if not y_col_match:
            for fallback in y_fallbacks:
                y_col_match = self._find_column(df, fallback)
                if y_col_match and y_col_match != x_col_match:  # Don't use same column
                    break
            if not y_col_match and len(numeric_cols) > 1:
                # Find a different column than x
                for col in numeric_cols:
                    if col != x_col_match:
                        y_col_match = col
                        break
            if not y_col_match:
                y_col_match = x_col_match  # Last resort: same column
        
        if not x_col_match or not y_col_match:
            available = ", ".join(df.columns[:5].tolist())
            self._show_scatter_message(f"Could not find suitable columns.\nAvailable: {available}...")
            return
        
        # Get data
        plot_df = df.head(top_n).copy()
        
        try:
            x_data = pd.to_numeric(plot_df[x_col_match], errors='coerce').fillna(0)
            y_data = pd.to_numeric(plot_df[y_col_match], errors='coerce').fillna(0)
        except:
            self._show_scatter_message("Could not convert data to numeric values.")
            return
        
        # For log scale, replace 0 with small value
        if log_x:
            x_data = x_data.replace(0, 0.5)
        if log_y:
            y_data = y_data.replace(0, 0.5)
        
        # Get labels from first column
        labels = plot_df.iloc[:, 0].astype(str).tolist()
        
        # Available numeric columns for reference
        numeric_cols = [c for c in plot_df.columns if pd.api.types.is_numeric_dtype(plot_df[c])]
        print(f"DEBUG Scatter: Available numeric columns: {numeric_cols}")
        
        # Size mapping
        sizes = 100  # default
        if size_by != "None":
            size_col = self._find_column(plot_df, size_by)
            print(f"DEBUG Scatter: Size by '{size_by}' -> found column: {size_col}")
            
            # If not found, try to use a fallback from available columns
            if not size_col and numeric_cols:
                # Prefer Rank or Documents for sizing
                for fallback in ["Rank", "Documents", "Number of documents"]:
                    size_col = self._find_column(plot_df, fallback)
                    if size_col:
                        print(f"DEBUG Scatter: Size fallback to: {size_col}")
                        break
            
            if size_col and size_col in plot_df.columns:
                try:
                    size_data = pd.to_numeric(plot_df[size_col], errors='coerce').fillna(0)
                    # Normalize to reasonable point sizes (20-500)
                    size_min, size_max = size_data.min(), size_data.max()
                    if size_max > size_min:
                        sizes = 20 + 480 * (size_data - size_min) / (size_max - size_min)
                        print(f"DEBUG Scatter: Size range: {size_min} to {size_max}")
                except Exception as e:
                    print(f"DEBUG Scatter: Size calculation failed: {e}")
        
        # Color mapping
        colors = self.theme["accent_primary"]
        use_colorbar = False
        color_col = None
        
        if color_by != "None":
            color_col = self._find_column(plot_df, color_by)
            print(f"DEBUG Scatter: Color by '{color_by}' -> found column: {color_col}")
            
            # If not found, try to use a fallback from available columns
            if not color_col and numeric_cols:
                # Prefer Rank or Documents for coloring
                for fallback in ["Rank", "Documents", "Number of documents", "% Docs"]:
                    color_col = self._find_column(plot_df, fallback)
                    if color_col:
                        print(f"DEBUG Scatter: Color fallback to: {color_col}")
                        break
            
            if color_col and color_col in plot_df.columns:
                try:
                    color_data = pd.to_numeric(plot_df[color_col], errors='coerce').fillna(0)
                    colors = color_data
                    use_colorbar = True
                    print(f"DEBUG Scatter: Color values range: {color_data.min()} to {color_data.max()}")
                except Exception as e:
                    print(f"DEBUG Scatter: Color calculation failed: {e}")
                    colors = self.theme["accent_primary"]
                    use_colorbar = False
        
        # Add AI button header
        ai_header = tk.Frame(self.scatter_frame, bg=self.theme["bg_card"])
        ai_header.pack(fill=tk.X, padx=4, pady=(4, 2))
        
        plot_info = {
            "type": "scatter plot",
            "title": f"{entity_label}: {x_col_match} vs {y_col_match}",
            "x_label": x_col_match,
            "y_label": y_col_match,
            "data_summary": f"{len(plot_df)} points. Log scale: X={log_x}, Y={log_y}",
        }
        
        def on_ai_click(info=plot_info, frame=self.scatter_frame):
            self._ai_describe_plot(info, frame)
        
        ai_btn = tk.Button(
            ai_header, text="🤖 AI Describe",
            font=("Segoe UI", 9),
            bg=self.theme["accent_primary"], fg="white",
            relief=tk.FLAT, cursor="hand2", padx=8, pady=2,
            command=on_ai_click,
        )
        ai_btn.pack(side=tk.RIGHT, padx=4)
        
        # Create plot (no toolbar to avoid focus issues with notebook tabs)
        plot = PlotFrame(self.scatter_frame, theme=self.theme_name, figsize=(9, 7), show_toolbar=False)
        plot.pack(fill=tk.BOTH, expand=True)
        
        fig, ax = plot.get_figure()
        
        # Scatter plot
        if use_colorbar:
            scatter = ax.scatter(
                x_data, y_data, 
                c=colors,
                cmap='viridis',
                alpha=0.7, 
                s=sizes,
                edgecolors='white',
                linewidths=0.5
            )
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
            cbar.set_label(color_col if color_col else color_by)
        else:
            scatter = ax.scatter(
                x_data, y_data, 
                c=colors, 
                alpha=0.7, 
                s=sizes,
                edgecolors='white',
                linewidths=0.5
            )
        
        ax.set_xlabel(x_col_match)
        ax.set_ylabel(y_col_match)
        ax.set_title(f"{entity_label}: {x_col_match} vs {y_col_match}")
        ax.grid(False)
        
        # Log scales
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        
        # Add labels based on option
        if label_opt == "Name":
            for i, label in enumerate(labels):
                short_label = label[:15] + "..." if len(label) > 15 else label
                ax.annotate(
                    short_label,
                    (x_data.iloc[i], y_data.iloc[i]),
                    fontsize=7,
                    alpha=0.8,
                    xytext=(3, 3),
                    textcoords='offset points'
                )
        elif label_opt == "Name (top 10)":
            for i, label in enumerate(labels[:10]):
                short_label = label[:15] + "..." if len(label) > 15 else label
                ax.annotate(
                    short_label,
                    (x_data.iloc[i], y_data.iloc[i]),
                    fontsize=7,
                    alpha=0.8,
                    xytext=(3, 3),
                    textcoords='offset points'
                )
        
        fig.tight_layout()
        plot.refresh()
    
    def _find_column(self, df, col_name: str) -> Optional[str]:
        """Find a column matching the given name (case-insensitive, handles abbreviations)."""
        if col_name == "None":
            return None
            
        col_lower = col_name.lower()
        
        # Exact match first
        for col in df.columns:
            if col.lower() == col_lower:
                return col
        
        # Handle common abbreviations/variations (using standard biblium column names)
        variations = {
            "average year": ["avg year", "avg_year", "avgyear", "mean year"],
            "avg year": ["average year", "avg_year", "avgyear", "mean year"],
            "h-index": ["h index", "hindex", "h_index"],
            "g-index": ["g index", "gindex", "g_index"],
            "a-index": ["a index", "aindex", "a_index"],
            "r-index": ["r index", "rindex", "r_index"],
            "h(2)-index": ["h2-index", "h2 index", "h2index", "h(2) index"],
            "w-index": ["w index", "windex", "w_index"],
            "t-index": ["t index", "tindex", "t_index"],
            "pi-index": ["pi index", "piindex", "pi_index"],
            "hg-index": ["hg index", "hgindex", "hg_index"],
            "chi-index": ["chi index", "chiindex", "chi_index"],
            "gini index": ["gini", "gini_index", "giniindex"],
            "total citations": ["citations", "cited by", "total_citations", "totalcitations"],
            "citations": ["total citations", "cited by", "total_citations"],
            "number of documents": ["documents", "ndocs", "n_docs", "num docs", "doc count", "docs (stats)"],
            "documents": ["number of documents", "ndocs", "n_docs", "docs (stats)"],
            "name": ["source", "author", "country", "affiliation", "keyword", "author(s) id"],
            "rank": ["percentrank", "ranking"],
            "percentage of documents": ["% docs", "prop docs", "proportion of documents"],
            "proportion of documents": ["percentage of documents", "% docs", "prop docs"],
            "first year": ["first_year", "firstyear", "min year"],
            "last year": ["last_year", "lastyear", "max year"],
            "median year": ["median_year", "medianyear"],
            "interdisciplinarity": ["diversity", "interdisciplinary"],
            "collaboration index": ["collaboration_index", "collab index"],
        }
        
        # Check variations
        if col_lower in variations:
            for var in variations[col_lower]:
                for col in df.columns:
                    if col.lower() == var:
                        return col
        
        # Partial match - check if search term words appear in column
        search_words = col_lower.replace("-", " ").replace("_", " ").split()
        for col in df.columns:
            col_words = col.lower().replace("-", " ").replace("_", " ")
            # Check if all search words appear in column name
            if all(word in col_words for word in search_words):
                return col
        
        # Reverse partial match - column words in search term
        for col in df.columns:
            if col.lower() in col_lower or col_lower in col.lower():
                return col
        
        return None
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
BIBLIOMETRIC STATISTICS
═══════════════════════

Calculate impact indices for authors, sources, countries, and keywords.

ANALYSIS LEVELS
───────────────
• Core: H-index, citations, documents
• Extended: + G-index, C-index
• Full: + A, R, W, T, Pi, HG, Chi indices

CORE INDICES
────────────
• H-index (Hirsch index)
  An author has index h if h papers have at least h citations.
  Balances productivity and impact.
  
• Total Citations
  Sum of all citations received.
  
• Number of Documents
  Total publications count.

EXTENDED INDICES
────────────────
• G-index (Egghe)
  Top g papers have together ≥ g² citations.
  Gives more weight to highly cited papers.
  
• C-index
  Complement to H-index for remaining papers.

FULL INDICES
────────────
• A-index: Average citations of papers in H-core
  A = (1/h) × Σ citations for top h papers

• R-index: Square root of citations in H-core
  R = √(Σ citations for top h papers)

• H(2)-index: H-index of H-core papers only
  Second-order h-index

• W-index (Wu): Papers with ≥ 10w citations
  Higher bar than H-index (10× factor)

• T-index: Modified H accounting for career length

• Pi-index: √(Citations of most cited paper) × h
  Combines peak impact with breadth

• HG-index: Geometric mean of H and G
  HG = √(h × g)

• Chi-index (χ): A² / h
  Ratio of average H-core citations to h

• Gini Index: Citation inequality (0-1)
  0 = equal distribution, 1 = maximum inequality

ENTITY TYPES
────────────
• Authors: Individual researcher impact
• Sources: Journal/venue performance
• Countries: National research output
• Affiliations: Institutional productivity
• Keywords: Topic popularity

VISUALIZATION
─────────────
• Table: Ranked statistics
• Bar Chart: Top N comparison
• Scatter Plot: Index relationships

INTERPRETATION
──────────────
• Compare within same field/period
• H-index is age-dependent
• G-index rewards highly cited papers
• Use multiple indices for full picture
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
                pass  # No selection
        
        def copy_all(widget):
            widget.config(state=tk.NORMAL)
            content = widget.get("1.0", tk.END)
            widget.config(state=tk.DISABLED)
            widget.clipboard_clear()
            widget.clipboard_append(content.strip())
        
        text_widget.bind("<Button-3>", show_context_menu)  # Right-click
        text_widget.bind("<Control-c>", lambda e: copy_selected(text_widget))

    def _export_results(self):
        """Export results."""
        if self._stats_df is None:
            messagebox.showwarning("No Results", "Run analysis first.")
            return
        
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")],
            title="Export Statistics"
        )
        
        if filepath:
            try:
                if filepath.endswith('.csv'):
                    self._stats_df.to_csv(filepath, index=False)
                else:
                    self._stats_df.to_excel(filepath, index=False)
                messagebox.showinfo("Exported", f"Statistics exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _manual_count_entities(self, entity_key: str, top_n: int) -> Optional[pd.DataFrame]:
        """Manually count entities when biblium's methods fail."""
        print(f"DEBUG Statistics: Manual counting for {entity_key}")
        
        # Define column mappings for different entity types
        entity_columns = {
            "authors": ["Authors", "Author(s) ID"],
            "countries": ["Corresponding Author Country", "Countries of Authors", "Country"],
            "affiliations": ["Affiliations"],
            "keywords": ["Author Keywords"],
            "sources": ["Source title"],
        }
        
        # Get the columns to try
        cols_to_try = entity_columns.get(entity_key, [])
        
        for col_name in cols_to_try:
            if col_name not in self.bib.df.columns:
                continue
            
            data = self.bib.df[col_name].dropna()
            data = data[data != ""]
            
            if len(data) == 0:
                continue
            
            print(f"DEBUG Statistics: Using column '{col_name}' with {len(data)} values")
            
            # For authors/keywords/affiliations, split by delimiter
            if entity_key in ["authors", "keywords", "affiliations"]:
                # Split by common delimiters
                all_items = []
                for val in data:
                    if isinstance(val, str):
                        # Try semicolon first, then comma
                        if ";" in val:
                            items = [x.strip() for x in val.split(";") if x.strip()]
                        else:
                            items = [val.strip()]
                        all_items.extend(items)
                
                if not all_items:
                    continue
                    
                counts = pd.Series(all_items).value_counts().reset_index()
            else:
                # For countries/sources, count directly
                counts = data.value_counts().reset_index()
            
            # Standard column names based on entity type
            name_col = {
                "authors": "Author",
                "countries": "Country",
                "affiliations": "Affiliation",
                "keywords": "Keyword",
                "sources": "Source",
            }.get(entity_key, "Name")
            
            counts.columns = [name_col, "Number of documents"]
            counts["Proportion of documents"] = counts["Number of documents"] / len(self.bib.df)
            counts["Percentage of documents"] = counts["Proportion of documents"] * 100
            counts["Rank"] = range(1, len(counts) + 1)
            counts["Percentrank"] = counts["Rank"] / len(counts) * 100
            
            print(f"DEBUG Statistics: Manual count found {len(counts)} {entity_key}")
            return counts.head(top_n)
        
        return None
    
    def _compute_entity_stats(self, count_df: pd.DataFrame, entity_key: str, top_n: int) -> pd.DataFrame:
        """Compute full statistics (H-index, citations, etc.) for entities."""
        print(f"DEBUG Statistics: Computing full stats for {entity_key}")
        
        stats_df = count_df.head(top_n).copy()
        
        # Find the name column
        name_col = None
        for col in ["Author", "Source", "Country", "Affiliation", "Keyword", "Author(s) ID"]:
            if col in stats_df.columns:
                name_col = col
                break
        
        if name_col is None and len(stats_df.columns) > 0:
            name_col = stats_df.columns[0]
        
        if name_col is None:
            return stats_df
        
        # Check if we have citation data
        citation_col = None
        for col in ["Cited by", "Citations", "Total citations"]:
            if col in self.bib.df.columns:
                citation_col = col
                break
        
        if citation_col is None:
            print(f"DEBUG Statistics: No citation column found, returning basic stats")
            return stats_df
        
        # Find the entity column in the original data
        entity_col = None
        entity_columns = {
            "authors": ["Authors", "Author(s) ID"],
            "countries": ["Corresponding Author Country", "Countries of Authors"],
            "affiliations": ["Affiliations"],
            "keywords": ["Author Keywords"],
            "sources": ["Source title"],
        }
        
        for col in entity_columns.get(entity_key, []):
            if col in self.bib.df.columns:
                entity_col = col
                break
        
        if entity_col is None:
            print(f"DEBUG Statistics: No entity column found for {entity_key}")
            return stats_df
        
        print(f"DEBUG Statistics: Computing stats using entity_col='{entity_col}', citation_col='{citation_col}'")
        
        # Compute stats for each entity in the top_n
        total_citations = []
        h_indices = []
        avg_years = []
        
        for entity_name in stats_df[name_col]:
            # Find documents associated with this entity
            mask = self.bib.df[entity_col].fillna("").str.contains(str(entity_name), case=False, regex=False)
            entity_docs = self.bib.df[mask]
            
            if len(entity_docs) == 0:
                total_citations.append(0)
                h_indices.append(0)
                avg_years.append(0)
                continue
            
            # Total citations
            cites = pd.to_numeric(entity_docs[citation_col], errors='coerce').fillna(0)
            total_cit = cites.sum()
            total_citations.append(int(total_cit))
            
            # H-index calculation
            sorted_cites = sorted(cites.tolist(), reverse=True)
            h = 0
            for i, c in enumerate(sorted_cites, 1):
                if c >= i:
                    h = i
                else:
                    break
            h_indices.append(h)
            
            # Average year
            if "Year" in entity_docs.columns:
                years = pd.to_numeric(entity_docs["Year"], errors='coerce').dropna()
                avg_year = years.mean() if len(years) > 0 else 0
                avg_years.append(round(avg_year, 1))
            else:
                avg_years.append(0)
        
        # Add computed columns
        stats_df["Total citations"] = total_citations
        stats_df["H-index"] = h_indices
        stats_df["Average year"] = avg_years
        
        # Reorder columns to put stats after basic counts
        cols = list(stats_df.columns)
        # Move name column to front
        if name_col in cols:
            cols.remove(name_col)
            cols = [name_col] + cols
        stats_df = stats_df[cols]
        
        print(f"DEBUG Statistics: Computed stats - Total citations range: {min(total_citations)}-{max(total_citations)}, H-index range: {min(h_indices)}-{max(h_indices)}")
        
        return stats_df
    
    def _ai_describe_plot(self, plot_info: dict, container):
        """Generate AI description of a plot."""
        from biblium.gui.widgets.tables import DataTable
        settings = DataTable.get_llm_settings()
        
        if not settings.get("api_key"):
            from tkinter import messagebox
            messagebox.showinfo("Configure AI", 
                "Please configure your AI API key in Settings first.\n\n"
                "Go to Settings (⚙️) and enter your API key.")
            return
        
        # Remove existing AI result
        if hasattr(self, '_ai_result_frame') and self._ai_result_frame:
            try:
                self._ai_result_frame.destroy()
            except:
                pass
        
        # Show loading
        self._ai_loading = tk.Label(
            container, text="⏳ Generating AI description...",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        )
        self._ai_loading.pack(side=tk.BOTTOM, pady=4)
        
        import threading
        def do_generate():
            try:
                from biblium.llm_utils import llm_describe_plot
                result_text = llm_describe_plot(
                    plot_type=plot_info.get("type", "chart"),
                    title=plot_info.get("title", ""),
                    data_summary=plot_info.get("data_summary", ""),
                    x_axis=plot_info.get("x_label", ""),
                    y_axis=plot_info.get("y_label", ""),
                    context=plot_info.get("context", ""),
                    provider=settings["provider"],
                    model=settings["model"],
                    api_key=settings["api_key"],
                )
                self.after(0, lambda r=result_text: self._show_ai_result(r, container))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.after(0, lambda msg=error_msg: self._show_ai_result(msg, container))
        
        thread = threading.Thread(target=do_generate, daemon=True)
        thread.start()
    
    def _show_ai_result(self, text: str, container):
        """Show AI description result inline."""
        if hasattr(self, '_ai_loading') and self._ai_loading:
            try:
                self._ai_loading.destroy()
            except:
                pass
        if hasattr(self, '_ai_result_frame') and self._ai_result_frame:
            try:
                self._ai_result_frame.destroy()
            except:
                pass
        
        self._ai_result_frame = tk.Frame(container, bg=self.theme["bg_card"])
        self._ai_result_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0))
        
        header = tk.Frame(self._ai_result_frame, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header, text="🤖 AI Description",
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT, padx=4)
        
        tk.Button(
            header, text="✕", font=("Segoe UI", 8),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            relief=tk.FLAT, command=lambda: self._ai_result_frame.destroy(),
            cursor="hand2", width=2,
        ).pack(side=tk.RIGHT, padx=2)
        
        def copy_text():
            try:
                self.clipboard_clear()
                self.clipboard_append(text)
                copy_btn.config(text="✓ Copied")
                self.after(1500, lambda: copy_btn.config(text="📋 Copy"))
            except:
                pass
        
        copy_btn = tk.Button(
            header, text="📋 Copy", font=("Segoe UI", 8),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            relief=tk.FLAT, command=copy_text, cursor="hand2",
        )
        copy_btn.pack(side=tk.RIGHT, padx=2)
        
        text_widget = tk.Text(
            self._ai_result_frame, wrap=tk.WORD,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            relief=tk.FLAT, height=4, padx=8, pady=4,
        )
        text_widget.pack(fill=tk.X, padx=4, pady=(2, 4))
        text_widget.insert("1.0", text)
        
        def on_key(e):
            if e.state & 0x4 and e.keysym.lower() in ('c', 'a'):
                return
            return "break"
        text_widget.bind("<Key>", on_key)
        text_widget.bind("<Button-1>", lambda e: text_widget.focus_set())
