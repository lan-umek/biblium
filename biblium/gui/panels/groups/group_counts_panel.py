# -*- coding: utf-8 -*-
"""
Group Counts Panel
==================
Panel for counting entities across document groups.

Supports counting:
- Sources
- Authors
- Author Keywords
- Index Keywords
- CA Countries
- All Countries
- Affiliations
- References
- N-grams (Abstract)
- N-grams (Title)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Dict, List, Optional, Any

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame
from biblium.gui.widgets.tooltips import ToolTip

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GroupCountsPanel(BasePanel):
    """
    Panel for counting entities across document groups.
    
    This panel provides access to all group_count_* methods from BiblioGroup,
    allowing users to compare entity frequencies across groups.
    """
    
    title = "Group Counts"
    icon = "📊"
    description = "Count entities across document groups"
    requires_data = True
    
    # Entity configurations
    ENTITY_CONFIGS = {
        "sources": {
            "label": "Sources (Journals)",
            "method": "group_count_sources",
            "result_attr": "group_sources_counts_df",
            "description": "Count journal/source occurrences per group",
        },
        "authors": {
            "label": "Authors",
            "method": "group_count_authors",
            "result_attr": "group_authors_counts_df",
            "description": "Count author occurrences per group",
        },
        "author_keywords": {
            "label": "Author Keywords",
            "method": "group_count_author_keywords",
            "result_attr": "group_author_keywords_counts_df",
            "description": "Count author keyword occurrences per group",
        },
        "index_keywords": {
            "label": "Index Keywords",
            "method": "group_count_index_keywords",
            "result_attr": "group_index_keywords_counts_df",
            "description": "Count index keyword occurrences per group",
        },
        "ca_countries": {
            "label": "Corresponding Author Countries",
            "method": "group_count_ca_countries",
            "result_attr": "group_ca_countries_counts_df",
            "description": "Count CA country occurrences per group",
        },
        "all_countries": {
            "label": "All Countries",
            "method": "group_count_all_countries",
            "result_attr": "group_all_countries_counts_df",
            "description": "Count all collaborating countries per group",
        },
        "affiliations": {
            "label": "Affiliations",
            "method": "group_count_affiliations",
            "result_attr": "group_affiliations_counts_df",
            "description": "Count affiliation occurrences per group",
        },
        "references": {
            "label": "References",
            "method": "group_count_references",
            "result_attr": "group_references_counts_df",
            "description": "Count cited reference occurrences per group",
        },
        "ngrams_abstract": {
            "label": "N-grams (Abstract)",
            "method": "group_count_ngrams_abstract",
            "result_attr": "group_words_abs_counts_df",
            "description": "Count n-gram occurrences in abstracts per group",
        },
        "ngrams_title": {
            "label": "N-grams (Title)",
            "method": "group_count_ngrams_title",
            "result_attr": "group_words_tit_counts_df",
            "description": "Count n-gram occurrences in titles per group",
        },
    }
    
    @property
    def bib_group(self):
        """Get BiblioGroup instance."""
        # First check stored attribute (set by BasePanel or directly)
        if hasattr(self, '_bib_group') and self._bib_group is not None:
            return self._bib_group
        # Try to get from workspace (master.master since self.master is Notebook)
        try:
            workspace = self.master.master
            if hasattr(workspace, 'bib_group'):
                return workspace.bib_group
            if hasattr(workspace, '_shared_bib_group'):
                return workspace._shared_bib_group
        except:
            pass
        return None
    
    @bib_group.setter
    def bib_group(self, value):
        self._bib_group = value
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Check for groups
        if not self._check_groups():
            return
        
        # Entity Selection Card
        entity_card = Card(
            self.options_content,
            title="📋 Entity Selection",
            theme=self.theme_name
        )
        entity_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Entity dropdown
        entity_values = [config["label"] for config in self.ENTITY_CONFIGS.values()]
        self.entity_var = tk.StringVar(value=entity_values[0])
        
        self.entity_combo = LabeledCombobox(
            entity_card.content,
            label="Entity Type:",
            values=entity_values,
            variable=self.entity_var,
            theme=self.theme_name,
            label_width=12
        )
        self.entity_combo.pack(fill=tk.X, pady=4)
        
        # Description label
        self.entity_desc_label = tk.Label(
            entity_card.content,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        )
        self.entity_desc_label.pack(fill=tk.X, pady=4)
        
        # Bind to update description
        self.entity_combo.combobox.bind("<<ComboboxSelected>>", self._update_entity_description)
        self._update_entity_description()
        
        # Options Card
        options_card = Card(
            self.options_content,
            title="⚙️ Options",
            theme=self.theme_name
        )
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Merge type
        self.merge_type_var = tk.StringVar(value="all items")
        
        merge_frame = tk.Frame(options_card.content, bg=self.theme["bg_card"])
        merge_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            merge_frame,
            text="Merge Type:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            width=12,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        tk.Radiobutton(
            merge_frame,
            text="All Items",
            variable=self.merge_type_var,
            value="all items",
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_secondary"],
            font=FONTS.get_font("body"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        tk.Radiobutton(
            merge_frame,
            text="Shared Only",
            variable=self.merge_type_var,
            value="shared items",
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_secondary"],
            font=FONTS.get_font("body"),
        ).pack(side=tk.LEFT)
        
        ToolTip(merge_frame, "'All Items' includes all entities; 'Shared Only' includes only entities present in multiple groups")
        
        # Display Options Card - expanded by default to show plot controls
        display_card = CollapsibleCard(
            self.options_content,
            title="📊 Display Options",
            theme=self.theme_name,
            collapsed=False
        )
        display_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Top N items to display in table
        self.top_n_display = LabeledSpinbox(
            display_card.content,
            label="Show Top N:",
            from_=10, to=500, default=50,
            theme=self.theme_name,
            label_width=14
        )
        self.top_n_display.pack(fill=tk.X, pady=4)
        ToolTip(self.top_n_display, "Number of items to show in results table")
        
        # Top N for plot
        self.top_n_plot = LabeledSpinbox(
            display_card.content,
            label="Top N for Plot:",
            from_=3, to=20, default=5,
            theme=self.theme_name,
            label_width=14
        )
        self.top_n_plot.pack(fill=tk.X, pady=4)
        ToolTip(self.top_n_plot, "Number of top items per group to show in plot")
        
        # Show plot
        self.show_plot_var = tk.BooleanVar(value=True)
        self.show_plot_check = LabeledCheckbox(
            display_card.content,
            label="Show comparison plot",
            variable=self.show_plot_var,
            theme=self.theme_name,
        )
        self.show_plot_check.pack(fill=tk.X, pady=2)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame,
            text="Count Entities",
            icon="📊",
            command=self._run_count,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # Quick actions
        tk.Label(
            btn_frame,
            text="Quick Actions:",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
        ).pack(fill=tk.X, pady=(8, 4))
        
        quick_frame = tk.Frame(btn_frame, bg=self.theme["bg_secondary"])
        quick_frame.pack(fill=tk.X)
        
        ThemedButton(
            quick_frame,
            text="Count All",
            command=self._count_all_entities,
            theme=self.theme_name,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        ThemedButton(
            quick_frame,
            text="Export Results",
            command=self._export_results,
            theme=self.theme_name,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
    
    def _check_groups(self) -> bool:
        """Check if groups are available."""
        if not self.bib:
            self._show_no_data_message("Please load a dataset first.")
            return False
        
        if not self.bib_group:
            self._show_no_groups_message()
            return False
        
        return True
    
    def _show_no_data_message(self, message: str):
        """Show no data message."""
        tk.Label(
            self.options_content,
            text=f"📂 {message}\n\nGo to DATA → Load Dataset",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _show_no_groups_message(self):
        """Show message when no groups are defined."""
        frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        tk.Label(
            frame,
            text="⚠️ No Groups Defined",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
        ).pack(pady=(0, 8))
        
        tk.Label(
            frame,
            text="Please create document groups first.\n\n"
                 "Go to GROUPS → Setup Groups to define your groups.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(pady=(0, 16))
        
        ActionButton(
            frame,
            text="Go to Group Setup",
            icon="⚙️",
            command=lambda: event_bus.emit(EventBus.PANEL_CHANGED, {"panel": "group_setup"}),
            theme=self.theme_name,
        ).pack()
    
    def _update_entity_description(self, event=None):
        """Update the entity description label."""
        selected_label = self.entity_var.get()
        
        for key, config in self.ENTITY_CONFIGS.items():
            if config["label"] == selected_label:
                self.entity_desc_label.config(text=config["description"])
                break
    
    def _get_selected_entity_key(self) -> str:
        """Get the key for the selected entity."""
        selected_label = self.entity_var.get()
        
        for key, config in self.ENTITY_CONFIGS.items():
            if config["label"] == selected_label:
                return key
        
        return "sources"
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook with permanent Info tab
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="📊 Results")
        
        # Info tab (always visible)
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
        # Show placeholder in results tab
        self._show_results_placeholder()
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading state within the results tab (not destroying notebook)."""
        from biblium.gui.widgets.progress import LoadingSpinner
        
        self._stop_active_spinners()
        
        # Only clear the results tab content, not the entire notebook
        try:
            if hasattr(self, 'results_tab') and self.results_tab.winfo_exists():
                for widget in self.results_tab.winfo_children():
                    try:
                        widget.destroy()
                    except:
                        pass
                
                # Add loading indicator inside results_tab
                loading_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
                loading_frame.pack(expand=True)
                
                spinner = LoadingSpinner(loading_frame, size=32, theme=self.theme_name)
                spinner.pack()
                spinner.start()
                
                # Track active spinner for cleanup
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
            pass  # Widget was destroyed
    
    def _show_error(self, message: str):
        """Show error message within the results tab."""
        self._stop_active_spinners()
        
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
                
            # Clear results tab
            for widget in self.results_tab.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
            
            error_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
            error_frame.pack(expand=True)
            
            tk.Label(
                error_frame,
                text="❌",
                font=("Segoe UI", 32),
                bg=self.theme["bg_card"],
            ).pack()
            
            tk.Label(
                error_frame,
                text="Error",
                font=FONTS.get_font("heading", bold=True),
                bg=self.theme["bg_card"],
                fg=self.theme["danger"],
            ).pack(pady=(8, 4))
            
            tk.Label(
                error_frame,
                text=message,
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
                wraplength=400,
            ).pack()
        except tk.TclError:
            pass  # Widget was destroyed
    
    def _show_initial_message(self, message: str = None):
        """Show initial message within the results tab."""
        self._stop_active_spinners()
        
        if message is None:
            message = "Click 'Run' to see results here.\nSee Info tab for documentation."
        
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
                
            # Clear results tab
            for widget in self.results_tab.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
            
            tk.Label(
                self.results_tab,
                text=message,
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
                justify=tk.CENTER,
            ).pack(expand=True)
        except tk.TclError:
            pass  # Widget was destroyed
    
    def _show_results_placeholder(self):
        """Show placeholder in results tab."""
        # Check if results_tab still exists
        if not hasattr(self, 'results_tab'):
            return
        try:
            if not self.results_tab.winfo_exists():
                return
        except Exception:
            return
        
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        tk.Label(
            self.results_tab,
            text="Click 'Run' to see results here.\nSee Info tab for documentation.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

    def _run_count(self):
        """Run the counting operation."""
        if not self._check_groups():
            return
        
        entity_key = self._get_selected_entity_key()
        config = self.ENTITY_CONFIGS[entity_key]
        
        self._show_loading(f"Counting {config['label']}...")
        
        def do_count():
            try:
                # Get the method
                method_name = config["method"]
                method = getattr(self.bib_group, method_name, None)
                
                if method is None:
                    raise AttributeError(f"Method {method_name} not found on BiblioGroup")
                
                # Call the method
                merge_type = self.merge_type_var.get()
                result_df = method(merge_type=merge_type)
                
                # Store result
                self.current_result = result_df
                self.current_entity = entity_key
                
                self._safe_after(lambda: self._display_results(result_df, config))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._safe_after(lambda msg=str(e): self._show_error(msg))
        
        threading.Thread(target=do_count, daemon=True).start()
    
    def _display_results(self, df: pd.DataFrame, config: Dict):
        """Display counting results."""
        # Check if results_tab still exists
        if not hasattr(self, 'results_tab'):
            return
        try:
            if not self.results_tab.winfo_exists():
                return
        except Exception:
            return
        
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        
        if df is None or df.empty:
            self._show_initial_message("No results found.")
            return
        
        # Store for later use
        self._current_df = df
        self._current_config = config
        
        # Get group names
        group_names = list(self.bib_group.groups.keys())
        
        # Summary cards
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        # Total unique items
        n_items = len(df)
        grid.add_card(StatsCard(grid, "Items", f"{n_items:,}", "📋", self.theme_name, accent=True))
        
        # Number of groups
        n_groups = len(group_names)
        grid.add_card(StatsCard(grid, "Groups", str(n_groups), "📊", self.theme_name))
        
        # Find numeric columns for totals
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            total_count = df[numeric_cols].sum().sum()
            grid.add_card(StatsCard(grid, "Total Count", f"{int(total_count):,}", "🔢", self.theme_name))
        
        # === Tabbed Interface (inside results_tab) ===
        inner_notebook = ttk.Notebook(self.results_tab)
        inner_notebook.pack(fill=tk.BOTH, expand=True, pady=8)
        
        # --- Table Tab ---
        table_frame = tk.Frame(inner_notebook, bg=self.theme["bg_card"])
        inner_notebook.add(table_frame, text="📋 Table")
        
        # Header
        tk.Label(
            table_frame,
            text=f"{config['label']} Counts by Group",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4), padx=8)
        
        # Show row count info
        tk.Label(
            table_frame,
            text=f"Showing {len(df)} items",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.E, padx=8)
        
        # Full table with all counts
        full_table = DataTable(table_frame, theme=self.theme_name, max_rows=500)
        full_table.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
        full_table.set_data(df.reset_index() if df.index.name else df)
        
        # --- Plot Tab ---
        if HAS_MATPLOTLIB and len(group_names) > 0:
            plot_frame = tk.Frame(inner_notebook, bg=self.theme["bg_card"])
            inner_notebook.add(plot_frame, text="📊 Chart")

            

            # Info tab

            info_frame = tk.Frame(inner_notebook, bg=self.theme["bg_card"])

            inner_notebook.add(info_frame, text="ℹ️ Info")

            self._create_info_content(info_frame)
            
            # Store plot frame for later use
            self._plot_frame = plot_frame
            self._group_names = group_names
            
            # Create plot content
            self._create_plot_tab(plot_frame, df, group_names, config)
    
    def _create_plot_tab(self, parent, df: pd.DataFrame, group_names: List[str], config: Dict):
        """Create the plot tab content."""
        top_n_plot = self.top_n_plot.get()
        
        # Header
        tk.Label(
            parent,
            text=f"Top {top_n_plot} Items Comparison",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4), padx=8)
        
        # Create the plot
        self._create_comparison_plot(df, group_names, config, parent=parent)

    def _create_comparison_plot(self, df: pd.DataFrame, group_names: List[str], config: Dict, parent=None):
        """Create comparison plot matching biblium's plot_top_items_by_group style."""
        # Use parent if provided, otherwise use results_content
        target = parent if parent is not None else self.results_tab
        
        try:
            import matplotlib.pyplot as plt
            from collections import Counter
            
            # Get top_n from user setting
            top_n = self.top_n_plot.get()
            
            # Prepare DataFrame - biblium expects first column to be item names
            if df.index.name:
                plot_df = df.reset_index()
            else:
                plot_df = df.copy()
            
            
            # Find item column (first column)
            item_col = plot_df.columns[0]
            
            # Find value columns matching "Number of documents" pattern
            value_cols = [col for col in plot_df.columns if col.startswith("Number of documents")]
            
            if not value_cols:
                # Try alternative: columns containing group names in parentheses
                for col in plot_df.columns:
                    if pd.api.types.is_numeric_dtype(plot_df[col]):
                        for g in group_names:
                            if f"({g})" in str(col):
                                value_cols.append(col)
                                break
            
            if not value_cols:
                tk.Label(
                    target,
                    text=f"No value columns found. Columns: {list(plot_df.columns)[:10]}",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"],
                    fg=self.theme.get("warning", "#f59e0b"),
                    wraplength=500
                ).pack(fill=tk.X, pady=4)
                return
            
            # Build records and collect actual group names from columns
            records = []
            extracted_groups = []
            
            for col in value_cols:
                # Extract group name from column "Number of documents (GroupName)"
                group = col.split("(")[-1].rstrip(")").strip()
                extracted_groups.append(group)
                
                temp_df = plot_df[[item_col, col]].copy()
                temp_df.columns = ["Item", "Value"]
                temp_df["Group"] = group
                
                # Convert to numeric
                temp_df["Value"] = pd.to_numeric(temp_df["Value"], errors='coerce')
                temp_df = temp_df.dropna(subset=["Value"])
                temp_df = temp_df[temp_df["Value"] > 0]
                
                # Get top N items
                top_items = temp_df.sort_values("Value", ascending=False).head(top_n)
                
                if len(top_items) > 0:
                    top_items = top_items.copy()
                    top_items["RankGroup"] = group
                    records.append(top_items)
            
            
            if not records:
                tk.Label(
                    target,
                    text="No data to plot - all values are zero or missing",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"],
                    fg=self.theme.get("warning", "#f59e0b"),
                ).pack(fill=tk.X, pady=4)
                return
            
            plot_data = pd.concat(records)
            
            # Filter out "Combined" group - we only want the actual groups
            plot_data = plot_data[~plot_data["Group"].str.lower().str.contains("combined")].copy()
            
            if len(plot_data) == 0:
                tk.Label(
                    target,
                    text="No data to plot after filtering",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"],
                    fg=self.theme.get("warning", "#f59e0b"),
                ).pack(fill=tk.X, pady=4)
                return
            
            # Define group order: use extracted_groups order (which comes from column order)
            # excluding Combined
            group_order = [g for g in extracted_groups if g.lower() != "combined"]
            
            # Create a mapping for group ordering - REVERSE so first group appears at bottom
            # This way when reading top-to-bottom, you see last group first
            group_order_map = {g: i for i, g in enumerate(reversed(group_order))}
            
            # Add order column and sort
            # Sort by group order descending (so first group ends up at bottom of plot)
            # and by Value ascending (so highest value is at bottom within each group, appearing at top when plotted)
            plot_data["_order"] = plot_data["Group"].map(group_order_map)
            plot_data = plot_data.sort_values(by=["_order", "Value"], ascending=[True, True])
            plot_data = plot_data.drop(columns=["_order"])
            
            # Get unique groups in the display order (reversed for legend)
            unique_groups = [g for g in group_order if g in plot_data["Group"].values]
            
            # Build group colors from the extracted group names (not the original group_names)
            group_colors = getattr(self.bib_group, 'group_colors', None)
            
            # Create a color mapping for the extracted groups
            palette = list(plt.cm.tab10.colors)
            if group_colors is None:
                group_colors = {}
            
            # Build final color dict for all groups in data
            final_colors = {}
            for i, g in enumerate(unique_groups):
                if g in group_colors:
                    final_colors[g] = group_colors[g]
                else:
                    final_colors[g] = palette[i % len(palette)]
            
            
            # Calculate figure height - fit to window with reasonable limits
            n_bars = len(plot_data)
            # Use 0.3 per bar for more compact display, min 4, max 10 inches
            fig_height = max(4, min(10, 0.3 * n_bars + 1))
            
            # Create plot frame - use fill=tk.BOTH and expand=True for responsive sizing
            plot_frame = PlotFrame(
                target,
                theme=self.theme_name,
                figsize=(10, fig_height)
            , show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 16))
            
            fig, ax = plot_frame.get_figure()
            
            # Map colors using final_colors dict
            colors = [final_colors.get(g, palette[0]) for g in plot_data["Group"]]
            
            # Handle duplicate item names (biblium style - add invisible spaces)
            counts = Counter(plot_data["Item"])
            labels = []
            seen = Counter()
            for item in plot_data["Item"]:
                label = str(item)
                if counts[item] > 1:
                    label += " " * seen[item]  # add space padding for repeated items
                    seen[item] += 1
                labels.append(label)
            
            
            # Create horizontal bar chart
            bars = ax.barh(labels, plot_data["Value"], color=colors)
            
            # Add value labels (like biblium)
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2,
                       f"{width:.0f}", va="center", ha="left", fontsize=9)
            
            ax.set_xlabel("Number of documents")
            ax.set_title(f"Top {top_n} {config['label']} by Group")
            
            # Add legend in the correct order
            handles = [plt.Line2D([0], [0], color=final_colors[g], lw=6) for g in group_order if g in unique_groups]
            legend_labels = [g for g in group_order if g in unique_groups]
            ax.legend(handles, legend_labels, title="Groups")
            
            fig.tight_layout()
            plot_frame.refresh()
            print("DEBUG: Plot completed successfully")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(
                target,
                text=f"Could not create plot: {str(e)}",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme.get("warning", "#f59e0b"),
            ).pack(fill=tk.X, pady=4)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(
                target,
                text=f"Could not create plot: {str(e)}",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme.get("warning", "#f59e0b"),
            ).pack(fill=tk.X, pady=4)
    
    def _count_all_entities(self):
        """Count all entity types."""
        if not self._check_groups():
            return
        
        self._show_loading("Counting all entities...")
        
        def do_count_all():
            results = {}
            errors = []
            
            for key, config in self.ENTITY_CONFIGS.items():
                try:
                    method_name = config["method"]
                    method = getattr(self.bib_group, method_name, None)
                    
                    if method:
                        merge_type = self.merge_type_var.get()
                        result_df = method(merge_type=merge_type)
                        results[key] = result_df
                except Exception as e:
                    errors.append(f"{config['label']}: {str(e)}")
            
            self._safe_after(lambda: self._display_count_all_results(results, errors))
        
        threading.Thread(target=do_count_all, daemon=True).start()
    
    def _display_count_all_results(self, results: Dict[str, pd.DataFrame], errors: List[str]):
        """Display results from counting all entities."""
        # Check if results_tab still exists
        if not hasattr(self, 'results_tab'):
            return
        try:
            if not self.results_tab.winfo_exists():
                return
        except Exception:
            return
        
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        
        # Success summary
        n_success = len(results)
        n_errors = len(errors)
        
        grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        grid.add_card(StatsCard(grid, "Completed", str(n_success), "✅", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Errors", str(n_errors), "❌", self.theme_name))
        grid.add_card(StatsCard(grid, "Groups", str(len(self.bib_group.groups)), "📊", self.theme_name))
        
        # Summary table
        tk.Label(
            self.results_tab,
            text="📋 Count Summary by Entity Type",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4))
        
        summary_data = []
        for key, df in results.items():
            config = self.ENTITY_CONFIGS[key]
            if df is not None and not df.empty:
                summary_data.append({
                    "Entity Type": config["label"],
                    "Unique Items": len(df),
                    "Status": "✅ Success",
                })
        
        for error in errors:
            entity_name = error.split(":")[0]
            summary_data.append({
                "Entity Type": entity_name,
                "Unique Items": 0,
                "Status": "❌ Error",
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        table = DataTable(self.results_tab, theme=self.theme_name, height=12)
        table.pack(fill=tk.BOTH, expand=True, pady=8)
        table.set_data(summary_df)
        
        # Show errors if any
        if errors:
            tk.Label(
                self.results_tab,
                text="⚠️ Some errors occurred:\n" + "\n".join(errors[:5]),
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme.get("accent_warning", self.theme.get("warning", "#f59e0b")),
                justify=tk.LEFT,
            ).pack(fill=tk.X, pady=8)
    
    def _export_results(self):
        """Export current results to Excel."""
        if not hasattr(self, 'current_result') or self.current_result is None:
            messagebox.showwarning("No Results", "Please run a count first.")
            return
        
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")],
            title="Export Results",
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    self.current_result.to_csv(filename)
                else:
                    self.current_result.to_excel(filename)
                
                messagebox.showinfo("Success", f"Results exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
GROUP COUNTS
════════════

Compare entity frequencies across groups.

ENTITIES COUNTED
────────────────
• Authors per group
• Sources per group
• Keywords per group
• Countries per group
• Any categorical field

OUTPUT
──────
• Side-by-side frequency tables
• Count in each group
• Percentage of group
• Difference between groups

STATISTICAL TESTS
─────────────────
• Chi-square: Overall difference
• Per-entity significance
• Multiple testing correction

VISUALIZATION
─────────────
• Grouped bar charts
• Stacked bar charts
• Difference plots

NORMALIZATION
─────────────
• Raw counts
• Within-group percentages
• Per-document averages
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

    def refresh(self):
        """Refresh the panel when bib_group changes."""
        # Clear and recreate options
        for widget in self.options_content.winfo_children():
            widget.destroy()
        
        self._create_options()
        
        # Clear results content and show initial message
        # Check if results_tab still exists
        if not hasattr(self, 'results_tab'):
            return
        try:
            if not self.results_tab.winfo_exists():
                return
        except Exception:
            return
        
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        
        self._show_initial_message(
            "Select an entity type and click 'Count Entities'\n\n"
            "This will count occurrences across all groups\n"
            "and display comparison statistics."
        )
