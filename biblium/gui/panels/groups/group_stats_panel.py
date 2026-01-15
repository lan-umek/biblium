# -*- coding: utf-8 -*-
"""
Group Statistics Panel
======================
Panel for computing performance statistics across document groups.

Supports statistics for:
- Sources, Authors, Keywords, Countries, Affiliations, etc.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, List, Optional, Any

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox, LabeledEntry
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


class GroupStatsPanel(BasePanel):
    """
    Panel for computing performance statistics across document groups.
    
    This panel provides access to all get_group_*_stats methods from BiblioGroup.
    """
    
    title = "Group Statistics"
    icon = "📈"
    description = "Compute performance statistics across groups"
    requires_data = True
    
    # Entity configurations for statistics
    ENTITY_CONFIGS = {
        "sources": {
            "label": "Sources (Journals)",
            "method": "get_group_sources_stats",
            "description": "Performance statistics for journals across groups",
        },
        "authors": {
            "label": "Authors",
            "method": "get_group_authors_stats",
            "description": "Author performance statistics across groups",
        },
        "author_keywords": {
            "label": "Author Keywords",
            "method": "get_group_author_keywords_stats",
            "description": "Author keyword statistics across groups",
        },
        "index_keywords": {
            "label": "Index Keywords",
            "method": "get_group_index_keywords_stats",
            "description": "Index keyword statistics across groups",
        },
        "ca_countries": {
            "label": "CA Countries",
            "method": "get_group_ca_countries_stats",
            "description": "Corresponding author country statistics",
        },
        "all_countries": {
            "label": "All Countries",
            "method": "get_group_all_countries_stats",
            "description": "All collaborating country statistics",
        },
        "affiliations": {
            "label": "Affiliations",
            "method": "get_group_affiliations_stats",
            "description": "Affiliation statistics across groups",
        },
        "references": {
            "label": "References",
            "method": "get_group_references_stats",
            "description": "Reference statistics across groups",
        },
        "ngrams_abstract": {
            "label": "N-grams (Abstract)",
            "method": "get_group_ngrams_abstract_stats",
            "description": "Abstract n-gram statistics across groups",
        },
        "ngrams_title": {
            "label": "N-grams (Title)",
            "method": "get_group_ngrams_title_stats",
            "description": "Title n-gram statistics across groups",
        },
        "fields": {
            "label": "Subject Fields (Scopus)",
            "method": "get_group_fields_stats",
            "description": "Subject field statistics (Scopus only)",
        },
        "areas": {
            "label": "Subject Areas (Scopus)",
            "method": "get_group_areas_stats",
            "description": "Subject area statistics (Scopus only)",
        },
        "sciences": {
            "label": "Broad Sciences (Scopus)",
            "method": "get_group_sciences_stats",
            "description": "Broad science statistics (Scopus only)",
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
        
        if not self._check_groups():
            return
        
        # Entity Selection Card
        entity_card = Card(
            self.options_content,
            title="📋 Entity Selection",
            theme=self.theme_name
        )
        entity_card.pack(fill=tk.X, padx=8, pady=8)
        
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
        
        # Description
        self.entity_desc = tk.Label(
            entity_card.content,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        )
        self.entity_desc.pack(fill=tk.X, pady=4)
        self.entity_combo.combobox.bind("<<ComboboxSelected>>", self._update_description)
        self._update_description()
        
        # Parameters Card
        params_card = Card(
            self.options_content,
            title="⚙️ Parameters",
            theme=self.theme_name
        )
        params_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Top N
        self.top_n = LabeledSpinbox(
            params_card.content,
            label="Top N Items:",
            from_=10, to=500, default=100,
            theme=self.theme_name,
            label_width=14
        )
        self.top_n.pack(fill=tk.X, pady=4)
        
        # Output format
        self.output_format_var = tk.StringVar(value="wide")
        format_frame = tk.Frame(params_card.content, bg=self.theme["bg_card"])
        format_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            format_frame,
            text="Output Format:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            width=14,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        tk.Radiobutton(
            format_frame,
            text="Wide",
            variable=self.output_format_var,
            value="wide",
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_secondary"],
            font=FONTS.get_font("body"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        tk.Radiobutton(
            format_frame,
            text="Long",
            variable=self.output_format_var,
            value="long",
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_secondary"],
            font=FONTS.get_font("body"),
        ).pack(side=tk.LEFT)
        
        ToolTip(format_frame, "'Wide': groups as columns; 'Long': groups as rows")
        
        # Include indicators
        self.include_indicators_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            params_card.content,
            label="Include performance indicators",
            variable=self.include_indicators_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Filtering Card
        filter_card = CollapsibleCard(
            self.options_content,
            title="🔍 Filtering",
            theme=self.theme_name,
            collapsed=True
        )
        filter_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Items of interest
        self.items_of_interest = LabeledEntry(
            filter_card.content,
            label="Include Items:",
            default="",
            theme=self.theme_name,
            label_width=14
        )
        self.items_of_interest.pack(fill=tk.X, pady=4)
        ToolTip(self.items_of_interest, "Comma-separated list or regex pattern")
        
        # Exclude items
        self.exclude_items = LabeledEntry(
            filter_card.content,
            label="Exclude Items:",
            default="",
            theme=self.theme_name,
            label_width=14
        )
        self.exclude_items.pack(fill=tk.X, pady=4)
        
        # Display Options Card
        display_card = CollapsibleCard(
            self.options_content,
            title="📊 Display Options",
            theme=self.theme_name,
            collapsed=True
        )
        display_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Show plot
        self.show_plot_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content,
            label="Show comparison plot",
            variable=self.show_plot_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Plot metric - these match the actual column names produced by group_entity_stats
        self.plot_metric_var = tk.StringVar(value="number of documents")
        self.plot_metric_combo = LabeledCombobox(
            display_card.content,
            label="Plot Metric:",
            values=["number of documents", "fraction of documents", "percentage of documents"],
            variable=self.plot_metric_var,
            theme=self.theme_name,
            label_width=12
        )
        self.plot_metric_combo.pack(fill=tk.X, pady=4)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame,
            text="Compute Statistics",
            icon="📈",
            command=self._run_stats,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # Quick actions
        quick_frame = tk.Frame(btn_frame, bg=self.theme["bg_secondary"])
        quick_frame.pack(fill=tk.X, pady=(8, 0))
        
        ThemedButton(
            quick_frame,
            text="All Stats",
            command=self._compute_all_stats,
            theme=self.theme_name,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        ThemedButton(
            quick_frame,
            text="Export",
            command=self._export_results,
            theme=self.theme_name,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
    
    def _check_groups(self) -> bool:
        """Check if groups are available."""
        if not self.bib:
            self._show_message("📂 Please load a dataset first.")
            return False
        
        if not self.bib_group:
            self._show_no_groups_message()
            return False
        
        return True
    
    def _show_message(self, message: str):
        """Show a message in options content."""
        tk.Label(
            self.options_content,
            text=message,
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
    
    def _update_description(self, event=None):
        """Update entity description."""
        selected = self.entity_var.get()
        for key, config in self.ENTITY_CONFIGS.items():
            if config["label"] == selected:
                self.entity_desc.config(text=config["description"])
                break
    
    def _get_selected_entity_key(self) -> str:
        """Get key for selected entity."""
        selected = self.entity_var.get()
        for key, config in self.ENTITY_CONFIGS.items():
            if config["label"] == selected:
                return key
        return "sources"
    
    def _create_results(self):
        """Create the results panel."""
        super()._create_results()
        
        # Create notebook for results and info
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="📊 Statistics")
        
        # Info tab
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
        # Show placeholder
        tk.Label(
            self.results_tab,
            text="Select entity type and click 'Compute Statistics'\n\n"
                 "Computes document counts and performance metrics per group.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
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
            message = "Select entity type and click 'Compute Statistics'\n\nComputes document counts and performance metrics per group."
        
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
    
    def _run_stats(self):
        """Run statistics computation."""
        if not self._check_groups():
            return
        
        entity_key = self._get_selected_entity_key()
        config = self.ENTITY_CONFIGS[entity_key]
        
        self._show_loading(f"Computing {config['label']} statistics...")
        
        def do_stats():
            try:
                method_name = config["method"]
                method = getattr(self.bib_group, method_name, None)
                
                if method is None:
                    raise AttributeError(f"Method {method_name} not found")
                
                # Build parameters
                kwargs = {
                    "top_n": self.top_n.get(),
                    "output_format": self.output_format_var.get(),
                    "indicators": self.include_indicators_var.get(),
                }
                
                # Add filtering if specified
                items_include = self.items_of_interest.get().strip()
                if items_include:
                    kwargs["items_of_interest"] = [x.strip() for x in items_include.split(",")]
                
                items_exclude = self.exclude_items.get().strip()
                if items_exclude:
                    kwargs["exclude_items"] = [x.strip() for x in items_exclude.split(",")]
                
                result_df = method(**kwargs)
                
                self.current_result = result_df
                self.current_entity = entity_key
                
                self.after(0, lambda: self._display_results(result_df, config))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self._show_error(str(e)))
        
        threading.Thread(target=do_stats, daemon=True).start()
    
    def _display_results(self, df: pd.DataFrame, config: Dict):
        """Display statistics results."""
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        
        if df is None or df.empty:
            self._show_initial_message("No results found.")
            return
        
        # Summary cards
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        n_items = len(df)
        n_groups = len(self.bib_group.groups)
        
        grid.add_card(StatsCard(grid, "Items", f"{n_items:,}", "📋", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Groups", str(n_groups), "📊", self.theme_name))
        
        # Find key metrics in results - look for meaningful columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Try to find a meaningful stat column (any group)
        max_docs_col = None
        for col in numeric_cols:
            col_lower = col.lower()
            if 'number of documents' in col_lower and max_docs_col is None:
                max_docs_col = col
                break
        
        if max_docs_col:
            max_docs = df[max_docs_col].max()
            if pd.notna(max_docs) and max_docs > 0:
                grid.add_card(StatsCard(grid, "Max Docs", str(int(max_docs)), "📈", self.theme_name))
        elif len(numeric_cols) > 0:
            # Find first column with non-NaN values
            for col in numeric_cols:
                if "Abbreviated" not in col and df[col].notna().any():
                    avg_val = df[col].mean()
                    if pd.notna(avg_val):
                        # Shorten column name for display
                        short_name = col.split(" - ")[-1] if " - " in col else col
                        if len(short_name) > 20:
                            short_name = short_name[:17] + "..."
                        grid.add_card(StatsCard(grid, f"Avg {short_name}", f"{avg_val:.1f}", "📈", self.theme_name))
                        break
        
        # Plot if enabled
        if self.show_plot_var.get() and HAS_MATPLOTLIB:
            self._create_stats_plot(df, config)
        
        # Results table
        tk.Label(
            self.results_tab,
            text=f"📋 {config['label']} Statistics",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4))
        
        table = DataTable(self.results_tab, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True, pady=8)
        
        # Reset index if needed
        display_df = df.reset_index() if df.index.name else df.copy()
        
        # Clean up columns - remove columns that are all NaN or contain "Abbreviated" 
        # (these are metadata columns that don't add value to the stats view)
        cols_to_keep = []
        for col in display_df.columns:
            # Skip "Abbreviated Source Title" columns (they're all NaN in wide format)
            if "Abbreviated" in col:
                continue
            # Keep column if not all NaN
            if display_df[col].notna().any():
                cols_to_keep.append(col)
        
        if cols_to_keep:
            display_df = display_df[cols_to_keep]
        
        table.set_data(display_df)
    
    def _create_stats_plot(self, df: pd.DataFrame, config: Dict):
        """Create statistics comparison plot - grouped bar chart comparing metrics across groups."""
        try:
            import matplotlib.pyplot as plt
            
            metric = self.plot_metric_var.get()  # e.g., "h_index", "citations", "documents"
            group_names = list(self.bib_group.groups.keys())
            
            # Get group colors
            group_colors = getattr(self.bib_group, 'group_colors', None)
            if group_colors is None:
                palette = list(plt.cm.tab10.colors)
                group_colors = {g: palette[i % len(palette)] for i, g in enumerate(group_names)}
            
            # Get item column (first column or index)
            if df.index.name:
                item_col = df.index.name
                plot_df = df.reset_index()
            else:
                non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
                item_col = non_numeric[0] if non_numeric else df.columns[0]
                plot_df = df.copy()
            
            # Find the metric columns for each group
            # Pattern: "metric (group_name)" e.g., "H-index (Random 1)"
            metric_cols_by_group = {}
            metric_lower = metric.lower().replace("_", "-").replace(" ", "-")
            
            for col in plot_df.columns:
                col_lower = col.lower().replace("_", "-").replace(" ", "-")
                # Check if column contains the metric
                if metric_lower in col_lower or metric.lower() in col_lower:
                    # Find which group this column belongs to
                    for g in group_names:
                        if f"({g})" in col:
                            metric_cols_by_group[g] = col
                            break
            
            # If no specific metric columns found, try Number of documents
            if not metric_cols_by_group:
                for col in plot_df.columns:
                    if "number of documents" in col.lower():
                        for g in group_names:
                            if f"({g})" in col:
                                metric_cols_by_group[g] = col
                                break
            
            if not metric_cols_by_group:
                tk.Label(
                    self.results_tab,
                    text=f"Could not find metric '{metric}' columns for plotting",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"],
                    fg=self.theme.get("warning", "#f59e0b"),
                ).pack(fill=tk.X, pady=4)
                return
            
            # Get top 10 items by total across groups
            cols_to_sum = list(metric_cols_by_group.values())
            plot_df["_total"] = plot_df[cols_to_sum].sum(axis=1)
            top_df = plot_df.nlargest(10, "_total").copy()
            
            if len(top_df) == 0:
                return
            
            # Create grouped bar chart
            plot_frame = PlotFrame(
                self.results_tab,
                theme=self.theme_name,
                figsize=(10, 5)
            , show_ai_button=True)
            plot_frame.pack(fill=tk.X, pady=(0, 16))
            
            fig, ax = plot_frame.get_figure()
            
            # Get labels
            labels = top_df[item_col].astype(str).tolist()
            labels = [l[:25] + "..." if len(l) > 25 else l for l in labels]
            
            x = np.arange(len(labels))
            n_groups = len(metric_cols_by_group)
            width = 0.8 / max(1, n_groups)
            
            # Plot bars for each group
            for i, (group, col) in enumerate(metric_cols_by_group.items()):
                offset = (i - n_groups/2 + 0.5) * width
                color = group_colors.get(group, plt.cm.tab10.colors[i % 10])
                values = top_df[col].fillna(0)
                ax.bar(x + offset, values, width, label=group, color=color, alpha=0.85)
            
            ax.set_xlabel(config['label'])
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"Top 10 {config['label']} - {metric.replace('_', ' ').title()} Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend(title="Groups", fontsize='small')
            
            fig.tight_layout()
            plot_frame.refresh()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(
                self.results_tab,
                text=f"Could not create plot: {str(e)}",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme.get("warning", "#f59e0b"),
            ).pack(fill=tk.X, pady=4)
    
    def _compute_all_stats(self):
        """Compute statistics for all entity types."""
        if not self._check_groups():
            return
        
        self._show_loading("Computing all statistics...")
        
        def do_all():
            results = {}
            errors = []
            
            for key, config in self.ENTITY_CONFIGS.items():
                try:
                    method = getattr(self.bib_group, config["method"], None)
                    if method:
                        result = method(
                            top_n=self.top_n.get(),
                            output_format=self.output_format_var.get(),
                        )
                        results[key] = result
                except Exception as e:
                    errors.append(f"{config['label']}: {str(e)}")
            
            self.after(0, lambda: self._display_all_stats_summary(results, errors))
        
        threading.Thread(target=do_all, daemon=True).start()
    
    def _display_all_stats_summary(self, results: Dict, errors: List[str]):
        """Display summary of all statistics computations."""
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        
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
            text="📋 Statistics Summary",
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
                    "Items": len(df),
                    "Columns": len(df.columns),
                    "Status": "✅",
                })
        
        for error in errors:
            entity_name = error.split(":")[0]
            summary_data.append({
                "Entity Type": entity_name,
                "Items": 0,
                "Columns": 0,
                "Status": "❌",
            })
        
        table = DataTable(self.results_tab, theme=self.theme_name, height=15)
        table.pack(fill=tk.BOTH, expand=True, pady=8)
        table.set_data(pd.DataFrame(summary_data))
        
        # Store all results for export
        self.all_results = results
    
    def _export_results(self):
        """Export current results."""
        if not hasattr(self, 'current_result') or self.current_result is None:
            # Check for all_results
            if hasattr(self, 'all_results') and self.all_results:
                self._export_all_results()
                return
            messagebox.showwarning("No Results", "Please compute statistics first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")],
            title="Export Statistics",
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    self.current_result.to_csv(filename)
                else:
                    self.current_result.to_excel(filename)
                messagebox.showinfo("Success", f"Exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _export_all_results(self):
        """Export all results to Excel with multiple sheets."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            title="Export All Statistics",
        )
        
        if filename:
            try:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    for key, df in self.all_results.items():
                        if df is not None and not df.empty:
                            sheet_name = self.ENTITY_CONFIGS[key]["label"][:31]
                            df.to_excel(writer, sheet_name=sheet_name)
                messagebox.showinfo("Success", f"Exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
GROUP STATISTICS
════════════════

Descriptive statistics for each defined group.

STATISTICS COMPUTED
───────────────────
For each numeric variable:
• Count (non-null values)
• Mean (average)
• Std (standard deviation)
• Min, Max (range)
• 25%, 50%, 75% (quartiles)

COMMON VARIABLES
────────────────
• Citations: Impact measure
• Authors: Team size
• References: Bibliography size
• Year: Temporal distribution

COMPARISON VIEW
───────────────
         Group A    Group B
Count      1,234       987
Mean        45.2      32.1
Median      12.0       8.0
Std         89.4      67.2

VISUALIZATION
─────────────
• Box plots per group
• Violin plots
• Bar chart comparison

EXPORT
──────
• Statistics table (Excel/CSV)
• By group and variable
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

    def refresh(self):
        """Refresh panel when bib_group changes."""
        for widget in self.options_content.winfo_children():
            widget.destroy()
        self._create_options()
        
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        self._show_initial_message()
