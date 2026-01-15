# -*- coding: utf-8 -*-
"""
Group Compare Panel
===================
Panel for comparing document groups.

Features:
- Compare continuous variables across groups
- Compute group intersections/overlaps
- Main info by group
- Top cited documents by group
- Scientific production by group
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


class GroupComparePanel(BasePanel):
    """
    Panel for comparing document groups.
    
    Provides tools for:
    - Continuous variable comparison with statistical tests
    - Group intersection analysis
    - Main info comparison
    - Scientific production by group
    """
    
    title = "Group Compare"
    icon = "⚖️"
    description = "Compare variables and overlaps across groups"
    requires_data = True
    
    # Comparison types
    COMPARISON_TYPES = {
        "continuous_vars": {
            "label": "Compare Continuous Variables",
            "method": "compare_continuous_vars",
            "description": "Compare metrics like citations, year across groups with statistical tests",
        },
        "intersections": {
            "label": "Group Intersections",
            "method": "get_group_intersections",
            "description": "Analyze document overlap between groups",
        },
        "main_info": {
            "label": "Main Info by Group",
            "method": "get_main_info",
            "description": "Summary statistics for each group",
        },
        "top_cited": {
            "label": "Top Cited Documents",
            "method": "get_group_top_cited_documents",
            "description": "Most cited documents in each group",
        },
        "production": {
            "label": "Scientific Production",
            "method": "get_scientific_production",
            "description": "Publication counts over time by group",
        },
    }
    
    # Continuous variables that can be compared
    CONTINUOUS_VARS = [
        "Year",
        "Cited by",
        "Citation Count",
        "References Count",
        "Author Count",
        "Page Count",
        "Entropy",
        "Sentiment Score",
    ]
    
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
        
        # Comparison Type Card
        type_card = Card(
            self.options_content,
            title="📋 Comparison Type",
            theme=self.theme_name
        )
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        type_values = [config["label"] for config in self.COMPARISON_TYPES.values()]
        self.comparison_type_var = tk.StringVar(value=type_values[0])
        
        self.type_combo = LabeledCombobox(
            type_card.content,
            label="Analysis:",
            values=type_values,
            variable=self.comparison_type_var,
            theme=self.theme_name,
            label_width=10
        )
        self.type_combo.pack(fill=tk.X, pady=4)
        self.type_combo.combobox.bind("<<ComboboxSelected>>", self._on_type_changed)
        
        # Description
        self.type_desc = tk.Label(
            type_card.content,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        )
        self.type_desc.pack(fill=tk.X, pady=4)
        
        # Dynamic Options Frame
        self.dynamic_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        self.dynamic_frame.pack(fill=tk.X, padx=8, pady=8)
        
        # Initialize with first type
        self._on_type_changed()
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame,
            text="Run Comparison",
            icon="⚖️",
            command=self._run_comparison,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # Export button
        ThemedButton(
            btn_frame,
            text="📥 Export Results",
            command=self._export_results,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=(8, 0))
    
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
        """Show message in options."""
        tk.Label(
            self.options_content,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _show_no_groups_message(self):
        """Show no groups message."""
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
                 "Go to GROUPS → Setup Groups.",
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
    
    def _get_available_numeric_columns(self) -> List[str]:
        """Get numeric columns from data."""
        if not self.bib or not hasattr(self.bib, 'df'):
            return self.CONTINUOUS_VARS
        
        df = self.bib.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add common expected columns if present
        result = []
        for col in self.CONTINUOUS_VARS:
            if col in df.columns:
                result.append(col)
        
        # Add others
        for col in numeric_cols:
            if col not in result:
                result.append(col)
        
        return result
    
    def _on_type_changed(self, event=None):
        """Handle comparison type change."""
        # Update description
        selected = self.comparison_type_var.get()
        for key, config in self.COMPARISON_TYPES.items():
            if config["label"] == selected:
                self.type_desc.config(text=config["description"])
                self._create_type_options(key)
                break
    
    def _create_type_options(self, type_key: str):
        """Create options for selected comparison type."""
        # Clear dynamic frame
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()
        
        options_card = Card(
            self.dynamic_frame,
            title="⚙️ Options",
            theme=self.theme_name
        )
        options_card.pack(fill=tk.X)
        
        if type_key == "continuous_vars":
            self._create_continuous_vars_options(options_card.content)
        elif type_key == "intersections":
            self._create_intersections_options(options_card.content)
        elif type_key == "main_info":
            self._create_main_info_options(options_card.content)
        elif type_key == "top_cited":
            self._create_top_cited_options(options_card.content)
        elif type_key == "production":
            self._create_production_options(options_card.content)
    
    def _create_continuous_vars_options(self, parent):
        """Options for continuous variable comparison."""
        # Variable selection
        numeric_cols = self._get_available_numeric_columns()
        
        tk.Label(
            parent,
            text="Select variables to compare:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(0, 4))
        
        # Listbox for multi-selection
        list_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        list_frame.pack(fill=tk.X, pady=4)
        
        self.vars_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.MULTIPLE,
            height=6,
            font=FONTS.get_font("body"),
        )
        self.vars_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, command=self.vars_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.vars_listbox.config(yscrollcommand=scrollbar.set)
        
        for col in numeric_cols:
            self.vars_listbox.insert(tk.END, col)
        
        # Select first few by default
        for i in range(min(3, len(numeric_cols))):
            self.vars_listbox.select_set(i)
        
        # Output format
        self.cv_format_var = tk.StringVar(value="long")
        format_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        format_frame.pack(fill=tk.X, pady=8)
        
        tk.Label(
            format_frame,
            text="Output:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            width=8,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        tk.Radiobutton(
            format_frame,
            text="Long",
            variable=self.cv_format_var,
            value="long",
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_secondary"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        tk.Radiobutton(
            format_frame,
            text="Wide",
            variable=self.cv_format_var,
            value="wide",
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_secondary"],
        ).pack(side=tk.LEFT)
        
        # Show plot
        self.cv_plot_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            parent,
            label="Show distribution plots",
            variable=self.cv_plot_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
    
    def _create_intersections_options(self, parent):
        """Options for group intersections."""
        # Include document IDs
        self.include_docs_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            parent,
            label="Include document IDs in output",
            variable=self.include_docs_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Visualization type
        self.overlap_viz_var = tk.StringVar(value="venn")
        
        tk.Label(
            parent,
            text="Visualization:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4))
        
        viz_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        viz_frame.pack(fill=tk.X)
        
        for viz, label in [("venn", "Venn"), ("upset", "UpSet"), ("heatmap", "Heatmap")]:
            tk.Radiobutton(
                viz_frame,
                text=label,
                variable=self.overlap_viz_var,
                value=viz,
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
            ).pack(side=tk.LEFT, padx=(0, 8))
    
    def _create_main_info_options(self, parent):
        """Options for main info comparison."""
        # Info type
        self.main_info_type_var = tk.StringVar(value="descriptives")
        
        tk.Label(
            parent,
            text="Information Type:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(0, 4))
        
        for info_type, label in [
            ("descriptives", "Descriptive Statistics"),
            ("performance", "Performance Metrics"),
            ("time_series", "Time Series"),
        ]:
            tk.Radiobutton(
                parent,
                text=label,
                variable=self.main_info_type_var,
                value=info_type,
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
            ).pack(anchor=tk.W)
    
    def _create_top_cited_options(self, parent):
        """Options for top cited documents."""
        # Number of documents
        self.top_n_docs = LabeledSpinbox(
            parent,
            label="Top N per group:",
            from_=5, to=100, default=10,
            theme=self.theme_name,
            label_width=14
        )
        self.top_n_docs.pack(fill=tk.X, pady=4)
        
        # Include columns
        self.include_abstract_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            parent,
            label="Include abstracts",
            variable=self.include_abstract_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
    
    def _create_production_options(self, parent):
        """Options for scientific production."""
        # Cumulative vs annual
        self.production_cumulative_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            parent,
            label="Show cumulative production",
            variable=self.production_cumulative_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Show plot
        self.production_plot_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            parent,
            label="Show production plot",
            variable=self.production_plot_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
    
    def _get_selected_type_key(self) -> str:
        """Get key for selected comparison type."""
        selected = self.comparison_type_var.get()
        for key, config in self.COMPARISON_TYPES.items():
            if config["label"] == selected:
                return key
        return "continuous_vars"
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook for results and info
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="📊 Comparison")
        
        # Info tab
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
        # Show placeholder
        tk.Label(
            self.results_tab,
            text="Select comparison type and click 'Run Comparison'\n\n"
                 "Compare groups using statistical tests and visualizations.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _run_comparison(self):
        """Run the selected comparison."""
        if not self._check_groups():
            return
        
        type_key = self._get_selected_type_key()
        config = self.COMPARISON_TYPES[type_key]
        
        self._show_loading(f"Running {config['label']}...")
        
        def do_comparison():
            try:
                if type_key == "continuous_vars":
                    result = self._run_continuous_vars_comparison()
                elif type_key == "intersections":
                    result = self._run_intersections()
                elif type_key == "main_info":
                    result = self._run_main_info()
                elif type_key == "top_cited":
                    result = self._run_top_cited()
                elif type_key == "production":
                    result = self._run_production()
                else:
                    result = None
                
                self.current_result = result
                self.current_type = type_key
                
                # Use safe after with widget existence check
                self._safe_display(result, type_key)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self._safe_error(error_msg)
        
        threading.Thread(target=do_comparison, daemon=True).start()
    
    def _safe_display(self, result, type_key: str):
        """Safely schedule display on main thread with widget check."""
        try:
            if self.winfo_exists():
                self.after(0, lambda: self._display_results(result, type_key))
        except tk.TclError:
            pass  # Widget was destroyed
    
    def _safe_error(self, msg: str):
        """Safely schedule error display on main thread."""
        try:
            if self.winfo_exists():
                self.after(0, lambda: self._show_error(msg))
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
            message = "Select comparison type and click 'Run Comparison'\n\nCompare groups using statistical tests and visualizations."
        
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
    
    def _run_continuous_vars_comparison(self):
        """Run continuous variable comparison."""
        # Get selected variables
        selected_indices = self.vars_listbox.curselection()
        variables = [self.vars_listbox.get(i) for i in selected_indices]
        
        if not variables:
            raise ValueError("Please select at least one variable to compare")
        
        method = getattr(self.bib_group, "compare_continuous_vars", None)
        if method:
            return method(
                vrs=variables,
                output_format=self.cv_format_var.get(),
            )
        else:
            # Fallback using utilsbib
            from biblium import utilsbib
            return utilsbib.compare_continuous_by_binary_groups(
                df=self.bib_group.df,
                vrs=variables,
                group_matrix=self.bib_group.group_matrix,
                output_format=self.cv_format_var.get(),
            )
    
    def _run_intersections(self):
        """Run group intersections analysis."""
        method = getattr(self.bib_group, "get_group_intersections", None)
        if method:
            return method(include_ids=self.include_docs_var.get())
        else:
            from biblium import utilsbib
            return utilsbib.compute_group_intersections(
                self.bib_group.group_matrix,
                include_ids=self.include_docs_var.get(),
            )
    
    def _run_main_info(self):
        """Run main info comparison."""
        info_type = self.main_info_type_var.get()
        method = getattr(self.bib_group, "get_main_info", None)
        if method:
            return method(info_type=info_type)
        return None
    
    def _run_top_cited(self):
        """Run top cited documents."""
        method = getattr(self.bib_group, "get_group_top_cited_documents", None)
        if method:
            return method(
                top_n=self.top_n_docs.get(),
                include_abstract=self.include_abstract_var.get(),
            )
        return None
    
    def _run_production(self):
        """Run scientific production analysis."""
        method = getattr(self.bib_group, "get_scientific_production", None)
        if method:
            return method(cumulative=self.production_cumulative_var.get())
        return None
    
    def _display_results(self, result, type_key: str):
        """Display comparison results."""
        # Safety check - ensure results_tab still exists
        try:
            if not self.winfo_exists():
                return
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
        except tk.TclError:
            return  # Widget was destroyed
        
        # Clear existing content safely
        try:
            for widget in self.results_tab.winfo_children():
                widget.destroy()
        except tk.TclError:
            return  # Widget was destroyed during iteration
        
        if result is None:
            self._show_initial_message("No results available.")
            return
        
        config = self.COMPARISON_TYPES[type_key]
        
        # Type-specific display
        if type_key == "continuous_vars":
            self._display_continuous_vars_results(result)
        elif type_key == "intersections":
            self._display_intersections_results(result)
        else:
            self._display_table_results(result, config)
    
    def _display_continuous_vars_results(self, result):
        """Display continuous variable comparison results."""
        if isinstance(result, pd.DataFrame):
            # Summary cards
            grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
            grid.pack(fill=tk.X, pady=(0, 16))
            
            n_vars = result['Variable'].nunique() if 'Variable' in result.columns else len(result)
            n_groups = len(self.bib_group.groups)
            
            grid.add_card(StatsCard(grid, "Variables", str(n_vars), "📊", self.theme_name, accent=True))
            grid.add_card(StatsCard(grid, "Groups", str(n_groups), "📋", self.theme_name))
            
            # Check for significant differences
            if 'p_value' in result.columns:
                n_sig = (result['p_value'] < 0.05).sum()
                grid.add_card(StatsCard(grid, "Significant", str(n_sig), "✓", self.theme_name))
            
            # Plot if enabled
            if hasattr(self, 'cv_plot_var') and self.cv_plot_var.get() and HAS_MATPLOTLIB:
                self._create_distribution_plot(result)
            
            # Results table
            tk.Label(
                self.results_tab,
                text="📋 Comparison Results",
                font=FONTS.get_font("subheading"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
            ).pack(fill=tk.X, pady=(8, 4))
            
            table = DataTable(self.results_tab, theme=self.theme_name)
            table.pack(fill=tk.BOTH, expand=True, pady=8)
            table.set_data(result)
    
    def _display_intersections_results(self, result):
        """Display intersections results."""
        # Summary
        grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        n_groups = len(self.bib_group.groups)
        total_docs = len(self.bib_group.df)
        
        grid.add_card(StatsCard(grid, "Groups", str(n_groups), "📊", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Total Docs", f"{total_docs:,}", "📄", self.theme_name))
        
        # Overlap visualization
        if HAS_MATPLOTLIB and hasattr(self, 'overlap_viz_var'):
            viz_type = self.overlap_viz_var.get()
            self._create_overlap_visualization(result, viz_type)
        
        # Results table
        if isinstance(result, pd.DataFrame):
            tk.Label(
                self.results_tab,
                text="📋 Intersection Details",
                font=FONTS.get_font("subheading"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
            ).pack(fill=tk.X, pady=(8, 4))
            
            table = DataTable(self.results_tab, theme=self.theme_name)
            table.pack(fill=tk.BOTH, expand=True, pady=8)
            table.set_data(result)
    
    def _display_table_results(self, result, config: Dict):
        """Display generic table results."""
        if isinstance(result, pd.DataFrame):
            # Summary
            grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
            grid.pack(fill=tk.X, pady=(0, 16))
            
            grid.add_card(StatsCard(grid, "Rows", f"{len(result):,}", "📋", self.theme_name, accent=True))
            grid.add_card(StatsCard(grid, "Columns", str(len(result.columns)), "📊", self.theme_name))
            grid.add_card(StatsCard(grid, "Groups", str(len(self.bib_group.groups)), "📁", self.theme_name))
            
            # Table
            tk.Label(
                self.results_tab,
                text=f"📋 {config['label']}",
                font=FONTS.get_font("subheading"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
            ).pack(fill=tk.X, pady=(8, 4))
            
            table = DataTable(self.results_tab, theme=self.theme_name)
            table.pack(fill=tk.BOTH, expand=True, pady=8)
            
            display_df = result.reset_index() if result.index.name else result
            table.set_data(display_df)
    
    def _create_distribution_plot(self, result: pd.DataFrame):
        """Create distribution comparison plot."""
        plot_frame = PlotFrame(
            self.results_tab,
            theme=self.theme_name,
            figsize=(10, 4)
        , show_ai_button=True)
        plot_frame.pack(fill=tk.X, pady=(0, 16))
        
        fig, ax = plot_frame.get_figure()
        
        # Simple bar plot of means by group
        if 'Variable' in result.columns and 'Group' in result.columns:
            pivot = result.pivot(index='Variable', columns='Group', values='Mean')
            if not pivot.empty:
                pivot.plot(kind='bar', ax=ax, alpha=0.8)
                ax.set_xlabel("Variable")
                ax.set_ylabel("Mean Value")
                ax.set_title("Mean Values by Group")
                ax.legend(title="Group", fontsize='small')
                plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout()
        plot_frame.refresh()
    
    def _create_overlap_visualization(self, result, viz_type: str):
        """Create overlap visualization."""
        plot_frame = PlotFrame(
            self.results_tab,
            theme=self.theme_name,
            figsize=(8, 6)
        , show_ai_button=True)
        plot_frame.pack(fill=tk.X, pady=(0, 16))
        
        fig, ax = plot_frame.get_figure()
        
        # Get group matrix for visualization
        group_matrix = self.bib_group.group_matrix
        group_names = list(group_matrix.columns)
        
        if viz_type == "heatmap":
            # Similarity heatmap
            from biblium import utilsbib
            similarity = utilsbib.compute_group_similarity_matrices(group_matrix)
            if 'jaccard' in similarity:
                sim_df = similarity['jaccard']
                im = ax.imshow(sim_df, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(range(len(group_names)))
                ax.set_xticklabels(group_names, rotation=45, ha='right')
                ax.set_yticks(range(len(group_names)))
                ax.set_yticklabels(group_names)
                ax.set_title("Group Similarity (Jaccard)")
                fig.colorbar(im, ax=ax, label='Jaccard Similarity')
        
        elif viz_type == "venn" and len(group_names) <= 3:
            # Venn diagram for 2-3 groups
            try:
                from matplotlib_venn import venn2, venn3
                
                sets = []
                for col in group_names:
                    sets.append(set(group_matrix[group_matrix[col]].index))
                
                if len(sets) == 2:
                    venn2(sets, set_labels=group_names, ax=ax)
                elif len(sets) == 3:
                    venn3(sets, set_labels=group_names, ax=ax)
                
                ax.set_title("Group Overlap")
            except ImportError:
                ax.text(0.5, 0.5, "matplotlib-venn not installed",
                       ha='center', va='center', transform=ax.transAxes)
        
        else:
            # Default: bar chart of group sizes
            sizes = group_matrix.sum()
            colors = self.bib_group.group_colors if hasattr(self.bib_group, 'group_colors') else {}
            bar_colors = [colors.get(g, f'C{i}') for i, g in enumerate(group_names)]
            
            ax.bar(group_names, sizes, color=bar_colors, alpha=0.8)
            ax.set_xlabel("Group")
            ax.set_ylabel("Documents")
            ax.set_title("Group Sizes")
            plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout()
        plot_frame.refresh()
    
    def _export_results(self):
        """Export current results."""
        if not hasattr(self, 'current_result') or self.current_result is None:
            messagebox.showwarning("No Results", "Please run a comparison first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")],
            title="Export Results",
        )
        
        if filename:
            try:
                if isinstance(self.current_result, pd.DataFrame):
                    if filename.endswith('.csv'):
                        self.current_result.to_csv(filename, index=False)
                    else:
                        self.current_result.to_excel(filename, index=False)
                    messagebox.showinfo("Success", f"Exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
GROUP COMPARISON
════════════════

Statistical comparison between groups.

STATISTICAL TESTS
─────────────────
Numeric variables:
• T-test: Two groups, normal
• Mann-Whitney: Non-normal
• ANOVA: 3+ groups
• Kruskal-Wallis: Non-parametric

Categorical variables:
• Chi-square: Independence
• Fisher's exact: Small samples

EFFECT SIZES
────────────
• Cohen's d: Mean difference
  0.2 small, 0.5 medium, 0.8 large
  
• Eta-squared: Variance explained
  0.01 small, 0.06 medium, 0.14 large

MULTIPLE COMPARISONS
────────────────────
• Bonferroni correction
• FDR (Benjamini-Hochberg)
• Report adjusted p-values

OUTPUT
──────
• Test statistics
• P-values
• Effect sizes
• Confidence intervals
• Significance flags
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
        """Refresh panel."""
        try:
            if not self.winfo_exists():
                return
            for widget in self.options_content.winfo_children():
                widget.destroy()
            self._create_options()
            
            if hasattr(self, 'results_tab') and self.results_tab.winfo_exists():
                for widget in self.results_tab.winfo_children():
                    widget.destroy()
                self._show_initial_message()
        except tk.TclError:
            pass  # Widget was destroyed
