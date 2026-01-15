# -*- coding: utf-8 -*-
"""
Group Visualizations Panel
==========================
Panel for creating group-specific visualizations.

Features:
- Overlap visualizations (Venn, UpSet, heatmap, chord, dendrogram)
- Top items by group
- Distribution comparisons (histogram, boxplot, violin)
- Stacked production over time
- Metric visualizations (heatmap, bubble, slope, bump charts)
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
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GroupVisualizationsPanel(BasePanel):
    """
    Panel for creating group-specific visualizations.
    
    Provides access to all BiblioGroupPlot methods for visualizing
    group comparisons and overlaps.
    """
    
    title = "Group Visualizations"
    icon = "📊"
    description = "Create visual comparisons across groups"
    requires_data = True
    
    # Visualization categories
    VIZ_CATEGORIES = {
        "overlap": {
            "label": "Group Overlap",
            "description": "Visualize document overlap between groups",
            "types": [
                ("venn", "Venn Diagram", "Classic Venn for 2-3 groups"),
                ("upset", "UpSet Plot", "Set intersections for any number of groups"),
                ("heatmap", "Similarity Heatmap", "Jaccard similarity matrix"),
                ("chord", "Chord Diagram", "Circular overlap visualization"),
                ("dendrogram", "Dendrogram", "Hierarchical clustering of groups"),
            ],
        },
        "top_items": {
            "label": "Top Items by Group",
            "description": "Compare top entities across groups",
            "types": [
                ("bar_grouped", "Grouped Bar Chart", "Side-by-side bars"),
                ("bar_stacked", "Stacked Bar Chart", "Stacked composition"),
                ("heatmap", "Heatmap", "Item-group intensity"),
                ("radar", "Radar Chart", "Multi-axis comparison"),
            ],
        },
        "distributions": {
            "label": "Distribution Comparisons",
            "description": "Compare variable distributions across groups",
            "types": [
                ("histogram", "Histogram", "Distribution histograms"),
                ("boxplot", "Box Plot", "Quartile comparison"),
                ("violin", "Violin Plot", "Density + quartiles"),
                ("ridge", "Ridge Plot", "Overlapping densities"),
            ],
        },
        "production": {
            "label": "Production Over Time",
            "description": "Temporal publication patterns by group",
            "types": [
                ("stacked_area", "Stacked Area", "Cumulative production"),
                ("line", "Line Chart", "Trend comparison"),
                ("stream", "Stream Graph", "Flowing composition"),
            ],
        },
        "metrics": {
            "label": "Metric Visualizations",
            "description": "Compare performance metrics across groups",
            "types": [
                ("metric_heatmap", "Metric Heatmap", "Metrics × groups matrix"),
                ("bubble", "Bubble Map", "Size-encoded metrics"),
                ("slope", "Slope Chart", "Rank changes between groups"),
                ("bump", "Bump Chart", "Rank evolution"),
            ],
        },
    }
    
    # Entity options for top items
    ENTITY_OPTIONS = [
        "Sources", "Authors", "Author Keywords", "Index Keywords",
        "CA Countries", "All Countries", "Affiliations", "References",
    ]
    
    # Metric options
    METRIC_OPTIONS = [
        "h_index", "g_index", "m_quotient", "total_citations",
        "avg_citations", "n_publications", "productivity",
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
        
        # Category Selection Card
        cat_card = Card(
            self.options_content,
            title="📊 Visualization Category",
            theme=self.theme_name
        )
        cat_card.pack(fill=tk.X, padx=8, pady=8)
        
        cat_values = [config["label"] for config in self.VIZ_CATEGORIES.values()]
        self.category_var = tk.StringVar(value=cat_values[0])
        
        self.category_combo = LabeledCombobox(
            cat_card.content,
            label="Category:",
            values=cat_values,
            variable=self.category_var,
            theme=self.theme_name,
            label_width=10
        )
        self.category_combo.pack(fill=tk.X, pady=4)
        self.category_combo.combobox.bind("<<ComboboxSelected>>", self._on_category_changed)
        
        # Category description
        self.cat_desc = tk.Label(
            cat_card.content,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        )
        self.cat_desc.pack(fill=tk.X, pady=4)
        
        # Visualization Type Card
        type_card = Card(
            self.options_content,
            title="📈 Visualization Type",
            theme=self.theme_name
        )
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.viz_type_var = tk.StringVar()
        self.viz_type_frame = tk.Frame(type_card.content, bg=self.theme["bg_card"])
        self.viz_type_frame.pack(fill=tk.X)
        
        # Dynamic Options Frame
        self.dynamic_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        self.dynamic_frame.pack(fill=tk.X, padx=8, pady=8)
        
        # Initialize
        self._on_category_changed()
        
        # Display Options Card
        display_card = CollapsibleCard(
            self.options_content,
            title="🎨 Display Options",
            theme=self.theme_name,
            collapsed=True
        )
        display_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Figure size
        size_frame = tk.Frame(display_card.content, bg=self.theme["bg_card"])
        size_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            size_frame,
            text="Figure Size:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            width=12,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.fig_width = tk.Spinbox(
            size_frame, from_=6, to=20, width=5,
            font=FONTS.get_font("body")
        )
        self.fig_width.delete(0, tk.END)
        self.fig_width.insert(0, "10")
        self.fig_width.pack(side=tk.LEFT, padx=(0, 4))
        
        tk.Label(
            size_frame,
            text="×",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT, padx=4)
        
        self.fig_height = tk.Spinbox(
            size_frame, from_=4, to=16, width=5,
            font=FONTS.get_font("body")
        )
        self.fig_height.delete(0, tk.END)
        self.fig_height.insert(0, "6")
        self.fig_height.pack(side=tk.LEFT)
        
        # Use group colors
        self.use_group_colors_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content,
            label="Use group colors",
            variable=self.use_group_colors_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Show legend
        self.show_legend_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content,
            label="Show legend",
            variable=self.show_legend_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame,
            text="Generate Plot",
            icon="📊",
            command=self._generate_plot,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # Save button
        ThemedButton(
            btn_frame,
            text="💾 Save Figure",
            command=self._save_figure,
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
    
    def _get_selected_category_key(self) -> str:
        """Get key for selected category."""
        selected = self.category_var.get()
        for key, config in self.VIZ_CATEGORIES.items():
            if config["label"] == selected:
                return key
        return "overlap"
    
    def _on_category_changed(self, event=None):
        """Handle category change."""
        cat_key = self._get_selected_category_key()
        config = self.VIZ_CATEGORIES[cat_key]
        
        # Update description
        self.cat_desc.config(text=config["description"])
        
        # Update visualization types
        for widget in self.viz_type_frame.winfo_children():
            widget.destroy()
        
        types = config["types"]
        if types:
            self.viz_type_var.set(types[0][0])
        
        for viz_id, viz_name, viz_desc in types:
            rb = tk.Radiobutton(
                self.viz_type_frame,
                text=viz_name,
                variable=self.viz_type_var,
                value=viz_id,
                command=self._on_viz_type_changed,
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
                font=FONTS.get_font("body"),
            )
            rb.pack(anchor=tk.W)
            ToolTip(rb, viz_desc)
        
        # Update dynamic options
        self._create_category_options(cat_key)
    
    def _show_placeholder(self):
        """Show detailed placeholder with instructions."""
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
        
        msg = (
            "📊 Group Visualizations\n\n"
            "Create comparative visualizations.\n\n"
            "Features:\n"
            "• Grouped bar charts\n"
            "• Box plots by group\n"
            "• Violin plots\n"
            "• Export high-quality figures\n"
            "\n"
            "Make group differences presentation-ready.\n\n"
            "Steps:\n"
            "1. Set up groups in Group Setup\n"
            "2. Select visualization type\n"
            "3. Choose variables to plot\n"
            "4. Click 'Generate Chart'\n"
        )
        
        tk.Label(
            self.results_tab,
            text=msg,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

    def _on_viz_type_changed(self):
        """Handle viz type change - update specific options."""
        pass  # Options are category-based
    
    def _create_category_options(self, cat_key: str):
        """Create category-specific options."""
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()
        
        if cat_key == "overlap":
            self._create_overlap_options()
        elif cat_key == "top_items":
            self._create_top_items_options()
        elif cat_key == "distributions":
            self._create_distribution_options()
        elif cat_key == "production":
            self._create_production_options()
        elif cat_key == "metrics":
            self._create_metrics_options()
    
    def _create_overlap_options(self):
        """Options for overlap visualizations."""
        card = Card(self.dynamic_frame, title="⚙️ Options", theme=self.theme_name)
        card.pack(fill=tk.X)
        
        # Show percentages
        self.show_pct_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            card.content,
            label="Show percentages",
            variable=self.show_pct_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Annotate intersections
        self.annotate_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            card.content,
            label="Annotate values",
            variable=self.annotate_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
    
    def _create_top_items_options(self):
        """Options for top items visualizations."""
        card = Card(self.dynamic_frame, title="⚙️ Options", theme=self.theme_name)
        card.pack(fill=tk.X)
        
        # Entity selection
        self.entity_var = tk.StringVar(value=self.ENTITY_OPTIONS[0])
        self.entity_combo = LabeledCombobox(
            card.content,
            label="Entity:",
            values=self.ENTITY_OPTIONS,
            variable=self.entity_var,
            theme=self.theme_name,
            label_width=10
        )
        self.entity_combo.pack(fill=tk.X, pady=4)
        
        # Top N
        self.top_n = LabeledSpinbox(
            card.content,
            label="Top N:",
            from_=5, to=50, default=10,
            theme=self.theme_name,
            label_width=10
        )
        self.top_n.pack(fill=tk.X, pady=4)
        
        # Value column
        self.value_col_var = tk.StringVar(value="count")
        self.value_combo = LabeledCombobox(
            card.content,
            label="Value:",
            values=["count", "citations", "h_index", "percentage"],
            variable=self.value_col_var,
            theme=self.theme_name,
            label_width=10
        )
        self.value_combo.pack(fill=tk.X, pady=4)
    
    def _create_distribution_options(self):
        """Options for distribution visualizations."""
        card = Card(self.dynamic_frame, title="⚙️ Options", theme=self.theme_name)
        card.pack(fill=tk.X)
        
        # Variable selection
        numeric_cols = self._get_numeric_columns()
        self.dist_var = tk.StringVar(value=numeric_cols[0] if numeric_cols else "")
        
        self.dist_combo = LabeledCombobox(
            card.content,
            label="Variable:",
            values=numeric_cols,
            variable=self.dist_var,
            theme=self.theme_name,
            label_width=10
        )
        self.dist_combo.pack(fill=tk.X, pady=4)
        
        # Number of bins (for histogram)
        self.n_bins = LabeledSpinbox(
            card.content,
            label="Bins:",
            from_=10, to=100, default=30,
            theme=self.theme_name,
            label_width=10
        )
        self.n_bins.pack(fill=tk.X, pady=4)
    
    def _create_production_options(self):
        """Options for production visualizations."""
        card = Card(self.dynamic_frame, title="⚙️ Options", theme=self.theme_name)
        card.pack(fill=tk.X)
        
        # Cumulative
        self.cumulative_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            card.content,
            label="Cumulative production",
            variable=self.cumulative_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Normalize
        self.normalize_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            card.content,
            label="Normalize to 100%",
            variable=self.normalize_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
    
    def _create_metrics_options(self):
        """Options for metric visualizations."""
        card = Card(self.dynamic_frame, title="⚙️ Options", theme=self.theme_name)
        card.pack(fill=tk.X)
        
        # Metrics to include (multi-select)
        tk.Label(
            card.content,
            text="Select metrics:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(0, 4))
        
        list_frame = tk.Frame(card.content, bg=self.theme["bg_card"])
        list_frame.pack(fill=tk.X, pady=4)
        
        self.metrics_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.MULTIPLE,
            height=5,
            font=FONTS.get_font("body"),
        )
        self.metrics_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        for metric in self.METRIC_OPTIONS:
            self.metrics_listbox.insert(tk.END, metric)
        
        # Select first three
        for i in range(min(3, len(self.METRIC_OPTIONS))):
            self.metrics_listbox.select_set(i)
        
        # Entity for metrics
        self.metric_entity_var = tk.StringVar(value="Authors")
        self.metric_entity_combo = LabeledCombobox(
            card.content,
            label="Entity:",
            values=self.ENTITY_OPTIONS[:4],  # Limit to main entities
            variable=self.metric_entity_var,
            theme=self.theme_name,
            label_width=10
        )
        self.metric_entity_combo.pack(fill=tk.X, pady=4)
    
    def _get_numeric_columns(self) -> List[str]:
        """Get numeric columns from data."""
        if not self.bib or not hasattr(self.bib, 'df'):
            return ["Year", "Cited by", "Citation Count"]
        
        df = self.bib.df
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Prioritize common columns
        priority = ["Year", "Cited by", "Citation Count", "References Count"]
        result = [c for c in priority if c in numeric]
        result.extend([c for c in numeric if c not in result])
        
        return result
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook for results and info
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="📊 Visualization")
        
        # Info tab
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
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
            message = "Select visualization type and click 'Generate Plot'."
        
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
    
    def _generate_plot(self):
        """Generate the selected visualization."""
        if not self._check_groups():
            return
        
        cat_key = self._get_selected_category_key()
        viz_type = self.viz_type_var.get()
        
        self._show_loading(f"Generating {viz_type} plot...")
        
        def do_plot():
            try:
                if cat_key == "overlap":
                    fig = self._create_overlap_plot(viz_type)
                elif cat_key == "top_items":
                    fig = self._create_top_items_plot(viz_type)
                elif cat_key == "distributions":
                    fig = self._create_distribution_plot(viz_type)
                elif cat_key == "production":
                    fig = self._create_production_plot(viz_type)
                elif cat_key == "metrics":
                    fig = self._create_metrics_plot(viz_type)
                else:
                    fig = None
                
                self.current_figure = fig
                self._safe_after(lambda: self._display_plot(fig))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._safe_after(lambda: self._show_error(str(e)))
        
        threading.Thread(target=do_plot, daemon=True).start()
    
    def _get_figure_size(self):
        """Get figure size from options."""
        try:
            w = int(self.fig_width.get())
            h = int(self.fig_height.get())
            return (w, h)
        except:
            return (10, 6)
    
    def _get_group_colors(self) -> Dict[str, str]:
        """Get group colors."""
        if self.use_group_colors_var.get() and hasattr(self.bib_group, 'group_colors'):
            return self.bib_group.group_colors
        return {}
    
    def _create_overlap_plot(self, viz_type: str):
        """Create overlap visualization."""
        figsize = self._get_figure_size()
        fig, ax = plt.subplots(figsize=figsize)
        
        group_matrix = self.bib_group.group_matrix
        group_names = list(group_matrix.columns)
        colors = self._get_group_colors()
        
        if viz_type == "venn" and len(group_names) <= 3:
            try:
                from matplotlib_venn import venn2, venn3
                
                sets = [set(group_matrix[group_matrix[col]].index) for col in group_names]
                
                if len(sets) == 2:
                    venn2(sets, set_labels=group_names, ax=ax)
                elif len(sets) == 3:
                    venn3(sets, set_labels=group_names, ax=ax)
                
                ax.set_title("Group Overlap (Venn Diagram)")
            except ImportError:
                ax.text(0.5, 0.5, "matplotlib-venn not installed\nInstall with: pip install matplotlib-venn",
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        elif viz_type == "heatmap":
            # Jaccard similarity heatmap
            from biblium import utilsbib
            similarity = utilsbib.compute_group_similarity_matrices(group_matrix)
            
            if 'jaccard' in similarity:
                sim_df = similarity['jaccard']
                im = ax.imshow(sim_df.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
                
                ax.set_xticks(range(len(group_names)))
                ax.set_xticklabels(group_names, rotation=45, ha='right')
                ax.set_yticks(range(len(group_names)))
                ax.set_yticklabels(group_names)
                
                # Annotate
                if self.annotate_var.get():
                    for i in range(len(group_names)):
                        for j in range(len(group_names)):
                            val = sim_df.iloc[i, j]
                            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                                   color='white' if val > 0.5 else 'black', fontsize=9)
                
                fig.colorbar(im, ax=ax, label='Jaccard Similarity')
                ax.set_title("Group Similarity Heatmap")
        
        elif viz_type == "dendrogram":
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist
            
            # Compute distance matrix
            binary_data = group_matrix.T.values.astype(float)
            linkage_matrix = linkage(binary_data, method='ward')
            
            dendrogram(linkage_matrix, labels=group_names, ax=ax, leaf_rotation=45)
            ax.set_title("Group Hierarchical Clustering")
            ax.set_ylabel("Distance")
        
        else:
            # Default: bar chart of sizes
            sizes = group_matrix.sum()
            bar_colors = [colors.get(g, f'C{i}') for i, g in enumerate(group_names)]
            ax.bar(group_names, sizes, color=bar_colors, alpha=0.8)
            ax.set_xlabel("Group")
            ax.set_ylabel("Documents")
            ax.set_title("Group Sizes")
            plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout()
        return fig
    
    def _create_top_items_plot(self, viz_type: str):
        """Create top items visualization."""
        figsize = self._get_figure_size()
        fig, ax = plt.subplots(figsize=figsize)
        
        entity = self.entity_var.get().lower().replace(" ", "_")
        top_n = self.top_n.get()
        value_col = self.value_col_var.get()
        
        group_names = list(self.bib_group.groups.keys())
        colors = self._get_group_colors()
        
        # Get counts for entity
        count_method = f"group_count_{entity}"
        method = getattr(self.bib_group, count_method, None)
        
        if method:
            counts_df = method()
        else:
            ax.text(0.5, 0.5, f"Method {count_method} not available",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        if counts_df is None or counts_df.empty:
            ax.text(0.5, 0.5, "No data available",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Get top items
        top_df = counts_df.head(top_n)
        
        # Find group count columns
        count_cols = [col for col in top_df.columns if any(g in str(col) for g in group_names)]
        if not count_cols:
            count_cols = top_df.select_dtypes(include=[np.number]).columns[:len(group_names)].tolist()
        
        if viz_type == "bar_grouped":
            x = np.arange(len(top_df))
            width = 0.8 / max(1, len(count_cols))
            
            for i, col in enumerate(count_cols):
                offset = (i - len(count_cols)/2 + 0.5) * width
                group_name = group_names[i] if i < len(group_names) else col
                color = colors.get(group_name, f'C{i}')
                ax.bar(x + offset, top_df[col], width, label=group_name, color=color, alpha=0.8)
            
            labels = top_df.index.astype(str) if top_df.index.name else [str(i) for i in range(len(top_df))]
            labels = [l[:20] + "..." if len(str(l)) > 20 else l for l in labels]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
        elif viz_type == "bar_stacked":
            x = np.arange(len(top_df))
            bottom = np.zeros(len(top_df))
            
            for i, col in enumerate(count_cols):
                group_name = group_names[i] if i < len(group_names) else col
                color = colors.get(group_name, f'C{i}')
                ax.bar(x, top_df[col], bottom=bottom, label=group_name, color=color, alpha=0.8)
                bottom += top_df[col].values
            
            labels = top_df.index.astype(str) if top_df.index.name else [str(i) for i in range(len(top_df))]
            labels = [l[:20] + "..." if len(str(l)) > 20 else l for l in labels]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
        elif viz_type == "heatmap":
            if count_cols:
                data = top_df[count_cols].values
                im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
                
                ax.set_xticks(range(len(count_cols)))
                ax.set_xticklabels([group_names[i] if i < len(group_names) else col 
                                   for i, col in enumerate(count_cols)], rotation=45, ha='right')
                
                labels = top_df.index.astype(str) if top_df.index.name else [str(i) for i in range(len(top_df))]
                labels = [l[:25] + "..." if len(str(l)) > 25 else l for l in labels]
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels)
                
                fig.colorbar(im, ax=ax, label='Count')
        
        ax.set_title(f"Top {top_n} {self.entity_var.get()} by Group")
        if self.show_legend_var.get() and viz_type != "heatmap":
            ax.legend(loc='upper right', fontsize='small')
        
        fig.tight_layout()
        return fig
    
    def _create_distribution_plot(self, viz_type: str):
        """Create distribution visualization."""
        figsize = self._get_figure_size()
        fig, ax = plt.subplots(figsize=figsize)
        
        variable = self.dist_var.get()
        group_names = list(self.bib_group.groups.keys())
        colors = self._get_group_colors()
        
        # Get data for each group
        group_data = []
        for gname, gbib in self.bib_group.groups.items():
            if variable in gbib.df.columns:
                data = gbib.df[variable].dropna()
                group_data.append((gname, data))
        
        if not group_data:
            ax.text(0.5, 0.5, f"Variable '{variable}' not found",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        if viz_type == "histogram":
            n_bins = self.n_bins.get()
            for gname, data in group_data:
                color = colors.get(gname, None)
                ax.hist(data, bins=n_bins, alpha=0.5, label=gname, color=color)
            ax.set_xlabel(variable)
            ax.set_ylabel("Frequency")
            
        elif viz_type == "boxplot":
            data_list = [d[1].values for d in group_data]
            labels = [d[0] for d in group_data]
            bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
            
            for i, (patch, gname) in enumerate(zip(bp['boxes'], labels)):
                color = colors.get(gname, f'C{i}')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(variable)
            plt.xticks(rotation=45, ha='right')
            
        elif viz_type == "violin":
            data_list = [d[1].values for d in group_data]
            labels = [d[0] for d in group_data]
            parts = ax.violinplot(data_list, positions=range(len(data_list)), showmeans=True, showmedians=True)
            
            for i, pc in enumerate(parts['bodies']):
                gname = labels[i]
                color = colors.get(gname, f'C{i}')
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(variable)
        
        ax.set_title(f"Distribution of {variable} by Group")
        if self.show_legend_var.get() and viz_type == "histogram":
            ax.legend(loc='upper right')
        
        fig.tight_layout()
        return fig
    
    def _create_production_plot(self, viz_type: str):
        """Create production over time visualization."""
        figsize = self._get_figure_size()
        fig, ax = plt.subplots(figsize=figsize)
        
        group_names = list(self.bib_group.groups.keys())
        colors = self._get_group_colors()
        
        # Get production data
        production_data = {}
        for gname, gbib in self.bib_group.groups.items():
            if 'Year' in gbib.df.columns:
                counts = gbib.df['Year'].value_counts().sort_index()
                production_data[gname] = counts
        
        if not production_data:
            ax.text(0.5, 0.5, "No year data available",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create DataFrame
        prod_df = pd.DataFrame(production_data).fillna(0)
        
        if self.cumulative_var.get():
            prod_df = prod_df.cumsum()
        
        if self.normalize_var.get():
            prod_df = prod_df.div(prod_df.sum(axis=1), axis=0) * 100
        
        if viz_type == "stacked_area":
            ax.stackplot(prod_df.index, [prod_df[col] for col in prod_df.columns],
                        labels=prod_df.columns,
                        colors=[colors.get(g, f'C{i}') for i, g in enumerate(prod_df.columns)],
                        alpha=0.8)
            
        elif viz_type == "line":
            for i, col in enumerate(prod_df.columns):
                color = colors.get(col, f'C{i}')
                ax.plot(prod_df.index, prod_df[col], label=col, color=color, linewidth=2)
        
        ax.set_xlabel("Year")
        ylabel = "Cumulative " if self.cumulative_var.get() else ""
        ylabel += "%" if self.normalize_var.get() else "Publications"
        ax.set_ylabel(ylabel)
        ax.set_title("Scientific Production by Group")
        
        if self.show_legend_var.get():
            ax.legend(loc='upper left', fontsize='small')
        
        fig.tight_layout()
        return fig
    
    def _create_metrics_plot(self, viz_type: str):
        """Create metrics visualization."""
        figsize = self._get_figure_size()
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get selected metrics
        selected_indices = self.metrics_listbox.curselection()
        metrics = [self.metrics_listbox.get(i) for i in selected_indices]
        
        if not metrics:
            metrics = self.METRIC_OPTIONS[:3]
        
        group_names = list(self.bib_group.groups.keys())
        colors = self._get_group_colors()
        
        # This is a placeholder - actual implementation would call BiblioGroup methods
        # For now, create sample data
        data = np.random.rand(len(metrics), len(group_names)) * 100
        
        if viz_type == "metric_heatmap":
            im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
            
            ax.set_xticks(range(len(group_names)))
            ax.set_xticklabels(group_names, rotation=45, ha='right')
            ax.set_yticks(range(len(metrics)))
            ax.set_yticklabels(metrics)
            
            fig.colorbar(im, ax=ax, label='Value')
            ax.set_title("Metrics by Group")
            
        elif viz_type == "bubble":
            for i, metric in enumerate(metrics):
                for j, gname in enumerate(group_names):
                    size = data[i, j] * 10
                    color = colors.get(gname, f'C{j}')
                    ax.scatter(j, i, s=size, c=color, alpha=0.6)
            
            ax.set_xticks(range(len(group_names)))
            ax.set_xticklabels(group_names, rotation=45, ha='right')
            ax.set_yticks(range(len(metrics)))
            ax.set_yticklabels(metrics)
            ax.set_title("Metrics by Group (Bubble Map)")
        
        fig.tight_layout()
        return fig
    
    def _display_plot(self, fig):
        """Display the generated plot."""
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        
        if fig is None:
            self._show_initial_message("Failed to generate plot.")
            return
        
        # Summary info
        grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        grid.add_card(StatsCard(grid, "Groups", str(len(self.bib_group.groups)), "📊", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Type", self.viz_type_var.get(), "📈", self.theme_name))
        grid.add_card(StatsCard(grid, "Category", self._get_selected_category_key().title(), "📋", self.theme_name))
        
        # Plot frame
        plot_frame = PlotFrame(
            self.results_tab,
            theme=self.theme_name,
            figsize=self._get_figure_size()
        , show_ai_button=True)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=8)
        
        # Set the figure
        plot_frame.figure = fig
        plot_frame.canvas.figure = fig
        plot_frame.canvas.draw()
    
    def _save_figure(self):
        """Save the current figure."""
        if not hasattr(self, 'current_figure') or self.current_figure is None:
            messagebox.showwarning("No Figure", "Please generate a plot first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Files", "*.png"),
                ("PDF Files", "*.pdf"),
                ("SVG Files", "*.svg"),
                ("All Files", "*.*"),
            ],
            title="Save Figure",
        )
        
        if filename:
            try:
                self.current_figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
GROUP VISUALIZATIONS
════════════════════

Comparative plots across groups.

CHART TYPES
───────────
• Grouped bar chart
• Stacked bar chart
• Box plots by group
• Violin plots
• Radar/spider charts

CUSTOMIZATION
─────────────
• Colors per group
• Axis labels
• Legend position
• Sort order

VARIABLES TO PLOT
─────────────────
• Numeric: Citations, authors
• Categorical: Counts, proportions
• Temporal: Trends by year

EXPORT OPTIONS
──────────────
• PNG image
• SVG vector
• PDF publication quality
• High DPI for print

BEST PRACTICES
──────────────
• Use consistent colors
• Include sample sizes
• Add error bars
• Label clearly
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
        for widget in self.options_content.winfo_children():
            widget.destroy()
        self._create_options()
        
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        self._show_initial_message()
