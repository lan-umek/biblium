# -*- coding: utf-8 -*-
"""
Collaboration Metrics Panel
===========================
Panel for analyzing collaboration patterns in bibliometric data.

@author: Claude (Anthropic) for Lan.Umek
@version: 2.7.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledEntry
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import add_plot_context_menu, make_canvas_resizable
from biblium.gui.widgets.plots import PlotFrame

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class CollaborationPanel(BasePanel):
    """Panel for analyzing collaboration patterns."""
    
    title = "Collaboration Metrics"
    icon = "👥"
    description = "Analyze collaboration patterns and team sizes"
    requires_data = True
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        if not self.bib:
            self._show_no_data_message()
            return
        
        # Info Card
        info_frame = tk.Frame(self.options_content, bg="#e3f2fd", relief=tk.FLAT, bd=1)
        info_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        info_inner = tk.Frame(info_frame, bg="#e3f2fd", padx=8, pady=6)
        info_inner.pack(fill=tk.X)
        
        tk.Label(
            info_inner, text="👥 Collaboration Metrics",
            font=FONTS.get_font("body_bold"), bg="#e3f2fd", fg="#1565c0",
        ).pack(anchor=tk.W)
        
        tk.Label(
            info_inner, 
            text="Analyze author collaboration patterns,\nteam sizes, and trends over time.",
            font=FONTS.get_font("small"), bg="#e3f2fd", fg="#1565c0",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        
        # Column Selection Card
        col_card = Card(self.options_content, title="🔧 Settings", theme=self.theme_name)
        col_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Find author columns
        auth_cols = []
        for col in self.bib.df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ["author", "au", "inventor"]):
                auth_cols.append(col)
        
        if not auth_cols:
            auth_cols = list(self.bib.df.columns)
        
        # Prefer "Authors" exactly as default
        default_auth = auth_cols[0] if auth_cols else ""
        for col in auth_cols:
            if col == "Authors":
                default_auth = col
                break
        
        self.authors_combo = LabeledCombobox(
            col_card.content, label="Authors column:",
            values=auth_cols, default=default_auth,
            theme=self.theme_name, label_width=15,
            tooltip="Column containing author names"
        )
        self.authors_combo.pack(fill=tk.X, pady=4)
        
        # Year column
        year_cols = [c for c in self.bib.df.columns if "year" in c.lower() or c == "PY"]
        if not year_cols:
            year_cols = list(self.bib.df.columns)
        
        # Prefer "Year" exactly as default
        default_year = year_cols[0] if year_cols else ""
        for col in year_cols:
            if col == "Year":
                default_year = col
                break
        
        self.year_combo = LabeledCombobox(
            col_card.content, label="Year column:",
            values=year_cols, default=default_year,
            theme=self.theme_name, label_width=15,
            tooltip="Column containing publication year"
        )
        self.year_combo.pack(fill=tk.X, pady=4)
        
        # Separator - set default based on database
        default_sep = "; "  # Default
        if self.bib:
            # Use the database's default separator
            default_sep = getattr(self.bib, 'default_separator', "; ")
            db_type = getattr(self.bib, 'db', '').lower()
            
            # Some databases have specific separators
            if db_type == 'openalex':
                default_sep = "|"
            elif db_type in ['scopus', 'wos', 'pubmed']:
                default_sep = "; "
        
        self.sep_entry = LabeledEntry(
            col_card.content, label="Author separator:",
            default=default_sep, theme=self.theme_name, label_width=15,
            tooltip="Character(s) separating author names (auto-detected from database)"
        )
        self.sep_entry.pack(fill=tk.X, pady=4)
        
        # Show detected database info
        if self.bib:
            db_type = getattr(self.bib, 'db', 'Unknown')
            db_label = tk.Label(
                col_card.content,
                text=f"Database: {db_type} (separator: '{default_sep}')",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            )
            db_label.pack(anchor=tk.W, pady=(0, 4))
        
        # Options Card
        options_card = Card(self.options_content, title="⚙️ Display Options", theme=self.theme_name)
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_distribution_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            options_card.content, label="Show author distribution",
            variable=self.show_distribution_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_trend_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            options_card.content, label="Show temporal trend",
            variable=self.show_trend_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_types_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            options_card.content, label="Show collaboration types",
            variable=self.show_types_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_metrics_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            options_card.content, label="Show detailed metrics table",
            variable=self.show_metrics_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Action Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Analyze Collaboration", icon="👥",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
    
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
            "🤝 Collaboration Analysis\n\n"
            "Explore co-authorship and institutional partnerships.\n\n"
            "Features:\n"
            "• Co-authorship network metrics\n"
            "• International collaboration rates\n"
            "• Institutional partnerships\n"
            "• Collaboration trends over time\n"
            "\n"
            "Reveals research team structures and linkages.\n\n"
            "Steps:\n"
            "1. Load a dataset with author affiliations\n"
            "2. Select collaboration type\n"
            "3. Choose visualization\n"
            "4. Click 'Analyze'\n"
        )
        
        tk.Label(
            self.results_tab,
            text=msg,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

    def _run_analysis(self):
        """Run collaboration analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        # Close any existing matplotlib figures first
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        self._show_loading("Analyzing collaboration patterns...")
        self.update_idletasks()
        
        try:
            authors_col = self.authors_combo.get()
            year_col = self.year_combo.get()
            sep = self.sep_entry.get() or "; "
            
            result = self.bib.analyze_collaboration(
                authors_col=authors_col,
                year_col=year_col,
                sep=sep,
                verbose=False
            )
            self._display_results(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_error(str(e))
    
    def _display_results(self, result: Dict):
        """Display collaboration analysis results."""
        # Close matplotlib figures before clearing widgets
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        # Clear previous results
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
        
        # Summary Cards - Row 1
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 8))
        
        grid.add_card(StatsCard(grid, "Papers", f"{result['n_papers']:,}", "📄", self.theme_name))
        grid.add_card(StatsCard(grid, "CI", f"{result['collaboration_index']:.2f}", "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "DC", f"{result['degree_of_collaboration']:.1%}", "📈", self.theme_name))
        grid.add_card(StatsCard(grid, "CC", f"{result['collaboration_coefficient']:.3f}", "📉", self.theme_name))
        
        # Summary Cards - Row 2
        grid2 = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid2.pack(fill=tk.X, pady=(0, 12))
        
        single_pct = result['single_author_papers'] / result['n_papers'] * 100
        multi_pct = result['multi_author_papers'] / result['n_papers'] * 100
        
        grid2.add_card(StatsCard(grid2, "Single Author", f"{result['single_author_papers']:,}", "👤", self.theme_name))
        grid2.add_card(StatsCard(grid2, "Multi-Author", f"{result['multi_author_papers']:,}", "👥", self.theme_name))
        grid2.add_card(StatsCard(grid2, "Mean Authors", f"{result['basic_stats']['mean']:.1f}", "📊", self.theme_name))
        grid2.add_card(StatsCard(grid2, "Max Authors", f"{result['max_authors']}", "🔝", self.theme_name))
        
        # Create Notebook (tabs) for different views
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        
        # Tab 1: Author Distribution
        if self.show_distribution_var.get() and HAS_MATPLOTLIB:
            dist_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(dist_frame, text="📊 Distribution")
            self._plot_distribution(result, dist_frame)
        
        # Tab 2: Temporal Trend
        if self.show_trend_var.get() and HAS_MATPLOTLIB and len(result['temporal_trend']) > 0:
            trend_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(trend_frame, text="📈 Trend")
            self._plot_trend(result, trend_frame)
        
        # Tab 3: Collaboration Types
        if self.show_types_var.get() and HAS_MATPLOTLIB:
            types_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(types_frame, text="🏷️ Types")
            self._plot_types(result, types_frame)
        
        # Tab 4: Detailed Metrics
        if self.show_metrics_var.get():
            metrics_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(metrics_frame, text="📋 Metrics")
            self._show_metrics_table(result, metrics_frame)
        
        # Tab 5: Distribution Table
        table_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(table_frame, text="📏 Data")

        

        # Info tab

        info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])

        notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        self._show_distribution_table(result, table_frame)
        
        # Export buttons
        self._add_export_buttons(result)
        
        self._current_result = result
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Collaboration Metrics"})
    
    def _plot_distribution(self, result: Dict, parent: tk.Frame):
        """Plot author count distribution."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(10, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        ax = fig.add_subplot(111)
        
        dist = result['author_distribution']
        dist_plot = dist[dist['n_authors'] <= 15].copy()
        
        if len(dist_plot) == 0:
            ax.text(0.5, 0.5, 'No distribution data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        elif len(dist_plot) == 1:
            # Only one category - show single bar with clear label
            n_auth = int(dist_plot['n_authors'].iloc[0])
            n_papers = int(dist_plot['n_papers'].iloc[0])
            pct = float(dist_plot['percentage'].iloc[0])
            
            bars = ax.bar([str(n_auth)], [n_papers], color='#2196F3', alpha=0.8, 
                         edgecolor='white', width=0.5)
            ax.text(0, n_papers + n_papers * 0.02, f'{pct:.1f}%', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add note about limited variation
            if n_auth == 1:
                ax.text(0.5, 0.85, f'All {n_papers:,} papers have single author',
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=10, style='italic', color='#666')
            
            ax.set_xlabel('Number of Authors', fontsize=10)
            ax.set_ylabel('Number of Papers', fontsize=10)
            ax.set_xlim(-0.5, 0.5)
        else:
            # Multiple categories - normal bar chart
            x_labels = dist_plot['n_authors'].astype(int).astype(str).tolist()
            bars = ax.bar(x_labels, dist_plot['n_papers'], color='#2196F3', 
                         alpha=0.8, edgecolor='white')
            
            # Add percentage labels
            max_papers = dist_plot['n_papers'].max()
            for bar, pct in zip(bars, dist_plot['percentage']):
                if pct >= 3:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_papers * 0.02,
                           f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Number of Authors', fontsize=10)
            ax.set_ylabel('Number of Papers', fontsize=10)
        
        ax.set_title('Author Count Distribution', fontsize=12, fontweight='bold')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        plot.refresh()
    
    def _plot_trend(self, result: Dict, parent: tk.Frame):
        """Plot collaboration trend over time."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(10, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        ax = fig.add_subplot(111)
        
        trend = result['temporal_trend']
        
        # Check if there's meaningful variation in the data
        has_variation = False
        if len(trend) >= 2:
            mean_std = trend['mean_authors'].std()
            has_variation = mean_std > 0.01  # More than 1% variation
        
        if len(trend) < 2:
            ax.text(0.5, 0.5, 'Insufficient temporal data\n(need at least 2 years with data)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        elif not has_variation:
            # All values are the same (e.g., all single-author papers)
            mean_val = trend['mean_authors'].mean()
            ax.text(0.5, 0.5, f'No variation in collaboration over time\n\n'
                             f'Mean authors per paper: {mean_val:.2f}\n'
                             f'(constant across {len(trend)} years)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
        else:
            # Main line: Mean authors
            line1 = ax.plot(trend['year'], trend['mean_authors'], marker='o', linewidth=2, 
                           color='#2196F3', label='Mean Authors', markersize=5)
            ax.fill_between(trend['year'], trend['mean_authors'], alpha=0.2, color='#2196F3')
            
            # Set reasonable y-axis limits
            y_min = max(0, trend['mean_authors'].min() * 0.9)
            y_max = trend['mean_authors'].max() * 1.1
            if y_max > y_min:
                ax.set_ylim(y_min, y_max)
            
            # Add secondary axis only if there's meaningful variation in multi-author %
            if 'multi_author_pct' in trend.columns:
                pct_std = trend['multi_author_pct'].std()
                if pct_std > 1:  # More than 1% std
                    ax2 = ax.twinx()
                    line2 = ax2.plot(trend['year'], trend['multi_author_pct'], marker='s', linewidth=2,
                                    color='#E91E63', label='Multi-Author %', linestyle='--', markersize=4)
                    ax2.set_ylabel('Multi-Author Papers (%)', fontsize=9, color='#E91E63')
                    ax2.tick_params(axis='y', labelcolor='#E91E63')
                    ax2.spines['top'].set_visible(False)
                    ax2.grid(False)
                    
                    # Set reasonable limits
                    pct_min = max(0, trend['multi_author_pct'].min() - 5)
                    pct_max = min(100, trend['multi_author_pct'].max() + 5)
                    if pct_max > pct_min:
                        ax2.set_ylim(pct_min, pct_max)
                    
                    # Combined legend
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='best', frameon=False, fontsize=8)
                else:
                    ax.legend(loc='best', frameon=False, fontsize=8)
            else:
                ax.legend(loc='best', frameon=False, fontsize=8)
            
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel('Mean Authors per Paper', fontsize=10)
        
        ax.set_title('Collaboration Trend Over Time', fontsize=12, fontweight='bold')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        plot.refresh()
    
    def _plot_types(self, result: Dict, parent: tk.Frame):
        """Plot collaboration types as horizontal bar chart."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(10, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        ax = fig.add_subplot(111)
        
        types_df = result['collaboration_types']
        # Filter to only types with data
        types_with_data = types_df[types_df['Count'] > 0]
        
        if len(types_with_data) == 0:
            ax.text(0.5, 0.5, 'No collaboration type data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            # Horizontal bar chart - Tufte approved!
            bars = ax.barh(types_with_data['Type'], types_with_data['Count'], 
                          color='#2196F3', alpha=0.8, edgecolor='white', height=0.6)
            
            max_count = types_with_data['Count'].max()
            for bar, (count, pct) in zip(bars, zip(types_with_data['Count'], types_with_data['Percentage'])):
                ax.text(bar.get_width() + max_count * 0.02, bar.get_y() + bar.get_height()/2,
                       f'{count:,} ({pct:.1f}%)', va='center', fontsize=9)
            
            ax.set_xlim(0, max_count * 1.35)
            ax.set_xlabel('Number of Papers', fontsize=10)
        
        ax.set_title('Collaboration Types (Team Sizes)', fontsize=12, fontweight='bold')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        plot.refresh()
    
    def _show_metrics_table(self, result: Dict, parent: tk.Frame):
        """Show detailed metrics table."""
        metrics_data = [
            ("Total Papers", f"{result['n_papers']:,}"),
            ("", ""),
            ("Collaboration Index (CI)", f"{result['collaboration_index']:.3f}"),
            ("Degree of Collaboration (DC)", f"{result['degree_of_collaboration']:.4f}"),
            ("Collaboration Coefficient (CC)", f"{result['collaboration_coefficient']:.4f}"),
            ("", ""),
            ("Single-Author Papers", f"{result['single_author_papers']:,}"),
            ("Multi-Author Papers", f"{result['multi_author_papers']:,}"),
            ("Single-Author %", f"{result['single_author_papers']/result['n_papers']*100:.2f}%"),
            ("Multi-Author %", f"{result['multi_author_papers']/result['n_papers']*100:.2f}%"),
            ("", ""),
            ("Mean Authors", f"{result['basic_stats']['mean']:.2f}"),
            ("Median Authors", f"{result['basic_stats']['median']:.1f}"),
            ("Mode Authors", f"{result['basic_stats']['mode']}"),
            ("Std Deviation", f"{result['basic_stats']['std']:.2f}"),
            ("Max Authors", f"{result['max_authors']}"),
        ]
        
        metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])
        
        # Add descriptions card
        desc_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        desc_frame.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            desc_frame, text="Collaboration Metrics Explained",
            font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(0, 4))
        
        explanations = [
            "• CI (Collaboration Index): Average authors per paper. CI > 1 indicates collaboration.",
            "• DC (Degree of Collaboration): Proportion of multi-authored papers (0-1).",
            "• CC (Collaboration Coefficient): Weighted measure accounting for team sizes (0-1).",
        ]
        for exp in explanations:
            tk.Label(
                desc_frame, text=exp, font=FONTS.get_font("small"),
                bg=self.theme["bg_card"], fg=self.theme["text_muted"],
                justify=tk.LEFT, anchor=tk.W,
            ).pack(anchor=tk.W)
        
        # Table
        table = DataTable(parent, theme=self.theme_name, height=14)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))
        table.set_data(metrics_df)
    
    def _show_distribution_table(self, result: Dict, parent: tk.Frame):
        """Show author distribution table."""
        tk.Label(
            parent, text="Author Count Distribution",
            font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(8, 4), padx=8)
        
        # Format the dataframe
        dist_df = result['author_distribution'].copy()
        dist_df['percentage'] = dist_df['percentage'].apply(lambda x: f"{x:.2f}%")
        dist_df.columns = ['Authors', 'Papers', 'Proportion', 'Percentage']
        
        table = DataTable(parent, theme=self.theme_name, height=15)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        table.set_data(dist_df)
    
    def _add_export_buttons(self, result: Dict):
        """Add export buttons."""
        btn_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=8)
        
        ThemedButton(
            btn_frame, text="📥 Export Data",
            command=lambda: self._export_data(result), theme=self.theme_name
        ).pack(side=tk.LEFT, padx=4)
        
        ThemedButton(
            btn_frame, text="💾 Save Plots",
            command=lambda: self._save_plots(result), theme=self.theme_name
        ).pack(side=tk.LEFT, padx=4)
    
    def _export_data(self, result: Dict):
        """Export collaboration data."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Export Collaboration Data"
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.xlsx'):
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    # Metrics sheet
                    metrics_df = pd.DataFrame([
                        {"Metric": "Total Papers", "Value": result['n_papers']},
                        {"Metric": "Collaboration Index", "Value": result['collaboration_index']},
                        {"Metric": "Degree of Collaboration", "Value": result['degree_of_collaboration']},
                        {"Metric": "Collaboration Coefficient", "Value": result['collaboration_coefficient']},
                        {"Metric": "Single-Author Papers", "Value": result['single_author_papers']},
                        {"Metric": "Multi-Author Papers", "Value": result['multi_author_papers']},
                        {"Metric": "Mean Authors", "Value": result['basic_stats']['mean']},
                        {"Metric": "Median Authors", "Value": result['basic_stats']['median']},
                        {"Metric": "Max Authors", "Value": result['max_authors']},
                    ])
                    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
                    
                    # Distribution sheet
                    result['author_distribution'].to_excel(writer, sheet_name="Author Distribution", index=False)
                    
                    # Types sheet
                    result['collaboration_types'].to_excel(writer, sheet_name="Collaboration Types", index=False)
                    
                    # Temporal trend sheet
                    if len(result['temporal_trend']) > 0:
                        result['temporal_trend'].to_excel(writer, sheet_name="Temporal Trend", index=False)
            else:
                result['author_distribution'].to_csv(filepath, index=False)
            
            messagebox.showinfo("Success", f"Data exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def _save_plots(self, result: Dict):
        """Save plots to files."""
        from biblium import plotbib
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")],
            title="Save Plots"
        )
        
        if not filepath:
            return
        
        try:
            base = filepath.rsplit('.', 1)[0] if '.' in filepath else filepath
            
            plotbib.plot_collaboration(result, filename=base, show=False)
            
            messagebox.showinfo("Success", f"Plots saved to:\n{base}.png/pdf/svg")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{str(e)}")
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading indicator."""
        # Close matplotlib figures first
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
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
        
        frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        frame.pack(expand=True)
        
        tk.Label(
            frame, text="⏳", font=("Segoe UI", 32),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(20, 10))
        
        tk.Label(
            frame, text=message, font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"]
        ).pack()
    
    def _show_error(self, message: str):
        """Show error message."""
        # Close matplotlib figures first
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
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
            text=f"❌ Error\n\n{message}",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["error"], justify=tk.CENTER, wraplength=400,
        ).pack(expand=True)
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
COLLABORATION ANALYSIS
══════════════════════

Analyze co-authorship and research collaboration patterns.

COLLABORATION METRICS
─────────────────────
• Single-author papers: Solo research count
• Multi-author papers: Collaborative research
• Average authors/paper: Team size indicator
• Collaboration Index (CI): Weighted average

COLLABORATION TYPES
───────────────────
• SCP (Single Country Publications)
  All authors from one country
  
• MCP (Multiple Country Publications)
  International collaboration
  
• MCP Ratio: MCP / Total publications
  Measures internationalization

AUTHOR METRICS
──────────────
• Unique authors: Total researcher count
• Authors per paper: Average team size
• Papers per author: Productivity
• Collaboration coefficient

INSTITUTIONAL ANALYSIS
──────────────────────
• Unique affiliations
• Intra-institutional: Same organization
• Inter-institutional: Different organizations
• Institution collaboration network

COUNTRY ANALYSIS
────────────────
• Countries represented
• Corresponding author country
• International co-authorship rate
• Country collaboration pairs

TEMPORAL TRENDS
───────────────
• Collaboration rate over time
• Average authors over time
• International rate trends

OUTPUT
──────
• Summary statistics
• Collaboration network
• Trend visualizations
• Top collaborators list
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

    def destroy(self):
        """Clean up resources before destroying panel."""
        # Close all matplotlib figures
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        # Clear any stored results
        if hasattr(self, '_current_result'):
            self._current_result = None
        
        # Call parent destroy
        super().destroy()
