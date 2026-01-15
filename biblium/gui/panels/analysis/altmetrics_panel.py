# -*- coding: utf-8 -*-
"""
Altmetrics Analysis Panel
=========================
GUI panel for analyzing alternative metrics (social media, news, policy mentions, etc.).
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, Optional
import pandas as pd

from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, StatsCard, CardGrid
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledEntry
from biblium.gui.widgets.buttons import ActionButton
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame
from biblium.gui.config import FONTS, get_theme
from biblium.gui.core.events import EventBus

try:
    from biblium.utilsbib_modules.stats import analyze_altmetrics
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

event_bus = EventBus()


class AltmetricsPanel(BasePanel):
    """
    Panel for altmetrics analysis.
    
    Analyzes alternative impact metrics including:
    - Social media (Twitter, Facebook, Reddit)
    - Academic social (Mendeley readers)
    - News and blog coverage
    - Policy document citations
    - Patent citations
    - Wikipedia mentions
    """
    
    title = "📊 Altmetrics Analysis"
    description = "Analyze alternative impact metrics beyond citations"
    
    def __init__(self, parent, bib=None, theme: str = "light", **kwargs):
        self._current_result = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create options panel."""
        super()._create_options()
        
        if not self.bib:
            tk.Label(
                self.options_content,
                text="Load a dataset to analyze altmetrics",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"],
                fg=self.theme["text_muted"],
            ).pack(pady=20)
            return
        
        # Column Selection Card
        col_card = Card(self.options_content, title="📋 Column Selection", theme=self.theme_name)
        col_card.pack(fill=tk.X, padx=8, pady=8)
        
        # ID column - include unique-id for OpenAlex and other ID-like columns
        id_cols = []
        if "unique-id" in self.bib.df.columns:
            id_cols.append("unique-id")
        for c in self.bib.df.columns:
            if c not in id_cols and ("id" in c.lower() or c in ["EID", "UT"]):
                id_cols.append(c)
        
        if not id_cols:
            id_cols = list(self.bib.df.columns)
        
        default_id = id_cols[0] if id_cols else ""
        
        self.id_combo = LabeledCombobox(
            col_card.content, label="ID column:",
            values=id_cols,
            default=default_id,
            theme=self.theme_name, label_width=15,
        )
        self.id_combo.pack(fill=tk.X, pady=4)
        
        # DOI column
        doi_cols = [c for c in self.bib.df.columns if "doi" in c.lower()]
        default_doi = "DOI" if "DOI" in self.bib.df.columns else (
            doi_cols[0] if doi_cols else ""
        )
        
        self.doi_combo = LabeledCombobox(
            col_card.content, label="DOI column:",
            values=doi_cols if doi_cols else list(self.bib.df.columns),
            default=default_doi,
            theme=self.theme_name, label_width=15,
        )
        self.doi_combo.pack(fill=tk.X, pady=4)
        
        # Citations column
        cit_cols = [c for c in self.bib.df.columns if any(x in c.lower() for x in ["cited", "citation", "cit"])]
        default_cit = "Cited by" if "Cited by" in self.bib.df.columns else (
            cit_cols[0] if cit_cols else ""
        )
        
        self.cit_combo = LabeledCombobox(
            col_card.content, label="Citations column:",
            values=cit_cols if cit_cols else list(self.bib.df.columns),
            default=default_cit,
            theme=self.theme_name, label_width=15,
        )
        self.cit_combo.pack(fill=tk.X, pady=4)
        
        # Year column
        year_cols = [c for c in self.bib.df.columns if "year" in c.lower() or c == "PY"]
        default_year = "Year" if "Year" in self.bib.df.columns else (
            year_cols[0] if year_cols else ""
        )
        
        self.year_combo = LabeledCombobox(
            col_card.content, label="Year column:",
            values=year_cols if year_cols else list(self.bib.df.columns),
            default=default_year,
            theme=self.theme_name, label_width=15,
        )
        self.year_combo.pack(fill=tk.X, pady=4)
        
        # Data Source Card - API Keys
        source_card = Card(self.options_content, title="🔑 API Configuration", theme=self.theme_name)
        source_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Altmetric API Key
        self.altmetric_key = LabeledEntry(
            source_card.content, label="Altmetric API Key:",
            theme=self.theme_name, label_width=18,
        )
        self.altmetric_key.pack(fill=tk.X, pady=2)
        
        # PlumX API Key  
        self.plumx_key = LabeledEntry(
            source_card.content, label="PlumX API Key:",
            theme=self.theme_name, label_width=18,
        )
        self.plumx_key.pack(fill=tk.X, pady=2)
        
        tk.Label(
            source_card.content,
            text="Enter API keys to fetch real altmetric data.\n"
                 "Leave empty and enable simulation for demo.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 8))
        
        # Simulation option (disabled by default)
        self.simulate_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            source_card.content, label="Simulate altmetric data (for demonstration)",
            variable=self.simulate_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        tk.Label(
            source_card.content,
            text="Simulation creates realistic altmetric\n"
                 "distributions correlated with citations.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 0))
        
        # Display Options Card
        display_card = Card(self.options_content, title="⚙️ Display Options", theme=self.theme_name)
        display_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_overview_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show overview plots",
            variable=self.show_overview_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_sources_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show source breakdown",
            variable=self.show_sources_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_trends_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show temporal trends",
            variable=self.show_trends_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_table_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show top papers table",
            variable=self.show_table_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Action Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Analyze Altmetrics", icon="📊",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook with Info tab always visible
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab (for analysis output)
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="📊 Results")
        
        # Info tab (always visible)
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
        # Show placeholder in results tab
        self._show_placeholder()
    
    def _show_placeholder(self):
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
            text="📊 Altmetrics Analysis\n\n"
                 "Select columns and click 'Analyze Altmetrics'\n"
                 "to explore social media impact and alternative metrics.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

    def _run_analysis(self):
        """Run altmetrics analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_ANALYSIS:
            messagebox.showerror("Error", "Altmetrics analysis module not available.")
            return
        
        try:
            plt.close('all')
        except:
            pass
        
        self._show_loading("Analyzing altmetrics...")
        self.update_idletasks()
        
        try:
            id_col = self.id_combo.get()
            doi_col = self.doi_combo.get()
            cit_col = self.cit_combo.get()
            year_col = self.year_combo.get()
            simulate = self.simulate_var.get()
            
            result = analyze_altmetrics(
                self.bib.df,
                id_col=id_col,
                doi_col=doi_col,
                citations_col=cit_col,
                year_col=year_col,
                simulate=simulate,
                verbose=False,
            )
            
            self._display_results(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_error(str(e))
    
    def _display_results(self, result: Dict):
        """Display altmetrics analysis results."""
        try:
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
        
        stats = result["statistics"]
        
        # Summary Cards - Row 1
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 8))
        
        grid.add_card(StatsCard(grid, "Papers", f"{stats['total_papers']:,}", "📄", self.theme_name))
        grid.add_card(StatsCard(grid, "With Attention", f"{stats['with_attention']:,}", "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "Attention Rate", f"{stats['attention_rate']:.1f}%", "📈", self.theme_name))
        grid.add_card(StatsCard(grid, "Mean Score", f"{stats['mean_score']:.2f}", "⭐", self.theme_name))
        
        # Summary Cards - Row 2
        grid2 = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid2.pack(fill=tk.X, pady=(0, 12))
        
        grid2.add_card(StatsCard(grid2, "Twitter", f"{stats['with_twitter']:,}", "🐦", self.theme_name))
        grid2.add_card(StatsCard(grid2, "Mendeley", f"{stats['with_mendeley']:,}", "📚", self.theme_name))
        grid2.add_card(StatsCard(grid2, "News", f"{stats['with_news']:,}", "📰", self.theme_name))
        grid2.add_card(StatsCard(grid2, "Policy", f"{stats['with_policy']:,}", "🏛️", self.theme_name))
        
        # Create Notebook
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        
        if self.show_overview_var.get() and HAS_MATPLOTLIB:
            overview_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(overview_frame, text="📊 Overview")
            self._plot_overview(result, overview_frame)
        
        if self.show_sources_var.get() and HAS_MATPLOTLIB:
            sources_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(sources_frame, text="📡 Sources")
            self._plot_sources(result, sources_frame)
        
        if self.show_trends_var.get() and HAS_MATPLOTLIB:
            trends_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(trends_frame, text="📈 Trends")
            self._plot_trends(result, trends_frame)
        
        if self.show_table_var.get():
            table_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(table_frame, text="🏆 Top Papers")
            self._show_top_papers(result, table_frame)
        
        coverage_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(coverage_frame, text="📋 Coverage")
        self._show_coverage(result, coverage_frame)
        
        # Info tab
        info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        
        self._add_export_buttons(result)
        self._current_result = result
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Altmetrics Analysis"})
    
    def _plot_overview(self, result: Dict, parent: tk.Frame):
        """Plot altmetrics overview - score distribution."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(8, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        df = result["summary_df"]
        scores = df["Altmetric Score"]
        scores_nonzero = scores[scores > 0]
        
        if len(scores_nonzero) > 0:
            ax.hist(np.log1p(scores_nonzero), bins=30, color='steelblue', 
                   edgecolor='white', alpha=0.8)
            ax.set_xlabel('log(Altmetric Score + 1)', fontsize=10)
            ax.set_ylabel('Number of Papers', fontsize=10)
            n_with = len(scores_nonzero)
            pct = n_with / len(scores) * 100
            ax.set_title(f'Altmetric Score Distribution ({n_with} papers, {pct:.1f}%)', 
                        fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No altmetric scores available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        plot.canvas.draw()
    
    def _plot_sources(self, result: Dict, parent: tk.Frame):
        """Plot source breakdown - single color, ordered by value."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(8, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        df = result["summary_df"]
        source_cols = ["Twitter", "Mendeley", "News", "Blogs", "Reddit", 
                      "Wikipedia", "Policy", "Patents", "GitHub"]
        
        available = [c for c in source_cols if c in df.columns]
        
        if available:
            totals = {k: df[k].sum() for k in available if df[k].sum() > 0}
            if totals:
                # Sort ascending for horizontal bar (highest at top)
                sorted_items = sorted(totals.items(), key=lambda x: x[1])
                names = [x[0] for x in sorted_items]
                values = [x[1] for x in sorted_items]
                
                bars = ax.barh(range(len(names)), values, color='steelblue', 
                              alpha=0.8, height=0.6, edgecolor='white')
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names)
                ax.set_xlabel('Total Mentions/Readers', fontsize=10)
                ax.set_title('Total by Source', fontsize=11, fontweight='bold')
                
                max_val = max(values)
                for bar, val in zip(bars, values):
                    ax.text(bar.get_width() + max_val * 0.02, 
                           bar.get_y() + bar.get_height()/2,
                           f'{val:,}', va='center', fontsize=9)
                ax.set_xlim(0, max_val * 1.15)
            else:
                ax.text(0.5, 0.5, 'No source data', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No source columns', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        plot.canvas.draw()
    
    def _plot_trends(self, result: Dict, parent: tk.Frame):
        """Plot temporal trends - mean score over time."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(8, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        temporal = result.get("temporal_trends", pd.DataFrame())
        
        if len(temporal) == 0:
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            plot.canvas.draw()
            return
        
        df = temporal.dropna(subset=["Year"])
        df = df[(df["Year"] >= 2000) & (df["Year"] <= 2030)]
        
        if "Mean Score" in df.columns and len(df) > 1:
            ax.plot(df["Year"], df["Mean Score"], marker='o', linewidth=2, 
                   color='steelblue', markersize=6)
            ax.fill_between(df["Year"], df["Mean Score"], alpha=0.2, color='steelblue')
            ax.set_xlabel('Publication Year', fontsize=10)
            ax.set_ylabel('Mean Altmetric Score', fontsize=10)
            ax.set_title('Mean Altmetric Score by Year', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient trend data', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        plot.canvas.draw()
    
    def _show_top_papers(self, result: Dict, parent: tk.Frame):
        """Show top papers table."""
        top_papers = result.get("top_papers", pd.DataFrame())
        
        if len(top_papers) == 0:
            tk.Label(parent, text="No top papers data", font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        table = DataTable(parent, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        display_cols = ["Title", "Year", "Altmetric Score", "Citations", "Twitter", "Mendeley", "News"]
        available_cols = [c for c in display_cols if c in top_papers.columns]
        
        display_df = top_papers[available_cols].copy()
        if "Title" in display_df.columns:
            display_df["Title"] = display_df["Title"].apply(
                lambda x: str(x)[:60] + "..." if pd.notna(x) and len(str(x)) > 60 else x
            )
        table.set_data(display_df)
    
    def _show_coverage(self, result: Dict, parent: tk.Frame):
        """Show source coverage details."""
        coverage = result.get("source_coverage", {})
        stats = result.get("statistics", {})
        
        canvas = tk.Canvas(parent, bg=self.theme["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.theme["bg_card"])
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        cov_card = Card(scrollable, title="📡 Source Coverage", theme=self.theme_name)
        cov_card.pack(fill=tk.X, pady=4)
        
        for source, pct in sorted(coverage.items(), key=lambda x: -x[1]):
            row = tk.Frame(cov_card.content, bg=self.theme["bg_card"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=source, font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                    fg=self.theme["text_primary"], width=15, anchor=tk.W).pack(side=tk.LEFT)
            bar_frame = tk.Frame(row, bg=self.theme["bg_secondary"], height=16, width=200)
            bar_frame.pack(side=tk.LEFT, padx=8)
            bar_frame.pack_propagate(False)
            tk.Frame(bar_frame, bg='#4CAF50', height=16, width=int(pct * 2)).pack(side=tk.LEFT)
            tk.Label(row, text=f"{pct:.1f}%", font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                    fg=self.theme["text_primary"]).pack(side=tk.LEFT, padx=8)
        
        stats_card = Card(scrollable, title="📊 Statistics", theme=self.theme_name)
        stats_card.pack(fill=tk.X, pady=4)
        
        stat_items = [("Total Papers", stats.get("total_papers", 0)),
                     ("With Attention", stats.get("with_attention", 0)),
                     ("Attention Rate", f"{stats.get('attention_rate', 0):.1f}%"),
                     ("Mean Score", f"{stats.get('mean_score', 0):.2f}"),
                     ("Max Score", f"{stats.get('max_score', 0):.2f}")]
        
        if "citation_altmetric_correlation" in stats:
            stat_items.append(("Citation-Altmetric Corr.", f"{stats['citation_altmetric_correlation']:.3f}"))
        
        for label, value in stat_items:
            row = tk.Frame(stats_card.content, bg=self.theme["bg_card"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label + ":", font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                    fg=self.theme["text_muted"], width=25, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(row, text=str(value), font=FONTS.get_font("body_bold"), bg=self.theme["bg_card"],
                    fg=self.theme["text_primary"]).pack(side=tk.LEFT)
    
    def _add_export_buttons(self, result: Dict):
        """Add export buttons."""
        btn_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=8)
        
        ActionButton(btn_frame, text="Export to Excel", icon="📊",
                    command=lambda: self._export_excel(result), theme=self.theme_name).pack(side=tk.LEFT, padx=4)
        ActionButton(btn_frame, text="Save Plots", icon="💾",
                    command=lambda: self._save_plots(result), theme=self.theme_name).pack(side=tk.LEFT, padx=4)
    
    def _export_excel(self, result: Dict):
        """Export results to Excel."""
        filename = filedialog.asksaveasfilename(defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")], title="Save Altmetrics Results")
        if not filename:
            return
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                result["summary_df"].to_excel(writer, sheet_name="Summary", index=False)
                if "top_papers" in result:
                    result["top_papers"].to_excel(writer, sheet_name="Top Papers", index=False)
                if "temporal_trends" in result and len(result["temporal_trends"]) > 0:
                    result["temporal_trends"].to_excel(writer, sheet_name="Trends", index=False)
            messagebox.showinfo("Success", f"Results exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def _save_plots(self, result: Dict):
        """Save plots to files."""
        base = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG files", "*.png")], title="Save Altmetrics Plots")
        if not base:
            return
        if base.endswith(('.png', '.pdf', '.svg')):
            base = base.rsplit('.', 1)[0]
        try:
            from biblium import plotbib
            plotbib.plot_altmetrics(result, filename=base + "_overview", show=False)
            plotbib.plot_altmetric_sources(result, filename=base + "_sources", show=False)
            messagebox.showinfo("Success", f"Plots saved to:\n{base}_*.png/pdf/svg")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{str(e)}")
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading indicator."""
        try:
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
        tk.Label(frame, text="⏳", font=("Segoe UI", 32), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]).pack(pady=(20, 10))
        tk.Label(frame, text=message, font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]).pack()
    
    def _show_error(self, message: str):
        """Show error message."""
        try:
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
        tk.Label(self.results_tab, text=f"❌ Error\n\n{message}", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["error"], justify=tk.CENTER,
                wraplength=400).pack(expand=True)
    
    def _create_info_content(self, parent):
        """Create Info tab with comprehensive documentation."""
        info_text = """
ALTMETRICS ANALYSIS
═══════════════════

Analyze alternative impact metrics beyond traditional citations.

METRICS ANALYZED
────────────────
• Social Media
  - Twitter/X mentions
  - Facebook shares
  - Reddit discussions
  
• Academic Social
  - Mendeley readers
  - CiteULike saves
  
• Media Coverage
  - News mentions
  - Blog posts
  - Wikipedia citations
  
• Real-World Impact
  - Policy document citations
  - Patent citations

COMPOSITE SCORES
────────────────
• Social Score: Social media attention
• Scholarly Score: Academic engagement
• Public Score: News/blog/Wikipedia
• Practice Score: Policy/patent impact
• Overall Attention Score: Weighted total

DATA REQUIREMENTS
─────────────────
• DOIs for Altmetric API lookup
• Or pre-existing altmetric columns
• OpenAlex includes some attention data

OUTPUT
──────
• Overview: Score distribution
• Sources: Breakdown by platform
• Trends: Attention over time
• Top Papers: Highest attention
• Coverage: Data completeness

INTERPRETATION
──────────────
• Compare within field (norms vary)
• Health/environment score higher
• New papers: high altmetrics, few citations
• Controversial topics get more attention
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
        """Clean up resources."""
        try:
            plt.close('all')
        except:
            pass
        self._current_result = None
        super().destroy()
