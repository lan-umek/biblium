# -*- coding: utf-8 -*-
"""
Citation Distribution Panel
===========================
Panel for analyzing citation distribution characteristics.

@author: Claude (Anthropic) for Lan.Umek
@version: 2.7.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox
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
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class CitationDistributionPanel(BasePanel):
    """Panel for analyzing citation distribution."""
    
    title = "Citation Distribution"
    icon = "📊"
    description = "Analyze citation distribution and impact metrics"
    requires_data = True
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        if not self.bib:
            self._show_no_data_message()
            return
        
        # Info Card
        info_frame = tk.Frame(self.options_content, bg="#e8f5e9", relief=tk.FLAT, bd=1)
        info_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        info_inner = tk.Frame(info_frame, bg="#e8f5e9", padx=8, pady=6)
        info_inner.pack(fill=tk.X)
        
        tk.Label(
            info_inner, text="📊 Citation Distribution",
            font=FONTS.get_font("body_bold"), bg="#e8f5e9", fg="#2e7d32",
        ).pack(anchor=tk.W)
        
        tk.Label(
            info_inner, 
            text="Analyze how citations are distributed\nacross your publications.",
            font=FONTS.get_font("small"), bg="#e8f5e9", fg="#2e7d32",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        
        # Column Selection Card
        col_card = Card(self.options_content, title="🔧 Settings", theme=self.theme_name)
        col_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Find citation columns
        cit_cols = []
        for col in self.bib.df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ["cited", "citation", "tc", "times"]):
                cit_cols.append(col)
        
        if not cit_cols:
            cit_cols = list(self.bib.df.columns)
        
        self.citations_combo = LabeledCombobox(
            col_card.content, label="Citations column:",
            values=cit_cols, default=cit_cols[0] if cit_cols else "",
            theme=self.theme_name, label_width=15,
            tooltip="Column containing citation counts"
        )
        self.citations_combo.pack(fill=tk.X, pady=4)
        
        # Options Card
        options_card = Card(self.options_content, title="⚙️ Display Options", theme=self.theme_name)
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_histogram_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            options_card.content, label="Show histogram",
            variable=self.show_histogram_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_log_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            options_card.content, label="Show log-scale histogram",
            variable=self.show_log_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_classes_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            options_card.content, label="Show citation classes",
            variable=self.show_classes_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_metrics_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            options_card.content, label="Show detailed metrics",
            variable=self.show_metrics_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Action Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Analyze Distribution", icon="📊",
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
            "📈 Citation Distribution\n\n"
            "Analyze how citations are distributed across publications.\n\n"
            "Features:\n"
            "• Distribution histograms and density plots\n"
            "• Percentile rankings\n"
            "• Highly cited paper identification\n"
            "• Skewness and kurtosis statistics\n"
            "\n"
            "Citation distributions are typically highly skewed.\n\n"
            "Steps:\n"
            "1. Load a dataset with citation counts\n"
            "2. Select distribution type\n"
            "3. Configure bins and range\n"
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
        """Run citation distribution analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Analyzing citation distribution...")
        
        def do_analysis():
            try:
                citations_col = self.citations_combo.get()
                result = self.bib.analyze_citation_distribution(
                    citations_col=citations_col,
                    verbose=False
                )
                self.after(0, lambda r=result: self._display_results(r))
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _display_results(self, result: Dict):
        """Display citation distribution results."""
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
        
        # Summary Cards
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 12))
        
        grid.add_card(StatsCard(grid, "Papers", f"{result['n_papers']:,}", "📄", self.theme_name))
        grid.add_card(StatsCard(grid, "H-index", str(result['h_index']), "📈", self.theme_name))
        grid.add_card(StatsCard(grid, "G-index", str(result['g_index']), "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "Gini", f"{result['gini_coefficient']:.3f}", "⚖️", self.theme_name))
        
        # Second row of cards
        grid2 = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid2.pack(fill=tk.X, pady=(0, 12))
        
        grid2.add_card(StatsCard(grid2, "Mean", f"{result['basic_stats']['mean']:.1f}", "📉", self.theme_name))
        grid2.add_card(StatsCard(grid2, "Median", f"{result['basic_stats']['median']:.0f}", "📉", self.theme_name))
        grid2.add_card(StatsCard(grid2, "Max", f"{result['basic_stats']['max']:,}", "🔝", self.theme_name))
        grid2.add_card(StatsCard(grid2, "Uncited", f"{result['uncited']['percentage']:.1f}%", "⭕", self.theme_name))
        
        # Create Notebook (tabs) for different views
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        
        # Tab 1: Histogram
        if self.show_histogram_var.get() and HAS_MATPLOTLIB:
            hist_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(hist_frame, text="📊 Histogram")
            self._plot_histogram(result, hist_frame)
        
        # Tab 2: Log-scale Histogram
        if self.show_log_var.get() and HAS_MATPLOTLIB:
            log_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(log_frame, text="📈 Log Scale")
            self._plot_histogram_log(result, log_frame)
        
        # Tab 3: Citation Classes
        if self.show_classes_var.get() and HAS_MATPLOTLIB:
            classes_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(classes_frame, text="🏷️ Classes")
            self._plot_classes(result, classes_frame)
        
        # Tab 4: Detailed Metrics
        if self.show_metrics_var.get():
            metrics_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(metrics_frame, text="📋 Metrics")
            self._show_metrics_table(result, metrics_frame)
        
        # Tab 5: Percentiles
        percentiles_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(percentiles_frame, text="📏 Percentiles")

        

        # Info tab

        info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])

        notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        self._show_percentiles(result, percentiles_frame)
        
        # Buttons at the bottom
        self._add_export_buttons(result)
        
        self._current_result = result
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Citation Distribution"})
    
    def _plot_histogram(self, result: Dict, parent: tk.Frame):
        """Plot citation histogram."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(10, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        ax = fig.add_subplot(111)
        
        citations = result['citations']
        p99 = result['percentiles'][99]
        clipped = citations[citations <= p99]
        
        ax.hist(clipped, bins=50, color='#2196F3', alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.axvline(x=result['basic_stats']['mean'], color='#E91E63', linestyle='--', 
                  linewidth=2, label=f"Mean ({result['basic_stats']['mean']:.1f})")
        ax.axvline(x=result['basic_stats']['median'], color='#4CAF50', linestyle='-', 
                  linewidth=2, label=f"Median ({result['basic_stats']['median']:.0f})")
        
        ax.set_xlabel('Citations', fontsize=10)
        ax.set_ylabel('Number of Papers', fontsize=10)
        ax.set_title('Citation Distribution (clipped at 99th percentile)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', frameon=False, fontsize=8)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        plot.refresh()
    
    def _plot_histogram_log(self, result: Dict, parent: tk.Frame):
        """Plot citation histogram with log scale."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(10, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        ax = fig.add_subplot(111)
        
        citations = result['citations']
        positive_cit = citations[citations > 0]
        
        if len(positive_cit) > 0:
            log_bins = np.logspace(0, np.log10(max(positive_cit) + 1), 40)
            ax.hist(positive_cit, bins=log_bins, color='#2196F3', alpha=0.8, 
                   edgecolor='white', linewidth=0.5)
            ax.set_xscale('log')
            ax.set_xlabel('Citations (log scale)', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No cited papers', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_xlabel('Citations', fontsize=10)
        
        ax.set_ylabel('Number of Papers', fontsize=10)
        ax.set_title('Citation Distribution (Log Scale, excluding zeros)', fontsize=12, fontweight='bold')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        plot.refresh()
    
    def _plot_classes(self, result: Dict, parent: tk.Frame):
        """Plot citation classes."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(10, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        ax = fig.add_subplot(111)
        
        class_df = result['citation_classes']
        
        # Single color for all bars
        bars = ax.barh(class_df['Class'], class_df['Count'], color='#2196F3', 
                      alpha=0.8, edgecolor='white', height=0.6)
        
        # Add count and percentage labels
        max_count = max(class_df['Count']) if max(class_df['Count']) > 0 else 1
        for bar, (count, pct) in zip(bars, zip(class_df['Count'], class_df['Percentage'])):
            width = bar.get_width()
            ax.text(width + max_count * 0.02, bar.get_y() + bar.get_height()/2,
                   f'{count:,} ({pct:.1f}%)', va='center', fontsize=9)
        
        ax.set_xlabel('Number of Papers', fontsize=10)
        ax.set_title('Citation Classes', fontsize=12, fontweight='bold')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Extend x limit to fit labels
        ax.set_xlim(0, max_count * 1.35)
        
        fig.tight_layout()
        fig.subplots_adjust(left=0.25)  # More space for y-axis labels
        plot.refresh()
    
    def _show_metrics_table(self, result: Dict, parent: tk.Frame):
        """Show detailed metrics table."""
        # Create metrics dataframe
        metrics_data = [
            ("Total Papers", f"{result['n_papers']:,}"),
            ("Total Citations", f"{result['basic_stats']['sum']:,}"),
            ("", ""),
            ("Mean Citations", f"{result['basic_stats']['mean']:.2f}"),
            ("Median Citations", f"{result['basic_stats']['median']:.1f}"),
            ("Std Deviation", f"{result['basic_stats']['std']:.2f}"),
            ("Max Citations", f"{result['basic_stats']['max']:,}"),
            ("Min Citations", f"{result['basic_stats']['min']}"),
            ("", ""),
            ("H-index", f"{result['h_index']}"),
            ("G-index", f"{result['g_index']}"),
            ("Gini Coefficient", f"{result['gini_coefficient']:.4f}"),
            ("", ""),
            ("Skewness", f"{result['skewness']:.2f}"),
            ("Kurtosis", f"{result['kurtosis']:.2f}"),
            ("", ""),
            ("Uncited Papers", f"{result['uncited']['count']:,} ({result['uncited']['percentage']:.1f}%)"),
            ("Highly Cited Threshold", f"≥{result['highly_cited']['threshold']:.0f}"),
            ("Highly Cited Count", f"{result['highly_cited']['count']:,}"),
        ]
        
        metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])
        
        tk.Label(
            parent, text="Detailed Citation Metrics",
            font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(8, 4), padx=8)
        
        table = DataTable(parent, theme=self.theme_name, height=18)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        table.set_data(metrics_df)
    
    def _show_percentiles(self, result: Dict, parent: tk.Frame):
        """Show percentiles table."""
        percentiles_data = []
        for p, v in sorted(result['percentiles'].items()):
            percentiles_data.append({
                "Percentile": f"{p}th",
                "Citations": f"{v:.0f}",
                "Interpretation": self._interpret_percentile(p)
            })
        
        percentiles_df = pd.DataFrame(percentiles_data)
        
        tk.Label(
            parent, text="Citation Percentiles",
            font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(8, 4), padx=8)
        
        tk.Label(
            parent, 
            text="A paper at the Xth percentile has more citations than X% of papers.",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W, padx=8)
        
        table = DataTable(parent, theme=self.theme_name, height=8)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))
        table.set_data(percentiles_df)
    
    def _interpret_percentile(self, p: int) -> str:
        """Interpret what a percentile means."""
        interpretations = {
            25: "Lower quartile",
            50: "Median (typical paper)",
            75: "Upper quartile",
            90: "Top 10% threshold",
            95: "Top 5% threshold",
            99: "Top 1% threshold",
        }
        return interpretations.get(p, "")
    
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
        """Export citation distribution data."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Export Citation Distribution Data"
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.xlsx'):
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    # Metrics sheet
                    metrics_df = pd.DataFrame([
                        {"Metric": "Total Papers", "Value": result['n_papers']},
                        {"Metric": "Total Citations", "Value": result['basic_stats']['sum']},
                        {"Metric": "Mean Citations", "Value": result['basic_stats']['mean']},
                        {"Metric": "Median Citations", "Value": result['basic_stats']['median']},
                        {"Metric": "Std Deviation", "Value": result['basic_stats']['std']},
                        {"Metric": "Max Citations", "Value": result['basic_stats']['max']},
                        {"Metric": "H-index", "Value": result['h_index']},
                        {"Metric": "G-index", "Value": result['g_index']},
                        {"Metric": "Gini Coefficient", "Value": result['gini_coefficient']},
                        {"Metric": "Skewness", "Value": result['skewness']},
                        {"Metric": "Kurtosis", "Value": result['kurtosis']},
                        {"Metric": "Uncited Count", "Value": result['uncited']['count']},
                        {"Metric": "Uncited Percentage", "Value": result['uncited']['percentage']},
                    ])
                    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
                    
                    # Percentiles sheet
                    perc_df = pd.DataFrame([
                        {"Percentile": p, "Citations": v} 
                        for p, v in sorted(result['percentiles'].items())
                    ])
                    perc_df.to_excel(writer, sheet_name="Percentiles", index=False)
                    
                    # Classes sheet
                    result['citation_classes'].to_excel(writer, sheet_name="Citation Classes", index=False)
                    
                    # Histogram data
                    result['histogram_data'].to_excel(writer, sheet_name="Histogram Data", index=False)
            else:
                # Just export classes for CSV
                result['citation_classes'].to_csv(filepath, index=False)
            
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
            
            plotbib.plot_citation_distribution(result, filename=base, show=False)
            
            messagebox.showinfo("Success", f"Plots saved to:\n{base}.png/pdf/svg")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{str(e)}")
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading indicator."""
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
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
CITATION DISTRIBUTION
═════════════════════

Analyze how citations are distributed across publications.

DISTRIBUTION CHARACTERISTICS
────────────────────────────
Citation distributions are highly skewed:
• Few papers get many citations
• Most papers get few/no citations
• Follows power law distribution

KEY STATISTICS
──────────────
• Total Citations: Sum across all papers
• Mean: Average citations per paper
• Median: Middle value (more robust)
• Max: Highest cited paper
• Uncited: Papers with zero citations

PERCENTILE ANALYSIS
───────────────────
• Top 1%: Highly cited threshold
• Top 10%: High impact threshold
• Top 25%: Above average
• Papers at each percentile level

VISUALIZATION
─────────────
• Histogram: Citation frequency
• Log-scale: Better for skewed data
• Box plot: Quartiles and outliers
• Cumulative distribution

SKEWNESS METRICS
────────────────
• Skewness coefficient
• Gini coefficient (inequality)
• 80/20 ratio (concentration)

INTERPRETATION
──────────────
• Mean >> Median: High skewness
• High Gini: Concentrated citations
• Compare within field and year cohort
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

    def _show_error(self, message: str):
        """Show error message."""
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
