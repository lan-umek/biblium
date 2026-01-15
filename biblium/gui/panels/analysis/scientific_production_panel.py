# -*- coding: utf-8 -*-
"""
Scientific Production Panel
===========================
Analyze annual scientific production and citations over time.
Uses biblium's get_production() and plotbib.plot_timeseries().
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
from typing import Dict, List, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledSpinbox
from biblium.gui.widgets.plots import add_plot_context_menu, make_canvas_resizable

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


class ScientificProductionPanel(BasePanel):
    """Panel for analyzing scientific production over time."""
    
    title = "Scientific Production"
    icon = "📊"
    description = "Analyze annual document production and citation trends"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._result_df = None
        self._current_fig = None
        self._canvas = None
        self._toolbar = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Time Period Card
        period_card = Card(self.options_content, title="📅 Time Period", theme=self.theme_name)
        period_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Year range
        year_frame = tk.Frame(period_card.content, bg=self.theme["bg_card"])
        year_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            year_frame, text="From:", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.year_from_spin = tk.Spinbox(
            year_frame, from_=1900, to=2030, width=6,
            font=FONTS.get_font("body"),
        )
        self.year_from_spin.pack(side=tk.LEFT, padx=(4, 16))
        self.year_from_spin.delete(0, tk.END)
        self.year_from_spin.insert(0, "1900")
        
        tk.Label(
            year_frame, text="To:", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.year_to_spin = tk.Spinbox(
            year_frame, from_=1900, to=2030, width=6,
            font=FONTS.get_font("body"),
        )
        self.year_to_spin.pack(side=tk.LEFT, padx=4)
        self.year_to_spin.delete(0, tk.END)
        self.year_to_spin.insert(0, "2030")
        
        # Merge years option
        merge_frame = tk.Frame(period_card.content, bg=self.theme["bg_card"])
        merge_frame.pack(fill=tk.X, pady=4)
        
        self.use_cut_year_var = tk.BooleanVar(value=False)
        self.use_cut_year_cb = tk.Checkbutton(
            merge_frame,
            text="Merge years before:",
            variable=self.use_cut_year_var,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            selectcolor=self.theme["bg_secondary"],
        )
        self.use_cut_year_cb.pack(side=tk.LEFT)
        
        self.cut_year_spin = tk.Spinbox(
            merge_frame, from_=1950, to=2020, width=6,
            font=FONTS.get_font("body"),
        )
        self.cut_year_spin.pack(side=tk.LEFT, padx=4)
        self.cut_year_spin.delete(0, tk.END)
        self.cut_year_spin.insert(0, "2000")
        
        # Plot Options Card
        plot_card = Card(self.options_content, title="📈 Plot Options", theme=self.theme_name)
        plot_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.bar_metric = LabeledCombobox(
            plot_card.content, label="Bars (primary):",
            values=["Documents", "Total Citations", "None"],
            default="Documents",
            theme=self.theme_name, label_width=14,
        )
        self.bar_metric.pack(fill=tk.X, pady=4)
        
        self.line_metric = LabeledCombobox(
            plot_card.content, label="Line (secondary):",
            values=["Cumulative Citations", "Cumulative Documents", "Total Citations", "Avg Citations", "None"],
            default="Cumulative Citations",
            theme=self.theme_name, label_width=14,
        )
        self.line_metric.pack(fill=tk.X, pady=4)
        
        # Show bar labels checkbox
        self.show_bar_labels_var = tk.BooleanVar(value=False)
        bar_labels_frame = tk.Frame(plot_card.content, bg=self.theme["bg_card"])
        bar_labels_frame.pack(fill=tk.X, pady=4)
        
        self.show_bar_labels_cb = tk.Checkbutton(
            bar_labels_frame,
            text="Show bar labels",
            variable=self.show_bar_labels_var,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            selectcolor=self.theme["bg_secondary"],
        )
        self.show_bar_labels_cb.pack(side=tk.LEFT)
        
        # Options Card
        options_card = Card(self.options_content, title="⚙️ Options", theme=self.theme_name)
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Include cumulative checkbox
        self.cumulative_var = tk.BooleanVar(value=True)
        cumulative_frame = tk.Frame(options_card.content, bg=self.theme["bg_card"])
        cumulative_frame.pack(fill=tk.X, pady=4)
        
        self.cumulative_check = tk.Checkbutton(
            cumulative_frame,
            text="Include Cumulative Counts",
            variable=self.cumulative_var,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            selectcolor=self.theme["bg_secondary"],
        )
        self.cumulative_check.pack(side=tk.LEFT)
        
        # Predict last year checkbox
        self.predict_var = tk.BooleanVar(value=True)
        predict_frame = tk.Frame(options_card.content, bg=self.theme["bg_card"])
        predict_frame.pack(fill=tk.X, pady=4)
        
        self.predict_check = tk.Checkbutton(
            predict_frame,
            text="Predict Incomplete Last Year",
            variable=self.predict_var,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            selectcolor=self.theme["bg_secondary"],
        )
        self.predict_check.pack(side=tk.LEFT)
        
        # Run button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_primary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        self.run_btn = ActionButton(
            btn_frame, text="Generate Plot",
            command=self._run_analysis,
            icon="▶",
            theme=self.theme_name,
        )
        self.run_btn.pack(fill=tk.X)
        
        # Export buttons
        self.export_plot_btn = ThemedButton(
            btn_frame, text="Export Plot",
            command=self._export_plot,
            style="secondary",
            icon="🖼️",
            theme=self.theme_name,
        )
        self.export_plot_btn.pack(fill=tk.X, pady=(8, 0))
        
        self.export_data_btn = ThemedButton(
            btn_frame, text="Export Data",
            command=self._export_data,
            style="secondary",
            icon="📊",
            theme=self.theme_name,
        )
        self.export_data_btn.pack(fill=tk.X, pady=(8, 0))
    
    def _create_results(self):
        """Create the results panel."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Plot tab
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="📈 Plot")
        
        # Data tab
        self.data_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.data_frame, text="📋 Data")
        
        # Info tab
        self.info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.info_frame, text="ℹ️ Info")
        
        # Info tab
        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        
        self._show_placeholder()
        self._create_info_tab()
    
    def _show_placeholder(self):
        """Show placeholder message."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame,
            text="Click 'Generate Plot' to see\nscientific production over time",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _create_info_tab(self):
        """Create the info tab."""
        for widget in self.info_frame.winfo_children():
            widget.destroy()
        
        info_text = """
SCIENTIFIC PRODUCTION ANALYSIS
══════════════════════════════

This analysis shows the overall publication output and citation 
patterns over time for the entire dataset.

TIME PERIOD
───────────
• From/To: Filter years to display
• Merge years before: Combine early years into a single bar

PLOT OPTIONS
────────────
• Bars (primary): What to show as bars (Documents, Citations)
• Line (secondary): What to show as line overlay
  - Cumulative Citations: Running total of citations
  - Cumulative Documents: Running total of documents
  - Total Citations: Citations per year
  - Avg Citations: Average citations per document per year
• Show bar labels: Display values on top of bars

OPTIONS
───────
• Include Cumulative Counts: Compute cumulative columns
• Predict Incomplete Last Year: Estimate full-year values

INTERPRETATION
──────────────
• Rising bars indicate growth in publication output
• S-curve in cumulative documents suggests field maturation
• Higher average citations for older years is expected (citation lag)
"""
        
        text_widget = tk.Text(
            self.info_frame,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            wrap=tk.WORD,
            padx=16, pady=16,
            relief=tk.FLAT,
        )
        text_widget.insert("1.0", info_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
    
    def _set_year_defaults(self):
        """Set year defaults from data."""
        if not self.bib or not hasattr(self.bib, 'df'):
            return
        
        if 'Year' in self.bib.df.columns:
            years = pd.to_numeric(self.bib.df['Year'], errors='coerce').dropna()
            if len(years) > 0:
                self.year_from_spin.delete(0, tk.END)
                self.year_from_spin.insert(0, str(int(years.min())))
                self.year_to_spin.delete(0, tk.END)
                self.year_to_spin.insert(0, str(int(years.max())))
    
    def _run_analysis(self):
        """Run the scientific production analysis."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self.run_btn.config(state=tk.DISABLED, text="⏳ Analyzing...")
        
        # Capture values
        year_from = int(self.year_from_spin.get())
        year_to = int(self.year_to_spin.get())
        use_cut_year = self.use_cut_year_var.get()
        cut_year = int(self.cut_year_spin.get()) if use_cut_year else None
        bar_metric = self.bar_metric.get()
        line_metric = self.line_metric.get()
        show_bar_labels = self.show_bar_labels_var.get()
        cumulative = self.cumulative_var.get()
        predict = self.predict_var.get()
        
        def do_analysis():
            error_info = None
            try:
                from biblium import utilsbib, plotbib
                
                # Use biblium's get_production() method
                if hasattr(self.bib, 'get_production'):
                    self.bib.get_production(
                        cumulative=cumulative,
                        predict_last_year=predict,
                    )
                    data = self.bib.production_df.copy()
                else:
                    # Fallback to utilsbib directly
                    data = utilsbib.get_scientific_production(
                        self.bib.df,
                        cumulative=cumulative,
                        predict_last_year=predict,
                    )
                
                if data is None or data.empty:
                    raise ValueError("No production data available")
                
                # Apply cut_year grouping (like biblium's plot_timeseries)
                if cut_year is not None:
                    before_df = data[data['Year'] < cut_year].copy()
                    after_df = data[data['Year'] >= cut_year].copy()
                    
                    if not before_df.empty:
                        combined = {'Year': f"<{cut_year}"}
                        for col in data.columns:
                            if col == 'Year':
                                continue
                            if 'Cumulative' in str(col):
                                combined[col] = before_df[col].max()
                            elif pd.api.types.is_numeric_dtype(data[col]):
                                combined[col] = before_df[col].sum()
                            else:
                                combined[col] = None
                        
                        before_row = pd.DataFrame([combined])
                        data = pd.concat([before_row, after_df], ignore_index=True)
                    else:
                        data = after_df
                else:
                    # Filter by year range
                    data = data[(data['Year'] >= year_from) & (data['Year'] <= year_to)].copy()
                
                # Standardize column names for plotting
                col_mapping = {
                    'Total_Citations': 'Total Citations',
                    'Cumulative_Citations': 'Cumulative Citations',
                    'Cumulative_Documents': 'Cumulative Documents',
                }
                data = data.rename(columns={k: v for k, v in col_mapping.items() if k in data.columns})
                
                # Add average citations if needed
                if 'Documents' in data.columns:
                    cit_col = 'Total Citations' if 'Total Citations' in data.columns else 'Total_Citations'
                    if cit_col in data.columns:
                        data['Avg Citations'] = (data[cit_col] / data['Documents'].replace(0, 1)).round(2)
                
                self._result_df = data
                
                # Create the plot using plotbib.plot_timeseries
                bar_y = None if bar_metric == "None" else bar_metric
                line_y = None if line_metric == "None" else line_metric
                
                # Map metric names to column names
                metric_to_col = {
                    'Documents': 'Documents',
                    'Total Citations': 'Total Citations',
                    'Cumulative Citations': 'Cumulative Citations',
                    'Cumulative Documents': 'Cumulative Documents',
                    'Avg Citations': 'Avg Citations',
                }
                
                if bar_y and bar_y in metric_to_col:
                    bar_y = metric_to_col[bar_y]
                if line_y and line_y in metric_to_col:
                    line_y = metric_to_col[line_y]
                
                # Check if columns exist
                if bar_y and bar_y not in data.columns:
                    bar_y = None
                if line_y and line_y not in data.columns:
                    line_y = None
                
                # Use plotbib.plot_timeseries
                plotbib.plot_timeseries(
                    data,
                    x='Year',
                    bar_y=bar_y,
                    line_y=line_y,
                    bar_labels=show_bar_labels,
                    filename=None,
                    show=False,
                )
                
                fig = plt.gcf()
                fig.suptitle('Scientific Production', fontsize=12, y=1.02)
                fig.tight_layout()
                
                self._current_fig = fig
                
                self.after(0, self._on_analysis_success)
                
            except Exception as exc:
                import traceback
                error_info = (str(exc), traceback.format_exc())
                self.after(0, lambda ei=error_info: self._on_analysis_error(ei[0], ei[1]))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _on_analysis_success(self):
        """Handle successful analysis."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Generate Plot")
        self._update_plot_tab()
        self._update_data_tab()
        self.notebook.select(0)
    
    def _on_analysis_error(self, error_msg: str, traceback_str: str):
        """Handle analysis error."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Generate Plot")
        messagebox.showerror("Analysis Error", f"Error: {error_msg}\n\n{traceback_str[:500]}")
    
    def _update_plot_tab(self):
        """Update the plot tab."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        if self._current_fig is None:
            tk.Label(
                self.plot_frame,
                text="No plot available",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["danger"],
            ).pack(expand=True)
            return
        
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            
            # Add AI button header at top
            ai_header = tk.Frame(self.plot_frame, bg=self.theme["bg_card"])
            ai_header.pack(fill=tk.X, padx=4, pady=(4, 0))
            
            ai_btn = tk.Button(
                ai_header,
                text="🤖 AI Describe Plot",
                font=FONTS.get_font("body"),
                bg=self.theme["accent_primary"],
                fg="white",
                relief=tk.FLAT,
                cursor="hand2",
                padx=12,
                pady=4,
                command=lambda: self._ai_describe_plot(self.plot_frame),
            )
            ai_btn.pack(side=tk.RIGHT, padx=4)
            
            tk.Label(
                ai_header,
                text="Right-click chart for more options",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(side=tk.LEFT, padx=4)
            
            container = tk.Frame(self.plot_frame, bg=self.theme["bg_card"])
            container.pack(fill=tk.BOTH, expand=True)
            
            canvas = FigureCanvasTkAgg(self._current_fig, master=container)
            canvas.draw()
            
            toolbar_frame = tk.Frame(container, bg=self.theme["bg_card"])
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            _canvas_widget = canvas.get_tk_widget()
            _canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Add right-click save menu and resize handling
            add_plot_context_menu(_canvas_widget, self._current_fig)
            make_canvas_resizable(canvas, self._current_fig, container)
            
            self._canvas = canvas
            self._toolbar = toolbar
            
        except Exception as e:
            tk.Label(
                self.plot_frame,
                text=f"Plot generated. Use 'Export Plot' to save.\n\n({e})",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
                wraplength=400,
            ).pack(expand=True, padx=20, pady=20)
    
    def _ai_describe_plot(self, container):
        """Generate AI description of the current plot."""
        from biblium.gui.widgets.tables import DataTable
        settings = DataTable.get_llm_settings()
        
        if not settings.get("api_key"):
            from tkinter import messagebox
            messagebox.showinfo("Configure AI", 
                "Please configure AI settings first:\n\n"
                "1. Expand '🤖 AI Analysis Settings' below\n"
                "2. Enter your API key\n"
                "3. Click this button again")
            return
        
        # Extract plot info
        plot_info = {"type": "bar chart with line overlay", "title": "", "x_label": "", "y_label": "", "data_summary": "", "context": ""}
        try:
            if self._current_fig and self._current_fig.axes:
                ax = self._current_fig.axes[0]
                plot_info["title"] = ax.get_title() or "Annual Scientific Production"
                plot_info["x_label"] = ax.get_xlabel() or "Year"
                plot_info["y_label"] = ax.get_ylabel() or "Documents"
                
                # Get data from result dataframe
                if self._result_df is not None and not self._result_df.empty:
                    plot_info["data_summary"] = f"{len(self._result_df)} time periods, "
                    if 'Documents' in self._result_df.columns:
                        docs = self._result_df['Documents']
                        plot_info["data_summary"] += f"Documents: {docs.sum()} total, range [{docs.min()}, {docs.max()}], "
                    if 'Cumulative Citations' in self._result_df.columns:
                        cites = self._result_df['Cumulative Citations']
                        plot_info["data_summary"] += f"Cumulative Citations: max {cites.max()}"
                
                legend = ax.get_legend()
                if legend:
                    texts = [t.get_text() for t in legend.get_texts()]
                    if texts:
                        plot_info["context"] = f"Legend: {', '.join(texts[:5])}"
        except Exception as e:
            plot_info["context"] = str(e)
        
        # Show loading
        if hasattr(self, '_ai_result_frame') and self._ai_result_frame:
            try: self._ai_result_frame.destroy()
            except: pass
        
        self._ai_loading_frame = tk.Frame(container, bg=self.theme["bg_card"])
        self._ai_loading_frame.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(self._ai_loading_frame, text="⏳ Generating AI description...",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], 
                fg=self.theme["text_muted"]).pack(pady=4)
        
        import threading
        def do_generate():
            try:
                from biblium.llm_utils import llm_describe_plot
                result = llm_describe_plot(
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
                self.after(0, lambda c=container, r=result: self._show_ai_plot_result(c, r))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.after(0, lambda c=container, msg=error_msg: self._show_ai_plot_result(c, msg))
        
        thread = threading.Thread(target=do_generate, daemon=True)
        thread.start()
    
    def _show_ai_plot_result(self, container, text):
        """Show AI plot description result."""
        if hasattr(self, '_ai_loading_frame') and self._ai_loading_frame:
            try: self._ai_loading_frame.destroy()
            except: pass
        if hasattr(self, '_ai_result_frame') and self._ai_result_frame:
            try: self._ai_result_frame.destroy()
            except: pass
        
        self._ai_result_frame = tk.Frame(container, bg=self.theme["bg_card"])
        self._ai_result_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0))
        
        header = tk.Frame(self._ai_result_frame, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        tk.Label(header, text="🤖 AI Plot Description", font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT, padx=4)
        
        def copy_text():
            try:
                self.clipboard_clear()
                self.clipboard_append(text)
                copy_btn.config(text="✓")
                self.after(1500, lambda: copy_btn.config(text="📋"))
            except: pass
        
        tk.Button(header, text="✕", font=("Segoe UI", 8), bg=self.theme["bg_secondary"],
                  fg=self.theme["text_primary"], relief=tk.FLAT,
                  command=lambda: self._ai_result_frame.destroy(), cursor="hand2", width=2).pack(side=tk.RIGHT, padx=2)
        copy_btn = tk.Button(header, text="📋", font=("Segoe UI", 8), bg=self.theme["bg_secondary"],
                             fg=self.theme["text_primary"], relief=tk.FLAT, command=copy_text, cursor="hand2", width=2)
        copy_btn.pack(side=tk.RIGHT, padx=2)
        
        text_widget = tk.Text(self._ai_result_frame, wrap=tk.WORD, font=FONTS.get_font("small"),
                              bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
                              relief=tk.FLAT, height=4, padx=8, pady=4)
        text_widget.pack(fill=tk.X, padx=4, pady=(2, 4))
        text_widget.insert("1.0", text)
        def on_key(e):
            if e.state & 0x4 and e.keysym.lower() in ('c', 'a'): return
            return "break"
        text_widget.bind("<Key>", on_key)
        text_widget.bind("<Button-1>", lambda e: text_widget.focus_set())
    
    def _update_data_tab(self):
        """Update the data table tab."""
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        
        if self._result_df is None or self._result_df.empty:
            tk.Label(
                self.data_frame,
                text="No data available",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
            ).pack(expand=True)
            return
        
        from biblium.gui.widgets.tables import DataTable
        
        table = DataTable(
            self.data_frame,
            dataframe=self._result_df,
            theme=self.theme_name,
            show_index=False,
        )
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    
    def _export_plot(self):
        """Export the plot."""
        if self._current_fig is None:
            messagebox.showwarning("No Plot", "Please generate a plot first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
            ],
            title="Export Plot",
        )
        
        if filepath:
            try:
                self._current_fig.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Export Complete", f"Plot exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def _export_data(self):
        """Export the data table."""
        if self._result_df is None or self._result_df.empty:
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
            ],
            title="Export Data",
        )
        
        if filepath:
            try:
                if filepath.endswith(".xlsx"):
                    self._result_df.to_excel(filepath, index=False)
                else:
                    self._result_df.to_csv(filepath, index=False)
                messagebox.showinfo("Export Complete", f"Data exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def on_data_loaded(self, bib):
        """Handle data loaded event."""
        self.bib = bib
        self._cleanup_figure()
        self._result_df = None
        self._show_placeholder()
        self._set_year_defaults()
    
    def _cleanup_figure(self):
        """Clean up matplotlib figure."""
        if self._current_fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._current_fig)
            except:
                pass
            self._current_fig = None
        self._canvas = None
        self._toolbar = None
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
SCIENTIFIC PRODUCTION
═════════════════════

Comprehensive production analysis.

DOCUMENT METRICS
────────────────
• Total publications
• By document type
• By source type
• Annual counts

AUTHOR METRICS
──────────────
• Unique authors
• Authors per paper
• Single vs multi-author
• Author productivity

SOURCE METRICS
──────────────
• Unique sources
• Papers per source
• Core journals
• Bradford analysis

TEMPORAL METRICS
────────────────
• Date range
• Growth rate
• Trend direction
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
        self._cleanup_figure()
        super().destroy()
