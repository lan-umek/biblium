# -*- coding: utf-8 -*-
"""
Diversity Indices Panel
=======================
Compute and visualize research diversity indices (Shannon, Simpson, Gini)
for bibliometric entities.

@author: Lan.Umek
@version: 2.9.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
from typing import Dict, List, Optional, Tuple

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import add_plot_context_menu, make_canvas_resizable

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DiversityIndicesPanel(BasePanel):
    """
    Panel for computing research diversity indices.
    
    Computes Shannon, Simpson, and Gini indices for bibliometric entities
    to measure diversity across sources, authors, countries, keywords, etc.
    """
    
    title = "Diversity Indices"
    icon = "📊"
    description = "Compute Shannon, Simpson, and Gini diversity indices for research entities"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._result = None
        self._current_fig = None
        self._entity_vars = {}
        self._canvas = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Entity Selection Card
        entity_card = Card(self.options_content, title="📋 Entities to Analyze", theme=self.theme_name)
        entity_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Info label
        info_label = tk.Label(
            entity_card.content,
            text="Select entities for diversity analysis:",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
        )
        info_label.pack(fill=tk.X, pady=(0, 8))
        
        # Entity checkboxes
        self._entity_checkboxes_frame = tk.Frame(entity_card.content, bg=self.theme["bg_card"])
        self._entity_checkboxes_frame.pack(fill=tk.X)
        
        # Available entities (will be updated when data loads)
        default_entities = [
            "Sources", "Authors", "Countries", "Affiliations",
            "Author Keywords", "Index Keywords", "Subject Areas",
            "Document Types", "SDGs", "Years"
        ]
        
        for entity in default_entities:
            var = tk.BooleanVar(value=True)
            self._entity_vars[entity] = var
            
            cb = tk.Checkbutton(
                self._entity_checkboxes_frame,
                text=entity,
                variable=var,
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                activebackground=self.theme["bg_card"],
                selectcolor=self.theme["bg_card"],
            )
            cb.pack(anchor=tk.W, pady=1)
        
        # Select/Deselect all buttons
        btn_frame = tk.Frame(entity_card.content, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=(8, 0))
        
        select_all_btn = tk.Button(
            btn_frame, text="Select All",
            command=self._select_all_entities,
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
            relief=tk.FLAT,
        )
        select_all_btn.pack(side=tk.LEFT, padx=(0, 4))
        
        deselect_all_btn = tk.Button(
            btn_frame, text="Deselect All",
            command=self._deselect_all_entities,
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
            relief=tk.FLAT,
        )
        deselect_all_btn.pack(side=tk.LEFT)
        
        # Benchmark Card
        bench_card = Card(self.options_content, title="🌐 Benchmarking", theme=self.theme_name)
        bench_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.benchmark_var = tk.BooleanVar(value=True)
        self.benchmark_check = LabeledCheckbox(
            bench_card.content, label="Compare against OpenAlex global",
            variable=self.benchmark_var,
            theme=self.theme_name,
        )
        self.benchmark_check.pack(fill=tk.X, pady=4)
        
        bench_help = tk.Label(
            bench_card.content,
            text="Fetches global diversity from OpenAlex for comparison.\nSupports: Sources, Countries, Document Types, SDGs",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.LEFT,
        )
        bench_help.pack(fill=tk.X, pady=(0, 4))
        
        # Visualization Card
        viz_card = Card(self.options_content, title="📈 Visualization", theme=self.theme_name)
        viz_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.plot_type = LabeledCombobox(
            viz_card.content, label="Plot Type:",
            values=["Radar Chart", "Bar Chart"],
            default="Radar Chart",
            theme=self.theme_name, label_width=12,
        )
        self.plot_type.pack(fill=tk.X, pady=4)
        
        # Run button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_primary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        self.run_btn = ActionButton(
            btn_frame, text="Compute Diversity",
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
        self.notebook.add(self.plot_frame, text="📊 Visualization")
        
        # Data tab
        self.data_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.data_frame, text="📋 Data")
        
        # Comparison tab (for benchmark)
        self.comparison_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.comparison_frame, text="⚖️ Comparison")
        
        # Summary tab
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📝 Summary")
        
        # Info tab
        self.info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.info_frame, text="ℹ️ Info")

        

        # Info tab

        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])

        self.notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        # Initialize with placeholder
        self._show_placeholder()
        self._create_info_tab()
    
    def _show_placeholder(self):
        """Show placeholder message."""
        # Clean up any existing matplotlib figure
        if self._current_fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._current_fig)
            except:
                pass
            self._current_fig = None
        
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame,
            text="Select entities and click 'Compute Diversity'\nto analyze research diversity",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _create_info_tab(self):
        """Create the info tab with documentation."""
        for widget in self.info_frame.winfo_children():
            widget.destroy()
        
        info_text = """
RESEARCH DIVERSITY INDICES
══════════════════════════

This analysis computes diversity metrics for bibliometric entities
to measure how diverse/concentrated your research corpus is.

DIVERSITY INDICES
─────────────────

Shannon Index (H')
• Measures entropy/uncertainty in the distribution
• H = -Σ(pᵢ × ln(pᵢ))
• Normalized to 0-1 scale: H' = H / ln(n)
• Higher = more diverse (evenly distributed)

Simpson Index (1-D)
• Probability that two randomly selected items are different
• D = Σ(pᵢ²), reported as 1-D (diversity form)
• Range: 0 to 1
• Higher = more diverse

Gini Index (G)
• Measures inequality/concentration
• Range: 0 to 1
• 0 = perfect equality (all categories equal)
• 1 = perfect inequality (one category dominates)
• Displayed as "Equality (1-G)" in plots

SUPPORTED ENTITIES
──────────────────
• Sources/Journals
• Authors
• Countries
• Affiliations/Institutions
• Author Keywords
• Index Keywords
• Subject Areas/Categories
• Document Types
• SDGs (Sustainable Development Goals)
• Publication Years

INTERPRETATION GUIDE
────────────────────
Shannon (normalized):
  0.8-1.0: Very high diversity
  0.6-0.8: High diversity
  0.4-0.6: Moderate diversity
  0.2-0.4: Low diversity
  0.0-0.2: Very low diversity

Gini coefficient:
  0.0-0.3: Low inequality
  0.3-0.5: Moderate inequality
  0.5-0.7: High inequality
  0.7-1.0: Very high inequality

BENCHMARKING
────────────
When enabled, fetches global diversity from OpenAlex
to compare your dataset against worldwide research patterns.

Supported for benchmarking:
• Sources, Countries, Document Types, SDGs

VISUALIZATION
─────────────
• Radar Chart: Shows all indices across entities
• Bar Chart: Grouped bars for each index
• Dashed lines show OpenAlex benchmark (if enabled)
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
    
    def _select_all_entities(self):
        """Select all entity checkboxes."""
        for var in self._entity_vars.values():
            var.set(True)
    
    def _deselect_all_entities(self):
        """Deselect all entity checkboxes."""
        for var in self._entity_vars.values():
            var.set(False)
    
    def _get_selected_entities(self) -> List[str]:
        """Get list of selected entities."""
        return [entity for entity, var in self._entity_vars.items() if var.get()]
    
    def _run_analysis(self):
        """Run the diversity analysis."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        selected_entities = self._get_selected_entities()
        if not selected_entities:
            messagebox.showwarning("No Entities", "Please select at least one entity to analyze.")
            return
        
        self.run_btn.config(state=tk.DISABLED, text="⏳ Computing...")
        
        # Capture values before thread
        fetch_benchmark = self.benchmark_var.get()
        plot_type = self.plot_type.get()
        separator = getattr(self.bib, 'default_separator', '; ')
        
        def do_analysis():
            error_info = None
            try:
                from biblium.diversity import (
                    compute_research_diversity_with_benchmark,
                    list_available_entities,
                )
                
                # Filter to available entities
                available = list_available_entities(self.bib.df)
                entities_to_analyze = [e for e in selected_entities if e in available]
                
                if not entities_to_analyze:
                    raise ValueError(
                        f"None of the selected entities are available in dataset.\n"
                        f"Available: {available}"
                    )
                
                # Get year range for benchmark
                year_range = None
                if "Year" in self.bib.df.columns:
                    year_range = (
                        int(self.bib.df["Year"].min()),
                        int(self.bib.df["Year"].max())
                    )
                
                # Compute diversity
                result = compute_research_diversity_with_benchmark(
                    self.bib.df,
                    entities=entities_to_analyze,
                    separator=separator,
                    fetch_benchmark=fetch_benchmark,
                    year_range=year_range,
                )
                
                self._result = result
                
                # Create plots in main thread
                self.after(0, lambda: self._create_plots(result, plot_type))
                self.after(0, self._on_analysis_success)
                
            except Exception as exc:
                import traceback
                error_info = (str(exc), traceback.format_exc())
                self.after(0, lambda ei=error_info: self._on_analysis_error(ei[0], ei[1]))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _create_plots(self, result, plot_type: str):
        """Create the visualization plots using library functions."""
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from biblium.diversity import plot_diversity_radar, plot_diversity_bars
        import matplotlib.pyplot as plt
        
        # Clean up any existing figure first
        if self._current_fig is not None:
            try:
                plt.close(self._current_fig)
            except:
                pass
            self._current_fig = None
        
        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        entities = list(result.results.keys())
        n_entities = len(entities)
        
        if n_entities == 0:
            tk.Label(
                self.plot_frame,
                text="No diversity results to display",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
            ).pack(expand=True)
            return
        
        # Use library plotting functions
        try:
            if plot_type == "Radar Chart" and n_entities >= 3:
                fig, ax = plot_diversity_radar(
                    result, 
                    title="Research Diversity Profile",
                    show_benchmark=bool(result.benchmark),
                )
            else:
                fig, ax = plot_diversity_bars(
                    result,
                    title="Research Diversity Indices",
                    show_benchmark=bool(result.benchmark),
                )
            
            # Store for export
            self._current_fig = fig
            
            # Embed in tkinter with proper scaling
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            add_plot_context_menu(canvas_widget, fig)
            
            self._canvas = canvas
            
            # Bind resize to update figure size
            self._resize_after_id = None
            def on_resize(event):
                if event.width > 100 and event.height > 100:
                    # Cancel pending resize
                    if self._resize_after_id:
                        canvas_widget.after_cancel(self._resize_after_id)
                    
                    def do_resize():
                        try:
                            dpi = fig.get_dpi()
                            fig.set_size_inches(event.width / dpi, event.height / dpi)
                            fig.tight_layout(pad=1.5, rect=[0.1, 0.1, 0.95, 0.95])
                            canvas.draw_idle()
                        except:
                            pass
                    
                    # Debounce resize
                    self._resize_after_id = canvas_widget.after(100, do_resize)
            
            canvas_widget.bind("<Configure>", on_resize)
            
        except Exception as e:
            tk.Label(
                self.plot_frame,
                text=f"Error creating plot: {e}",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
            ).pack(expand=True)
            import traceback
            traceback.print_exc()
    
    def _on_analysis_success(self):
        """Handle successful analysis completion."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Compute Diversity")
        
        # Update data table
        self._update_data_table()
        
        # Update comparison table
        self._update_comparison_table()
        
        # Update summary
        self._update_summary()
        
        # Switch to plot tab
        self.notebook.select(0)
        
        # Status update
        event_bus.emit("status_update", {
            "message": f"Diversity analysis complete. {len(self._result.results)} entities analyzed.",
            "level": "success"
        })
    
    def _on_analysis_error(self, error_msg: str, traceback_str: str):
        """Handle analysis error."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Compute Diversity")
        
        messagebox.showerror(
            "Analysis Error",
            f"An error occurred:\n\n{error_msg}\n\nSee console for details."
        )
        print(f"Error details:\n{traceback_str}")
        
        event_bus.emit("status_update", {
            "message": f"Analysis failed: {error_msg}",
            "level": "error"
        })
    
    def _update_data_table(self):
        """Update the data table with results."""
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        
        if self._result is None:
            return
        
        df = self._result.to_dataframe()
        
        # Create treeview
        columns = list(df.columns)
        tree = ttk.Treeview(self.data_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            width = max(100, len(col) * 10)
            tree.column(col, width=width, minwidth=80)
        
        # Add data
        for _, row in df.iterrows():
            values = []
            for col in columns:
                val = row[col]
                if isinstance(val, float):
                    values.append(f"{val:.3f}")
                else:
                    values.append(str(val)[:30])
            tree.insert("", tk.END, values=values)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(self.data_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.data_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        self.data_frame.grid_rowconfigure(0, weight=1)
        self.data_frame.grid_columnconfigure(0, weight=1)
    
    def _update_comparison_table(self):
        """Update the comparison table with benchmark data."""
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()
        
        if self._result is None:
            return
        
        if not self._result.benchmark:
            tk.Label(
                self.comparison_frame,
                text="No benchmark data available.\nEnable 'Compare against OpenAlex global' to see comparison.",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
                justify=tk.CENTER,
            ).pack(expand=True)
            return
        
        df = self._result.to_comparison_dataframe()
        
        # Create treeview
        columns = list(df.columns)
        tree = ttk.Treeview(self.comparison_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            width = max(90, len(col) * 9)
            tree.column(col, width=width, minwidth=70)
        
        # Add data
        for _, row in df.iterrows():
            values = []
            for col in columns:
                val = row[col]
                if pd.isna(val):
                    values.append("N/A")
                elif isinstance(val, float):
                    if "Δ" in col:
                        values.append(f"{val:+.3f}")
                    else:
                        values.append(f"{val:.3f}")
                else:
                    values.append(str(val))
            tree.insert("", tk.END, values=values)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(self.comparison_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.comparison_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        self.comparison_frame.grid_rowconfigure(0, weight=1)
        self.comparison_frame.grid_columnconfigure(0, weight=1)
    
    def _update_summary(self):
        """Update the summary tab."""
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        
        if self._result is None:
            return
        
        summary_text = self._result.summary()
        
        # Add interpretation section
        summary_text += "\n\n" + "=" * 50
        summary_text += "\nINTERPRETATION BY ENTITY\n" + "=" * 50
        
        from biblium.diversity import interpret_diversity
        
        for entity, res in self._result.results.items():
            summary_text += "\n\n" + interpret_diversity(res)
        
        # Create text widget
        text_widget = tk.Text(
            self.summary_frame,
            font=("Consolas", 10),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            wrap=tk.WORD,
            padx=16, pady=16,
            relief=tk.FLAT,
        )
        text_widget.insert("1.0", summary_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
    
    def _export_plot(self):
        """Export the current plot."""
        if self._current_fig is None:
            messagebox.showinfo("No Plot", "Please run analysis first.")
            return
        
        filetypes = [
            ("PNG Image", "*.png"),
            ("SVG Vector", "*.svg"),
            ("PDF Document", "*.pdf"),
        ]
        filepath = filedialog.asksaveasfilename(
            title="Save Plot",
            filetypes=filetypes,
            defaultextension=".png",
        )
        if filepath:
            self._current_fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            messagebox.showinfo("Saved", f"Plot saved to:\n{filepath}")
    
    def _export_data(self):
        """Export the results data."""
        if self._result is None:
            messagebox.showinfo("No Data", "Please run analysis first.")
            return
        
        filetypes = [
            ("Excel File", "*.xlsx"),
            ("CSV File", "*.csv"),
        ]
        filepath = filedialog.asksaveasfilename(
            title="Save Data",
            filetypes=filetypes,
            defaultextension=".xlsx",
        )
        if filepath:
            df = self._result.to_dataframe()
            if filepath.endswith('.csv'):
                df.to_csv(filepath, index=False)
            else:
                df.to_excel(filepath, index=False)
            messagebox.showinfo("Saved", f"Data saved to:\n{filepath}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
DIVERSITY INDICES
═════════════════

Measure diversity and concentration of bibliometric entities.

INDICES COMPUTED
────────────────
• Shannon Index (H')
  H' = -Σ pᵢ × ln(pᵢ)
  Higher = more diverse
  Range: 0 to ln(S)
  
• Simpson Index (1-D)
  D = Σ pᵢ²
  1-D = probability two items differ
  Higher = more diverse
  
• Gini Index (G)
  Measures inequality
  0 = perfect equality
  1 = maximum inequality
  
• Richness (S)
  Count of unique categories
  
• Evenness (J)
  J = H' / ln(S)
  How evenly distributed
  Range: 0 to 1

ENTITY TYPES
────────────
• Authors: Productivity concentration
• Sources: Journal distribution
• Keywords: Topic breadth
• Countries: Geographic spread

INTERPRETATION
──────────────
High diversity: Broad, even coverage
Low diversity: Concentrated, dominated
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

    def update_bib(self, bib):
        """Update the BiblioAnalysis instance."""
        self.bib = bib
        # Clean up matplotlib figure before reset
        if self._current_fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._current_fig)
            except:
                pass
        # Reset results when data changes
        self._result = None
        self._current_fig = None
        self._canvas = None
        self._show_placeholder()
