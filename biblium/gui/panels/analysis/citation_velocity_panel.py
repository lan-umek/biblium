# -*- coding: utf-8 -*-
"""
Citation Velocity Panel
=======================
Analyze citation velocity and momentum for documents.

@author: Lan.Umek
@version: 2.9.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, List, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledEntry
from biblium.gui.widgets.plots import add_plot_context_menu, make_canvas_resizable

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


class CitationVelocityPanel(BasePanel):
    """
    Panel for citation velocity and momentum analysis.
    """
    
    title = "Citation Velocity"
    icon = "🚀"
    description = "Analyze citation velocity and momentum"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._result = None
        self._current_fig = None
        self._canvas = None
        self._resize_after_id = None
        self._has_embedded_counts = False
        self._is_openalex_db = False
        self._analysis_thread = None
        self._stop_requested = False
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Info Card
        info_card = Card(self.options_content, title="ℹ️ About", theme=self.theme_name)
        info_card.pack(fill=tk.X, padx=8, pady=8)
        
        info_text = (
            "Analyze citation dynamics:\n"
            "• Velocity: Citations per year (rate)\n"
            "• Momentum: Acceleration/deceleration\n"
            "• Trend: Rising, Stable, Declining\n\n"
            "Requires yearly citation counts."
        )
        tk.Label(
            info_card.content,
            text=info_text,
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 8))
        
        # Settings Card
        settings_card = Card(self.options_content, title="⚙️ Settings", theme=self.theme_name)
        settings_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Data source info label
        self.data_source_label = tk.Label(
            settings_card.content,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.LEFT,
        )
        self.data_source_label.pack(fill=tk.X, pady=(0, 4))
        
        # Use OpenAlex checkbox
        self.use_openalex_var = tk.BooleanVar(value=False)
        self.openalex_cb_frame = tk.Frame(settings_card.content, bg=self.theme["bg_card"])
        self.openalex_cb_frame.pack(fill=tk.X, pady=4)
        
        self.openalex_checkbox = tk.Checkbutton(
            self.openalex_cb_frame,
            text="Use OpenAlex API",
            variable=self.use_openalex_var,
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            selectcolor=self.theme["bg_card"],
            font=FONTS.get_font("body"),
        )
        self.openalex_checkbox.pack(anchor=tk.W)
        
        self.max_papers = LabeledEntry(
            settings_card.content, label="Max papers:",
            default="500", width=8,
            theme=self.theme_name, label_width=12,
        )
        self.max_papers.pack(fill=tk.X, pady=4)
        
        self.recent_window = LabeledEntry(
            settings_card.content, label="Recent window:",
            default="3", width=8,
            theme=self.theme_name, label_width=12,
        )
        self.recent_window.pack(fill=tk.X, pady=4)
        
        tk.Label(
            settings_card.content,
            text="(years for velocity calculation)",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
        ).pack(anchor=tk.W)
        
        self.min_age = LabeledEntry(
            settings_card.content, label="Min age (years):",
            default="2", width=8,
            theme=self.theme_name, label_width=12,
        )
        self.min_age.pack(fill=tk.X, pady=4)
        
        # Visualization Card
        viz_card = Card(self.options_content, title="📊 Visualization", theme=self.theme_name)
        viz_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.plot_type = LabeledCombobox(
            viz_card.content, label="Plot Type:",
            values=["Trend Distribution", "Velocity Distribution", "Velocity vs Age", 
                    "Momentum Distribution", "Top Accelerating", "Top Velocity", "Trajectories"],
            default="Trend Distribution",
            theme=self.theme_name, label_width=12,
        )
        self.plot_type.pack(fill=tk.X, pady=4)
        
        self.selection_var = LabeledCombobox(
            viz_card.content, label="Selection:",
            values=["Top Velocity", "Top Accelerating", "Rising Stars"],
            default="Top Velocity",
            theme=self.theme_name, label_width=12,
        )
        self.selection_var.pack(fill=tk.X, pady=4)
        
        self.n_items = LabeledEntry(
            viz_card.content, label="Show top N:",
            default="15", width=8,
            theme=self.theme_name, label_width=12,
        )
        self.n_items.pack(fill=tk.X, pady=4)
        
        # Run button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_primary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        self.run_btn = ActionButton(
            btn_frame, text="Analyze Velocity",
            command=self._run_analysis,
            icon="▶",
            theme=self.theme_name,
        )
        self.run_btn.pack(fill=tk.X)
        
        self.stop_btn = ThemedButton(
            btn_frame, text="Stop",
            command=self._stop_analysis,
            style="secondary",
            icon="⏹",
            theme=self.theme_name,
        )
        self.stop_btn.pack(fill=tk.X, pady=(4, 0))
        self.stop_btn.config(state=tk.DISABLED)
        
        self.update_plot_btn = ThemedButton(
            btn_frame, text="Update Plot",
            command=self._update_plot,
            style="secondary",
            icon="🔄",
            theme=self.theme_name,
        )
        self.update_plot_btn.pack(fill=tk.X, pady=(8, 0))
        
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
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Plot tab
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="📊 Visualization")
        
        # Data tab
        self.data_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.data_frame, text="📋 Data")
        
        # Summary tab
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📝 Summary")

        

        # Info tab

        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])

        self.notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Show placeholder message."""
        if self._current_fig is not None:
            try:
                plt.close(self._current_fig)
            except:
                pass
            self._current_fig = None
        
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        msg = "Click 'Analyze Velocity' to compute\ncitation velocity and momentum metrics."
        if not self._has_embedded_counts and not self._is_openalex_db:
            msg += "\n\n⚠️ Dataset lacks yearly citation data.\nEnable OpenAlex API or use OpenAlex dataset."
        
        tk.Label(
            self.plot_frame,
            text=msg,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _run_analysis(self):
        """Run the velocity analysis."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._stop_requested = False
        self.run_btn.config(state=tk.DISABLED, text="⏳ Analyzing...")
        self.stop_btn.config(state=tk.NORMAL)
        
        # Capture values
        use_openalex = self.use_openalex_var.get()
        
        try:
            max_papers = int(self.max_papers.get())
        except:
            max_papers = 500
        
        try:
            recent_window = int(self.recent_window.get())
        except:
            recent_window = 3
        
        try:
            min_age = int(self.min_age.get())
        except:
            min_age = 2
        
        def do_analysis():
            try:
                from biblium.citation_velocity import analyze_citation_velocity
                
                # Check stop flag before starting
                if self._stop_requested:
                    try:
                        self.after(0, self._on_analysis_stopped)
                    except RuntimeError:
                        pass
                    return
                
                result = analyze_citation_velocity(
                    self.bib.df,
                    use_openalex=use_openalex,
                    max_papers=max_papers,
                    recent_window=recent_window,
                    min_age=min_age,
                    verbose=True,
                    stop_flag=lambda: self._stop_requested,
                )
                
                # Check if stopped during analysis
                if self._stop_requested:
                    try:
                        self.after(0, self._on_analysis_stopped)
                    except RuntimeError:
                        pass
                    return
                
                self._result = result
                
                try:
                    self.after(0, self._on_analysis_success)
                except RuntimeError:
                    # Main thread not in main loop (window closed)
                    pass
                
            except Exception as exc:
                import traceback
                error_msg = str(exc)
                error_trace = traceback.format_exc()
                print(f"Analysis error: {error_msg}\n{error_trace}")
                
                try:
                    self.after(0, lambda: self._on_analysis_error(error_msg, error_trace))
                except RuntimeError:
                    # Main thread not in main loop (window closed)
                    pass
        
        self._analysis_thread = threading.Thread(target=do_analysis, daemon=True)
        self._analysis_thread.start()
    
    def _stop_analysis(self):
        """Request to stop the running analysis."""
        self._stop_requested = True
        self.stop_btn.config(state=tk.DISABLED, text="⏹ Stopping...")
        event_bus.emit("status_update", {
            "message": "Stopping analysis...",
            "level": "info"
        })
    
    def _on_analysis_stopped(self):
        """Handle stopped analysis."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Analyze Velocity")
        self.stop_btn.config(state=tk.DISABLED, text="⏹ Stop")
        self._stop_requested = False
        event_bus.emit("status_update", {
            "message": "Analysis stopped by user",
            "level": "info"
        })
    
    def _on_analysis_success(self):
        """Handle successful analysis."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Analyze Velocity")
        self.stop_btn.config(state=tk.DISABLED, text="⏹ Stop")
        self._update_plot()
        self._update_data_table()
        self._update_summary()
        self.notebook.select(0)
        
        if self._result.n_analyzed == 0:
            event_bus.emit("status_update", {
                "message": "No documents with citation history found. Enable OpenAlex API or use OpenAlex dataset.",
                "level": "warning"
            })
            messagebox.showwarning(
                "No Data", 
                "No documents with yearly citation data found.\n\n"
                "This analysis requires citation counts by year.\n\n"
                "Options:\n"
                "• Use an OpenAlex dataset with counts_by_year column\n"
                "• Enable 'Use OpenAlex API' checkbox to fetch data"
            )
        else:
            event_bus.emit("status_update", {
                "message": f"Citation velocity analysis complete ({self._result.n_analyzed} documents)",
                "level": "success"
            })
    
    def _on_analysis_error(self, error_msg: str, traceback_str: str):
        """Handle analysis error."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Analyze Velocity")
        self.stop_btn.config(state=tk.DISABLED, text="⏹ Stop")
        self._stop_requested = False
        messagebox.showerror("Analysis Error", f"Error:\n{error_msg}")
        print(f"Error details:\n{traceback_str}")
    
    def _update_plot(self):
        """Update the visualization."""
        if self._result is None:
            return
        
        # Clean up previous
        if self._current_fig is not None:
            try:
                plt.close(self._current_fig)
            except:
                pass
        
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Check if we have any data
        if not self._result.metrics:
            tk.Label(
                self.plot_frame,
                text="No documents with citation history found.\n\n"
                     "This analysis requires yearly citation counts.\n"
                     "Options:\n"
                     "• Use an OpenAlex dataset with counts_by_year\n"
                     "• Enable 'Use OpenAlex API' to fetch citation history",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
                justify=tk.CENTER,
            ).pack(expand=True)
            return
        
        plot_type_map = {
            "Trend Distribution": "trend",
            "Velocity Distribution": "velocity",
            "Velocity vs Age": "vs_age",
            "Momentum Distribution": "momentum",
            "Top Accelerating": "top_accelerating",
            "Top Velocity": "top_velocity",
            "Trajectories": "trajectories",
        }
        
        selection_map = {
            "Top Velocity": "top_velocity",
            "Top Accelerating": "top_accelerating",
            "Rising Stars": "rising_stars",
        }
        
        plot_type = plot_type_map.get(self.plot_type.get(), "trend")
        selection = selection_map.get(self.selection_var.get(), "top_velocity")
        
        try:
            n = int(self.n_items.get())
        except:
            n = 15
        
        try:
            from biblium.citation_velocity import (
                plot_trend_distribution,
                plot_velocity_distribution,
                plot_velocity_vs_age,
                plot_momentum_distribution,
                plot_top_accelerating,
                plot_top_velocity,
                plot_velocity_trajectories,
            )
            
            if plot_type == "trend":
                fig, ax = plot_trend_distribution(self._result)
            elif plot_type == "velocity":
                fig, ax = plot_velocity_distribution(self._result)
            elif plot_type == "vs_age":
                fig, ax = plot_velocity_vs_age(self._result)
            elif plot_type == "momentum":
                fig, ax = plot_momentum_distribution(self._result)
            elif plot_type == "top_accelerating":
                fig, ax = plot_top_accelerating(self._result, n=n)
            elif plot_type == "top_velocity":
                fig, ax = plot_top_velocity(self._result, n=n)
            elif plot_type == "trajectories":
                fig, ax = plot_velocity_trajectories(self._result, n_examples=min(n, 5), selection=selection)
            else:
                fig, ax = plot_trend_distribution(self._result)
            
            self._current_fig = fig
            
            # Create canvas with scaling
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            add_plot_context_menu(canvas_widget, fig)
            self._canvas = canvas
            
            # Bind resize
            def on_resize(event):
                if event.width > 100 and event.height > 100:
                    if self._resize_after_id:
                        canvas_widget.after_cancel(self._resize_after_id)
                    
                    def do_resize():
                        try:
                            dpi = fig.get_dpi()
                            fig.set_size_inches(event.width / dpi, event.height / dpi)
                            fig.tight_layout(pad=1.5)
                            canvas.draw_idle()
                        except:
                            pass
                    
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
    
    def _update_data_table(self):
        """Update the data table."""
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        
        if self._result is None:
            return
        
        df = self._result.to_dataframe()
        
        if df.empty:
            tk.Label(
                self.data_frame,
                text="No data available",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
            ).pack(expand=True)
            return
        
        columns = list(df.columns)
        tree = ttk.Treeview(self.data_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            tree.heading(col, text=col)
            width = 80 if col not in ["Title", "DOI"] else 180
            tree.column(col, width=width, minwidth=50)
        
        for _, row in df.head(500).iterrows():
            values = []
            for v in row:
                if isinstance(v, float):
                    values.append(f"{v:.2f}")
                else:
                    val_str = str(v)
                    values.append(val_str[:40] + "..." if len(val_str) > 40 else val_str)
            tree.insert("", tk.END, values=values)
        
        vsb = ttk.Scrollbar(self.data_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.data_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        self.data_frame.grid_rowconfigure(0, weight=1)
        self.data_frame.grid_columnconfigure(0, weight=1)
    
    def _update_summary(self):
        """Update the summary tab."""
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        
        if self._result is None:
            return
        
        summary_text = self._result.summary()
        
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
        
        filepath = filedialog.asksaveasfilename(
            title="Save Plot",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")],
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
        
        filepath = filedialog.asksaveasfilename(
            title="Save Data",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")],
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
CITATION VELOCITY
═════════════════

Measure citation accumulation speed over time.

VELOCITY METRICS
────────────────
• Early Velocity: Citations in first 2-3 years
• Current Velocity: Recent annual citations
• Peak Velocity: Maximum annual citations
• Acceleration: Velocity change rate

VELOCITY PATTERNS
─────────────────
• Accelerating: Gaining momentum
• Constant: Steady citation rate
• Decelerating: Slowing down
• Dormant: Very low recent citations

APPLICATIONS
────────────
• Identify "hot" papers gaining traction
• Predict future high-impact work
• Detect emerging research fronts
• Compare papers of similar age

OUTPUT
──────
• Velocity scores per paper
• Velocity trends over time
• Pattern classification
• Ranking by velocity
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
        if self._current_fig is not None:
            try:
                plt.close(self._current_fig)
            except:
                pass
        self._result = None
        self._current_fig = None
        self._canvas = None
        
        # Check for embedded counts_by_year column
        self._has_embedded_counts = False
        self._has_split_counts = False
        self._is_openalex_db = False
        
        if bib is not None and hasattr(bib, 'df'):
            # Check for counts_by_year column (single embedded column)
            for col in ["counts_by_year", "Counts by Year", "citation_counts_by_year"]:
                if col in bib.df.columns:
                    sample = bib.df[col].dropna().head(5)
                    if len(sample) > 0:
                        self._has_embedded_counts = True
                        break
            
            # Check for OpenAlex split format columns
            # Note: BiblioPlot may rename counts_by_year.cited_by_count to "Citations by Year"
            has_year_col = any(c.lower() == "counts_by_year.year" for c in bib.df.columns)
            has_count_col = any(c.lower() in ["counts_by_year.cited_by_count", "citations by year"] for c in bib.df.columns)
            if has_year_col and has_count_col:
                self._has_split_counts = True
            
            # Check if database is OpenAlex
            openalex_indicators = ["openalex_id", "id", "cited_by_count", "counts_by_year", 
                                   "authorships", "primary_location", "open_access",
                                   "counts_by_year.year", "counts_by_year.cited_by_count"]
            openalex_cols = sum(1 for col in openalex_indicators if col in bib.df.columns)
            
            if openalex_cols >= 3:
                self._is_openalex_db = True
            
            if hasattr(bib, 'db') and bib.db:
                if 'openalex' in str(bib.db).lower():
                    self._is_openalex_db = True
        
        # Update UI based on data source
        if self._has_embedded_counts or self._has_split_counts:
            self.data_source_label.config(
                text="✓ Dataset contains citation history\n   (No API calls needed)",
                fg="green"
            )
            self.use_openalex_var.set(False)
            self.openalex_checkbox.config(state=tk.DISABLED)
            self.max_papers.entry.config(state=tk.DISABLED)
        elif self._is_openalex_db:
            self.data_source_label.config(
                text="✓ OpenAlex database detected\n   (API enabled by default)",
                fg="green"
            )
            self.use_openalex_var.set(True)
            self.openalex_checkbox.config(state=tk.NORMAL)
            self.max_papers.entry.config(state=tk.NORMAL)
        else:
            self.data_source_label.config(
                text="⚠️ No yearly citation data\n   (Enable API or use OpenAlex data)",
                fg="orange"
            )
            self.use_openalex_var.set(False)
            self.openalex_checkbox.config(state=tk.NORMAL)
            self.max_papers.entry.config(state=tk.NORMAL)
        
        self._show_placeholder()
