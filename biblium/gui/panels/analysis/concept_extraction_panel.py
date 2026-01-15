# -*- coding: utf-8 -*-
"""
Concept Extraction Panel
========================
GUI panel for concept and keyword extraction analysis.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, List, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
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


class ConceptExtractionPanel(BasePanel):
    """Panel for concept and keyword extraction analysis."""
    
    title = "Concept Extraction"
    icon = "🏷️"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", **kwargs):
        self._stop_requested = False
        self._analysis_running = False
        self._result = None
        self._current_fig = None
        self._canvas = None
        
        super().__init__(parent, theme=theme, **kwargs)
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create options panel."""
        self._add_title()
        
        # About card
        about_card = Card(self.options_content, title="ℹ️ About", theme=self.theme_name)
        about_card.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        about_text = tk.Label(
            about_card.content,
            text=(
                "Extract key terminology from your dataset:\n"
                "• Concepts (OpenAlex hierarchical)\n"
                "• Keywords (OpenAlex/Scopus/WoS)\n"
                "• Topics (OpenAlex primary/secondary)\n"
                "• N-grams from titles/abstracts (TF-IDF)\n\n"
                "Analyze frequency, co-occurrence, and trends."
            ),
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.LEFT,
            anchor="w",
        )
        about_text.pack(fill=tk.X, padx=8, pady=4)
        
        # Data source indicator
        self.data_source_label = tk.Label(
            about_card.content,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
        )
        self.data_source_label.pack(fill=tk.X, padx=8, pady=(0, 4))
        
        # Settings card
        settings_card = Card(self.options_content, title="⚙️ Settings", theme=self.theme_name)
        settings_card.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        # Extract n-grams checkbox
        self.extract_ngrams_var = tk.BooleanVar(value=True)
        ngrams_frame = tk.Frame(settings_card.content, bg=self.theme["bg_card"])
        ngrams_frame.pack(fill=tk.X, padx=8, pady=4)
        
        tk.Checkbutton(
            ngrams_frame,
            text="Extract n-grams (TF-IDF)",
            variable=self.extract_ngrams_var,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_card"],
            activebackground=self.theme["bg_card"],
        ).pack(side=tk.LEFT)
        
        # N-gram range
        ngram_range_frame = tk.Frame(settings_card.content, bg=self.theme["bg_card"])
        ngram_range_frame.pack(fill=tk.X, padx=8, pady=4)
        
        tk.Label(
            ngram_range_frame, text="N-gram range:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.ngram_min_var = tk.StringVar(value="1")
        tk.Entry(
            ngram_range_frame, textvariable=self.ngram_min_var,
            width=3, font=FONTS.get_font("body"),
        ).pack(side=tk.LEFT, padx=(8, 2))
        
        tk.Label(
            ngram_range_frame, text="to",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT, padx=2)
        
        self.ngram_max_var = tk.StringVar(value="3")
        tk.Entry(
            ngram_range_frame, textvariable=self.ngram_max_var,
            width=3, font=FONTS.get_font("body"),
        ).pack(side=tk.LEFT, padx=(2, 0))
        
        # Max n-grams
        max_ngrams_frame = tk.Frame(settings_card.content, bg=self.theme["bg_card"])
        max_ngrams_frame.pack(fill=tk.X, padx=8, pady=4)
        
        tk.Label(
            max_ngrams_frame, text="Max n-grams:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.max_ngrams_var = tk.StringVar(value="500")
        tk.Entry(
            max_ngrams_frame, textvariable=self.max_ngrams_var,
            width=6, font=FONTS.get_font("body"),
        ).pack(side=tk.LEFT, padx=(8, 0))
        
        # Top N to show
        top_n_frame = tk.Frame(settings_card.content, bg=self.theme["bg_card"])
        top_n_frame.pack(fill=tk.X, padx=8, pady=4)
        
        tk.Label(
            top_n_frame, text="Show top N:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.top_n_var = tk.StringVar(value="50")
        tk.Entry(
            top_n_frame, textvariable=self.top_n_var,
            width=6, font=FONTS.get_font("body"),
        ).pack(side=tk.LEFT, padx=(8, 0))
        
        # Visualization card
        viz_card = Card(self.options_content, title="📊 Visualization", theme=self.theme_name)
        viz_card.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        # Plot type
        plot_frame = tk.Frame(viz_card.content, bg=self.theme["bg_card"])
        plot_frame.pack(fill=tk.X, padx=8, pady=4)
        
        tk.Label(
            plot_frame, text="Plot Type:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.plot_type_var = tk.StringVar(value="Top Concepts")
        self.plot_type_combo = ttk.Combobox(
            plot_frame, textvariable=self.plot_type_var,
            values=[
                "Top Concepts",
                "Top Keywords",
                "Top Topics",
                "Top N-grams (TF-IDF)",
                "Co-occurrence Matrix",
                "Temporal Trends",
            ],
            state="readonly",
            width=22,
        )
        self.plot_type_combo.pack(side=tk.LEFT, padx=(8, 0))
        
        # Action buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        
        self.analyze_btn = ActionButton(
            btn_frame,
            text="Extract Concepts",
            icon="▶",
            command=self._run_analysis,
            theme=self.theme_name,
        )
        self.analyze_btn.pack(fill=tk.X)
        
        self.stop_btn = ActionButton(
            btn_frame,
            text="Stop",
            icon="⏹",
            command=self._stop_analysis,
            theme=self.theme_name,
        )
        self.stop_btn.pack(fill=tk.X, pady=(4, 0))
        self.stop_btn.set_enabled(False)
        
        self.update_btn = ActionButton(
            btn_frame,
            text="Update Plot",
            icon="🔄",
            command=self._update_plot,
            theme=self.theme_name,
        )
        self.update_btn.pack(fill=tk.X, pady=(4, 0))
        
        # Export buttons
        export_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        export_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        ActionButton(
            export_frame,
            text="Export Plot",
            icon="🖼️",
            command=self._export_plot,
            theme=self.theme_name,
        ).pack(fill=tk.X)
        
        ActionButton(
            export_frame,
            text="Export Data",
            icon="📊",
            command=self._export_data,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=(4, 0))
    
    def _create_results(self):
        """Create results panel with tabs."""
        self.results_card = tk.Frame(
            self.results_frame,
            bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.results_card)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Visualization tab
        self.viz_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.viz_frame, text="📊 Visualization")
        
        # Concepts tab
        self.concepts_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.concepts_frame, text="🏷️ Concepts")
        
        # Keywords tab
        self.keywords_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.keywords_frame, text="🔑 Keywords")
        
        # N-grams tab
        self.ngrams_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.ngrams_frame, text="📝 N-grams")
        
        # Summary tab
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📈 Summary")
        
        # Info tab
        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Show placeholder message."""
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        placeholder = tk.Label(
            self.viz_frame,
            text="Load a dataset and click 'Extract Concepts'\nto analyze key terminology.\n\n"
                 "Works with OpenAlex, Scopus, and Web of Science datasets.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        )
        placeholder.pack(expand=True)
    
    def _run_analysis(self):
        """Run concept extraction analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if self._analysis_running:
            return
        
        self._analysis_running = True
        self._stop_requested = False
        self.analyze_btn.set_enabled(False)
        self.stop_btn.set_enabled(True)
        
        event_bus.emit(EventBus.ANALYSIS_STARTED, {"name": "Concept Extraction"})
        
        def do_analysis():
            try:
                from biblium.concept_extraction import analyze_concepts
                
                ngram_min = int(self.ngram_min_var.get())
                ngram_max = int(self.ngram_max_var.get())
                max_ngrams = int(self.max_ngrams_var.get())
                top_n = int(self.top_n_var.get())
                
                result = analyze_concepts(
                    self.bib.df,
                    extract_ngrams=self.extract_ngrams_var.get(),
                    ngram_range=(ngram_min, ngram_max),
                    max_ngrams=max_ngrams,
                    top_n=top_n,
                    verbose=True,
                    stop_flag=lambda: self._stop_requested,
                )
                
                if not self._stop_requested:
                    self.after(0, lambda: self._on_analysis_complete(result))
                else:
                    self.after(0, self._on_analysis_stopped)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self._on_analysis_error(str(e)))
        
        thread = threading.Thread(target=do_analysis, daemon=True)
        thread.start()
    
    def _stop_analysis(self):
        """Stop running analysis."""
        self._stop_requested = True
        self.stop_btn.set_enabled(False)
    
    def _on_analysis_complete(self, result):
        """Handle analysis completion."""
        self._analysis_running = False
        self._result = result
        self.analyze_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)
        
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Concept Extraction"})
        
        self._update_plot()
        self._update_concepts_table()
        self._update_keywords_table()
        self._update_ngrams_table()
        self._update_summary()
        
        messagebox.showinfo(
            "Complete",
            f"Concept extraction complete.\n\n"
            f"Unique concepts: {result.n_unique_concepts}\n"
            f"Unique keywords: {result.n_unique_keywords}\n"
            f"N-grams extracted: {len(result.ngrams)}"
        )
    
    def _on_analysis_stopped(self):
        """Handle analysis stopped."""
        self._analysis_running = False
        self.analyze_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)
        event_bus.emit(EventBus.ANALYSIS_CANCELLED, {})
    
    def _on_analysis_error(self, error: str):
        """Handle analysis error."""
        self._analysis_running = False
        self.analyze_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)
        event_bus.emit(EventBus.ERROR_OCCURRED, {"message": error})
        messagebox.showerror("Error", f"Analysis failed:\n{error}")
    
    def _update_plot(self):
        """Update the visualization."""
        if not self._result:
            return
        
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        plot_type = self.plot_type_var.get()
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from biblium.concept_extraction import (
                plot_top_concepts, plot_concept_cooccurrence, plot_temporal_trends
            )
            
            try:
                top_n = int(self.top_n_var.get())
            except:
                top_n = 50
            
            if plot_type == "Top Concepts":
                fig = plot_top_concepts(self._result, n=min(top_n, 30), concept_type="concepts")
            elif plot_type == "Top Keywords":
                fig = plot_top_concepts(self._result, n=min(top_n, 30), concept_type="keywords", color='#2ecc71')
            elif plot_type == "Top Topics":
                fig = plot_top_concepts(self._result, n=min(top_n, 30), concept_type="topics", color='#9b59b6')
            elif plot_type == "Top N-grams (TF-IDF)":
                fig = plot_top_concepts(self._result, n=min(top_n, 30), concept_type="ngrams", color='#e74c3c')
            elif plot_type == "Co-occurrence Matrix":
                fig = plot_concept_cooccurrence(self._result)
            elif plot_type == "Temporal Trends":
                fig = plot_temporal_trends(self._result, top_n=10)
            else:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "Select a plot type", ha='center', va='center')
            
            self._current_fig = fig
            
            if self._canvas:
                self._canvas.get_tk_widget().destroy()
            
            self._canvas = FigureCanvasTkAgg(fig, self.viz_frame)
            self._canvas.draw()
            _self._canvas_widget = self._canvas.get_tk_widget()

            _self._canvas_widget.pack(fill=tk.BOTH, expand=True)
            add_plot_context_menu(canvas_widget, fig)

            add_plot_context_menu(_self._canvas_widget, fig)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_label = tk.Label(
                self.viz_frame,
                text=f"Error creating plot:\n{e}",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["danger"],
            )
            error_label.pack(expand=True)
    
    def _update_concepts_table(self):
        """Update concepts table."""
        for widget in self.concepts_frame.winfo_children():
            widget.destroy()
        
        if not self._result or not self._result.concepts:
            tk.Label(
                self.concepts_frame,
                text="No concepts found.\nOpenAlex datasets contain concept data.",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        self._create_table(
            self.concepts_frame,
            columns=("Rank", "Concept", "Count", "Doc Freq", "Avg Score"),
            data=[
                (i+1, c.name, c.count, c.doc_freq, f"{c.avg_score:.3f}")
                for i, c in enumerate(self._result.concepts)
            ]
        )
    
    def _update_keywords_table(self):
        """Update keywords table."""
        for widget in self.keywords_frame.winfo_children():
            widget.destroy()
        
        if not self._result or not self._result.keywords:
            tk.Label(
                self.keywords_frame,
                text="No keywords found.\nCheck Author Keywords or Index Keywords columns.",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        self._create_table(
            self.keywords_frame,
            columns=("Rank", "Keyword", "Count", "Doc Freq", "Avg Score"),
            data=[
                (i+1, k.name, k.count, k.doc_freq, f"{k.avg_score:.3f}" if k.avg_score else "-")
                for i, k in enumerate(self._result.keywords)
            ]
        )
    
    def _update_ngrams_table(self):
        """Update n-grams table."""
        for widget in self.ngrams_frame.winfo_children():
            widget.destroy()
        
        if not self._result or not self._result.ngrams:
            tk.Label(
                self.ngrams_frame,
                text="No n-grams extracted.\nEnable n-gram extraction in settings.",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        self._create_table(
            self.ngrams_frame,
            columns=("Rank", "N-gram", "TF", "Doc Freq", "Avg TF-IDF"),
            data=[
                (i+1, n.name, n.count, n.doc_freq, f"{n.avg_score:.4f}")
                for i, n in enumerate(self._result.ngrams)
            ]
        )
    
    def _create_table(self, parent, columns, data):
        """Create a table with scrollbars."""
        tree_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        
        for col in columns:
            tree.heading(col, text=col)
            if col == "Rank":
                tree.column(col, width=50, anchor="center")
            elif col in ["Count", "TF", "Doc Freq"]:
                tree.column(col, width=80, anchor="center")
            elif col in ["Avg Score", "Avg TF-IDF"]:
                tree.column(col, width=90, anchor="center")
            else:
                tree.column(col, width=250)
        
        for row in data:
            display_row = list(row)
            # Truncate long text
            if len(str(display_row[1])) > 50:
                display_row[1] = str(display_row[1])[:47] + "..."
            tree.insert("", tk.END, values=display_row)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _update_summary(self):
        """Update summary view."""
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        
        if not self._result:
            return
        
        r = self._result
        summary_text = f"""
Concept Extraction Summary
==========================

Dataset Overview:
  Total documents: {r.n_documents}
  Documents with concepts: {r.n_with_concepts}
  Documents with keywords: {r.n_with_keywords}
  Data source: {r.data_source}

Extraction Results:
  Unique concepts: {r.n_unique_concepts}
  Unique keywords: {r.n_unique_keywords}
  Unique topics: {len(r.topics)}
  N-grams extracted: {len(r.ngrams)}

Top 10 Concepts:
"""
        for i, c in enumerate(r.concepts[:10], 1):
            summary_text += f"  {i}. {c.name} ({c.count})\n"
        
        summary_text += "\nTop 10 Keywords:\n"
        for i, k in enumerate(r.keywords[:10], 1):
            summary_text += f"  {i}. {k.name} ({k.count})\n"
        
        if r.ngrams:
            summary_text += "\nTop 10 N-grams (by TF-IDF):\n"
            for i, n in enumerate(r.ngrams[:10], 1):
                summary_text += f"  {i}. {n.name} (tfidf={n.avg_score:.4f})\n"
        
        text_widget = tk.Text(
            self.summary_frame,
            font=FONTS.get_font("code"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            wrap=tk.WORD,
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        text_widget.insert("1.0", summary_text)
        text_widget.config(state=tk.DISABLED)
    
    def _export_plot(self):
        """Export current plot."""
        if not self._current_fig:
            messagebox.showwarning("No Plot", "No plot to export. Run analysis first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            title="Export Plot"
        )
        
        if filepath:
            self._current_fig.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Exported", f"Plot saved to:\n{filepath}")
    
    def _export_data(self):
        """Export analysis data."""
        if not self._result:
            messagebox.showwarning("No Data", "No data to export. Run analysis first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")],
            title="Export Data"
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.xlsx'):
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    if self._result.concept_df is not None:
                        self._result.concept_df.to_excel(writer, sheet_name='Concepts', index=False)
                    if self._result.keyword_df is not None:
                        self._result.keyword_df.to_excel(writer, sheet_name='Keywords', index=False)
                    if self._result.topic_df is not None:
                        self._result.topic_df.to_excel(writer, sheet_name='Topics', index=False)
                    if self._result.ngram_df is not None and not self._result.ngram_df.empty:
                        self._result.ngram_df.to_excel(writer, sheet_name='N-grams', index=False)
                    if self._result.cooccurrence_matrix is not None:
                        self._result.cooccurrence_matrix.to_excel(writer, sheet_name='Co-occurrence')
                    if self._result.temporal_trends is not None and not self._result.temporal_trends.empty:
                        self._result.temporal_trends.to_excel(writer, sheet_name='Temporal Trends')
            else:
                # Export concepts as CSV
                if self._result.concept_df is not None:
                    self._result.concept_df.to_csv(filepath, index=False)
            
            messagebox.showinfo("Exported", f"Data saved to:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
CONCEPT EXTRACTION
══════════════════

Extract key concepts from text.

METHODS
───────
• TF-IDF extraction
• Noun phrase extraction
• Keyword extraction
• Named entity recognition

OUTPUT
──────
• Ranked concept list
• Frequency counts
• TF-IDF scores
• Co-occurrence patterns

TEXT SOURCES
────────────
• Titles
• Abstracts
• Keywords
• Full text (if available)

APPLICATIONS
────────────
• Vocabulary analysis
• Theme identification
• Ontology building
• Knowledge mapping
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

    def on_data_loaded(self, bib):
        """Handle dataset loaded event."""
        self.bib = bib
        self._result = None
        self._current_fig = None
        
        # Check for concept/keyword columns
        if bib is not None and hasattr(bib, 'df'):
            cols = [c.lower() for c in bib.df.columns]
            
            sources = []
            if 'concepts.display_name' in cols:
                sources.append("OpenAlex concepts")
            if 'keywords.display_name' in cols:
                sources.append("OpenAlex keywords")
            if 'author keywords' in cols:
                sources.append("Author keywords")
            if 'index keywords' in cols:
                sources.append("Index keywords")
            if 'title' in cols or 'abstract' in cols:
                sources.append("Text (for n-grams)")
            
            if sources:
                self.data_source_label.config(
                    text=f"✓ Found: {', '.join(sources)}",
                    fg="green"
                )
            else:
                self.data_source_label.config(
                    text="⚠️ No keyword/concept columns found",
                    fg="orange"
                )
        
        self._show_placeholder()
