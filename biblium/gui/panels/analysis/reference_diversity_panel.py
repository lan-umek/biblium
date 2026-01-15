# -*- coding: utf-8 -*-
"""
Reference Diversity Panel
=========================
GUI panel for analyzing reference diversity metrics.
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


class ReferenceDiversityPanel(BasePanel):
    """Panel for reference diversity analysis."""
    
    title = "Reference Diversity"
    icon = "📚"
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
                "Analyze diversity of paper references:\n"
                "• Source Diversity: Different journals/sources\n"
                "• Field Diversity: Cross-disciplinary citations\n"
                "• Age Diversity: Temporal spread of refs\n"
                "• Interdisciplinarity: Rao-Stirling index\n\n"
                "Requires referenced_works column.\n"
                "OpenAlex API enriches the analysis."
            ),
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.LEFT,
            anchor="w",
        )
        about_text.pack(fill=tk.X, padx=8, pady=4)
        
        # Settings card
        settings_card = Card(self.options_content, title="⚙️ Settings", theme=self.theme_name)
        settings_card.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        # Use OpenAlex API checkbox
        self.use_openalex_var = tk.BooleanVar(value=True)
        openalex_frame = tk.Frame(settings_card.content, bg=self.theme["bg_card"])
        openalex_frame.pack(fill=tk.X, padx=8, pady=4)
        
        self.openalex_checkbox = tk.Checkbutton(
            openalex_frame,
            text="Use OpenAlex API",
            variable=self.use_openalex_var,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_card"],
            activebackground=self.theme["bg_card"],
        )
        self.openalex_checkbox.pack(side=tk.LEFT)
        
        # Data source indicator
        self.data_source_label = tk.Label(
            settings_card.content,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
        )
        self.data_source_label.pack(fill=tk.X, padx=8, pady=(0, 4))
        
        # Max papers
        max_papers_frame = tk.Frame(settings_card.content, bg=self.theme["bg_card"])
        max_papers_frame.pack(fill=tk.X, padx=8, pady=4)
        
        tk.Label(
            max_papers_frame, text="Max papers:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.max_papers_var = tk.StringVar(value="100")
        self.max_papers_entry = tk.Entry(
            max_papers_frame, textvariable=self.max_papers_var,
            width=8, font=FONTS.get_font("body"),
        )
        self.max_papers_entry.pack(side=tk.LEFT, padx=(8, 0))
        
        # Max refs per paper
        max_refs_frame = tk.Frame(settings_card.content, bg=self.theme["bg_card"])
        max_refs_frame.pack(fill=tk.X, padx=8, pady=4)
        
        tk.Label(
            max_refs_frame, text="Max refs/paper:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.max_refs_var = tk.StringVar(value="50")
        self.max_refs_entry = tk.Entry(
            max_refs_frame, textvariable=self.max_refs_var,
            width=8, font=FONTS.get_font("body"),
        )
        self.max_refs_entry.pack(side=tk.LEFT, padx=(8, 0))
        
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
        
        self.plot_type_var = tk.StringVar(value="Diversity Distribution")
        self.plot_type_combo = ttk.Combobox(
            plot_frame, textvariable=self.plot_type_var,
            values=[
                "Diversity Distribution",
                "Source Diversity",
                "Field Diversity",
                "Reference Age Distribution",
                "Diversity by Year",
                "Top Diverse Papers",
            ],
            state="readonly",
            width=20,
        )
        self.plot_type_combo.pack(side=tk.LEFT, padx=(8, 0))
        
        # Show top N
        top_n_frame = tk.Frame(viz_card.content, bg=self.theme["bg_card"])
        top_n_frame.pack(fill=tk.X, padx=8, pady=4)
        
        tk.Label(
            top_n_frame, text="Show top N:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.top_n_var = tk.StringVar(value="15")
        self.top_n_entry = tk.Entry(
            top_n_frame, textvariable=self.top_n_var,
            width=8, font=FONTS.get_font("body"),
        )
        self.top_n_entry.pack(side=tk.LEFT, padx=(8, 0))
        
        # Action buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        
        self.analyze_btn = ActionButton(
            btn_frame,
            text="Analyze Diversity",
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
        # Results card
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
        
        # Data tab
        self.data_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.data_frame, text="📋 Data")
        
        # Summary tab
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📈 Summary")
        
        # Info tab
        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        
        # Show placeholder
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Show placeholder message."""
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        placeholder = tk.Label(
            self.viz_frame,
            text="Load a dataset with referenced_works column\n"
                 "and click 'Analyze Diversity' to see results.\n\n"
                 "OpenAlex datasets include this data.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        )
        placeholder.pack(expand=True)
    
    def _run_analysis(self):
        """Run reference diversity analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if self._analysis_running:
            return
        
        self._analysis_running = True
        self._stop_requested = False
        self.analyze_btn.set_enabled(False)
        self.stop_btn.set_enabled(True)
        
        event_bus.emit(EventBus.ANALYSIS_STARTED, {"name": "Reference Diversity"})
        
        def do_analysis():
            try:
                from biblium.reference_diversity import analyze_reference_diversity
                
                max_papers = int(self.max_papers_var.get())
                max_refs = int(self.max_refs_var.get())
                
                result = analyze_reference_diversity(
                    self.bib.df,
                    use_openalex=self.use_openalex_var.get(),
                    max_papers=max_papers,
                    max_refs_per_paper=max_refs,
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
        
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Reference Diversity"})
        
        if result.n_analyzed == 0:
            messagebox.showwarning(
                "No Data",
                "No documents with reference data found.\n\n"
                "This analysis requires a 'referenced_works' column.\n"
                "Use an OpenAlex dataset or enable OpenAlex API."
            )
            return
        
        # Add computed metrics to original dataframe
        self._add_metrics_to_dataframe(result)
        
        self._update_plot()
        self._update_data_view()
        self._update_summary()
        
        messagebox.showinfo(
            "Complete",
            f"Analyzed {result.n_analyzed} documents.\n\n"
            f"Average references: {result.avg_reference_count:.1f}\n"
            f"Average source diversity: {result.avg_source_diversity:.3f}\n"
            f"Average field diversity: {result.avg_field_diversity:.3f}\n\n"
            f"New columns added to dataset:\n"
            f"• Ref_Source_Diversity\n"
            f"• Ref_Field_Diversity\n"
            f"• Ref_Diversity_Level\n"
            f"• Ref_Median_Age"
        )
    
    def _add_metrics_to_dataframe(self, result):
        """Add computed reference diversity metrics to the original dataframe."""
        if not self.bib or not hasattr(self.bib, 'df'):
            return
        
        # Create a mapping from DOI/title to metrics
        metrics_data = {}
        for m in result.metrics:
            # Use DOI as primary key, fall back to title
            key = m.doi if m.doi else m.title
            if key:
                metrics_data[key] = {
                    'Ref_Source_Diversity': round(m.source_diversity, 4),
                    'Ref_Field_Diversity': round(m.field_diversity, 4),
                    'Ref_Topic_Diversity': round(m.topic_diversity, 4),
                    'Ref_Rao_Stirling': round(m.rao_stirling_index, 4),
                    'Ref_Unique_Sources': m.unique_sources,
                    'Ref_Unique_Fields': m.unique_fields,
                    'Ref_Median_Age': round(m.median_ref_age, 1),
                    'Ref_Mean_Age': round(m.mean_ref_age, 1),
                    'Ref_Diversity_Level': m.diversity_level,
                }
        
        # Initialize new columns with NaN
        new_cols = [
            'Ref_Source_Diversity', 'Ref_Field_Diversity', 'Ref_Topic_Diversity',
            'Ref_Rao_Stirling', 'Ref_Unique_Sources', 'Ref_Unique_Fields',
            'Ref_Median_Age', 'Ref_Mean_Age', 'Ref_Diversity_Level'
        ]
        
        for col in new_cols:
            if col not in self.bib.df.columns:
                self.bib.df[col] = None
        
        # Find DOI and title columns
        doi_col = None
        title_col = None
        for c in self.bib.df.columns:
            c_lower = c.lower()
            if c_lower == 'doi' and doi_col is None:
                doi_col = c
            elif c_lower in ['title', 'document title', 'ti'] and title_col is None:
                title_col = c
        
        # Map metrics to rows
        matched = 0
        for idx, row in self.bib.df.iterrows():
            key = None
            if doi_col and pd.notna(row.get(doi_col)):
                key = str(row[doi_col])
            if key not in metrics_data and title_col and pd.notna(row.get(title_col)):
                key = str(row[title_col])
            
            if key and key in metrics_data:
                for col, val in metrics_data[key].items():
                    self.bib.df.at[idx, col] = val
                matched += 1
        
        print(f"  ✓ Added diversity metrics to {matched} rows in dataframe")
    
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
        if not self._result or not self._result.metrics:
            return
        
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        plot_type = self.plot_type_var.get()
        
        # Handle "Top Diverse Papers" as table instead of plot
        if plot_type == "Top Diverse Papers":
            self._show_top_diverse_table()
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from biblium.reference_diversity import (
                plot_diversity_distribution,
                plot_source_diversity,
                plot_field_diversity,
                plot_reference_age_distribution,
                plot_diversity_by_year,
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(self.theme["bg_card"])
            ax.set_facecolor(self.theme["bg_card"])
            
            if plot_type == "Diversity Distribution":
                plot_diversity_distribution(self._result, ax=ax)
            elif plot_type == "Source Diversity":
                plot_source_diversity(self._result, ax=ax)
            elif plot_type == "Field Diversity":
                plot_field_diversity(self._result, ax=ax)
            elif plot_type == "Reference Age Distribution":
                plot_reference_age_distribution(self._result, ax=ax)
            elif plot_type == "Diversity by Year":
                plot_diversity_by_year(self._result, ax=ax)
            
            # Remove gridlines
            ax.grid(False)
            
            fig.tight_layout()
            self._current_fig = fig
            
            if self._canvas:
                self._canvas.get_tk_widget().destroy()
            
            self._canvas = FigureCanvasTkAgg(fig, self.viz_frame)
            self._canvas.draw()
            self._canvas_widget = self._canvas.get_tk_widget()

            self._canvas_widget.pack(fill=tk.BOTH, expand=True)
            add_plot_context_menu(self._canvas_widget, fig)
            
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
    
    def _show_top_diverse_table(self):
        """Show top diverse papers as a table."""
        try:
            n = int(self.top_n_var.get())
        except:
            n = 15
        
        from biblium.reference_diversity import get_top_diverse_papers
        
        df = get_top_diverse_papers(self._result, n=n)
        
        # Create scrollable treeview
        tree_frame = tk.Frame(self.viz_frame, bg=self.theme["bg_card"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Title
        title_label = tk.Label(
            tree_frame,
            text=f"Top {n} Most Diverse Papers",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        )
        title_label.pack(pady=(8, 4))
        
        # Table frame
        table_frame = tk.Frame(tree_frame, bg=self.theme["bg_card"])
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("Rank", "Title", "Year", "Refs", "Sources", "Fields", "Src Div", "Fld Div", "Combined", "Level")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        
        # Column config
        tree.heading("Rank", text="#")
        tree.column("Rank", width=40, anchor="center")
        
        tree.heading("Title", text="Title")
        tree.column("Title", width=300)
        
        tree.heading("Year", text="Year")
        tree.column("Year", width=60, anchor="center")
        
        tree.heading("Refs", text="Refs")
        tree.column("Refs", width=50, anchor="center")
        
        tree.heading("Sources", text="Sources")
        tree.column("Sources", width=60, anchor="center")
        
        tree.heading("Fields", text="Fields")
        tree.column("Fields", width=50, anchor="center")
        
        tree.heading("Src Div", text="Src Div")
        tree.column("Src Div", width=70, anchor="center")
        
        tree.heading("Fld Div", text="Fld Div")
        tree.column("Fld Div", width=70, anchor="center")
        
        tree.heading("Combined", text="Combined")
        tree.column("Combined", width=80, anchor="center")
        
        tree.heading("Level", text="Level")
        tree.column("Level", width=80, anchor="center")
        
        # Insert data
        for _, row in df.iterrows():
            title = row['Title']
            if len(title) > 50:
                title = title[:47] + '...'
            tree.insert("", tk.END, values=(
                row['Rank'],
                title,
                row['Year'],
                row['References'],
                row['Unique Sources'],
                row['Unique Fields'],
                row['Source Diversity'],
                row['Field Diversity'],
                row['Combined'],
                row['Level'],
            ))
        
        # Scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        
        # No figure for this view
        self._current_fig = None
    
    def _update_data_view(self):
        """Update data table view."""
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        
        if not self._result or not self._result.metrics:
            return
        
        # Create scrollable treeview
        tree_frame = tk.Frame(self.data_frame, bg=self.theme["bg_card"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        columns = ("Title", "Refs", "Sources", "Fields", "Src Div", "Fld Div", "Med Age", "Level")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80)
        
        tree.column("Title", width=200)
        tree.column("Level", width=100)
        
        for m in self._result.metrics:
            tree.insert("", tk.END, values=(
                m.title[:40] + '...' if len(m.title) > 40 else m.title,
                m.reference_count,
                m.unique_sources,
                m.unique_fields,
                f"{m.source_diversity:.3f}",
                f"{m.field_diversity:.3f}",
                f"{m.median_ref_age:.1f}",
                m.diversity_level,
            ))
        
        # Scrollbars
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
        
        summary_text = f"""
Reference Diversity Analysis Summary
====================================

Dataset Overview:
  Total papers: {self._result.n_papers}
  Papers analyzed: {self._result.n_analyzed}
  Papers with references: {self._result.n_with_references}
  Data source: {self._result.data_source}
  API calls made: {self._result.api_calls_made}

Aggregate Metrics:
  Average reference count: {self._result.avg_reference_count:.1f}
  Average source diversity: {self._result.avg_source_diversity:.3f}
  Average field diversity: {self._result.avg_field_diversity:.3f}
  Average reference age: {self._result.avg_ref_age:.1f} years

Diversity Distribution:
"""
        for level, count in self._result.diversity_distribution.items():
            pct = (count / self._result.n_analyzed * 100) if self._result.n_analyzed > 0 else 0
            summary_text += f"  {level}: {count} ({pct:.1f}%)\n"
        
        summary_label = tk.Text(
            self.summary_frame,
            font=FONTS.get_font("code"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            wrap=tk.WORD,
            height=30,
        )
        summary_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        summary_label.insert("1.0", summary_text)
        summary_label.config(state=tk.DISABLED)
    
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
        if not self._result or not self._result.metrics:
            messagebox.showwarning("No Data", "No data to export. Run analysis first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")],
            title="Export Data"
        )
        
        if filepath:
            data = []
            for m in self._result.metrics:
                data.append({
                    'DOI': m.doi,
                    'Title': m.title,
                    'Year': m.pub_year,
                    'Reference Count': m.reference_count,
                    'Unique Sources': m.unique_sources,
                    'Unique Fields': m.unique_fields,
                    'Unique Topics': m.unique_topics,
                    'Source Diversity': m.source_diversity,
                    'Field Diversity': m.field_diversity,
                    'Topic Diversity': m.topic_diversity,
                    'Rao-Stirling Index': m.rao_stirling_index,
                    'Median Ref Age': m.median_ref_age,
                    'Mean Ref Age': m.mean_ref_age,
                    'Diversity Level': m.diversity_level,
                })
            
            import pandas as pd
            df = pd.DataFrame(data)
            
            if filepath.endswith('.xlsx'):
                df.to_excel(filepath, index=False)
            else:
                df.to_csv(filepath, index=False)
            
            messagebox.showinfo("Exported", f"Data saved to:\n{filepath}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
REFERENCE DIVERSITY
═══════════════════

Analyze diversity of cited references.

DIVERSITY DIMENSIONS
────────────────────
• Source diversity
  How many journals cited
  
• Field diversity
  Disciplines represented
  
• Temporal diversity
  Age spread of references
  
• Geographic diversity
  Countries of cited works

METRICS
───────
• Shannon index of refs
• Richness (unique sources)
• Age range of references
• Interdisciplinarity score

INTERPRETATION
──────────────
High diversity:
• Broad literature base
• Interdisciplinary work
• Comprehensive coverage

Low diversity:
• Focused citations
• Single-field work
• Potential gaps
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
        
        # Check for referenced_works column
        has_refs = False
        if bib is not None and hasattr(bib, 'df'):
            for col in ["referenced_works", "References", "Cited References"]:
                if col in bib.df.columns:
                    has_refs = True
                    n_with_refs = bib.df[col].notna().sum()
                    self.data_source_label.config(
                        text=f"✓ Found {col} column\n   ({n_with_refs} papers with refs)",
                        fg="green"
                    )
                    break
        
        if not has_refs:
            self.data_source_label.config(
                text="⚠️ No referenced_works column\n   Use OpenAlex dataset or API",
                fg="orange"
            )
        
        self._show_placeholder()
