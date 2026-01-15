# -*- coding: utf-8 -*-
"""
Top Items Timeline Panel
========================
Bubble plot visualization of top items production over time.
Uses biblium's plot_item_time_stats() for scatter/bubble visualization.
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
from biblium.gui.widgets.tables import DataTable

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

# Available items configuration
AVAILABLE_ITEMS_CONFIG = [
    ("Authors", "authors", ["Authors", "Author full names", "AU", "AF", "Author Full Names"]),
    ("Author Keywords", "author keywords", ["Author Keywords", "DE", "Keywords"]),
    ("Index Keywords", "index keywords", ["Index Keywords", "ID", "Keywords Plus", "Keyword Plus"]),
    ("Sources", "sources", ["Source title", "Source", "SO", "Journal", "Publication Name"]),
    ("Countries", "all countries", ["Countries", "Country", "Countries of Authors"]),
    ("Affiliations", "affiliations", ["Affiliations", "C1", "Addresses", "Author Affiliations"]),
    ("References", "references", ["References", "Cited References", "CR", "Referenced Works"]),
]


def get_available_items_for_df(df):
    """Get list of available items based on dataframe columns."""
    if df is None:
        return [item[0] for item in AVAILABLE_ITEMS_CONFIG]
    
    available = []
    for display_name, key, possible_cols in AVAILABLE_ITEMS_CONFIG:
        for col in possible_cols:
            if col in df.columns:
                available.append(display_name)
                break
    
    return available if available else [item[0] for item in AVAILABLE_ITEMS_CONFIG]


class TopItemsTimelinePanel(BasePanel):
    """Panel for bubble plot visualization of top items over time."""
    
    title = "Top Items Timeline"
    icon = "⏱️"
    description = "Bubble plot of top items production over time"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._result_df = None
        self._current_fig = None
        self._canvas = None
        self._toolbar = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Item Selection Card
        item_card = Card(self.options_content, title="📋 Item Selection", theme=self.theme_name)
        item_card.pack(fill=tk.X, padx=8, pady=8)
        
        available_items = self._get_available_items()
        self.item_type = LabeledCombobox(
            item_card.content, label="Item Type:",
            values=available_items,
            default=available_items[0] if available_items else "Authors",
            theme=self.theme_name, label_width=12,
        )
        self.item_type.pack(fill=tk.X, pady=4)
        
        # Settings Card
        settings_card = Card(self.options_content, title="⚙️ Settings", theme=self.theme_name)
        settings_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.top_n_spin = LabeledSpinbox(
            settings_card.content, label="Top N:",
            from_=5, to=50, default=10,
            theme=self.theme_name, label_width=12,
        )
        self.top_n_spin.pack(fill=tk.X, pady=4)
        
        self.min_docs_spin = LabeledSpinbox(
            settings_card.content, label="Min Documents:",
            from_=1, to=20, default=1,
            theme=self.theme_name, label_width=12,
        )
        self.min_docs_spin.pack(fill=tk.X, pady=4)
        
        # Plot Options Card
        plot_card = Card(self.options_content, title="🎨 Plot Options", theme=self.theme_name)
        plot_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.color_by = LabeledCombobox(
            plot_card.content, label="Color By:",
            values=["Citations per document", "Total citations", "H-index"],
            default="Citations per document",
            theme=self.theme_name, label_width=12,
        )
        self.color_by.pack(fill=tk.X, pady=4)
        
        self.cmap = LabeledCombobox(
            plot_card.content, label="Color Map:",
            values=["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Reds", "YlOrRd"],
            default="viridis",
            theme=self.theme_name, label_width=12,
        )
        self.cmap.pack(fill=tk.X, pady=4)
        
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
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="📈 Plot")
        
        self.data_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.data_frame, text="📋 Data")
        
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
            text="Click 'Generate Plot' to see\ntop items timeline",
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
TOP ITEMS TIMELINE (BUBBLE PLOT)
════════════════════════════════

This visualization shows top items as horizontal timelines with
bubble sizes representing document counts and colors representing
citation metrics.

VISUALIZATION
─────────────
• Y-axis: Top items (authors, keywords, etc.)
• X-axis: Publication years
• Bubble size: Number of documents in that year
• Bubble color: Citation metric (configurable)
• Horizontal lines: Activity span (first to last year)

INTERPRETATION
──────────────
• Long horizontal lines = sustained activity
• Large bubbles = high productivity periods
• Warm colors = high citation impact
• Gaps in lines = publication pauses

SETTINGS
────────
• Top N: Number of items to display
• Min Documents: Filter items with few publications
• Color By: Metric for bubble coloring
• Color Map: Color scheme for the metric

This uses biblium's plot_item_time_stats() function.
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
    
    def _get_available_items(self):
        """Get list of available items based on current data."""
        df = self.bib.df if self.bib is not None else None
        return get_available_items_for_df(df)
    
    def _get_item_key(self, display_name):
        """Get the key for a display name."""
        for dn, key, _ in AVAILABLE_ITEMS_CONFIG:
            if dn == display_name:
                return key
        return display_name.lower().replace(" ", "_")
    
    def _run_analysis(self):
        """Run the analysis."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self.run_btn.config(state=tk.DISABLED, text="⏳ Analyzing...")
        
        item_key = self._get_item_key(self.item_type.get())
        top_n = self.top_n_spin.get()
        min_docs = self.min_docs_spin.get()
        color_by = self.color_by.get()
        cmap = self.cmap.get()
        
        def do_analysis():
            error_info = None
            try:
                from biblium import utilsbib, plotbib
                
                col_mapping = {key: cols for _, key, cols in AVAILABLE_ITEMS_CONFIG}
                
                color_col_map = {
                    "Citations per document": "citations_per_article",
                    "Total citations": "total_citations",
                    "H-index": "item_h_index",
                }
                color_col = color_col_map.get(color_by, "citations_per_article")
                
                # Check if bib has mapping
                has_mapping = False
                if (hasattr(self.bib, 'mapping') 
                    and item_key in self.bib.mapping
                    and isinstance(self.bib.mapping.get(item_key), dict)
                    and "time production var" in self.bib.mapping.get(item_key, {})):
                    mapped_col = self.bib.mapping[item_key]["time production var"]
                    if mapped_col in self.bib.df.columns or f"Processed {mapped_col}" in self.bib.df.columns:
                        has_mapping = True
                
                if has_mapping:
                    fig = self.bib.plot_items_production_over_time(
                        items=item_key,
                        top_n=top_n,
                        compute_kwargs={"min_docs_per_item": min_docs},
                        plot_kwargs={
                            "color_col": color_col,
                            "cmap": cmap,
                        },
                    )
                    
                    import re
                    slug = re.sub(r"\W+", "_", str(item_key).strip().lower()).strip("_")
                    attr_name = f"{slug}_production_over_time_df"
                    result_df = getattr(self.bib, attr_name, None)
                else:
                    # Fallback
                    possible_cols = col_mapping.get(item_key, [item_key])
                    actual_col = None
                    for col in possible_cols:
                        if col in self.bib.df.columns:
                            actual_col = col
                            break
                    
                    if actual_col is None:
                        for col in self.bib.df.columns:
                            col_lower = col.lower()
                            item_lower = item_key.replace("_", " ").lower()
                            if col_lower == item_lower:
                                actual_col = col
                                break
                    
                    if actual_col is None:
                        raise ValueError(f"Could not find column for '{item_key}'")
                    
                    separator = getattr(self.bib, 'default_separator', '; ')
                    if hasattr(self.bib, 'db') and self.bib.db == 'oa':
                        separator = r"\|"
                    
                    result_df = utilsbib.compute_item_time_stats(
                        self.bib.df, 
                        actual_col,
                        list_separators=separator,
                        top_n=top_n,
                        min_docs_per_item=min_docs,
                    )
                    
                    fig = plotbib.plot_item_time_stats(
                        result_df,
                        color_col=color_col,
                        cmap=cmap,
                        y_label=actual_col,
                        tick_fontsize=8,
                        label_fontsize=10,
                    )
                
                self._result_df = result_df
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
            from biblium.gui.widgets.plots import ScaledImageFrame
            
            scaled_frame = ScaledImageFrame(
                self.plot_frame, 
                theme=self.theme_name,
                maintain_aspect=True,
                max_scale=1.5
            )
            scaled_frame.pack(fill=tk.BOTH, expand=True)
            scaled_frame.set_image_from_figure(self._current_fig, dpi=100)
            
            self._scaled_frame = scaled_frame
            self._add_save_menu(scaled_frame)
            
        except Exception as e:
            tk.Label(
                self.plot_frame,
                text=f"Plot generated successfully.\nUse 'Export Plot' to save.\n\n(Display error: {e})",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
                wraplength=400,
            ).pack(expand=True, padx=20, pady=20)
    
    def _add_save_menu(self, widget):
        """Add right-click context menu for saving."""
        def show_menu(event):
            menu = tk.Menu(widget, tearoff=0)
            menu.add_command(label="📄 Add to Report", command=self._add_plot_to_report)
            menu.add_separator()
            menu.add_command(label="💾 Save as PNG...", command=lambda: self._save_as("png"))
            menu.add_command(label="💾 Save as PDF...", command=lambda: self._save_as("pdf"))
            menu.add_command(label="💾 Save as SVG...", command=lambda: self._save_as("svg"))
            menu.tk_popup(event.x_root, event.y_root)
        
        widget.bind("<Button-3>", show_menu)
    
    def _add_plot_to_report(self):
        """Add current plot to report queue."""
        if self._current_fig is None:
            messagebox.showinfo("No Plot", "No plot to add to report.")
            return
        
        try:
            from biblium.gui.core.state import report_queue
            
            plot_title = "Top Items Timeline"
            if self._current_fig.axes:
                plot_title = self._current_fig.axes[0].get_title() or "Top Items Timeline"
            
            report_queue.add_plot(
                figure_or_bytes=self._current_fig,
                title=plot_title,
                source_panel=self.title,
            )
            
            messagebox.showinfo(
                "Added to Report",
                f"Plot '{plot_title}' has been added to the report queue.\n\n"
                f"Items in queue: {len(report_queue)}\n\n"
                "Go to Reports → Report Builder to generate your report."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add to report: {e}")
    
    def _save_as(self, fmt):
        """Save figure in specified format."""
        if self._current_fig is None:
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} files", f"*.{fmt}")],
            title=f"Save as {fmt.upper()}",
        )
        if filepath:
            try:
                self._current_fig.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Saved", f"Plot saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
    
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
        
        table = DataTable(
            self.data_frame,
            theme=self.theme_name,
            show_index=False,
        )
        table.set_data(self._result_df)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    
    def _export_plot(self):
        """Export the plot."""
        if self._current_fig is None:
            messagebox.showwarning("No Plot", "Please generate a plot first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            title="Export Plot",
        )
        
        if filepath:
            try:
                self._current_fig.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Export Complete", f"Plot exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def _export_data(self):
        """Export the data."""
        if self._result_df is None or self._result_df.empty:
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")],
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
        
        available_items = self._get_available_items()
        self.item_type.configure(values=available_items)
        if available_items:
            self.item_type.set(available_items[0])
    
    def _cleanup_figure(self):
        """Clean up matplotlib figure."""
        if self._current_fig is not None:
            try:
                plt.close(self._current_fig)
            except:
                pass
            self._current_fig = None
        self._canvas = None
        self._toolbar = None
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
TOP ITEMS TIMELINE
══════════════════

Track top entities across time periods.

WHAT IT SHOWS
─────────────
• Top N entities per period
• Ranking changes over time
• Entry and exit patterns
• Persistence analysis

ENTITY TYPES
────────────
• Top authors by period
• Top journals over time
• Top keywords by year
• Top countries annually

VISUALIZATION
─────────────
• Bump chart (rankings)
• Stream graph (volumes)
• Animated bar race
• Presence heatmap

ANALYSIS
────────
• Rising entities
• Declining entities
• Consistent leaders
• New entrants
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
