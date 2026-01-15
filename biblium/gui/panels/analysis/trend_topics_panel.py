# -*- coding: utf-8 -*-
"""
Trend Topics Panel
==================
Analyze trending topics over time using median year ordering.
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
from biblium.gui.widgets.forms import LabeledCombobox, LabeledSpinbox, LabeledEntry
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

# Available items for trend topics analysis with column mappings
AVAILABLE_ITEMS_CONFIG = [
    ("Author Keywords", "author keywords", ["Author Keywords", "DE", "Keywords"]),
    ("Index Keywords", "index keywords", ["Index Keywords", "ID", "Keywords Plus", "Keyword Plus"]),
    ("Authors", "authors", ["Authors", "Author full names", "AU", "AF", "Author Full Names"]),
    ("Sources", "sources", ["Source title", "Source", "SO", "Journal", "Publication Name"]),
    ("Countries", "all countries", ["Countries", "Country", "Countries of Authors"]),
    ("Affiliations", "affiliations", ["Affiliations", "C1", "Addresses", "Author Affiliations"]),
    ("References", "references", ["References", "Cited References", "CR", "Referenced Works"]),
]

def get_available_items_for_df(df):
    """Return list of available items based on columns in dataframe."""
    if df is None:
        return [(name, key) for name, key, _ in AVAILABLE_ITEMS_CONFIG]
    
    available = []
    columns_lower = {c.lower(): c for c in df.columns}
    
    for display_name, key, possible_cols in AVAILABLE_ITEMS_CONFIG:
        found = False
        for col in possible_cols:
            if col in df.columns or col.lower() in columns_lower:
                found = True
                break
        if found:
            available.append((display_name, key))
    
    return available if available else [(name, key) for name, key, _ in AVAILABLE_ITEMS_CONFIG]


class TrendTopicsPanel(BasePanel):
    """Panel for analyzing trend topics over time."""
    
    title = "Trend Topics"
    icon = "📈"
    description = "Analyze trending topics ordered by median publication year"
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
        item_card = Card(self.options_content, title="📊 Item Selection", theme=self.theme_name)
        item_card.pack(fill=tk.X, padx=8, pady=8)
        
        available_items = self._get_available_items()
        item_values = [f[0] for f in available_items]
        
        self.item_type = LabeledCombobox(
            item_card.content, label="Analyze:",
            values=item_values,
            default=item_values[0] if item_values else "Author Keywords",
            theme=self.theme_name, label_width=10,
        )
        self.item_type.pack(fill=tk.X, pady=4)
        
        # Settings Card
        settings_card = Card(self.options_content, title="⚙️ Settings", theme=self.theme_name)
        settings_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.min_docs_spin = LabeledSpinbox(
            settings_card.content, label="Min Documents:",
            from_=1, to=50, default=3,
            theme=self.theme_name, label_width=14,
        )
        self.min_docs_spin.pack(fill=tk.X, pady=4)
        
        self.top_n_year_spin = LabeledSpinbox(
            settings_card.content, label="Top N per Year:",
            from_=1, to=20, default=3,
            theme=self.theme_name, label_width=14,
        )
        self.top_n_year_spin.pack(fill=tk.X, pady=4)
        
        # Filter frame
        filter_frame = tk.Frame(settings_card.content, bg=self.theme["bg_card"])
        filter_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            filter_frame, text="Regex Filter:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            width=14, anchor="w",
        ).pack(side=tk.LEFT)
        
        self.regex_entry = tk.Entry(
            filter_frame,
            font=FONTS.get_font("body"),
            width=20,
        )
        self.regex_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Plot Options Card
        plot_card = Card(self.options_content, title="🎨 Plot Options", theme=self.theme_name)
        plot_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.color_by = LabeledCombobox(
            plot_card.content, label="Color By:",
            values=["Total citations", "Citations per document", "H-index"],
            default="Total citations",
            theme=self.theme_name, label_width=12,
        )
        self.color_by.pack(fill=tk.X, pady=4)
        
        self.cmap = LabeledCombobox(
            plot_card.content, label="Color Map:",
            values=["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Reds", "Greens"],
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
            btn_frame, text="Run Analysis",
            command=self._run_analysis,
            icon="▶",
            theme=self.theme_name,
        )
        self.run_btn.pack(fill=tk.X)
        
        # Export button
        self.export_btn = ThemedButton(
            btn_frame, text="Export Plot",
            command=self._export_plot,
            style="secondary",
            icon="🖼️",
            theme=self.theme_name,
        )
        self.export_btn.pack(fill=tk.X, pady=(8, 0))
    
    def _create_results(self):
        """Create the results panel."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Plot tab
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="📈 Trend Plot")
        
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
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame,
            text="Select an item type and click 'Run Analysis'\nto see trend topics over time",
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
TREND TOPICS ANALYSIS
═════════════════════

Trend Topics shows which topics (keywords, authors, sources, etc.) 
are trending based on their median publication year.

HOW IT WORKS
────────────
• Items are ordered by their median publication year (most recent first)
• Each row shows an item with its publication time span (Q1 to Q3)
• Circle size indicates the number of documents
• Circle color represents the selected metric (citations, h-index)
• Horizontal lines show the interquartile range (25th-75th percentile)

INTERPRETATION
──────────────
• Items at the TOP are the most recent trends
• Items at the BOTTOM are older/established topics
• Long horizontal lines indicate sustained activity
• Short lines indicate burst activity in a specific period
• Large circles show high-volume topics
• Bright colors indicate high-impact topics

SETTINGS
────────
• Min Documents: Minimum number of documents for an item to be included
• Top N per Year: Maximum items shown per median year (prevents overcrowding)
• Regex Filter: Optional pattern to filter items (e.g., "machine|deep" for ML topics)
• Color By: Metric for coloring (total citations, citations per doc, h-index)

USE CASES
─────────
• Identify emerging research topics
• Track keyword evolution over time
• Find rising authors or institutions
• Discover trending journals in a field
• Analyze temporal patterns in research themes
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
    
    def _get_item_key(self, display_name: str) -> str:
        """Convert display name to item key."""
        for name, key, _ in AVAILABLE_ITEMS_CONFIG:
            if name == display_name:
                return key
        return display_name.lower()
    
    def _run_analysis(self):
        """Run the trend topics analysis."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self.run_btn.config(state=tk.DISABLED, text="⏳ Analyzing...")
        
        # Capture values before thread
        item_display = self.item_type.get()
        item_key = self._get_item_key(item_display)
        min_docs = self.min_docs_spin.get()
        top_n_year = self.top_n_year_spin.get()
        regex_filter = self.regex_entry.get().strip() or None
        color_by = self.color_by.get()
        cmap = self.cmap.get()
        
        def do_analysis():
            error_info = None
            try:
                # Check if bib has the mapping for this item
                has_mapping = (
                    hasattr(self.bib, 'mapping')
                    and item_key in self.bib.mapping
                    and isinstance(self.bib.mapping.get(item_key), dict)
                    and "stats_attr" in self.bib.mapping.get(item_key, {})
                )
                
                if has_mapping:
                    # Use biblium's plot_trend_topics method
                    # Set colormap
                    if hasattr(self.bib, 'cmap'):
                        old_cmap = self.bib.cmap
                    else:
                        old_cmap = None
                    self.bib.cmap = cmap
                    
                    try:
                        fig = self.bib.plot_trend_topics(
                            items=item_display,
                            min_docs=min_docs,
                            top_n_year=top_n_year,
                            regex_filter=regex_filter,
                            color_by=color_by,
                        )
                    finally:
                        if old_cmap is not None:
                            self.bib.cmap = old_cmap
                else:
                    # Fallback: compute directly
                    fig = self._compute_trend_topics_fallback(
                        item_key, item_display, min_docs, top_n_year, 
                        regex_filter, color_by, cmap
                    )
                
                self._current_fig = fig
                self.after(0, self._on_analysis_success)
                
            except Exception as exc:
                import traceback
                error_info = (str(exc), traceback.format_exc())
                self.after(0, lambda ei=error_info: self._on_analysis_error(ei[0], ei[1]))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _compute_trend_topics_fallback(self, item_key, item_display, min_docs, 
                                        top_n_year, regex_filter, color_by, cmap):
        """Fallback computation when mapping is not available."""
        from biblium import utilsbib, plotbib
        import re
        
        # Build column mapping
        col_mapping = {key: cols for _, key, cols in AVAILABLE_ITEMS_CONFIG}
        
        # Find the actual column
        possible_cols = col_mapping.get(item_key, [item_key])
        actual_col = None
        for col in possible_cols:
            if col in self.bib.df.columns:
                actual_col = col
                break
        
        if actual_col is None:
            # Try case-insensitive
            columns_lower = {c.lower(): c for c in self.bib.df.columns}
            for col in possible_cols:
                if col.lower() in columns_lower:
                    actual_col = columns_lower[col.lower()]
                    break
        
        if actual_col is None:
            raise ValueError(f"Could not find column for '{item_key}'")
        
        # Compute stats using utilsbib
        separator = getattr(self.bib, 'default_separator', '; ')
        if hasattr(self.bib, 'db') and self.bib.db == 'oa':
            separator = r"\|"
        
        # Get item statistics with year info
        stats_df = utilsbib.compute_item_time_stats(
            self.bib.df,
            actual_col,
            list_separators=separator,
            top_n=100,  # Get more items for filtering
            min_docs_per_item=min_docs,
        )
        
        if stats_df.empty:
            raise ValueError("No items found after filtering")
        
        # Add median year if not present
        item_col = stats_df.columns[0]  # First column is usually the item
        
        # Compute median and quartiles per item
        per_item = stats_df.groupby(item_col).agg({
            'Year': ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
            'n_docs': 'sum',
            'total_citations': 'sum',
        })
        per_item.columns = ['Median Year', 'Q1', 'Q3', 'Number of documents', 'Total citations']
        per_item = per_item.reset_index()
        
        # Apply regex filter
        if regex_filter:
            pat = re.compile(regex_filter, flags=re.I)
            per_item = per_item[per_item[item_col].astype(str).apply(lambda s: bool(pat.search(s)))]
        
        # Apply top N per year filter
        if top_n_year and top_n_year > 0:
            per_item['_my'] = per_item['Median Year'].round().astype('Int64')
            per_item = (
                per_item.sort_values(['_my', 'Number of documents'], ascending=[False, False])
                .groupby('_my', group_keys=False)
                .head(top_n_year)
                .drop(columns=['_my'])
            )
        
        if per_item.empty:
            raise ValueError("No items left after filtering")
        
        # Order by median year
        order = per_item.sort_values(
            ['Median Year', 'Number of documents'], 
            ascending=[False, False]
        )[item_col].tolist()
        
        # Plot using plotbib
        fig = plotbib.plot_item_time_stats(
            per_item,
            year_col='Median Year',
            item_order=order,
            item_col=item_col,
            color_col=color_by.lower().replace(' ', '_'),
            cmap=cmap,
            y_label=item_display,
            line_width=1.0,
            line_alpha=0.9,
            size_legend=True,
            size_legend_title="Number of documents",
            tick_fontsize=8,
            label_fontsize=10,
        )
        
        return fig
    
    def _on_analysis_success(self):
        """Handle successful analysis."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Run Analysis")
        self._update_plot_tab()
        self.notebook.select(0)
    
    def _on_analysis_error(self, error_msg: str, traceback_str: str):
        """Handle analysis error."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Run Analysis")
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
            
            plot_title = "Trend Topics"
            if self._current_fig.axes:
                plot_title = self._current_fig.axes[0].get_title() or "Trend Topics"
            
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
    
    def _export_plot(self):
        """Export the plot."""
        if self._current_fig is None:
            messagebox.showwarning("No Plot", "Please run analysis first.")
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
        
        if not filepath:
            return
        
        try:
            self._current_fig.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Export Complete", f"Plot exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def on_data_loaded(self, bib):
        """Handle data loaded event."""
        self.bib = bib
        self._cleanup_figure()
        self._show_placeholder()
        self._update_available_items()
    
    def _update_available_items(self):
        """Update the item dropdown based on available columns."""
        if hasattr(self, 'item_type'):
            available_items = self._get_available_items()
            item_values = [f[0] for f in available_items]
            
            self.item_type.combo['values'] = item_values
            
            current = self.item_type.get()
            if current not in item_values and item_values:
                self.item_type.combo.set(item_values[0])
    
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
TREND TOPICS
════════════

Identify trending and declining topics.

DETECTION METHODS
─────────────────
• Growth rate analysis
• Burst detection
• Emergence detection
• Decline patterns

TOPIC CATEGORIES
────────────────
• 🔥 Hot: Rapid growth
• 🆕 Emerging: New appearances
• ➡️ Stable: Consistent
• 📉 Declining: Decreasing
• 💤 Dormant: Inactive

METRICS
───────
• Growth rate (%)
• Burst score
• Persistence (years active)
• Peak timing

OUTPUT
──────
• Trend classification
• Growth visualization
• Topic comparison
• Forecast indicators
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
