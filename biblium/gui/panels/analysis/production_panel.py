# -*- coding: utf-8 -*-
"""
Production Over Time Panel
==========================
Analyze top authors, sources, keywords, countries production over time.
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
from biblium.gui.widgets.forms import LabeledCombobox, LabeledSpinbox, LabeledCheckbox
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

# Available items for production over time analysis
# Format: (Display Name, key, [possible column names])
AVAILABLE_ITEMS_CONFIG = [
    ("Authors", "authors", ["Authors", "Author full names", "AU", "AF", "Author Full Names"]),
    ("Author Keywords", "author keywords", ["Author Keywords", "DE", "Keywords"]),
    ("Index Keywords", "index keywords", ["Index Keywords", "ID", "Keywords Plus", "Keyword Plus"]),
    ("Sources", "sources", ["Source title", "Source", "SO", "Journal", "Publication Name"]),
    ("Countries", "all countries", ["Countries", "Country", "Countries of Authors"]),
    ("Affiliations", "affiliations", ["Affiliations", "C1", "Addresses", "Author Affiliations"]),
    ("Document Types", "document types", ["Document Type", "DT", "Document type", "Publication Type", "Type"]),
    ("References", "references", ["References", "Cited References", "CR", "Referenced Works"]),
    ("Cited Sources", "cited sources", ["Cited Sources", "Cited Journals", "Cited Source"]),
    ("Cited Authors", "cited authors", ["Cited Authors", "Cited Author", "First Authors of Cited References"]),
    ("Research Areas", "research areas", ["Research Areas", "WoS Categories", "SC", "Subject Area", "Fields"]),
    ("Subject Categories", "subject categories", ["Web of Science Categories", "WC", "Subject Categories", "Categories"]),
]

def get_available_items_for_df(df):
    """Return list of available items based on columns in dataframe."""
    if df is None:
        return [(name, key) for name, key, _ in AVAILABLE_ITEMS_CONFIG]
    
    available = []
    columns_lower = {c.lower(): c for c in df.columns}
    
    for display_name, key, possible_cols in AVAILABLE_ITEMS_CONFIG:
        # Check if any of the possible columns exist
        found = False
        for col in possible_cols:
            if col in df.columns:
                found = True
                break
            # Case-insensitive check
            if col.lower() in columns_lower:
                found = True
                break
        if found:
            available.append((display_name, key))
    
    return available if available else [(name, key) for name, key, _ in AVAILABLE_ITEMS_CONFIG]


class ProductionOverTimePanel(BasePanel):
    """Panel for analyzing production over time of top items."""
    
    title = "Production Over Time"
    icon = "📅"
    description = "Analyze production over time of top authors, sources, keywords, countries"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._result_df = None
        self._current_fig = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Item Selection Card
        item_card = Card(self.options_content, title="📊 Item Selection", theme=self.theme_name)
        item_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Get available items based on current data
        available_items = self._get_available_items()
        item_values = [f[0] for f in available_items]
        
        self.item_type = LabeledCombobox(
            item_card.content, label="Analyze:",
            values=item_values,
            default=item_values[0] if item_values else "Authors",
            theme=self.theme_name, label_width=10,
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
        
        # Plot Type Card
        plot_type_card = Card(self.options_content, title="📈 Plot Type", theme=self.theme_name)
        plot_type_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.plot_type = LabeledCombobox(
            plot_type_card.content, label="View:",
            values=["Per Year", "Cumulative"],
            default="Per Year",
            theme=self.theme_name, label_width=12,
        )
        self.plot_type.pack(fill=tk.X, pady=4)
        
        # Plot Options Card
        plot_card = Card(self.options_content, title="🎨 Plot Options", theme=self.theme_name)
        plot_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.cmap = LabeledCombobox(
            plot_card.content, label="Color Map:",
            values=["tab10", "Set1", "Set2", "Dark2", "Paired", "viridis", "plasma"],
            default="tab10",
            theme=self.theme_name, label_width=12,
        )
        self.cmap.pack(fill=tk.X, pady=4)
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
        
        # Initialize with placeholder
        self._show_placeholder()
        self._create_info_tab()
    
    def _show_placeholder(self):
        """Show placeholder message."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame,
            text="Select an item type and click 'Run Analysis'\nto see production over time",
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
PRODUCTION OVER TIME ANALYSIS
═════════════════════════════

This analysis shows the publication activity of top items (authors, 
sources, keywords, countries) over time.

VISUALIZATION
─────────────
• Each row represents one item (author, source, keyword, etc.)
• Horizontal lines show the publication span (first to last year)
• Circle size indicates number of documents in that year
• Circle color represents the selected metric (citations, h-index)

METRICS COMPUTED
────────────────
Per item, per year:
• Number of documents
• Total citations
• Citations per document

Per item (overall):
• Total documents
• Total citations
• H-index
• First and last publication year

AVAILABLE ITEMS
───────────────
• Authors: Individual author production patterns
• Author Keywords: Author-assigned keyword trends over time
• Index Keywords: Database-assigned keyword trends (Keywords Plus)
• Sources: Journal/venue publication patterns
• Countries: Geographic publication distribution
• Affiliations: Institutional production patterns
• Document Types: Distribution of article, review, conference paper, etc.
• References: Most cited references over time
• Cited Sources: Most cited journals in references
• Cited Authors: Most cited authors in references
• Research Areas: Subject area trends
• Subject Categories: WoS category distribution

INTERPRETATION
──────────────
• Rising lines indicate growth in output
• Steep slopes show rapid productivity increases
• Flat lines indicate stable production
• Gaps may indicate publication pauses

PLOT TYPES
──────────
• Per Year: Shows yearly document count as lines
• Cumulative: Shows accumulated output over time

SETTINGS
────────
• Top N: Number of top items to analyze (by total documents)
• Min Documents: Minimum documents required per item
• View: Choose between Per Year or Cumulative view
• Color Map: Color scheme for lines (tab10, Set1 best for distinct colors)
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
        """Run the production over time analysis."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self.run_btn.config(state=tk.DISABLED, text="⏳ Analyzing...")
        
        # Capture values before thread
        item_key = self._get_item_key(self.item_type.get())
        top_n = self.top_n_spin.get()
        min_docs = self.min_docs_spin.get()
        cmap = self.cmap.get()
        plot_type = self.plot_type.get()
        
        def do_analysis():
            error_info = None
            try:
                from biblium import utilsbib, plotbib
                
                # Build column mapping from config
                col_mapping = {key: cols for _, key, cols in AVAILABLE_ITEMS_CONFIG}
                
                # Find the actual column
                possible_cols = col_mapping.get(item_key, [item_key])
                actual_col = None
                for col in possible_cols:
                    if col in self.bib.df.columns:
                        actual_col = col
                        break
                
                if actual_col is None:
                    # Try case-insensitive match
                    for col in self.bib.df.columns:
                        col_lower = col.lower()
                        item_lower = item_key.replace("_", " ").lower()
                        if col_lower == item_lower or col_lower.replace(" ", "") == item_lower.replace(" ", ""):
                            actual_col = col
                            break
                
                if actual_col is None:
                    raise ValueError(f"Could not find column for '{item_key}' in dataset.")
                
                # Compute using utilsbib
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
                
                # Create line chart based on plot_type
                if plot_type == "Cumulative":
                    fig = self._create_cumulative_plot(result_df, actual_col, cmap)
                else:
                    # Per Year view
                    fig = self._create_per_year_plot(result_df, actual_col, cmap)
                
                self._result_df = result_df
                self._current_fig = fig
                
                self.after(0, self._on_analysis_success)
                
            except Exception as exc:
                import traceback
                error_info = (str(exc), traceback.format_exc())
                self.after(0, lambda ei=error_info: self._on_analysis_error(ei[0], ei[1]))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _create_per_year_plot(self, result_df, y_label, cmap):
        """Create per-year line chart for entities."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        from matplotlib.figure import Figure
        import numpy as np
        
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Find columns
        item_col = result_df.columns[0]
        year_col = None
        docs_col = None
        
        for col in ['Year', 'year', 'Publication Year']:
            if col in result_df.columns:
                year_col = col
                break
        
        for col in ['n_docs', 'Number of documents', 'Documents']:
            if col in result_df.columns:
                docs_col = col
                break
        
        if year_col is None or docs_col is None:
            raise ValueError(f"Required columns not found. Available: {list(result_df.columns)}")
        
        # Get unique items
        items = result_df[item_col].unique()
        
        # Get colormap
        import matplotlib.pyplot as plt
        if cmap in ['tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2']:
            colors = plt.cm.get_cmap(cmap).colors[:len(items)]
        else:
            colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, len(items)))
        
        # Plot line for each item
        for idx, item in enumerate(items):
            item_data = result_df[result_df[item_col] == item].copy()
            if item_data.empty:
                continue
            
            item_data = item_data.sort_values(year_col)
            
            ax.plot(
                item_data[year_col], 
                item_data[docs_col],
                color=colors[idx % len(colors)],
                linewidth=2,
                marker='o',
                markersize=4,
                label=str(item)[:30] + ('...' if len(str(item)) > 30 else ''),
            )
        
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel(f'Documents ({y_label})', fontsize=10)
        ax.set_title('Entity Production Per Year', fontsize=12)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(False)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        
        fig.tight_layout()
        return fig
    
    def _create_cumulative_plot(self, result_df, y_label, cmap):
        """Create cumulative growth plot for entities."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Find columns
        item_col = result_df.columns[0]
        year_col = None
        docs_col = None
        
        for col in ['Year', 'year', 'Publication Year']:
            if col in result_df.columns:
                year_col = col
                break
        
        for col in ['n_docs', 'Number of documents', 'Documents']:
            if col in result_df.columns:
                docs_col = col
                break
        
        if year_col is None or docs_col is None:
            raise ValueError(f"Required columns not found. Available: {list(result_df.columns)}")
        
        # Get unique items
        items = result_df[item_col].unique()
        
        # Get colormap - use qualitative colormaps for distinct colors
        if cmap in ['tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Accent']:
            colors = plt.cm.get_cmap(cmap).colors[:len(items)]
        else:
            colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, len(items)))
        
        # Plot cumulative line for each item
        for idx, item in enumerate(items):
            item_data = result_df[result_df[item_col] == item].copy()
            if item_data.empty:
                continue
            
            # Sort by year and compute cumulative
            item_data = item_data.sort_values(year_col)
            item_data['cumulative'] = item_data[docs_col].cumsum()
            
            # Plot
            ax.plot(
                item_data[year_col], 
                item_data['cumulative'],
                color=colors[idx % len(colors)],
                linewidth=2,
                marker='o',
                markersize=4,
                label=str(item)[:30] + ('...' if len(str(item)) > 30 else ''),
            )
        
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel(f'Cumulative Documents ({y_label})', fontsize=10)
        ax.set_title('Cumulative Growth Over Time', fontsize=12)
        ax.tick_params(axis='both', labelsize=8)
        
        # No gridlines
        ax.grid(False)
        
        # Legend in upper left corner
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        
        fig.tight_layout()
        return fig
    
    def _on_analysis_success(self):
        """Handle successful analysis."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Run Analysis")
        self._update_plot_tab()
        self._update_data_tab()
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
            
            plot_title = "Production Analysis"
            if self._current_fig.axes:
                plot_title = self._current_fig.axes[0].get_title() or "Production Analysis"
            
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
        
        # Format the dataframe for display
        df = self._result_df.copy()
        
        # Round numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if "citation" in col.lower() or "per" in col.lower():
                df[col] = df[col].round(2)
            else:
                df[col] = df[col].round(0).astype(int, errors='ignore')
        
        # Create data table
        table = DataTable(
            self.data_frame,
            theme=self.theme_name,
            show_index=False,
        )
        table.set_data(df)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    
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
        
        if not filepath:
            return
        
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
        
        # Update available items in dropdown
        self._update_available_items()
    
    def _update_available_items(self):
        """Update the item dropdown based on available columns."""
        if hasattr(self, 'item_type'):
            available_items = self._get_available_items()
            item_values = [f[0] for f in available_items]
            
            # Update combobox values
            self.item_type.combo['values'] = item_values
            
            # Keep current selection if still valid, otherwise reset
            current = self.item_type.get()
            if current not in item_values and item_values:
                self.item_type.combo.set(item_values[0])
    
    def _cleanup_figure(self):
        """Clean up matplotlib figure to prevent threading issues."""
        if self._current_fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._current_fig)
            except:
                pass
            self._current_fig = None
        
        # Clear canvas reference
        if hasattr(self, '_canvas'):
            self._canvas = None
        if hasattr(self, '_toolbar'):
            self._toolbar = None
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
SCIENTIFIC PRODUCTION
═════════════════════

Analyze publication output patterns.

METRICS
───────
• Total documents
• Annual output
• Growth rate
• Author productivity

AUTHOR ANALYSIS
───────────────
• Most productive authors
• Productivity distribution
• Lotka's Law fit

SOURCE ANALYSIS
───────────────
• Core journals
• Bradford zones
• Source productivity

TEMPORAL PATTERNS
─────────────────
• Publication trends
• Growth modeling
• Period comparisons
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
        """Clean up resources before destroying widget."""
        self._cleanup_figure()
        super().destroy()
