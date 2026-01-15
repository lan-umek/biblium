# -*- coding: utf-8 -*-
"""
SDG Identifier Panel
====================
Panel for identifying Sustainable Development Goals in bibliometric data.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import pandas as pd
import io
import os
from typing import Optional

from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.buttons import ActionButton, ThemedButton
from biblium.gui.widgets.cards import Card, StatsCard, CardGrid
from biblium.gui.widgets.forms import LabeledCombobox, LabeledEntry, LabeledCheckbox
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import ScaledImageFrame
from biblium.gui.config import FONTS, get_theme
from biblium.gui.core.events import event_bus, EventBus


# SDG Names for display
SDG_NAMES = {
    "SDG01": "No Poverty",
    "SDG02": "Zero Hunger",
    "SDG03": "Good Health and Well-being",
    "SDG04": "Quality Education",
    "SDG05": "Gender Equality",
    "SDG06": "Clean Water and Sanitation",
    "SDG07": "Affordable and Clean Energy",
    "SDG08": "Decent Work and Economic Growth",
    "SDG09": "Industry, Innovation and Infrastructure",
    "SDG10": "Reduced Inequalities",
    "SDG11": "Sustainable Cities and Communities",
    "SDG12": "Responsible Consumption and Production",
    "SDG13": "Climate Action",
    "SDG14": "Life Below Water",
    "SDG15": "Life on Land",
    "SDG16": "Peace, Justice and Strong Institutions",
    "SDG17": "Partnerships for the Goals",
}


class SDGPanel(BasePanel):
    """Panel for SDG identification and analysis."""
    
    title = "SDG Identifier"
    icon = "🌍"
    description = "Identify Sustainable Development Goals in your dataset"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", **kwargs):
        self._sdg_results = None
        self._last_image_bytes = None
        super().__init__(parent, theme=theme, **kwargs)
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Text Column Selection Card
        text_card = Card(self.options_content, title="📝 Text Analysis", theme=self.theme_name)
        text_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.text_col_combo = LabeledCombobox(
            text_card.content,
            label="Text Column:",
            values=["Abstract", "Title", "Combined Text"],
            default="Abstract",
            theme=self.theme_name,
            label_width=15,
        )
        self.text_col_combo.pack(fill=tk.X, pady=4)
        
        tk.Label(
            text_card.content,
            text="Select the column containing text to analyze for SDG keywords",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 4))
        
        # SDG Queries File Card
        queries_card = Card(self.options_content, title="📋 SDG Queries", theme=self.theme_name)
        queries_card.pack(fill=tk.X, padx=8, pady=8)
        
        # File selection
        file_frame = tk.Frame(queries_card.content, bg=self.theme["bg_card"])
        file_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            file_frame, text="Queries File:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], width=12, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self.queries_file_var = tk.StringVar(value="(default)")
        tk.Entry(
            file_frame, textvariable=self.queries_file_var,
            font=FONTS.get_font("body"), width=15,
            state="readonly",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        ThemedButton(
            file_frame, text="Browse", style="ghost", size="small",
            command=self._browse_queries_file, theme=self.theme_name,
        ).pack(side=tk.LEFT)
        
        tk.Label(
            queries_card.content,
            text="Uses Scopus SDG metadata by default",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W)
        
        # Output Options Card
        output_card = Card(self.options_content, title="⚙️ Output Options", theme=self.theme_name)
        output_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.perspectives_cb = LabeledCheckbox(
            output_card.content,
            label="Include Perspectives",
            default=True,
            theme=self.theme_name,
        )
        self.perspectives_cb.pack(fill=tk.X, pady=2)
        
        self.dimensions_cb = LabeledCheckbox(
            output_card.content,
            label="Include Dimensions",
            default=True,
            theme=self.theme_name,
        )
        self.dimensions_cb.pack(fill=tk.X, pady=2)
        
        self.add_to_df_cb = LabeledCheckbox(
            output_card.content,
            label="Add results to dataset",
            default=True,
            theme=self.theme_name,
        )
        self.add_to_df_cb.pack(fill=tk.X, pady=2)
        
        # Action Buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Identify SDGs", icon="🔍",
            command=self._run_identification, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        ThemedButton(
            btn_frame, text="Export Results", style="secondary",
            command=self._export_results, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
    
    def _create_results(self):
        """Create the results panel."""
        # Results card
        self.results_card = tk.Frame(
            self.results_frame, bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Header with tabs
        header = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header, text="SDG Analysis Results",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            padx=12, pady=8,
        ).pack(side=tk.LEFT)
        
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.results_card)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Summary tab
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📊 Summary")
        
        # Distribution tab
        self.dist_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.dist_frame, text="📈 Distribution")
        
        # Documents tab
        self.docs_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.docs_frame, text="📄 Documents")
        
        # Info tab
        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        
        # Initial placeholder
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Show placeholder message."""
        for frame in [self.summary_frame, self.dist_frame, self.docs_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        tk.Label(
            self.summary_frame,
            text="Click 'Identify SDGs' to analyze your dataset\nfor Sustainable Development Goals",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True, pady=50)
    
    def _browse_queries_file(self):
        """Browse for SDG queries file."""
        filepath = filedialog.askopenfilename(
            title="Select SDG Queries File",
            filetypes=[("Excel Files", "*.xlsx *.xls"), ("All Files", "*.*")],
        )
        if filepath:
            self.queries_file_var.set(os.path.basename(filepath))
            self._custom_queries_path = filepath
        else:
            self._custom_queries_path = None
    
    def _run_identification(self):
        """Run SDG identification."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        # Get text column
        text_col = self.text_col_combo.get()
        
        # Find actual column name in data
        col_map = {
            "Abstract": ["Abstract", "AB", "abstract"],
            "Title": ["Title", "TI", "title", "Document Title"],
            "Combined Text": ["Combined Text", "combined_text", "Text"],
        }
        
        actual_col = None
        for candidate in col_map.get(text_col, [text_col]):
            if candidate in self.bib.df.columns:
                actual_col = candidate
                break
        
        if not actual_col:
            messagebox.showerror("Error", f"Column '{text_col}' not found in dataset.")
            return
        
        # Show progress
        self._show_progress("Identifying SDGs in documents...")
        
        # Get options
        return_perspectives = self.perspectives_cb.get()
        return_dimensions = self.dimensions_cb.get()
        queries_path = getattr(self, '_custom_queries_path', None)
        
        def do_identification():
            try:
                from biblium.sdg_identifier import identify_sdgs
                
                result_df = identify_sdgs(
                    self.bib.df,
                    text_column=actual_col,
                    sdg_queries_path=queries_path,
                    return_perspectives=return_perspectives,
                    return_dimensions=return_dimensions,
                )
                
                self._sdg_results = result_df
                
                # Add to bib.df if requested
                add_to_df = self.add_to_df_cb.get()
                
                self.after(0, lambda: self._show_results(result_df, add_to_df))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self._show_error(str(e)))
        
        threading.Thread(target=do_identification, daemon=True).start()
    
    def _show_progress(self, message: str):
        """Show progress message."""
        for frame in [self.summary_frame, self.dist_frame, self.docs_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        tk.Label(
            self.summary_frame,
            text=f"⏳ {message}",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(expand=True, pady=50)
    
    def _show_error(self, error: str):
        """Show error message."""
        for frame in [self.summary_frame, self.dist_frame, self.docs_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        tk.Label(
            self.summary_frame,
            text=f"❌ Error: {error}",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["danger"],
            wraplength=400,
        ).pack(expand=True, pady=50)
    
    def _show_results(self, result_df: pd.DataFrame, add_to_df: bool):
        """Show identification results."""
        # Clear all frames
        for frame in [self.summary_frame, self.dist_frame, self.docs_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        # Get SDG columns
        import re
        sdg_cols = [c for c in result_df.columns if re.match(r'SDG\d+', c)]
        
        # Calculate statistics
        total_docs = len(result_df)
        docs_with_sdg = (result_df['any_SDG'] == 1).sum() if 'any_SDG' in result_df.columns else 0
        coverage_pct = (docs_with_sdg / total_docs * 100) if total_docs > 0 else 0
        
        # SDG counts
        sdg_counts = {}
        for col in sdg_cols:
            count = result_df[col].sum()
            if count > 0:
                sdg_counts[col] = count
        
        # === SUMMARY TAB ===
        # Stats cards
        stats_frame = tk.Frame(self.summary_frame, bg=self.theme["bg_card"])
        stats_frame.pack(fill=tk.X, pady=(8, 16))
        
        grid = CardGrid(stats_frame, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X)
        
        grid.add_card(StatsCard(grid, "Total Documents", f"{total_docs:,}", "📄", self.theme_name))
        grid.add_card(StatsCard(grid, "With SDG", f"{docs_with_sdg:,}", "🌍", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Coverage", f"{coverage_pct:.1f}%", "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "SDGs Found", f"{len(sdg_counts)}", "🎯", self.theme_name))
        
        # SDG breakdown table
        tk.Label(
            self.summary_frame,
            text="SDG Breakdown",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, padx=8, pady=(16, 8))
        
        if sdg_counts:
            breakdown_data = []
            for sdg, count in sorted(sdg_counts.items()):
                pct = count / total_docs * 100
                name = SDG_NAMES.get(sdg, sdg)
                breakdown_data.append({
                    "SDG": sdg,
                    "Name": name,
                    "Documents": count,
                    "Percentage": f"{pct:.1f}%",
                })
            
            breakdown_df = pd.DataFrame(breakdown_data)
            table = DataTable(self.summary_frame, theme=self.theme_name, max_rows=20)
            table.pack(fill=tk.BOTH, expand=True, padx=8)
            table.set_data(breakdown_df)
        
        # === DISTRIBUTION TAB ===
        self._create_distribution_chart(result_df, sdg_counts)
        
        # === DOCUMENTS TAB ===
        self._create_documents_table(result_df, sdg_cols)
        
        # Add results to main dataframe if requested
        if add_to_df and self.bib:
            # Add SDG columns to bib.df
            for col in sdg_cols + ['any_SDG']:
                if col in result_df.columns:
                    self.bib.df[col] = result_df[col]
            
            # Add perspective columns
            persp_cols = [c for c in result_df.columns if c.startswith('perspective_')]
            for col in persp_cols:
                self.bib.df[col] = result_df[col]
            
            # Add dimension columns
            dim_cols = [c for c in result_df.columns if c.startswith('dimension_')]
            for col in dim_cols:
                self.bib.df[col] = result_df[col]
            
            # Emit event
            event_bus.emit(EventBus.ANALYSIS_COMPLETED, {
                "name": "SDG Identification",
                "added_columns": sdg_cols + ['any_SDG'] + persp_cols + dim_cols,
            })
    
    def _create_distribution_chart(self, result_df: pd.DataFrame, sdg_counts: dict):
        """Create SDG distribution bar chart."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            if not sdg_counts:
                tk.Label(
                    self.dist_frame,
                    text="No SDGs found in documents",
                    font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"],
                    fg=self.theme["text_muted"],
                ).pack(expand=True, pady=50)
                return
            
            # Sort by SDG number
            sorted_sdgs = sorted(sdg_counts.items())
            labels = [sdg for sdg, _ in sorted_sdgs]
            values = [count for _, count in sorted_sdgs]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Color by SDG category
            colors = []
            for sdg in labels:
                num = int(sdg.replace('SDG', ''))
                if num <= 3:
                    colors.append('#e74c3c')  # Life/poverty
                elif num <= 5:
                    colors.append('#9b59b6')  # Equality
                elif num <= 7:
                    colors.append('#3498db')  # Resources
                elif num <= 9:
                    colors.append('#f39c12')  # Economic
                elif num <= 12:
                    colors.append('#1abc9c')  # Social/resources
                else:
                    colors.append('#27ae60')  # Environment
            
            bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=0.5)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:,}', va='center', fontsize=9)
            
            ax.set_xlabel('Number of Documents', fontsize=11)
            ax.set_title('SDG Distribution in Dataset', fontsize=14, fontweight='bold', pad=15)
            ax.invert_yaxis()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            fig.tight_layout()
            
            # Convert to image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            self._last_image_bytes = buf.getvalue()
            buf.close()
            plt.close(fig)
            
            # Display
            from PIL import Image
            pil_image = Image.open(io.BytesIO(self._last_image_bytes))
            
            scaled_frame = ScaledImageFrame(self.dist_frame, theme=self.theme_name)
            scaled_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            scaled_frame.set_image(pil_image)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(
                self.dist_frame,
                text=f"Could not create chart: {e}",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True, pady=50)
    
    def _create_documents_table(self, result_df: pd.DataFrame, sdg_cols: list):
        """Create documents table showing SDG assignments."""
        # Filter to documents with at least one SDG
        if 'any_SDG' in result_df.columns:
            sdg_docs = result_df[result_df['any_SDG'] == 1].copy()
        else:
            sdg_docs = result_df.copy()
        
        if len(sdg_docs) == 0:
            tk.Label(
                self.docs_frame,
                text="No documents matched any SDG",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True, pady=50)
            return
        
        # Create SDGs column showing which SDGs each document matches
        def get_sdgs(row):
            matched = []
            for col in sdg_cols:
                if row.get(col, 0) == 1:
                    matched.append(col)
            return "; ".join(matched)
        
        sdg_docs['Matched SDGs'] = sdg_docs.apply(get_sdgs, axis=1)
        
        # Select display columns
        display_cols = ['Matched SDGs']
        for col in ['Title', 'Year', 'Authors', 'Source title']:
            if col in sdg_docs.columns:
                display_cols.append(col)
        
        # Reorder columns
        display_df = sdg_docs[display_cols].head(500)
        
        # Info label
        tk.Label(
            self.docs_frame,
            text=f"Showing {min(len(sdg_docs), 500)} of {len(sdg_docs)} documents with SDG matches",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W, padx=8, pady=(8, 4))
        
        # Table
        table = DataTable(self.docs_frame, theme=self.theme_name, max_rows=500)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        table.set_data(display_df)
    
    def _export_results(self):
        """Export SDG results to file."""
        if self._sdg_results is None:
            messagebox.showwarning("No Results", "Please run SDG identification first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Export SDG Results",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel Files", "*.xlsx"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*"),
            ],
        )
        
        if filepath:
            try:
                if filepath.endswith('.xlsx'):
                    self._sdg_results.to_excel(filepath, index=False)
                else:
                    self._sdg_results.to_csv(filepath, index=False)
                messagebox.showinfo("Exported", f"Results saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
SDG IDENTIFICATION
══════════════════

Map publications to UN Sustainable Development Goals.

THE 17 SDGs
───────────
SDG 1: No Poverty
SDG 2: Zero Hunger
SDG 3: Good Health and Well-being
SDG 4: Quality Education
SDG 5: Gender Equality
SDG 6: Clean Water and Sanitation
SDG 7: Affordable and Clean Energy
SDG 8: Decent Work and Economic Growth
SDG 9: Industry, Innovation, Infrastructure
SDG 10: Reduced Inequalities
SDG 11: Sustainable Cities and Communities
SDG 12: Responsible Consumption/Production
SDG 13: Climate Action
SDG 14: Life Below Water
SDG 15: Life on Land
SDG 16: Peace, Justice, Strong Institutions
SDG 17: Partnerships for the Goals

CLASSIFICATION METHODS
──────────────────────
• Keyword matching
• OpenAlex SDG tags
• Custom rules/patterns

OUTPUT
──────
• SDG per document
• Multi-SDG papers flagged
• SDG distribution chart
• SDG trends over time
• SDG co-occurrence

APPLICATIONS
────────────
• Assess sustainability focus
• Track SDG research growth
• Identify research gaps
• Policy alignment analysis
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
        """Handle data loaded event."""
        super().on_data_loaded(bib)
        
        # Update text column options based on available columns
        text_cols = []
        for col in ["Abstract", "Title", "Combined Text"]:
            candidates = {
                "Abstract": ["Abstract", "AB", "abstract"],
                "Title": ["Title", "TI", "title", "Document Title"],
                "Combined Text": ["Combined Text", "combined_text", "Text"],
            }
            for candidate in candidates.get(col, []):
                if candidate in bib.df.columns:
                    text_cols.append(col)
                    break
        
        if text_cols:
            self.text_col_combo.combo.configure(values=text_cols)
            if "Abstract" in text_cols:
                self.text_col_combo.set("Abstract")
            elif text_cols:
                self.text_col_combo.set(text_cols[0])
