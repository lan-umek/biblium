# -*- coding: utf-8 -*-
"""
K-Fields Panel (Three-Field Plot)
=================================
Sankey diagram showing relationships between multiple bibliometric fields.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import tempfile
import os
from typing import Dict, List, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledSpinbox, LabeledCheckbox

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Available fields for selection with column mappings
# Format: (Display Name, key, [possible column names])
AVAILABLE_FIELDS_CONFIG = [
    ("Authors", "authors", ["Authors", "Author full names", "AU", "AF", "Author Full Names"]),
    ("Author Keywords", "author keywords", ["Author Keywords", "DE", "Keywords"]),
    ("Index Keywords", "index keywords", ["Index Keywords", "ID", "Keywords Plus", "Keyword Plus"]),
    ("Sources", "sources", ["Source title", "Source", "SO", "Journal", "Publication Name"]),
    ("Countries", "all countries", ["Countries", "Country", "Countries of Authors", "Affiliations"]),
    ("Affiliations", "affiliations", ["Affiliations", "C1", "Addresses", "Author Affiliations"]),
    ("References", "references", ["References", "Cited References", "CR"]),
    ("Cited Sources", "cited sources", ["Cited Sources", "Cited Journals"]),
    ("Cited Authors", "cited authors", ["Cited Authors", "First Authors of Cited References"]),
    ("Document Types", "document types", ["Document Type", "DT", "Publication Type", "Type"]),
    ("Research Areas", "research areas", ["Research Areas", "WoS Categories", "SC", "Subject Area"]),
]

def get_available_fields_for_df(df):
    """Return list of available fields based on columns in dataframe."""
    if df is None:
        return [(name, key) for name, key, _ in AVAILABLE_FIELDS_CONFIG]
    
    available = []
    columns_lower = {c.lower(): c for c in df.columns}
    
    for display_name, key, possible_cols in AVAILABLE_FIELDS_CONFIG:
        found = False
        for col in possible_cols:
            if col in df.columns or col.lower() in columns_lower:
                found = True
                break
        if found:
            available.append((display_name, key))
    
    return available if available else [(name, key) for name, key, _ in AVAILABLE_FIELDS_CONFIG]


class KFieldsPanel(BasePanel):
    """Panel for K-Fields Sankey diagram."""
    
    title = "K-Fields Plot"
    icon = "🔀"
    description = "Visualize relationships between K bibliometric fields using Sankey diagram"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._result_fig = None
        self._image_label = None
        self._temp_png = None
        self._field_combos = []
        self._field_spins = []
        self._fields_frame = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # K Selection Card
        k_card = Card(self.options_content, title="🔢 Number of Fields (K)", theme=self.theme_name)
        k_card.pack(fill=tk.X, padx=8, pady=8)
        
        k_frame = tk.Frame(k_card.content, bg=self.theme["bg_card"])
        k_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            k_frame, text="K:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.k_var = tk.IntVar(value=3)
        self.k_spin = tk.Spinbox(
            k_frame, from_=2, to=6, width=5,
            textvariable=self.k_var,
            font=FONTS.get_font("body"),
            command=self._on_k_changed,
        )
        self.k_spin.pack(side=tk.LEFT, padx=(8, 0))
        self.k_spin.bind("<Return>", lambda e: self._on_k_changed())
        self.k_spin.bind("<FocusOut>", lambda e: self._on_k_changed())
        
        tk.Label(
            k_frame, text="(2-6 fields)",
            font=FONTS.get_font("caption"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
        ).pack(side=tk.LEFT, padx=(8, 0))
        
        # Field Selection Card
        self.fields_card = Card(self.options_content, title="📊 Field Selection", theme=self.theme_name)
        self.fields_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Container for dynamic field selectors
        self._fields_frame = tk.Frame(self.fields_card.content, bg=self.theme["bg_card"])
        self._fields_frame.pack(fill=tk.X)
        
        # Initialize field selectors
        self._create_field_selectors()
        
        # Color Options Card
        color_card = Card(self.options_content, title="🎨 Coloring", theme=self.theme_name)
        color_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.color_option = LabeledCombobox(
            color_card.content, label="Color By:",
            values=["Average year", "Citations per document", "None"],
            default="Average year",
            theme=self.theme_name, label_width=12,
        )
        self.color_option.pack(fill=tk.X, pady=4)
        
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
        self.export_png_btn = ThemedButton(
            btn_frame, text="Export PNG",
            command=self._export_png,
            style="secondary",
            icon="🖼️",
            theme=self.theme_name,
        )
        self.export_png_btn.pack(fill=tk.X, pady=(8, 0))
        
        self.export_html_btn = ThemedButton(
            btn_frame, text="Export HTML",
            command=self._export_html,
            style="secondary",
            icon="🌐",
            theme=self.theme_name,
        )
        self.export_html_btn.pack(fill=tk.X, pady=(8, 0))
    
    def _on_k_changed(self):
        """Handle K value change."""
        try:
            k = self.k_var.get()
            if k < 2:
                k = 2
                self.k_var.set(2)
            elif k > 6:
                k = 6
                self.k_var.set(6)
            self._create_field_selectors()
        except tk.TclError:
            pass
    
    def _create_field_selectors(self):
        """Create field selectors based on current K value."""
        # Clear existing selectors
        for widget in self._fields_frame.winfo_children():
            widget.destroy()
        
        self._field_combos = []
        self._field_spins = []
        
        k = self.k_var.get()
        available_fields = self._get_available_fields()
        field_values = [f[0] for f in available_fields]
        
        # Default field selections - filter to only available fields
        all_defaults = ["Authors", "Author Keywords", "Sources", "Countries", "Affiliations", "References"]
        defaults = [d for d in all_defaults if d in field_values]
        # Pad with remaining available fields if needed
        for fv in field_values:
            if fv not in defaults:
                defaults.append(fv)
        
        for i in range(k):
            # Field row frame
            row_frame = tk.Frame(self._fields_frame, bg=self.theme["bg_card"])
            row_frame.pack(fill=tk.X, pady=4)
            
            # Field label
            tk.Label(
                row_frame, text=f"Field {i+1}:",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                width=8, anchor="w",
            ).pack(side=tk.LEFT)
            
            # Field combobox
            default_val = defaults[i] if i < len(defaults) else (field_values[0] if field_values else "Authors")
            combo = ttk.Combobox(
                row_frame, values=field_values,
                state="readonly", width=18,
                font=FONTS.get_font("body"),
            )
            combo.set(default_val if default_val in field_values else (field_values[0] if field_values else ""))
            combo.pack(side=tk.LEFT, padx=(4, 8))
            self._field_combos.append(combo)
            
            # Top N label
            tk.Label(
                row_frame, text="Top:",
                font=FONTS.get_font("caption"),
                bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            ).pack(side=tk.LEFT)
            
            # Top N spinbox
            spin = tk.Spinbox(
                row_frame, from_=5, to=50, width=4,
                font=FONTS.get_font("body"),
            )
            spin.delete(0, tk.END)
            spin.insert(0, "10")
            spin.pack(side=tk.LEFT, padx=(4, 0))
            self._field_spins.append(spin)
    
    def _create_results(self):
        """Create the results panel."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Plot tab
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="📊 Sankey Plot")
        
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
            text="Select K fields and click 'Generate Plot'\nto create a K-Fields Sankey diagram",
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
K-FIELDS PLOT (SANKEY DIAGRAM)
══════════════════════════════

A K-Fields Plot visualizes the relationships between K different 
bibliometric entities using a Sankey diagram.

HOW IT WORKS
────────────
• Set K (2-6) to choose the number of fields
• Each column represents a field (e.g., Authors, Keywords, Sources)
• Nodes represent individual items within each field
• Links show co-occurrence relationships between items
• Link thickness indicates the strength of the relationship
• Node colors can represent average publication year or citations

COMMON COMBINATIONS
───────────────────
K=2: Authors → Keywords
     Simple author-topic analysis

K=3: Authors → Keywords → Sources
     Shows which authors work on which topics and publish in which journals

K=4: Countries → Authors → Keywords → Sources  
     Full geographic to publication venue flow

K=5: Affiliations → Authors → Keywords → Sources → Cited Sources
     Complete knowledge flow analysis

INTERPRETATION
──────────────
• Thick links indicate strong co-occurrence
• Central nodes with many connections are key bridges
• Color gradients show temporal or citation patterns
• Isolated nodes indicate niche or specialized items

SETTINGS
────────
• K: Number of fields to analyze (2-6)
• Top N: Number of items to show per field (5-50)
• Color By: Average year (temporal) or Citations per document
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
    
    def _get_available_fields(self):
        """Get list of available fields based on current data."""
        df = self.bib.df if self.bib is not None else None
        return get_available_fields_for_df(df)
    
    def _get_field_key(self, display_name: str) -> str:
        """Convert display name to field key."""
        for name, key, _ in AVAILABLE_FIELDS_CONFIG:
            if name == display_name:
                return key
        return display_name.lower()
    
    def _run_analysis(self):
        """Run the K-Fields analysis."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        # Validate unique fields
        selected_fields = [combo.get() for combo in self._field_combos]
        if len(selected_fields) != len(set(selected_fields)):
            messagebox.showwarning("Duplicate Fields", "Please select unique fields for each position.")
            return
        
        self.run_btn.config(state=tk.DISABLED, text="⏳ Generating...")
        
        # Capture values before thread
        fields = [self._get_field_key(f) for f in selected_fields]
        top_n = []
        for spin in self._field_spins:
            try:
                top_n.append(int(spin.get()))
            except ValueError:
                top_n.append(10)
        
        color_opt = self.color_option.get()
        if color_opt == "None":
            color_opt = None
        
        def do_analysis():
            error_info = None
            try:
                # Build binary matrices using BiblioAnalysis methods
                field_dfs = self._build_field_matrices(fields)
                
                # Import k_fields_plot
                from biblium.plotbib import k_fields_plot
                
                # Create temp file for PNG
                self._temp_png = tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ).name
                
                # Prepare df_main with numeric columns for coloring
                df_main = self.bib.df.copy()
                
                # Ensure Year is numeric
                if "Year" in df_main.columns:
                    df_main["Year"] = pd.to_numeric(df_main["Year"], errors="coerce")
                elif "PY" in df_main.columns:
                    df_main["Year"] = pd.to_numeric(df_main["PY"], errors="coerce")
                
                # Ensure Cited by is numeric
                for cite_col in ["Cited by", "Cited By", "Times Cited", "TC"]:
                    if cite_col in df_main.columns:
                        df_main["Cited by"] = pd.to_numeric(df_main[cite_col], errors="coerce")
                        break
                
                # Generate plot
                fig = k_fields_plot(
                    field_dfs=field_dfs,
                    df_main=df_main,
                    fields=fields,
                    top_n=top_n,
                    color_option=color_opt,
                    save_png=None,  # Don't save yet
                    save_html=None,
                )
                
                # Calculate dynamic height based on max nodes
                max_nodes = max(top_n)
                fig_height = max(600, 80 * max_nodes)
                fig_width = max(900, 200 * len(fields))
                
                # Update layout with dynamic size
                fig.update_layout(
                    width=fig_width,
                    height=fig_height,
                )
                
                # Now save the image
                fig.write_image(self._temp_png, width=fig_width, height=fig_height)
                
                self._result_fig = fig
                self.after(0, self._on_analysis_success)
                
            except Exception as exc:
                import traceback
                error_info = (str(exc), traceback.format_exc())
                self.after(0, lambda ei=error_info: self._on_analysis_error(ei[0], ei[1]))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _build_field_matrices(self, fields: List[str]) -> Dict[str, pd.DataFrame]:
        """Build binary indicator matrices for each field."""
        field_dfs = {}
        
        # Build column mapping from config
        col_mapping = {key: cols for _, key, cols in AVAILABLE_FIELDS_CONFIG}
        
        separator = getattr(self.bib, "default_separator", "; ")
        
        for field in fields:
            # Try to find the column
            possible_cols = col_mapping.get(field, [field])
            actual_col = None
            
            for col in possible_cols:
                if col in self.bib.df.columns:
                    actual_col = col
                    break
            
            if actual_col is None:
                # Try case-insensitive search
                for col in self.bib.df.columns:
                    if col.lower() == field.lower() or col.lower().replace(" ", "_") == field.lower().replace(" ", "_"):
                        actual_col = col
                        break
            
            if actual_col is None:
                raise ValueError(f"Could not find column for field '{field}'")
            
            # Special handling for countries from affiliations
            if field == "all countries" and actual_col == "Affiliations":
                binary_df = self._extract_countries_binary(actual_col, separator)
            else:
                binary_df = self._build_binary_matrix(actual_col, separator)
            
            field_dfs[field] = binary_df
        
        return field_dfs
    
    def _build_binary_matrix(self, col: str, separator: str) -> pd.DataFrame:
        """Build binary indicator matrix from a multi-valued column."""
        df = self.bib.df
        
        # Get all unique items
        all_items = set()
        for idx, val in df[col].dropna().items():
            if isinstance(val, str):
                items = [x.strip() for x in val.split(separator) if x.strip()]
                all_items.update(items)
        
        all_items = sorted(all_items)
        
        # Build binary matrix
        binary_data = []
        for idx, val in df[col].items():
            row = {item: 0 for item in all_items}
            if pd.notna(val) and isinstance(val, str):
                items = [x.strip() for x in val.split(separator) if x.strip()]
                for item in items:
                    if item in row:
                        row[item] = 1
            binary_data.append(row)
        
        binary_df = pd.DataFrame(binary_data, index=df.index)
        return binary_df
    
    def _extract_countries_binary(self, col: str, separator: str) -> pd.DataFrame:
        """Extract countries from affiliations and build binary matrix."""
        df = self.bib.df
        
        # Check if Countries column exists
        if "Countries" in df.columns:
            return self._build_binary_matrix("Countries", separator)
        
        # Try to extract from affiliations
        all_countries = set()
        
        for idx, val in df[col].dropna().items():
            if isinstance(val, str):
                affs = [x.strip() for x in val.split(separator) if x.strip()]
                for aff in affs:
                    parts = aff.split(",")
                    if parts:
                        country = parts[-1].strip()
                        if len(country) > 1 and not country.isdigit():
                            all_countries.add(country)
        
        all_countries = sorted(all_countries)
        
        # Build binary matrix
        binary_data = []
        for idx, val in df[col].items():
            row = {c: 0 for c in all_countries}
            if pd.notna(val) and isinstance(val, str):
                affs = [x.strip() for x in val.split(separator) if x.strip()]
                for aff in affs:
                    parts = aff.split(",")
                    if parts:
                        country = parts[-1].strip()
                        if country in row:
                            row[country] = 1
            binary_data.append(row)
        
        binary_df = pd.DataFrame(binary_data, index=df.index)
        return binary_df
    
    def _on_analysis_success(self):
        """Handle successful analysis."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Generate Plot")
        self._display_plot()
        self.notebook.select(0)
    
    def _on_analysis_error(self, error_msg: str, traceback_str: str):
        """Handle analysis error."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Generate Plot")
        messagebox.showerror("Analysis Error", f"Error: {error_msg}\n\n{traceback_str[:500]}")
    
    def _display_plot(self):
        """Display the generated plot image."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        if not self._temp_png or not os.path.exists(self._temp_png):
            tk.Label(
                self.plot_frame,
                text="No plot available",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["danger"],
            ).pack(expand=True)
            return
        
        if not HAS_PIL:
            tk.Label(
                self.plot_frame,
                text="PIL not available for image display.\nUse Export PNG to save the plot.",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
            ).pack(expand=True)
            return
        
        try:
            from biblium.gui.widgets.plots import ScaledImageFrame
            
            scaled_frame = ScaledImageFrame(
                self.plot_frame, 
                theme=self.theme_name,
                maintain_aspect=True,
                max_scale=1.0
            )
            scaled_frame.pack(fill=tk.BOTH, expand=True)
            scaled_frame.set_image_from_file(self._temp_png)
            self._scaled_frame = scaled_frame
            
        except Exception as e:
            tk.Label(
                self.plot_frame,
                text=f"Error displaying plot: {e}",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["danger"],
            ).pack(expand=True)
    
    def _export_png(self):
        """Export plot as PNG."""
        if self._result_fig is None:
            messagebox.showwarning("No Plot", "Please generate a plot first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            title="Export Sankey Plot as PNG",
        )
        
        if not filepath:
            return
        
        try:
            self._result_fig.write_image(filepath)
            messagebox.showinfo("Export Complete", f"Plot exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def _export_html(self):
        """Export plot as interactive HTML."""
        if self._result_fig is None:
            messagebox.showwarning("No Plot", "Please generate a plot first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
            title="Export Sankey Plot as HTML",
        )
        
        if not filepath:
            return
        
        try:
            self._result_fig.write_html(filepath)
            messagebox.showinfo("Export Complete", f"Interactive plot exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def on_data_loaded(self, bib):
        """Handle data loaded event."""
        self.bib = bib
        self._result_fig = None
        self._show_placeholder()
        # Refresh field selectors with available fields
        self._create_field_selectors()
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
K-FIELDS PLOT
═════════════

Keyword dynamics across time periods.

WHAT IT SHOWS
─────────────
• Keyword presence over time
• Emerging keywords (new)
• Growing keywords (increasing)
• Declining keywords (decreasing)
• Disappeared (no longer used)

VISUALIZATION
─────────────
• X-axis: Time periods
• Y-axis: Keywords
• Size: Frequency
• Color: Trend direction

KEYWORD CATEGORIES
──────────────────
• 🆕 Emerging: First appearance
• 📈 Growing: Frequency increasing
• ➡️ Stable: Consistent presence
• 📉 Declining: Decreasing use
• ❌ Disappeared: No longer present

APPLICATIONS
────────────
• Track research evolution
• Identify hot topics
• Find declining areas
• Vocabulary shifts
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
        """Clean up temp files."""
        if self._temp_png and os.path.exists(self._temp_png):
            try:
                os.unlink(self._temp_png)
            except:
                pass
        super().destroy()
