# -*- coding: utf-8 -*-
"""
Relationship Analysis Panel
===========================
Analyze relationships between TWO DIFFERENT bibliometric entity types.
Examples: Authors vs Keywords, Countries vs Sources, Keywords vs Years, etc.
Includes comprehensive item filtering (include/exclude lists, regex, file loading).
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, Optional, List, Tuple, Set
import io
import os
import re

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledSpinbox, LabeledCheckbox, LabeledEntry

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageTk
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class EntityFilterFrame(tk.Frame):
    """Reusable filtering widget for an entity - matches the design in the screenshot."""
    
    def __init__(self, parent, theme_dict, theme_name="light", **kwargs):
        self.theme = theme_dict
        self.theme_name = theme_name
        super().__init__(parent, bg=theme_dict["bg_card"], **kwargs)
        
        # Include only (one per line)
        inc_header = tk.Frame(self, bg=self.theme["bg_card"])
        inc_header.pack(fill=tk.X, pady=(0, 2))
        
        tk.Label(inc_header, text="Include only (one per line):", 
                 font=FONTS.get_font("small"), bg=self.theme["bg_card"], 
                 fg=self.theme["text_secondary"], anchor='w').pack(side=tk.LEFT)
        
        self.inc_load_btn = tk.Button(inc_header, text="📁 Load", font=FONTS.get_font("small"),
                                       relief=tk.FLAT, cursor="hand2",
                                       command=lambda: self._load_file("include"))
        self.inc_load_btn.pack(side=tk.RIGHT)
        
        self.include_text = tk.Text(self, height=4, width=30, font=FONTS.get_font("small"),
                                     relief=tk.SOLID, borderwidth=1)
        self.include_text.pack(fill=tk.X, pady=(0, 8))
        
        # Exclude (one per line)
        exc_header = tk.Frame(self, bg=self.theme["bg_card"])
        exc_header.pack(fill=tk.X, pady=(0, 2))
        
        tk.Label(exc_header, text="Exclude (one per line):", 
                 font=FONTS.get_font("small"), bg=self.theme["bg_card"], 
                 fg=self.theme["text_secondary"], anchor='w').pack(side=tk.LEFT)
        
        self.exc_load_btn = tk.Button(exc_header, text="📁 Load", font=FONTS.get_font("small"),
                                       relief=tk.FLAT, cursor="hand2",
                                       command=lambda: self._load_file("exclude"))
        self.exc_load_btn.pack(side=tk.RIGHT)
        
        self.exclude_text = tk.Text(self, height=4, width=30, font=FONTS.get_font("small"),
                                     relief=tk.SOLID, borderwidth=1)
        self.exclude_text.pack(fill=tk.X, pady=(0, 8))
        
        # Regex include
        regex_inc_frame = tk.Frame(self, bg=self.theme["bg_card"])
        regex_inc_frame.pack(fill=tk.X, pady=(0, 4))
        
        tk.Label(regex_inc_frame, text="Regex include:", 
                 font=FONTS.get_font("small"), bg=self.theme["bg_card"], 
                 fg=self.theme["text_secondary"], width=12, anchor='w').pack(side=tk.LEFT)
        
        self.regex_include = tk.Entry(regex_inc_frame, font=FONTS.get_font("small"),
                                       relief=tk.SOLID, borderwidth=1)
        self.regex_include.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.regex_include.insert(0, "")
        # Add placeholder
        self._add_placeholder(self.regex_include, "e.g., ^Smith|^Jones (regex pattern)")
        
        # Regex exclude
        regex_exc_frame = tk.Frame(self, bg=self.theme["bg_card"])
        regex_exc_frame.pack(fill=tk.X, pady=(0, 4))
        
        tk.Label(regex_exc_frame, text="Regex exclude:", 
                 font=FONTS.get_font("small"), bg=self.theme["bg_card"], 
                 fg=self.theme["text_secondary"], width=12, anchor='w').pack(side=tk.LEFT)
        
        self.regex_exclude = tk.Entry(regex_exc_frame, font=FONTS.get_font("small"),
                                       relief=tk.SOLID, borderwidth=1)
        self.regex_exclude.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.regex_exclude.insert(0, "")
        self._add_placeholder(self.regex_exclude, "e.g., Unknown|Anonymous")
        
        # Max items
        max_frame = tk.Frame(self, bg=self.theme["bg_card"])
        max_frame.pack(fill=tk.X, pady=(0, 4))
        
        tk.Label(max_frame, text="Max items:", 
                 font=FONTS.get_font("small"), bg=self.theme["bg_card"], 
                 fg=self.theme["text_secondary"], width=12, anchor='w').pack(side=tk.LEFT)
        
        self.max_items = tk.Spinbox(max_frame, from_=0, to=500, width=10, 
                                     font=FONTS.get_font("small"),
                                     relief=tk.SOLID, borderwidth=1)
        self.max_items.delete(0, tk.END)
        self.max_items.insert(0, "0")
        self.max_items.pack(side=tk.LEFT)
        
        # Skip header checkbox
        self.skip_header_var = tk.BooleanVar(value=True)
        self.skip_header_cb = tk.Checkbutton(self, text="Skip header row when loading files",
                                              variable=self.skip_header_var, 
                                              font=FONTS.get_font("small"),
                                              bg=self.theme["bg_card"], 
                                              fg=self.theme["text_secondary"],
                                              activebackground=self.theme["bg_card"],
                                              selectcolor=self.theme["bg_card"])
        self.skip_header_cb.pack(anchor='w', pady=(4, 4))
        
        # Note
        note_text = "Note: 'Include only' overrides Top N. Max items=0 means no limit.\nLoad from .txt, .csv, or .xlsx files."
        tk.Label(self, text=note_text, font=FONTS.get_font("small"), 
                 bg=self.theme["bg_card"], fg=self.theme["text_muted"], 
                 justify='left', anchor='w').pack(anchor='w')
    
    def _add_placeholder(self, entry, placeholder_text):
        """Add placeholder text to an entry widget."""
        entry.placeholder = placeholder_text
        entry.showing_placeholder = True
        entry.config(fg='gray')
        entry.delete(0, tk.END)
        entry.insert(0, placeholder_text)
        
        def on_focus_in(event):
            if entry.showing_placeholder:
                entry.delete(0, tk.END)
                entry.config(fg='black')
                entry.showing_placeholder = False
        
        def on_focus_out(event):
            if not entry.get():
                entry.showing_placeholder = True
                entry.config(fg='gray')
                entry.insert(0, entry.placeholder)
        
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)
    
    def _load_file(self, target):
        """Load items from file into include or exclude text area."""
        filepath = filedialog.askopenfilename(
            title=f"Select file for {target}",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), 
                      ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        try:
            items = []
            skip_header = self.skip_header_var.get()
            
            if filepath.endswith('.xlsx'):
                try:
                    df = pd.read_excel(filepath, header=0 if skip_header else None)
                    items = df.iloc[:, 0].dropna().astype(str).tolist()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to read Excel file: {e}")
                    return
            elif filepath.endswith('.csv'):
                try:
                    df = pd.read_csv(filepath, header=0 if skip_header else None)
                    items = df.iloc[:, 0].dropna().astype(str).tolist()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to read CSV file: {e}")
                    return
            else:
                # Text file
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    start_idx = 1 if skip_header and lines else 0
                    for line in lines[start_idx:]:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            items.append(line)
            
            # Insert into appropriate text area
            text_widget = self.include_text if target == "include" else self.exclude_text
            text_widget.delete("1.0", tk.END)
            text_widget.insert("1.0", "\n".join(items))
            
            messagebox.showinfo("Loaded", f"Loaded {len(items)} items from file.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def _get_entry_value(self, entry):
        """Get entry value, ignoring placeholder text."""
        if hasattr(entry, 'showing_placeholder') and entry.showing_placeholder:
            return ""
        return entry.get().strip()
    
    def get_filter_config(self) -> dict:
        """Get the current filter configuration."""
        include_items = set()
        exclude_items = set()
        
        # Parse include text
        inc_text = self.include_text.get("1.0", tk.END).strip()
        if inc_text:
            for line in inc_text.split('\n'):
                line = line.strip()
                if line:
                    include_items.add(line)
        
        # Parse exclude text
        exc_text = self.exclude_text.get("1.0", tk.END).strip()
        if exc_text:
            for line in exc_text.split('\n'):
                line = line.strip()
                if line:
                    exclude_items.add(line)
        
        # Get regex patterns (check for placeholder)
        regex_inc = self._get_entry_value(self.regex_include)
        regex_exc = self._get_entry_value(self.regex_exclude)
        
        # Get max items
        try:
            max_items = int(self.max_items.get())
        except ValueError:
            max_items = 0
        
        return {
            "include_items": include_items,
            "exclude_items": exclude_items,
            "regex_include": regex_inc,
            "regex_exclude": regex_exc,
            "max_items": max_items,
        }


class RelationshipPanel(BasePanel):
    """Panel for analyzing relationships between two different entity types."""
    
    title = "Relationship Analysis"
    icon = "🔗"
    description = "Analyze relationships between two entity types"
    requires_data = True
    
    # Entity configurations: (column_candidates, value_type, separator)
    ENTITY_CONFIG = {
        "Author Keywords": (["Processed Author Keywords", "Author Keywords", "DE"], "list", "; "),
        "Index Keywords": (["Processed Index Keywords", "Index Keywords", "ID"], "list", "; "),
        "Authors": (["Authors", "Authors or Inventors", "AU"], "list", "; "),
        "Sources": (["Source title", "Source Title", "Journal", "SO"], "string", None),
        "Countries": (["Countries", "Country"], "list", "; "),
        "Affiliations": (["Affiliations", "Addresses", "C1"], "list", "; "),
        "Year": (["Year", "Publication Year", "PY"], "string", None),
        "Document Type": (["Document Type", "DT"], "string", None),
        "Language": (["Language", "Language of Original Document", "LA"], "string", None),
        "Subject Area": (["Subject Area", "WC", "SC"], "list", "; "),
    }
    
    VIS_TYPES = ["Contingency Heatmap", "Cluster Heatmap", "Top Pairs (Bubble Chart)", 
                 "Correspondence Analysis", "Sankey Diagram"]
    
    ASSOC_MEASURES = ["Raw Counts", "Chi-square Residuals", "Normalized (Row %)", 
                     "Normalized (Column %)", "Log Odds Ratio"]
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._current_fig = None
        self._photo_image = None
        self._contingency_matrix = None
        self._last_image_bytes = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Entity 1 Selection (Rows)
        entity1_card = Card(self.options_content, title="📊 Row Entity", theme=self.theme_name)
        entity1_card.pack(fill=tk.X, padx=8, pady=4)
        
        self.entity1_combo = LabeledCombobox(
            entity1_card.content, label="Entity:", values=list(self.ENTITY_CONFIG.keys()),
            default="Authors", theme=self.theme_name, label_width=8)
        self.entity1_combo.pack(fill=tk.X, pady=2)
        
        self.top_n1_spin = LabeledSpinbox(
            entity1_card.content, label="Top N:", from_=5, to=200, default=10,
            theme=self.theme_name, label_width=8)
        self.top_n1_spin.pack(fill=tk.X, pady=2)
        
        # Filtering for Entity 1
        filter1_card = Card(self.options_content, title="🔍 Row Entity Filtering", theme=self.theme_name)
        filter1_card.pack(fill=tk.X, padx=8, pady=4)
        
        self.filter1 = EntityFilterFrame(filter1_card.content, self.theme, self.theme_name)
        self.filter1.pack(fill=tk.X)
        
        # Entity 2 Selection (Columns)
        entity2_card = Card(self.options_content, title="📋 Column Entity", theme=self.theme_name)
        entity2_card.pack(fill=tk.X, padx=8, pady=4)
        
        self.entity2_combo = LabeledCombobox(
            entity2_card.content, label="Entity:", values=list(self.ENTITY_CONFIG.keys()),
            default="Author Keywords", theme=self.theme_name, label_width=8)
        self.entity2_combo.pack(fill=tk.X, pady=2)
        
        self.top_n2_spin = LabeledSpinbox(
            entity2_card.content, label="Top N:", from_=5, to=200, default=10,
            theme=self.theme_name, label_width=8)
        self.top_n2_spin.pack(fill=tk.X, pady=2)
        
        # Filtering for Entity 2
        filter2_card = Card(self.options_content, title="🔍 Column Entity Filtering", theme=self.theme_name)
        filter2_card.pack(fill=tk.X, padx=8, pady=4)
        
        self.filter2 = EntityFilterFrame(filter2_card.content, self.theme, self.theme_name)
        self.filter2.pack(fill=tk.X)
        
        # Visualization Settings
        vis_card = Card(self.options_content, title="📈 Visualization", theme=self.theme_name)
        vis_card.pack(fill=tk.X, padx=8, pady=4)
        
        self.vis_combo = LabeledCombobox(
            vis_card.content, label="Plot:", values=self.VIS_TYPES,
            default="Contingency Heatmap", theme=self.theme_name, label_width=8)
        self.vis_combo.pack(fill=tk.X, pady=2)
        self.vis_combo.bind("<<ComboboxSelected>>", self._on_vis_change)
        
        self.measure_combo = LabeledCombobox(
            vis_card.content, label="Measure:", values=self.ASSOC_MEASURES,
            default="Raw Counts", theme=self.theme_name, label_width=8)
        self.measure_combo.pack(fill=tk.X, pady=2)
        
        # Top Pairs Settings (shown conditionally)
        self.pairs_frame = tk.Frame(vis_card.content, bg=self.theme["bg_card"])
        
        self.pairs_top_n = LabeledSpinbox(
            self.pairs_frame, label="Top pairs:", from_=10, to=100, default=30,
            theme=self.theme_name, label_width=8)
        self.pairs_top_n.pack(fill=tk.X, pady=2)
        
        self.pairs_sign = LabeledCombobox(
            self.pairs_frame, label="Show:", values=["Both (+/-)", "Positive only", "Negative only"],
            default="Both (+/-)", theme=self.theme_name, label_width=8)
        self.pairs_sign.pack(fill=tk.X, pady=2)
        
        # Appearance Settings
        appear_card = Card(self.options_content, title="🎨 Appearance", theme=self.theme_name)
        appear_card.pack(fill=tk.X, padx=8, pady=4)
        
        self.colormap_combo = LabeledCombobox(
            appear_card.content, label="Colormap:",
            values=["viridis", "plasma", "coolwarm", "RdBu_r", "YlGnBu", "Blues", "Reds", "Spectral", "YlOrRd"],
            default="YlGnBu", theme=self.theme_name, label_width=8)
        self.colormap_combo.pack(fill=tk.X, pady=2)
        
        self.show_values_cb = LabeledCheckbox(
            appear_card.content, label="Show values in cells", default=True, theme=self.theme_name)
        self.show_values_cb.pack(fill=tk.X, pady=2)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        
        ActionButton(btn_frame, text="Run Analysis", icon="🔗",
            command=self._run_analysis, theme=self.theme_name).pack(fill=tk.X)
        
        ThemedButton(btn_frame, text="Export Results", style="secondary",
            command=self._export_results, theme=self.theme_name).pack(fill=tk.X, pady=(4, 0))
    
    def _on_vis_change(self, event=None):
        """Update UI based on visualization type."""
        vis_type = self.vis_combo.get()
        if vis_type == "Top Pairs (Bubble Chart)":
            self.pairs_frame.pack(fill=tk.X, pady=2)
            self.colormap_combo.set("coolwarm")
            self.measure_combo.set("Chi-square Residuals")
        else:
            self.pairs_frame.pack_forget()
            if vis_type in ["Contingency Heatmap", "Cluster Heatmap"]:
                self.colormap_combo.set("YlGnBu")
    
    def _create_results(self):
        """Create results panel - optimized for larger plot display."""
        self.results_card = tk.Frame(
            self.results_frame, bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"], highlightthickness=1)
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.results_notebook = ttk.Notebook(self.results_card)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        self.plot_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.plot_frame, text="📊 Plot")
        
        self.image_label = tk.Label(self.plot_frame, bg=self.theme["bg_card"])
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        self.data_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.data_frame, text="📋 Table")
        
        # Info tab
        info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        
        self.tree_frame = tk.Frame(self.data_frame, bg=self.theme["bg_card"])
        self.tree_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Note: ScaledImageFrame handles its own resize events
    
    def _on_plot_resize(self, event=None):
        # Deprecated - ScaledImageFrame handles resize internally
        pass
    
    def _reset_resize_flag(self):
        pass
    
    def _show_placeholder(self, message: str):
        """Show a placeholder message in the plot frame."""
        # Clear all widgets in plot_frame
        for widget in self.plot_frame.winfo_children():
            try:
                widget.destroy()
            except tk.TclError:
                pass
        
        # Create a new label for the message
        self.image_label = tk.Label(
            self.plot_frame, 
            text=message, 
            font=FONTS.get_font("body"), 
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        )
        self.image_label.pack(expand=True)
    
    def _apply_filters(self, items: List[str], filter_config: dict, top_n: int) -> List[str]:
        """Apply all filters to a list of items."""
        result = items.copy()
        
        # Apply include filter (if specified, only keep matching items)
        if filter_config["include_items"]:
            result = [i for i in result if i in filter_config["include_items"]]
        
        # Apply exclude filter
        if filter_config["exclude_items"]:
            result = [i for i in result if i not in filter_config["exclude_items"]]
        
        # Apply regex include
        if filter_config["regex_include"]:
            try:
                pattern = re.compile(filter_config["regex_include"], re.IGNORECASE)
                result = [i for i in result if pattern.search(i)]
            except re.error:
                pass
        
        # Apply regex exclude
        if filter_config["regex_exclude"]:
            try:
                pattern = re.compile(filter_config["regex_exclude"], re.IGNORECASE)
                result = [i for i in result if not pattern.search(i)]
            except re.error:
                pass
        
        # Apply max items limit
        max_items = filter_config["max_items"]
        if max_items > 0:
            result = result[:max_items]
        elif not filter_config["include_items"]:
            # If no include filter and max_items=0, use top_n
            result = result[:top_n]
        
        return result
    
    def _run_analysis(self):
        """Run relationship analysis between two entity types."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        entity1 = self.entity1_combo.get()
        entity2 = self.entity2_combo.get()
        
        if entity1 == entity2:
            messagebox.showwarning("Same Entity", 
                "Please select two different entity types.\nFor co-occurrence within the same entity, use Co-occurrence Networks.")
            return
        
        event_bus.emit(EventBus.ANALYSIS_STARTED, {"name": self.title})
        self._is_running = True
        self._show_placeholder(f"⏳ Computing: {entity1} × {entity2}...")
        
        top_n1, top_n2 = self.top_n1_spin.get(), self.top_n2_spin.get()
        vis_type, measure = self.vis_combo.get(), self.measure_combo.get()
        colormap, show_values = self.colormap_combo.get(), self.show_values_cb.get()
        
        # Get filter configurations
        filter1_config = self.filter1.get_filter_config()
        filter2_config = self.filter2.get_filter_config()
        
        def do_analysis():
            try:
                config1, config2 = self.ENTITY_CONFIG.get(entity1), self.ENTITY_CONFIG.get(entity2)
                if not config1 or not config2:
                    raise ValueError("Unknown entity configuration")
                
                col1_candidates, value_type1, sep1 = config1
                col2_candidates, value_type2, sep2 = config2
                
                col1 = self._find_column(col1_candidates)
                col2 = self._find_column(col2_candidates)
                
                if not col1:
                    raise ValueError(f"Column not found for {entity1}")
                if not col2:
                    raise ValueError(f"Column not found for {entity2}")
                
                print(f"[DEBUG] Relationship: {entity1} ({col1}) × {entity2} ({col2})")
                
                contingency = self._build_contingency_matrix(
                    col1, col2, value_type1, value_type2, sep1, sep2, 
                    top_n1, top_n2, entity1, entity2,
                    filter1_config, filter2_config)
                
                if contingency is None or contingency.empty:
                    raise ValueError("No relationship data found")
                
                self._contingency_matrix = contingency
                display_matrix = self._compute_measure(contingency, measure)
                
                result = self._create_visualization(display_matrix, contingency, vis_type, 
                    entity1, entity2, colormap, show_values, measure)
                
                if result is not None:
                    # Check if result is bytes (from Sankey) or a matplotlib figure
                    if isinstance(result, bytes):
                        image_bytes = result
                        self._current_fig = None
                    else:
                        # It's a matplotlib figure
                        fig = result
                        self._current_fig = fig
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                        buf.seek(0)
                        image_bytes = buf.getvalue()
                        buf.close()
                        plt.close(fig)
                    self.after(0, lambda ib=image_bytes: self._show_results(ib, display_matrix))
                else:
                    self.after(0, lambda: self._on_error("Failed to create visualization"))
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._on_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _find_column(self, candidates):
        if not self.bib or not hasattr(self.bib, 'df'):
            return None
        for col in candidates:
            if col in self.bib.df.columns:
                return col
            for df_col in self.bib.df.columns:
                if df_col.lower() == col.lower():
                    return df_col
        return None
    
    def _build_contingency_matrix(self, col1, col2, vt1, vt2, sep1, sep2, 
                                   top_n1, top_n2, name1, name2,
                                   filter1_config, filter2_config):
        df = self.bib.df.copy()
        default_sep = getattr(self.bib, 'default_separator', '; ')
        sep1, sep2 = sep1 or default_sep, sep2 or default_sep
        
        def extract_items(series, vt, sep):
            if vt == "list":
                items = series.dropna().astype(str).str.split(sep).explode().str.strip()
            else:
                items = series.dropna().astype(str).str.strip()
            return items[items != '']
        
        items1, items2 = extract_items(df[col1], vt1, sep1), extract_items(df[col2], vt2, sep2)
        
        # Get candidate items (more than needed for filtering)
        has_filter1 = filter1_config["include_items"] or filter1_config["exclude_items"] or filter1_config["regex_include"] or filter1_config["regex_exclude"]
        has_filter2 = filter2_config["include_items"] or filter2_config["exclude_items"] or filter2_config["regex_include"] or filter2_config["regex_exclude"]
        
        multiplier1 = 5 if has_filter1 else 1
        multiplier2 = 5 if has_filter2 else 1
        
        candidates1 = items1.value_counts().head(top_n1 * multiplier1).index.tolist()
        candidates2 = items2.value_counts().head(top_n2 * multiplier2).index.tolist()
        
        # Apply filters
        top1 = self._apply_filters(candidates1, filter1_config, top_n1)
        top2 = self._apply_filters(candidates2, filter2_config, top_n2)
        
        if not top1 or not top2:
            return pd.DataFrame()
        
        print(f"[DEBUG] After filtering: {len(top1)} × {len(top2)} items")
        
        contingency = pd.DataFrame(0, index=top1, columns=top2)
        contingency.index.name, contingency.columns.name = name1, name2
        
        for _, row in df.iterrows():
            v1, v2 = row[col1], row[col2]
            if pd.isna(v1) or pd.isna(v2):
                continue
            
            di1 = [x.strip() for x in str(v1).split(sep1) if x.strip()] if vt1 == "list" else [str(v1).strip()]
            di2 = [x.strip() for x in str(v2).split(sep2) if x.strip()] if vt2 == "list" else [str(v2).strip()]
            
            for i1 in di1:
                if i1 in top1:
                    for i2 in di2:
                        if i2 in top2:
                            contingency.loc[i1, i2] += 1
        
        return contingency.loc[(contingency.sum(axis=1) > 0), (contingency.sum(axis=0) > 0)]
    
    def _compute_measure(self, cont, measure):
        if measure == "Raw Counts":
            return cont
        
        obs = cont.values.astype(float)
        row_s, col_s, total = obs.sum(axis=1, keepdims=True), obs.sum(axis=0, keepdims=True), obs.sum()
        
        if total == 0:
            return cont
        
        if measure == "Chi-square Residuals":
            exp = (row_s @ col_s) / total
            with np.errstate(divide='ignore', invalid='ignore'):
                res = (obs - exp) / np.sqrt(exp)
                res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            return pd.DataFrame(res, index=cont.index, columns=cont.columns)
        
        elif measure == "Normalized (Row %)":
            return cont.div(cont.sum(axis=1), axis=0).fillna(0) * 100
        
        elif measure == "Normalized (Column %)":
            return cont.div(cont.sum(axis=0), axis=1).fillna(0) * 100
        
        elif measure == "Log Odds Ratio":
            obs_s = obs + 0.5
            exp = (obs_s.sum(axis=1, keepdims=True) @ obs_s.sum(axis=0, keepdims=True)) / obs_s.sum()
            with np.errstate(divide='ignore', invalid='ignore'):
                lo = np.log(obs_s / exp)
                lo = np.nan_to_num(lo, nan=0.0, posinf=3.0, neginf=-3.0)
            return pd.DataFrame(lo, index=cont.index, columns=cont.columns)
        
        return cont
    
    def _create_visualization(self, disp, raw, vis, e1, e2, cmap, show_val, measure):
        plt.ioff()
        if vis == "Contingency Heatmap":
            return self._plot_heatmap(disp, e1, e2, cmap, show_val, measure)
        elif vis == "Cluster Heatmap":
            return self._plot_clustermap(disp, e1, e2, cmap, show_val, measure)
        elif vis == "Top Pairs (Bubble Chart)":
            return self._plot_top_pairs(disp, e1, e2, cmap, measure)
        elif vis == "Correspondence Analysis":
            return self._plot_correspondence(raw, e1, e2)
        elif vis == "Sankey Diagram":
            return self._plot_sankey(raw, e1, e2, cmap)
        return None
    
    def _plot_heatmap(self, mat, e1, e2, cmap, show_val, measure):
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(12, 10))
        
        fmt = ".0f" if measure == "Raw Counts" else ".1f" if "%" in measure else ".2f"
        center = 0 if measure in ["Chi-square Residuals", "Log Odds Ratio"] else None
        if center is not None:
            cmap = "coolwarm"
        
        sns.heatmap(mat, annot=show_val, fmt=fmt, cmap=cmap, center=center,
            cbar_kws={"label": measure}, ax=ax, annot_kws={"fontsize": 8})
        
        ax.set_title(f"{e1} × {e2}", fontsize=14, pad=10)
        ax.set_xlabel(e2, fontsize=11)
        ax.set_ylabel(e1, fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        fig.tight_layout()
        return fig
    
    def _plot_clustermap(self, mat, e1, e2, cmap, show_val, measure):
        import seaborn as sns
        if mat.shape[0] < 2 or mat.shape[1] < 2:
            return self._plot_heatmap(mat, e1, e2, cmap, show_val, measure)
        
        fmt = ".0f" if measure == "Raw Counts" else ".1f" if "%" in measure else ".2f"
        center = 0 if measure in ["Chi-square Residuals", "Log Odds Ratio"] else None
        if center is not None:
            cmap = "coolwarm"
        
        g = sns.clustermap(mat, annot=show_val, fmt=fmt, cmap=cmap, center=center,
            figsize=(12, 10), dendrogram_ratio=0.12, cbar_kws={"label": measure}, annot_kws={"fontsize": 7})
        
        g.fig.suptitle(f"Clustered: {e1} × {e2}", y=1.01, fontsize=14)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)
        g.ax_heatmap.set_xlabel(e2, fontsize=11)
        g.ax_heatmap.set_ylabel(e1, fontsize=11)
        return g.fig
    
    def _plot_top_pairs(self, mat, e1, e2, cmap, measure):
        top_n = self.pairs_top_n.get()
        sign_opt = self.pairs_sign.get()
        
        pairs = []
        for r in mat.index:
            for c in mat.columns:
                v = mat.loc[r, c]
                if pd.notna(v) and v != 0:
                    raw = self._contingency_matrix.loc[r, c] if self._contingency_matrix is not None else abs(v)
                    pairs.append({"Row": str(r), "Col": str(c), "Val": float(v), "Cnt": float(raw)})
        
        if not pairs:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "No pairs found", ha='center', va='center')
            ax.set_axis_off()
            return fig
        
        pdf = pd.DataFrame(pairs)
        if sign_opt == "Positive only":
            pdf = pdf[pdf["Val"] > 0]
        elif sign_opt == "Negative only":
            pdf = pdf[pdf["Val"] < 0]
        
        pdf["abs"] = pdf["Val"].abs()
        pdf = pdf.sort_values("abs", ascending=False).head(top_n)
        
        if pdf.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "No pairs match criteria", ha='center', va='center')
            ax.set_axis_off()
            return fig
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        r_lbl = list(pdf.groupby("Row")["abs"].sum().sort_values(ascending=False).index)
        c_lbl = list(pdf.groupby("Col")["abs"].sum().sort_values(ascending=False).index)
        
        x_pos, y_pos = {l: i for i, l in enumerate(r_lbl)}, {l: i for i, l in enumerate(c_lbl)}
        pdf = pdf[pdf["Row"].isin(r_lbl) & pdf["Col"].isin(c_lbl)]
        
        x = [x_pos[r] for r in pdf["Row"]]
        y = [y_pos[c] for c in pdf["Col"]]
        max_cnt = pdf["Cnt"].max() if pdf["Cnt"].max() > 0 else 1
        sizes = 50 + (pdf["Cnt"] / max_cnt) * 500
        
        sc = ax.scatter(x, y, s=sizes, c=pdf["Val"], cmap=cmap, alpha=0.7, edgecolors='white', linewidth=1)
        plt.colorbar(sc, ax=ax, label=measure)
        
        ax.set_xticks(range(len(r_lbl)))
        ax.set_xticklabels(r_lbl, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(c_lbl)))
        ax.set_yticklabels(c_lbl, fontsize=9)
        ax.set_xlabel(e1, fontsize=11)
        ax.set_ylabel(e2, fontsize=11)
        ax.set_title(f"Top {len(pdf)} Pairs: {e1} × {e2}", fontsize=14, pad=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    
    def _plot_correspondence(self, cont, e1, e2):
        from sklearn.decomposition import TruncatedSVD
        
        data = np.nan_to_num(cont.values.astype(float), nan=0.0)
        row_s, col_s, total = data.sum(axis=1, keepdims=True), data.sum(axis=0, keepdims=True), data.sum()
        
        if total == 0 or min(data.shape) < 3:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "Not enough data for CA", ha='center', va='center')
            ax.set_axis_off()
            return fig
        
        exp = (row_s @ col_s) / total
        with np.errstate(divide='ignore', invalid='ignore'):
            res = (data - exp) / np.sqrt(exp)
            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        
        svd = TruncatedSVD(n_components=2)
        row_c = svd.fit_transform(res)
        col_c = svd.components_.T * svd.singular_values_
        expl = svd.explained_variance_ratio_ * 100
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        ax.scatter(row_c[:, 0], row_c[:, 1], c='steelblue', s=120, alpha=0.7, label=e1)
        for i, l in enumerate(cont.index):
            ax.annotate(str(l), (row_c[i, 0], row_c[i, 1]), fontsize=9, color='steelblue', fontweight='bold')
        
        ax.scatter(col_c[:, 0], col_c[:, 1], c='coral', s=120, alpha=0.7, marker='s', label=e2)
        for i, l in enumerate(cont.columns):
            ax.annotate(str(l), (col_c[i, 0], col_c[i, 1]), fontsize=9, color='coral', fontweight='bold')
        
        ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
        ax.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
        ax.set_xlabel(f"Dim 1 ({expl[0]:.1f}%)", fontsize=11)
        ax.set_ylabel(f"Dim 2 ({expl[1]:.1f}%)", fontsize=11)
        ax.set_title(f"Correspondence Analysis: {e1} × {e2}", fontsize=14, pad=10)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    
    def _plot_sankey(self, cont, e1, e2, cmap):
        """Create Sankey diagram using plotbib implementation."""
        from biblium import plotbib
        import tempfile
        import os
        
        # Use plotbib's bipartite sankey function
        fig = plotbib.plot_bipartite_sankey(
            contingency_matrix=cont,
            left_label=e1,
            right_label=e2,
            title=f"Flow: {e1} → {e2}",
            max_links=30,
            min_value=0,
            colorscale="Blues",
            width=1000,
            height=700,
        )
        
        # Convert plotly figure to static image bytes
        try:
            img_bytes = fig.to_image(format="png", scale=2)
            return img_bytes
        except Exception as e:
            # Fallback: save to temp file and read
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                fig.write_image(tmp_path, scale=2)
                with open(tmp_path, 'rb') as f:
                    img_bytes = f.read()
                os.unlink(tmp_path)
                return img_bytes
            except Exception as e2:
                print(f"Sankey export error: {e2}")
                # Return None to trigger error
                return None
    
    def _display_image(self, img_bytes):
        # Clear existing widgets in plot_frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        try:
            from biblium.gui.widgets.plots import ScaledImageFrame
            
            buf = io.BytesIO(img_bytes)
            img = Image.open(buf)
            
            scaled_frame = ScaledImageFrame(
                self.plot_frame, 
                theme=self.theme_name,
                maintain_aspect=True,
                max_scale=1.5
            )
            scaled_frame.pack(fill=tk.BOTH, expand=True)
            scaled_frame.set_image(img)
            
            self._scaled_frame = scaled_frame
        except Exception as e:
            # Fallback to original method
            buf = io.BytesIO(img_bytes)
            img = Image.open(buf)
            self._photo_image = ImageTk.PhotoImage(img)
            self.image_label = tk.Label(self.plot_frame, image=self._photo_image, bg=self.theme["bg_card"])
            self.image_label.pack(fill=tk.BOTH, expand=True)
    
    def _show_results(self, img_bytes, mat):
        self._is_running = False
        self._last_image_bytes = img_bytes
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": self.title})
        
        self._display_image(img_bytes)
        self._update_matrix_table(mat)
    
    def _update_matrix_table(self, mat):
        for w in self.tree_frame.winfo_children():
            w.destroy()
        
        cols = [mat.index.name or "Row"] + [str(c) for c in mat.columns]
        tree = ttk.Treeview(self.tree_frame, columns=cols, show='headings', height=15)
        
        vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        tree.column(cols[0], width=120, anchor='w')
        tree.heading(cols[0], text=cols[0])
        for c in cols[1:]:
            tree.column(c, width=80, anchor='center')
            tree.heading(c, text=str(c)[:15])
        
        for idx, row in mat.iterrows():
            vals = [str(idx)[:20]] + [f"{v:.1f}" if isinstance(v, (int, float)) else str(v) for v in row]
            tree.insert("", "end", values=vals)
        
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_columnconfigure(0, weight=1)
    
    def _on_error(self, msg):
        self._is_running = False
        event_bus.emit(EventBus.ERROR_OCCURRED, {"message": msg})
        self._show_placeholder(f"❌ Error: {msg}")
        messagebox.showerror("Error", msg)
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
RELATIONSHIP ANALYSIS
═════════════════════

Analyze entity co-occurrence relationships.

RELATIONSHIP TYPES
──────────────────
• Author-Keyword
• Journal-Topic
• Country-Field
• Institution-Theme

METHODS
───────
• Co-occurrence matrix
• Association strength
• Correlation analysis
• Network analysis

OUTPUT
──────
• Relationship matrix
• Heatmap visualization
• Network diagram
• Top associations

METRICS
───────
• Co-occurrence count
• Jaccard similarity
• Lift (above expected)
• PMI (pointwise mutual info)
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

    def _export_results(self):
        if self._contingency_matrix is None:
            messagebox.showwarning("No Results", "Run analysis first.")
            return
        
        fp = filedialog.asksaveasfilename(defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("All", "*.*")])
        
        if fp:
            try:
                if fp.endswith('.xlsx'):
                    self._contingency_matrix.to_excel(fp, sheet_name='Contingency')
                else:
                    self._contingency_matrix.to_csv(fp)
                messagebox.showinfo("Exported", f"Saved to {fp}")
                if self._current_fig:
                    self._current_fig.savefig(fp.rsplit('.', 1)[0] + '_plot.png', dpi=300, bbox_inches='tight')
            except Exception as e:
                messagebox.showerror("Export Error", str(e))
