# -*- coding: utf-8 -*-
"""
Word Cloud Panel
================
Generate word cloud visualizations from bibliometric data using biblium's implementation.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Dict, Optional
import io

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageTk
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


class WordCloudPanel(BasePanel):
    """Panel for word cloud visualization using biblium's plotbib.plot_wordcloud."""
    
    title = "Word Cloud"
    icon = "☁️"
    description = "Generate word cloud visualizations from entity frequencies"
    requires_data = True
    
    # Entity configurations: (stats_attr, getter_method, label_col_candidates)
    # label_col_candidates is a list of possible column names to try
    ENTITY_CONFIG = {
        "Words from Abstract": ("ngrams_abstract_stats_df", "get_ngrams_abstract_stats", ["Word", "Word - Phrase", "Phrase"]),
        "Words from Title": ("ngrams_title_stats_df", "get_ngrams_title_stats", ["Word", "Word - Phrase", "Phrase"]),
        "Author Keywords": ("author_keywords_stats_df", "get_author_keywords_stats", ["Keyword", "Author Keywords"]),
        "Index Keywords": ("index_keywords_stats_df", "get_index_keywords_stats", ["Keyword", "Index Keywords"]),
        "Authors": ("authors_stats_df", "get_authors_stats", ["Author", "Author(s) ID", "Authors"]),
        "Sources": ("sources_stats_df", "get_sources_stats", ["Source title", "Source", "Journal"]),
        "Countries": ("ca_countries_stats_df", "get_ca_countries_stats", ["Country", "Countries"]),
        "Affiliations": ("affiliations_stats_df", "get_affiliations_stats", ["Affiliation", "Affiliations"]),
    }
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._current_fig = None
        self._photo_image = None
        self._stats_df = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._generate_wordcloud
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Entity Selection
        entity_card = Card(self.options_content, title="📊 Entity Selection", theme=self.theme_name)
        entity_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.entity_combo = LabeledCombobox(
            entity_card.content, label="Entity:",
            values=list(self.ENTITY_CONFIG.keys()),
            default="Words from Abstract",
            theme=self.theme_name, label_width=12,
        )
        self.entity_combo.pack(fill=tk.X, pady=4)
        self.entity_combo.bind("<<ComboboxSelected>>", self._on_entity_change)
        
        self.top_n_spin = LabeledSpinbox(
            entity_card.content, label="Top N:",
            from_=10, to=200, default=50,
            theme=self.theme_name, label_width=12,
        )
        self.top_n_spin.pack(fill=tk.X, pady=4)
        
        # Size & Color Settings
        visual_card = Card(self.options_content, title="📏 Size & Color", theme=self.theme_name)
        visual_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.size_col_combo = LabeledCombobox(
            visual_card.content, label="Size by:",
            values=["Number of documents", "Total citations", "H-index", "Citations per document"],
            default="Number of documents",
            theme=self.theme_name, label_width=12,
        )
        self.size_col_combo.pack(fill=tk.X, pady=4)
        
        self.color_col_combo = LabeledCombobox(
            visual_card.content, label="Color by:",
            values=["None", "Number of documents", "Total citations", "H-index", 
                    "Citations per document", "Average year", "First year", "Last year"],
            default="Average year",
            theme=self.theme_name, label_width=12,
        )
        self.color_col_combo.pack(fill=tk.X, pady=4)
        
        # Appearance Settings
        appear_card = Card(self.options_content, title="🎨 Appearance", theme=self.theme_name)
        appear_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.colormap_combo = LabeledCombobox(
            appear_card.content, label="Colormap:",
            values=["viridis", "plasma", "inferno", "magma", "cividis", 
                    "Blues", "Greens", "Reds", "Oranges", "Purples",
                    "YlOrRd", "YlGnBu", "RdYlBu", "Spectral", "coolwarm"],
            default="viridis",
            theme=self.theme_name, label_width=12,
        )
        self.colormap_combo.pack(fill=tk.X, pady=4)
        
        self.bg_combo = LabeledCombobox(
            appear_card.content, label="Background:",
            values=["white", "black", "lightgray", "darkgray"],
            default="white",
            theme=self.theme_name, label_width=12,
        )
        self.bg_combo.pack(fill=tk.X, pady=4)
        
        self.show_cbar_cb = LabeledCheckbox(
            appear_card.content, label="Show colorbar",
            default=True, theme=self.theme_name,
        )
        self.show_cbar_cb.pack(fill=tk.X, pady=2)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Generate Word Cloud", icon="☁️",
            command=self._generate_wordcloud, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        ThemedButton(
            btn_frame, text="Export Image", style="secondary",
            command=self._export_image, theme=self.theme_name,
        ).pack(fill=tk.X, pady=(8, 0))
    
    def _on_entity_change(self, event=None):
        """Update available columns when entity changes."""
        # Could dynamically update size/color options based on entity
        pass
    
    def _create_results(self):
        """Create the results panel."""
        super()._create_results()
        
        # Create notebook for results and info
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Image display tab
        self.image_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.image_frame, text="☁️ Word Cloud")
        
        # Info tab
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
        self._show_placeholder("Select entity and click 'Generate Word Cloud'")
    
    def _show_placeholder(self, message: str):
        """Show placeholder message."""
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.image_frame, text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _generate_wordcloud(self):
        """Generate word cloud visualization using biblium's plot_wordcloud."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_WORDCLOUD:
            messagebox.showerror("Missing Library", 
                "WordCloud library is required.\nInstall with: pip install wordcloud")
            return
        
        if not HAS_MATPLOTLIB:
            messagebox.showerror("Missing Library", "Matplotlib is required.")
            return
        
        # Emit start event
        event_bus.emit(EventBus.ANALYSIS_STARTED, {"name": self.title})
        self._is_running = True
        
        self._show_loading("Generating word cloud...")
        
        entity = self.entity_combo.get()
        top_n = self.top_n_spin.get()
        size_col = self.size_col_combo.get()
        color_col = self.color_col_combo.get()
        colormap = self.colormap_combo.get()
        bg_color = self.bg_combo.get()
        show_cbar = self.show_cbar_cb.get()
        
        def do_generate():
            try:
                # Helper to find column with case-insensitive matching
                def find_column(name, df_cols):
                    """Find column by name with case-insensitive fallback."""
                    if name in df_cols:
                        return name
                    # Try case-insensitive match
                    name_lower = name.lower()
                    for col in df_cols:
                        if col.lower() == name_lower:
                            return col
                    return None
                
                # Get stats dataframe from biblium
                df = self._get_stats_df(entity, top_n)
                
                if df is None or len(df) == 0:
                    # Provide more helpful error message based on entity type
                    if entity in ["Words from Abstract"]:
                        raise ValueError(f"No Abstract column found in dataset. Try 'Words from Title' instead.")
                    elif entity in ["Words from Title"]:
                        raise ValueError(f"No Title column found in dataset.")
                    elif entity in ["Countries"]:
                        raise ValueError(f"No Countries data found. This requires affiliation data to extract countries.")
                    else:
                        raise ValueError(f"No data found for {entity}. The required column may not exist in your dataset.")
                
                # Make a copy to avoid SettingWithCopyWarning
                df = df.copy()
                self._stats_df = df.copy()
                
                # Compute "Citations per document" if not present
                if "Citations per document" not in df.columns:
                    if "Total citations" in df.columns and "Number of documents" in df.columns:
                        df["Citations per document"] = df["Total citations"] / df["Number of documents"].replace(0, np.nan)
                        df["Citations per document"] = df["Citations per document"].fillna(0)
                
                # DEBUG: Print DataFrame info
                print(f"[DEBUG] WordCloud - Entity: {entity}")
                print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
                print(f"[DEBUG] DataFrame shape: {df.shape}")
                print(f"[DEBUG] First 3 rows:\n{df.head(3)}")
                
                # Get entity config
                config = self.ENTITY_CONFIG.get(entity)
                label_col_candidates = config[2] if config else [df.columns[0]]
                
                # Ensure it's a list
                if isinstance(label_col_candidates, str):
                    label_col_candidates = [label_col_candidates]
                
                # Find the actual item column from candidates
                item_col = None
                for candidate in label_col_candidates:
                    item_col = find_column(candidate, df.columns)
                    if item_col:
                        break
                
                if item_col is None:
                    # Fallback: use first non-numeric column or first column
                    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
                    item_col = non_numeric[0] if non_numeric else df.columns[0]
                    print(f"[DEBUG] None of {label_col_candidates} found, using '{item_col}'")
                
                print(f"[DEBUG] item_col: {item_col}")
                print(f"[DEBUG] size_col requested: {size_col}")
                print(f"[DEBUG] color_col requested: {color_col}")
                
                # Validate and map size column
                size_col_actual = find_column(size_col, df.columns)
                if size_col_actual is None:
                    # Fallback to first numeric column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    print(f"[DEBUG] size_col '{size_col}' not in columns, numeric cols: {numeric_cols}")
                    size_col_actual = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]
                
                print(f"[DEBUG] size_col_actual: {size_col_actual}")
                
                # Validate and map color column
                if color_col == "None":
                    color_by = None
                    print(f"[DEBUG] color_by set to None (user selected None)")
                else:
                    color_by = find_column(color_col, df.columns)
                    if color_by is None:
                        print(f"[DEBUG] color_by set to None (color_col '{color_col}' not in columns)")
                    else:
                        print(f"[DEBUG] color_by: {color_by}")
                        print(f"[DEBUG] color values sample: {df[color_by].head(5).tolist()}")
                
                # Use biblium's plot_wordcloud
                from biblium import plotbib
                
                # Create figure without showing
                import matplotlib.pyplot as plt
                plt.ioff()
                
                # Call biblium's plot_wordcloud but capture the figure
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Prepare data
                work = df.head(top_n).copy()
                items = work[item_col].astype(str)
                sizes = pd.to_numeric(work[size_col_actual], errors="coerce").fillna(0.0).astype(float)
                
                if (sizes <= 0).all():
                    raise ValueError(f"All values in '{size_col_actual}' are non-positive")
                
                # Color mapping
                import matplotlib.colors as mcolors
                from matplotlib import cm
                
                colorbar_type = None
                norm = None
                cmap_obj = None
                
                if color_by is None:
                    color_map = {item: "#808080" for item in items}
                    print(f"[DEBUG] Using gray color for all items (no color_by)")
                else:
                    color_values = work[color_by]
                    print(f"[DEBUG] color_values dtype: {color_values.dtype}")
                    print(f"[DEBUG] color_values is numeric: {pd.api.types.is_numeric_dtype(color_values)}")
                    
                    if pd.api.types.is_numeric_dtype(color_values):
                        vmin = float(np.nanmin(color_values))
                        vmax = float(np.nanmax(color_values))
                        print(f"[DEBUG] vmin: {vmin}, vmax: {vmax}")
                        if vmin == vmax:
                            vmax = vmin + 1e-12
                        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                        cmap_obj = cm.get_cmap(colormap)
                        color_map = {
                            item: mcolors.to_hex(cmap_obj(norm(val if np.isfinite(val) else vmin)))
                            for item, val in zip(items, color_values)
                        }
                        print(f"[DEBUG] Sample colors: {list(color_map.items())[:3]}")
                        colorbar_type = "continuous"
                    else:
                        color_map = {item: "#808080" for item in items}
                
                # Create frequencies dict
                frequencies = dict(zip(items, sizes))
                
                # Color function
                def color_func(word, **_):
                    return color_map.get(word, "#808080")
                
                # Generate wordcloud
                wc = WordCloud(
                    width=1200,
                    height=800,
                    background_color=bg_color,
                    prefer_horizontal=0.9,
                    random_state=42,
                    color_func=color_func,
                    collocations=False,
                    relative_scaling=0,
                    max_words=len(frequencies),
                )
                wc.generate_from_frequencies(frequencies)
                
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title(f"Word Cloud: {entity} (Top {top_n})", fontsize=14, pad=10)
                
                # Add colorbar if requested
                if show_cbar and color_by is not None and colorbar_type == "continuous":
                    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
                    sm.set_array([])
                    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
                    cb.set_label(str(color_by), fontsize=10)
                
                fig.tight_layout()
                self._current_fig = fig
                
                # Convert to image bytes
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                image_bytes = buf.getvalue()
                buf.close()
                
                plt.close(fig)
                
                self.after(0, lambda: self._show_image_from_bytes(image_bytes))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._on_error(msg))
        
        threading.Thread(target=do_generate, daemon=True).start()
    
    def _get_stats_df(self, entity: str, top_n: int) -> Optional[pd.DataFrame]:
        """Get stats dataframe from biblium."""
        config = self.ENTITY_CONFIG.get(entity)
        
        print(f"[DEBUG] _get_stats_df called for entity: {entity}")
        print(f"[DEBUG] config: {config}")
        
        if config:
            stats_attr, getter_method, label_col_candidates = config
            
            print(f"[DEBUG] stats_attr: {stats_attr}, getter_method: {getter_method}")
            print(f"[DEBUG] hasattr(bib, stats_attr): {hasattr(self.bib, stats_attr)}")
            
            # Check if stats already computed
            existing_df = getattr(self.bib, stats_attr, None)
            print(f"[DEBUG] existing stats df is None: {existing_df is None}")
            
            if existing_df is None:
                # Call getter method
                print(f"[DEBUG] hasattr(bib, getter_method): {hasattr(self.bib, getter_method)}")
                if hasattr(self.bib, getter_method):
                    print(f"[DEBUG] Calling {getter_method}(top_n={top_n})")
                    try:
                        getattr(self.bib, getter_method)(top_n=top_n)
                    except Exception as e:
                        print(f"[DEBUG] Error calling {getter_method}: {e}")
            
            df = getattr(self.bib, stats_attr, None)
            if df is not None:
                print(f"[DEBUG] Got stats df with columns: {df.columns.tolist()}")
                print(f"[DEBUG] Stats df shape: {df.shape}")
                return df.head(top_n)
            else:
                print(f"[DEBUG] Stats df is still None after calling getter")
        
        # Fallback: manual counting
        print(f"[DEBUG] Using fallback counts")
        return self._fallback_counts(entity, top_n)
    
    def _fallback_counts(self, entity: str, top_n: int) -> Optional[pd.DataFrame]:
        """Fallback counting when biblium stats not available."""
        import re
        df = self.bib.df
        
        def find_col(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None
        
        # Stopwords for text processing
        stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'was', 
                     'were', 'been', 'have', 'has', 'had', 'not', 'but', 'can', 'which',
                     'their', 'they', 'these', 'there', 'than', 'also', 'into', 'such',
                     'more', 'other', 'some', 'what', 'when', 'where', 'who', 'how',
                     'its', 'our', 'using', 'based', 'study', 'results', 'analysis',
                     'used', 'use', 'may', 'two', 'one', 'new', 'however', 'well',
                     'show', 'showed', 'shown', 'between', 'through', 'during', 'after',
                     'before', 'both', 'each', 'most', 'all', 'only', 'over', 'under'}
        
        if entity == "Words from Abstract":
            col = find_col(["Processed Abstract", "Abstract", "AB", "abstract"])
            if col:
                all_text = df[col].dropna().str.cat(sep=' ')
                words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
                words = [w for w in words if w not in stopwords]
                counts = pd.Series(words).value_counts().head(top_n).reset_index()
                counts.columns = ['Word', 'Number of documents']
                return counts
            return None
            
        elif entity == "Words from Title":
            col = find_col(["Processed Title", "Title", "TI", "Article Title"])
            if col:
                all_text = df[col].dropna().str.cat(sep=' ')
                words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
                words = [w for w in words if w not in stopwords]
                counts = pd.Series(words).value_counts().head(top_n).reset_index()
                counts.columns = ['Word', 'Number of documents']
                return counts
            return None
        
        elif entity == "Author Keywords":
            col = find_col(["Author Keywords", "Keywords", "DE"])
            label = "Keyword"
        elif entity == "Index Keywords":
            col = find_col(["Index Keywords", "ID"])
            label = "Keyword"
        elif entity == "Authors":
            col = find_col(["Authors", "Authors or Inventors", "AU"])
            label = "Author"
        elif entity == "Sources":
            col = find_col(["Source title", "Source Title", "Journal", "SO"])
            label = "Source title"
        elif entity == "Countries":
            col = find_col(["Countries", "Country"])
            label = "Country"
        elif entity == "Affiliations":
            col = find_col(["Affiliations", "Addresses", "C1"])
            label = "Affiliation"
        else:
            return None
        
        if col is None:
            return None
        
        # Count items
        if entity in ["Author Keywords", "Index Keywords", "Authors", "Affiliations"]:
            items = df[col].dropna().str.split(";").explode().str.strip()
            items = items[items != '']
        else:
            items = df[col].dropna()
        
        counts = items.value_counts().head(top_n).reset_index()
        counts.columns = [label, 'Number of documents']
        
        return counts
    
    def _show_image_from_bytes(self, image_bytes):
        """Display image from bytes in main thread."""
        self._is_running = False
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": self.title})
        
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        # Add AI button header
        ai_header = tk.Frame(self.image_frame, bg=self.theme["bg_card"])
        ai_header.pack(fill=tk.X, padx=4, pady=(4, 2))
        
        ai_btn = tk.Button(
            ai_header, text="🤖 AI Describe",
            font=("Segoe UI", 9),
            bg=self.theme["accent_primary"], fg="white",
            relief=tk.FLAT, cursor="hand2", padx=8, pady=2,
            command=self._ai_describe_wordcloud,
        )
        ai_btn.pack(side=tk.RIGHT, padx=4)
        self._ai_btn = ai_btn
        
        try:
            from biblium.gui.widgets.plots import ScaledImageFrame
            
            buf = io.BytesIO(image_bytes)
            img = Image.open(buf)
            
            scaled_frame = ScaledImageFrame(
                self.image_frame, 
                theme=self.theme_name,
                maintain_aspect=True,
                max_scale=1.5
            )
            scaled_frame.pack(fill=tk.BOTH, expand=True)
            scaled_frame.set_image(img)
            
            self._scaled_frame = scaled_frame
            scaled_frame.bind("<Button-3>", self._on_right_click)
        except Exception as e:
            # Fallback to original method
            buf = io.BytesIO(image_bytes)
            img = Image.open(buf)
            self._photo_image = ImageTk.PhotoImage(img)
            label = tk.Label(self.image_frame, image=self._photo_image, bg=self.theme["bg_card"])
            label.pack(expand=True)
            label.bind("<Button-3>", self._on_right_click)
    
    def _on_right_click(self, event):
        """Handle right-click for context menu."""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Save as PNG", command=lambda: self._save_image("png"))
        menu.add_command(label="Save as PDF", command=lambda: self._save_image("pdf"))
        menu.add_command(label="Save as SVG", command=lambda: self._save_image("svg"))
        menu.tk_popup(event.x_root, event.y_root)
    
    def _save_image(self, fmt: str):
        """Save the current image."""
        if self._current_fig is None:
            messagebox.showwarning("No Image", "Generate a word cloud first.")
            return
        
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} files", f"*.{fmt}"), ("All files", "*.*")],
        )
        if filepath:
            self._current_fig.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Saved", f"Image saved to {filepath}")
    
    def _export_image(self):
        """Export current image."""
        self._save_image("png")
    
    def _on_error(self, message: str):
        """Handle error."""
        self._is_running = False
        event_bus.emit(EventBus.ERROR_OCCURRED, {"message": message})
        self._show_placeholder(f"❌ Error: {message}")
    
    def _ai_describe_wordcloud(self):
        """Generate AI description of the wordcloud."""
        from biblium.gui.widgets.tables import DataTable
        settings = DataTable.get_llm_settings()
        
        if not settings.get("api_key"):
            messagebox.showinfo("Configure AI", 
                "Please configure your AI API key in Settings first.\n\n"
                "Go to Settings (⚙️) and enter your API key.")
            return
        
        if hasattr(self, '_ai_btn'):
            self._ai_btn.config(text="⏳ Generating...", state=tk.DISABLED)
        
        # Get current settings for context
        entity = self.entity_combo.get() if hasattr(self, 'entity_combo') else "Unknown"
        top_n = int(self.top_n_spin.get()) if hasattr(self, 'top_n_spin') else 50
        
        plot_info = {
            "type": "word cloud",
            "title": f"Word Cloud: {entity}",
            "data_summary": f"Top {top_n} terms for {entity}. Size represents frequency.",
            "context": "Bibliometric word cloud showing most frequent terms",
        }
        
        import threading
        def do_generate():
            try:
                from biblium.llm_utils import llm_describe_plot
                result = llm_describe_plot(
                    plot_type=plot_info["type"],
                    title=plot_info["title"],
                    data_summary=plot_info["data_summary"],
                    context=plot_info["context"],
                    provider=settings["provider"],
                    model=settings["model"],
                    api_key=settings["api_key"],
                )
                self.after(0, lambda r=result: self._show_wc_ai_result(r))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.after(0, lambda msg=error_msg: self._show_wc_ai_result(msg))
        
        thread = threading.Thread(target=do_generate, daemon=True)
        thread.start()
    
    def _show_wc_ai_result(self, text: str):
        """Show AI result for wordcloud."""
        if hasattr(self, '_ai_btn'):
            self._ai_btn.config(text="🤖 AI Describe", state=tk.NORMAL)
        
        if hasattr(self, '_ai_result_frame') and self._ai_result_frame:
            try:
                self._ai_result_frame.destroy()
            except:
                pass
        
        self._ai_result_frame = tk.Frame(self.image_frame, bg=self.theme["bg_card"])
        self._ai_result_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0))
        
        header = tk.Frame(self._ai_result_frame, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header, text="🤖 AI Description",
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT, padx=4)
        
        tk.Button(
            header, text="✕", font=("Segoe UI", 8),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            relief=tk.FLAT, command=lambda: self._ai_result_frame.destroy(),
            cursor="hand2", width=2,
        ).pack(side=tk.RIGHT, padx=2)
        
        def copy_text():
            try:
                self.clipboard_clear()
                self.clipboard_append(text)
                copy_btn.config(text="✓ Copied")
                self.after(1500, lambda: copy_btn.config(text="📋 Copy"))
            except:
                pass
        
        copy_btn = tk.Button(
            header, text="📋 Copy", font=("Segoe UI", 8),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            relief=tk.FLAT, command=copy_text, cursor="hand2",
        )
        copy_btn.pack(side=tk.RIGHT, padx=2)
        
        text_widget = tk.Text(
            self._ai_result_frame, wrap=tk.WORD,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            relief=tk.FLAT, height=4, padx=8, pady=4,
        )
        text_widget.pack(fill=tk.X, padx=4, pady=(2, 4))
        text_widget.insert("1.0", text)
        
        def on_key(e):
            if e.state & 0x4 and e.keysym.lower() in ('c', 'a'):
                return
            return "break"
        text_widget.bind("<Key>", on_key)
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
WORD CLOUD
══════════

Visual representation of term frequencies.

TEXT SOURCES
────────────
• Author keywords
• Index keywords
• Titles
• Abstracts

WORD SIZING
───────────
• By frequency (count)
• By TF-IDF weight
• By citation impact

CUSTOMIZATION
─────────────
• Maximum words
• Minimum frequency
• Color scheme
• Font selection
• Shape/mask

STOPWORDS
─────────
• Default English stopwords
• Custom stopword list
• Domain-specific filtering

OUTPUT
──────
• Word cloud image
• Word frequency table
• Export to PNG/SVG

INTERPRETATION
──────────────
• Larger words = more frequent
• Color can show categories
• Position is random
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
                pass
        
        def copy_all(widget):
            widget.config(state=tk.NORMAL)
            content = widget.get("1.0", tk.END)
            widget.config(state=tk.DISABLED)
            widget.clipboard_clear()
            widget.clipboard_append(content.strip())
        
        text_widget.bind("<Button-3>", show_context_menu)
        text_widget.bind("<Control-c>", lambda e: copy_selected(text_widget))

    def _show_loading(self, message: str = "Loading..."):
        """Show loading indicator."""
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.image_frame, text=f"⏳ {message}",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(expand=True)
