# -*- coding: utf-8 -*-
"""
Treemap Panel
=============
Generate treemap visualizations from bibliometric data using biblium's implementation.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Dict, Optional
import io
import textwrap

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
    import matplotlib.colors as mcolors
    from matplotlib import cm
    from PIL import Image, ImageTk
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import squarify
    HAS_SQUARIFY = True
except ImportError:
    HAS_SQUARIFY = False


class TreemapPanel(BasePanel):
    """Panel for treemap visualization using biblium's plotbib.plot_treemap."""
    
    title = "Treemap"
    icon = "🗃️"
    description = "Generate treemap visualizations showing hierarchical data"
    requires_data = True
    
    # Entity configurations: (stats_attr, getter_method, label_col)
    ENTITY_CONFIG = {
        "Author Keywords": ("author_keywords_stats_df", "get_author_keywords_stats", "Keyword"),
        "Index Keywords": ("index_keywords_stats_df", "get_index_keywords_stats", "Keyword"),
        "Authors": ("authors_stats_df", "get_authors_stats", "Author"),
        "Sources": ("sources_stats_df", "get_sources_stats", "Source title"),
        "Countries": ("ca_countries_stats_df", "get_ca_countries_stats", "Country"),
        "Affiliations": ("affiliations_stats_df", "get_affiliations_stats", "Affiliation"),
        "Document Types": ("document_type_stats_df", "get_document_type_stats", "Document Type"),
    }
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._current_fig = None
        self._photo_image = None
        self._stats_df = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._generate_treemap
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Entity Selection
        entity_card = Card(self.options_content, title="📊 Entity Selection", theme=self.theme_name)
        entity_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.entity_combo = LabeledCombobox(
            entity_card.content, label="Entity:",
            values=list(self.ENTITY_CONFIG.keys()),
            default="Author Keywords",
            theme=self.theme_name, label_width=12,
        )
        self.entity_combo.pack(fill=tk.X, pady=4)
        
        self.top_n_spin = LabeledSpinbox(
            entity_card.content, label="Top N:",
            from_=5, to=100, default=20,
            theme=self.theme_name, label_width=12,
        )
        self.top_n_spin.pack(fill=tk.X, pady=4)
        
        # Size & Color Settings
        visual_card = Card(self.options_content, title="📏 Size & Color", theme=self.theme_name)
        visual_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.size_col_combo = LabeledCombobox(
            visual_card.content, label="Size by:",
            values=["Number of documents", "Total citations", "h-index", "Average citations"],
            default="Number of documents",
            theme=self.theme_name, label_width=12,
        )
        self.size_col_combo.pack(fill=tk.X, pady=4)
        
        self.color_col_combo = LabeledCombobox(
            visual_card.content, label="Color by:",
            values=["None", "Number of documents", "Total citations", "h-index", 
                    "Average citations", "Average year", "First year", "Last year"],
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
                    "YlOrRd", "YlGnBu", "RdYlBu", "Spectral", "coolwarm",
                    "tab10", "tab20", "Set1", "Set2", "Pastel1"],
            default="viridis",
            theme=self.theme_name, label_width=12,
        )
        self.colormap_combo.pack(fill=tk.X, pady=4)
        
        self.show_labels_cb = LabeledCheckbox(
            appear_card.content, label="Show labels",
            default=True, theme=self.theme_name,
        )
        self.show_labels_cb.pack(fill=tk.X, pady=2)
        
        self.show_values_cb = LabeledCheckbox(
            appear_card.content, label="Show values",
            default=True, theme=self.theme_name,
        )
        self.show_values_cb.pack(fill=tk.X, pady=2)
        
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
            btn_frame, text="Generate Treemap", icon="🗃️",
            command=self._generate_treemap, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        ThemedButton(
            btn_frame, text="Export Image", style="secondary",
            command=self._export_image, theme=self.theme_name,
        ).pack(fill=tk.X, pady=(8, 0))
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook for results and info
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.image_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.image_frame, text="🗺️ Treemap")
        
        # Info tab
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
        # Image display area
        self.image_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        self._show_placeholder("Select entity and click 'Generate Treemap'")
    
    def _show_placeholder(self, message: str):
        """Show placeholder message."""
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.image_frame, text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _generate_treemap(self):
        """Generate treemap visualization using biblium's approach."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_SQUARIFY:
            messagebox.showerror("Missing Library", 
                "squarify library is required.\nInstall with: pip install squarify")
            return
        
        if not HAS_MATPLOTLIB:
            messagebox.showerror("Missing Library", "Matplotlib is required.")
            return
        
        # Emit start event
        event_bus.emit(EventBus.ANALYSIS_STARTED, {"name": self.title})
        self._is_running = True
        
        self._show_loading("Generating treemap...")
        
        entity = self.entity_combo.get()
        top_n = self.top_n_spin.get()
        size_col = self.size_col_combo.get()
        color_col = self.color_col_combo.get()
        colormap_name = self.colormap_combo.get()
        show_labels = self.show_labels_cb.get()
        show_values = self.show_values_cb.get()
        show_cbar = self.show_cbar_cb.get()
        
        def do_generate():
            try:
                # Get stats dataframe from biblium
                df = self._get_stats_df(entity, top_n)
                
                if df is None or len(df) == 0:
                    raise ValueError(f"No data found for {entity}")
                
                self._stats_df = df
                
                # Get entity config
                config = self.ENTITY_CONFIG.get(entity)
                item_col = config[2] if config else df.columns[0]
                
                # Validate size column
                if size_col not in df.columns:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    size_col_actual = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]
                else:
                    size_col_actual = size_col
                
                # Validate color column
                if color_col == "None" or color_col not in df.columns:
                    color_by = None
                else:
                    color_by = color_col
                
                # Prepare data
                work = df.head(top_n).copy()
                sizes = pd.to_numeric(work[size_col_actual], errors="coerce").fillna(0.0).values
                labels = work[item_col].astype(str).values
                
                if (sizes <= 0).all():
                    raise ValueError(f"All values in '{size_col_actual}' are non-positive")
                
                # Create display labels
                if show_labels:
                    if show_values:
                        display_labels = [
                            f"{textwrap.fill(str(l), width=15)}\n({int(s)})"
                            for l, s in zip(labels, sizes)
                        ]
                    else:
                        display_labels = [textwrap.fill(str(l), width=15) for l in labels]
                else:
                    display_labels = [''] * len(labels)
                
                # Color mapping
                colorbar_type = None
                norm = None
                cmap_obj = cm.get_cmap(colormap_name)
                
                if color_by is None:
                    # Color by size
                    norm = mcolors.Normalize(vmin=min(sizes), vmax=max(sizes))
                    colors = [cmap_obj(norm(s)) for s in sizes]
                    colorbar_type = "continuous"
                    color_label = size_col_actual
                else:
                    color_values = work[color_by].values
                    if pd.api.types.is_numeric_dtype(work[color_by]):
                        vmin = float(np.nanmin(color_values))
                        vmax = float(np.nanmax(color_values))
                        if vmin == vmax:
                            vmax = vmin + 1e-12
                        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                        colors = [cmap_obj(norm(v if np.isfinite(v) else vmin)) for v in color_values]
                        colorbar_type = "continuous"
                        color_label = color_by
                    else:
                        # Categorical - just use size for color
                        norm = mcolors.Normalize(vmin=min(sizes), vmax=max(sizes))
                        colors = [cmap_obj(norm(s)) for s in sizes]
                        colorbar_type = "continuous"
                        color_label = size_col_actual
                
                # Create figure
                fig, ax = plt.subplots(figsize=(14, 10))
                
                # Normalize sizes for squarify
                normed_sizes = squarify.normalize_sizes(sizes, 100, 100)
                boxes = squarify.squarify(normed_sizes, 0, 0, 100, 100)
                
                # Draw boxes
                for i, (box, color) in enumerate(zip(boxes, colors)):
                    rect = plt.Rectangle(
                        (box['x'], box['y']), box['dx'], box['dy'],
                        facecolor=color,
                        edgecolor='white',
                        linewidth=2
                    )
                    ax.add_patch(rect)
                
                # Add labels
                min_area = sum(sizes) * 0.01
                for i, (box, label, size) in enumerate(zip(boxes, display_labels, sizes)):
                    if size >= min_area and label:
                        x = box['x'] + box['dx'] / 2
                        y = box['y'] + box['dy'] / 2
                        fontsize = max(6, min(14, int(box['dx'] * 0.8)))
                        
                        ax.text(x, y, label, 
                               ha='center', va='center', 
                               fontsize=fontsize,
                               color='white' if self._is_dark_color(colors[i]) else 'black',
                               weight='bold',
                               wrap=True)
                
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.axis('off')
                ax.set_title(f"Treemap: {entity} (Top {top_n})", fontsize=16, pad=15)
                
                # Add colorbar
                if show_cbar and colorbar_type == "continuous":
                    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', 
                                       fraction=0.05, pad=0.02)
                    cbar.set_label(color_label, fontsize=10)
                
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
                self.after(0, lambda: self._on_error(str(e)))
        
        threading.Thread(target=do_generate, daemon=True).start()
    
    def _is_dark_color(self, color):
        """Check if a color is dark (for text contrast)."""
        if isinstance(color, tuple) and len(color) >= 3:
            r, g, b = color[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            return luminance < 0.5
        return False
    
    def _get_stats_df(self, entity: str, top_n: int) -> Optional[pd.DataFrame]:
        """Get stats dataframe from biblium."""
        config = self.ENTITY_CONFIG.get(entity)
        
        if config:
            stats_attr, getter_method, label_col = config
            
            # Check if stats already computed
            if not hasattr(self.bib, stats_attr) or getattr(self.bib, stats_attr) is None:
                # Call getter method
                if hasattr(self.bib, getter_method):
                    getattr(self.bib, getter_method)(top_n=top_n)
            
            df = getattr(self.bib, stats_attr, None)
            if df is not None:
                return df.head(top_n)
        
        # Fallback: manual counting
        return self._fallback_counts(entity, top_n)
    
    def _fallback_counts(self, entity: str, top_n: int) -> Optional[pd.DataFrame]:
        """Fallback counting when biblium stats not available."""
        df = self.bib.df
        
        def find_col(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None
        
        if entity == "Author Keywords":
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
        elif entity == "Document Types":
            col = find_col(["Document Type", "Document type", "DT", "Type"])
            label = "Document Type"
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
        menu.add_command(label="📄 Add to Report", command=self._add_plot_to_report)
        menu.add_separator()
        menu.add_command(label="💾 Save as PNG", command=lambda: self._save_image("png"))
        menu.add_command(label="💾 Save as PDF", command=lambda: self._save_image("pdf"))
        menu.add_command(label="💾 Save as SVG", command=lambda: self._save_image("svg"))
        menu.tk_popup(event.x_root, event.y_root)
    
    def _add_plot_to_report(self):
        """Add current plot to report queue."""
        if self._current_fig is None:
            messagebox.showinfo("No Plot", "No plot to add to report.")
            return
        
        try:
            from biblium.gui.core.state import report_queue
            
            plot_title = "Treemap"
            if self._current_fig.axes:
                plot_title = self._current_fig.axes[0].get_title() or "Treemap"
            
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
    
    def _save_image(self, fmt: str):
        """Save the current image."""
        if self._current_fig is None:
            messagebox.showwarning("No Image", "Generate a treemap first.")
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
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
TREEMAP VISUALIZATION
═════════════════════

Hierarchical visualization of bibliometric entities.

WHAT IS A TREEMAP?
──────────────────
• Rectangles sized by value
• Nested hierarchy
• Space-filling layout
• Color by category

ENTITY TYPES
────────────
• Authors by publications
• Sources by papers
• Keywords by frequency
• Countries by output
• Subjects by count

HIERARCHY LEVELS
────────────────
• Single level: Flat treemap
• Two levels: Nested groups
• Color grouping

SIZE METRIC
───────────
• Document count
• Citation count
• Any numeric field

CUSTOMIZATION
─────────────
• Color palette
• Label display
• Minimum size threshold
• Border styling

EXPORT
──────
• PNG image
• SVG vector
• Interactive HTML
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
