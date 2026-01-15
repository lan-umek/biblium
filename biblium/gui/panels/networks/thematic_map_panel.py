# -*- coding: utf-8 -*-
"""
Thematic Map Panel
==================
Strategic diagram / thematic map for keyword analysis.
Based on Callon's centrality-density strategic diagram.
Uses biblium.utilsbib.build_thematic_map for core computation.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import io
from typing import Dict, List, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox
from biblium.gui.widgets.tables import DataTable

try:
    import pandas as pd
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ThematicMapPanel(BasePanel):
    """Panel for thematic map / strategic diagram analysis."""
    
    title = "Thematic Map"
    icon = "🗺️"
    description = "Strategic diagram showing research themes by centrality and density"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._clusters = None
        self._cluster_df = None
        self._items_df = None
        self._current_fig = None
        self._photo_image = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._generate_map  # Set primary action for toolbar Run button
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Analysis Type Card
        type_card = Card(self.options_content, title="📊 Analysis Settings", theme=self.theme_name)
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.field_type = LabeledCombobox(
            type_card.content, label="Field:",
            values=["Keywords", "Index Keywords", "Authors", "Sources", "Countries", "Institutions"],
            default="Keywords",
            theme=self.theme_name, label_width=12,
        )
        self.field_type.pack(fill=tk.X, pady=4)
        
        self.min_occurrences = LabeledSpinbox(
            type_card.content, label="Min Occurrences:",
            from_=1, to=50, default=2,
            theme=self.theme_name, label_width=15,
        )
        self.min_occurrences.pack(fill=tk.X, pady=4)
        
        self.top_n = LabeledSpinbox(
            type_card.content, label="Top N Items:",
            from_=10, to=200, default=50,
            theme=self.theme_name, label_width=15,
        )
        self.top_n.pack(fill=tk.X, pady=4)
        
        # Display Options
        display_card = Card(self.options_content, title="🎨 Display Options", theme=self.theme_name)
        display_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_labels = LabeledCheckbox(
            display_card.content, label="Show cluster labels",
            default=True, theme=self.theme_name,
        )
        self.show_labels.pack(fill=tk.X, pady=2)
        
        self.show_quadrant_labels = LabeledCheckbox(
            display_card.content, label="Show quadrant names",
            default=True, theme=self.theme_name,
        )
        self.show_quadrant_labels.pack(fill=tk.X, pady=2)
        
        self.bubble_size_by = LabeledCombobox(
            display_card.content, label="Bubble Size:",
            values=["Occurrences", "Items", "Uniform"],
            default="Occurrences",
            theme=self.theme_name, label_width=12,
        )
        self.bubble_size_by.pack(fill=tk.X, pady=4)
        
        # Synonyms Card
        synonyms_card = CollapsibleCard(
            self.options_content, title="🔗 Synonyms (Merge Items)",
            collapsed=True, theme=self.theme_name,
        )
        synonyms_card.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            synonyms_card.content, 
            text="Format: keyword1 = keyword2\n(one per line)",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 4))
        
        self.synonyms_text = tk.Text(
            synonyms_card.content, height=5, width=30,
            font=FONTS.get_font("body"),
            wrap=tk.WORD,
        )
        self.synonyms_text.pack(fill=tk.X, pady=4)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Generate Thematic Map", icon="🗺️",
            command=self._generate_map, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        ThemedButton(
            btn_frame, text="Export Results", style="secondary",
            command=self._export_results, theme=self.theme_name,
        ).pack(fill=tk.X, pady=(8, 0))
    
    def _create_results(self):
        """Create the results panel."""
        self.results_card = tk.Frame(
            self.results_frame, bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Header
        header = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header, text="Thematic Map Results",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            padx=12, pady=8,
        ).pack(side=tk.LEFT)
        
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.results_card)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Map tab
        self.map_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.map_frame, text="  🗺️ Map  ")
        
        # Clusters tab
        self.clusters_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.clusters_frame, text="  📊 Clusters  ")
        
        # Items tab
        self.items_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.items_frame, text="  📝 Items  ")

        

        # Info tab

        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])

        self.notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        # Set results_content for base class compatibility
        self.results_content = self.map_frame
        
        self._show_placeholder("Configure options and click 'Generate Thematic Map'")
    
    def _show_placeholder(self, message: str):
        """Show message in results area."""
        for frame in [self.map_frame, self.clusters_frame, self.items_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
            tk.Label(
                frame, text=message,
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            ).pack(expand=True)
    
    def _show_loading(self, message: str):
        """Show loading message."""
        self._show_placeholder(f"⏳ {message}")
    
    def _generate_map(self):
        """Generate the thematic map using biblium core."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_NETWORKX:
            messagebox.showerror("Missing Library", "NetworkX is required for thematic map analysis.")
            return
        
        # Emit start event
        event_bus.emit(EventBus.ANALYSIS_STARTED, {"name": self.title})
        self._is_running = True
        self._cancel_requested = False
        
        self._show_loading("Generating thematic map...")
        
        # Get options
        field_type = self.field_type.get().lower().replace(" ", "_")
        min_occ = self.min_occurrences.get()
        top_n = self.top_n.get()
        
        # Parse synonyms
        synonyms = {}
        synonyms_text = self.synonyms_text.get("1.0", tk.END).strip()
        if synonyms_text:
            for line in synonyms_text.split("\n"):
                if "=" in line:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        value = parts[1].strip()
                        synonyms[key] = value
        
        def do_generate():
            try:
                from biblium import utilsbib
                
                # Get separator
                separator = getattr(self.bib, 'default_separator', '; ')
                db = getattr(self.bib, 'db', None)
                
                # Call biblium's build_thematic_map
                result = utilsbib.build_thematic_map(
                    df=self.bib.df,
                    field=field_type,
                    min_occurrences=min_occ,
                    top_n=top_n,
                    separator=separator,
                    synonyms=synonyms if synonyms else None,
                    db=db,
                )
                
                self.after(0, lambda r=result: self._on_generate_success(r))
                
            except Exception as e:
                import traceback
                error_msg = str(e)
                traceback.print_exc()
                self.after(0, lambda msg=error_msg: self._on_generate_error(msg))
        
        threading.Thread(target=do_generate, daemon=True).start()
    
    def _on_generate_success(self, result: Dict):
        """Handle successful generation."""
        self._cluster_df = result["clusters_df"]
        self._items_df = result["items_df"]
        self._graph = result.get("graph")
        
        # Show map
        self._show_map(result)
        
        # Show clusters table
        self._show_clusters_table(result["clusters_df"])
        
        # Show items table
        self._show_items_table(result["items_df"])
        
        self._is_running = False
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": self.title})
    
    def _on_generate_error(self, error: str):
        """Handle generation error."""
        self._is_running = False
        self._show_placeholder(f"❌ Error: {error}")
        event_bus.emit(EventBus.ERROR_OCCURRED, {"message": error})
    
    def _show_map(self, result: Dict):
        """Display the thematic map (strategic diagram) as static image."""
        for widget in self.map_frame.winfo_children():
            widget.destroy()
        
        if not HAS_MATPLOTLIB:
            tk.Label(
                self.map_frame, text="Matplotlib required",
                bg=self.theme["bg_card"],
            ).pack(expand=True)
            return
        
        cluster_df = result["clusters_df"]
        
        if len(cluster_df) == 0:
            tk.Label(
                self.map_frame, text="No clusters found",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        try:
            from PIL import Image, ImageTk
            
            # Create larger figure to accommodate labels
            fig = Figure(figsize=(14, 11), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            
            self._current_fig = fig
            
            # Get centrality and density
            x = cluster_df["Centrality"].values
            y = cluster_df["Density"].values
            labels = cluster_df["Label"].values
            
            # Bubble sizes
            size_by = self.bubble_size_by.get()
            if size_by == "Occurrences":
                sizes = cluster_df["Occurrences"].values
            elif size_by == "Items":
                sizes = cluster_df["Items"].values
            else:
                sizes = np.array([100] * len(cluster_df))
            
            # Normalize sizes
            sizes = np.array(sizes, dtype=float)
            if sizes.max() > sizes.min():
                sizes = 200 + 1000 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
            else:
                sizes = np.array([600] * len(sizes))
            
            # Get medians for quadrant lines
            x_median = cluster_df["Centrality_Median"].iloc[0] if "Centrality_Median" in cluster_df.columns else np.median(x)
            y_median = cluster_df["Density_Median"].iloc[0] if "Density_Median" in cluster_df.columns else np.median(y)
            
            # Calculate plot bounds with more padding for labels
            x_range = max(x) - min(x) if max(x) != min(x) else 0.1
            y_range = max(y) - min(y) if max(y) != min(y) else 0.1
            x_min = min(x) - 0.20 * x_range
            x_max = max(x) + 0.20 * x_range
            y_min = min(y) - 0.20 * y_range
            y_max = max(y) + 0.25 * y_range  # Extra padding on top for labels
            
            # Quadrant colors (light backgrounds)
            colors_quad = {
                "Motor": "#c8e6c9",      # Green - top right
                "Basic": "#ffe0b2",       # Orange - top left
                "Emerging": "#bbdefb",    # Blue - bottom left  
                "Niche": "#f8bbd9",       # Pink - bottom right
            }
            
            # Draw quadrant rectangles
            ax.add_patch(Rectangle((x_median, y_median), x_max - x_median, y_max - y_median, 
                                   facecolor=colors_quad["Motor"], alpha=0.4, zorder=0))
            ax.add_patch(Rectangle((x_min, y_median), x_median - x_min, y_max - y_median,
                                   facecolor=colors_quad["Basic"], alpha=0.4, zorder=0))
            ax.add_patch(Rectangle((x_min, y_min), x_median - x_min, y_median - y_min,
                                   facecolor=colors_quad["Emerging"], alpha=0.4, zorder=0))
            ax.add_patch(Rectangle((x_median, y_min), x_max - x_median, y_median - y_min,
                                   facecolor=colors_quad["Niche"], alpha=0.4, zorder=0))
            
            # Draw quadrant lines
            ax.axhline(y=y_median, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
            ax.axvline(x=x_median, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
            
            # Quadrant labels
            if self.show_quadrant_labels.get():
                font_props = {'fontsize': 10, 'style': 'italic', 'fontweight': 'bold'}
                ax.text(x_max - 0.02 * x_range, y_max - 0.02 * y_range, 
                       "MOTOR THEMES", ha='right', va='top', color='#2e7d32', **font_props)
                ax.text(x_min + 0.02 * x_range, y_max - 0.02 * y_range,
                       "BASIC THEMES", ha='left', va='top', color='#e65100', **font_props)
                ax.text(x_min + 0.02 * x_range, y_min + 0.02 * y_range,
                       "EMERGING/DECLINING", ha='left', va='bottom', color='#1565c0', **font_props)
                ax.text(x_max - 0.02 * x_range, y_min + 0.02 * y_range,
                       "NICHE THEMES", ha='right', va='bottom', color='#ad1457', **font_props)
            
            # Plot bubbles with colors based on quadrant
            bubble_colors = []
            color_map = {"Motor": "#4caf50", "Basic": "#ff9800", "Emerging/Declining": "#2196f3", "Niche": "#e91e63"}
            for _, row in cluster_df.iterrows():
                quadrant = row.get("Quadrant", "Motor")
                bubble_colors.append(color_map.get(quadrant, "#9e9e9e"))
            
            scatter = ax.scatter(x, y, s=sizes, c=bubble_colors, alpha=0.7, 
                                edgecolors='white', linewidth=2, zorder=2)
            
            # Add labels - show full labels without truncation
            if self.show_labels.get():
                for i, label in enumerate(labels):
                    # Use full label, wrap if needed by splitting on comma
                    display_label = label.replace(", ", "\n")
                    ax.annotate(display_label, (x[i], y[i]), fontsize=9, ha='center', va='bottom',
                               xytext=(0, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5),
                               zorder=3)
            
            ax.set_xlabel("Centrality (External Links → Theme Importance)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Density (Internal Cohesion → Theme Development)", fontsize=11, fontweight='bold')
            ax.set_title("Thematic Map - Strategic Diagram", fontsize=14, fontweight='bold', pad=15)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle=':')
            
            fig.tight_layout()
            
            # Render to static image using ScaledImageFrame
            from biblium.gui.widgets.plots import ScaledImageFrame
            
            scaled_frame = ScaledImageFrame(
                self.map_frame, 
                theme=self.theme_name,
                maintain_aspect=True,
                max_scale=1.5
            )
            scaled_frame.pack(fill=tk.BOTH, expand=True)
            scaled_frame.set_image_from_figure(fig, dpi=100)
            
            self._scaled_frame = scaled_frame
            
            # Right-click menu for saving
            def show_menu(event):
                menu = tk.Menu(scaled_frame, tearoff=0)
                menu.add_command(label="📄 Add to Report", command=self._add_plot_to_report)
                menu.add_separator()
                menu.add_command(label="💾 Save as PNG...", command=lambda: self._save_plot("png"))
                menu.add_command(label="💾 Save as PDF...", command=lambda: self._save_plot("pdf"))
                menu.add_command(label="💾 Save as SVG...", command=lambda: self._save_plot("svg"))
                menu.tk_popup(event.x_root, event.y_root)
            scaled_frame.bind("<Button-3>", show_menu)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(
                self.map_frame,
                text=f"Plot error: {e}",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["danger"],
            ).pack(expand=True)
    
    def _save_plot(self, fmt: str):
        """Save plot to file."""
        if not self._current_fig:
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} files", f"*.{fmt}")],
        )
        if filepath:
            try:
                self._current_fig.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Saved", f"Saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def _add_plot_to_report(self):
        """Add current plot to report queue."""
        if not self._current_fig:
            messagebox.showinfo("No Plot", "No plot to add to report.")
            return
        
        try:
            from biblium.gui.core.state import report_queue
            
            plot_title = "Thematic Map"
            if self._current_fig.axes:
                plot_title = self._current_fig.axes[0].get_title() or "Thematic Map"
            
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
    
    def _show_clusters_table(self, df: pd.DataFrame):
        """Display clusters table."""
        for widget in self.clusters_frame.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            tk.Label(
                self.clusters_frame, text="No cluster data",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            ).pack(expand=True)
            return
        
        # Select columns to display
        display_cols = ["Cluster", "Label", "Quadrant", "Items", "Occurrences", "Centrality", "Density", "Top_Items"]
        display_df = df[[c for c in display_cols if c in df.columns]].copy()
        
        table = DataTable(self.clusters_frame, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(display_df)
    
    def _show_items_table(self, df: pd.DataFrame):
        """Display items table."""
        for widget in self.items_frame.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            tk.Label(
                self.items_frame, text="No items data",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            ).pack(expand=True)
            return
        
        table = DataTable(self.items_frame, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)
    
    def _export_results(self):
        """Export thematic map results."""
        if self._cluster_df is None:
            messagebox.showinfo("No Data", "Generate a thematic map first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")],
            title="Export Thematic Map Data",
        )
        
        if filename:
            try:
                if filename.endswith(".csv"):
                    self._cluster_df.to_csv(filename, index=False)
                else:
                    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                        self._cluster_df.to_excel(writer, sheet_name="Clusters", index=False)
                        if self._items_df is not None:
                            self._items_df.to_excel(writer, sheet_name="Items", index=False)
                messagebox.showinfo("Success", f"Exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
THEMATIC MAP
════════════

Strategic diagram of research themes (Callon's method).

QUADRANT STRUCTURE
──────────────────
Based on centrality (X) and density (Y):

• MOTOR THEMES (Upper-Right)
  High centrality + High density
  Core, well-developed topics
  Drive the field forward
  
• NICHE THEMES (Upper-Left)
  Low centrality + High density
  Specialized, peripheral
  Developed but isolated
  
• BASIC THEMES (Lower-Right)
  High centrality + Low density
  Important but undeveloped
  Transversal, foundational
  
• EMERGING/DECLINING (Lower-Left)
  Low centrality + Low density
  New or fading topics
  Weakly developed, marginal

METRICS
───────
• Centrality: External connections
  How connected to other themes
  
• Density: Internal cohesion
  How developed/mature the theme is

BUBBLE SIZE
───────────
• By document count
• By citation count
• By occurrence frequency

INTERPRETATION
──────────────
• Upper-right: Focus areas
• Upper-left: Specializations
• Lower-right: Foundations
• Lower-left: Monitor for emergence
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
        if self._current_fig:
            try:
                plt.close(self._current_fig)
            except:
                pass
        self._current_fig = None
        self._photo_image = None
        super().destroy()
