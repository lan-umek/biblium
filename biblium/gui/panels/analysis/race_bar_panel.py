# -*- coding: utf-8 -*-
"""
Race Bar Animation Panel
========================
GUI panel for creating animated race bar charts showing cumulative counts over time.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, Optional, Any
import pandas as pd
import os
import threading

from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, StatsCard, CardGrid
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledEntry
from biblium.gui.widgets.plots import add_plot_context_menu, make_canvas_resizable
from biblium.gui.widgets.buttons import ActionButton
from biblium.gui.widgets.plots import PlotFrame
from biblium.gui.config import FONTS, get_theme
from biblium.gui.core.events import EventBus

# Import race bar functions
try:
    from biblium.bibplot_modules.race_bar import RaceBarMixin, create_race_bar_from_dataframe
    HAS_RACEBAR = True
except ImportError:
    HAS_RACEBAR = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.animation import FuncAnimation
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

event_bus = EventBus()


class RaceBarPanel(BasePanel):
    """
    Panel for creating race bar animations.
    
    Shows cumulative counts over time for various entities:
    - Sources/Journals
    - Authors
    - Keywords (Author/Index)
    - Countries
    - Affiliations
    - Custom columns
    """
    
    title = "🏁 Race Bar Animation"
    description = "Create animated bar chart races showing cumulative counts over time"
    
    # Predefined entity types with their settings
    # separator=None means single value (no splitting)
    # We'll detect the actual separator based on the data
    ENTITY_TYPES = {
        "Sources/Journals": {
            "columns": ["Source title", "Source", "Journal", "primary_location.source.display_name"],
            "separator": None,  # Sources are single-valued
            "title": "Top Sources by Cumulative Publications"
        },
        "Authors": {
            "columns": ["Authors", "Author", "authorships.author.display_name"],
            "separator": "; ",  # Will be auto-detected
            "title": "Top Authors by Cumulative Publications"
        },
        "Author Keywords": {
            "columns": ["Author Keywords", "Keywords", "keywords.display_name"],
            "separator": "; ",  # Will be auto-detected
            "title": "Top Author Keywords by Cumulative Frequency"
        },
        "Index Keywords": {
            "columns": ["Index Keywords", "Indexed Keywords"],
            "separator": "; ",  # Will be auto-detected
            "title": "Top Index Keywords by Cumulative Frequency"
        },
        "Countries": {
            "columns": ["Countries", "Country", "Affiliations Country", "authorships.countries"],
            "separator": "; ",  # Will be auto-detected
            "title": "Top Countries by Cumulative Publications"
        },
        "Affiliations": {
            "columns": ["Affiliations", "Affiliation", "Institution", "authorships.institutions"],
            "separator": "; ",  # Will be auto-detected
            "title": "Top Affiliations by Cumulative Publications"
        },
        "Custom Column": {
            "columns": [],
            "separator": "; ",
            "title": "Cumulative Counts Over Time"
        }
    }
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self.bib = bib
        self._current_animation = None
        self._preview_frame = None
        self._animation_running = False
        self._export_thread = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create options panel."""
        super()._create_options()
        
        # Entity Selection Card
        entity_card = Card(self.options_content, title="📊 Entity Selection", theme=self.theme_name)
        entity_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Entity type dropdown
        self.entity_var = tk.StringVar(value="Sources/Journals")
        entity_combo = LabeledCombobox(
            entity_card.content, label="Entity type:",
            values=list(self.ENTITY_TYPES.keys()),
            variable=self.entity_var, theme=self.theme_name,
        )
        entity_combo.pack(fill=tk.X, pady=4)
        self.entity_var.trace_add("write", self._on_entity_change)
        
        # Custom column selection (for "Custom Column" option)
        self.custom_col_var = tk.StringVar()
        self.custom_combo = LabeledCombobox(
            entity_card.content, label="Custom column:",
            values=[], variable=self.custom_col_var, theme=self.theme_name,
        )
        self.custom_combo.pack(fill=tk.X, pady=4)
        self.custom_combo.pack_forget()  # Hide initially
        
        # Separator entry
        sep_frame = tk.Frame(entity_card.content, bg=self.theme["bg_card"])
        sep_frame.pack(fill=tk.X, pady=4)
        tk.Label(sep_frame, text="Separator:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.separator_var = tk.StringVar(value="")  # Empty = no separator for Sources
        self.sep_entry = tk.Entry(sep_frame, textvariable=self.separator_var,
                            font=FONTS.get_font("body"), width=8)
        self.sep_entry.pack(side=tk.LEFT, padx=8)
        self.sep_hint = tk.Label(sep_frame, text="(empty = single values)", font=FONTS.get_font("small"),
                bg=self.theme["bg_card"], fg=self.theme["text_muted"])
        self.sep_hint.pack(side=tk.LEFT)
        
        # Year column
        self.year_col_var = tk.StringVar(value="Year")
        year_combo = LabeledCombobox(
            entity_card.content, label="Year column:",
            values=["Year"], variable=self.year_col_var, theme=self.theme_name,
        )
        year_combo.pack(fill=tk.X, pady=4)
        self.year_combo = year_combo
        
        # Animation Settings Card
        anim_card = Card(self.options_content, title="🎬 Animation Settings", theme=self.theme_name)
        anim_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Top N
        topn_frame = tk.Frame(anim_card.content, bg=self.theme["bg_card"])
        topn_frame.pack(fill=tk.X, pady=4)
        tk.Label(topn_frame, text="Top N items:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.topn_var = tk.StringVar(value="15")
        topn_entry = tk.Entry(topn_frame, textvariable=self.topn_var,
                             font=FONTS.get_font("body"), width=8)
        topn_entry.pack(side=tk.LEFT, padx=8)
        
        # Duration
        dur_frame = tk.Frame(anim_card.content, bg=self.theme["bg_card"])
        dur_frame.pack(fill=tk.X, pady=4)
        tk.Label(dur_frame, text="Duration (seconds):", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="15")
        dur_entry = tk.Entry(dur_frame, textvariable=self.duration_var,
                            font=FONTS.get_font("body"), width=8)
        dur_entry.pack(side=tk.LEFT, padx=8)
        
        # FPS
        fps_frame = tk.Frame(anim_card.content, bg=self.theme["bg_card"])
        fps_frame.pack(fill=tk.X, pady=4)
        tk.Label(fps_frame, text="FPS:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="20")  # Lower default for faster export
        fps_entry = tk.Entry(fps_frame, textvariable=self.fps_var,
                            font=FONTS.get_font("body"), width=8)
        fps_entry.pack(side=tk.LEFT, padx=8)
        
        # Interpolation frames
        interp_frame = tk.Frame(anim_card.content, bg=self.theme["bg_card"])
        interp_frame.pack(fill=tk.X, pady=4)
        tk.Label(interp_frame, text="Smooth frames:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.interp_var = tk.StringVar(value="5")  # Lower default for faster export
        interp_entry = tk.Entry(interp_frame, textvariable=self.interp_var,
                               font=FONTS.get_font("body"), width=8)
        interp_entry.pack(side=tk.LEFT, padx=8)
        
        # Year Range Card
        range_card = Card(self.options_content, title="📅 Year Range", theme=self.theme_name)
        range_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Min year
        min_frame = tk.Frame(range_card.content, bg=self.theme["bg_card"])
        min_frame.pack(fill=tk.X, pady=4)
        tk.Label(min_frame, text="Start year:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.min_year_var = tk.StringVar(value="")
        min_entry = tk.Entry(min_frame, textvariable=self.min_year_var,
                            font=FONTS.get_font("body"), width=8)
        min_entry.pack(side=tk.LEFT, padx=8)
        tk.Label(min_frame, text="(empty = auto)", font=FONTS.get_font("small"),
                bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(side=tk.LEFT)
        
        # Max year
        max_frame = tk.Frame(range_card.content, bg=self.theme["bg_card"])
        max_frame.pack(fill=tk.X, pady=4)
        tk.Label(max_frame, text="End year:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.max_year_var = tk.StringVar(value="")
        max_entry = tk.Entry(max_frame, textvariable=self.max_year_var,
                            font=FONTS.get_font("body"), width=8)
        max_entry.pack(side=tk.LEFT, padx=8)
        tk.Label(max_frame, text="(empty = auto)", font=FONTS.get_font("small"),
                bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(side=tk.LEFT)
        
        # Display Options Card
        display_card = Card(self.options_content, title="🎨 Display Options", theme=self.theme_name)
        display_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_values_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show value labels",
            variable=self.show_values_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.dark_mode_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            display_card.content, label="Dark mode",
            variable=self.dark_mode_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Custom title
        title_frame = tk.Frame(display_card.content, bg=self.theme["bg_card"])
        title_frame.pack(fill=tk.X, pady=4)
        tk.Label(title_frame, text="Title:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.title_var = tk.StringVar(value="")
        title_entry = tk.Entry(title_frame, textvariable=self.title_var,
                              font=FONTS.get_font("body"), width=30)
        title_entry.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        
        # Action Buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Preview", icon="👁️",
            command=self._preview_animation, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        ActionButton(
            btn_frame, text="Export Animation", icon="💾",
            command=self._generate_animation, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook for results and info
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="🏁 Animation")
        
        # Info tab
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
        # Show placeholder in results tab
        tk.Label(
            self.results_tab,
            text="🏁 Race Bar Animation\n\n"
                 "Create animated bar chart races showing rankings over time.\n\n"
                 "Select entity type and click 'Preview' or 'Export Animation'.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _detect_separator(self, column: str) -> Optional[str]:
        """Detect the separator used in a column by examining the data."""
        if self.bib is None or not hasattr(self.bib, 'df'):
            return "; "
        
        df = self.bib.df
        if column not in df.columns:
            return "; "
        
        # Sample some non-null values
        sample = df[column].dropna().head(20)
        if len(sample) == 0:
            return "; "
        
        # Check for common separators
        separators = ["|", "; ", ", ", ";"]
        sep_counts = {sep: 0 for sep in separators}
        
        for val in sample:
            val_str = str(val)
            for sep in separators:
                if sep in val_str:
                    sep_counts[sep] += val_str.count(sep)
        
        # Find the most common separator
        best_sep = max(sep_counts, key=sep_counts.get)
        if sep_counts[best_sep] > 0:
            return best_sep
        
        # No separator found - probably single values
        return None
    
    def _on_entity_change(self, *args):
        """Handle entity type change."""
        entity = self.entity_var.get()
        settings = self.ENTITY_TYPES.get(entity, {})
        
        if entity == "Custom Column":
            self.custom_combo.pack(fill=tk.X, pady=4, after=self.custom_combo.master.winfo_children()[0])
            # Populate with available columns
            if self.bib is not None and hasattr(self.bib, 'df'):
                cols = list(self.bib.df.columns)
                self.custom_combo.combobox['values'] = cols
            self.separator_var.set("; ")
            self.sep_hint.config(text="(for multi-value fields)")
        else:
            self.custom_combo.pack_forget()
            
            # Find the actual column in data
            column = self._find_column(settings.get("columns", []))
            
            if column and settings.get("separator") is not None:
                # Detect separator from actual data
                detected_sep = self._detect_separator(column)
                if detected_sep is None:
                    self.separator_var.set("")
                    self.sep_hint.config(text="(single values - no splitting)")
                else:
                    self.separator_var.set(detected_sep)
                    self.sep_hint.config(text=f"(detected: '{detected_sep}')")
            elif settings.get("separator") is None:
                # Sources - explicitly no separator
                self.separator_var.set("")
                self.sep_hint.config(text="(single values - no splitting)")
    
    def _find_column(self, candidates: list) -> Optional[str]:
        """Find the first matching column from candidates."""
        if self.bib is None or not hasattr(self.bib, 'df'):
            return None
        
        df_cols = list(self.bib.df.columns)
        df_cols_lower = [c.lower() for c in df_cols]
        
        for candidate in candidates:
            # Exact match
            if candidate in df_cols:
                return candidate
            # Case-insensitive match
            candidate_lower = candidate.lower()
            if candidate_lower in df_cols_lower:
                idx = df_cols_lower.index(candidate_lower)
                return df_cols[idx]
        
        return None
    
    def _get_settings(self) -> Dict:
        """Get current settings for animation."""
        entity = self.entity_var.get()
        settings = self.ENTITY_TYPES.get(entity, {})
        
        # Determine column
        if entity == "Custom Column":
            column = self.custom_col_var.get()
        else:
            column = self._find_column(settings.get("columns", []))
        
        # Get separator - empty string means None (no splitting)
        sep = self.separator_var.get()
        separator = sep if sep.strip() else None
        
        # Get numeric values
        try:
            top_n = int(self.topn_var.get())
        except ValueError:
            top_n = 15
        
        try:
            duration = float(self.duration_var.get())
        except ValueError:
            duration = 15.0
        
        try:
            fps = int(self.fps_var.get())
        except ValueError:
            fps = 20
        
        try:
            interp = int(self.interp_var.get())
        except ValueError:
            interp = 5
        
        # Year range
        min_year = None
        max_year = None
        if self.min_year_var.get().strip():
            try:
                min_year = int(self.min_year_var.get())
            except ValueError:
                pass
        if self.max_year_var.get().strip():
            try:
                max_year = int(self.max_year_var.get())
            except ValueError:
                pass
        
        # Title
        title = self.title_var.get().strip()
        if not title:
            title = settings.get("title", f"Cumulative {column} Over Time")
        
        return {
            "column": column,
            "year_column": self.year_col_var.get(),
            "separator": separator,
            "top_n": top_n,
            "min_year": min_year,
            "max_year": max_year,
            "title": title,
            "duration": duration,
            "fps": fps,
            "interpolate_frames": interp,
            "show_value_labels": self.show_values_var.get(),
            "dark_mode": self.dark_mode_var.get(),
        }
    
    def _preview_animation(self):
        """Preview the animation (shows final state)."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_RACEBAR:
            messagebox.showerror("Error", "Race bar module not available.")
            return
        
        settings = self._get_settings()
        
        if not settings["column"]:
            messagebox.showwarning("No Column", "Please select a column for the animation.")
            return
        
        self._show_loading("Generating preview...")
        
        try:
            # Create a wrapper class to use the mixin
            class Wrapper(RaceBarMixin):
                def __init__(self, df):
                    self.df = df
                    self.default_separator = "; "
                
                def _get_column(self, name, required=False):
                    if name in self.df.columns:
                        return name
                    for col in self.df.columns:
                        if col.lower() == name.lower():
                            return col
                    return None
            
            wrapper = Wrapper(self.bib.df)
            
            # Prepare data
            race_df = wrapper._prepare_race_data(
                column=settings["column"],
                year_column=settings["year_column"],
                separator=settings["separator"],
                top_n=settings["top_n"],
                min_year=settings["min_year"],
                max_year=settings["max_year"],
            )
            
            self._display_preview(race_df, settings)
            
        except Exception as e:
            self._show_error(str(e))
    
    def _display_preview(self, race_df: pd.DataFrame, settings: Dict):
        """Display preview of the animation."""
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        
        # Info cards
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 8))
        
        years = race_df.index.tolist()
        items = race_df.columns.tolist()
        total_frames = (len(years) - 1) * settings['interpolate_frames'] + 1
        
        grid.add_card(StatsCard(grid, "Years", f"{min(years)} - {max(years)}", "📅", self.theme_name))
        grid.add_card(StatsCard(grid, "Items", f"{len(items)}", "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "Duration", f"{settings['duration']}s", "⏱️", self.theme_name))
        grid.add_card(StatsCard(grid, "Frames", f"{total_frames}", "🎞️", self.theme_name))
        
        # Create preview plot (last frame showing final state)
        plot_frame = PlotFrame(self.results_tab, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig, ax = plot_frame.get_figure()
        
        # Get final year data
        final_year = max(years)
        final_data = race_df.loc[final_year].sort_values(ascending=True)
        
        # Plot horizontal bars
        from biblium.colors import get_colors
        colors = get_colors(len(final_data))
        
        y_pos = range(len(final_data))
        bars = ax.barh(y_pos, final_data.values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(x)[:40] for x in final_data.index], fontsize=9)
        ax.set_xlabel("Cumulative Count", fontsize=11)
        ax.set_title(f"{settings['title']} (Final: {final_year})", fontsize=12)
        
        # Add value labels
        if settings["show_value_labels"]:
            for bar, val in zip(bars, final_data.values):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                       f" {int(val)}", va='center', fontsize=8)
        
        fig.subplots_adjust(left=0.3, right=0.95, top=0.92, bottom=0.1)
        plot_frame.set_preserve_margins(True)
        plot_frame.refresh()
        
        # Export button
        btn_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=8)
        
        ActionButton(
            btn_frame, text="Export Animation", icon="💾",
            command=self._generate_animation, theme=self.theme_name,
        ).pack(side=tk.LEFT, padx=4)
        
        # Store data for export
        self._preview_data = race_df
        self._preview_settings = settings
    
    def _generate_animation(self):
        """Generate and save the animation."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_RACEBAR:
            messagebox.showerror("Error", "Race bar module not available.")
            return
        
        settings = self._get_settings()
        
        if not settings["column"]:
            messagebox.showwarning("No Column", "Please select a column for the animation.")
            return
        
        # Ask for output format and path
        format_dialog = tk.Toplevel(self)
        format_dialog.title("Export Animation")
        format_dialog.geometry("400x280")
        format_dialog.transient(self)
        format_dialog.grab_set()
        
        tk.Label(format_dialog, text="Select output format:", font=FONTS.get_font("body")).pack(pady=10)
        
        format_var = tk.StringVar(value="mp4")
        formats = [
            ("MP4 Video (requires ffmpeg)", "mp4"), 
            ("GIF Animation (slower export)", "gif"), 
            ("HTML (interactive, large file)", "html")
        ]
        
        for text, val in formats:
            tk.Radiobutton(format_dialog, text=text, variable=format_var, value=val,
                          font=FONTS.get_font("body")).pack(anchor=tk.W, padx=20)
        
        # Tips
        tk.Label(format_dialog, text="\nTips for faster export:", 
                font=FONTS.get_font("body_bold")).pack(anchor=tk.W, padx=20)
        tk.Label(format_dialog, text="• Lower FPS (15-20)\n• Fewer smooth frames (3-5)\n• Shorter duration",
                font=FONTS.get_font("small"), justify=tk.LEFT).pack(anchor=tk.W, padx=30)
        
        def do_export():
            fmt = format_var.get()
            format_dialog.destroy()
            
            # Get save path
            ext_map = {"mp4": ".mp4", "gif": ".gif", "html": ".html"}
            filetypes = {
                "mp4": [("MP4 Video", "*.mp4")],
                "gif": [("GIF Animation", "*.gif")],
                "html": [("HTML File", "*.html")],
            }
            
            filename = filedialog.asksaveasfilename(
                defaultextension=ext_map[fmt],
                filetypes=filetypes[fmt],
                title="Save Animation"
            )
            
            if not filename:
                return
            
            self._export_animation(settings, filename, fmt)
        
        btn_frame = tk.Frame(format_dialog)
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text="Export", command=do_export,
                 font=FONTS.get_font("body"), width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=format_dialog.destroy,
                 font=FONTS.get_font("body"), width=10).pack(side=tk.LEFT, padx=5)
    
    def _export_animation(self, settings: Dict, output_path: str, output_format: str):
        """Export the animation to file in a background thread."""
        
        # Show progress
        self._show_loading(
            f"Exporting {output_format.upper()} animation...\n\n"
            "This may take several minutes.\n"
            "Please wait..."
        )
        self.update_idletasks()
        
        def do_export():
            try:
                # Use the standalone function
                result = create_race_bar_from_dataframe(
                    df=self.bib.df,
                    column=settings["column"],
                    year_column=settings["year_column"],
                    separator=settings["separator"],
                    top_n=settings["top_n"],
                    min_year=settings["min_year"],
                    max_year=settings["max_year"],
                    title=settings["title"],
                    duration=settings["duration"],
                    fps=settings["fps"],
                    interpolate_frames=settings["interpolate_frames"],
                    show_value_labels=settings["show_value_labels"],
                    dark_mode=settings["dark_mode"],
                    output_path=output_path,
                    output_format=output_format,
                )
                
                # Schedule UI update on main thread
                self.after(0, lambda: self._export_complete(output_path, None))
                
            except Exception as e:
                self.after(0, lambda: self._export_complete(None, str(e)))
        
        # Run export in background thread
        self._export_thread = threading.Thread(target=do_export, daemon=True)
        self._export_thread.start()
    
    def _export_complete(self, output_path: Optional[str], error: Optional[str]):
        """Called when export completes."""
        if error:
            self._show_error(f"Export failed:\n{error}")
            messagebox.showerror("Export Failed", error)
        else:
            # Show success and preview again
            if hasattr(self, '_preview_data') and hasattr(self, '_preview_settings'):
                self._display_preview(self._preview_data, self._preview_settings)
            messagebox.showinfo("Success", f"Animation saved to:\n{output_path}")
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading indicator."""
        try:
            plt.close('all')
        except:
            pass
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        frame.pack(expand=True)
        tk.Label(frame, text="⏳", font=("Segoe UI", 32), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]).pack(pady=(20, 10))
        tk.Label(frame, text=message, font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"], justify=tk.CENTER).pack()
    
    def _show_error(self, message: str):
        """Show error message."""
        try:
            plt.close('all')
        except:
            pass
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        error_color = self.theme.get("error", self.theme.get("danger", "#e74c3c"))
        tk.Label(self.results_tab, text=f"❌ Error\n\n{message}", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=error_color, justify=tk.CENTER,
                wraplength=400).pack(expand=True)
    
    def set_bib(self, bib):
        """Set the bibliometric data object."""
        self.bib = bib
        
        # Update column dropdowns
        if bib is not None and hasattr(bib, 'df'):
            cols = list(bib.df.columns)
            self.custom_combo.combobox['values'] = cols
            self.year_combo.combobox['values'] = cols
            
            # Try to find year column
            for col in cols:
                if col.lower() in ['year', 'publication year', 'pub year', 'publication_year']:
                    self.year_col_var.set(col)
                    break
            
            # Re-detect separator for current entity type
            self._on_entity_change()
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
ANIMATED BAR CHART RACE
═══════════════════════

Animated visualization of rankings over time.

WHAT IT SHOWS
─────────────
• Top N entities per year
• Ranking changes animated
• Rise and fall of leaders
• Dynamic competition

ENTITY TYPES
────────────
• Authors over time
• Journals over time
• Keywords over time
• Countries over time

METRICS
───────
• Cumulative publications
• Cumulative citations
• Annual values

ANIMATION CONTROLS
──────────────────
• Speed (frames per second)
• Duration per year
• Transition smoothness
• Bar colors

OUTPUT
──────
• Animated GIF
• MP4 video
• Frame-by-frame images

CUSTOMIZATION
─────────────
• Number of bars (Top N)
• Color scheme
• Title and labels
• Year range
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

    def destroy(self):
        """Clean up resources."""
        try:
            plt.close('all')
        except:
            pass
        self._current_animation = None
        super().destroy()
