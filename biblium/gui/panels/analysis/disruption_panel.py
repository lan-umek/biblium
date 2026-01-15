# -*- coding: utf-8 -*-
"""
Disruption Index Panel
======================
Panel for computing and visualizing Disruption Index metrics.

@author: Claude (Anthropic) for Lan.Umek
@version: 2.7.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DisruptionIndexPanel(BasePanel):
    """Panel for Disruption Index analysis."""
    
    title = "Disruption Index"
    icon = "💥"
    description = "Measure consolidating vs. disruptive research impact"
    requires_data = True
    
    ENTITY_TYPES = [
        ("documents", "Documents", "Document-level disruption"),
        ("sources", "Sources/Journals", "Average disruption by journal"),
        ("authors", "Authors", "Average disruption by author"),
        ("countries", "Countries", "Average disruption by country"),
        ("affiliations", "Affiliations", "Average disruption by institution"),
        ("years", "Years", "Disruption trend over time"),
    ]
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        if not self.bib:
            self._show_no_data_message()
            return
        
        # Warning about OpenAlex
        warning_frame = tk.Frame(self.options_content, bg="#fff3cd", relief=tk.FLAT, bd=1)
        warning_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        warning_inner = tk.Frame(warning_frame, bg="#fff3cd", padx=8, pady=6)
        warning_inner.pack(fill=tk.X)
        
        tk.Label(
            warning_inner, text="⚠️ Best with OpenAlex data",
            font=FONTS.get_font("body_bold"), bg="#fff3cd", fg="#856404",
        ).pack(anchor=tk.W)
        
        tk.Label(
            warning_inner, 
            text="CD Index requires internal citations.\nOpenAlex provides matching IDs.\nScopus/WoS may show limited results.",
            font=FONTS.get_font("small"), bg="#fff3cd", fg="#856404",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        
        # Analysis Type Card
        type_card = Card(self.options_content, title="📊 Analysis Type", theme=self.theme_name)
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.entity_var = tk.StringVar(value="documents")
        
        for entity_id, entity_name, entity_desc in self.ENTITY_TYPES:
            frame = tk.Frame(type_card.content, bg=self.theme["bg_card"])
            frame.pack(fill=tk.X, pady=2)
            
            rb = tk.Radiobutton(
                frame, text=entity_name, variable=self.entity_var, value=entity_id,
                command=self._on_entity_changed,
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
                font=FONTS.get_font("body"),
            )
            rb.pack(side=tk.LEFT)
            
            tk.Label(frame, text=f"  ({entity_desc})", font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(side=tk.LEFT)
        
        # Parameters Card
        params_card = Card(self.options_content, title="⚙️ Parameters", theme=self.theme_name)
        params_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.min_docs = LabeledSpinbox(
            params_card.content, label="Min Documents:", from_=1, to=100, default=3,
            theme=self.theme_name, label_width=15,
            tooltip="Minimum documents for entity inclusion"
        )
        self.min_docs.pack(fill=tk.X, pady=4)
        
        self.top_n = LabeledSpinbox(
            params_card.content, label="Top N:", from_=5, to=100, default=20,
            theme=self.theme_name, label_width=15,
            tooltip="Number of top entities to display"
        )
        self.top_n.pack(fill=tk.X, pady=4)
        
        self.add_to_df_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            params_card.content, label="Add disruption to dataset",
            variable=self.add_to_df_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # View Data link
        view_data_btn = tk.Label(
            params_card.content, text="📋 View Data →", 
            font=FONTS.get_font("body"), bg=self.theme["bg_card"], 
            fg=self.theme["accent_primary"], cursor="hand2",
        )
        view_data_btn.pack(anchor=tk.W, pady=(8, 4))
        view_data_btn.bind("<Button-1>", lambda e: self._view_data())
        view_data_btn.bind("<Enter>", lambda e: view_data_btn.config(font=FONTS.get_font("body_bold")))
        view_data_btn.bind("<Leave>", lambda e: view_data_btn.config(font=FONTS.get_font("body")))
        
        # Plot Options Card
        plot_card = CollapsibleCard(self.options_content, title="📈 Visualization",
                                    theme=self.theme_name, collapsed=False)
        plot_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.plot_type = LabeledCombobox(
            plot_card.content, label="Plot Type:",
            values=["Distribution", "Bar Chart", "Time Trend", "Scatter Plot"],
            default="Distribution", theme=self.theme_name, label_width=15,
        )
        self.plot_type.pack(fill=tk.X, pady=4)
        
        self.show_plot_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            plot_card.content, label="Show visualization",
            variable=self.show_plot_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        self.show_stats_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            plot_card.content, label="Show statistics on plot",
            variable=self.show_stats_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # Action Buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Compute Disruption Index", icon="💥",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        quick_frame = tk.Frame(btn_frame, bg=self.theme["bg_secondary"])
        quick_frame.pack(fill=tk.X, pady=(8, 0))
        
        ThemedButton(
            quick_frame, text="Top Disruptive", command=self._show_top_disruptive,
            theme=self.theme_name,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        
        ThemedButton(
            quick_frame, text="Top Consolidating", command=self._show_top_consolidating,
            theme=self.theme_name,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
    
    def _on_entity_changed(self):
        """Handle entity type change."""
        defaults = {"documents": 1, "sources": 3, "authors": 2, 
                   "countries": 5, "affiliations": 3, "years": 1}
        self.min_docs.set(defaults.get(self.entity_var.get(), 3))
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook for results and info
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="📊 Results")
        
        # Info tab
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Show detailed placeholder with instructions."""
        # Check if results_tab still exists
        if not hasattr(self, 'results_tab'):
            return
        try:
            if not self.results_tab.winfo_exists():
                return
        except Exception:
            return
        
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        
        msg = (
            "💥 Disruption Index\n\n"
            "Measure whether papers consolidate or disrupt fields.\n\n"
            "Features:\n"
            "• CD Index calculation\n"
            "• DI1, DI5, DIp variants\n"
            "• Disruption classification\n"
            "• Field-normalized scores\n"
            "\n"
            "Positive = disruptive; Negative = consolidating work.\n\n"
            "Steps:\n"
            "1. Load a dataset with citation data\n"
            "2. Select index variant\n"
            "3. Set parameters\n"
            "4. Click 'Analyze'\n"
        )
        
        tk.Label(
            self.results_tab,
            text=msg,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

    def _run_analysis(self):
        """Run disruption analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        entity = self.entity_var.get()
        self._show_loading(f"Computing disruption for {entity}...")
        
        def do_analysis():
            try:
                # Compute document disruption first
                if not hasattr(self.bib, 'disruption_df') or self.bib.disruption_df is None:
                    self.bib.compute_disruption_index(
                        verbose=True, add_to_df=self.add_to_df_var.get()
                    )
                
                # Get data based on entity type
                if entity == "documents":
                    result_df = self.bib.disruption_df.copy()
                    # Add metadata
                    if result_df is not None and len(result_df) > 0:
                        title_col = self.bib.mapping.get("Title", "Title")
                        year_col = self.bib.mapping.get("Year", "Year")
                        id_col = self.bib.mapping.get("unique-id", "unique-id")
                        
                        meta_cols = [c for c in [id_col, title_col, year_col] if c in self.bib.df.columns]
                        if meta_cols:
                            meta = self.bib.df[meta_cols].copy()
                            meta['_join'] = meta[id_col].astype(str).str.strip().str.lower()
                            result_df['_join'] = result_df['doc_id'].astype(str).str.strip().str.lower()
                            result_df = result_df.merge(meta.drop_duplicates('_join'), on='_join', how='left')
                            result_df = result_df.drop(columns=['_join'])
                elif entity == "sources":
                    result_df = self.bib.get_source_disruption(min_docs=self.min_docs.get())
                elif entity == "authors":
                    result_df = self.bib.get_author_disruption(min_docs=self.min_docs.get())
                elif entity == "countries":
                    result_df = self.bib.get_country_disruption(min_docs=self.min_docs.get())
                elif entity == "affiliations":
                    result_df = self.bib.get_affiliation_disruption(min_docs=self.min_docs.get())
                elif entity == "years":
                    result_df = self.bib.get_year_disruption()
                else:
                    result_df = pd.DataFrame()
                
                self.after(0, lambda r=result_df, ent=entity: self._on_success(r, ent))
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _on_success(self, df: pd.DataFrame, entity: str):
        """Display analysis results."""
        # Check if results_tab still exists
        if not hasattr(self, 'results_tab'):
            return
        try:
            if not self.results_tab.winfo_exists():
                return
        except Exception:
            return
        
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            tk.Label(
                self.results_tab,
                text="No disruption data available.\n\n"
                     "This may be because:\n"
                     "• Papers don't cite each other within the dataset\n"
                     "• References column is missing or empty",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"], justify=tk.CENTER,
            ).pack(expand=True)
            return
        
        self._current_result = df
        self._current_entity = entity
        
        # Summary Cards
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        grid.add_card(StatsCard(grid, entity.title(), f"{len(df):,}", "📊", self.theme_name))
        
        cd_col = 'cd_index' if 'cd_index' in df.columns else 'cd_index_mean'
        if cd_col in df.columns:
            valid_data = df[cd_col].dropna()
            if len(valid_data) > 0:
                mean_cd = valid_data.mean()
                grid.add_card(StatsCard(grid, "Mean CD", f"{mean_cd:.3f}",
                             "📈" if mean_cd > 0 else "📉", self.theme_name))
                
                thresh = 0.25 if entity == "documents" else 0.1
                n_disruptive = (valid_data > thresh).sum()
                n_consolidating = (valid_data < -thresh).sum()
                
                grid.add_card(StatsCard(grid, "Disruptive", f"{n_disruptive:,}",
                             "💥", self.theme_name, accent=True))
                grid.add_card(StatsCard(grid, "Consolidating", f"{n_consolidating:,}",
                             "🔗", self.theme_name))
        
        # Visualization
        if self.show_plot_var.get() and HAS_MATPLOTLIB:
            self._create_visualization(df, entity)
        
        # Data Table
        tk.Label(
            self.results_tab, text=f"📋 {entity.title()} Data",
            font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4))
        
        display_df = df.head(100).copy()
        for col in display_df.select_dtypes(include=[np.number]).columns:
            display_df[col] = display_df[col].round(4)
        
        table = DataTable(self.results_tab, theme=self.theme_name, height=15)
        table.pack(fill=tk.BOTH, expand=True, pady=8)
        table.set_data(display_df)
        
        # Export buttons
        export_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        export_frame.pack(fill=tk.X, pady=8)
        
        ThemedButton(export_frame, text="📋 View Data",
                    command=self._view_data, theme=self.theme_name).pack(side=tk.LEFT, padx=4)
        ThemedButton(export_frame, text="📥 Export to Excel",
                    command=self._export_to_excel, theme=self.theme_name).pack(side=tk.LEFT, padx=4)
        ThemedButton(export_frame, text="📊 Save Plot",
                    command=self._save_plot, theme=self.theme_name).pack(side=tk.LEFT, padx=4)
    
    def _view_data(self):
        """Navigate to View Data panel."""
        event_bus.emit(EventBus.PANEL_CHANGED, {"panel_id": "view"})
    
    def _create_visualization(self, df: pd.DataFrame, entity: str):
        """Create visualization using biblium plot methods."""
        if not HAS_MATPLOTLIB:
            return
        
        plot_type = self.plot_type.get()
        
        plot_frame = PlotFrame(self.results_tab, theme=self.theme_name, figsize=(10, 5), show_ai_button=True)
        plot_frame.pack(fill=tk.X, pady=(0, 16))
        
        fig, ax = plot_frame.get_figure()
        
        try:
            # Use biblium's plot methods with ax parameter
            if plot_type == "Distribution" and entity == "documents":
                if hasattr(self.bib, 'plot_disruption_distribution'):
                    self.bib.plot_disruption_distribution(
                        show_stats=self.show_stats_var.get(),
                        ax=ax
                    )
                else:
                    self._plot_distribution_fallback(df, ax)
            
            elif plot_type == "Bar Chart" or entity in ["sources", "authors", "countries", "affiliations"]:
                if hasattr(self.bib, 'plot_disruption_by_entity'):
                    self.bib.plot_disruption_by_entity(
                        entity=entity if entity != "documents" else "sources",
                        top_n=self.top_n.get(),
                        ax=ax
                    )
                else:
                    self._plot_bar_fallback(df, entity, ax)
            
            elif plot_type == "Time Trend" or entity == "years":
                if hasattr(self.bib, 'plot_disruption_over_time'):
                    self.bib.plot_disruption_over_time(ax=ax)
                else:
                    self._plot_time_fallback(df, ax)
            
            elif plot_type == "Scatter Plot":
                if hasattr(self.bib, 'plot_disruption_scatter'):
                    self.bib.plot_disruption_scatter(ax=ax)
                else:
                    self._plot_scatter_fallback(df, ax)
            
            else:
                if hasattr(self.bib, 'plot_disruption_distribution'):
                    self.bib.plot_disruption_distribution(ax=ax)
                else:
                    self._plot_distribution_fallback(df, ax)
            
            fig.tight_layout()
            plot_frame.refresh()
        except Exception as e:
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            plot_frame.refresh()
    
    def _plot_distribution_fallback(self, df: pd.DataFrame, ax):
        """Fallback distribution plot."""
        cd_col = 'cd_index' if 'cd_index' in df.columns else 'cd_index_mean'
        
        if cd_col not in df.columns:
            ax.text(0.5, 0.5, "No CD Index data", ha='center', va='center', transform=ax.transAxes)
            return
        
        data = df[cd_col].dropna()
        if len(data) == 0:
            ax.text(0.5, 0.5, "No valid CD Index values", ha='center', va='center', transform=ax.transAxes)
            return
        
        bins = np.linspace(-1, 1, 41)
        n, bins_edges, patches = ax.hist(data, bins=bins, alpha=0.7, edgecolor='white')
        
        for i, patch in enumerate(patches):
            bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
            if bin_center > 0.25:
                patch.set_facecolor('#27ae60')
            elif bin_center < -0.25:
                patch.set_facecolor('#e74c3c')
            else:
                patch.set_facecolor('#3498db')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axvline(x=0.25, color='green', linestyle='--', alpha=0.3)
        ax.axvline(x=-0.25, color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel('CD Index')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Disruption Index')
        ax.set_xlim(-1.05, 1.05)
        
        if self.show_stats_var.get():
            stats_text = f'n={len(data):,}\nMean={data.mean():.3f}\nMedian={data.median():.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_bar_fallback(self, df: pd.DataFrame, entity: str, ax):
        """Fallback bar chart."""
        entity_col = df.columns[0]
        metric_col = 'cd_index_mean' if 'cd_index_mean' in df.columns else 'cd_index'
        
        if metric_col not in df.columns:
            ax.text(0.5, 0.5, "No CD Index data", ha='center', va='center', transform=ax.transAxes)
            return
        
        top_n = min(self.top_n.get(), len(df))
        df_sorted = df.nlargest(top_n, metric_col).copy()
        df_sorted[entity_col] = df_sorted[entity_col].astype(str).str[:40]
        
        colors = ['#27ae60' if v > 0.1 else '#e74c3c' if v < -0.1 else '#3498db'
                 for v in df_sorted[metric_col].fillna(0)]
        
        y_pos = range(len(df_sorted))
        ax.barh(y_pos, df_sorted[metric_col], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted[entity_col], fontsize=9)
        ax.set_xlabel('Mean CD Index')
        ax.set_title(f'Disruption Index by {entity.title()}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.invert_yaxis()
    
    def _plot_time_fallback(self, df: pd.DataFrame, ax):
        """Fallback time trend plot."""
        year_col = 'Year' if 'Year' in df.columns else df.columns[0]
        metric_col = 'cd_index_mean' if 'cd_index_mean' in df.columns else 'cd_index'
        
        if metric_col not in df.columns:
            ax.text(0.5, 0.5, "No CD Index data", ha='center', va='center', transform=ax.transAxes)
            return
        
        df_sorted = df.sort_values(year_col)
        ax.plot(df_sorted[year_col], df_sorted[metric_col], 'o-', color='#3498db', linewidth=2, markersize=6)
        
        if 'cd_index_std' in df_sorted.columns:
            ax.fill_between(df_sorted[year_col],
                           df_sorted[metric_col] - df_sorted['cd_index_std'],
                           df_sorted[metric_col] + df_sorted['cd_index_std'],
                           alpha=0.2, color='#3498db')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(y=0.25, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=-0.25, color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel('Year')
        ax.set_ylabel('Mean CD Index')
        ax.set_title('Disruption Index Over Time')
        ax.set_ylim(-1, 1)
    
    def _plot_scatter_fallback(self, df: pd.DataFrame, ax):
        """Fallback scatter plot."""
        year_col = self.bib.mapping.get("Year", "Year")
        cd_col = 'cd_index'
        
        if year_col in df.columns and cd_col in df.columns:
            valid = df[[year_col, cd_col]].dropna()
            ax.scatter(valid[year_col], valid[cd_col], alpha=0.6, c='#3498db')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.set_xlabel('Year')
            ax.set_ylabel('CD Index')
            ax.set_title('Disruption vs Year')
        else:
            ax.text(0.5, 0.5, "Missing data for scatter plot", ha='center', va='center', transform=ax.transAxes)
    
    def _show_top_disruptive(self):
        """Show top disruptive papers."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Finding top disruptive papers...")
        
        def do_analysis():
            try:
                if not hasattr(self.bib, 'disruption_df') or self.bib.disruption_df is None:
                    self.bib.compute_disruption_index(verbose=True)
                result = self.bib.get_top_disruptive(n=self.top_n.get())
                self.after(0, lambda r=result: self._on_success(r, "top_disruptive"))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _show_top_consolidating(self):
        """Show top consolidating papers."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Finding top consolidating papers...")
        
        def do_analysis():
            try:
                if not hasattr(self.bib, 'disruption_df') or self.bib.disruption_df is None:
                    self.bib.compute_disruption_index(verbose=True)
                result = self.bib.get_top_consolidating(n=self.top_n.get())
                self.after(0, lambda r=result: self._on_success(r, "top_consolidating"))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _export_to_excel(self):
        """Export results to Excel."""
        if not hasattr(self, '_current_result') or self._current_result is None:
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=f"disruption_{self._current_entity}.xlsx"
        )
        
        if filename:
            try:
                self._current_result.to_excel(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export: {str(e)}")
    
    def _save_plot(self):
        """Save current plot."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf")],
            initialfile=f"disruption_{self._current_entity}.png"
        )
        
        if filename:
            try:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save: {str(e)}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
DISRUPTION INDEX
════════════════

Measure whether papers consolidate or disrupt fields.

THE CD INDEX
────────────
Formula: CD = (nᵢ - nⱼ) / (nᵢ + nⱼ + nₖ)

For papers citing the focal paper:
• nᵢ = cite focal but NOT its references (disrupting)
• nⱼ = cite BOTH focal AND its references (consolidating)
• nₖ = cite references but NOT focal

INTERPRETATION
──────────────
• CD = +1: Fully disruptive
  All citers ignore references
  Paper "replaces" prior work

• CD = 0: Neutral
  Equal disrupting/consolidating

• CD = -1: Fully consolidating
  All citers also cite references
  Paper builds incrementally

INDEX VARIANTS
──────────────
• DI1: Standard disruption
• DI5: 5-year citation window
• DIp: Proportional variant

TYPICAL VALUES
──────────────
• Most papers: -0.3 to +0.3
• Highly disruptive: > +0.5
• Highly consolidating: < -0.5

DATA REQUIREMENTS
─────────────────
• Parsed references
• Citation data
• Sufficient time window
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

    def _show_no_data_message(self):
        """Show no data message."""
        tk.Label(
            self.options_content,
            text="📂 Please load a dataset first\n\nGo to DATA → Load Dataset",
            font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"], justify=tk.CENTER,
        ).pack(expand=True)
