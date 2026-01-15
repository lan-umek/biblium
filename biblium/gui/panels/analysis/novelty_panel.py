# -*- coding: utf-8 -*-
"""
Novelty Analysis Panel
======================
GUI panel for analyzing novelty and atypicality of papers.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, Optional
import pandas as pd

from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, StatsCard, CardGrid
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledEntry
from biblium.gui.widgets.buttons import ActionButton
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame
from biblium.gui.config import FONTS, get_theme
from biblium.gui.core.events import EventBus

try:
    from biblium.utilsbib_modules.stats import analyze_novelty, fetch_openalex_novelty
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

event_bus = EventBus()


class NoveltyPanel(BasePanel):
    """
    Panel for novelty and atypicality analysis.
    
    Analyzes within-dataset novelty based on:
    - Keyword combination rarity
    - Subject category bridging
    - Reference diversity
    
    Optionally fetches broader novelty data from OpenAlex API.
    """
    
    title = "🔬 Novelty Analysis"
    description = "Identify novel and atypical research based on unusual combinations"
    
    def __init__(self, parent, bib=None, theme: str = "light", **kwargs):
        self._current_result = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create options panel."""
        super()._create_options()
        
        if not self.bib:
            tk.Label(
                self.options_content,
                text="Load a dataset to analyze novelty",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"],
                fg=self.theme["text_muted"],
            ).pack(pady=20)
            return
        
        # Column Selection Card
        col_card = Card(self.options_content, title="📋 Column Selection", theme=self.theme_name)
        col_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Keywords column
        kw_cols = []
        for col in ["Author Keywords", "Keywords", "DE", "ID", "Index Keywords"]:
            if col in self.bib.df.columns:
                kw_cols.append(col)
        kw_cols = kw_cols + [c for c in self.bib.df.columns if c not in kw_cols]
        
        default_kw = kw_cols[0] if kw_cols else ""
        
        self.kw_combo = LabeledCombobox(
            col_card.content, label="Keywords column:",
            values=kw_cols,
            default=default_kw,
            theme=self.theme_name, label_width=18,
        )
        self.kw_combo.pack(fill=tk.X, pady=4)
        
        # Subject column
        subj_cols = []
        for col in ["Subject Area", "WC", "SC", "Categories", "Subject Categories", 
                    "Research Areas", "Web of Science Categories"]:
            if col in self.bib.df.columns:
                subj_cols.append(col)
        subj_cols = subj_cols + [c for c in self.bib.df.columns if c not in subj_cols]
        
        default_subj = subj_cols[0] if subj_cols else ""
        
        self.subj_combo = LabeledCombobox(
            col_card.content, label="Subject column:",
            values=subj_cols,
            default=default_subj,
            theme=self.theme_name, label_width=18,
        )
        self.subj_combo.pack(fill=tk.X, pady=4)
        
        # References column
        ref_cols = []
        for col in ["References", "Cited References", "CR", "cited_by_ids"]:
            if col in self.bib.df.columns:
                ref_cols.append(col)
        ref_cols = ref_cols + [c for c in self.bib.df.columns if c not in ref_cols]
        
        default_ref = ref_cols[0] if ref_cols else ""
        
        self.ref_combo = LabeledCombobox(
            col_card.content, label="References column:",
            values=ref_cols,
            default=default_ref,
            theme=self.theme_name, label_width=18,
        )
        self.ref_combo.pack(fill=tk.X, pady=4)
        
        # Year column
        year_cols = ["Year"] if "Year" in self.bib.df.columns else []
        for col in self.bib.df.columns:
            if col not in year_cols and ("year" in col.lower() or col == "PY"):
                year_cols.append(col)
        year_cols = year_cols + [c for c in self.bib.df.columns if c not in year_cols]
        
        default_year = year_cols[0] if year_cols else ""
        
        self.year_combo = LabeledCombobox(
            col_card.content, label="Year column:",
            values=year_cols,
            default=default_year,
            theme=self.theme_name, label_width=18,
        )
        self.year_combo.pack(fill=tk.X, pady=4)
        
        # OpenAlex API Card
        api_card = Card(self.options_content, title="🌐 OpenAlex API (Optional)", theme=self.theme_name)
        api_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.use_openalex_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            api_card.content, label="Fetch broader novelty from OpenAlex",
            variable=self.use_openalex_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.email_entry = LabeledEntry(
            api_card.content, label="Email (for API):",
            theme=self.theme_name, label_width=18,
        )
        self.email_entry.pack(fill=tk.X, pady=2)
        
        tk.Label(
            api_card.content,
            text="OpenAlex provides concept-based novelty\n"
                 "comparing to global literature patterns.\n"
                 "Requires DOI column in dataset.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))
        
        # Display Options Card
        display_card = Card(self.options_content, title="⚙️ Display Options", theme=self.theme_name)
        display_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_distribution_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show novelty distribution",
            variable=self.show_distribution_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_components_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show component breakdown",
            variable=self.show_components_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_trends_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show temporal trends",
            variable=self.show_trends_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_table_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show top novel papers",
            variable=self.show_table_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Action Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Analyze Novelty", icon="🔬",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook with permanent Info tab
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="📊 Results")
        
        # Info tab (always visible)
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
        # Show placeholder in results tab
        self._show_results_placeholder()
    
    def _show_results_placeholder(self):
        """Show placeholder in results tab."""
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
        tk.Label(
            self.results_tab,
            text="Click 'Run' to see results here.\nSee Info tab for documentation.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

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
            "✨ Novelty & Atypicality\n\n"
            "Measure how papers combine knowledge in new ways.\n\n"
            "Features:\n"
            "• Novelty scores (new combinations)\n"
            "• Atypicality measures\n"
            "• Conventionality index\n"
            "• Reference diversity metrics\n"
            "\n"
            "High impact often comes from novel combinations.\n\n"
            "Steps:\n"
            "1. Load dataset with references\n"
            "2. Select novelty metrics\n"
            "3. Set baseline period\n"
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
        """Run novelty analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_ANALYSIS:
            messagebox.showerror("Error", "Novelty analysis module not available.")
            return
        
        try:
            plt.close('all')
        except:
            pass
        
        self._show_loading("Analyzing novelty...")
        self.update_idletasks()
        
        try:
            kw_col = self.kw_combo.get() if self.kw_combo.get() else None
            subj_col = self.subj_combo.get() if self.subj_combo.get() else None
            ref_col = self.ref_combo.get() if self.ref_combo.get() else None
            year_col = self.year_combo.get() if self.year_combo.get() else "Year"
            
            result = analyze_novelty(
                self.bib.df,
                keywords_col=kw_col,
                subject_col=subj_col,
                references_col=ref_col,
                year_col=year_col,
                verbose=False,
            )
            
            # Optional OpenAlex enhancement
            if self.use_openalex_var.get():
                email = self.email_entry.get()
                doi_col = None
                for col in ["DOI", "doi", "DI"]:
                    if col in self.bib.df.columns:
                        doi_col = col
                        break
                
                if doi_col:
                    self._show_loading("Fetching OpenAlex data...")
                    self.update_idletasks()
                    
                    dois = self.bib.df[doi_col].dropna().tolist()[:100]  # Limit to 100
                    try:
                        oa_result = fetch_openalex_novelty(dois, email=email, verbose=False)
                        result["openalex"] = oa_result
                    except Exception as e:
                        print(f"OpenAlex fetch error: {e}")
                        result["openalex"] = None
                else:
                    messagebox.showwarning("No DOI", "DOI column not found for OpenAlex lookup.")
                    result["openalex"] = None
            
            self._display_results(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_error(str(e))
    
    def _display_results(self, result: Dict):
        """Display novelty analysis results."""
        try:
            plt.close('all')
        except:
            pass
        
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
        
        stats = result["statistics"]
        
        # Summary Cards
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 8))
        
        grid.add_card(StatsCard(grid, "Papers", f"{stats['n_papers']:,}", "📄", self.theme_name))
        grid.add_card(StatsCard(grid, "Mean Novelty", f"{stats['mean_novelty']:.3f}", "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "Highly Novel", f"{stats['highly_novel_count']:,}", "🔬", self.theme_name))
        grid.add_card(StatsCard(grid, "Top 10% Threshold", f"{stats['highly_novel_threshold']:.3f}", "📈", self.theme_name))
        
        # Data availability row
        grid2 = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
        grid2.pack(fill=tk.X, pady=(0, 12))
        
        kw_status = "✓" if stats['has_keywords'] else "✗"
        subj_status = "✓" if stats['has_subjects'] else "✗"
        ref_status = "✓" if stats['has_references'] else "✗"
        
        grid2.add_card(StatsCard(grid2, "Keywords", kw_status, "🏷️", self.theme_name))
        grid2.add_card(StatsCard(grid2, "Subjects", subj_status, "📚", self.theme_name))
        grid2.add_card(StatsCard(grid2, "References", ref_status, "📖", self.theme_name))
        
        # Create Notebook
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        
        if self.show_distribution_var.get() and HAS_MATPLOTLIB:
            dist_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(dist_frame, text="📊 Distribution")
            self._plot_distribution(result, dist_frame)
        
        if self.show_components_var.get() and HAS_MATPLOTLIB:
            comp_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(comp_frame, text="📈 Components")
            self._plot_components(result, comp_frame)
        
        if self.show_trends_var.get() and HAS_MATPLOTLIB:
            trend_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(trend_frame, text="📅 Trends")
            self._plot_trends(result, trend_frame)
        
        if self.show_table_var.get():
            table_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(table_frame, text="🏆 Top Novel")
            self._show_top_papers(result, table_frame)
        
        # Details tab
        details_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(details_frame, text="📋 Details")

        

        # Info tab

        info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])

        notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        self._show_details(result, details_frame)
        
        self._add_export_buttons(result)
        self._current_result = result
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Novelty Analysis"})
    
    def _plot_distribution(self, result: Dict, parent: tk.Frame):
        """Plot novelty distribution."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(8, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        novelty_df = result["novelty_df"]
        scores = novelty_df["Composite Novelty"]
        
        ax.hist(scores, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        
        # Add percentile line
        p90 = scores.quantile(0.90)
        ax.axvline(p90, color='firebrick', linestyle='--', linewidth=2)
        ax.text(p90, ax.get_ylim()[1] * 0.9, f' Top 10%: {p90:.3f}', 
               color='firebrick', fontsize=10)
        
        ax.set_xlabel('Composite Novelty Score', fontsize=10)
        ax.set_ylabel('Number of Papers', fontsize=10)
        ax.set_title('Distribution of Novelty Scores', fontsize=11, fontweight='bold')
        
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        plot.canvas.draw()
    
    def _plot_components(self, result: Dict, parent: tk.Frame):
        """Plot novelty components."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(8, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        novelty_df = result["novelty_df"]
        
        components = []
        means = []
        
        if "Keyword Novelty" in novelty_df.columns:
            kw_mean = novelty_df["Keyword Novelty"].mean()
            if kw_mean > 0:
                components.append("Keyword\nNovelty")
                means.append(kw_mean)
        
        if "Subject Bridging" in novelty_df.columns:
            sb_mean = novelty_df["Subject Bridging"].mean()
            if sb_mean > 0:
                components.append("Subject\nBridging")
                means.append(sb_mean)
        
        if "Reference Diversity" in novelty_df.columns:
            rd_mean = novelty_df["Reference Diversity"].mean()
            if rd_mean > 0:
                components.append("Reference\nDiversity")
                means.append(rd_mean)
        
        if components:
            # Sort ascending (highest at top)
            sorted_pairs = sorted(zip(means, components))
            means = [p[0] for p in sorted_pairs]
            components = [p[1] for p in sorted_pairs]
            
            bars = ax.barh(range(len(components)), means, color='steelblue', 
                          alpha=0.8, height=0.5, edgecolor='white')
            ax.set_yticks(range(len(components)))
            ax.set_yticklabels(components)
            ax.set_xlabel('Mean Score', fontsize=10)
            ax.set_title('Novelty Components', fontsize=11, fontweight='bold')
            
            max_val = max(means) if means else 1
            for bar, val in zip(bars, means):
                ax.text(bar.get_width() + max_val * 0.02, 
                       bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=10)
            ax.set_xlim(0, max_val * 1.15)
        else:
            ax.text(0.5, 0.5, 'No component data', ha='center', va='center', 
                   transform=ax.transAxes)
        
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        plot.canvas.draw()
    
    def _plot_trends(self, result: Dict, parent: tk.Frame):
        """Plot novelty trends."""
        plot = PlotFrame(parent, theme=self.theme_name, figsize=(8, 5), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        fig = plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        temporal = result.get("temporal_trends")
        
        if temporal is None or len(temporal) == 0:
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.grid(False)
            plot.canvas.draw()
            return
        
        df = temporal[(temporal["Year"] >= 1990) & (temporal["Year"] <= 2030)]
        
        if len(df) > 1:
            ax.plot(df["Year"], df["Mean Novelty"], marker='o', linewidth=2, 
                   color='steelblue', markersize=6)
            ax.fill_between(df["Year"], df["Mean Novelty"], alpha=0.2, color='steelblue')
            ax.set_xlabel('Publication Year', fontsize=10)
            ax.set_ylabel('Mean Novelty Score', fontsize=10)
            ax.set_title('Novelty Trend Over Time', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient trend data', ha='center', va='center', 
                   transform=ax.transAxes)
        
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        plot.canvas.draw()
    
    def _show_top_papers(self, result: Dict, parent: tk.Frame):
        """Show top novel papers table."""
        top_papers = result.get("top_novel_papers", pd.DataFrame())
        
        if len(top_papers) == 0:
            tk.Label(parent, text="No top papers data", font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        table = DataTable(parent, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        display_cols = ["Title", "Year", "Composite Novelty", "Keyword Novelty", 
                       "Subject Bridging", "Reference Diversity"]
        available_cols = [c for c in display_cols if c in top_papers.columns]
        
        display_df = top_papers[available_cols].copy()
        if "Title" in display_df.columns:
            display_df["Title"] = display_df["Title"].apply(
                lambda x: str(x)[:60] + "..." if pd.notna(x) and len(str(x)) > 60 else x
            )
        
        # Round numeric columns
        for col in ["Composite Novelty", "Keyword Novelty", "Subject Bridging", "Reference Diversity"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)
        
        table.set_data(display_df)
    
    def _show_details(self, result: Dict, parent: tk.Frame):
        """Show detailed analysis information."""
        canvas = tk.Canvas(parent, bg=self.theme["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.theme["bg_card"])
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Keyword Novelty Details
        kw_novelty = result.get("keyword_novelty")
        if kw_novelty:
            kw_card = Card(scrollable, title="🏷️ Keyword Novelty", theme=self.theme_name)
            kw_card.pack(fill=tk.X, pady=4)
            
            stats_text = (
                f"Unique keywords: {kw_novelty['n_unique_keywords']}\n"
                f"Keyword pairs: {kw_novelty['n_keyword_pairs']}\n"
                f"Novel pairs (appear once): {kw_novelty['n_unique_pairs']}\n"
                f"Novel pair ratio: {kw_novelty['unique_pair_ratio']:.1%}\n"
                f"Mean novelty: {kw_novelty['mean_novelty']:.3f}"
            )
            tk.Label(kw_card.content, text=stats_text, font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    justify=tk.LEFT).pack(anchor=tk.W, pady=4)
            
            # Show common pairs
            if kw_novelty.get("most_common_pairs"):
                tk.Label(kw_card.content, text="\nMost common pairs:", 
                        font=FONTS.get_font("body_bold"),
                        bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(anchor=tk.W)
                for pair, count in kw_novelty["most_common_pairs"][:5]:
                    tk.Label(kw_card.content, text=f"  {pair[0]} + {pair[1]}: {count}",
                            font=FONTS.get_font("small"),
                            bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(anchor=tk.W)
        
        # Subject Bridging Details
        subj_bridging = result.get("subject_bridging")
        if subj_bridging:
            subj_card = Card(scrollable, title="📚 Subject Bridging", theme=self.theme_name)
            subj_card.pack(fill=tk.X, pady=4)
            
            stats_text = (
                f"Subject categories: {subj_bridging['n_subjects']}\n"
                f"Subject pairs: {subj_bridging['n_subject_pairs']}\n"
                f"Rare bridges (≤2 papers): {subj_bridging['n_rare_bridges']}\n"
                f"Mean bridging: {subj_bridging['mean_bridging']:.3f}"
            )
            tk.Label(subj_card.content, text=stats_text, font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    justify=tk.LEFT).pack(anchor=tk.W, pady=4)
        
        # Reference Diversity Details
        ref_diversity = result.get("reference_diversity")
        if ref_diversity:
            ref_card = Card(scrollable, title="📖 Reference Diversity", theme=self.theme_name)
            ref_card.pack(fill=tk.X, pady=4)
            
            stats_text = (
                f"Mean references per paper: {ref_diversity['mean_references']:.1f}\n"
                f"Median references: {ref_diversity['median_references']:.1f}\n"
                f"Papers with references: {ref_diversity['papers_with_refs']}\n"
                f"Mean diversity (Simpson): {ref_diversity['mean_diversity']:.3f}"
            )
            tk.Label(ref_card.content, text=stats_text, font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    justify=tk.LEFT).pack(anchor=tk.W, pady=4)
        
        # OpenAlex Results
        oa_result = result.get("openalex")
        if oa_result and oa_result.get("n_fetched", 0) > 0:
            oa_card = Card(scrollable, title="🌐 OpenAlex Data", theme=self.theme_name)
            oa_card.pack(fill=tk.X, pady=4)
            
            stats_text = (
                f"Papers fetched: {oa_result['n_fetched']} / {oa_result['n_requested']}\n"
                f"Concept-based novelty available for matched papers"
            )
            tk.Label(oa_card.content, text=stats_text, font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    justify=tk.LEFT).pack(anchor=tk.W, pady=4)
    
    def _add_export_buttons(self, result: Dict):
        """Add export buttons."""
        btn_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=8)
        
        ActionButton(btn_frame, text="Export to Excel", icon="📊",
                    command=lambda: self._export_excel(result), theme=self.theme_name).pack(side=tk.LEFT, padx=4)
        ActionButton(btn_frame, text="Save Plots", icon="💾",
                    command=lambda: self._save_plots(result), theme=self.theme_name).pack(side=tk.LEFT, padx=4)
    
    def _export_excel(self, result: Dict):
        """Export results to Excel."""
        filename = filedialog.asksaveasfilename(defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")], title="Save Novelty Results")
        if not filename:
            return
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                result["novelty_df"].to_excel(writer, sheet_name="Novelty Scores", index=False)
                if "top_novel_papers" in result:
                    result["top_novel_papers"].to_excel(writer, sheet_name="Top Novel", index=False)
                if "temporal_trends" in result and result["temporal_trends"] is not None:
                    result["temporal_trends"].to_excel(writer, sheet_name="Trends", index=False)
            messagebox.showinfo("Success", f"Results exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def _save_plots(self, result: Dict):
        """Save plots to files."""
        base = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG files", "*.png")], title="Save Novelty Plots")
        if not base:
            return
        if base.endswith(('.png', '.pdf', '.svg')):
            base = base.rsplit('.', 1)[0]
        try:
            from biblium import plotbib
            plotbib.plot_novelty_distribution(result, filename=base + "_distribution", show=False)
            plotbib.plot_novelty_components(result, filename=base + "_components", show=False)
            plotbib.plot_novelty_trend(result, filename=base + "_trend", show=False)
            messagebox.showinfo("Success", f"Plots saved to:\n{base}_*.png")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{str(e)}")
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading indicator."""
        try:
            plt.close('all')
        except:
            pass
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
        frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        frame.pack(expand=True)
        tk.Label(frame, text="⏳", font=("Segoe UI", 32), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]).pack(pady=(20, 10))
        tk.Label(frame, text=message, font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]).pack()
    
    def _show_error(self, message: str):
        """Show error message."""
        try:
            plt.close('all')
        except:
            pass
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
        tk.Label(self.results_tab, text=f"❌ Error\n\n{message}", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["error"], justify=tk.CENTER,
                wraplength=400).pack(expand=True)
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
NOVELTY & ATYPICALITY ANALYSIS
══════════════════════════════

Measure how papers combine knowledge in novel ways.

NOVELTY SCORE
─────────────
Measures unusual combinations of referenced journals:
• Examines all pairs of journals in references
• Compares to baseline co-citation frequency
• Rare combinations = higher novelty

Calculation:
1. Extract journal pairs from paper's references
2. Look up each pair's co-citation frequency
3. Rare pairs contribute more to novelty score
4. Average across all pairs

ATYPICALITY SCORE
─────────────────
Measures deviation from typical citation patterns:
• How different from average paper in field
• Based on reference list composition
• High atypicality = unusual knowledge base

INTERPRETATION
──────────────
Novelty Score:
• Low (< 50th %ile): Conventional combinations
• Medium (50-90th): Balanced approach
• High (> 90th): Unusual combinations

Research shows:
• Moderate novelty → highest long-term impact
• Too conventional → limited contribution
• Too novel → may face acceptance barriers

DATA REQUIREMENTS
─────────────────
• Parsed references with journal names
• Sufficient references per paper (10+)
• Baseline corpus for comparison

OUTPUT
──────
• Novelty score per paper
• Atypicality score
• Percentile rankings
• Distribution plots
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
        try:
            plt.close('all')
        except:
            pass
        self._current_result = None
        super().destroy()
