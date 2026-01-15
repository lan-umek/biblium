# -*- coding: utf-8 -*-
"""
Sentiment Analysis Panel
========================
GUI panel for comprehensive sentiment analysis of bibliographic text data.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, Optional, Any
import pandas as pd

from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, StatsCard, CardGrid
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledEntry
from biblium.gui.widgets.buttons import ActionButton
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame
from biblium.gui.config import FONTS, get_theme
from biblium.gui.core.events import EventBus

# Import sentiment analysis functions
try:
    from biblium.utilsbib_modules.stats import (
        analyze_sentiment_advanced,
        get_sentiment_by_entity,
        compare_sentiment_groups,
    )
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False

# Import sentiment plotting functions
try:
    from biblium.plotting.sentiment_plots import (
        plot_sentiment_distribution,
        plot_sentiment_categories,
        plot_sentiment_temporal,
        plot_sentiment_certainty,
        plot_sentiment_by_source,
        plot_sentiment_heatmap,
    )
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

event_bus = EventBus()


class SentimentPanel(BasePanel):
    """
    Panel for comprehensive sentiment analysis.
    
    Analyzes sentiment in bibliographic text using:
    - VADER sentiment (compound scores)
    - TextBlob sentiment (polarity and subjectivity)
    - Scientific certainty markers (domain-specific)
    
    Features:
    - Temporal sentiment trends
    - Sentiment by source/journal
    - Word analysis per sentiment category
    - Correlation analysis
    - Multiple visualization options
    """
    
    title = "💭 Sentiment Analysis"
    description = "Analyze sentiment, tone, and certainty in scientific text"
    
    def __init__(self, parent, bib=None, theme: str = "light", **kwargs):
        self._current_result = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create options panel."""
        super()._create_options()
        
        if not self.bib:
            tk.Label(
                self.options_content,
                text="Load a dataset to analyze sentiment",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"],
                fg=self.theme["text_muted"],
            ).pack(pady=20)
            return
        
        # Column Selection Card
        col_card = Card(self.options_content, title="📋 Column Selection", theme=self.theme_name)
        col_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Text column selection
        text_cols = []
        for col in ["Abstract", "Processed Abstract", "Title", "AB", "TI", "Combined Text"]:
            if col in self.bib.df.columns:
                text_cols.append(col)
        text_cols = text_cols + [c for c in self.bib.df.columns if c not in text_cols]
        
        default_text = text_cols[0] if text_cols else ""
        
        self.text_combo = LabeledCombobox(
            col_card.content, label="Text column:",
            values=text_cols,
            default=default_text,
            theme=self.theme_name, label_width=18,
        )
        self.text_combo.pack(fill=tk.X, pady=4)
        
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
        
        # Source column
        source_cols = []
        for col in ["Source", "Source title", "Journal", "SO", "Publication Name"]:
            if col in self.bib.df.columns:
                source_cols.append(col)
        source_cols = source_cols + [c for c in self.bib.df.columns if c not in source_cols]
        
        default_source = source_cols[0] if source_cols else ""
        
        self.source_combo = LabeledCombobox(
            col_card.content, label="Source column:",
            values=source_cols,
            default=default_source,
            theme=self.theme_name, label_width=18,
        )
        self.source_combo.pack(fill=tk.X, pady=4)
        
        # Analysis Methods Card
        methods_card = Card(self.options_content, title="🔬 Analysis Methods", theme=self.theme_name)
        methods_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.use_vader_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            methods_card.content, label="Use VADER sentiment analysis",
            variable=self.use_vader_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.use_textblob_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            methods_card.content, label="Use TextBlob sentiment analysis",
            variable=self.use_textblob_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.use_scientific_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            methods_card.content, label="Use scientific certainty markers",
            variable=self.use_scientific_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Threshold setting
        threshold_frame = tk.Frame(methods_card.content, bg=self.theme["bg_card"])
        threshold_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            threshold_frame, text="Sentiment threshold:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            width=18, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self.threshold_var = tk.StringVar(value="0.05")
        threshold_entry = tk.Entry(
            threshold_frame, textvariable=self.threshold_var,
            font=FONTS.get_font("body"), width=10,
        )
        threshold_entry.pack(side=tk.LEFT, padx=4)
        
        # Analysis Options Card
        options_card = Card(self.options_content, title="⚙️ Analysis Options", theme=self.theme_name)
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.analyze_temporal_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            options_card.content, label="Analyze temporal trends",
            variable=self.analyze_temporal_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.analyze_source_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            options_card.content, label="Analyze by source/journal",
            variable=self.analyze_source_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Display Options Card
        display_card = Card(self.options_content, title="📊 Display Options", theme=self.theme_name)
        display_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_distribution_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show sentiment distribution",
            variable=self.show_distribution_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_temporal_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show temporal trends",
            variable=self.show_temporal_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_certainty_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show certainty analysis",
            variable=self.show_certainty_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_table_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content, label="Show top documents",
            variable=self.show_table_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Action Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Analyze Sentiment", icon="💭",
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
            "😊 Sentiment Analysis\n\n"
            "Analyze emotional tone in abstracts and titles.\n\n"
            "Features:\n"
            "• Positive/negative sentiment scores\n"
            "• Sentiment distribution\n"
            "• Temporal sentiment trends\n"
            "• Word-level sentiment\n"
            "\n"
            "Reveals emotional framing in research communication.\n\n"
            "Steps:\n"
            "1. Load dataset with abstracts\n"
            "2. Select text field\n"
            "3. Choose sentiment model\n"
            "4. Click 'Analyze Sentiment'\n"
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
        """Run sentiment analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_ANALYSIS:
            messagebox.showerror("Error", "Sentiment analysis module not available.")
            return
        
        try:
            plt.close('all')
        except:
            pass
        
        self._show_loading("Analyzing sentiment...")
        self.update_idletasks()
        
        try:
            text_col = self.text_combo.get() if self.text_combo.get() else None
            year_col = self.year_combo.get() if self.year_combo.get() else "Year"
            source_col = self.source_combo.get() if self.source_combo.get() else "Source"
            
            # Parse threshold
            try:
                threshold = float(self.threshold_var.get())
            except ValueError:
                threshold = 0.05
            
            result = analyze_sentiment_advanced(
                self.bib.df,
                text_column=text_col,
                year_column=year_col,
                sentiment_threshold=threshold,
                use_vader=self.use_vader_var.get(),
                use_textblob=self.use_textblob_var.get(),
                use_scientific=self.use_scientific_var.get(),
                analyze_temporal=self.analyze_temporal_var.get(),
                analyze_by_source=self.analyze_source_var.get(),
                source_column=source_col,
                verbose=False,
            )
            
            self._display_results(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_error(str(e))
    
    def _display_results(self, result: Dict):
        """Display sentiment analysis results."""
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
        
        stats = result.get("statistics", {})
        
        # Summary Cards
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 8))
        
        grid.add_card(StatsCard(grid, "Documents", f"{stats.get('n_documents', 0):,}", "📄", self.theme_name))
        grid.add_card(StatsCard(grid, "Mean Sentiment", f"{stats.get('mean_sentiment', 0):.3f}", "📊", self.theme_name))
        
        # Sentiment distribution
        pos_pct = stats.get('positive_pct', 0)
        neg_pct = stats.get('negative_pct', 0)
        neu_pct = stats.get('neutral_pct', 0)
        
        grid.add_card(StatsCard(grid, "Positive", f"{stats.get('positive_count', 0):,} ({pos_pct:.1f}%)", "✅", self.theme_name))
        grid.add_card(StatsCard(grid, "Negative", f"{stats.get('negative_count', 0):,} ({neg_pct:.1f}%)", "❌", self.theme_name))
        
        # Second row - certainty info
        grid2 = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid2.pack(fill=tk.X, pady=(0, 12))
        
        grid2.add_card(StatsCard(grid2, "Neutral", f"{stats.get('neutral_count', 0):,} ({neu_pct:.1f}%)", "⚪", self.theme_name))
        grid2.add_card(StatsCard(grid2, "Std Dev", f"{stats.get('std_sentiment', 0):.3f}", "📈", self.theme_name))
        
        certainty_mean = stats.get('certainty_mean', 0)
        hedging_mean = stats.get('hedging_mean', 0)
        grid2.add_card(StatsCard(grid2, "Certainty", f"{certainty_mean:.3f}", "🎯", self.theme_name))
        grid2.add_card(StatsCard(grid2, "Hedging", f"{hedging_mean:.3f}", "🔄", self.theme_name))
        
        # Create Notebook for tabbed content
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        
        # Distribution tab
        if self.show_distribution_var.get() and HAS_MATPLOTLIB and HAS_PLOTS:
            dist_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(dist_frame, text="📊 Distribution")
            self._plot_distribution(result, dist_frame)
        
        # Categories tab
        cat_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(cat_frame, text="📈 Categories")
        self._plot_categories(result, cat_frame)
        
        # Temporal trends tab
        if self.show_temporal_var.get() and HAS_MATPLOTLIB and HAS_PLOTS:
            trend_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(trend_frame, text="📅 Temporal")
            self._plot_temporal(result, trend_frame)
        
        # Certainty tab
        if self.show_certainty_var.get() and HAS_MATPLOTLIB and HAS_PLOTS:
            cert_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(cert_frame, text="🎯 Certainty")
            self._plot_certainty(result, cert_frame)
        
        # Top documents tab
        if self.show_table_var.get():
            table_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(table_frame, text="🏆 Top Documents")
            self._show_top_documents(result, table_frame)
        
        # Word analysis tab
        words_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(words_frame, text="📝 Words")
        self._show_word_analysis(result, words_frame)
        
        # Correlations tab
        if result.get("correlations") is not None:
            corr_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(corr_frame, text="🔗 Correlations")

            

            # Info tab

            info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])

            notebook.add(info_frame, text="ℹ️ Info")

            self._create_info_content(info_frame)
            self._plot_correlations(result, corr_frame)
        
        # Export buttons
        self._add_export_buttons(result)
        self._current_result = result
    
    def _plot_distribution(self, result: Dict, parent: tk.Frame):
        """Plot sentiment distribution."""
        if not HAS_PLOTS:
            tk.Label(parent, text="Plotting module not available", 
                    bg=self.theme["bg_card"]).pack(expand=True)
            return
        
        try:
            plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            
            fig, ax = plot_frame.get_figure()
            plot_sentiment_distribution(result, ax=ax, show=False)
            
            # Adjust margins for labels
            fig.subplots_adjust(bottom=0.15, left=0.12, right=0.95, top=0.92)
            plot_frame.set_preserve_margins(True)
            plot_frame.refresh()
        except Exception as e:
            tk.Label(parent, text=f"Error: {str(e)}", 
                    bg=self.theme["bg_card"], fg=self.theme.get("error", self.theme.get("danger", "#e74c3c"))).pack(expand=True)
    
    def _plot_categories(self, result: Dict, parent: tk.Frame):
        """Plot sentiment categories."""
        if not HAS_PLOTS:
            tk.Label(parent, text="Plotting module not available", 
                    bg=self.theme["bg_card"]).pack(expand=True)
            return
        
        try:
            plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            
            fig, ax = plot_frame.get_figure()
            plot_sentiment_categories(result, ax=ax, show=False)
            
            # Adjust margins for labels
            fig.subplots_adjust(bottom=0.15, left=0.12, right=0.95, top=0.88)
            plot_frame.set_preserve_margins(True)
            plot_frame.refresh()
        except Exception as e:
            tk.Label(parent, text=f"Error: {str(e)}", 
                    bg=self.theme["bg_card"], fg=self.theme.get("error", self.theme.get("danger", "#e74c3c"))).pack(expand=True)
    
    def _plot_temporal(self, result: Dict, parent: tk.Frame):
        """Plot temporal trends."""
        if not HAS_PLOTS:
            tk.Label(parent, text="Plotting module not available", 
                    bg=self.theme["bg_card"]).pack(expand=True)
            return
        
        temporal = result.get("temporal_trends")
        if temporal is None or len(temporal) == 0:
            tk.Label(parent, text="No temporal data available\n(Year column may be missing)", 
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        try:
            plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            
            fig, ax = plot_frame.get_figure()
            plot_sentiment_temporal(result, ax=ax, show=False)
            
            # Adjust margins for labels
            fig.subplots_adjust(bottom=0.15, left=0.12, right=0.95, top=0.92)
            plot_frame.set_preserve_margins(True)
            plot_frame.refresh()
        except Exception as e:
            tk.Label(parent, text=f"Error: {str(e)}", 
                    bg=self.theme["bg_card"], fg=self.theme.get("error", self.theme.get("danger", "#e74c3c"))).pack(expand=True)
    
    def _plot_certainty(self, result: Dict, parent: tk.Frame):
        """Plot certainty analysis."""
        if not HAS_PLOTS:
            tk.Label(parent, text="Plotting module not available", 
                    bg=self.theme["bg_card"]).pack(expand=True)
            return
        
        try:
            plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            
            fig, ax = plot_frame.get_figure()
            plot_sentiment_certainty(result, ax=ax, show=False)
            
            # Adjust margins for labels
            fig.subplots_adjust(bottom=0.15, left=0.12, right=0.95, top=0.92)
            plot_frame.set_preserve_margins(True)
            plot_frame.refresh()
        except Exception as e:
            tk.Label(parent, text=f"Error: {str(e)}", 
                    bg=self.theme["bg_card"], fg=self.theme.get("error", self.theme.get("danger", "#e74c3c"))).pack(expand=True)
    
    def _plot_correlations(self, result: Dict, parent: tk.Frame):
        """Plot correlations."""
        if not HAS_PLOTS:
            tk.Label(parent, text="Plotting module not available", 
                    bg=self.theme["bg_card"]).pack(expand=True)
            return
        
        try:
            plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            
            fig, ax = plot_frame.get_figure()
            plot_sentiment_heatmap(result, ax=ax, show=False)
            
            # Adjust margins for rotated labels and preserve on resize
            fig.subplots_adjust(bottom=0.25, left=0.2)
            plot_frame.set_preserve_margins(True)
            plot_frame.refresh()
        except Exception as e:
            tk.Label(parent, text=f"Error: {str(e)}", 
                    bg=self.theme["bg_card"], fg=self.theme.get("error", self.theme.get("danger", "#e74c3c"))).pack(expand=True)
    
    def _show_top_documents(self, result: Dict, parent: tk.Frame):
        """Show top positive and negative documents."""
        # Create two-column layout
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Top positive
        pos_frame = tk.Frame(paned, bg=self.theme["bg_card"])
        paned.add(pos_frame, weight=1)
        
        tk.Label(pos_frame, text="✅ Most Positive Documents", 
                font=FONTS.get_font("body_bold"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=4)
        
        top_positive = result.get("top_positive", pd.DataFrame())
        if len(top_positive) > 0:
            pos_table = DataTable(pos_frame, theme=self.theme_name, height=15)
            pos_table.pack(fill=tk.BOTH, expand=True)
            
            display_cols = ["Title", "Year", "Composite Sentiment"]
            available_cols = [c for c in display_cols if c in top_positive.columns]
            
            display_df = top_positive[available_cols].copy()
            if "Title" in display_df.columns:
                display_df["Title"] = display_df["Title"].apply(
                    lambda x: str(x)[:50] + "..." if pd.notna(x) and len(str(x)) > 50 else x
                )
            if "Composite Sentiment" in display_df.columns:
                display_df["Composite Sentiment"] = display_df["Composite Sentiment"].round(3)
            
            pos_table.set_data(display_df)
        
        # Top negative
        neg_frame = tk.Frame(paned, bg=self.theme["bg_card"])
        paned.add(neg_frame, weight=1)
        
        tk.Label(neg_frame, text="❌ Most Negative Documents", 
                font=FONTS.get_font("body_bold"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=4)
        
        top_negative = result.get("top_negative", pd.DataFrame())
        if len(top_negative) > 0:
            neg_table = DataTable(neg_frame, theme=self.theme_name, height=15)
            neg_table.pack(fill=tk.BOTH, expand=True)
            
            display_cols = ["Title", "Year", "Composite Sentiment"]
            available_cols = [c for c in display_cols if c in top_negative.columns]
            
            display_df = top_negative[available_cols].copy()
            if "Title" in display_df.columns:
                display_df["Title"] = display_df["Title"].apply(
                    lambda x: str(x)[:50] + "..." if pd.notna(x) and len(str(x)) > 50 else x
                )
            if "Composite Sentiment" in display_df.columns:
                display_df["Composite Sentiment"] = display_df["Composite Sentiment"].round(3)
            
            neg_table.set_data(display_df)
    
    def _show_word_analysis(self, result: Dict, parent: tk.Frame):
        """Show word analysis by sentiment category."""
        word_analysis = result.get("word_analysis", {})
        
        if not word_analysis:
            tk.Label(parent, text="No word analysis data available", 
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg=self.theme["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.theme["bg_card"])
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Display word analysis for each category
        category_colors = {
            "Positive": "#2ecc71",
            "Negative": "#e74c3c",
            "Neutral": "#3498db",
        }
        
        for category, data in word_analysis.items():
            card = Card(scrollable, title=f"📝 {category} Documents", theme=self.theme_name)
            card.pack(fill=tk.X, pady=4)
            
            info_text = (
                f"Documents: {data.get('documents', 0):,}\n"
                f"Total words: {data.get('total_words', 0):,}\n"
                f"Unique words: {data.get('unique_words', 0):,}"
            )
            tk.Label(card.content, text=info_text, font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    justify=tk.LEFT).pack(anchor=tk.W, pady=4)
            
            # Top words
            top_words = data.get("top_words", [])
            if top_words:
                tk.Label(card.content, text="Top words:", font=FONTS.get_font("body_bold"),
                        bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(anchor=tk.W)
                
                words_text = ", ".join([f"{word} ({count})" for word, count in top_words[:15]])
                tk.Label(card.content, text=words_text, font=FONTS.get_font("small"),
                        bg=self.theme["bg_card"], fg=self.theme["text_muted"],
                        wraplength=400, justify=tk.LEFT).pack(anchor=tk.W, pady=2)
    
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
            filetypes=[("Excel files", "*.xlsx")], title="Save Sentiment Results")
        if not filename:
            return
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                result["sentiment_df"].to_excel(writer, sheet_name="Sentiment Scores", index=False)
                
                if result.get("top_positive") is not None:
                    result["top_positive"].to_excel(writer, sheet_name="Top Positive", index=False)
                
                if result.get("top_negative") is not None:
                    result["top_negative"].to_excel(writer, sheet_name="Top Negative", index=False)
                
                if result.get("temporal_trends") is not None:
                    result["temporal_trends"].to_excel(writer, sheet_name="Temporal Trends", index=False)
                
                if result.get("source_analysis") is not None:
                    result["source_analysis"].to_excel(writer, sheet_name="By Source", index=False)
                
                if result.get("correlations") is not None:
                    result["correlations"].to_excel(writer, sheet_name="Correlations")
                
                # Statistics
                stats_df = pd.DataFrame([result.get("statistics", {})])
                stats_df.to_excel(writer, sheet_name="Statistics", index=False)
                
            messagebox.showinfo("Success", f"Results exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def _save_plots(self, result: Dict):
        """Save plots to files."""
        base = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG files", "*.png")], title="Save Sentiment Plots")
        if not base:
            return
        if base.endswith(('.png', '.pdf', '.svg')):
            base = base.rsplit('.', 1)[0]
        
        try:
            if HAS_PLOTS:
                plot_sentiment_distribution(result, filename=base + "_distribution", show=False)
                plot_sentiment_categories(result, filename=base + "_categories", show=False)
                plot_sentiment_temporal(result, filename=base + "_temporal", show=False)
                plot_sentiment_certainty(result, filename=base + "_certainty", show=False)
                if result.get("correlations") is not None:
                    plot_sentiment_heatmap(result, filename=base + "_correlations", show=False)
                
                messagebox.showinfo("Success", f"Plots saved to:\n{base}_*.png")
            else:
                messagebox.showwarning("Warning", "Plotting module not available")
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
        error_color = self.theme.get("error", self.theme.get("danger", "#e74c3c"))
        tk.Label(self.results_tab, text=f"❌ Error\n\n{message}", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=error_color, justify=tk.CENTER,
                wraplength=400).pack(expand=True)
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
SENTIMENT ANALYSIS
══════════════════

Analyze sentiment, tone, and certainty in scientific text.

SENTIMENT METHODS
─────────────────
• VADER (Valence Aware Dictionary)
  - Compound score: -1 to +1
  - Optimized for short texts
  - Handles negations, intensifiers
  
• TextBlob
  - Polarity: -1 (negative) to +1 (positive)
  - Subjectivity: 0 (objective) to 1 (subjective)

SCIENTIFIC CERTAINTY
────────────────────
Domain-specific analysis for academic text:

• Certainty Score
  Presence of definite language:
  "demonstrate", "prove", "establish"
  
• Hedging Score  
  Tentative/cautious language:
  "may", "might", "suggests", "possibly"
  
• Boosting
  Intensifying language:
  "strongly", "clearly", "significantly"

OUTPUT METRICS
──────────────
• Compound: Overall VADER sentiment
• Polarity: TextBlob positive/negative
• Subjectivity: Objective vs subjective
• Certainty: Scientific confidence level
• Hedging: Cautious language rate

VISUALIZATIONS
──────────────
• Sentiment Distribution: Score histograms
• Sentiment Categories: Pos/Neg/Neutral counts
• Temporal Trends: Sentiment over time
• Certainty Analysis: Hedging vs boosting
• By Source: Journal-level sentiment
• Heatmap: Multi-dimensional view

TEXT FIELDS
───────────
• Titles: Usually neutral
• Abstracts: Most variation
• Can combine multiple fields

INTERPRETATION
──────────────
Academic text characteristics:
• Generally more neutral than social media
• Hedging is common and expected
• High certainty may indicate strong findings
• Compare within your field's norms
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
