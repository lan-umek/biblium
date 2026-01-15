"""
Topic Modeling Panel
====================

Comprehensive topic modeling analysis with LDA, NMF, and LSA models.
Includes coherence analysis, visualizations, and trend analysis.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import threading

from biblium.gui.panels.base import BasePanel
from biblium.gui.config import FONTS, get_theme
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame
from biblium.gui.widgets.forms import LabeledEntry, LabeledCombobox, LabeledSpinbox
from biblium.gui.widgets.cards import Card, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ActionButton

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TopicModelingPanel(BasePanel):
    """
    Panel for topic modeling analysis.
    
    Supports LDA, NMF, and LSA models with automatic topic selection,
    coherence analysis, and various visualizations.
    """
    
    title = "Topic Modeling"
    icon = "📚"
    description = "Discover latent topics in your corpus"
    requires_data = True
    
    MODELS = ["LDA", "NMF", "LSA"]
    
    TEXT_COLUMNS = [
        "Processed Abstract", "Abstract",
        "Processed Title", "Title",
        "Author Keywords", "Processed Author Keywords",
    ]
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._result = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Text Selection Card
        text_card = Card(self.options_content, title="📄 Text Selection", theme=self.theme_name)
        text_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.text_combo = LabeledCombobox(
            text_card.content, label="Text Column:",
            values=self.TEXT_COLUMNS, default="Processed Abstract",
            theme=self.theme_name, label_width=14,
            tooltip="Column containing text to model"
        )
        self.text_combo.pack(fill=tk.X, pady=4)
        
        # Model Card
        model_card = Card(self.options_content, title="🔬 Model Settings", theme=self.theme_name)
        model_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.model_combo = LabeledCombobox(
            model_card.content, label="Model Type:",
            values=self.MODELS, default="LDA",
            theme=self.theme_name, label_width=14,
            tooltip="LDA: Probabilistic\nNMF: Non-negative\nLSA: SVD-based"
        )
        self.model_combo.pack(fill=tk.X, pady=4)
        
        # Auto-select topics
        self.auto_topics_var = tk.BooleanVar(value=True)
        auto_frame = tk.Frame(model_card.content, bg=self.theme["bg_card"])
        auto_frame.pack(fill=tk.X, pady=4)
        
        tk.Checkbutton(
            auto_frame, text="Auto-select number of topics",
            variable=self.auto_topics_var, bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], font=FONTS.get_font("body"),
            activebackground=self.theme["bg_card"],
            command=self._on_auto_topics_change
        ).pack(side=tk.LEFT)
        
        # Number of topics
        self.n_topics_spin = LabeledSpinbox(
            model_card.content, label="Number of Topics:",
            from_=2, to=30, default=5,
            theme=self.theme_name, label_width=16,
        )
        self.n_topics_spin.pack(fill=tk.X, pady=4)
        
        # Max topics for auto-selection
        self.max_topics_spin = LabeledSpinbox(
            model_card.content, label="Max Topics (auto):",
            from_=5, to=30, default=15,
            theme=self.theme_name, label_width=16,
        )
        self.max_topics_spin.pack(fill=tk.X, pady=4)
        
        # Advanced Options (collapsible)
        advanced_card = Card(self.options_content, title="⚙️ Advanced Options", theme=self.theme_name)
        advanced_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.max_features_spin = LabeledSpinbox(
            advanced_card.content, label="Max Features:",
            from_=500, to=20000, default=5000,
            theme=self.theme_name, label_width=14,
        )
        self.max_features_spin.pack(fill=tk.X, pady=4)
        
        self.min_df_spin = LabeledSpinbox(
            advanced_card.content, label="Min Doc Freq:",
            from_=1, to=20, default=2,
            theme=self.theme_name, label_width=14,
            tooltip="Minimum documents a term must appear in"
        )
        self.min_df_spin.pack(fill=tk.X, pady=4)
        
        # N-gram range
        ngram_frame = tk.Frame(advanced_card.content, bg=self.theme["bg_card"])
        ngram_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            ngram_frame, text="N-gram Range:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], width=14, anchor="w"
        ).pack(side=tk.LEFT)
        
        self.ngram_min_spin = tk.Spinbox(
            ngram_frame, from_=1, to=3, width=5,
            font=FONTS.get_font("body")
        )
        self.ngram_min_spin.delete(0, tk.END)
        self.ngram_min_spin.insert(0, "1")
        self.ngram_min_spin.pack(side=tk.LEFT, padx=2)
        
        tk.Label(
            ngram_frame, text="to",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=4)
        
        self.ngram_max_spin = tk.Spinbox(
            ngram_frame, from_=1, to=4, width=5,
            font=FONTS.get_font("body")
        )
        self.ngram_max_spin.delete(0, tk.END)
        self.ngram_max_spin.insert(0, "2")
        self.ngram_max_spin.pack(side=tk.LEFT, padx=2)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Run Topic Modeling", icon="📚",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        # Export
        export_card = Card(self.options_content, title="📥 Export", theme=self.theme_name)
        export_card.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Button(
            export_card.content, text="Export Results to Excel",
            command=self._export_results,
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            font=FONTS.get_font("body"), relief=tk.FLAT, cursor="hand2"
        ).pack(fill=tk.X, pady=4)
        
        # Initialize UI state
        self._on_auto_topics_change()
    
    def _on_auto_topics_change(self):
        """Handle auto-topics checkbox change."""
        auto = self.auto_topics_var.get()
        if auto:
            self.n_topics_spin.spinbox.config(state="disabled")
        else:
            self.n_topics_spin.spinbox.config(state="normal")
    
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
            "🎯 Topic Modeling\n\n"
            "Discover latent themes in your corpus.\n\n"
            "Features:\n"
            "• LDA (Latent Dirichlet Allocation)\n"
            "• NMF (Non-negative Matrix Factorization)\n"
            "• Topic coherence scores\n"
            "• Topic-document distributions\n"
            "\n"
            "Reveals hidden thematic structure.\n\n"
            "Steps:\n"
            "1. Load dataset with abstracts\n"
            "2. Select algorithm\n"
            "3. Set number of topics\n"
            "4. Click 'Run Topic Model'\n"
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
        """Run topic modeling analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        text_column = self.text_combo.get()
        
        # Check if column exists
        if text_column not in self.bib.df.columns:
            # Try fallbacks
            fallbacks = ["Abstract", "Processed Title", "Title"]
            found = False
            for fb in fallbacks:
                if fb in self.bib.df.columns:
                    text_column = fb
                    found = True
                    break
            if not found:
                messagebox.showerror("Error", f"No suitable text column found.")
                return
        
        self._show_loading("Running topic modeling...")
        
        def do_analysis():
            try:
                print(f"[DEBUG] Starting topic modeling on {text_column}")
                
                # Build parameters
                params = {
                    "text_column": text_column,
                    "model_type": self.model_combo.get(),
                    "max_features": self.max_features_spin.get(),
                    "min_df": self.min_df_spin.get(),
                    "max_df": 0.95,
                    "ngram_range": (int(self.ngram_min_spin.get()), int(self.ngram_max_spin.get())),
                    "auto_select_topics": self.auto_topics_var.get(),
                }
                
                if self.auto_topics_var.get():
                    params["n_topics"] = None
                    params["max_topics"] = self.max_topics_spin.get()
                else:
                    params["n_topics"] = self.n_topics_spin.get()
                    params["max_topics"] = params["n_topics"]
                
                print(f"[DEBUG] Topic modeling params: {params}")
                
                # Call extended topic modeling
                if hasattr(self.bib, 'get_topics_extended'):
                    result = self.bib.get_topics_extended(**params)
                else:
                    from biblium import utilsbib
                    result = utilsbib.topic_modeling_extended(self.bib.df, **params)
                
                n_topics = result.get("optimal_n_topics", 0)
                print(f"[DEBUG] Topic modeling complete. {n_topics} topics found.")
                
                self._result = result
                self.after(0, lambda: self._on_success(result))
                
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[DEBUG] Error: {e}")
                print(f"[DEBUG] Traceback: {tb}")
                self.after(0, lambda: self._show_error(f"{e}\n\n{tb}"))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _on_success(self, result):
        """Display topic modeling results."""
        self._stop_active_spinners()
        self._safe_clear_results()
        
        if result.get("model") is None:
            tk.Label(
                self.results_tab, text="Topic modeling did not produce results.\nCheck that text column has content.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            ).pack(expand=True, pady=50)
            return
        
        # Create notebook for results
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Summary
        summary_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(summary_frame, text="Summary")
        self._create_summary_tab(summary_frame, result)
        
        # Tab 2: Topic Terms
        terms_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(terms_frame, text="Topic Terms")
        self._create_terms_tab(terms_frame, result)
        
        # Tab 3: Coherence (if available)
        if result.get("coherence_scores"):
            coherence_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(coherence_frame, text="Coherence")
            self._create_coherence_tab(coherence_frame, result)
        
        # Tab 4: Topic Similarity
        if result.get("topic_similarity") is not None:
            sim_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(sim_frame, text="Similarity")
            self._create_similarity_tab(sim_frame, result)
        
        # Tab 5: Trends (if year data available)
        if result.get("topic_trends") is not None and not result["topic_trends"].empty:
            trends_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(trends_frame, text="Trends")
            self._create_trends_tab(trends_frame, result)
        
        # Tab 6: Document Distribution
        if HAS_MATPLOTLIB:
            dist_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(dist_frame, text="Distribution")
            self._create_distribution_tab(dist_frame, result)
        
        # Tab 7: Word Clouds
        wordcloud_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(wordcloud_frame, text="Word Clouds")

        

        # Info tab

        info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])

        notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        self._create_wordcloud_tab(wordcloud_frame, result)
        
        n_topics = result.get("optimal_n_topics", 0)
        model_type = self.model_combo.get()
        messagebox.showinfo("Complete", f"{model_type} model fitted with {n_topics} topics.")
    
    def _create_summary_tab(self, parent, result):
        """Create summary statistics tab."""
        # Title
        tk.Label(
            parent, text="Topic Modeling Summary",
            font=FONTS.get_font("heading2"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(16, 8))
        
        # Stats cards - overall statistics
        n_topics = result.get("optimal_n_topics", 0)
        n_docs = len(result.get("df_out", pd.DataFrame()))
        coherence = result.get("topic_coherence", {})
        avg_coherence = np.mean(list(coherence.values())) if coherence else 0
        
        # Get total citations if available
        topic_stats = result.get("topic_stats", pd.DataFrame())
        total_citations = 0
        has_citations = False
        has_years = False
        
        if not topic_stats.empty:
            if "Total Citations" in topic_stats.columns:
                total_citations = topic_stats["Total Citations"].sum()
                has_citations = True
            if "Avg Year" in topic_stats.columns:
                has_years = True
        
        grid = CardGrid(parent, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, padx=16, pady=8)
        
        grid.add_card(StatsCard(grid, "Topics", f"{n_topics}", "📚", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Documents", f"{n_docs:,}", "📄", self.theme_name))
        grid.add_card(StatsCard(grid, "Model", self.model_combo.get(), "🔬", self.theme_name))
        grid.add_card(StatsCard(grid, "Avg Coherence", f"{avg_coherence:.3f}", "📊", self.theme_name))
        
        if has_citations:
            grid2 = CardGrid(parent, columns=4, theme=self.theme_name)
            grid2.pack(fill=tk.X, padx=16, pady=4)
            grid2.add_card(StatsCard(grid2, "Total Citations", f"{int(total_citations):,}", "📈", self.theme_name))
        
        # Topic statistics table
        if not topic_stats.empty:
            tk.Label(
                parent, text="Topic Statistics",
                font=FONTS.get_font("heading3"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]
            ).pack(pady=(16, 8))
            
            # Reorder columns for better display
            display_cols = ["Topic", "Dominant Documents", "Document Share (%)"]
            if has_citations:
                display_cols.extend(["Total Citations", "Avg Citations"])
            if has_years:
                display_cols.extend(["Avg Year", "Min Year", "Max Year"])
            display_cols.extend(["Avg Weight", "Top Terms"])
            
            # Only include columns that exist
            display_cols = [c for c in display_cols if c in topic_stats.columns]
            display_df = topic_stats[display_cols].copy()
            
            # Format numeric columns to 2 decimal places
            if "Document Share (%)" in display_df.columns:
                display_df["Document Share (%)"] = display_df["Document Share (%)"].round(2)
            if "Avg Citations" in display_df.columns:
                display_df["Avg Citations"] = display_df["Avg Citations"].round(2)
            if "Avg Year" in display_df.columns:
                display_df["Avg Year"] = display_df["Avg Year"].round(2)
            if "Avg Weight" in display_df.columns:
                display_df["Avg Weight"] = display_df["Avg Weight"].round(2)
            
            table = DataTable(parent, theme=self.theme_name)
            table.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
            table.set_data(display_df)
    
    def _create_terms_tab(self, parent, result):
        """Create topic terms tab."""
        topics_df = result.get("topics_df", pd.DataFrame())
        
        if topics_df.empty:
            tk.Label(
                parent, text="No topic terms available.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
            return
        
        # Topic filter
        control_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        control_frame.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            control_frame, text="Filter by Topic:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        topics = ["All"] + sorted(topics_df["Topic"].unique().tolist())
        self._topic_filter_var = tk.StringVar(value="All")
        topic_combo = ttk.Combobox(
            control_frame, textvariable=self._topic_filter_var,
            values=topics, state="readonly", width=15
        )
        topic_combo.pack(side=tk.LEFT, padx=4)
        
        # Table
        table_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        self._terms_table = DataTable(table_frame, theme=self.theme_name)
        self._terms_table.pack(fill=tk.BOTH, expand=True)
        self._topics_df = topics_df
        self._update_terms_table()
        
        topic_combo.bind("<<ComboboxSelected>>", lambda e: self._update_terms_table())
    
    def _update_terms_table(self):
        """Update terms table based on filter."""
        if not hasattr(self, '_topics_df'):
            return
        
        filter_val = self._topic_filter_var.get()
        
        if filter_val == "All":
            filtered = self._topics_df
        else:
            filtered = self._topics_df[self._topics_df["Topic"] == filter_val]
        
        # Show top 15 terms per topic
        if filter_val == "All":
            filtered = filtered.groupby("Topic").head(15)
        
        self._terms_table.set_data(filtered)
    
    def _create_coherence_tab(self, parent, result):
        """Create coherence scores tab."""
        coherence_scores = result.get("coherence_scores", {})
        
        if not coherence_scores:
            tk.Label(
                parent, text="No coherence scores available.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
            return
        
        if HAS_MATPLOTLIB:
            plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True)
            
            fig, ax = plot_frame.get_figure()
            
            ks = sorted(coherence_scores.keys())
            scores = [coherence_scores[k] for k in ks]
            
            ax.plot(ks, scores, marker='o', color='#3b82f6', linewidth=2, markersize=8)
            
            # Highlight optimal
            optimal_k = result.get("optimal_n_topics", ks[np.argmax(scores)])
            optimal_score = coherence_scores.get(optimal_k, max(scores))
            ax.scatter([optimal_k], [optimal_score], color='red', s=150, zorder=5, 
                       label=f'Selected: k={optimal_k}')
            ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.5)
            
            ax.set_xlabel("Number of Topics")
            ax.set_ylabel("Coherence Score (UMass)")
            ax.set_title("Topic Coherence by Number of Topics")
            ax.legend()
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            fig.tight_layout()
            plot_frame.refresh()
    
    def _create_similarity_tab(self, parent, result):
        """Create topic similarity heatmap tab."""
        similarity_df = result.get("topic_similarity")
        
        if similarity_df is None or similarity_df.empty:
            tk.Label(
                parent, text="No similarity data available.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
            return
        
        if HAS_MATPLOTLIB:
            plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True)
            
            fig, ax = plot_frame.get_figure()
            
            import seaborn as sns
            sns.heatmap(
                similarity_df, 
                cmap="RdYlBu_r", 
                annot=True, 
                fmt=".2f",
                square=True,
                linewidths=0.5,
                cbar_kws={"label": "Cosine Similarity"},
                ax=ax
            )
            ax.set_title("Topic Similarity Matrix")
            
            fig.tight_layout()
            plot_frame.refresh()
    
    def _create_trends_tab(self, parent, result):
        """Create topic trends over time tab."""
        trends_df = result.get("topic_trends", pd.DataFrame())
        
        if trends_df.empty:
            tk.Label(
                parent, text="No trend data available (requires Year column).",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
            return
        
        if HAS_MATPLOTLIB:
            plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True)
            
            fig, ax = plot_frame.get_figure()
            
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(trends_df.columns)))
            
            ax.stackplot(
                trends_df.index, 
                trends_df.T.values,
                labels=trends_df.columns,
                colors=colors,
                alpha=0.8
            )
            
            ax.set_xlabel("Year")
            ax.set_ylabel("Proportion")
            ax.set_title("Topic Prevalence Over Time")
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            fig.tight_layout()
            plot_frame.refresh()
    
    def _create_distribution_tab(self, parent, result):
        """Create document distribution plot."""
        topic_stats = result.get("topic_stats", pd.DataFrame())
        
        if topic_stats.empty:
            tk.Label(
                parent, text="No distribution data available.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
            return
        
        plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        fig, ax = plot_frame.get_figure()
        
        topics = topic_stats["Topic"].tolist()
        counts = topic_stats["Dominant Documents"].tolist()
        
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(topics)))
        bars = ax.bar(topics, counts, color=colors, edgecolor='white')
        
        ax.set_xlabel("Topic")
        ax.set_ylabel("Number of Documents")
        ax.set_title("Document Distribution Across Topics")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        plot_frame.refresh()
    
    def _create_wordcloud_tab(self, parent, result):
        """Create word clouds for topics with comparative coloring."""
        topics_df = result.get("topics_df", pd.DataFrame())
        
        if topics_df.empty:
            tk.Label(
                parent, text="No topic data for word clouds.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
            return
        
        try:
            from wordcloud import WordCloud
            HAS_WORDCLOUD = True
        except ImportError:
            HAS_WORDCLOUD = False
        
        if not HAS_WORDCLOUD or not HAS_MATPLOTLIB:
            tk.Label(
                parent, text="Word clouds require 'wordcloud' package.\npip install wordcloud",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
            return
        
        # Store for later use
        self._wordcloud_topics_df = topics_df
        self._wordcloud_result = result
        
        # Create notebook for different wordcloud views
        wc_notebook = ttk.Notebook(parent)
        wc_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Standard word clouds
        std_frame = tk.Frame(wc_notebook, bg=self.theme["bg_card"])
        wc_notebook.add(std_frame, text="Standard")
        self._create_standard_wordclouds(std_frame, topics_df)
        
        # Tab 2: Comparative word clouds (only if > 1 topic)
        topics = sorted(topics_df["Topic"].unique())
        if len(topics) > 1:
            comp_frame = tk.Frame(wc_notebook, bg=self.theme["bg_card"])
            wc_notebook.add(comp_frame, text="Comparative")
            self._create_comparative_wordclouds(comp_frame, topics_df, result)
    
    def _create_standard_wordclouds(self, parent, topics_df):
        """Create standard word clouds for each topic."""
        from wordcloud import WordCloud
        
        plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        fig, ax = plot_frame.get_figure()
        
        # Determine term column
        term_col = "Term" if "Term" in topics_df.columns else "Word"
        
        topics = sorted(topics_df["Topic"].unique())
        n_topics = len(topics)
        n_cols = min(3, n_topics)
        n_rows = int(np.ceil(n_topics / n_cols))
        
        # Clear and create subplots
        fig.clear()
        axes = fig.subplots(n_rows, n_cols) if n_topics > 1 else [[fig.add_subplot(111)]]
        
        if n_topics == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        
        for i, topic in enumerate(topics):
            row, col = i // n_cols, i % n_cols
            ax = axes[row][col] if n_rows > 1 else axes[0][col]
            
            topic_data = topics_df[topics_df["Topic"] == topic]
            word_weights = dict(zip(topic_data[term_col], topic_data["Weight"]))
            
            if word_weights:
                wc = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='viridis',
                    max_words=25,
                ).generate_from_frequencies(word_weights)
                
                ax.imshow(wc, interpolation='bilinear')
            
            ax.set_title(str(topic), fontsize=10)
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(n_topics, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            if n_rows > 1:
                axes[row][col].axis('off')
            else:
                axes[0][col].axis('off')
        
        fig.tight_layout()
        plot_frame.refresh()
    
    def _create_comparative_wordclouds(self, parent, topics_df, result):
        """Create comparative word clouds colored by association to reference topic."""
        from wordcloud import WordCloud
        import matplotlib.colors as mcolors
        
        # Control frame for reference topic selection
        control_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        control_frame.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            control_frame, text="Reference Topic:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        topics = sorted(topics_df["Topic"].unique())
        
        # Default to topic with most documents
        topic_stats = result.get("topic_stats", pd.DataFrame())
        if not topic_stats.empty and "Dominant Documents" in topic_stats.columns:
            default_ref = topic_stats.loc[topic_stats["Dominant Documents"].idxmax(), "Topic"]
        else:
            default_ref = topics[0]
        
        self._ref_topic_var = tk.StringVar(value=str(default_ref))
        ref_combo = ttk.Combobox(
            control_frame, textvariable=self._ref_topic_var,
            values=[str(t) for t in topics], state="readonly", width=15
        )
        ref_combo.pack(side=tk.LEFT, padx=4)
        
        tk.Button(
            control_frame, text="Update",
            command=lambda: self._update_comparative_wordclouds(parent),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            font=FONTS.get_font("body"), relief=tk.FLAT, cursor="hand2"
        ).pack(side=tk.LEFT, padx=8)
        
        # Info label
        tk.Label(
            control_frame, 
            text="Colors: Blue = associated with reference topic, Red = associated with comparison topic",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        ).pack(side=tk.LEFT, padx=16)
        
        # Plot frame
        self._comp_plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        self._comp_plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial plot
        self._update_comparative_wordclouds(parent)
    
    def _update_comparative_wordclouds(self, parent):
        """Update comparative word clouds based on selected reference topic."""
        from wordcloud import WordCloud
        import matplotlib.colors as mcolors
        
        topics_df = self._wordcloud_topics_df
        result = self._wordcloud_result
        
        term_col = "Term" if "Term" in topics_df.columns else "Word"
        topics = sorted(topics_df["Topic"].unique())
        ref_topic = self._ref_topic_var.get()
        
        # Get other topics (excluding reference)
        other_topics = [t for t in topics if str(t) != ref_topic]
        
        if not other_topics:
            return
        
        # Get topic-term matrix for computing associations
        topic_term_matrix = result.get("topic_term_matrix")
        feature_names = result.get("feature_names", [])
        
        if topic_term_matrix is None or len(feature_names) == 0:
            # Fallback: build from topics_df
            topic_term_matrix, feature_names = self._build_topic_term_from_df(topics_df, term_col, topics)
        
        # Convert feature_names to list for indexing
        feature_names_list = list(feature_names) if hasattr(feature_names, '__iter__') else []
        
        # Find reference topic index
        ref_idx = None
        for i, t in enumerate(topics):
            if str(t) == ref_topic:
                ref_idx = i
                break
        
        if ref_idx is None:
            print(f"[DEBUG] Reference topic {ref_topic} not found in topics: {topics}")
            return
        
        # Setup figure
        fig, ax = self._comp_plot_frame.get_figure()
        fig.clear()
        
        n_other = len(other_topics)
        n_cols = min(3, n_other)
        n_rows = int(np.ceil(n_other / n_cols))
        
        if n_other == 1:
            axes = [[fig.add_subplot(111)]]
        else:
            axes_flat = fig.subplots(n_rows, n_cols)
            if n_rows == 1:
                axes = [list(axes_flat) if n_cols > 1 else [axes_flat]]
            elif n_cols == 1:
                axes = [[ax] for ax in axes_flat]
            else:
                axes = [list(row) for row in axes_flat]
        
        # Create diverging colormap (blue for ref, red for comparison)
        cmap_div = plt.cm.get_cmap('RdYlBu_r')  # Red to Blue
        
        for i, comp_topic in enumerate(other_topics):
            row, col = i // n_cols, i % n_cols
            ax = axes[row][col]
            
            # Find comparison topic index
            comp_idx = None
            for j, t in enumerate(topics):
                if str(t) == str(comp_topic):
                    comp_idx = j
                    break
            
            if comp_idx is None:
                ax.axis('off')
                continue
            
            # Compute word associations
            ref_weights = topic_term_matrix[ref_idx]
            comp_weights = topic_term_matrix[comp_idx]
            
            # Get words that appear in either topic
            topic_data_ref = topics_df[topics_df["Topic"] == ref_topic]
            topic_data_comp = topics_df[topics_df["Topic"] == comp_topic]
            
            ref_words = set(topic_data_ref[term_col].values)
            comp_words = set(topic_data_comp[term_col].values)
            all_words = ref_words | comp_words
            
            # Build word frequencies and colors
            word_weights = {}
            word_colors = {}
            
            for word in all_words:
                # Get weights from topic_term_matrix if possible
                if word in feature_names_list:
                    word_idx = feature_names_list.index(word)
                    w_ref = float(ref_weights[word_idx])
                    w_comp = float(comp_weights[word_idx])
                else:
                    # Fallback to topics_df
                    w_ref_arr = topic_data_ref[topic_data_ref[term_col] == word]["Weight"].values
                    w_ref = float(w_ref_arr[0]) if len(w_ref_arr) > 0 else 0.0
                    w_comp_arr = topic_data_comp[topic_data_comp[term_col] == word]["Weight"].values
                    w_comp = float(w_comp_arr[0]) if len(w_comp_arr) > 0 else 0.0
                
                # Frequency is max of both
                freq = max(w_ref, w_comp, 0.001)
                word_weights[word] = freq
                
                # Association score: -1 (fully ref) to +1 (fully comp)
                total = w_ref + w_comp
                if total > 0:
                    assoc = (w_comp - w_ref) / total  # -1 to 1
                else:
                    assoc = 0.0
                
                word_colors[word] = assoc
            
            if not word_weights:
                ax.axis('off')
                continue
            
            # Create color function with captured word_colors dict
            # Need to create a new closure for each iteration
            def make_color_func(colors_dict, colormap):
                def color_func(word, **kwargs):
                    assoc = colors_dict.get(word, 0)
                    # Map -1 to 1 -> 0 to 1 for colormap
                    norm_val = (assoc + 1) / 2
                    rgba = colormap(norm_val)
                    return tuple(int(c * 255) for c in rgba[:3])
                return color_func
            
            color_func = make_color_func(word_colors.copy(), cmap_div)
            
            try:
                wc = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    max_words=30,
                    color_func=color_func,
                ).generate_from_frequencies(word_weights)
                
                ax.imshow(wc, interpolation='bilinear')
                ax.set_title(f"{ref_topic} vs {comp_topic}", fontsize=10)
            except Exception as e:
                print(f"[DEBUG] WordCloud error for {comp_topic}: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)[:30]}", ha='center', va='center')
            
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(n_other, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row][col].axis('off')
        
        # Add colorbar legend
        fig.tight_layout()
        
        # Add a simple legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2166ac', label=f'→ {ref_topic}'),
            Patch(facecolor='#b2182b', label='→ Other topic'),
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        self._comp_plot_frame.refresh()
    
    def _build_topic_term_from_df(self, topics_df, term_col, topics):
        """Build topic-term matrix from topics_df."""
        # Get all unique terms
        all_terms = topics_df[term_col].unique()
        feature_names = np.array(sorted(all_terms))
        
        # Build matrix
        n_topics = len(topics)
        n_terms = len(feature_names)
        matrix = np.zeros((n_topics, n_terms))
        
        term_to_idx = {term: i for i, term in enumerate(feature_names)}
        
        for i, topic in enumerate(topics):
            topic_data = topics_df[topics_df["Topic"] == topic]
            for _, row in topic_data.iterrows():
                term = row[term_col]
                weight = row["Weight"]
                if term in term_to_idx:
                    matrix[i, term_to_idx[term]] = weight
        
        return matrix, feature_names
    
    def _export_results(self):
        """Export topic modeling results."""
        if self._result is None:
            messagebox.showwarning("No Results", "Run topic modeling first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Export Topic Modeling Results"
        )
        
        if not filename:
            return
        
        try:
            result = self._result
            
            with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                # Topic terms
                if result.get("topics_df") is not None:
                    result["topics_df"].to_excel(writer, sheet_name="Topic Terms", index=False)
                
                # Topic statistics
                if result.get("topic_stats") is not None:
                    result["topic_stats"].to_excel(writer, sheet_name="Topic Stats", index=False)
                
                # Similarity matrix
                if result.get("topic_similarity") is not None:
                    result["topic_similarity"].to_excel(writer, sheet_name="Similarity")
                
                # Trends
                if result.get("topic_trends") is not None and not result["topic_trends"].empty:
                    result["topic_trends"].to_excel(writer, sheet_name="Trends")
                
                # Coherence scores
                if result.get("coherence_scores"):
                    coh_df = pd.DataFrame([
                        {"K": k, "Coherence": v} 
                        for k, v in result["coherence_scores"].items()
                    ])
                    coh_df.to_excel(writer, sheet_name="Coherence Scores", index=False)
                
                # Per-topic coherence
                if result.get("topic_coherence"):
                    tc_df = pd.DataFrame([
                        {"Topic": k, "Coherence": v}
                        for k, v in result["topic_coherence"].items()
                    ])
                    tc_df.to_excel(writer, sheet_name="Topic Coherence", index=False)
            
            messagebox.showinfo("Success", f"Results exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
TOPIC MODELING
══════════════

Discover latent themes using unsupervised machine learning.

AVAILABLE MODELS
────────────────
• LDA (Latent Dirichlet Allocation)
  - Probabilistic generative model
  - Topics as word distributions
  - Documents as topic mixtures
  - Most popular method
  
• NMF (Non-negative Matrix Factorization)
  - Matrix decomposition approach
  - Often produces cleaner topics
  - Faster computation
  - Deterministic results
  
• LSA (Latent Semantic Analysis)
  - SVD-based dimensionality reduction
  - Oldest method
  - Good for document similarity
  - Less interpretable topics

PARAMETERS
──────────
• Number of topics (K): 5-50 typical
• Auto-select: Uses coherence optimization
• Min document frequency: Filter rare terms
• Max document frequency: Filter common terms
• Preprocessing: Stemming, stopwords

TOPIC COHERENCE
───────────────
Measures topic quality (higher = better):
• C_v: Normalized PMI-based
• UMass: Co-occurrence based
• Used for automatic K selection

OUTPUT
──────
• Top words per topic
• Topic-document matrix
• Topic prevalence
• Coherence scores
• Word clouds
• Topic trends over time

BEST PRACTICES
──────────────
• Start with K=10-20
• Use coherence for K selection
• Run multiple times (LDA varies)
• Validate with domain expertise
• Remove generic stopwords
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

    def set_bib(self, bib):
        """Set the bibliometric data object."""
        self.bib = bib
        self._result = None
        
        # Update text column options
        if bib and hasattr(bib, 'df'):
            available_cols = []
            for col in self.TEXT_COLUMNS:
                if col in bib.df.columns:
                    available_cols.append(col)
            
            # Add other text columns
            for col in bib.df.columns:
                if col not in available_cols and bib.df[col].dtype == 'object':
                    # Check if it looks like a text column
                    sample = bib.df[col].dropna().head(5)
                    if any(len(str(v)) > 50 for v in sample):
                        available_cols.append(col)
            
            if available_cols:
                self.text_combo.combobox['values'] = available_cols
                if "Processed Abstract" in available_cols:
                    self.text_combo.set("Processed Abstract")
                elif "Abstract" in available_cols:
                    self.text_combo.set("Abstract")
                elif available_cols:
                    self.text_combo.set(available_cols[0])
