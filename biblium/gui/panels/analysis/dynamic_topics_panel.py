"""
Dynamic Topic Modeling Panel
============================

Panel for Sequential and Dynamic Topic Modeling analysis.
Tracks how topics evolve over time.
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
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DynamicTopicsPanel(BasePanel):
    """Panel for Dynamic and Sequential Topic Modeling."""
    
    title = "Dynamic Topics"
    icon = "🔄"
    description = "Track topic evolution over time"
    requires_data = True
    
    MODELS = ["LDA", "NMF", "LSA"]
    METHODS = ["Dynamic (DTM)", "Sequential (STM)"]
    TEXT_COLUMNS = ["Processed Abstract", "Abstract", "Processed Title", "Title"]
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._result = None
        self._method = "DTM"
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Method Selection
        method_card = Card(self.options_content, title="🔄 Method", theme=self.theme_name)
        method_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.method_combo = LabeledCombobox(
            method_card.content, label="Analysis Type:",
            values=self.METHODS, default="Dynamic (DTM)",
            theme=self.theme_name, label_width=14,
        )
        self.method_combo.pack(fill=tk.X, pady=4)
        self.method_combo.combobox.bind("<<ComboboxSelected>>", self._on_method_change)
        
        # Text Selection
        text_card = Card(self.options_content, title="📄 Text Selection", theme=self.theme_name)
        text_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.text_combo = LabeledCombobox(
            text_card.content, label="Text Column:",
            values=self.TEXT_COLUMNS, default="Processed Abstract",
            theme=self.theme_name, label_width=14,
        )
        self.text_combo.pack(fill=tk.X, pady=4)
        
        self.time_combo = LabeledCombobox(
            text_card.content, label="Time Column:",
            values=["Year", "Publication Year", "PY"],
            default="Year", theme=self.theme_name, label_width=14,
        )
        self.time_combo.pack(fill=tk.X, pady=4)
        
        # Model Settings
        model_card = Card(self.options_content, title="🔬 Model Settings", theme=self.theme_name)
        model_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.model_combo = LabeledCombobox(
            model_card.content, label="Model Type:",
            values=self.MODELS, default="LDA",
            theme=self.theme_name, label_width=14,
        )
        self.model_combo.pack(fill=tk.X, pady=4)
        
        self.n_topics_spin = LabeledSpinbox(
            model_card.content, label="Number of Topics:",
            from_=2, to=20, default=5,
            theme=self.theme_name, label_width=16,
        )
        self.n_topics_spin.pack(fill=tk.X, pady=4)
        
        self.n_slices_spin = LabeledSpinbox(
            model_card.content, label="Time Slices:",
            from_=3, to=20, default=5,
            theme=self.theme_name, label_width=16,
        )
        self.n_slices_spin.pack(fill=tk.X, pady=4)
        
        # Advanced
        advanced_card = Card(self.options_content, title="⚙️ Advanced", theme=self.theme_name)
        advanced_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.max_features_spin = LabeledSpinbox(
            advanced_card.content, label="Max Features:",
            from_=500, to=10000, default=3000,
            theme=self.theme_name, label_width=14,
        )
        self.max_features_spin.pack(fill=tk.X, pady=4)
        
        self.variance_frame = tk.Frame(advanced_card.content, bg=self.theme["bg_card"])
        self.variance_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            self.variance_frame, text="Smoothing:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], width=14, anchor="w"
        ).pack(side=tk.LEFT)
        
        self.variance_scale = tk.Scale(
            self.variance_frame, from_=0.01, to=0.5,
            resolution=0.01, orient=tk.HORIZONTAL,
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            highlightthickness=0, length=150
        )
        self.variance_scale.set(0.1)
        self.variance_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Run Analysis", icon="🔄",
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
    
    def _on_method_change(self, event=None):
        """Handle method selection change."""
        is_dtm = "Dynamic" in self.method_combo.get()
        if is_dtm:
            self.n_slices_spin.pack(fill=tk.X, pady=4)
            self.variance_frame.pack(fill=tk.X, pady=4)
        else:
            self.n_slices_spin.pack_forget()
            self.variance_frame.pack_forget()
    
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
            "📅 Dynamic Topic Modeling\n\n"
            "Discover how topics evolve over time.\n\n"
            "Features:\n"
            "• Structural Topic Models (STM)\n"
            "• Dynamic Topic Models (DTM)\n"
            "• Topic prevalence over time\n"
            "• Emerging and declining topics\n"
            "\n"
            "Captures topic evolution and research fronts.\n\n"
            "Steps:\n"
            "1. Load dataset with abstracts and years\n"
            "2. Select model type\n"
            "3. Set number of topics\n"
            "4. Click 'Run Analysis'\n"
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
        """Run analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        method = self.method_combo.get()
        is_dtm = "Dynamic" in method
        self._method = "DTM" if is_dtm else "STM"
        
        text_column = self.text_combo.get()
        time_column = self.time_combo.get()
        
        # Check columns
        if text_column not in self.bib.df.columns:
            for fb in ["Abstract", "Processed Title", "Title"]:
                if fb in self.bib.df.columns:
                    text_column = fb
                    break
        
        if time_column not in self.bib.df.columns:
            for fb in ["Year", "Publication Year", "PY"]:
                if fb in self.bib.df.columns:
                    time_column = fb
                    break
        
        if time_column not in self.bib.df.columns:
            messagebox.showerror("Error", "No time/year column found.")
            return
        
        self._show_loading(f"Running {'Dynamic' if is_dtm else 'Sequential'} Topic Modeling...")
        
        def do_analysis():
            try:
                from biblium import utilsbib
                
                if is_dtm:
                    result = utilsbib.dynamic_topic_modeling(
                        df=self.bib.df,
                        text_column=text_column,
                        time_column=time_column,
                        n_topics=self.n_topics_spin.get(),
                        n_time_slices=self.n_slices_spin.get(),
                        model_type=self.model_combo.get(),
                        max_features=self.max_features_spin.get(),
                        chain_variance=self.variance_scale.get(),
                    )
                else:
                    result = utilsbib.sequential_topic_modeling(
                        df=self.bib.df,
                        text_column=text_column,
                        time_column=time_column,
                        n_topics=self.n_topics_spin.get(),
                        model_type=self.model_combo.get(),
                        max_features=self.max_features_spin.get(),
                    )
                
                self._result = result
                self.after(0, lambda: self._on_success(result))
                
            except Exception as e:
                import traceback
                self.after(0, lambda: self._show_error(f"{e}\n\n{traceback.format_exc()}"))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _on_success(self, result):
        """Display results."""
        self._stop_active_spinners()
        self._safe_clear_results()
        
        is_dtm = self._method == "DTM"
        
        if is_dtm and result.get("n_topics", 0) == 0:
            self._show_error("No topics found.")
            return
        if not is_dtm and not result.get("period_topics"):
            self._show_error("No topics found.")
            return
        
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary
        summary_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(summary_frame, text="Summary")
        self._create_summary_tab(summary_frame, result, is_dtm)
        
        # Prevalence
        prevalence_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(prevalence_frame, text="Prevalence")
        self._create_prevalence_tab(prevalence_frame, result, is_dtm)
        
        # Streams (DTM)
        if is_dtm and HAS_MATPLOTLIB:
            streams_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(streams_frame, text="Topic Streams")
            self._create_streams_tab(streams_frame, result)
        
        # Word Evolution (DTM)
        if is_dtm:
            word_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(word_frame, text="Word Evolution")
            self._create_word_evolution_tab(word_frame, result)
        
        # Global Topics (DTM) / Alignment (STM)
        if is_dtm:
            global_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(global_frame, text="Global Topics")
            self._create_global_topics_tab(global_frame, result)
        else:
            evo_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(evo_frame, text="Topic Alignment")
            self._create_evolution_tab(evo_frame, result)
            
            period_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(period_frame, text="Period Topics")
            self._create_period_topics_tab(period_frame, result)
        
        # Term Evolution tab (for both DTM and STM)
        if HAS_MATPLOTLIB:
            term_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(term_frame, text="Term Evolution")

            

            # Info tab

            info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])

            notebook.add(info_frame, text="ℹ️ Info")

            self._create_info_content(info_frame)
            self._create_term_evolution_tab(term_frame)
        
        n_topics = result.get("n_topics", len(result.get("period_topics", {})))
        messagebox.showinfo("Complete", f"Analysis complete with {n_topics} topics.")
    
    def _create_summary_tab(self, parent, result, is_dtm):
        """Create summary tab."""
        tk.Label(
            parent, text=f"{'Dynamic' if is_dtm else 'Sequential'} Topic Modeling Summary",
            font=FONTS.get_font("heading2"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(16, 8))
        
        grid = CardGrid(parent, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, padx=16, pady=8)
        
        n_topics = result.get("n_topics", 0)
        
        if is_dtm:
            n_slices = result.get("n_time_slices", 0)
            grid.add_card(StatsCard(grid, "Topics", f"{n_topics}", "📚", self.theme_name, accent=True))
            grid.add_card(StatsCard(grid, "Time Slices", f"{n_slices}", "📅", self.theme_name))
            grid.add_card(StatsCard(grid, "Method", "DTM", "🔄", self.theme_name))
            grid.add_card(StatsCard(grid, "Model", self.model_combo.get(), "🔬", self.theme_name))
            
            time_info = result.get("time_slice_info", pd.DataFrame())
            if not time_info.empty:
                tk.Label(parent, text="Time Slices", font=FONTS.get_font("heading3"),
                        bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=(16, 8))
                table = DataTable(parent, theme=self.theme_name)
                table.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
                table.set_data(time_info)
        else:
            n_periods = len(result.get("periods", []))
            grid.add_card(StatsCard(grid, "Topics/Period", f"{n_topics}", "📚", self.theme_name, accent=True))
            grid.add_card(StatsCard(grid, "Periods", f"{n_periods}", "📅", self.theme_name))
            grid.add_card(StatsCard(grid, "Method", "STM", "🔄", self.theme_name))
            grid.add_card(StatsCard(grid, "Model", self.model_combo.get(), "🔬", self.theme_name))
            
            period_stats = result.get("period_stats", pd.DataFrame())
            if not period_stats.empty:
                tk.Label(parent, text="Period Statistics", font=FONTS.get_font("heading3"),
                        bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=(16, 8))
                table = DataTable(parent, theme=self.theme_name)
                table.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
                table.set_data(period_stats)
    
    def _create_prevalence_tab(self, parent, result, is_dtm):
        """Create prevalence plot."""
        if not HAS_MATPLOTLIB:
            tk.Label(parent, text="Matplotlib required.", font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        prevalence_df = result.get("topic_prevalence_evolution" if is_dtm else "topic_prevalence", pd.DataFrame())
        time_col = "Time_Slice" if is_dtm else "Period"
        
        if prevalence_df.empty:
            tk.Label(parent, text="No prevalence data.", font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        control_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        control_frame.pack(fill=tk.X, padx=8, pady=8)
        
        self._stacked_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            control_frame, text="Stacked", variable=self._stacked_var,
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            command=lambda: self._update_prevalence(prevalence_df, time_col)
        ).pack(side=tk.LEFT, padx=8)
        
        self._prev_plot = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        self._prev_plot.pack(fill=tk.BOTH, expand=True)
        self._update_prevalence(prevalence_df, time_col)
    
    def _update_prevalence(self, df, time_col):
        """Update prevalence plot."""
        fig, ax = self._prev_plot.get_figure()
        ax.clear()
        
        pivot = df.pivot(index=time_col, columns='Topic', values='Prevalence').fillna(0)
        topics = pivot.columns.tolist()
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(topics)))
        
        if self._stacked_var.get():
            ax.stackplot(pivot.index, pivot.T.values, labels=topics, colors=colors, alpha=0.8)
        else:
            for i, t in enumerate(topics):
                ax.plot(pivot.index, pivot[t], label=t, color=colors[i], linewidth=2, marker='o')
        
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Prevalence")
        ax.set_title("Topic Prevalence Over Time")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        self._prev_plot.refresh()
    
    def _create_streams_tab(self, parent, result):
        """Create topic streams."""
        df = result.get("topic_prevalence_evolution", pd.DataFrame())
        if df.empty:
            tk.Label(parent, text="No data.", font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        plot = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True)
        fig, ax = plot.get_figure()
        
        pivot = df.pivot(index='Time_Slice', columns='Topic', values='Prevalence').fillna(0)
        pivot = pivot.div(pivot.sum(axis=1), axis=0)
        
        topics = pivot.columns.tolist()
        x = pivot.index.values
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(topics)))
        
        y_stack = np.row_stack([pivot[t].values for t in topics])
        baseline = -y_stack.sum(axis=0) / 2
        y_bottom = baseline.copy()
        
        for i, topic in enumerate(topics):
            y_top = y_bottom + pivot[topic].values
            ax.fill_between(x, y_bottom, y_top, label=topic, color=colors[i], alpha=0.8)
            y_bottom = y_top
        
        ax.set_xlabel("Time Slice")
        ax.set_ylabel("Relative Prevalence")
        ax.set_title("Topic Streams Over Time")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(False)
        fig.tight_layout()
        plot.refresh()
    
    def _create_word_evolution_tab(self, parent, result):
        """Create word evolution tab."""
        evo = result.get("topic_word_evolution", {})
        if not evo:
            tk.Label(parent, text="No data.", font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        control = tk.Frame(parent, bg=self.theme["bg_card"])
        control.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(control, text="Topic:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        
        topics = list(evo.keys())
        self._evo_var = tk.StringVar(value=topics[0] if topics else "")
        combo = ttk.Combobox(control, textvariable=self._evo_var, values=topics, state="readonly", width=15)
        combo.pack(side=tk.LEFT, padx=4)
        
        self._evo_table = DataTable(parent, theme=self.theme_name)
        self._evo_table.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._evo_data = evo
        
        def update(*args):
            t = self._evo_var.get()
            if t in self._evo_data:
                self._evo_table.set_data(self._evo_data[t])
        
        combo.bind("<<ComboboxSelected>>", update)
        update()
    
    def _create_global_topics_tab(self, parent, result):
        """Create global topics tab."""
        df = result.get("global_topics", pd.DataFrame())
        if df.empty:
            tk.Label(parent, text="No data.", font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        tk.Label(parent, text="Global Topic-Word Weights", font=FONTS.get_font("heading3"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=(16, 8))
        
        table = DataTable(parent, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        table.set_data(df)
    
    def _create_evolution_tab(self, parent, result):
        """Create topic alignment tab (STM)."""
        df = result.get("topic_evolution", pd.DataFrame())
        if df.empty:
            tk.Label(parent, text="No data.", font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        tk.Label(parent, text="Topic Alignment Across Periods", font=FONTS.get_font("heading3"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=(16, 8))
        
        table = DataTable(parent, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        table.set_data(df)
    
    def _create_period_topics_tab(self, parent, result):
        """Create period topics tab (STM)."""
        period_topics = result.get("period_topics", {})
        if not period_topics:
            tk.Label(parent, text="No data.", font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        control = tk.Frame(parent, bg=self.theme["bg_card"])
        control.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(control, text="Period:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        
        periods = sorted(period_topics.keys())
        self._period_var = tk.StringVar(value=str(periods[0]) if periods else "")
        combo = ttk.Combobox(control, textvariable=self._period_var,
                            values=[str(p) for p in periods], state="readonly", width=15)
        combo.pack(side=tk.LEFT, padx=4)
        
        self._period_table = DataTable(parent, theme=self.theme_name)
        self._period_table.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._period_topics = period_topics
        
        def update(*args):
            try:
                p = int(self._period_var.get())
                if p in self._period_topics:
                    self._period_table.set_data(self._period_topics[p])
            except ValueError:
                pass
        
        combo.bind("<<ComboboxSelected>>", update)
        update()
    
    def _export_results(self):
        """Export results."""
        if self._result is None:
            messagebox.showwarning("No Results", "Run analysis first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")],
            title="Export Results"
        )
        if not filename:
            return
        
        try:
            result = self._result
            is_dtm = self._method == "DTM"
            
            with pd.ExcelWriter(filename, engine="xlsxwriter") as w:
                if is_dtm:
                    result["topic_prevalence_evolution"].to_excel(w, sheet_name="Prevalence", index=False)
                    result["time_slice_info"].to_excel(w, sheet_name="Time Slices", index=False)
                    result["global_topics"].to_excel(w, sheet_name="Global Topics", index=False)
                    for t, df in result["topic_word_evolution"].items():
                        df.to_excel(w, sheet_name=t[:31], index=False)
                else:
                    result["topic_evolution"].to_excel(w, sheet_name="Evolution", index=False)
                    result["topic_prevalence"].to_excel(w, sheet_name="Prevalence", index=False)
                    result["period_stats"].to_excel(w, sheet_name="Period Stats", index=False)
                    for p, df in result["period_topics"].items():
                        df.to_excel(w, sheet_name=f"Period_{p}"[:31], index=False)
                
                # Add term evolution if computed
                if hasattr(self, '_term_evolution_df') and self._term_evolution_df is not None:
                    self._term_evolution_df.to_excel(w, sheet_name="Term Evolution", index=False)
            
            messagebox.showinfo("Success", f"Exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def _create_term_evolution_tab(self, parent):
        """Create term evolution analysis tab."""
        # Controls
        control_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        control_frame.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            control_frame, text="Top N Terms:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=(0, 4))
        
        self._term_top_n = tk.IntVar(value=15)
        top_n_spin = tk.Spinbox(
            control_frame, from_=5, to=30, textvariable=self._term_top_n, width=5
        )
        top_n_spin.pack(side=tk.LEFT, padx=4)
        
        tk.Label(
            control_frame, text="Plot Type:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=(16, 4))
        
        self._term_plot_type = tk.StringVar(value="heatmap")
        plot_combo = ttk.Combobox(
            control_frame, textvariable=self._term_plot_type,
            values=["heatmap", "line", "area", "bump"], state="readonly", width=10
        )
        plot_combo.pack(side=tk.LEFT, padx=4)
        
        # Time span controls
        control_frame2 = tk.Frame(parent, bg=self.theme["bg_card"])
        control_frame2.pack(fill=tk.X, padx=8, pady=4)
        
        tk.Label(
            control_frame2, text="Time Span:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=(0, 4))
        
        tk.Label(
            control_frame2, text="From:",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        ).pack(side=tk.LEFT, padx=(8, 2))
        
        self._term_year_from = tk.StringVar(value="")
        year_from_entry = tk.Entry(control_frame2, textvariable=self._term_year_from, width=6)
        year_from_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Label(
            control_frame2, text="To:",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        ).pack(side=tk.LEFT, padx=(8, 2))
        
        self._term_year_to = tk.StringVar(value="")
        year_to_entry = tk.Entry(control_frame2, textvariable=self._term_year_to, width=6)
        year_to_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Label(
            control_frame2, text="(leave empty for all years)",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        ).pack(side=tk.LEFT, padx=(8, 4))
        
        tk.Button(
            control_frame2, text="Compute & Plot",
            command=self._compute_term_evolution,
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            font=FONTS.get_font("body"), relief=tk.FLAT, cursor="hand2"
        ).pack(side=tk.LEFT, padx=16)
        
        # Plot frame
        self._term_plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        self._term_plot_frame.pack(fill=tk.BOTH, expand=True, pady=8)
        
        # Initial message
        fig, ax = self._term_plot_frame.get_figure()
        ax.text(0.5, 0.5, "Click 'Compute & Plot' to analyze term evolution",
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        self._term_plot_frame.refresh()
    
    def _compute_term_evolution(self):
        """Compute and plot term evolution."""
        if not self.bib:
            messagebox.showwarning("No Data", "No data loaded.")
            return
        
        text_column = self.text_combo.get()
        time_column = self.time_combo.get()
        
        # Find valid columns
        if text_column not in self.bib.df.columns:
            for fb in ["Abstract", "Processed Title", "Title"]:
                if fb in self.bib.df.columns:
                    text_column = fb
                    break
        
        if time_column not in self.bib.df.columns:
            for fb in ["Year", "Publication Year", "PY"]:
                if fb in self.bib.df.columns:
                    time_column = fb
                    break
        
        try:
            from biblium import utilsbib
            
            top_n = self._term_top_n.get()
            
            # Get time span filter
            year_from_str = self._term_year_from.get().strip()
            year_to_str = self._term_year_to.get().strip()
            
            # Filter dataframe by time span if specified
            df_filtered_time = self.bib.df.copy()
            df_filtered_time[time_column] = pd.to_numeric(df_filtered_time[time_column], errors='coerce')
            
            if year_from_str:
                try:
                    year_from = int(year_from_str)
                    df_filtered_time = df_filtered_time[df_filtered_time[time_column] >= year_from]
                except ValueError:
                    pass
            
            if year_to_str:
                try:
                    year_to = int(year_to_str)
                    df_filtered_time = df_filtered_time[df_filtered_time[time_column] <= year_to]
                except ValueError:
                    pass
            
            if len(df_filtered_time) == 0:
                messagebox.showwarning("No Data", "No documents in the selected time span.")
                return
            
            # Compute term evolution
            term_evo_df = utilsbib.compute_term_evolution(
                df=df_filtered_time,
                text_column=text_column,
                time_column=time_column,
                top_n_terms=top_n,
                normalize=True,
            )
            
            self._term_evolution_df = term_evo_df
            
            if term_evo_df.empty:
                messagebox.showwarning("No Data", "No term evolution data computed.")
                return
            
            # Plot
            plot_type = self._term_plot_type.get()
            fig, ax = self._term_plot_frame.get_figure()
            ax.clear()
            fig.clear()
            
            # Get top terms
            term_avg = term_evo_df.groupby('Term')['Frequency'].mean()
            top_terms = term_avg.nlargest(top_n).index.tolist()
            df_filtered = term_evo_df[term_evo_df['Term'].isin(top_terms)]
            
            # Pivot for plotting
            pivot = df_filtered.pivot(index='Term', columns='Period', values='Frequency').fillna(0)
            pivot = pivot.reindex(top_terms)
            
            if plot_type == "heatmap":
                ax = fig.add_subplot(111)
                import seaborn as sns
                hm = sns.heatmap(
                    pivot, cmap='viridis',
                    annot=len(pivot.columns) <= 10,
                    fmt='.4f', ax=ax,
                    cbar_kws={'label': 'Relative Frequency', 'orientation': 'horizontal', 'pad': 0.15}
                )
                ax.set_xlabel("Year")
                ax.set_ylabel("Term")
                ax.set_title("Term Frequency Evolution (Relative Frequency per Year)")
                
            elif plot_type == "line":
                ax = fig.add_subplot(111)
                colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(top_terms)))
                
                for i, term in enumerate(top_terms):
                    if term in pivot.index:
                        ax.plot(pivot.columns, pivot.loc[term], label=term,
                               color=colors[i], linewidth=2, marker='o', markersize=4)
                
                ax.set_xlabel("Year")
                ax.set_ylabel("Relative Frequency")
                ax.set_title("Term Trends Over Time (Relative Frequency per Year)")
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
                ax.grid(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
            elif plot_type == "area":
                ax = fig.add_subplot(111)
                colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(top_terms)))
                
                ax.stackplot(pivot.columns, pivot.values,
                            labels=pivot.index.tolist(), colors=colors, alpha=0.8)
                ax.set_xlabel("Year")
                ax.set_ylabel("Cumulative Relative Frequency")
                ax.set_title("Stacked Term Frequencies (Relative Frequency per Year)")
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
                ax.grid(False)
                
            elif plot_type == "bump":
                ax = fig.add_subplot(111)
                
                # Compute ranks
                df_filtered_copy = df_filtered.copy()
                df_filtered_copy['Rank'] = df_filtered_copy.groupby('Period')['Frequency'].rank(
                    ascending=False, method='min'
                )
                pivot_rank = df_filtered_copy.pivot(index='Period', columns='Term', values='Rank')
                
                periods = pivot_rank.index.tolist()
                colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(top_terms)))
                
                for i, term in enumerate(top_terms):
                    if term in pivot_rank.columns:
                        ranks = pivot_rank[term].values
                        ax.plot(periods, ranks, label=term, color=colors[i],
                               linewidth=3, marker='o', markersize=8)
                
                ax.invert_yaxis()
                ax.set_xlabel("Year")
                ax.set_ylabel("Rank")
                ax.set_title("Term Rank Evolution (Bump Chart)")
                ax.set_yticks(range(1, min(top_n + 1, 16)))
                ax.grid(False)
            
            fig.tight_layout()
            self._term_plot_frame.refresh()
            
        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Failed to compute: {e}\n\n{traceback.format_exc()}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
DYNAMIC TOPIC MODELING
══════════════════════

Track how topics evolve over time periods.

AVAILABLE METHODS
─────────────────
• DTM (Dynamic Topic Model)
  - Topics evolve through time slices
  - Word distributions change over time
  - Captures vocabulary evolution
  - Best for tracking term changes
  
• STM (Structural Topic Model)
  - Incorporates document metadata
  - Topic prevalence varies with covariates
  - Better for hypothesis testing
  - Includes topic correlations

DTM APPROACH
────────────
• Divides corpus by time period
• Topics linked across periods
• Words can gain/lose relevance
• Shows topic evolution trajectory

STM APPROACH
────────────
• Metadata influences topic proportions
• Prevalence regression on time
• Content can vary with covariates
• More flexible modeling

PARAMETERS
──────────
• Number of topics
• Time period granularity
• Text preprocessing options
• Minimum documents per period

OUTPUT
──────
• Topic prevalence over time
• Topic word evolution
• Emerging/declining topics
• Topic correlation network
• Period-specific top words

USE CASES
─────────
• Track research front emergence
• Identify methodology shifts
• Understand field evolution
• Compare topic trajectories
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
        """Set bibliometric data."""
        self.bib = bib
        self._result = None
        
        if bib and hasattr(bib, 'df'):
            available = [c for c in self.TEXT_COLUMNS if c in bib.df.columns]
            if available:
                self.text_combo.combobox['values'] = available
                self.text_combo.set(available[0])
            
            time_cols = [c for c in ["Year", "Publication Year", "PY"] if c in bib.df.columns]
            if time_cols:
                self.time_combo.combobox['values'] = time_cols
                self.time_combo.set(time_cols[0])
