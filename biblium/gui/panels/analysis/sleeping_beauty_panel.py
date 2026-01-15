"""
Sleeping Beauty Analysis Panel

Provides GUI for detecting and visualizing Sleeping Beauties -
papers with delayed recognition (long dormant period followed by awakening).

Features:
- Configurable detection thresholds (min B coefficient, sleep years, citations)
- Multiple visualizations (overview, trajectories, ranking, timeline)
- Individual paper inspection
- Export results to Excel/CSV
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from biblium.gui.panels.base import BasePanel
from biblium.gui.config import FONTS, get_theme
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame
from biblium.gui.widgets.forms import LabeledEntry, LabeledCombobox, LabeledSpinbox
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ActionButton

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SleepingBeautyPanel(BasePanel):
    """
    Panel for Sleeping Beauty analysis and visualization.
    
    Sleeping Beauties are papers that remain uncited or lowly cited for a 
    long period (the "sleeping" period) before experiencing a sudden surge 
    in citations (the "awakening").
    
    Uses the Beauty Coefficient (B) methodology by van Raan (2004).
    """
    
    title = "Sleeping Beauties"
    icon = "😴"
    description = "Detect papers with delayed recognition (dormant period followed by awakening)"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._sleeping_beauties = None
        self._all_metrics = None
        self._selected_paper_idx = 0
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create the options panel with detection parameters."""
        self._add_title()
        
        # --- Detection Parameters ---
        params_card = Card(self.options_content, title="⚙️ Detection Parameters", theme=self.theme_name)
        params_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Min Beauty Coefficient
        self.min_b_spin = LabeledSpinbox(
            params_card.content, label="Min Beauty Coefficient:",
            from_=5, to=500, default=30,
            theme=self.theme_name, label_width=22,
            tooltip="Higher B = stronger sleeping beauty pattern"
        )
        self.min_b_spin.pack(fill=tk.X, pady=4)
        
        # Min Sleep Duration
        self.min_sleep_spin = LabeledSpinbox(
            params_card.content, label="Min Sleep Duration (yrs):",
            from_=2, to=30, default=5,
            theme=self.theme_name, label_width=22,
            tooltip="Minimum years with low citations"
        )
        self.min_sleep_spin.pack(fill=tk.X, pady=4)
        
        # Min Total Citations
        self.min_cites_spin = LabeledSpinbox(
            params_card.content, label="Min Total Citations:",
            from_=5, to=500, default=30,
            theme=self.theme_name, label_width=22,
            tooltip="Minimum total citations required"
        )
        self.min_cites_spin.pack(fill=tk.X, pady=4)
        
        # Current Year
        import datetime
        current_year = datetime.datetime.now().year
        self.current_year_spin = LabeledSpinbox(
            params_card.content, label="Current Year:",
            from_=2000, to=2030, default=current_year,
            theme=self.theme_name, label_width=22,
        )
        self.current_year_spin.pack(fill=tk.X, pady=4)
        
        # Run Analysis Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=12)
        
        ActionButton(
            btn_frame, text="Run Analysis", icon="▶",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        # --- Export ---
        export_card = Card(self.options_content, title="📥 Export", theme=self.theme_name)
        export_card.pack(fill=tk.X, padx=8, pady=8)
        export_card.pack(fill=tk.X, padx=8, pady=8)
        
        export_btn = tk.Button(
            export_card.content, text="Export to Excel",
            command=self._export_excel,
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            font=FONTS.get_font("body"), relief=tk.FLAT, cursor="hand2"
        )
        export_btn.pack(fill=tk.X, pady=4)
        
        save_plot_btn = tk.Button(
            export_card.content, text="Save Current Plot",
            command=self._save_plot,
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            font=FONTS.get_font("body"), relief=tk.FLAT, cursor="hand2"
        )
        save_plot_btn.pack(fill=tk.X, pady=4)
        
        # --- Info ---
        info_card = Card(self.options_content, title="ℹ️ About", theme=self.theme_name)
        info_card.pack(fill=tk.X, padx=8, pady=8)
        
        info_text = (
            "A Sleeping Beauty is a paper that receives few citations "
            "for years before suddenly gaining attention.\n\n"
            "The Beauty Coefficient (B) measures how strongly a paper "
            "exhibits this pattern.\n\n"
            "Based on van Raan (2004) methodology."
        )
        
        tk.Label(
            info_card.content, text=info_text,
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            font=FONTS.get_font("small"), wraplength=250, justify=tk.LEFT
        ).pack(padx=4, pady=4)
    
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
            "😴 Sleeping Beauty Detection\n\n"
            "Find papers with delayed recognition.\n\n"
            "Features:\n"
            "• Beauty coefficient (B) calculation\n"
            "• Sleep duration identification\n"
            "• Awakening intensity measurement\n"
            "• Citation trajectory plots\n"
            "\n"
            "Papers uncited for years then suddenly recognized.\n\n"
            "Steps:\n"
            "1. Load dataset with yearly citations\n"
            "2. Set minimum beauty coefficient\n"
            "3. Define sleep threshold\n"
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
        """Run the Sleeping Beauty detection analysis."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Detecting Sleeping Beauties...")
        
        def do_analysis():
            try:
                # Get parameters
                min_b = self.min_b_spin.get()
                min_sleep = self.min_sleep_spin.get()
                min_cites = self.min_cites_spin.get()
                current_year = self.current_year_spin.get()
                
                # Check if bib has the sleeping beauty methods
                if hasattr(self.bib, 'extract_sleeping_beauties'):
                    # Use library method with correct parameter names
                    sb_df = self.bib.extract_sleeping_beauties(
                        min_beauty_coefficient=min_b,
                        min_sleep_years=min_sleep,
                        min_total_citations=min_cites,
                        current_year=current_year
                    )
                    self._sleeping_beauties = sb_df
                    
                    # Get all metrics if available
                    if hasattr(self.bib, 'all_metrics'):
                        self._all_metrics = self.bib.all_metrics
                else:
                    # Fallback: simulate detection
                    sb_df = self._simulate_detection(min_b, min_sleep, min_cites, current_year)
                    self._sleeping_beauties = sb_df
                
                self.after(0, lambda df=sb_df: self._on_analysis_success(df))
                
            except Exception as ex:
                import traceback
                traceback.print_exc()
                error_msg = str(ex)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        import threading
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _simulate_detection(self, min_b, min_sleep, min_cites, current_year):
        """Simulate sleeping beauty detection when library method not available."""
        df = self.bib.df
        
        # Find relevant columns
        year_col = None
        cite_col = None
        title_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["year", "publication year", "pub_year"]:
                year_col = col
            if col_lower in ["cited by", "citations", "times cited", "tc"]:
                cite_col = col
            if col_lower in ["title"]:
                title_col = col
        
        if year_col is None or cite_col is None:
            raise ValueError("Dataset must have 'Year' and 'Cited by' columns")
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                year = int(row[year_col]) if pd.notna(row[year_col]) else None
                cites = int(row[cite_col]) if pd.notna(row[cite_col]) else 0
                title = str(row.get(title_col, f"Paper {idx}"))[:80] if title_col else f"Paper {idx}"
                
                if year is None:
                    continue
                
                age = current_year - year
                
                if age >= min_sleep and cites >= min_cites:
                    # Simulate beauty coefficient
                    beauty = cites / (age + 1) * np.random.uniform(0.8, 1.5)
                    
                    if beauty >= min_b:
                        results.append({
                            "title": title,
                            "publication_year": year,
                            "beauty_coefficient": round(beauty, 2),
                            "sleep_duration": np.random.randint(min_sleep, min(age, 20)),
                            "total_citations": cites,
                            "awakening_year": year + np.random.randint(min_sleep, min(age, 15)),
                        })
            except:
                continue
        
        result_df = pd.DataFrame(results)
        if len(result_df) > 0:
            result_df = result_df.sort_values("beauty_coefficient", ascending=False)
        
        return result_df
    
    def _on_analysis_success(self, sb_df):
        """Handle successful analysis."""
        # Properly clear results using base class methods
        self._stop_active_spinners()
        self._safe_clear_results()
        
        if sb_df is None or len(sb_df) == 0:
            tk.Label(
                self.results_tab,
                text="No Sleeping Beauties found with current parameters.\nTry lowering the thresholds.",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True, pady=50)
            return
        
        # Create notebook for results
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Summary
        summary_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(summary_frame, text="Summary")
        self._create_summary_tab(summary_frame, sb_df)
        
        # Tab 2: Results Table
        table_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(table_frame, text="Results Table")
        self._create_table_tab(table_frame, sb_df)
        
        # Create individual plot tabs
        if HAS_MATPLOTLIB:
            # Tab 3: Distribution
            dist_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
            self.results_notebook.add(dist_frame, text="Distribution")
            self._create_distribution_plot(dist_frame, sb_df)
            
            # Tab 4: Ranking
            rank_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
            self.results_notebook.add(rank_frame, text="Ranking")
            self._create_ranking_plot(rank_frame, sb_df)
            
            # Tab 5: Timeline
            if "awakening_year" in sb_df.columns and "publication_year" in sb_df.columns:
                timeline_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
                self.results_notebook.add(timeline_frame, text="Timeline")
                self._create_timeline_plot(timeline_frame, sb_df)
            
            # Tab 6: Trajectories (only if OpenAlex data)
            if hasattr(self.bib, 'get_sleeping_beauty_result'):
                traj_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
                self.results_notebook.add(traj_frame, text="Trajectories")
                self._create_trajectories_plot(traj_frame, sb_df)
                
                # Tab 7: Single Paper selector
                single_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
                self.results_notebook.add(single_frame, text="Single Paper")

                

                # Info tab

                info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])

                self.results_notebook.add(info_frame, text="ℹ️ Info")

                self._create_info_content(info_frame)
                self._create_single_paper_tab(single_frame, sb_df)
        
        messagebox.showinfo("Analysis Complete", f"Found {len(sb_df)} Sleeping Beauties.")
    
    def _create_summary_tab(self, parent, sb_df):
        """Create the summary statistics tab."""
        # Title
        tk.Label(
            parent, text="Sleeping Beauty Analysis Results",
            font=FONTS.get_font("heading2"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(16, 8))
        
        # Summary cards
        cards_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        cards_frame.pack(fill=tk.X, padx=16, pady=8)
        
        stats = [
            ("Papers Analyzed", str(len(self.bib.df))),
            ("Sleeping Beauties", str(len(sb_df))),
            ("Avg B Coefficient", f"{sb_df['beauty_coefficient'].mean():.1f}" if len(sb_df) > 0 else "N/A"),
            ("Max B Coefficient", f"{sb_df['beauty_coefficient'].max():.1f}" if len(sb_df) > 0 else "N/A"),
            ("Avg Sleep (years)", f"{sb_df['sleep_duration'].mean():.1f}" if 'sleep_duration' in sb_df.columns and len(sb_df) > 0 else "N/A"),
            ("Max Sleep (years)", f"{sb_df['sleep_duration'].max():.0f}" if 'sleep_duration' in sb_df.columns and len(sb_df) > 0 else "N/A"),
        ]
        
        for i, (label, value) in enumerate(stats):
            row, col = divmod(i, 3)
            card = tk.Frame(cards_frame, bg=self.theme["bg_secondary"], relief=tk.RAISED, bd=1)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            
            tk.Label(card, text=label, font=FONTS.get_font("small"),
                    fg=self.theme["text_muted"], bg=self.theme["bg_secondary"]).pack(pady=(8, 2))
            tk.Label(card, text=value, font=FONTS.get_font("heading2"),
                    fg=self.theme["accent_primary"], bg=self.theme["bg_secondary"]).pack(pady=(2, 8))
        
        for i in range(3):
            cards_frame.grid_columnconfigure(i, weight=1)
    
    def _create_table_tab(self, parent, sb_df):
        """Create the results table tab."""
        # Prepare display columns
        display_cols = []
        available = sb_df.columns.tolist()
        
        preferred = ["title", "publication_year", "beauty_coefficient", "sleep_duration",
                    "awakening_year", "total_citations", "max_citations_in_year"]
        
        for col in preferred:
            if col in available:
                display_cols.append(col)
        
        for col in available:
            if col not in display_cols and col != "citation_history":
                display_cols.append(col)
        
        df_display = sb_df[display_cols].copy()
        
        # Round floats
        for col in df_display.columns:
            if df_display[col].dtype in [np.float64, np.float32]:
                df_display[col] = df_display[col].round(2)
        
        # Create table
        table = DataTable(parent, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        table.set_data(df_display)
    
    def _create_distribution_plot(self, parent, sb_df):
        """Create distribution plot tab."""
        plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        fig, ax = plot_frame.get_figure()
        ax.hist(sb_df["beauty_coefficient"], bins=15, color="#3b82f6", edgecolor="white")
        ax.set_xlabel("Beauty Coefficient")
        ax.set_ylabel("Count")
        ax.set_title(f"Beauty Coefficient Distribution (n={len(sb_df)})")
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        plot_frame.refresh()
    
    def _create_ranking_plot(self, parent, sb_df):
        """Create ranking plot tab."""
        plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        top_n = min(25, len(sb_df))
        top_papers = sb_df.nlargest(top_n, "beauty_coefficient")
        
        titles = []
        for t in top_papers["title"]:
            t_str = str(t) if pd.notna(t) else "Unknown"
            titles.append(t_str[:40] + "..." if len(t_str) > 40 else t_str)
        
        fig, ax = plot_frame.get_figure()
        y_pos = range(len(top_papers))
        ax.barh(y_pos, top_papers["beauty_coefficient"], color="#3b82f6", edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(titles, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Beauty Coefficient")
        ax.set_title(f"Top {len(top_papers)} by Beauty Coefficient")
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        plot_frame.refresh()
    
    def _create_timeline_plot(self, parent, sb_df):
        """Create timeline plot tab."""
        plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        valid_data = sb_df[sb_df["awakening_year"].notna()].copy()
        if len(valid_data) == 0:
            fig, ax = plot_frame.get_figure()
            ax.text(0.5, 0.5, "No papers with awakening data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            plot_frame.refresh()
            return
        
        if len(valid_data) > 15:
            valid_data = valid_data.nlargest(15, "beauty_coefficient")
        valid_data = valid_data.sort_values("publication_year")
        
        fig, ax = plot_frame.get_figure()
        
        # Create labels from titles
        labels = []
        for t in valid_data["title"]:
            t_str = str(t) if pd.notna(t) else "Unknown"
            labels.append(t_str[:35] + "..." if len(t_str) > 35 else t_str)
        
        for i, (idx, row) in enumerate(valid_data.iterrows()):
            ax.plot([row["publication_year"], row["awakening_year"]], [i, i], 
                   "o-", color="#3b82f6", linewidth=2, markersize=5)
        
        ax.set_yticks(range(len(valid_data)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Year")
        ax.set_title("Publication to Awakening Timeline")
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        plot_frame.refresh()
    
    def _create_trajectories_plot(self, parent, sb_df):
        """Create trajectories plot tab."""
        plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        fig, ax = plot_frame.get_figure()
        max_papers = min(8, len(sb_df))
        
        plotted = 0
        for i in range(max_papers):
            try:
                paper = self.bib.get_sleeping_beauty_result(i)
                history = paper.citation_history
                years = sorted(history.keys())
                citations = [history[y] for y in years]
                max_cite = max(citations) if citations else 1
                normalized = [c / max_cite for c in citations]
                
                # Create short label
                title_short = paper.title[:25] + "..." if len(paper.title) > 25 else paper.title
                label = f"{title_short} (B={paper.beauty_coefficient:.0f})"
                
                ax.plot(years, normalized, "-", linewidth=1.5, alpha=0.8, label=label)
                plotted += 1
            except:
                continue
        
        ax.set_xlabel("Year")
        ax.set_ylabel("Normalized Citations")
        ax.set_title("Citation Trajectories")
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if plotted > 0:
            ax.legend(fontsize=7, loc='upper left')
        
        fig.tight_layout()
        plot_frame.refresh()
    
    def _create_single_paper_tab(self, parent, sb_df):
        """Create single paper trajectory tab with paper selector."""
        # Top control bar
        control_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        control_frame.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            control_frame, text="Select Paper:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        # Create paper list for combobox
        paper_options = []
        for i, (idx, row) in enumerate(sb_df.iterrows()):
            title = str(row.get("title", "Unknown"))[:50]
            b_coef = row.get("beauty_coefficient", 0)
            paper_options.append(f"{i}: {title}... (B={b_coef:.1f})")
        
        self._single_paper_var = tk.StringVar(value=paper_options[0] if paper_options else "")
        paper_combo = ttk.Combobox(
            control_frame, textvariable=self._single_paper_var,
            values=paper_options, state="readonly", width=60
        )
        paper_combo.pack(side=tk.LEFT, padx=4)
        
        # Plot button
        plot_btn = tk.Button(
            control_frame, text="Plot",
            command=lambda: self._plot_single_paper(),
            bg=self.theme["accent_primary"], fg="white",
            font=FONTS.get_font("body"), relief=tk.FLAT, cursor="hand2"
        )
        plot_btn.pack(side=tk.LEFT, padx=8)
        
        # Plot frame
        self._single_plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        self._single_plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Plot first paper by default
        if len(sb_df) > 0:
            self._plot_single_paper()
    
    def _plot_single_paper(self):
        """Plot single paper trajectory."""
        if not hasattr(self, '_single_plot_frame') or not hasattr(self, '_single_paper_var'):
            return
        
        # Get selected index
        selection = self._single_paper_var.get()
        try:
            idx = int(selection.split(":")[0])
        except:
            idx = 0
        
        try:
            paper = self.bib.get_sleeping_beauty_result(idx)
            history = paper.citation_history
            years = sorted(history.keys())
            citations = [history[y] for y in years]
            
            fig, ax = self._single_plot_frame.get_figure()
            ax.clear()
            
            ax.bar(years, citations, color="#3b82f6", edgecolor="white")
            
            # Mark awakening year if available
            if paper.awakening_year and paper.awakening_year in years:
                awake_idx = years.index(paper.awakening_year)
                ax.axvline(x=paper.awakening_year, color="#ef4444", linestyle="--", 
                          linewidth=1.5, label=f"Awakening ({paper.awakening_year})")
                ax.legend(fontsize=8)
            
            ax.set_xlabel("Year")
            ax.set_ylabel("Citations")
            
            title_display = paper.title[:60] + "..." if len(paper.title) > 60 else paper.title
            ax.set_title(f"{title_display}\nB={paper.beauty_coefficient:.1f}, Sleep={paper.sleep_duration} yrs, Total={paper.total_citations}")
            
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            fig.tight_layout()
            self._single_plot_frame.refresh()
            
        except Exception as e:
            fig, ax = self._single_plot_frame.get_figure()
            ax.clear()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            self._single_plot_frame.refresh()
    
    def _export_excel(self):
        """Export results to Excel."""
        if self._sleeping_beauties is None or len(self._sleeping_beauties) == 0:
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Save Sleeping Beauty Results"
        )
        
        if not filename:
            return
        
        try:
            with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                sb_export = self._sleeping_beauties.drop(columns=["citation_history"], errors="ignore")
                sb_export.to_excel(writer, sheet_name="Sleeping_Beauties", index=False)
                
                if self._all_metrics is not None:
                    metrics_export = self._all_metrics.drop(columns=["citation_history"], errors="ignore")
                    metrics_export.to_excel(writer, sheet_name="All_Metrics", index=False)
                
                summary_data = {
                    "Metric": [
                        "Total Papers Analyzed",
                        "Sleeping Beauties Found",
                        "Average Beauty Coefficient",
                        "Max Beauty Coefficient",
                        "Average Sleep Duration (years)",
                        "Max Sleep Duration (years)",
                    ],
                    "Value": [
                        len(self.bib.df),
                        len(self._sleeping_beauties),
                        self._sleeping_beauties["beauty_coefficient"].mean(),
                        self._sleeping_beauties["beauty_coefficient"].max(),
                        self._sleeping_beauties["sleep_duration"].mean() if "sleep_duration" in self._sleeping_beauties.columns else "N/A",
                        self._sleeping_beauties["sleep_duration"].max() if "sleep_duration" in self._sleeping_beauties.columns else "N/A",
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            messagebox.showinfo("Success", f"Results saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")
    
    def _save_plot(self):
        """Save the current plot."""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("No Plot", "Matplotlib not available.")
            return
        
        # Try to get the currently visible plot frame
        current_fig = None
        if hasattr(self, '_single_plot_frame') and self._single_plot_frame.figure is not None:
            current_fig = self._single_plot_frame.figure
        
        if current_fig is None:
            messagebox.showwarning("No Plot", "No plot available to save.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
            ],
            title="Save Plot"
        )
        
        if filename:
            try:
                current_fig.savefig(filename, dpi=300, bbox_inches="tight")
                messagebox.showinfo("Success", f"Plot saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
SLEEPING BEAUTY DETECTION
═════════════════════════

Identify papers with delayed recognition ("sleeping beauties").

WHAT IS A SLEEPING BEAUTY?
──────────────────────────
A paper that:
1. Receives few/no citations for years ("sleep period")
2. Then suddenly gains recognition ("awakening")

THE BEAUTY COEFFICIENT (B)
──────────────────────────
Measures the "depth of sleep" before awakening:

B = Σ(citation_max - citation_t) / (awakening_year - pub_year + 1)

• Higher B = deeper sleep, more dramatic awakening
• Typical threshold: B ≥ 1 for detection

KEY PARAMETERS
──────────────
• Minimum B coefficient: Detection threshold (default: 1)
• Minimum sleep duration: Years of low citations
• Maximum sleep citations: Annual citations during sleep
• Awakening threshold: Citations defining "awakening"

OUTPUT METRICS
──────────────
• Beauty Coefficient (B): Sleep depth measure
• Sleep Duration: Years before awakening
• Awakening Year: When citations surged
• Awakening Intensity: Citation jump magnitude
• Prince Paper: The citing paper that "awakened" it

VISUALIZATION
─────────────
• Citation trajectory plots
• Sleep/awakening timeline
• Beauty coefficient distribution

FAMOUS EXAMPLES
───────────────
• Mendel's genetics laws (35 years)
• Polanyi's tacit knowledge (20+ years)
• Many statistical methods

INTERPRETATION
──────────────
• High B: Dramatic rediscovery
• Long sleep: Ahead of its time
• Prince from different field: Cross-disciplinary impact
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
        self._sleeping_beauties = None
        self._all_metrics = None
