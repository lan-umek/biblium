# -*- coding: utf-8 -*-
"""
Life Cycle Panel
================
Scientific production life cycle analysis using logistic growth model.
Implements BiblioShiny-style life cycle analysis.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledSpinbox, LabeledCheckbox
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import add_plot_context_menu, make_canvas_resizable
from biblium.gui.widgets.plots import PlotFrame

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class LifeCyclePanel(BasePanel):
    """Panel for scientific production life cycle analysis."""
    
    title = "Life Cycle Analysis"
    icon = "🔄"
    description = "Analyze the life cycle of scientific production using logistic growth model"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._result = None
        self._canvas = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Model Settings Card
        settings_card = Card(self.options_content, title="⚙️ Model Settings", theme=self.theme_name)
        settings_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Forecast years
        self.forecast_years_spin = LabeledSpinbox(
            settings_card.content,
            label="Forecast Years:",
            from_=10, to=100, default=50,
            theme=self.theme_name, label_width=14,
        )
        self.forecast_years_spin.pack(fill=tk.X, pady=4)
        
        # Target years for projections
        target_frame = tk.Frame(settings_card.content, bg=self.theme["bg_card"])
        target_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            target_frame, text="Projection Years:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.target_years_entry = tk.Entry(
            target_frame, font=FONTS.get_font("body"), width=20,
        )
        self.target_years_entry.pack(side=tk.LEFT, padx=(8, 0))
        self.target_years_entry.insert(0, "2025, 2030, 2035")
        
        # Display Options Card
        display_card = Card(self.options_content, title="📊 Display Options", theme=self.theme_name)
        display_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_milestones_cb = LabeledCheckbox(
            display_card.content,
            label="Show Milestone Years",
            default=True,
            theme=self.theme_name,
        )
        self.show_milestones_cb.pack(fill=tk.X, pady=2)
        
        self.show_forecast_cb = LabeledCheckbox(
            display_card.content,
            label="Show Forecast Period",
            default=True,
            theme=self.theme_name,
        )
        self.show_forecast_cb.pack(fill=tk.X, pady=2)
        
        # Plot years limit
        self.plot_years_spin = LabeledSpinbox(
            display_card.content,
            label="Plot Forecast Limit:",
            from_=10, to=100, default=30,
            theme=self.theme_name, label_width=16,
        )
        self.plot_years_spin.pack(fill=tk.X, pady=4)
        
        # Run button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_primary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        self.run_btn = ActionButton(
            btn_frame, text="Run Analysis",
            command=self._run_analysis,
            icon="▶",
            theme=self.theme_name,
        )
        self.run_btn.pack(fill=tk.X)
        
        # Export button
        self.export_btn = ThemedButton(
            btn_frame, text="Export Results",
            command=self._export_results,
            style="secondary",
            icon="📥",
            theme=self.theme_name,
        )
        self.export_btn.pack(fill=tk.X, pady=(8, 0))
    
    def _create_results(self):
        """Create the results panel with tabs."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Summary tab
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📋 Summary")
        
        # Plot tab
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="📈 Plot")
        
        # Forecast tab
        self.forecast_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.forecast_frame, text="🔮 Forecast")
        
        # Info tab
        self.info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.info_frame, text="ℹ️ Info & References")
        
        # Info tab
        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        
        # Initialize with placeholder
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Show placeholder message."""
        for frame in [self.summary_frame, self.plot_frame, self.forecast_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        tk.Label(
            self.summary_frame,
            text="Click 'Run Analysis' to analyze the life cycle\nof scientific production",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.CENTER,
        ).pack(expand=True)
        
        # Add info content
        self._create_info_tab()
    
    def _create_info_tab(self):
        """Create the info/references tab."""
        for widget in self.info_frame.winfo_children():
            widget.destroy()
        
        # Scrollable frame
        canvas = tk.Canvas(self.info_frame, bg=self.theme["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.info_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.theme["bg_card"])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Info content
        info_text = """
LIFE CYCLE ANALYSIS
═══════════════════

This analysis models the growth of scientific literature on a topic using 
a logistic (S-curve) growth model. The model estimates when the field will 
reach saturation and identifies the current phase of development.

MODEL FORMULA
─────────────
N(t) = K / (1 + exp(-r × (t - Tm)))

Where:
• K  = Carrying capacity (saturation level)
• r  = Growth rate parameter  
• Tm = Midpoint (inflection point, when growth is fastest)

KEY METRICS
───────────
• Saturation (K): Maximum cumulative publications the topic will reach
• Peak Year (Tm): Year of maximum annual growth (inflection point)
• Peak Annual: Maximum publications per year (at inflection)
• Growth Duration (Δt): Years from 10% to 90% of saturation
  Formula: Δt = 2 × ln(9) / r ≈ 4.39 / r

GROWTH PHASES
─────────────
• Emergence (<10% of K): Topic is just beginning
• Rapid Growth (10-50% of K): Accelerating publication rate  
• Maturity (50-90% of K): Decelerating growth
• Saturation (>90% of K): Approaching carrying capacity

MODEL FIT QUALITY
─────────────────
• Excellent: R² ≥ 0.95
• Good: R² ≥ 0.85
• Fair: R² ≥ 0.70
• Poor: R² < 0.70

REFERENCES
──────────
• Egghe, L., & Ravichandra Rao, I. K. (1992). Classification of growth 
  models based on growth rates and its applications. Scientometrics.

• Gupta, B. M., & Karisiddappa, C. R. (1999). Modelling the growth of 
  literature in the area of theoretical population genetics. 
  Scientometrics.

• Bornmann, L., & Mutz, R. (2015). Growth rates of modern science: 
  A bibliometric analysis based on the number of publications and 
  cited references. JASIST.
"""
        
        text_widget = tk.Text(
            scrollable_frame,
            font=FONTS.get_font("monospace"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            wrap=tk.WORD,
            padx=16, pady=16,
            relief=tk.FLAT,
            height=30,
        )
        text_widget.insert("1.0", info_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
    
    def _run_analysis(self):
        """Run the life cycle analysis."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self.run_btn.config(state=tk.DISABLED, text="⏳ Analyzing...")
        
        def do_analysis():
            try:
                # Parse target years
                target_years_str = self.target_years_entry.get().strip()
                target_years = None
                if target_years_str:
                    try:
                        target_years = [int(y.strip()) for y in target_years_str.split(",")]
                    except ValueError:
                        pass
                
                forecast_years = self.forecast_years_spin.get()
                
                # Import and run analysis
                from biblium.addons.advanced_statistics import analyze_life_cycle
                
                # Find year column
                year_col = "Year"
                for col in ["Year", "PY", "Publication Year"]:
                    if col in self.bib.df.columns:
                        year_col = col
                        break
                
                result = analyze_life_cycle(
                    self.bib.df,
                    year_col=year_col,
                    forecast_years=forecast_years,
                    target_years=target_years,
                    verbose=False,
                )
                
                self._result = result
                self.after(0, lambda: self._on_analysis_success(result))
                
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_analysis_error(str(e), tb))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _on_analysis_success(self, result):
        """Handle successful analysis."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Run Analysis")
        
        # Update all tabs
        self._update_summary_tab(result)
        self._update_plot_tab(result)
        self._update_forecast_tab(result)
        
        # Switch to summary tab
        self.notebook.select(0)
    
    def _on_analysis_error(self, error_msg, traceback_str):
        """Handle analysis error."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Run Analysis")
        messagebox.showerror("Analysis Error", f"Error: {error_msg}\n\n{traceback_str[:500]}")
    
    def _update_summary_tab(self, result):
        """Update the summary tab with results."""
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        
        # Scrollable frame
        canvas = tk.Canvas(self.summary_frame, bg=self.theme["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.summary_frame, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=self.theme["bg_card"])
        
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Model Overview Card
        overview_card = Card(content, title="📈 Model Overview", theme=self.theme_name)
        overview_card.pack(fill=tk.X, padx=8, pady=8)
        
        overview_grid = tk.Frame(overview_card.content, bg=self.theme["bg_card"])
        overview_grid.pack(fill=tk.X, pady=8)
        
        metrics = [
            ("SATURATION (K)", f"{result.saturation_k:,.0f}", "pubs"),
            ("PEAK YEAR (Tm)", f"{result.peak_year_tm:.1f}", ""),
            ("PEAK ANNUAL", f"{result.peak_annual:,.0f}", "pubs/year"),
            ("GROWTH DURATION (Δt)", f"{result.growth_duration_delta_t:.1f}", "years"),
        ]
        
        for i, (label, value, unit) in enumerate(metrics):
            frame = tk.Frame(overview_grid, bg=self.theme["bg_secondary"], padx=12, pady=8)
            frame.grid(row=0, column=i, padx=4, pady=4, sticky="nsew")
            overview_grid.columnconfigure(i, weight=1)
            
            tk.Label(
                frame, text=label, font=FONTS.get_font("caption"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"],
            ).pack()
            
            tk.Label(
                frame, text=value, font=FONTS.get_font("h2"),
                bg=self.theme["bg_secondary"], fg=self.theme["accent_primary"],
            ).pack()
            
            if unit:
                tk.Label(
                    frame, text=unit, font=FONTS.get_font("caption"),
                    bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"],
                ).pack()
        
        # Model Fit Quality Card
        quality_color = {
            "Excellent": self.theme["success"],
            "Good": self.theme["accent_primary"],
            "Fair": self.theme["warning"],
            "Poor": self.theme["danger"],
        }.get(result.fit_quality, self.theme["text_primary"])
        
        fit_card = Card(content, title="✓ Model Fit Quality", theme=self.theme_name)
        fit_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Quality badge
        badge_frame = tk.Frame(fit_card.content, bg=self.theme["bg_card"])
        badge_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            badge_frame, text=result.fit_quality,
            font=FONTS.get_font("body_bold"),
            bg=quality_color, fg="white",
            padx=8, pady=2,
        ).pack(side=tk.LEFT)
        
        # Fit metrics
        fit_grid = tk.Frame(fit_card.content, bg=self.theme["bg_card"])
        fit_grid.pack(fill=tk.X, pady=8)
        
        fit_metrics = [
            ("R²", f"{result.r_squared:.3f}"),
            ("RMSE", f"{result.rmse:.2f}"),
            ("AIC", f"{result.aic:.1f}"),
            ("BIC", f"{result.bic:.1f}"),
        ]
        
        for i, (label, value) in enumerate(fit_metrics):
            frame = tk.Frame(fit_grid, bg=self.theme["bg_secondary"], padx=12, pady=8)
            frame.grid(row=0, column=i, padx=4, pady=4, sticky="nsew")
            fit_grid.columnconfigure(i, weight=1)
            
            tk.Label(
                frame, text=label, font=FONTS.get_font("caption"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"],
            ).pack()
            
            tk.Label(
                frame, text=value, font=FONTS.get_font("h3"),
                bg=self.theme["bg_secondary"], fg=self.theme["accent_primary"],
            ).pack()
        
        # Current Status, Milestone Years, Forecast in a row
        bottom_row = tk.Frame(content, bg=self.theme["bg_card"])
        bottom_row.pack(fill=tk.X, padx=8, pady=8)
        
        # Current Status Card
        status_card = Card(bottom_row, title="ℹ️ Current Status", theme=self.theme_name)
        status_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        
        status_items = [
            ("Last Observed Year:", str(result.last_observed_year)),
            ("Annual Publications:", str(result.last_annual_pubs)),
            ("Cumulative Total:", f"{result.cumulative_total:,}"),
            ("Progress to Saturation:", f"{result.progress_to_saturation*100:.1f}%"),
        ]
        
        for label, value in status_items:
            row = tk.Frame(status_card.content, bg=self.theme["bg_card"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(
                row, text=label, font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            ).pack(side=tk.LEFT)
            tk.Label(
                row, text=value, font=FONTS.get_font("body_bold"),
                bg=self.theme["bg_card"], fg=self.theme["accent_primary"],
            ).pack(side=tk.RIGHT)
        
        # Progress bar
        progress_frame = tk.Frame(status_card.content, bg=self.theme["bg_secondary"], height=20)
        progress_frame.pack(fill=tk.X, pady=(8, 4))
        progress_frame.pack_propagate(False)
        
        progress_pct = min(result.progress_to_saturation, 1.0)
        progress_bar = tk.Frame(progress_frame, bg=self.theme["accent_primary"])
        progress_bar.place(relx=0, rely=0, relwidth=progress_pct, relheight=1)
        
        tk.Label(
            progress_frame, text=f"{progress_pct*100:.1f}%",
            font=FONTS.get_font("caption"),
            bg=self.theme["accent_primary"] if progress_pct > 0.15 else self.theme["bg_secondary"],
            fg="white" if progress_pct > 0.15 else self.theme["text_primary"],
        ).place(relx=0.02, rely=0.5, anchor="w")
        
        # Milestone Years Card
        milestone_card = Card(bottom_row, title="🎯 Milestone Years", theme=self.theme_name)
        milestone_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        
        for pct, year in result.milestone_years.items():
            years_from_now = year - result.last_observed_year
            row = tk.Frame(milestone_card.content, bg=self.theme["bg_card"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(
                row, text=f"{pct} of K:",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            ).pack(side=tk.LEFT)
            tk.Label(
                row, text=f"{year:.1f}",
                font=FONTS.get_font("body_bold"),
                bg=self.theme["bg_card"], fg=self.theme["accent_primary"],
            ).pack(side=tk.RIGHT, padx=(0, 8))
            tk.Label(
                row, text=f"({years_from_now:+.0f} years)",
                font=FONTS.get_font("caption"),
                bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            ).pack(side=tk.RIGHT)
        
        # Phase indicator
        phase_frame = tk.Frame(milestone_card.content, bg=self.theme["bg_secondary"], padx=8, pady=4)
        phase_frame.pack(fill=tk.X, pady=(8, 0))
        
        tk.Label(
            phase_frame, text="💡 " + result.phase_description,
            font=FONTS.get_font("caption"),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            wraplength=200,
        ).pack()
        
        # Forecast Card
        forecast_card = Card(bottom_row, title="🔮 Forecast", theme=self.theme_name)
        forecast_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        
        if result.projections:
            years = sorted(result.projections.keys())
            forecast_period = f"{years[0]} - {years[-1]}" if len(years) > 1 else years[0]
            
            row = tk.Frame(forecast_card.content, bg=self.theme["bg_card"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(
                row, text="Forecast Period:",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            ).pack(side=tk.LEFT)
            tk.Label(
                row, text=forecast_period,
                font=FONTS.get_font("body_bold"),
                bg=self.theme["bg_card"], fg=self.theme["accent_primary"],
            ).pack(side=tk.RIGHT)
            
            for year, proj in result.projections.items():
                row = tk.Frame(forecast_card.content, bg=self.theme["bg_card"])
                row.pack(fill=tk.X, pady=2)
                tk.Label(
                    row, text=f"Projection for {year}:",
                    font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                ).pack(side=tk.LEFT)
                tk.Label(
                    row, text=f"{proj['cumulative']:,} cumulative",
                    font=FONTS.get_font("body_bold"),
                    bg=self.theme["bg_card"], fg=self.theme["accent_primary"],
                ).pack(side=tk.RIGHT)
                tk.Label(
                    row, text=f"({proj['annual']:,} annual)",
                    font=FONTS.get_font("caption"),
                    bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                ).pack(side=tk.RIGHT, padx=(0, 8))
    
    def _update_plot_tab(self, result):
        """Update the plot tab with life cycle visualization."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        if not HAS_MATPLOTLIB:
            tk.Label(
                self.plot_frame, text="Matplotlib not available",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["danger"],
            ).pack(expand=True)
            return
        
        try:
            # Create PlotFrame with toolbar
            plot_widget = PlotFrame(
                self.plot_frame, 
                theme=self.theme_name, 
                figsize=(11, 8),
                show_toolbar=True
            , show_ai_button=True)
            plot_widget.pack(fill=tk.BOTH, expand=True)
            
            # Get figure and create subplots
            fig = plot_widget.figure
            fig.clear()
            
            # Create 2 subplots (cumulative on top, annual on bottom)
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            
            # Get data
            df = result.forecast_df.copy()
            forecast_limit = self.plot_years_spin.get()
            last_obs = result.last_observed_year
            df = df[df["Year"] <= last_obs + forecast_limit]
            
            observed_df = df[~df["Is_Forecast"]]
            forecast_df_data = df[df["Is_Forecast"]]
            
            # --- Top plot: Cumulative ---
            # Observed data points
            ax1.scatter(
                observed_df["Year"], observed_df["Observed_Cumulative"],
                color="#3b82f6", s=40, label="Observed", zorder=3, edgecolors="white", linewidth=0.5
            )
            
            # Fitted logistic curve
            ax1.plot(
                df["Year"], df["Fitted_Cumulative"],
                color="#ef4444", linewidth=2.5, label="Logistic Model", zorder=2
            )
            
            # Forecast shading
            if self.show_forecast_cb.get() and len(forecast_df_data) > 0:
                ax1.axvspan(last_obs, df["Year"].max(), alpha=0.08, color="gray", label="Forecast")
            
            # Saturation line
            ax1.axhline(
                y=result.saturation_k, color="#10b981", linestyle="--",
                linewidth=2, label=f"Saturation (K={result.saturation_k:,.0f})"
            )
            
            # Milestone markers
            if self.show_milestones_cb.get():
                colors = {"10%": "#f59e0b", "50%": "#8b5cf6", "90%": "#6366f1", "99%": "#64748b"}
                for pct, year in result.milestone_years.items():
                    if not np.isnan(year) and df["Year"].min() <= year <= df["Year"].max():
                        proportion = float(pct.rstrip("%")) / 100
                        y_val = result.saturation_k * proportion
                        ax1.axvline(x=year, color=colors.get(pct, "gray"), linestyle=":", alpha=0.7, linewidth=1.5)
                        ax1.scatter([year], [y_val], color=colors.get(pct, "gray"), s=60, zorder=4, marker="D")
                        ax1.annotate(
                            f" {pct}", xy=(year, y_val), fontsize=9, fontweight="bold",
                            color=colors.get(pct, "gray"), va="center"
                        )
            
            ax1.set_xlabel("Year", fontsize=10)
            ax1.set_ylabel("Cumulative Publications", fontsize=10)
            ax1.set_title(
                f"Life Cycle of Scientific Production\n"
                f"R² = {result.r_squared:.3f} | Phase: {result.current_phase.replace('_', ' ').title()} | "
                f"Progress: {result.progress_to_saturation*100:.1f}%",
                fontsize=11, fontweight="bold"
            )
            ax1.legend(loc="upper left", fontsize=9)
            ax1.set_facecolor("white")
            ax1.grid(False)
            
            # --- Bottom plot: Annual ---
            # Observed annual bars
            bar_width = 0.8
            ax2.bar(
                observed_df["Year"], observed_df["Observed_Annual"],
                width=bar_width, color="#3b82f6", alpha=0.7, label="Observed Annual", edgecolor="white"
            )
            
            # Fitted annual line
            ax2.plot(
                df["Year"], df["Fitted_Annual"],
                color="#ef4444", linewidth=2.5, label="Model Fit"
            )
            
            # Peak annual line
            ax2.axhline(
                y=result.peak_annual, color="#10b981", linestyle="--",
                alpha=0.8, linewidth=1.5, label=f"Peak ({result.peak_annual:,.0f}/year)"
            )
            
            # Mark peak year
            ax2.axvline(x=result.peak_year_tm, color="#8b5cf6", linestyle=":", alpha=0.7, linewidth=1.5)
            ax2.annotate(
                f" Peak Year\n {result.peak_year_tm:.0f}",
                xy=(result.peak_year_tm, result.peak_annual * 0.9),
                fontsize=9, color="#8b5cf6", fontweight="bold"
            )
            
            ax2.set_xlabel("Year", fontsize=10)
            ax2.set_ylabel("Annual Publications", fontsize=10)
            ax2.legend(loc="upper left", fontsize=9)
            ax2.set_facecolor("white")
            ax2.grid(False)
            
            # Adjust layout
            fig.tight_layout(pad=2.0)
            
            # Refresh canvas
            plot_widget.refresh()
            
        except Exception as e:
            import traceback
            error_msg = f"Error creating plot: {e}\n{traceback.format_exc()}"
            tk.Label(
                self.plot_frame, text=error_msg,
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["danger"],
                wraplength=600, justify=tk.LEFT
            ).pack(expand=True, padx=20, pady=20)
    
    def _update_forecast_tab(self, result):
        """Update the forecast data tab."""
        for widget in self.forecast_frame.winfo_children():
            widget.destroy()
        
        if not HAS_PANDAS:
            tk.Label(
                self.forecast_frame, text="Pandas not available",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["danger"],
            ).pack(expand=True)
            return
        
        # Create data table
        df = result.forecast_df.copy()
        
        # Format columns
        df["Year"] = df["Year"].astype(int)
        for col in ["Observed_Annual", "Observed_Cumulative", "Fitted_Annual", "Fitted_Cumulative"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
        
        df["Is_Forecast"] = df["Is_Forecast"].map({True: "Yes", False: "No"})
        
        # Rename columns for display
        df = df.rename(columns={
            "Observed_Annual": "Observed Annual",
            "Observed_Cumulative": "Observed Cumulative",
            "Fitted_Annual": "Fitted Annual",
            "Fitted_Cumulative": "Fitted Cumulative",
            "Is_Forecast": "Forecast",
        })
        
        table = DataTable(
            self.forecast_frame,
            theme=self.theme_name,
            show_index=False,
        )
        table.set_data(df)
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    
    def _export_results(self):
        """Export results to file."""
        if self._result is None:
            messagebox.showwarning("No Results", "Please run analysis first.")
            return
        
        from tkinter import filedialog
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
            ],
            title="Export Life Cycle Results",
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith(".xlsx"):
                with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                    # Summary sheet
                    summary_data = {
                        "Metric": [
                            "Saturation (K)",
                            "Peak Year (Tm)",
                            "Peak Annual",
                            "Growth Duration (Δt)",
                            "Growth Rate (r)",
                            "R²",
                            "RMSE",
                            "AIC",
                            "BIC",
                            "Last Observed Year",
                            "Cumulative Total",
                            "Progress to Saturation",
                            "Current Phase",
                        ],
                        "Value": [
                            f"{self._result.saturation_k:,.0f}",
                            f"{self._result.peak_year_tm:.1f}",
                            f"{self._result.peak_annual:,.0f}",
                            f"{self._result.growth_duration_delta_t:.1f}",
                            f"{self._result.growth_rate_r:.4f}",
                            f"{self._result.r_squared:.4f}",
                            f"{self._result.rmse:.2f}",
                            f"{self._result.aic:.1f}",
                            f"{self._result.bic:.1f}",
                            str(self._result.last_observed_year),
                            f"{self._result.cumulative_total:,}",
                            f"{self._result.progress_to_saturation*100:.1f}%",
                            self._result.current_phase,
                        ],
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
                    
                    # Milestones sheet
                    milestones_data = {
                        "Milestone": list(self._result.milestone_years.keys()),
                        "Year": [f"{y:.1f}" for y in self._result.milestone_years.values()],
                    }
                    pd.DataFrame(milestones_data).to_excel(writer, sheet_name="Milestones", index=False)
                    
                    # Forecast sheet
                    self._result.forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
                    
                    # Projections sheet
                    if self._result.projections:
                        proj_data = {
                            "Year": list(self._result.projections.keys()),
                            "Cumulative": [p["cumulative"] for p in self._result.projections.values()],
                            "Annual": [p["annual"] for p in self._result.projections.values()],
                        }
                        pd.DataFrame(proj_data).to_excel(writer, sheet_name="Projections", index=False)
            else:
                self._result.forecast_df.to_csv(filepath, index=False)
            
            messagebox.showinfo("Export Complete", f"Results exported to:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
RESEARCH LIFE CYCLE
═══════════════════

Analyze topic/field development stages.

LIFE CYCLE STAGES
─────────────────
1. EMERGENCE
   • New topic appears
   • Few publications
   • Limited citations
   
2. GROWTH
   • Rapid publication increase
   • Growing attention
   • New authors entering
   
3. MATURITY
   • Peak activity
   • Established methods
   • Standard references
   
4. DECLINE
   • Decreasing output
   • Paradigm shifts
   • Author exit
   
5. OBSOLESCENCE
   • Minimal activity
   • Historical interest only

INDICATORS
──────────
• Publication growth rate
• Citation velocity
• Author entry/exit
• Keyword evolution

VISUALIZATION
─────────────
• Life cycle curve
• Stage classification
• Growth rate trends
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

    def on_data_loaded(self, bib):
        """Handle data loaded event."""
        self.bib = bib
        self._result = None
        self._show_placeholder()
