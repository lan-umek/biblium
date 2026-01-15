# -*- coding: utf-8 -*-
"""
Growth Models Panel
===================
Panel for fitting and visualizing bibliometric growth models.

@author: Claude (Anthropic) for Lan.Umek
@version: 2.7.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledSpinbox, LabeledCheckbox
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import add_plot_context_menu, make_canvas_resizable

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


class GrowthModelsPanel(BasePanel):
    """Panel for fitting bibliometric growth models."""
    
    title = "Growth Models"
    icon = "📈"
    description = "Fit growth models and forecast publication trends"
    requires_data = True
    
    MODEL_TYPES = [
        ("auto", "Auto (Best Fit)"),
        ("exponential", "Exponential"),
        ("logistic", "Logistic (S-Curve)"),
        ("power", "Power Law"),
        ("linear", "Linear"),
    ]
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        if not self.bib:
            self._show_no_data_message()
            return
        
        # Info Card
        info_frame = tk.Frame(self.options_content, bg="#e3f2fd", relief=tk.FLAT, bd=1)
        info_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        info_inner = tk.Frame(info_frame, bg="#e3f2fd", padx=8, pady=6)
        info_inner.pack(fill=tk.X)
        
        tk.Label(
            info_inner, text="📈 Growth Models",
            font=FONTS.get_font("body_bold"), bg="#e3f2fd", fg="#1565c0",
        ).pack(anchor=tk.W)
        
        tk.Label(
            info_inner, 
            text="Model publication growth patterns\nand forecast future trends.",
            font=FONTS.get_font("small"), bg="#e3f2fd", fg="#1565c0",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        
        # Model Selection Card
        model_card = Card(self.options_content, title="🔧 Model Settings", theme=self.theme_name)
        model_card.pack(fill=tk.X, padx=8, pady=8)
        
        model_values = [name for _, name in self.MODEL_TYPES]
        self.model_combo = LabeledCombobox(
            model_card.content, label="Model type:",
            values=model_values, default="Auto (Best Fit)",
            theme=self.theme_name, label_width=12,
            tooltip="Growth model to fit. Auto selects best by AIC."
        )
        self.model_combo.pack(fill=tk.X, pady=4)
        
        # Forecast Settings Card
        forecast_card = Card(self.options_content, title="🔮 Forecast Settings", theme=self.theme_name)
        forecast_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.forecast_spin = LabeledSpinbox(
            forecast_card.content, label="Forecast years:",
            from_=1, to=30, default=5,
            theme=self.theme_name, label_width=14,
            tooltip="Number of years to forecast"
        )
        self.forecast_spin.pack(fill=tk.X, pady=4)
        
        self.min_year_spin = LabeledSpinbox(
            forecast_card.content, label="Min year:",
            from_=1900, to=2020, default=1900,
            theme=self.theme_name, label_width=14,
            tooltip="Exclude years before this"
        )
        self.min_year_spin.pack(fill=tk.X, pady=4)
        
        # Options Card
        options_card = Card(self.options_content, title="⚙️ Options", theme=self.theme_name)
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_comparison_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            options_card.content, label="Show model comparison",
            variable=self.show_comparison_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_residuals_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            options_card.content, label="Show residuals plot",
            variable=self.show_residuals_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        self.show_table_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            options_card.content, label="Show predictions table",
            variable=self.show_table_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Action Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Fit Growth Model", icon="📈",
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
            "📈 Growth Model Fitting\n\n"
            "Fit mathematical models to publication growth.\n\n"
            "Features:\n"
            "• Exponential growth models\n"
            "• Logistic (S-curve) models\n"
            "• Polynomial regression\n"
            "• Model comparison statistics\n"
            "\n"
            "Predict future volumes and identify field maturity.\n\n"
            "Steps:\n"
            "1. Load dataset with publication years\n"
            "2. Select model types to fit\n"
            "3. Set projection period\n"
            "4. Click 'Fit Models'\n"
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
        """Run growth model analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Fitting growth model...")
        
        def do_analysis():
            try:
                forecast_years = self.forecast_spin.get()
                min_year = self.min_year_spin.get()
                
                result = self._fit_growth_model(forecast_years, min_year)
                
                self.after(0, lambda r=result: self._display_results(r))
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _fit_growth_model(self, forecast_years: int, min_year: int) -> Dict:
        """Fit growth model using biblium."""
        # Get model type
        model_display = self.model_combo.get()
        model_type = "auto"
        for type_id, name in self.MODEL_TYPES:
            if name == model_display:
                model_type = type_id
                break
        
        result = self.bib.fit_growth_model(
            model_type=model_type,
            forecast_years=forecast_years,
            min_year=min_year if min_year > 1900 else None,
            verbose=False
        )
        
        return result
    
    def _display_results(self, result: Dict):
        """Display growth model results."""
        # Clear previous results
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
        
        # Summary Cards
        grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 12))
        
        model_name = result['model_type'].title()
        grid.add_card(StatsCard(grid, "Model", model_name, "📈", self.theme_name))
        grid.add_card(StatsCard(grid, "R²", f"{result['r_squared']:.4f}", "📊", self.theme_name))
        
        if result.get('doubling_time'):
            grid.add_card(StatsCard(grid, "Doubling Time", f"{result['doubling_time']:.1f} yr", "⏱️", self.theme_name))
        else:
            grid.add_card(StatsCard(grid, "Growth Rate", f"{result['growth_rate']:.4f}", "📈", self.theme_name))
        
        # Parameters card
        param_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        param_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(
            param_frame, text="📐 Model Parameters",
            font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(0, 4))
        
        params_text = "  •  ".join([f"{k}: {v:.4f}" for k, v in result['parameters'].items()])
        tk.Label(
            param_frame, text=params_text,
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W)
        
        # Create Notebook (tabs) for different views
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        
        # Tab 1: Growth Chart
        chart_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(chart_frame, text="📈 Growth Chart")
        
        if HAS_MATPLOTLIB:
            self._plot_growth_model(result, chart_frame)
        
        # Tab 2: Residuals Over Time (if option checked)
        if self.show_residuals_var.get() and HAS_MATPLOTLIB:
            residuals_time_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(residuals_time_frame, text="📉 Residuals Over Time")
            self._plot_residuals_time(result, residuals_time_frame)
            
            # Tab 3: Residuals Distribution
            residuals_dist_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(residuals_dist_frame, text="📊 Residuals Distribution")
            self._plot_residuals_distribution(result, residuals_dist_frame)
        
        # Tab 3: Model Comparison (if option checked)
        if self.show_comparison_var.get() and 'comparison_df' in result:
            comparison_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(comparison_frame, text="📊 Model Comparison")
            
            tk.Label(
                comparison_frame, text="Model Comparison (sorted by AIC)",
                font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
            ).pack(anchor=tk.W, pady=(8, 4), padx=8)
            
            table = DataTable(comparison_frame, theme=self.theme_name, height=5)
            table.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
            table.set_data(result['comparison_df'])
        
        # Tab 4: Predictions Table (if option checked)
        if self.show_table_var.get():
            table_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(table_frame, text="📋 Predictions")

            

            # Info tab

            info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])

            notebook.add(info_frame, text="ℹ️ Info")

            self._create_info_content(info_frame)
            
            tk.Label(
                table_frame, text="Historical + Forecast Predictions",
                font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
            ).pack(anchor=tk.W, pady=(8, 4), padx=8)
            
            pred_df = result['prediction_df'].copy()
            display_pred_df = pd.DataFrame({
                'Year': pred_df['Year'].astype(int),
                'Observed': pred_df['Observed'].apply(lambda x: int(x) if pd.notna(x) else '-'),
                'Fitted': pred_df['Fitted'].round(1),
                'Type': pred_df['Is Forecast'].map({False: 'Historical', True: 'Forecast'})
            })
            
            table = DataTable(table_frame, theme=self.theme_name, height=20)
            table.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
            table.set_data(display_pred_df)
        
        # Buttons at the bottom
        self._add_export_buttons(result)
        
        self._current_result = result
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Growth Model"})
    
    def _plot_growth_model(self, result: Dict, parent: tk.Frame = None):
        """Plot growth model inline."""
        if parent is None:
            parent = self.results_tab
            
        pred_df = result['prediction_df']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        hist_data = pred_df[~pred_df['Is Forecast']]
        forecast_data = pred_df[pred_df['Is Forecast']]
        
        # Observed as bars
        ax.bar(hist_data['Year'], hist_data['Observed'],
               color='#2196F3', alpha=0.7, label='Observed', width=0.8)
        
        # Fitted line
        ax.plot(hist_data['Year'], hist_data['Fitted'],
               color='#4CAF50', linewidth=2.5, label='Fitted')
        
        # Forecast
        if len(forecast_data) > 0:
            ax.plot(forecast_data['Year'], forecast_data['Fitted'],
                   color='#FF9800', linewidth=2.5, linestyle='--',
                   marker='o', markersize=4, label='Forecast')
            ax.fill_between(forecast_data['Year'], 0, forecast_data['Fitted'],
                           color='#FF9800', alpha=0.15)
        
        model_name = result['model_type'].title()
        r2 = result['r_squared']
        ax.set_title(f'{model_name} Growth Model (R² = {r2:.3f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Publications', fontsize=10)
        ax.legend(loc='upper left', frameon=False, fontsize=9)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if len(pred_df) > 20:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Embed
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        _canvas_widget = canvas.get_tk_widget()
        _canvas_widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        add_plot_context_menu(_canvas_widget, fig)
        plt.close(fig)
    
    def _plot_residuals_time(self, result: Dict, parent: tk.Frame = None):
        """Plot residuals over time."""
        if parent is None:
            parent = self.results_tab
            
        residuals = result.get('residuals')
        years = result.get('years')
        
        if residuals is None or years is None:
            tk.Label(
                parent, text="No residuals data available",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        fig, ax = plt.subplots(figsize=(11, 5))
        
        # Color bars based on positive/negative
        colors = ['#26A69A' if r >= 0 else '#EF5350' for r in residuals]
        
        ax.bar(years, residuals, color=colors, alpha=0.8, width=0.8, edgecolor='white', linewidth=0.5)
        ax.axhline(y=0, color='#424242', linestyle='-', linewidth=1.2)
        
        ax.set_title('Residuals Over Time', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Residual (Observed - Fitted)', fontsize=10)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend - smaller
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#26A69A', alpha=0.8, label='Positive'),
            Patch(facecolor='#EF5350', alpha=0.8, label='Negative')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
        
        if len(years) > 20:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Embed
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        _canvas_widget = canvas.get_tk_widget()
        _canvas_widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        add_plot_context_menu(_canvas_widget, fig)
        plt.close(fig)
    
    def _plot_residuals_distribution(self, result: Dict, parent: tk.Frame = None):
        """Plot residuals distribution histogram."""
        if parent is None:
            parent = self.results_tab
            
        residuals = result.get('residuals')
        
        if residuals is None:
            tk.Label(
                parent, text="No residuals data available",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        fig, ax = plt.subplots(figsize=(11, 5))
        
        # Histogram with nice gradient color
        n, bins, patches = ax.hist(residuals, bins=min(20, len(residuals)//2 + 1), 
                                   alpha=0.85, edgecolor='white', linewidth=0.8)
        
        # Color gradient based on position
        cm = plt.cm.get_cmap('RdYlGn_r')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min() + 0.001)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        ax.axvline(x=0, color='#424242', linestyle='--', linewidth=1.5, label='Zero')
        
        ax.set_title('Residuals Distribution', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Residual Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=False, fontsize=8)
        
        # Add stats box (without mean since it's always ~0)
        std_res = np.std(residuals)
        min_res = np.min(residuals)
        max_res = np.max(residuals)
        stats_text = f'Std: {std_res:.2f}\nMin: {min_res:.2f}\nMax: {max_res:.2f}'
        ax.annotate(stats_text, 
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', alpha=0.9, edgecolor='#1565C0'))
        
        plt.tight_layout()
        
        # Embed
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        _canvas_widget = canvas.get_tk_widget()
        _canvas_widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        add_plot_context_menu(_canvas_widget, fig)
        plt.close(fig)
    
    def _add_export_buttons(self, result: Dict):
        """Add export buttons."""
        btn_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=8)
        
        ThemedButton(
            btn_frame, text="📥 Export Data",
            command=lambda: self._export_data(result), theme=self.theme_name
        ).pack(side=tk.LEFT, padx=4)
        
        ThemedButton(
            btn_frame, text="💾 Save Plot",
            command=lambda: self._save_plot(result), theme=self.theme_name
        ).pack(side=tk.LEFT, padx=4)
    
    def _export_data(self, result: Dict):
        """Export prediction data."""
        pred_df = result.get('prediction_df')
        if pred_df is None or len(pred_df) == 0:
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Export Growth Model Data"
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.csv'):
                pred_df.to_csv(filepath, index=False)
            else:
                pred_df.to_excel(filepath, index=False)
            
            messagebox.showinfo("Success", f"Data exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def _save_plot(self, result: Dict):
        """Save plot to file."""
        from biblium import plotbib
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")],
            title="Save Plot"
        )
        
        if not filepath:
            return
        
        try:
            base = filepath.rsplit('.', 1)[0] if '.' in filepath else filepath
            
            plotbib.plot_growth_model(
                result, 
                filename=base, 
                show=False,
                show_residuals=self.show_residuals_var.get()
            )
            
            messagebox.showinfo("Success", f"Plot saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{str(e)}")
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading indicator."""
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
        
        tk.Label(
            frame, text="⏳", font=("Segoe UI", 32),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(20, 10))
        
        tk.Label(
            frame, text=message, font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"]
        ).pack()
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
GROWTH MODEL FITTING
════════════════════

Fit mathematical models to publication growth data.

AVAILABLE MODELS
────────────────
• Exponential
  N(t) = N₀ × e^(rt)
  - Constant percentage growth
  - Typical early-stage fields
  - Doubling time = ln(2)/r
  
• Logistic (S-Curve)
  N(t) = K / (1 + e^(-r(t-t₀)))
  - Growth with saturation
  - K = carrying capacity
  - t₀ = inflection point
  - Typical maturing fields
  
• Linear
  N(t) = a + bt
  - Constant absolute growth
  - Simple baseline model
  
• Power Law
  N(t) = a × t^b
  - Polynomial growth
  - Flexible curvature

MODEL COMPARISON
────────────────
• R² (coefficient of determination)
  - Variance explained
  - Higher = better fit
  
• AIC (Akaike Information Criterion)
  - Model selection
  - Lower = better (penalizes complexity)
  
• RMSE (Root Mean Square Error)
  - Prediction accuracy
  - Lower = better fit

FORECASTING
───────────
• Project future publication counts
• Confidence intervals
• Extrapolation cautions

OUTPUT
──────
• Fitted parameters
• Model comparison table
• Observed vs fitted plot
• Growth rate estimates
• Projections (optional)
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

    def _show_error(self, message: str):
        """Show error message."""
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
            text=f"❌ Error\n\n{message}",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["error"], justify=tk.CENTER, wraplength=400,
        ).pack(expand=True)
