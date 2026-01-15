# -*- coding: utf-8 -*-
"""
Overview Panel
==============
Dataset summary with comprehensive statistics.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid, ScrollableStatsRow
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCheckbox
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class OverviewPanel(BasePanel):
    """Panel for dataset overview and summary statistics."""
    
    title = "Overview"
    icon = "ℹ️"
    description = "Comprehensive summary of your bibliometric dataset"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._main_info = None
        
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._generate_overview  # Set primary action for toolbar Run button
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Quick Stats Card
        stats_card = Card(self.options_content, title="📊 Quick Statistics", theme=self.theme_name)
        stats_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.stats_content = tk.Frame(stats_card.content, bg=self.theme["bg_card"])
        self.stats_content.pack(fill=tk.X)
        
        if self.bib:
            self._show_quick_stats()
        else:
            tk.Label(
                self.stats_content, text="Load a dataset to see statistics",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            ).pack(pady=8)
        
        # Options Card
        options_card = Card(self.options_content, title="⚙️ Options", theme=self.theme_name)
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_plots_cb = LabeledCheckbox(
            options_card.content, label="Generate overview plots",
            default=True, theme=self.theme_name,
        )
        self.show_plots_cb.pack(fill=tk.X, pady=2)
        
        self.show_main_info_cb = LabeledCheckbox(
            options_card.content, label="Show detailed main info",
            default=True, theme=self.theme_name,
        )
        self.show_main_info_cb.pack(fill=tk.X, pady=2)
        
        self.show_coverage_cb = LabeledCheckbox(
            options_card.content, label="Show data coverage analysis",
            default=True, theme=self.theme_name,
        )
        self.show_coverage_cb.pack(fill=tk.X, pady=2)
        
        self.analyze_quality_cb = LabeledCheckbox(
            options_card.content, label="Analyze data quality (missing data)",
            default=True, theme=self.theme_name,
        )
        self.analyze_quality_cb.pack(fill=tk.X, pady=2)
        
        # Generate Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Generate Overview", icon="📊",
            command=self._generate_overview, theme=self.theme_name,
        ).pack(fill=tk.X)
    
    def _show_quick_stats(self):
        """Show quick statistics in options panel."""
        for widget in self.stats_content.winfo_children():
            widget.destroy()
        
        if not self.bib:
            return
        
        stats = [
            ("Documents", f"{self.bib.n:,}"),
        ]
        
        # Year range
        year_col = self.bib.mapping.get("Year", "Year")
        if year_col in self.bib.df.columns:
            # Convert to numeric, coercing errors to NaN
            years = pd.to_numeric(self.bib.df[year_col], errors="coerce").dropna()
            if len(years) > 0:
                min_year = int(years.min())
                max_year = int(years.max())
                stats.append(("Years", f"{min_year}-{max_year}"))
                stats.append(("Timespan", f"{max_year - min_year + 1} years"))
        
        # Database
        stats.append(("Database", self.bib.db.upper()))
        
        for label, value in stats:
            row = tk.Frame(self.stats_content, bg=self.theme["bg_card"])
            row.pack(fill=tk.X, pady=2)
            
            tk.Label(
                row, text=label + ":", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_secondary"], width=12, anchor=tk.W,
            ).pack(side=tk.LEFT)
            
            tk.Label(
                row, text=value, font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_card"], fg=self.theme["accent_primary"],
            ).pack(side=tk.LEFT)
    
    def _create_results(self):
        """Create the results panel."""
        # Results container
        self.results_card = tk.Frame(
            self.results_frame, bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Header
        header = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header, text="Dataset Overview",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            padx=12, pady=8,
        ).pack(side=tk.LEFT)
        
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        # Scrollable content
        canvas = tk.Canvas(self.results_card, bg=self.theme["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.results_card, orient=tk.VERTICAL, command=canvas.yview)
        
        self.results_content = tk.Frame(canvas, bg=self.theme["bg_card"])
        
        self.results_content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.results_content, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        # Bind mouse wheel
        def _on_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        
        self._show_initial_overview()
    
    def _show_initial_overview(self):
        """Show initial overview message."""
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.results_content, text="Click 'Generate Overview' to analyze your dataset",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(expand=True, pady=50)
    
    def _generate_overview(self):
        """Generate the dataset overview."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Generating overview...")
        
        def do_generate():
            try:
                info = {}
                df = self.bib.df
                
                # Use biblium's summary() for basic info
                if hasattr(self.bib, 'summary'):
                    try:
                        summary = self.bib.summary(verbose=False)
                        info["n_documents"] = summary.n_documents
                        info["n_columns"] = summary.n_columns
                        info["database"] = summary.database
                        info["year_range"] = summary.year_range
                        info["top_authors"] = summary.top_authors
                        info["top_sources"] = summary.top_sources
                        info["top_keywords"] = summary.top_keywords
                        info["missing_rates"] = summary.missing_rates
                        info["summary_available"] = True
                    except Exception as e:
                        print(f"summary() failed: {e}")
                        info["summary_available"] = False
                        info["n_documents"] = self.bib.n
                        info["n_columns"] = len(df.columns)
                else:
                    info["n_documents"] = self.bib.n
                    info["n_columns"] = len(df.columns)
                    info["summary_available"] = False
                
                # Use biblium's get_main_info() for detailed analysis
                if self.show_main_info_cb.get() and hasattr(self.bib, 'get_main_info'):
                    try:
                        self.bib.get_main_info()
                        info["descriptives_df"] = getattr(self.bib, 'descriptives_df', None)
                        info["performances_df"] = getattr(self.bib, 'performances_df', None)
                        info["production_df"] = getattr(self.bib, 'production_df', None)
                        info["time_series_stats_df"] = getattr(self.bib, 'time_series_stats_df', None)
                        info["main_info_available"] = True
                    except Exception as e:
                        print(f"get_main_info() failed: {e}")
                        info["main_info_available"] = False
                else:
                    info["main_info_available"] = False
                
                # Fallback: Year analysis from df if not available from biblium
                if "year_range" not in info or info.get("year_range") is None:
                    year_col = self.bib.mapping.get("Year", "Year")
                    if year_col in df.columns:
                        years = df[year_col].dropna()
                        if len(years) > 0:
                            info["year_min"] = int(years.min())
                            info["year_max"] = int(years.max())
                            info["year_counts"] = years.value_counts().sort_index().to_dict()
                
                # Extract year_counts from production_df if available
                if info.get("production_df") is not None and len(info["production_df"]) > 0:
                    prod_df = info["production_df"]
                    if "Year" in prod_df.columns and "Number of Documents" in prod_df.columns:
                        info["year_counts"] = dict(zip(prod_df["Year"], prod_df["Number of Documents"]))
                        info["year_min"] = int(prod_df["Year"].min())
                        info["year_max"] = int(prod_df["Year"].max())
                
                # Citation analysis from performances_df or fallback
                if info.get("performances_df") is not None:
                    try:
                        perf_df = info["performances_df"]
                        for _, row in perf_df.iterrows():
                            indicator = str(row.get("Indicator", "")).lower()
                            value = row.get("Value", 0)
                            if "total citation" in indicator:
                                info["total_citations"] = int(value) if pd.notna(value) else 0
                            elif "mean citation" in indicator or "average citation" in indicator:
                                info["mean_citations"] = float(value) if pd.notna(value) else 0
                    except:
                        pass
                
                # Fallback citation analysis
                if "total_citations" not in info:
                    cit_col = self.bib.mapping.get("Cited_by", "Cited by")
                    if cit_col in df.columns:
                        cits = pd.to_numeric(df[cit_col], errors='coerce').fillna(0)
                        info["total_citations"] = int(cits.sum())
                        info["mean_citations"] = round(cits.mean(), 2)
                        info["max_citations"] = int(cits.max())
                        info["median_citations"] = int(cits.median())
                
                # Document types from descriptives_df or fallback
                if info.get("descriptives_df") is not None:
                    try:
                        desc_df = info["descriptives_df"]
                        doc_types = desc_df[desc_df["Variable"].str.contains("Document Type", case=False, na=False)]
                        if len(doc_types) > 0:
                            info["document_types"] = dict(zip(doc_types["Indicator"], doc_types["Value"]))
                    except:
                        pass
                
                # Fallback document types
                if "document_types" not in info:
                    doctype_col = self.bib.mapping.get("Document_Type", "Document Type")
                    if doctype_col in df.columns:
                        info["document_types"] = df[doctype_col].value_counts().head(10).to_dict()
                
                # Coverage analysis
                if self.show_coverage_cb.get():
                    # Use missing_rates from summary if available
                    if info.get("missing_rates"):
                        coverage = {col: round(100 - rate, 1) for col, rate in info["missing_rates"].items()}
                        info["coverage"] = coverage
                    else:
                        coverage = {}
                        for col in df.columns:
                            non_null = df[col].notna().sum()
                            coverage[col] = round(non_null / len(df) * 100, 1)
                        info["coverage"] = coverage
                
                # Data quality analysis using biblium
                if self.analyze_quality_cb.get() and hasattr(self.bib, 'analyze_data_quality'):
                    try:
                        self.bib.analyze_data_quality()
                        info["quality_score"] = getattr(self.bib, 'data_quality_score', None)
                        info["quality_details"] = getattr(self.bib, 'data_quality_details', None)
                        info["missing_data_report"] = getattr(self.bib, 'missing_data_report', None)
                        info["data_quality_available"] = True
                    except Exception as e:
                        print(f"Data quality analysis failed: {e}")
                        info["data_quality_available"] = False
                else:
                    info["data_quality_available"] = False
                
                self.after(0, lambda: self._on_overview_success(info))
            except Exception as e:
                import traceback
                self.after(0, lambda: self._on_overview_error(str(e)))
        
        threading.Thread(target=do_generate, daemon=True).start()
    
    def _on_overview_success(self, info: Dict):
        """Display the overview results."""
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        # Key Metrics Section
        self._add_section_header("📊 Key Metrics")
        
        grid = ScrollableStatsRow(self.results_content, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        grid.add_card(StatsCard(grid.inner_frame, "Documents", f"{info['n_documents']:,}", "📄", self.theme_name, accent=True))
        
        if "total_citations" in info:
            grid.add_card(StatsCard(grid.inner_frame, "Total Citations", f"{info['total_citations']:,}", "📈", self.theme_name))
        if "mean_citations" in info:
            grid.add_card(StatsCard(grid.inner_frame, "Avg Citations", f"{info['mean_citations']:.1f}", "📊", self.theme_name))
        
        # Year range from biblium summary or fallback
        if "year_range" in info and info["year_range"]:
            grid.add_card(StatsCard(grid.inner_frame, "Year Range", str(info["year_range"]), "📅", self.theme_name))
        elif "year_min" in info and info.get("year_min"):
            timespan = info["year_max"] - info["year_min"] + 1
            grid.add_card(StatsCard(grid.inner_frame, "Timespan", f"{timespan} years", "📅", self.theme_name))
        
        if "database" in info and info["database"]:
            grid.add_card(StatsCard(grid.inner_frame, "Database", str(info["database"]).upper(), "🗄️", self.theme_name))
        
        # Descriptives from biblium (if available)
        if info.get("main_info_available") and info.get("descriptives_df") is not None:
            self._add_section_header("📋 Dataset Descriptives")
            
            desc_df = info["descriptives_df"]
            # Group by Variable
            try:
                variables = desc_df["Variable"].unique()
                for var in variables[:8]:  # Show first 8 variable groups
                    var_data = desc_df[desc_df["Variable"] == var]
                    if len(var_data) > 0:
                        var_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
                        var_frame.pack(fill=tk.X, pady=(0, 8))
                        
                        tk.Label(
                            var_frame, text=f"{var}:",
                            font=FONTS.get_font("body", bold=True),
                            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                        ).pack(anchor=tk.W)
                        
                        for _, row in var_data.head(5).iterrows():
                            indicator = row.get("Indicator", "")
                            value = row.get("Value", "")
                            
                            row_frame = tk.Frame(var_frame, bg=self.theme["bg_card"])
                            row_frame.pack(fill=tk.X, padx=(16, 0))
                            
                            tk.Label(
                                row_frame, text=f"  {indicator}:",
                                font=FONTS.get_font("small"),
                                bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                                width=30, anchor=tk.W,
                            ).pack(side=tk.LEFT)
                            
                            tk.Label(
                                row_frame, text=str(value),
                                font=FONTS.get_font("small"),
                                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                            ).pack(side=tk.LEFT)
            except Exception as e:
                print(f"Error displaying descriptives: {e}")
        
        # Top Authors, Sources, Keywords from summary
        if info.get("summary_available"):
            if info.get("top_authors"):
                self._add_section_header("👤 Top Authors")
                self._show_top_items(info["top_authors"])
            
            if info.get("top_sources"):
                self._add_section_header("📚 Top Sources")
                self._show_top_items(info["top_sources"])
            
            if info.get("top_keywords"):
                self._add_section_header("🏷️ Top Keywords")
                self._show_top_items(info["top_keywords"])
        
        # Production Over Time Plot (using production_df if available)
        if self.show_plots_cb.get() and HAS_MATPLOTLIB:
            if info.get("production_df") is not None and len(info["production_df"]) > 0:
                self._add_section_header("📈 Scientific Production Over Time")
                
                plot_info = {
                    "type": "bar chart with trend line",
                    "title": "Annual Scientific Production",
                    "x_label": "Year",
                    "y_label": "Number of Documents",
                    "data_summary": f"{len(info['production_df'])} years of data",
                }
                plot = PlotFrame(self.results_content, theme=self.theme_name, figsize=(10, 4),
                                 show_ai_button=True, plot_info=plot_info)
                plot.pack(fill=tk.X, pady=(0, 16))
                
                fig, ax = plot.get_figure()
                prod_df = info["production_df"]
                
                years = prod_df["Year"].tolist()
                counts = prod_df["Number of Documents"].tolist()
                
                ax.bar(years, counts, color=self.theme["accent_primary"], alpha=0.8)
                ax.set_xlabel("Year")
                ax.set_ylabel("Number of Documents")
                ax.set_title("Annual Scientific Production")
                
                # Add trend line
                if len(years) > 2:
                    import numpy as np
                    z = np.polyfit(years, counts, 1)
                    p = np.poly1d(z)
                    ax.plot(years, p(years), "r--", alpha=0.8, label="Trend")
                
                fig.tight_layout()
                plot.refresh()
            
            elif "year_counts" in info:
                self._add_section_header("📈 Scientific Production Over Time")
                
                plot_info = {
                    "type": "bar chart with trend line",
                    "title": "Annual Scientific Production",
                    "x_label": "Year",
                    "y_label": "Number of Documents",
                    "data_summary": f"{len(info['year_counts'])} years of data",
                }
                plot = PlotFrame(self.results_content, theme=self.theme_name, figsize=(10, 4),
                                 show_ai_button=True, plot_info=plot_info)
                plot.pack(fill=tk.X, pady=(0, 16))
                
                fig, ax = plot.get_figure()
                years = sorted(info["year_counts"].keys())
                counts = [info["year_counts"][y] for y in years]
                
                ax.bar(years, counts, color=self.theme["accent_primary"], alpha=0.8)
                ax.set_xlabel("Year")
                ax.set_ylabel("Number of Documents")
                ax.set_title("Annual Scientific Production")
                
                if len(years) > 2:
                    import numpy as np
                    z = np.polyfit(years, counts, 1)
                    p = np.poly1d(z)
                    ax.plot(years, p(years), "r--", alpha=0.8, label="Trend")
                
                fig.tight_layout()
                plot.refresh()
        
        # Performance metrics from biblium
        if info.get("main_info_available") and info.get("performances_df") is not None:
            self._add_section_header("🏆 Performance Metrics")
            
            perf_df = info["performances_df"]
            perf_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
            perf_frame.pack(fill=tk.X, pady=(0, 16))
            
            try:
                for _, row in perf_df.head(15).iterrows():
                    var = row.get("Variable", "")
                    indicator = row.get("Indicator", "")
                    value = row.get("Value", "")
                    
                    row_frame = tk.Frame(perf_frame, bg=self.theme["bg_card"])
                    row_frame.pack(fill=tk.X, pady=1)
                    
                    label_text = f"{var} - {indicator}" if var else indicator
                    tk.Label(
                        row_frame, text=label_text[:40],
                        font=FONTS.get_font("small"),
                        bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                        width=35, anchor=tk.W,
                    ).pack(side=tk.LEFT)
                    
                    tk.Label(
                        row_frame, text=str(value),
                        font=FONTS.get_font("small", bold=True),
                        bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    ).pack(side=tk.LEFT)
            except Exception as e:
                print(f"Error displaying performance: {e}")
        
        # Document Types
        if "document_types" in info:
            self._add_section_header("📄 Document Types")
            
            types_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
            types_frame.pack(fill=tk.X, pady=(0, 16))
            
            for dtype, count in list(info["document_types"].items())[:10]:
                row = tk.Frame(types_frame, bg=self.theme["bg_card"])
                row.pack(fill=tk.X, pady=2)
                
                try:
                    count_val = int(count) if pd.notna(count) else 0
                    pct = count_val / info["n_documents"] * 100
                except:
                    count_val = count
                    pct = 0
                
                tk.Label(
                    row, text=str(dtype)[:30], font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"], width=25, anchor=tk.W,
                ).pack(side=tk.LEFT)
                
                tk.Label(
                    row, text=f"{count_val:,} ({pct:.1f}%)" if pct > 0 else str(count_val),
                    font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                ).pack(side=tk.LEFT)
        
        # Data Coverage
        if self.show_coverage_cb.get() and "coverage" in info:
            self._add_section_header("📋 Data Coverage")
            
            coverage_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
            coverage_frame.pack(fill=tk.X, pady=(0, 16))
            
            # Show columns with coverage percentage
            sorted_coverage = sorted(info["coverage"].items(), key=lambda x: x[1], reverse=True)
            
            for col, pct in sorted_coverage[:15]:  # Top 15
                row = tk.Frame(coverage_frame, bg=self.theme["bg_card"])
                row.pack(fill=tk.X, pady=1)
                
                tk.Label(
                    row, text=col[:30], font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"], width=25, anchor=tk.W,
                ).pack(side=tk.LEFT)
                
                # Progress bar
                bar_frame = tk.Frame(row, bg=self.theme["border"], height=12, width=150)
                bar_frame.pack(side=tk.LEFT, padx=8)
                bar_frame.pack_propagate(False)
                
                fill_width = int(150 * pct / 100)
                color = self.theme["success"] if pct >= 80 else (self.theme["warning"] if pct >= 50 else self.theme["danger"])
                
                fill = tk.Frame(bar_frame, bg=color, width=fill_width, height=12)
                fill.pack(side=tk.LEFT)
                
                tk.Label(
                    row, text=f"{pct:.0f}%", font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"], width=5,
                ).pack(side=tk.LEFT)
        
        # Data Quality Analysis (from biblium)
        if info.get("data_quality_available"):
            self._add_section_header("🔍 Data Quality Analysis")
            
            quality_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
            quality_frame.pack(fill=tk.X, pady=(0, 16))
            
            # Overall quality score
            quality_score = info.get("quality_score")
            if quality_score is not None:
                try:
                    score_val = float(quality_score)
                    score_color = self.theme["success"] if score_val >= 80 else (self.theme["warning"] if score_val >= 60 else self.theme["danger"])
                    
                    score_row = tk.Frame(quality_frame, bg=self.theme["bg_card"])
                    score_row.pack(fill=tk.X, pady=(0, 12))
                    
                    tk.Label(
                        score_row, text="Overall Quality Score:",
                        font=FONTS.get_font("body", bold=True),
                        bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    ).pack(side=tk.LEFT)
                    
                    tk.Label(
                        score_row, text=f"  {score_val:.1f}/100",
                        font=FONTS.get_font("heading", bold=True),
                        bg=self.theme["bg_card"], fg=score_color,
                    ).pack(side=tk.LEFT)
                except:
                    pass
            
            # Quality details (key columns)
            quality_details = info.get("quality_details")
            if quality_details is not None and len(quality_details) > 0:
                tk.Label(
                    quality_frame, text="Key Column Completeness:",
                    font=FONTS.get_font("body", bold=True),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    anchor=tk.W,
                ).pack(fill=tk.X, pady=(8, 4))
                
                try:
                    for _, row_data in quality_details.iterrows():
                        # Get column name and completeness from correct columns
                        col_name = row_data.get("Column", row_data.iloc[0] if len(row_data) > 0 else "")
                        # Use "Completeness %" column if available
                        if "Completeness %" in row_data.index:
                            completeness = row_data["Completeness %"]
                        elif len(row_data) > 2:
                            completeness = row_data.iloc[2]  # Completeness % is 3rd column
                        else:
                            completeness = 0
                        
                        row = tk.Frame(quality_frame, bg=self.theme["bg_card"])
                        row.pack(fill=tk.X, pady=1)
                        
                        tk.Label(
                            row, text=str(col_name)[:25], font=FONTS.get_font("small"),
                            bg=self.theme["bg_card"], fg=self.theme["text_primary"], width=20, anchor=tk.W,
                        ).pack(side=tk.LEFT)
                        
                        try:
                            pct = float(completeness)
                        except:
                            pct = 0
                        
                        # Progress bar
                        bar_frame = tk.Frame(row, bg=self.theme["border"], height=10, width=120)
                        bar_frame.pack(side=tk.LEFT, padx=8)
                        bar_frame.pack_propagate(False)
                        
                        fill_width = int(120 * pct / 100)
                        color = self.theme["success"] if pct >= 80 else (self.theme["warning"] if pct >= 50 else self.theme["danger"])
                        
                        fill = tk.Frame(bar_frame, bg=color, width=fill_width, height=10)
                        fill.pack(side=tk.LEFT)
                        
                        tk.Label(
                            row, text=f"{pct:.1f}%", font=FONTS.get_font("small"),
                            bg=self.theme["bg_card"], fg=self.theme["text_muted"], width=6,
                        ).pack(side=tk.LEFT)
                except Exception as e:
                    print(f"Error displaying quality details: {e}")
            
            # Missing data report (top issues)
            missing_report = info.get("missing_data_report")
            if missing_report is not None and len(missing_report) > 0:
                tk.Label(
                    quality_frame, text="Top Missing Data Issues:",
                    font=FONTS.get_font("body", bold=True),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    anchor=tk.W,
                ).pack(fill=tk.X, pady=(12, 4))
                
                try:
                    # Filter to show only columns with actual missing data
                    if "% Missing" in missing_report.columns:
                        missing_filtered = missing_report[missing_report["% Missing"] > 0].head(10)
                    else:
                        missing_filtered = missing_report.head(10)
                    
                    if len(missing_filtered) == 0:
                        tk.Label(
                            quality_frame, text="  ✓ No significant missing data issues!",
                            font=FONTS.get_font("small"),
                            bg=self.theme["bg_card"], fg=self.theme["success"],
                        ).pack(fill=tk.X, pady=1)
                    else:
                        for i, (_, row_data) in enumerate(missing_filtered.iterrows()):
                            col_name = row_data.get("Column", row_data.iloc[0] if len(row_data) > 0 else "")
                            
                            # Get the missing percentage from the correct column
                            if "% Missing" in row_data.index:
                                missing_pct = float(row_data["% Missing"])
                            else:
                                missing_pct = 0
                            
                            if missing_pct > 0:
                                row = tk.Frame(quality_frame, bg=self.theme["bg_card"])
                                row.pack(fill=tk.X, pady=1)
                                
                                tk.Label(
                                    row, text=f"  • {str(col_name)[:30]}", font=FONTS.get_font("small"),
                                    bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                                ).pack(side=tk.LEFT)
                                
                                tk.Label(
                                    row, text=f" ({missing_pct:.1f}% missing)",
                                    font=FONTS.get_font("small"),
                                    bg=self.theme["bg_card"], fg=self.theme["danger"],
                                ).pack(side=tk.LEFT)
                except Exception as e:
                    print(f"Error displaying missing report: {e}")
        
        # Citation Distribution
        if self.show_plots_cb.get() and "mean_citations" in info and HAS_MATPLOTLIB:
            self._add_section_header("📊 Citation Statistics")
            
            cit_stats = tk.Frame(self.results_content, bg=self.theme["bg_card"])
            cit_stats.pack(fill=tk.X, pady=(0, 16))
            
            stats_data = [
                ("Total Citations", f"{info['total_citations']:,}"),
                ("Mean Citations", f"{info['mean_citations']:.2f}"),
                ("Median Citations", f"{info['median_citations']}"),
                ("Max Citations", f"{info['max_citations']:,}"),
            ]
            
            for label, value in stats_data:
                row = tk.Frame(cit_stats, bg=self.theme["bg_card"])
                row.pack(fill=tk.X, pady=2)
                
                tk.Label(
                    row, text=label + ":", font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_secondary"], width=20, anchor=tk.W,
                ).pack(side=tk.LEFT)
                
                tk.Label(
                    row, text=value, font=FONTS.get_font("body", bold=True),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                ).pack(side=tk.LEFT)
    
    def _add_section_header(self, title: str):
        """Add a section header."""
        tk.Label(
            self.results_content, text=title,
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(16, 8))
        
        ttk.Separator(self.results_content, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 8))
    
    def _show_top_items(self, items):
        """Show top items (authors, sources, keywords) from biblium summary."""
        if not items:
            return
        
        items_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
        items_frame.pack(fill=tk.X, pady=(0, 16))
        
        try:
            # Handle different formats (dict, list of tuples, DataFrame)
            if isinstance(items, dict):
                item_list = list(items.items())[:10]
            elif hasattr(items, 'items'):
                item_list = list(items.items())[:10]
            elif hasattr(items, 'iterrows'):  # DataFrame
                item_list = [(row.iloc[0], row.iloc[1]) for _, row in items.head(10).iterrows()]
            elif isinstance(items, (list, tuple)):
                item_list = items[:10]
            else:
                item_list = list(items)[:10]
            
            for item in item_list:
                row = tk.Frame(items_frame, bg=self.theme["bg_card"])
                row.pack(fill=tk.X, pady=1)
                
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    name, count = item[0], item[1]
                else:
                    name, count = str(item), ""
                
                tk.Label(
                    row, text=f"  • {str(name)[:35]}",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    anchor=tk.W,
                ).pack(side=tk.LEFT)
                
                if count:
                    tk.Label(
                        row, text=f"  ({count})",
                        font=FONTS.get_font("small"),
                        bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                    ).pack(side=tk.LEFT)
        except Exception as e:
            print(f"Error showing top items: {e}")
    
    def _on_overview_error(self, error: str):
        """Handle overview error."""
        self._show_error(f"Overview error: {error}")
    
    def _on_dataset_loaded(self, data):
        """Handle dataset loaded event."""
        super()._on_dataset_loaded(data)
        self._show_quick_stats()
