# -*- coding: utf-8 -*-
"""
Trends Panel
============
Temporal analysis and trend visualization.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Dict, List, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme, ENTITY_TYPES
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
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TrendsPanel(BasePanel):
    """Panel for temporal trend analysis."""
    
    title = "Trend Analysis"
    icon = "📈"
    description = "Analyze temporal patterns and trends in your data"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._trend_data = None
        
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._analyze_trends  # Set primary action for toolbar Run button
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Analysis Type Card
        type_card = Card(self.options_content, title="📊 Analysis Type", theme=self.theme_name)
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.analysis_type = LabeledCombobox(
            type_card.content, label="Type:",
            values=[
                "Scientific Production",
                "Citation Trends",
            ],
            default="Scientific Production",
            theme=self.theme_name, label_width=12,
        )
        self.analysis_type.pack(fill=tk.X, pady=4)
        
        # Time Period Card
        period_card = Card(self.options_content, title="📅 Time Period", theme=self.theme_name)
        period_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Year range
        year_frame = tk.Frame(period_card.content, bg=self.theme["bg_card"])
        year_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            year_frame, text="From:", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.year_from_spin = tk.Spinbox(
            year_frame, from_=1900, to=2030, width=6,
            font=FONTS.get_font("body"),
        )
        self.year_from_spin.pack(side=tk.LEFT, padx=(4, 16))
        
        tk.Label(
            year_frame, text="To:", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.year_to_spin = tk.Spinbox(
            year_frame, from_=1900, to=2030, width=6,
            font=FONTS.get_font("body"),
        )
        self.year_to_spin.pack(side=tk.LEFT, padx=4)
        
        # Cut year option - merge documents before this year
        cut_frame = tk.Frame(period_card.content, bg=self.theme["bg_card"])
        cut_frame.pack(fill=tk.X, pady=4)
        
        self.use_cut_year_cb = LabeledCheckbox(
            cut_frame, label="Merge years before:",
            default=False, theme=self.theme_name,
        )
        self.use_cut_year_cb.pack(side=tk.LEFT)
        
        self.cut_year_spin = tk.Spinbox(
            cut_frame, from_=1900, to=2030, width=6,
            font=FONTS.get_font("body"),
        )
        self.cut_year_spin.pack(side=tk.LEFT, padx=4)
        self.cut_year_spin.delete(0, tk.END)
        self.cut_year_spin.insert(0, "2000")
        
        # Set defaults from data
        if self.bib:
            self._set_year_defaults()
        
        # Aggregation
        self.aggregation = LabeledCombobox(
            period_card.content, label="Aggregation:",
            values=["Yearly", "5-Year", "Decade", "Cumulative"],
            default="Yearly",
            theme=self.theme_name, label_width=12,
        )
        self.aggregation.pack(fill=tk.X, pady=4)
        
        # Plot Options Card
        plot_card = Card(self.options_content, title="📈 Plot Options", theme=self.theme_name)
        plot_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Primary axis (bars)
        self.bar_metric = LabeledCombobox(
            plot_card.content, label="Bars (primary):",
            values=["Documents", "Total Citations", "None"],
            default="Documents",
            theme=self.theme_name, label_width=14,
        )
        self.bar_metric.pack(fill=tk.X, pady=4)
        
        # Secondary axis (line)
        self.line_metric = LabeledCombobox(
            plot_card.content, label="Line (secondary):",
            values=["Cumulative Citations", "Cumulative Documents", "Avg Citations", "None"],
            default="Cumulative Citations",
            theme=self.theme_name, label_width=14,
        )
        self.line_metric.pack(fill=tk.X, pady=4)
        
        self.show_bar_labels_cb = LabeledCheckbox(
            plot_card.content, label="Show bar labels",
            default=False, theme=self.theme_name,
        )
        self.show_bar_labels_cb.pack(fill=tk.X, pady=2)
        
        # Trendline Options Card
        trendline_card = CollapsibleCard(
            self.options_content, title="📉 Trendline Options",
            collapsed=True, theme=self.theme_name,
        )
        trendline_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_trend_cb = LabeledCheckbox(
            trendline_card.content, label="Show trendline",
            default=False, theme=self.theme_name,
        )
        self.show_trend_cb.pack(fill=tk.X, pady=2)
        
        self.trend_type = LabeledCombobox(
            trendline_card.content, label="Trendline type:",
            values=["Linear", "Exponential", "Polynomial (2)", "Polynomial (3)", "Moving Average"],
            default="Linear",
            theme=self.theme_name, label_width=14,
        )
        self.trend_type.pack(fill=tk.X, pady=4)
        
        self.moving_avg_window = LabeledSpinbox(
            trendline_card.content, label="Moving avg window:",
            from_=2, to=10, default=3,
            theme=self.theme_name, label_width=14,
        )
        self.moving_avg_window.pack(fill=tk.X, pady=4)
        
        self.show_avg_cb = LabeledCheckbox(
            trendline_card.content, label="Show average line",
            default=False, theme=self.theme_name,
        )
        self.show_avg_cb.pack(fill=tk.X, pady=2)
        
        # Advanced Options
        advanced_card = CollapsibleCard(
            self.options_content, title="Advanced Options",
            collapsed=True, theme=self.theme_name,
        )
        advanced_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.top_n = LabeledSpinbox(
            advanced_card.content, label="Top N items:",
            from_=3, to=20, default=10,
            theme=self.theme_name, label_width=12,
        )
        self.top_n.pack(fill=tk.X, pady=4)
        
        self.normalize_cb = LabeledCheckbox(
            advanced_card.content, label="Normalize values",
            default=False, theme=self.theme_name,
        )
        self.normalize_cb.pack(fill=tk.X, pady=2)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Analyze Trends", icon="📈",
            command=self._analyze_trends, theme=self.theme_name,
        ).pack(fill=tk.X)
    
    def _set_year_defaults(self):
        """Set year defaults from data."""
        if not self.bib:
            return
        
        year_col = self.bib.mapping.get("Year", "Year")
        if year_col in self.bib.df.columns:
            years = self.bib.df[year_col].dropna()
            if len(years) > 0:
                self.year_from_spin.delete(0, tk.END)
                self.year_from_spin.insert(0, str(int(years.min())))
                self.year_to_spin.delete(0, tk.END)
                self.year_to_spin.insert(0, str(int(years.max())))
    
    def _create_results(self):
        """Create the results panel."""
        self.results_card = tk.Frame(
            self.results_frame, bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Header
        header = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header, text="Trend Analysis Results",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            padx=12, pady=8,
        ).pack(side=tk.LEFT)
        
        ThemedButton(
            header, text="📥 Export", style="ghost", size="small",
            command=self._export_results, theme=self.theme_name,
        ).pack(side=tk.RIGHT, padx=8, pady=4)
        
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.results_card)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Chart tab
        self.chart_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.chart_frame, text="  📈 Chart  ")
        
        # Table tab
        self.table_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.table_frame, text="  📊 Data  ")
        
        # Statistics tab
        self.stats_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.stats_frame, text="  📉 Statistics  ")

        

        # Info tab

        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])

        self.notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        # Set results_content to chart_frame for base class compatibility
        self.results_content = self.chart_frame
        
        self._show_trends_message("Select analysis type and click 'Analyze Trends'")
    
    def _show_trends_message(self, message: str):
        """Show message in all tabs."""
        for frame in [self.chart_frame, self.table_frame, self.stats_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
            tk.Label(
                frame, text=message,
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            ).pack(expand=True)
    
    def _analyze_trends(self):
        """Run trend analysis using biblium methods."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Analyzing trends...")
        
        analysis_type = self.analysis_type.get()
        
        # Get options before starting thread
        year_from = int(self.year_from_spin.get())
        year_to = int(self.year_to_spin.get())
        use_cut_year = self.use_cut_year_cb.get()
        cut_year = int(self.cut_year_spin.get()) if use_cut_year else None
        top_n = self.top_n.get()
        
        # New plot options
        bar_metric = self.bar_metric.get()
        line_metric = self.line_metric.get()
        show_bar_labels = self.show_bar_labels_cb.get()
        show_trend = self.show_trend_cb.get()
        trend_type = self.trend_type.get()
        moving_avg_window = self.moving_avg_window.get()
        show_avg = self.show_avg_cb.get()
        
        def do_analysis():
            try:
                result = {}
                
                if analysis_type == "Scientific Production":
                    # Use biblium's get_production() method
                    self.bib.get_production(cumulative=True)
                    prod_df = getattr(self.bib, 'production_df', None)
                    
                    if prod_df is not None and len(prod_df) > 0:
                        # Work with the raw DataFrame first, before any filtering
                        data = prod_df.copy()
                        
                        # Apply cut_year grouping (like Biblium's plot_timeseries)
                        if cut_year is not None:
                            before_df = data[data['Year'] < cut_year].copy()
                            after_df = data[data['Year'] >= cut_year].copy()
                            
                            if not before_df.empty:
                                # Sum documents, max cumulative for years before cut_year
                                combined = {'Year': f"<{cut_year}"}
                                for col in data.columns:
                                    if col == 'Year':
                                        continue
                                    if 'Cumulative' in col:
                                        combined[col] = before_df[col].max()
                                    elif pd.api.types.is_numeric_dtype(data[col]):
                                        combined[col] = before_df[col].sum()
                                    else:
                                        combined[col] = None
                                
                                before_row = pd.DataFrame([combined])
                                data = pd.concat([before_row, after_df], ignore_index=True)
                            else:
                                data = after_df
                        else:
                            # Filter by year range
                            data = data[(data['Year'] >= year_from) & (data['Year'] <= year_to)].copy()
                        
                        # Standardize column names
                        col_mapping = {
                            'Documents': 'Documents',
                            'Number of Documents': 'Documents',
                            'Total_Citations': 'Total Citations',
                            'Cumulative Documents': 'Cumulative Documents',
                            'Cumulative Citations': 'Cumulative Citations',
                        }
                        data = data.rename(columns={k: v for k, v in col_mapping.items() if k in data.columns})
                        
                        # Add average citations if we have both docs and citations
                        if 'Documents' in data.columns and 'Total Citations' in data.columns:
                            data['Avg Citations'] = (data['Total Citations'] / data['Documents'].replace(0, 1)).round(2)
                        
                        result["raw_df"] = data
                        result["title"] = "Annual Scientific Production"
                        result["bar_metric"] = bar_metric
                        result["line_metric"] = line_metric
                        result["show_bar_labels"] = show_bar_labels
                        result["show_trend"] = show_trend
                        result["trend_type"] = trend_type
                        result["moving_avg_window"] = moving_avg_window
                        result["show_avg"] = show_avg
                        result["from_biblium"] = True
                    else:
                        result = self._fallback_scientific_production(year_from, year_to, cut_year)
                
                elif analysis_type == "Citation Trends":
                    # Use production_df which has citation data
                    self.bib.get_production(cumulative=True)
                    prod_df = getattr(self.bib, 'production_df', None)
                    
                    if prod_df is not None and len(prod_df) > 0:
                        data = prod_df.copy()
                        
                        # Apply cut_year grouping
                        if cut_year is not None:
                            before_df = data[data['Year'] < cut_year].copy()
                            after_df = data[data['Year'] >= cut_year].copy()
                            
                            if not before_df.empty:
                                combined = {'Year': f"<{cut_year}"}
                                for col in data.columns:
                                    if col == 'Year':
                                        continue
                                    if 'Cumulative' in col:
                                        combined[col] = before_df[col].max()
                                    elif pd.api.types.is_numeric_dtype(data[col]):
                                        combined[col] = before_df[col].sum()
                                    else:
                                        combined[col] = None
                                
                                before_row = pd.DataFrame([combined])
                                data = pd.concat([before_row, after_df], ignore_index=True)
                            else:
                                data = after_df
                        else:
                            data = data[(data['Year'] >= year_from) & (data['Year'] <= year_to)].copy()
                        
                        # Standardize column names
                        col_mapping = {
                            'Documents': 'Documents',
                            'Number of Documents': 'Documents',
                            'Total_Citations': 'Total Citations',
                            'Cumulative Documents': 'Cumulative Documents',
                            'Cumulative Citations': 'Cumulative Citations',
                        }
                        data = data.rename(columns={k: v for k, v in col_mapping.items() if k in data.columns})
                        
                        if 'Documents' in data.columns and 'Total Citations' in data.columns:
                            data['Avg Citations'] = (data['Total Citations'] / data['Documents'].replace(0, 1)).round(2)
                        
                        result["raw_df"] = data
                        result["title"] = "Citation Trends"
                        result["bar_metric"] = "Total Citations" if bar_metric == "Documents" else bar_metric
                        result["line_metric"] = line_metric
                        result["show_bar_labels"] = show_bar_labels
                        result["show_trend"] = show_trend
                        result["trend_type"] = trend_type
                        result["moving_avg_window"] = moving_avg_window
                        result["show_avg"] = show_avg
                        result["from_biblium"] = True
                    else:
                        result = self._fallback_citation_trends(year_from, year_to, cut_year)
                    
                elif analysis_type == "Keyword Evolution":
                    # Use biblium's plot_items_production_over_time
                    try:
                        self.bib.plot_items_production_over_time(items='author keywords', top_n=top_n)
                        kw_df = getattr(self.bib, 'author_keywords_production_over_time_df', None)
                        
                        if kw_df is not None and len(kw_df) > 0:
                            # Pivot to get items as columns
                            data = kw_df.pivot_table(
                                index='Year', 
                                columns='Item', 
                                values='n_docs', 
                                fill_value=0
                            )
                            
                            # Filter by year range
                            data = data[(data.index >= year_from) & (data.index <= year_to)]
                            
                            result["data"] = data
                            result["title"] = f"Top {top_n} Keyword Evolution"
                            result["ylabel"] = "Frequency"
                            result["multi_line"] = True
                            result["from_biblium"] = True
                        else:
                            result = self._fallback_keyword_evolution(year_from, year_to, top_n)
                    except Exception as e:
                        print(f"Keyword evolution from biblium failed: {e}")
                        result = self._fallback_keyword_evolution(year_from, year_to, top_n)
                    
                elif analysis_type == "Author Productivity":
                    # Use biblium's plot_items_production_over_time
                    try:
                        self.bib.plot_items_production_over_time(items='authors', top_n=top_n)
                        auth_df = getattr(self.bib, 'authors_production_over_time_df', None)
                        
                        if auth_df is not None and len(auth_df) > 0:
                            data = auth_df.pivot_table(
                                index='Year', 
                                columns='Item', 
                                values='n_docs', 
                                fill_value=0
                            )
                            
                            data = data[(data.index >= year_from) & (data.index <= year_to)]
                            
                            result["data"] = data
                            result["title"] = f"Top {top_n} Author Productivity"
                            result["ylabel"] = "Publications"
                            result["multi_line"] = True
                            result["from_biblium"] = True
                        else:
                            result = self._fallback_author_productivity(year_from, year_to, top_n)
                    except Exception as e:
                        print(f"Author productivity from biblium failed: {e}")
                        result = self._fallback_author_productivity(year_from, year_to, top_n)
                    
                elif analysis_type == "Source Growth":
                    # Use biblium's plot_items_production_over_time
                    try:
                        self.bib.plot_items_production_over_time(items='sources', top_n=top_n)
                        src_df = getattr(self.bib, 'sources_production_over_time_df', None)
                        
                        if src_df is not None and len(src_df) > 0:
                            data = src_df.pivot_table(
                                index='Year', 
                                columns='Item', 
                                values='n_docs', 
                                fill_value=0
                            )
                            
                            data = data[(data.index >= year_from) & (data.index <= year_to)]
                            
                            result["data"] = data
                            result["title"] = f"Top {top_n} Source Growth"
                            result["ylabel"] = "Publications"
                            result["multi_line"] = True
                            result["from_biblium"] = True
                        else:
                            result = self._fallback_source_growth(year_from, year_to, top_n)
                    except Exception as e:
                        print(f"Source growth from biblium failed: {e}")
                        result = self._fallback_source_growth(year_from, year_to, top_n)
                    
                elif analysis_type == "Country Trends":
                    # Use biblium's plot_items_production_over_time
                    try:
                        self.bib.plot_items_production_over_time(items='all countries', top_n=top_n)
                        country_df = getattr(self.bib, 'all_countries_production_over_time_df', None)
                        
                        if country_df is not None and len(country_df) > 0:
                            data = country_df.pivot_table(
                                index='Year', 
                                columns='Item', 
                                values='n_docs', 
                                fill_value=0
                            )
                            
                            data = data[(data.index >= year_from) & (data.index <= year_to)]
                            
                            result["data"] = data
                            result["title"] = f"Top {top_n} Country Trends"
                            result["ylabel"] = "Publications"
                            result["multi_line"] = True
                            result["from_biblium"] = True
                        else:
                            result = self._fallback_country_trends(year_from, year_to, top_n)
                    except Exception as e:
                        print(f"Country trends from biblium failed: {e}")
                        result = self._fallback_country_trends(year_from, year_to, top_n)
                
                self.after(0, lambda: self._on_trends_success(result))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self._on_trends_error(str(e)))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _fallback_scientific_production(self, year_from, year_to, cut_year=None):
        """Fallback method for scientific production."""
        df = self.bib.df
        year_col = self.bib.mapping.get("Year", "Year")
        
        df_filtered = df[(df[year_col] >= year_from) & (df[year_col] <= year_to)].copy()
        yearly = df_filtered[year_col].value_counts().sort_index()
        
        data = yearly.to_frame("Documents")
        data = data.reset_index()
        data.columns = ['Year', 'Documents']
        
        return {
            "raw_df": data,
            "title": "Annual Scientific Production",
        }
    
    def _fallback_citation_trends(self, year_from, year_to, cut_year=None):
        """Fallback method for citation trends."""
        df = self.bib.df
        year_col = self.bib.mapping.get("Year", "Year")
        cit_col = self.bib.mapping.get("Cited_by", "Cited by")
        
        if cit_col not in df.columns:
            raise ValueError("Citation column not found")
        
        df_filtered = df[(df[year_col] >= year_from) & (df[year_col] <= year_to)]
        yearly = df_filtered.groupby(year_col)[cit_col].agg(['sum', 'mean', 'count'])
        yearly.columns = ['Total Citations', 'Avg Citations', 'Documents']
        
        return {
            "data": yearly,
            "title": "Citation Trends",
            "ylabel": "Citations",
        }
    
    def _fallback_keyword_evolution(self, year_from, year_to, top_n):
        """Fallback method for keyword evolution."""
        df = self.bib.df
        year_col = self.bib.mapping.get("Year", "Year")
        kw_col = self.bib.mapping.get("Author_Keywords", "Author Keywords")
        
        df_filtered = df[(df[year_col] >= year_from) & (df[year_col] <= year_to)]
        
        all_kw = df_filtered[kw_col].dropna().str.split(";").explode().str.strip()
        top_kw = all_kw.value_counts().head(top_n).index.tolist()
        
        evolution = {}
        for year in sorted(df_filtered[year_col].dropna().unique()):
            year_kw = df_filtered[df_filtered[year_col] == year][kw_col].dropna()
            year_kw = year_kw.str.split(";").explode().str.strip()
            counts = year_kw.value_counts()
            for kw in top_kw:
                if kw not in evolution:
                    evolution[kw] = {}
                evolution[kw][year] = counts.get(kw, 0)
        
        data = pd.DataFrame(evolution)
        
        return {
            "data": data,
            "title": f"Top {top_n} Keyword Evolution",
            "ylabel": "Frequency",
            "multi_line": True,
        }
    
    def _fallback_author_productivity(self, year_from, year_to, top_n):
        """Fallback method for author productivity."""
        df = self.bib.df
        year_col = self.bib.mapping.get("Year", "Year")
        author_col = self.bib.mapping.get("Authors", "Authors")
        
        df_filtered = df[(df[year_col] >= year_from) & (df[year_col] <= year_to)]
        
        all_authors = df_filtered[author_col].dropna().str.split(";").explode().str.strip()
        top_authors = all_authors.value_counts().head(top_n).index.tolist()
        
        evolution = {}
        for year in sorted(df_filtered[year_col].dropna().unique()):
            year_auth = df_filtered[df_filtered[year_col] == year][author_col].dropna()
            year_auth = year_auth.str.split(";").explode().str.strip()
            counts = year_auth.value_counts()
            for auth in top_authors:
                if auth not in evolution:
                    evolution[auth] = {}
                evolution[auth][year] = counts.get(auth, 0)
        
        data = pd.DataFrame(evolution)
        
        return {
            "data": data,
            "title": f"Top {top_n} Author Productivity",
            "ylabel": "Publications",
            "multi_line": True,
        }
    
    def _fallback_source_growth(self, year_from, year_to, top_n):
        """Fallback method for source growth."""
        df = self.bib.df
        year_col = self.bib.mapping.get("Year", "Year")
        source_col = self.bib.mapping.get("Source_title", "Source title")
        
        df_filtered = df[(df[year_col] >= year_from) & (df[year_col] <= year_to)]
        top_sources = df_filtered[source_col].value_counts().head(top_n).index.tolist()
        
        evolution = {}
        for year in sorted(df_filtered[year_col].dropna().unique()):
            year_data = df_filtered[df_filtered[year_col] == year]
            counts = year_data[source_col].value_counts()
            for src in top_sources:
                if src not in evolution:
                    evolution[src] = {}
                evolution[src][year] = counts.get(src, 0)
        
        data = pd.DataFrame(evolution)
        
        return {
            "data": data,
            "title": f"Top {top_n} Source Growth",
            "ylabel": "Publications",
            "multi_line": True,
        }
    
    def _fallback_country_trends(self, year_from, year_to, top_n):
        """Fallback method for country trends."""
        # Try to use count data from biblium
        if hasattr(self.bib, 'all_countries_counts_df') and self.bib.all_countries_counts_df is not None:
            counts_df = self.bib.all_countries_counts_df
            if len(counts_df) > 0:
                # Get top countries from counts
                name_col = counts_df.columns[0]
                top_countries = counts_df.head(top_n)[name_col].tolist()
                
                # Create simple data structure
                data = pd.DataFrame({country: [0] for country in top_countries})
                data.index = [self.bib.df[self.bib.mapping.get("Year", "Year")].max()]
                
                return {
                    "data": data,
                    "title": f"Top {top_n} Country Trends",
                    "ylabel": "Publications",
                    "multi_line": True,
                }
        
        raise ValueError("Country data not available")
    
    def _on_trends_success(self, result: Dict):
        """Display trend analysis results."""
        self._trend_data = result
        
        # Get data from either raw_df or data key
        data = result.get("raw_df", result.get("data"))
        
        # Calculate statistics if not already present
        if data is not None and "stats" not in result:
            if isinstance(data, pd.DataFrame) and len(data) > 0:
                stats = {
                    "Years": len(data),
                }
                
                if not result.get("multi_line"):
                    # Find the main document column
                    doc_col = None
                    for col in ["Documents", "Number of Documents"]:
                        if col in data.columns:
                            doc_col = col
                            break
                    
                    if doc_col:
                        stats["Total Documents"] = int(data[doc_col].sum())
                        stats["Mean/Year"] = round(data[doc_col].mean(), 1)
                        stats["Max"] = int(data[doc_col].max())
                        
                        # Growth rate
                        if len(data) > 1:
                            first_val = data[doc_col].iloc[0]
                            last_val = data[doc_col].iloc[-1]
                            if first_val > 0:
                                growth = ((last_val - first_val) / first_val) * 100
                                stats["Growth (%)"] = round(growth, 1)
                    
                    # Citation stats if available
                    cite_col = None
                    for col in ["Total Citations", "Total_Citations"]:
                        if col in data.columns:
                            cite_col = col
                            break
                    
                    if cite_col:
                        stats["Total Citations"] = int(data[cite_col].sum())
                
                result["stats"] = stats
        
        # Show chart
        self._show_chart(result)
        
        # Show table
        self._show_table(result)
        
        # Show statistics
        self._show_statistics(result)
    
    def _on_trends_error(self, error: str):
        """Handle trend analysis error."""
        self._show_trends_message(f"Error: {error}")
    
    def _on_analysis_success(self, result: Dict):
        """Display analysis results (deprecated, use _on_trends_success)."""
        self._on_trends_success(result)
    
    def _on_analysis_error(self, error: str):
        """Handle analysis error (deprecated, use _on_trends_error)."""
        self._on_trends_error(error)
    
    def _show_chart(self, result: Dict):
        """Display trend chart following Biblium's visualization approach."""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        if not HAS_MATPLOTLIB:
            return
        
        # Check for new raw_df format or old data format
        if "raw_df" in result:
            data = result["raw_df"]
        elif "data" in result:
            data = result["data"]
        else:
            return
        
        if data is None or len(data) == 0:
            return
        
        # Add AI button header
        ai_header = tk.Frame(self.chart_frame, bg=self.theme["bg_card"])
        ai_header.pack(fill=tk.X, padx=4, pady=(4, 2))
        
        ai_btn = tk.Button(
            ai_header,
            text="🤖 AI Describe",
            font=FONTS.get_font("body"),
            bg=self.theme["accent_primary"],
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            padx=12,
            pady=2,
            command=lambda: self._ai_describe_chart(result, data),
        )
        ai_btn.pack(side=tk.RIGHT, padx=4)
        
        tk.Label(
            ai_header,
            text="💡 Right-click chart for save options",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(side=tk.LEFT, padx=4)
        
        plot = PlotFrame(self.chart_frame, theme=self.theme_name, figsize=(12, 6), show_toolbar=False, show_ai_button=True)
        plot.pack(fill=tk.BOTH, expand=True)
        
        fig, ax1 = plot.get_figure()
        ax2 = None
        
        # Get plot options
        bar_metric = result.get("bar_metric", self.bar_metric.get() if hasattr(self, 'bar_metric') else "Documents")
        line_metric = result.get("line_metric", self.line_metric.get() if hasattr(self, 'line_metric') else "Cumulative Citations")
        show_bar_labels = result.get("show_bar_labels", self.show_bar_labels_cb.get() if hasattr(self, 'show_bar_labels_cb') else False)
        show_trend = result.get("show_trend", self.show_trend_cb.get() if hasattr(self, 'show_trend_cb') else False)
        trend_type = result.get("trend_type", self.trend_type.get() if hasattr(self, 'trend_type') else "Linear")
        moving_avg_window = result.get("moving_avg_window", self.moving_avg_window.get() if hasattr(self, 'moving_avg_window') else 3)
        show_avg = result.get("show_avg", self.show_avg_cb.get() if hasattr(self, 'show_avg_cb') else False)
        
        # Handle multi-line charts (keyword evolution, author productivity, etc.)
        if result.get("multi_line"):
            for col in data.columns:
                ax1.plot(data.index, data[col], marker='o', markersize=4, linewidth=2, label=col[:20])
            ax1.legend(loc='upper left', fontsize=8)
            ax1.set_ylabel(result.get("ylabel", "Frequency"))
        else:
            # Single-series chart (Scientific Production, Citation Trends)
            # X-axis from Year column or index
            if 'Year' in data.columns:
                x_vals = data['Year'].astype(str).tolist()
            else:
                x_vals = [str(x) for x in data.index]
            x_pos = range(len(x_vals))
            
            # Determine bar and line columns
            bar_col = None
            line_col = None
            
            if bar_metric != "None":
                # Map metric name to column
                bar_col_map = {
                    "Documents": ["Documents", "Number of Documents"],
                    "Total Citations": ["Total Citations", "Total_Citations"],
                }
                for possible_col in bar_col_map.get(bar_metric, [bar_metric]):
                    if possible_col in data.columns:
                        bar_col = possible_col
                        break
            
            if line_metric != "None":
                # Map metric name to column
                line_col_map = {
                    "Cumulative Citations": ["Cumulative Citations"],
                    "Cumulative Documents": ["Cumulative Documents"],
                    "Avg Citations": ["Avg Citations"],
                }
                for possible_col in line_col_map.get(line_metric, [line_metric]):
                    if possible_col in data.columns:
                        line_col = possible_col
                        break
            
            # Plot bars on primary axis
            if bar_col and bar_col in data.columns:
                bar_color = "lightblue"
                bars = ax1.bar(x_pos, data[bar_col], color=bar_color, label=bar_col)
                ax1.set_ylabel(bar_col)
                ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                # Show bar labels if requested
                if show_bar_labels:
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(
                            bar.get_x() + bar.get_width() / 2,
                            height,
                            f"{int(height)}" if float(height).is_integer() else f"{height:.1f}",
                            ha="center", va="bottom", fontsize=8
                        )
                
                # Add trendline on primary axis
                if show_trend and len(x_pos) > 2:
                    y_vals = data[bar_col].values
                    x_numeric = np.array(list(x_pos))
                    
                    try:
                        if trend_type == "Linear":
                            z = np.polyfit(x_numeric, y_vals, 1)
                            p = np.poly1d(z)
                            ax1.plot(x_numeric, p(x_numeric), "r--", linewidth=2, label="Linear Trend")
                        elif trend_type == "Exponential":
                            # Fit exponential: y = a * exp(b * x)
                            y_log = np.log(y_vals + 1)  # +1 to avoid log(0)
                            z = np.polyfit(x_numeric, y_log, 1)
                            p = np.exp(z[1]) * np.exp(z[0] * x_numeric)
                            ax1.plot(x_numeric, p, "r--", linewidth=2, label="Exponential Trend")
                        elif trend_type.startswith("Polynomial"):
                            degree = int(trend_type.split("(")[1].replace(")", ""))
                            z = np.polyfit(x_numeric, y_vals, degree)
                            p = np.poly1d(z)
                            ax1.plot(x_numeric, p(x_numeric), "r--", linewidth=2, label=f"Poly({degree}) Trend")
                        elif trend_type == "Moving Average":
                            window = min(moving_avg_window, len(y_vals))
                            ma = pd.Series(y_vals).rolling(window=window, min_periods=1).mean()
                            ax1.plot(x_numeric, ma, "r-", linewidth=2, label=f"MA({window})")
                    except Exception as e:
                        print(f"Trendline error: {e}")
                
                # Show average line
                if show_avg:
                    avg = data[bar_col].mean()
                    ax1.axhline(y=avg, color='green', linestyle=':', linewidth=2, label=f"Avg: {avg:.1f}")
            
            # Plot line on secondary axis
            if line_col and line_col in data.columns:
                ax2 = ax1.twinx()
                line_color = "black"
                ax2.plot(x_pos, data[line_col], color=line_color, marker='o', 
                        linewidth=2, markersize=4, label=line_col)
                ax2.set_ylabel(line_col)
                ax2.ticklabel_format(style='plain', axis='y')
            
            # Set x-axis
            ax1.set_xticks(list(x_pos))
            ax1.set_xticklabels(x_vals, rotation=45, ha='right')
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            if ax2:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            elif lines1:
                ax1.legend(loc='upper left')
        
        ax1.set_xlabel("Year")
        ax1.set_title(result.get("title", "Trend Analysis"))
        ax1.grid(False)
        if ax2:
            ax2.grid(False)
        
        fig.tight_layout()
        plot.refresh()
    
    def _show_table(self, result: Dict):
        """Display data table."""
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        # Get data from either raw_df or data key
        data = result.get("raw_df", result.get("data"))
        
        if data is None or len(data) == 0:
            return
        
        # Make a copy for display
        display_data = data.copy()
        
        # Round float columns for cleaner display
        for col in display_data.columns:
            if display_data[col].dtype in ['float64', 'float32']:
                display_data[col] = display_data[col].round(2)
        
        # Reset index if Year is in the index
        if 'Year' not in display_data.columns and display_data.index.name == 'Year':
            display_data = display_data.reset_index()
        
        table = DataTable(self.table_frame, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(display_data)
    
    def _show_statistics(self, result: Dict):
        """Display statistics."""
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        if "stats" not in result:
            tk.Label(
                self.stats_frame, text="Statistics not available",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        stats = result["stats"]
        
        # Stats cards
        grid = CardGrid(self.stats_frame, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, padx=8, pady=8)
        
        for key, value in stats.items():
            grid.add_card(StatsCard(grid, key, str(value), "📊", self.theme_name))
    
    def _ai_describe_chart(self, result: Dict, data):
        """Generate AI description of the chart."""
        from biblium.gui.widgets.tables import DataTable
        settings = DataTable.get_llm_settings()
        
        if not settings.get("api_key"):
            from tkinter import messagebox
            messagebox.showinfo("Configure AI", 
                "Please configure AI settings first:\n\n"
                "1. Expand '🤖 AI Analysis Settings' in the options panel\n"
                "2. Enter your API key\n"
                "3. Click this button again")
            return
        
        # Extract chart info
        plot_info = {
            "type": "bar chart with trend line",
            "title": result.get("title", "Trend Analysis"),
            "x_label": "Year",
            "y_label": result.get("bar_metric", "Documents"),
            "data_summary": "",
            "context": "",
        }
        
        try:
            if data is not None and len(data) > 0:
                plot_info["data_summary"] = f"{len(data)} time periods. "
                if hasattr(data, 'columns'):
                    for col in data.columns[:3]:
                        if data[col].dtype in ['int64', 'float64']:
                            plot_info["data_summary"] += f"{col}: min={data[col].min():.0f}, max={data[col].max():.0f}, total={data[col].sum():.0f}. "
        except:
            pass
        
        # Show loading
        if hasattr(self, '_ai_result_frame') and self._ai_result_frame:
            try: self._ai_result_frame.destroy()
            except: pass
        
        self._ai_loading = tk.Label(
            self.chart_frame,
            text="⏳ Generating AI description...",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        )
        self._ai_loading.pack(side=tk.BOTTOM, pady=4)
        
        import threading
        def do_generate():
            try:
                from biblium.llm_utils import llm_describe_plot
                result_text = llm_describe_plot(
                    plot_type=plot_info["type"],
                    title=plot_info["title"],
                    data_summary=plot_info["data_summary"],
                    x_axis=plot_info["x_label"],
                    y_axis=plot_info["y_label"],
                    context=plot_info["context"],
                    provider=settings["provider"],
                    model=settings["model"],
                    api_key=settings["api_key"],
                    custom_prompt=settings.get("custom_prompt", ""),
                )
                self.after(0, lambda r=result_text: self._show_ai_chart_result(r))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.after(0, lambda msg=error_msg: self._show_ai_chart_result(msg))
        
        thread = threading.Thread(target=do_generate, daemon=True)
        thread.start()
    
    def _show_ai_chart_result(self, text: str):
        """Show AI chart description result."""
        if hasattr(self, '_ai_loading') and self._ai_loading:
            try: self._ai_loading.destroy()
            except: pass
        if hasattr(self, '_ai_result_frame') and self._ai_result_frame:
            try: self._ai_result_frame.destroy()
            except: pass
        
        self._ai_result_frame = tk.Frame(self.chart_frame, bg=self.theme["bg_card"])
        self._ai_result_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0))
        
        header = tk.Frame(self._ai_result_frame, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        tk.Label(header, text="🤖 AI Description", font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT, padx=4)
        
        def copy_text():
            try:
                self.clipboard_clear()
                self.clipboard_append(text)
                copy_btn.config(text="✓ Copied")
                self.after(1500, lambda: copy_btn.config(text="📋 Copy"))
            except: pass
        
        tk.Button(header, text="✕", font=("Segoe UI", 8), bg=self.theme["bg_secondary"],
                  fg=self.theme["text_primary"], relief=tk.FLAT,
                  command=lambda: self._ai_result_frame.destroy(), cursor="hand2", width=2).pack(side=tk.RIGHT, padx=2)
        copy_btn = tk.Button(header, text="📋 Copy", font=("Segoe UI", 8), bg=self.theme["bg_secondary"],
                             fg=self.theme["text_primary"], relief=tk.FLAT, command=copy_text, cursor="hand2")
        copy_btn.pack(side=tk.RIGHT, padx=2)
        
        text_widget = tk.Text(self._ai_result_frame, wrap=tk.WORD, font=FONTS.get_font("body"),
                              bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
                              relief=tk.FLAT, height=4, padx=8, pady=4)
        text_widget.pack(fill=tk.X, padx=4, pady=(2, 4))
        text_widget.insert("1.0", text)
        def on_key(e):
            if e.state & 0x4 and e.keysym.lower() in ('c', 'a'): return
            return "break"
        text_widget.bind("<Key>", on_key)
        text_widget.bind("<Button-1>", lambda e: text_widget.focus_set())
    
    def _on_analysis_error(self, error: str):
        """Handle analysis error."""
        for frame in [self.chart_frame, self.table_frame, self.stats_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
            
            tk.Label(
                frame, text=f"Error: {error}",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["danger"],
            ).pack(expand=True)
    
    def _export_results(self):
        """Export trend data."""
        if self._trend_data is None or "data" not in self._trend_data:
            messagebox.showwarning("No Data", "No trend data to export.")
            return
        
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")],
            title="Export Trend Data",
        )
        
        if filename:
            try:
                data = self._trend_data["data"]
                if filename.endswith(".xlsx"):
                    data.to_excel(filename)
                else:
                    data.to_csv(filename)
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
PUBLICATION TRENDS
══════════════════

Analyze temporal patterns in publication data.

TREND METRICS
─────────────
• Annual Publications: Count per year
• Cumulative: Running total
• Growth Rate: Year-over-year change
• CAGR: Compound annual growth rate

VISUALIZATION OPTIONS
─────────────────────
• Line chart: Trends over time
• Bar chart: Annual counts
• Area chart: Cumulative growth
• Combined: Multiple series

GROWTH ANALYSIS
───────────────
• Absolute growth: Papers added
• Relative growth: Percentage change
• Moving average: Smoothed trend
• Growth acceleration: Rate change

TREND PATTERNS
──────────────
• Linear growth: Constant additions
• Exponential: Accelerating growth
• Logistic: Growth then saturation
• Decline: Decreasing output

DATA FILTERS
────────────
• Year range selection
• Exclude incomplete years
• Entity-specific trends

COMMON FINDINGS
───────────────
• Most fields show growth
• COVID-19 effects (2020-2022)
• Database coverage changes
• Seasonal patterns (rare)

EXPORT
──────
• Time series data
• Growth statistics
• Trend visualization
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

    def _on_dataset_loaded(self, data):
        """Handle dataset loaded event."""
        super()._on_dataset_loaded(data)
        self._set_year_defaults()
