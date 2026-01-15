# -*- coding: utf-8 -*-
"""
Time series plotting functions - production over time, trends.

This module contains methods for:
- Scientific production plots
- Reference spectrograms
- Item production over time
- Trend topics
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class TimeSeriesPlotsMixin:
    """Mixin class providing time series plotting methods."""
    
    def plot_scientific_production(
        self,
        filename: str = "scientific production",
        **kwargs: Any,
    ) -> None:
        """
        Plot annual scientific production.

        If production_df is not available, computes it first.

        Parameters
        ----------
        filename : str
            Filename for saving the plot.
        **kwargs :
            Additional arguments for plotbib.plot_timeseries.
        """
        from biblium import plotbib
        
        if not hasattr(self, "production_df") or self.production_df is None:
            self.get_production()
        
        if self.res_folder is not None:
            plotbib.plot_timeseries(
                self.production_df,
                x="Year",
                y="Number of documents",
                filename=os.path.join(self.res_folder, "plots", filename),
                dpi=getattr(self, "dpi", 600),
                **kwargs,
            )

    def plot_reference_spectrogram(
        self,
        save_path: str = "spectrogram",
        **kwargs: Any,
    ) -> None:
        """
        Plot reference spectrogram showing temporal distribution of references.

        Parameters
        ----------
        save_path : str
            Filename for saving.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import plotbib, utilsbib
        
        if self.db == "scopus":
            spec_df = utilsbib.compute_reference_spectrogram_scopus(self.df)
        else:
            spec_df = utilsbib.compute_reference_spectrogram_openalex(self.df)
        
        if spec_df is not None and not spec_df.empty:
            if self.res_folder is not None:
                plotbib.plot_heatmap(
                    spec_df,
                    filename=os.path.join(self.res_folder, "plots", save_path),
                    **kwargs,
                )

    def plot_items_production_over_time(
        self,
        items: str = "keywords",
        top_n: int = 10,
        normalize: bool = False,
        cumulative: bool = False,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot production of top items over time.

        Parameters
        ----------
        items : str
            Type of items: "keywords", "sources", "authors", etc.
        top_n : int
            Number of top items to show.
        normalize : bool
            Whether to normalize values.
        cumulative : bool
            Whether to show cumulative values.
        filename : str, optional
            Filename for saving.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import plotbib, utilsbib
        import matplotlib.pyplot as plt
        
        # Get the appropriate data
        items_map = {
            "keywords": ("author_keywords_counts_df", "Keyword", "Author Keywords"),
            "author_keywords": ("author_keywords_counts_df", "Keyword", "Author Keywords"),
            "sources": ("sources_counts_df", "Source", "Source title"),
            "authors": ("authors_counts_df", "Author(s) ID", "Authors"),
        }
        
        if items not in items_map:
            raise ValueError(f"Unknown items type: {items}")
        
        attr_name, label_col, source_col = items_map[items]
        
        # Ensure counts exist
        if not hasattr(self, attr_name):
            count_method = f"count_{items}"
            if hasattr(self, count_method):
                getattr(self, count_method)()
        
        counts_df = getattr(self, attr_name, None)
        if counts_df is None or counts_df.empty:
            return
        
        # Get top items
        top_items = counts_df[label_col].head(top_n).tolist()
        
        # Compute time series for each item
        time_data = utilsbib.compute_item_time_stats(
            self.df,
            column=source_col,
            items=top_items,
            sep=getattr(self, "default_separator", "; "),
        )
        
        if time_data is not None and not time_data.empty:
            plotbib.plot_items_over_time(
                time_data,
                normalize=normalize,
                cumulative=cumulative,
                **kwargs,
            )
            
            if filename and self.res_folder:
                self._save_plot(filename)

    def plot_trend_topics(
        self,
        items: str = "keywords",
        top_n: int = 10,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        method: str = "loess",
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot trend analysis for topics over time.

        Parameters
        ----------
        items : str
            Type of items to analyze.
        top_n : int
            Number of top items.
        min_year, max_year : int, optional
            Year range filter.
        method : str
            Trend smoothing method.
        filename : str, optional
            Filename for saving.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import plotbib, utilsbib
        
        items_map = {
            "keywords": ("author_keywords_counts_df", "Keyword", "Author Keywords"),
            "author_keywords": ("author_keywords_counts_df", "Keyword", "Author Keywords"),
            "index_keywords": ("index_keywords_counts_df", "Keyword", "Index Keywords"),
        }
        
        if items not in items_map:
            raise ValueError(f"Unknown items type: {items}")
        
        attr_name, label_col, source_col = items_map[items]
        
        # Ensure counts exist
        if not hasattr(self, attr_name):
            count_method = f"count_{items}"
            if hasattr(self, count_method):
                getattr(self, count_method)()
        
        counts_df = getattr(self, attr_name, None)
        if counts_df is None or counts_df.empty:
            return
        
        # Filter by year if specified
        df = self.df.copy()
        if min_year:
            df = df[df["Year"] >= min_year]
        if max_year:
            df = df[df["Year"] <= max_year]
        
        # Get top items
        top_items = counts_df[label_col].head(top_n).tolist()
        
        # Compute trends
        trend_data = utilsbib.compute_item_time_stats(
            df,
            column=source_col,
            items=top_items,
            sep=getattr(self, "default_separator", "; "),
        )
        
        if trend_data is not None and not trend_data.empty:
            plotbib.plot_trend_topics(
                trend_data,
                method=method,
                **kwargs,
            )
            
            if filename and self.res_folder:
                self._save_plot(filename)
