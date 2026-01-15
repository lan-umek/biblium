# -*- coding: utf-8 -*-
"""
Basic plotting functions - distributions, bar charts, scatter plots.

This module contains methods for:
- Distribution plots (box, violin)
- Top items bar charts
- Scatter plots
- Average citations plots
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class BasicPlotsMixin:
    """Mixin class providing basic plotting methods."""
    
    def plot_average_citations_per_year(
        self,
        filename_base: str = "average citations per document",
        **kwargs: Any,
    ) -> None:
        """
        Plot average citations per document by publication year.

        Parameters
        ----------
        filename_base : str or None
            Base filename for saving. If None, plot is not saved.
        **kwargs :
            Additional arguments for plotbib.plot_average_citations_per_year.
        """
        from biblium import plotbib, utilsbib
        
        grouped = utilsbib.compute_average_citations_per_year(self.df)
        if filename_base is not None and self.res_folder is not None:
            filename_base = os.path.join(self.res_folder, "plots", filename_base)
            plotbib.plot_average_citations_per_year(grouped, **kwargs)

    def dist_plots(
        self,
        grouping_vars: List[str] = None,
        numeric_vars: List[str] = None,
        list_grouping_vars: Optional[List[str]] = None,
        max_groups: int = 5,
        order_by_size: bool = True,
        plot_type: str = "box",
        **kwargs: Any,
    ) -> None:
        """
        Generate box or violin plots for combinations of numeric and grouping variables.
    
        Parameters
        ----------
        grouping_vars : list of str
            Categorical variables for grouping (single value per row).
        numeric_vars : list of str
            Numerical variables to plot.
        list_grouping_vars : list of str, optional
            Variables with delimited lists.
        max_groups : int
            Maximum groups per plot.
        order_by_size : bool
            Order groups by size.
        plot_type : {"box", "violin"}
            Type of distribution plot.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import plotbib
        
        if grouping_vars is None:
            grouping_vars = ["Source title", "Document Type"]
        if numeric_vars is None:
            numeric_vars = ["Year", "Cited by"]
        
        plot_func = {
            "box": plotbib.plot_boxplot,
            "violin": plotbib.plot_violinplot,
        }.get(plot_type)
    
        if plot_func is None:
            raise ValueError("plot_type must be 'box' or 'violin'")
    
        if list_grouping_vars is None:
            list_grouping_vars = []
    
        # Keep only existing columns
        grouping_vars = [c for c in grouping_vars if c in self.df.columns]
        numeric_vars = [c for c in numeric_vars if c in self.df.columns]
    
        def _explode_list_column(df, col, sep):
            """Explode a delimited string column into multiple rows."""
            tmp = df.dropna(subset=[col]).copy()
            tmp[col] = tmp[col].astype(str).str.split(sep)
            tmp = tmp.explode(col)
            tmp[col] = tmp[col].str.strip()
            tmp = tmp[tmp[col] != ""]
            return tmp
    
        # Plots for standard grouping vars
        for group_var in grouping_vars:
            for numeric_var in numeric_vars:
                if self.res_folder is not None:
                    filename_base = os.path.join(
                        self.res_folder, "plots",
                        f"{numeric_var} by {group_var}_{plot_type}",
                    )
                else:
                    filename_base = None
                plot_func(
                    self.df,
                    value_column=numeric_var,
                    group_by=group_var,
                    max_groups=max_groups,
                    order_by_size=order_by_size,
                    filename_base=filename_base,
                    dpi=getattr(self, "dpi", 600),
                    **kwargs,
                )
    
        # Plots for list-like grouping vars
        resolved_list_grouping = []
        for gv in list_grouping_vars:
            actual_col = None
            display_name = gv
    
            if gv in ("Author Keywords", "author keywords"):
                if "Processed Author Keywords" in self.df.columns:
                    actual_col = "Processed Author Keywords"
                elif "Author Keywords" in self.df.columns:
                    actual_col = "Author Keywords"
                display_name = "Author Keywords"
            elif gv in ("Index Keywords", "index keywords"):
                if "Processed Index Keywords" in self.df.columns:
                    actual_col = "Processed Index Keywords"
                elif "Index Keywords" in self.df.columns:
                    actual_col = "Index Keywords"
                display_name = "Index Keywords"
            else:
                if gv in self.df.columns:
                    actual_col = gv
    
            if actual_col is not None:
                resolved_list_grouping.append((display_name, actual_col))
    
        for display_name, actual_col in resolved_list_grouping:
            df_exploded = _explode_list_column(
                self.df, actual_col, getattr(self, "default_separator", "; ")
            )
            if display_name != actual_col:
                df_exploded = df_exploded.rename(columns={actual_col: display_name})
    
            for numeric_var in numeric_vars:
                if self.res_folder is not None:
                    filename_base = os.path.join(
                        self.res_folder, "plots",
                        f"{numeric_var} by {display_name}_{plot_type}",
                    )
                else:
                    filename_base = None
                plot_func(
                    df_exploded,
                    value_column=numeric_var,
                    group_by=display_name,
                    max_groups=max_groups,
                    order_by_size=order_by_size,
                    filename_base=filename_base,
                    dpi=getattr(self, "dpi", 600),
                    **kwargs,
                )

    def plot_top_items(
        self,
        items: str = "sources",
        x: str = "Number of documents",
        top_n: int = 10,
        ax=None,
        figsize: tuple = (10, 6),
        title: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot top N items as a horizontal bar chart.

        Parameters
        ----------
        items : str
            Type of items: "sources", "authors", "keywords", etc.
        x : str
            Column name for the metric.
        top_n : int
            Number of top items to show.
        ax : matplotlib Axes, optional
            Axes to plot on.
        figsize : tuple
            Figure size.
        title : str, optional
            Plot title.
        filename : str, optional
            Filename for saving.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import plotbib
        import matplotlib.pyplot as plt
        
        # Get the appropriate counts DataFrame
        items_map = {
            "sources": ("sources_counts_df", "Source"),
            "authors": ("authors_counts_df", "Author(s) ID"),
            "author_keywords": ("author_keywords_counts_df", "Keyword"),
            "index_keywords": ("index_keywords_counts_df", "Keyword"),
            "keywords": ("keywords_counts_df", "Keyword"),
            "affiliations": ("affiliations_counts_df", "Affiliation"),
            "countries": ("ca_country_counts_df", "Country"),
        }
        
        if items not in items_map:
            raise ValueError(f"Unknown items type: {items}")
        
        attr_name, label_col = items_map[items]
        
        # Ensure counts exist
        if not hasattr(self, attr_name):
            count_method = f"count_{items}"
            if hasattr(self, count_method):
                getattr(self, count_method)()
        
        df = getattr(self, attr_name, None)
        if df is None or df.empty:
            return
        
        # Plot
        plot_df = df.head(top_n)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        plotbib.plot_barh(
            plot_df, 
            x=x, 
            y=label_col,
            ax=ax,
            title=title or f"Top {top_n} {items.replace('_', ' ').title()}",
            **kwargs
        )
        
        if filename and self.res_folder:
            self._save_plot(filename)

    def scatter_plot_top_sources(
        self,
        x: str = "Number of documents",
        y: str = "Total citations",
        top_n: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Scatter plot for top sources."""
        self._scatter_plot_top_items("sources", x, y, top_n, **kwargs)
    
    def scatter_plot_top_authors(
        self,
        x: str = "Number of documents",
        y: str = "Total citations",
        top_n: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Scatter plot for top authors."""
        self._scatter_plot_top_items("authors", x, y, top_n, **kwargs)
    
    def scatter_plot_top_countries(
        self,
        x: str = "Number of documents",
        y: str = "Total citations",
        top_n: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Scatter plot for top countries."""
        self._scatter_plot_top_items("countries", x, y, top_n, **kwargs)
    
    def _scatter_plot_top_items(
        self,
        items: str,
        x: str,
        y: str,
        top_n: Optional[int],
        **kwargs: Any,
    ) -> None:
        """Internal scatter plot helper."""
        from biblium import plotbib
        
        items_map = {
            "sources": "sources_counts_df",
            "authors": "authors_counts_df",
            "countries": "ca_country_counts_df",
        }
        
        attr_name = items_map.get(items)
        if not attr_name:
            return
        
        df = getattr(self, attr_name, None)
        if df is None:
            return
        
        if top_n:
            df = df.head(top_n)
        
        plotbib.plot_scatter(df, x=x, y=y, **kwargs)
