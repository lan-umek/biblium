# -*- coding: utf-8 -*-
"""
Plot Interface for BiblioStats Integration.

This module provides a PlotInterface class that connects BiblioStats/BiblioAnalysis
with the unified plotting module, allowing seamless access to all plot types
with full customization.

Usage
-----
    ba = BiblioAnalysis("data.csv", db="scopus")
    
    # Access through .plot attribute
    ba.plot.sources_bar()
    ba.plot.keywords_wordcloud()
    ba.plot.production_line()
    
    # With customization
    from biblium.plotting import PlotConfig
    ba.plot.sources_bar(config=PlotConfig.presentation())
    
    # Change backend
    ba.plot.set_backend("bokeh")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import pandas as pd
import numpy as np

# Import from submodules to avoid circular import
from biblium.plotting.config import PlotConfig

if TYPE_CHECKING:
    from biblium.bibstats import BiblioStats


def _get_plot():
    """Lazy import of Plot class to avoid circular imports."""
    from biblium.plotting import Plot
    return Plot


def _set_backend(name: str):
    """Lazy import of set_backend to avoid circular imports."""
    from biblium.plotting import set_backend
    set_backend(name)


def _get_backend_name():
    """Lazy import of get_backend_name."""
    from biblium.plotting import get_backend_name
    return get_backend_name()


class PlotInterface:
    """
    Unified plotting interface for BiblioStats.
    
    Provides convenient methods that prepare data from BiblioStats
    and pass it to the unified plotting module.
    
    Parameters
    ----------
    biblio : BiblioStats
        The BiblioStats instance to plot from.
    default_config : PlotConfig, optional
        Default configuration for all plots.
    
    Examples
    --------
    >>> ba = BiblioAnalysis("data.csv", db="scopus")
    >>> ba.plot.sources_bar(top_n=15)
    >>> ba.plot.keywords_wordcloud()
    >>> ba.plot.production_line(cumulative=True)
    """
    
    def __init__(
        self,
        biblio: "BiblioStats",
        default_config: Optional[PlotConfig] = None,
    ):
        self._biblio = biblio
        self._config = default_config or PlotConfig()
    
    def _find_column(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Find first matching column from candidates (case-insensitive)."""
        df_cols_lower = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
            if candidate.lower() in df_cols_lower:
                return df_cols_lower[candidate.lower()]
        return None
    
    def _Plot(self):
        """Lazy accessor for Plot class."""
        return _get_plot()
    
    def _get_save_path(self, filename: str) -> Optional[str]:
        """Get full save path if res_folder is set."""
        import os
        if self._biblio.res_folder:
            figures_dir = os.path.join(self._biblio.res_folder, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            return os.path.join(figures_dir, filename)
        return None
    
    def _merge_config(self, config: Optional[PlotConfig]) -> PlotConfig:
        """Merge provided config with default."""
        if config is None:
            return self._config
        return self._config.merge(config)
    
    def set_backend(self, name: str) -> None:
        """
        Set the plotting backend.
        
        Parameters
        ----------
        name : str
            Backend name: "matplotlib" or "bokeh"
        """
        _set_backend(name)
    
    def get_backend(self) -> str:
        """Get current backend name."""
        return _get_backend_name()
    
    def set_config(self, config: PlotConfig) -> None:
        """Set default configuration for all plots."""
        self._config = config
    
    # =========================================================================
    # SOURCES/JOURNALS
    # =========================================================================
    
    def sources_bar(
        self,
        top_n: int = 20,
        metric: str = "Number of documents",
        horizontal: bool = True,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "sources_bar",
        **kwargs,
    ) -> Any:
        """
        Plot top sources/journals as bar chart.
        
        Parameters
        ----------
        top_n : int
            Number of top sources to show.
        metric : str
            Metric to plot.
        horizontal : bool
            Use horizontal bars.
        config : PlotConfig, optional
            Plot configuration.
        filename : str, optional
            Filename for saving.
        """
        # Ensure source counts exist
        if not hasattr(self._biblio, "sources_counts_df") or self._biblio.sources_counts_df is None:
            self._biblio.count_sources()
        
        df = self._biblio.sources_counts_df.head(top_n).copy()
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"Top {top_n} Sources")
        if config.xlabel is None:
            config = config.update(xlabel=metric)
        
        if horizontal:
            fig = self._Plot().barh(df, x=metric, y="Source", config=config, **kwargs)
        else:
            fig = self._Plot().bar(df, x="Source", y=metric, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def sources_treemap(
        self,
        top_n: int = 20,
        metric: str = "Number of documents",
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "sources_treemap",
        **kwargs,
    ) -> Any:
        """Plot sources as treemap."""
        if not hasattr(self._biblio, "sources_counts_df") or self._biblio.sources_counts_df is None:
            self._biblio.count_sources()
        
        df = self._biblio.sources_counts_df.head(top_n).copy()
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"Top {top_n} Sources")
        
        fig = self._Plot().treemap(df, values=metric, labels="Source", config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    # =========================================================================
    # AUTHORS
    # =========================================================================
    
    def authors_bar(
        self,
        top_n: int = 20,
        metric: str = "Number of documents",
        horizontal: bool = True,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "authors_bar",
        **kwargs,
    ) -> Any:
        """Plot top authors as bar chart."""
        if not hasattr(self._biblio, "authors_counts_df") or self._biblio.authors_counts_df is None:
            self._biblio.count_authors()
        
        df = self._biblio.authors_counts_df.head(top_n).copy()
        
        # Find the author column (can vary by database)
        author_col = None
        for col in ["Author", "Author(s) ID", "Authors"]:
            if col in df.columns:
                author_col = col
                break
        if author_col is None:
            author_col = df.columns[0]  # Use first column as fallback
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"Top {top_n} Authors")
        
        if horizontal:
            fig = self._Plot().barh(df, x=metric, y=author_col, config=config, **kwargs)
        else:
            fig = self._Plot().bar(df, x=author_col, y=metric, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def authors_lollipop(
        self,
        top_n: int = 20,
        metric: str = "Number of documents",
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "authors_lollipop",
        **kwargs,
    ) -> Any:
        """Plot top authors as lollipop chart."""
        if not hasattr(self._biblio, "authors_counts_df") or self._biblio.authors_counts_df is None:
            self._biblio.count_authors()
        
        df = self._biblio.authors_counts_df.head(top_n).copy()
        
        # Find the author column
        author_col = None
        for col in ["Author", "Author(s) ID", "Authors"]:
            if col in df.columns:
                author_col = col
                break
        if author_col is None:
            author_col = df.columns[0]
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"Top {top_n} Authors")
        
        fig = self._Plot().lollipop(df, x=metric, y=author_col, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    # =========================================================================
    # KEYWORDS
    # =========================================================================
    
    def keywords_bar(
        self,
        top_n: int = 20,
        keyword_type: str = "author",
        metric: str = "Number of documents",
        horizontal: bool = True,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "keywords_bar",
        **kwargs,
    ) -> Any:
        """
        Plot top keywords as bar chart.
        
        Parameters
        ----------
        keyword_type : str
            "author" for author keywords, "index" for index keywords.
        """
        if keyword_type == "author":
            if not hasattr(self._biblio, "author_keywords_counts_df") or self._biblio.author_keywords_counts_df is None:
                self._biblio.count_author_keywords()
            df = self._biblio.author_keywords_counts_df.head(top_n).copy()
            label_col = "Keyword"  # Actual column name from count method
            title_label = "Author Keywords"
        else:
            if not hasattr(self._biblio, "index_keywords_counts_df") or self._biblio.index_keywords_counts_df is None:
                self._biblio.count_index_keywords()
            df = self._biblio.index_keywords_counts_df.head(top_n).copy()
            label_col = "Keyword"
            title_label = "Index Keywords"
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"Top {top_n} {title_label}")
        
        if horizontal:
            fig = self._Plot().barh(df, x=metric, y=label_col, config=config, **kwargs)
        else:
            fig = self._Plot().bar(df, x=label_col, y=metric, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def keywords_wordcloud(
        self,
        top_n: int = 100,
        keyword_type: str = "author",
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "keywords_wordcloud",
        **kwargs,
    ) -> Any:
        """Plot keywords as word cloud."""
        if keyword_type == "author":
            if not hasattr(self._biblio, "author_keywords_counts_df") or self._biblio.author_keywords_counts_df is None:
                self._biblio.count_author_keywords()
            df = self._biblio.author_keywords_counts_df.head(top_n).copy()
            # Find the keyword column name
            text_col = "Keyword"  # Standard name from count method
            title_label = "Author Keywords"
        else:
            if not hasattr(self._biblio, "index_keywords_counts_df") or self._biblio.index_keywords_counts_df is None:
                self._biblio.count_index_keywords()
            df = self._biblio.index_keywords_counts_df.head(top_n).copy()
            text_col = "Keyword"
            title_label = "Index Keywords"
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"{title_label} Word Cloud")
        
        # Find the count column
        weight_col = self._find_column(df, ["Number of documents", "Number of Documents", "Count"])
        if weight_col is None:
            weight_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        fig = self._Plot().wordcloud(df, text=text_col, weight=weight_col, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    # =========================================================================
    # COUNTRIES
    # =========================================================================
    
    def countries_bar(
        self,
        top_n: int = 20,
        country_type: str = "all",
        metric: str = "Number of documents",
        horizontal: bool = True,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "countries_bar",
        **kwargs,
    ) -> Any:
        """
        Plot top countries as bar chart.
        
        Parameters
        ----------
        country_type : str
            "all" for all countries, "ca" for corresponding author countries.
        """
        if country_type == "ca":
            if not hasattr(self._biblio, "ca_country_counts_df") or self._biblio.ca_country_counts_df is None:
                self._biblio.count_ca_countries()
            df = self._biblio.ca_country_counts_df.head(top_n).copy()
            title = f"Top {top_n} Corresponding Author Countries"
        else:
            if not hasattr(self._biblio, "all_countries_counts_df") or self._biblio.all_countries_counts_df is None:
                self._biblio.count_all_countries()
            df = self._biblio.all_countries_counts_df.head(top_n).copy()
            title = f"Top {top_n} Countries"
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=title)
        
        if horizontal:
            fig = self._Plot().barh(df, x=metric, y="Country", config=config, **kwargs)
        else:
            fig = self._Plot().bar(df, x="Country", y=metric, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def countries_treemap(
        self,
        top_n: int = 20,
        country_type: str = "all",
        metric: str = "Number of documents",
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "countries_treemap",
        **kwargs,
    ) -> Any:
        """Plot countries as treemap."""
        if country_type == "ca":
            if not hasattr(self._biblio, "ca_country_counts_df") or self._biblio.ca_country_counts_df is None:
                self._biblio.count_ca_countries()
            df = self._biblio.ca_country_counts_df.head(top_n).copy()
        else:
            if not hasattr(self._biblio, "all_countries_counts_df") or self._biblio.all_countries_counts_df is None:
                self._biblio.count_all_countries()
            df = self._biblio.all_countries_counts_df.head(top_n).copy()
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"Top {top_n} Countries")
        
        fig = self._Plot().treemap(df, values=metric, labels="Country", config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    # =========================================================================
    # TIME SERIES / PRODUCTION
    # =========================================================================
    
    def production_line(
        self,
        cumulative: bool = False,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "production_line",
        **kwargs,
    ) -> Any:
        """
        Plot scientific production over time.
        
        Parameters
        ----------
        cumulative : bool
            Show cumulative production.
        """
        if not hasattr(self._biblio, "production_df") or self._biblio.production_df is None:
            self._biblio.get_production()
        
        df = self._biblio.production_df.copy()
        
        # Find the count column (handles case variations)
        y_col = self._find_column(df, ["Number of Documents", "Number of documents", "Count", "Documents"])
        if y_col is None:
            raise ValueError(f"Could not find count column in production_df. Columns: {df.columns.tolist()}")
        
        if cumulative:
            if "Cumulative Documents" in df.columns:
                y_col = "Cumulative Documents"
            else:
                df["Cumulative"] = df[y_col].cumsum()
                y_col = "Cumulative"
        
        config = self._merge_config(config)
        if config.title is None:
            title = "Cumulative Scientific Production" if cumulative else "Annual Scientific Production"
            config = config.update(title=title)
        if config.xlabel is None:
            config = config.update(xlabel="Year")
        if config.ylabel is None:
            config = config.update(ylabel="Publications")
        
        fig = self._Plot().line(df, x="Year", y=y_col, config=config.update(show_markers=True), **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def production_area(
        self,
        cumulative: bool = False,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "production_area",
        **kwargs,
    ) -> Any:
        """Plot scientific production as area chart."""
        if not hasattr(self._biblio, "production_df") or self._biblio.production_df is None:
            self._biblio.get_production()
        
        df = self._biblio.production_df.copy()
        
        y_col = self._find_column(df, ["Number of Documents", "Number of documents", "Count"])
        if y_col is None:
            raise ValueError("Could not find count column")
        
        if cumulative:
            if "Cumulative Documents" in df.columns:
                y_col = "Cumulative Documents"
            else:
                df["Cumulative"] = df[y_col].cumsum()
                y_col = "Cumulative"
        
        config = self._merge_config(config)
        if config.title is None:
            title = "Cumulative Production" if cumulative else "Annual Production"
            config = config.update(title=title)
        
        fig = self._Plot().area(df, x="Year", y=y_col, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def production_bar(
        self,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "production_bar",
        **kwargs,
    ) -> Any:
        """Plot scientific production as bar chart."""
        if not hasattr(self._biblio, "production_df") or self._biblio.production_df is None:
            self._biblio.get_production()
        
        df = self._biblio.production_df.copy()
        
        y_col = self._find_column(df, ["Number of Documents", "Number of documents", "Count"])
        if y_col is None:
            raise ValueError("Could not find count column")
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title="Annual Scientific Production")
        
        fig = self._Plot().bar(df, x="Year", y=y_col, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    # =========================================================================
    # DOCUMENT TYPES
    # =========================================================================
    
    def doctypes_bar(
        self,
        horizontal: bool = True,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "doctypes_bar",
        **kwargs,
    ) -> Any:
        """Plot document types distribution."""
        if not hasattr(self._biblio, "document_types_counts_df") or self._biblio.document_types_counts_df is None:
            self._biblio.count_document_types()
        
        df = self._biblio.document_types_counts_df.copy()
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title="Document Types")
        
        if horizontal:
            fig = self._Plot().barh(df, x="Number of documents", y="Document Type", config=config, **kwargs)
        else:
            fig = self._Plot().bar(df, x="Document Type", y="Number of documents", config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def doctypes_donut(
        self,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "doctypes_bar",
        **kwargs,
    ) -> Any:
        """Plot document types as vertical bar chart."""
        if not hasattr(self._biblio, "document_types_counts_df") or self._biblio.document_types_counts_df is None:
            self._biblio.count_document_types()
        
        df = self._biblio.document_types_counts_df.copy()
        
        # Find the count column
        count_col = self._find_column(df, ["Number of documents", "Number of Documents", "Count"])
        if count_col is None:
            count_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title="Document Types")
        if config.xlabel is None:
            config = config.update(xlabel="Document Type")
        if config.ylabel is None:
            config = config.update(ylabel="Number of Documents")
        
        fig = self._Plot().bar(df, x="Document Type", y=count_col, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def doctypes_bar(
        self,
        horizontal: bool = False,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "doctypes_bar",
        **kwargs,
    ) -> Any:
        """Plot document types distribution as bar chart."""
        if not hasattr(self._biblio, "document_types_counts_df") or self._biblio.document_types_counts_df is None:
            self._biblio.count_document_types()
        
        df = self._biblio.document_types_counts_df.copy()
        
        # Find the count column
        count_col = self._find_column(df, ["Number of documents", "Number of Documents", "Count"])
        if count_col is None:
            count_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title="Document Types")
        
        if horizontal:
            fig = self._Plot().barh(df, x=count_col, y="Document Type", config=config, **kwargs)
        else:
            fig = self._Plot().bar(df, x="Document Type", y=count_col, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    # =========================================================================
    # CITATIONS
    # =========================================================================
    
    def citations_histogram(
        self,
        bins: int = 30,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "citations_histogram",
        **kwargs,
    ) -> Any:
        """Plot citation distribution."""
        cite_col = self._biblio._get_column("Cited by", required=False)
        if cite_col is None:
            raise ValueError("Citation column not found")
        
        df = self._biblio.df[[cite_col]].dropna().copy()
        df.columns = ["Citations"]
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title="Citation Distribution")
        if config.xlabel is None:
            config = config.update(xlabel="Citations")
        
        fig = self._Plot().histogram(df, x="Citations", bins=bins, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def citations_boxplot(
        self,
        by: str = "year",
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "citations_boxplot",
        **kwargs,
    ) -> Any:
        """
        Plot citation distribution by group.
        
        Parameters
        ----------
        by : str
            Group by "year", "doctype", or "source".
        """
        cite_col = self._biblio._get_column("Cited by", required=False)
        if cite_col is None:
            raise ValueError("Citation column not found")
        
        if by == "year":
            group_col = self._biblio._get_column("Year")
        elif by == "doctype":
            group_col = self._biblio._get_column("Document Type")
        elif by == "source":
            group_col = self._biblio._get_column("Source")
        else:
            group_col = by
        
        df = self._biblio.df[[group_col, cite_col]].dropna().copy()
        df.columns = ["Group", "Citations"]
        
        # Limit groups for readability
        top_groups = df.groupby("Group").size().nlargest(10).index
        df = df[df["Group"].isin(top_groups)]
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"Citations by {by.title()}")
        
        fig = self._Plot().boxplot(df, x="Group", y="Citations", config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    # =========================================================================
    # SCATTER / BUBBLE
    # =========================================================================
    
    def sources_bubble(
        self,
        top_n: int = 30,
        x_metric: str = "Number of documents",
        y_metric: str = "Total citations",
        size_metric: str = "h-index",
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "sources_bubble",
        **kwargs,
    ) -> Any:
        """
        Plot sources as bubble chart.
        
        Parameters
        ----------
        x_metric, y_metric : str
            Metrics for axes.
        size_metric : str
            Metric for bubble size.
        """
        if not hasattr(self._biblio, "sources_counts_df") or self._biblio.sources_counts_df is None:
            self._biblio.count_sources()
        
        df = self._biblio.sources_counts_df.head(top_n).copy()
        
        # Ensure required columns exist
        required = [x_metric, y_metric, size_metric]
        available = [c for c in required if c in df.columns]
        
        if len(available) < 3:
            # Fall back to basic metrics
            x_metric = "Number of documents"
            y_metric = "Total citations" if "Total citations" in df.columns else "Number of documents"
            size_metric = x_metric
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"Top {top_n} Sources")
        
        fig = self._Plot().bubble(
            df, 
            x=x_metric, 
            y=y_metric, 
            size=size_metric,
            label="Source",
            config=config,
            **kwargs
        )
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def sources_scatter(
        self,
        top_n: int = 20,
        x_metric: str = "Number of documents",
        y_metric: str = "Total citations",
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "sources_scatter",
        **kwargs,
    ) -> Any:
        """
        Scatter plot of top sources.
        
        Parameters
        ----------
        top_n : int
            Number of top sources to plot.
        x_metric : str
            Metric for x-axis (default: "Number of documents").
        y_metric : str
            Metric for y-axis (default: "Total citations").
        """
        if not hasattr(self._biblio, "sources_counts_df") or self._biblio.sources_counts_df is None:
            self._biblio.count_sources()
        
        df = self._biblio.sources_counts_df.head(top_n).copy()
        
        # Validate columns
        if x_metric not in df.columns:
            x_metric = "Number of documents"
        if y_metric not in df.columns:
            y_metric = "Total citations" if "Total citations" in df.columns else "Number of documents"
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"Top {top_n} Sources")
        
        fig = self._Plot().scatter(
            df,
            x=x_metric,
            y=y_metric,
            label="Source",
            config=config,
            **kwargs
        )
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def authors_scatter(
        self,
        top_n: int = 20,
        x_metric: str = "Number of documents",
        y_metric: str = "Total citations",
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "authors_scatter",
        **kwargs,
    ) -> Any:
        """
        Scatter plot of top authors.
        
        Parameters
        ----------
        top_n : int
            Number of top authors to plot.
        x_metric : str
            Metric for x-axis.
        y_metric : str
            Metric for y-axis.
        """
        if not hasattr(self._biblio, "authors_counts_df") or self._biblio.authors_counts_df is None:
            self._biblio.count_authors()
        
        df = self._biblio.authors_counts_df.head(top_n).copy()
        
        # Find the author column
        author_col = None
        for col in ["Author", "Authors", df.columns[0]]:
            if col in df.columns:
                author_col = col
                break
        
        if x_metric not in df.columns:
            x_metric = "Number of documents"
        if y_metric not in df.columns:
            y_metric = "Total citations" if "Total citations" in df.columns else "Number of documents"
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"Top {top_n} Authors")
        
        fig = self._Plot().scatter(
            df,
            x=x_metric,
            y=y_metric,
            label=author_col,
            config=config,
            **kwargs
        )
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    def countries_scatter(
        self,
        top_n: int = 20,
        x_metric: str = "Number of documents",
        y_metric: str = "Total citations",
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "countries_scatter",
        **kwargs,
    ) -> Any:
        """
        Scatter plot of top countries.
        
        Parameters
        ----------
        top_n : int
            Number of top countries to plot.
        x_metric : str
            Metric for x-axis.
        y_metric : str
            Metric for y-axis.
        """
        if not hasattr(self._biblio, "all_countries_counts_df") or self._biblio.all_countries_counts_df is None:
            self._biblio.count_all_countries()
        
        df = self._biblio.all_countries_counts_df.head(top_n).copy()
        
        if x_metric not in df.columns:
            x_metric = "Number of documents"
        if y_metric not in df.columns:
            y_metric = "Total citations" if "Total citations" in df.columns else "Number of documents"
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"Top {top_n} Countries")
        
        fig = self._Plot().scatter(
            df,
            x=x_metric,
            y=y_metric,
            label="Country",
            config=config,
            **kwargs
        )
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    
    def compare_periods(
        self,
        column: str,
        period1: tuple,
        period2: tuple,
        top_n: int = 15,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "compare_periods",
        **kwargs,
    ) -> Any:
        """
        Compare item counts between two periods using dumbbell chart.
        
        Parameters
        ----------
        column : str
            Column to compare (e.g., "Author Keywords", "Source").
        period1, period2 : tuple
            Year ranges as (start, end).
        top_n : int
            Number of items to compare.
        """
        year_col = self._biblio._get_column("Year")
        col = self._biblio._get_column(column)
        sep = self._biblio.default_separator
        
        from collections import Counter
        
        def count_items(df_subset):
            counts = Counter()
            for val in df_subset[col].dropna():
                if isinstance(val, str):
                    if sep in val:
                        for item in val.split(sep):
                            counts[item.strip()] += 1
                    else:
                        counts[val.strip()] += 1
            return counts
        
        df1 = self._biblio.df[(self._biblio.df[year_col] >= period1[0]) & 
                              (self._biblio.df[year_col] <= period1[1])]
        df2 = self._biblio.df[(self._biblio.df[year_col] >= period2[0]) & 
                              (self._biblio.df[year_col] <= period2[1])]
        
        counts1 = count_items(df1)
        counts2 = count_items(df2)
        
        # Get top items overall
        all_items = set(counts1.keys()) | set(counts2.keys())
        total_counts = {item: counts1.get(item, 0) + counts2.get(item, 0) for item in all_items}
        top_items = sorted(total_counts, key=total_counts.get, reverse=True)[:top_n]
        
        compare_df = pd.DataFrame({
            "Item": top_items,
            f"{period1[0]}-{period1[1]}": [counts1.get(i, 0) for i in top_items],
            f"{period2[0]}-{period2[1]}": [counts2.get(i, 0) for i in top_items],
        })
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"{column}: {period1[0]}-{period1[1]} vs {period2[0]}-{period2[1]}")
        
        fig = self._Plot().dumbbell(
            compare_df,
            y="Item",
            x_start=f"{period1[0]}-{period1[1]}",
            x_end=f"{period2[0]}-{period2[1]}",
            config=config,
            **kwargs
        )
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    # =========================================================================
    # HEATMAP
    # =========================================================================
    
    def cooccurrence_heatmap(
        self,
        column: str,
        top_n: int = 20,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "cooccurrence_heatmap",
        **kwargs,
    ) -> Any:
        """
        Plot co-occurrence matrix as heatmap.
        
        Parameters
        ----------
        column : str
            Column for co-occurrence (e.g., "Author Keywords").
        top_n : int
            Number of top items.
        """
        from biblium.utilsbib import compute_cooccurrence_matrix
        
        col = self._biblio._get_column(column)
        matrix = compute_cooccurrence_matrix(
            self._biblio.df, 
            col, 
            top_n=top_n,
            separator=self._biblio.default_separator,
        )
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title=f"{column} Co-occurrence")
        
        fig = self._Plot().heatmap(matrix, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    # =========================================================================
    # NETWORK
    # =========================================================================
    
    def keyword_network(
        self,
        top_n: int = 50,
        min_edge_weight: int = 2,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "keyword_network",
        **kwargs,
    ) -> Any:
        """Plot keyword co-occurrence network."""
        from biblium.utilsbib import compute_cooccurrence_matrix, build_links_from_matrix
        
        kw_col = self._biblio._get_column("Author Keywords", required=False)
        if kw_col is None:
            raise ValueError("Author Keywords column not found")
        
        matrix = compute_cooccurrence_matrix(
            self._biblio.df,
            kw_col,
            top_n=top_n,
            separator=self._biblio.default_separator,
        )
        
        # Build nodes and edges
        nodes = pd.DataFrame({"id": matrix.columns.tolist()})
        
        edges_df = build_links_from_matrix(matrix)
        edges_df = edges_df[edges_df["Weight"] >= min_edge_weight]
        edges_df = edges_df.rename(columns={"Source": "source", "Target": "target", "Weight": "weight"})
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title="Keyword Co-occurrence Network")
        
        fig = self._Plot().network(nodes, edges_df, config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
    
    # =========================================================================
    # RADAR COMPARISON
    # =========================================================================
    
    def sources_radar(
        self,
        sources: List[str] = None,
        top_n: int = 5,
        metrics: List[str] = None,
        config: Optional[PlotConfig] = None,
        filename: Optional[str] = "sources_radar",
        **kwargs,
    ) -> Any:
        """
        Compare sources using radar chart.
        
        Parameters
        ----------
        sources : list, optional
            Specific sources to compare. If None, uses top N.
        top_n : int
            Number of top sources if sources not specified.
        metrics : list, optional
            Metrics to compare.
        """
        if not hasattr(self._biblio, "sources_counts_df") or self._biblio.sources_counts_df is None:
            self._biblio.count_sources()
        
        df = self._biblio.sources_counts_df.copy()
        
        if sources:
            df = df[df["Source"].isin(sources)]
        else:
            df = df.head(top_n)
        
        if metrics is None:
            # Use available numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            metrics = [c for c in numeric_cols if c not in ["Source"]][:5]
        
        if len(metrics) < 3:
            raise ValueError("Need at least 3 metrics for radar chart")
        
        # Normalize metrics to 0-100 scale
        radar_df = df[["Source"] + metrics].copy()
        for m in metrics:
            max_val = radar_df[m].max()
            if max_val > 0:
                radar_df[m] = (radar_df[m] / max_val) * 100
        
        radar_df = radar_df.set_index("Source").T
        
        config = self._merge_config(config)
        if config.title is None:
            config = config.update(title="Source Comparison")
        
        fig = self._Plot().radar(radar_df, categories=metrics, values=list(df["Source"].head(top_n)), config=config, **kwargs)
        
        if filename:
            save_path = self._get_save_path(filename)
            if save_path:
                self._Plot().save(fig, save_path, config)
        
        return fig
