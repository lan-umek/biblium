# -*- coding: utf-8 -*-
"""
Sentiment Analysis Plots
========================
Visualization functions for sentiment analysis results.
Follows Biblium plotting conventions: single plots, no gridlines, simple colors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any, Tuple

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Default color following Biblium convention
DEFAULT_COLOR = "lightblue"


def save_plot(filename: str, dpi: int = 600):
    """Save plot to multiple formats."""
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(f"{filename}.{ext}", dpi=dpi, bbox_inches="tight")


def plot_sentiment_distribution(
    result: dict,
    column: str = "Composite Sentiment",
    filename: str = None,
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
    color: str = DEFAULT_COLOR,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """
    Plot sentiment score distribution as histogram.
    
    Parameters
    ----------
    result : dict
        Result from analyze_sentiment_advanced.
    column : str
        Sentiment column to plot.
    filename : str, optional
        Save path for the figure.
    dpi : int
        Figure resolution.
    show : bool
        Display the figure.
    figsize : tuple
        Figure size.
    color : str
        Bar color.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.figure.Figure or None
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.grid(False)
    
    sentiment_df = result.get("sentiment_df", pd.DataFrame())
    
    if column not in sentiment_df.columns:
        ax.text(0.5, 0.5, f'Column "{column}" not found', 
               ha='center', va='center', transform=ax.transAxes)
        if fig is not None:
            plt.tight_layout()
            if filename:
                save_plot(filename, dpi=dpi)
            if show:
                plt.show()
            plt.close()
        return fig
    
    scores = sentiment_df[column].dropna()
    
    if len(scores) == 0:
        ax.text(0.5, 0.5, 'No sentiment data available', 
               ha='center', va='center', transform=ax.transAxes)
        if fig is not None:
            plt.tight_layout()
            if filename:
                save_plot(filename, dpi=dpi)
            if show:
                plt.show()
            plt.close()
        return fig
    
    ax.hist(scores, bins=30, color=color, edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Sentiment Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Sentiment Score Distribution', fontsize=12)
    
    if HAS_SEABORN:
        sns.despine(ax=ax)
    
    if fig is not None:
        plt.tight_layout()
        if filename is not None:
            save_plot(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close()
    
    return fig


def plot_sentiment_categories(
    result: dict,
    filename: str = None,
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
    color: str = DEFAULT_COLOR,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """
    Plot sentiment category distribution as bar chart.
    
    Parameters
    ----------
    result : dict
        Result from analyze_sentiment_advanced.
    filename : str, optional
        Save path for the figure.
    dpi : int
        Figure resolution.
    show : bool
        Display the figure.
    figsize : tuple
        Figure size.
    color : str
        Bar color.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.figure.Figure or None
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.grid(False)
    
    sentiment_df = result.get("sentiment_df", pd.DataFrame())
    
    # Check for category column
    if "Sentiment Category" in sentiment_df.columns:
        category_col = "Sentiment Category"
    else:
        ax.text(0.5, 0.5, 'No category data available', 
               ha='center', va='center', transform=ax.transAxes)
        if fig is not None:
            plt.tight_layout()
            if filename:
                save_plot(filename, dpi=dpi)
            if show:
                plt.show()
            plt.close()
        return fig
    
    category_counts = sentiment_df[category_col].value_counts()
    
    if len(category_counts) == 0:
        ax.text(0.5, 0.5, 'No sentiment categories found', 
               ha='center', va='center', transform=ax.transAxes)
        if fig is not None:
            plt.tight_layout()
            if filename:
                save_plot(filename, dpi=dpi)
            if show:
                plt.show()
            plt.close()
        return fig
    
    bars = ax.bar(range(len(category_counts)), category_counts.values, color=color, 
                  edgecolor='white', alpha=0.7)
    ax.set_xticks(range(len(category_counts)))
    ax.set_xticklabels([str(c) for c in category_counts.index], fontsize=10)
    ax.set_ylabel('Number of Documents', fontsize=11)
    ax.set_title('Sentiment Category Distribution', fontsize=12)
    
    # Add value labels
    for bar, val in zip(bars, category_counts.values):
        pct = val / category_counts.sum() * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(category_counts) * 0.02,
                f"{val}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9)
    
    ax.set_ylim(0, max(category_counts) * 1.15)
    
    if HAS_SEABORN:
        sns.despine(ax=ax)
    
    if fig is not None:
        plt.tight_layout()
        if filename is not None:
            save_plot(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close()
    
    return fig


def plot_sentiment_temporal(
    result: dict,
    filename: str = None,
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
    color: str = DEFAULT_COLOR,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """
    Plot sentiment trends over time.
    
    Parameters
    ----------
    result : dict
        Result from analyze_sentiment_advanced.
    filename : str, optional
        Save path for the figure.
    dpi : int
        Figure resolution.
    show : bool
        Display the figure.
    figsize : tuple
        Figure size.
    color : str
        Line color.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.figure.Figure or None
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.grid(False)
    
    temporal = result.get("temporal_trends")
    
    if temporal is None or len(temporal) == 0:
        ax.text(0.5, 0.5, 'No temporal data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Sentiment Trend Over Time', fontsize=12)
        if fig is not None:
            plt.tight_layout()
            if filename:
                save_plot(filename, dpi=dpi)
            if show:
                plt.show()
            plt.close()
        return fig
    
    ax.plot(temporal["Year"], temporal["Mean Sentiment"], marker='o', linewidth=2,
            color=color, markersize=5)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Mean Sentiment Score', fontsize=11)
    ax.set_title('Sentiment Trend Over Time', fontsize=12)
    
    if HAS_SEABORN:
        sns.despine(ax=ax)
    
    if fig is not None:
        plt.tight_layout()
        if filename is not None:
            save_plot(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close()
    
    return fig


def plot_sentiment_certainty(
    result: dict,
    filename: str = None,
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 6),
    color: str = DEFAULT_COLOR,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """
    Plot certainty score distribution.
    
    Parameters
    ----------
    result : dict
        Result from analyze_sentiment_advanced.
    filename : str, optional
        Save path for the figure.
    dpi : int
        Figure resolution.
    show : bool
        Display the figure.
    figsize : tuple
        Figure size.
    color : str
        Bar color.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.figure.Figure or None
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.grid(False)
    
    sentiment_df = result.get("sentiment_df", pd.DataFrame())
    
    if "Certainty Score" not in sentiment_df.columns:
        ax.text(0.5, 0.5, 'No certainty data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Certainty Score Distribution', fontsize=12)
        if fig is not None:
            plt.tight_layout()
            if filename:
                save_plot(filename, dpi=dpi)
            if show:
                plt.show()
            plt.close()
        return fig
    
    scores = sentiment_df["Certainty Score"].dropna()
    
    if len(scores) == 0:
        ax.text(0.5, 0.5, 'No certainty data', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Certainty Score Distribution', fontsize=12)
        if fig is not None:
            plt.tight_layout()
            if filename:
                save_plot(filename, dpi=dpi)
            if show:
                plt.show()
            plt.close()
        return fig
    
    ax.hist(scores, bins=30, color=color, edgecolor='white', alpha=0.7)
    ax.set_xlabel('Certainty Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Certainty Score Distribution', fontsize=12)
    
    if HAS_SEABORN:
        sns.despine(ax=ax)
    
    if fig is not None:
        plt.tight_layout()
        if filename is not None:
            save_plot(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close()
    
    return fig


def plot_sentiment_by_source(
    result: dict,
    top_n: int = 20,
    filename: str = None,
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (10, 8),
    color: str = DEFAULT_COLOR,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """
    Plot sentiment scores by publication source/journal as horizontal bar chart.
    
    Parameters
    ----------
    result : dict
        Result from analyze_sentiment_advanced.
    top_n : int
        Number of sources to show.
    filename : str, optional
        Save path for the figure.
    dpi : int
        Figure resolution.
    show : bool
        Display the figure.
    figsize : tuple
        Figure size.
    color : str
        Bar color.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.figure.Figure or None
    """
    source_analysis = result.get("source_analysis")
    
    fig = None
    if ax is None:
        if source_analysis is None or len(source_analysis) == 0:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            top_sources = source_analysis.head(top_n)
            fig, ax = plt.subplots(figsize=(figsize[0], max(figsize[1], 0.4 * len(top_sources))))
    
    ax.grid(False)
    
    if source_analysis is None or len(source_analysis) == 0:
        ax.text(0.5, 0.5, 'No source analysis data available\n(Enable analyze_by_source=True)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Sentiment by Source', fontsize=12)
        if fig is not None:
            plt.tight_layout()
            if filename:
                save_plot(filename, dpi=dpi)
            if show:
                plt.show()
            plt.close()
        return fig
    
    # Get top N sources
    top_sources = source_analysis.head(top_n).copy()
    top_sources = top_sources.sort_values("Mean Sentiment", ascending=True)
    
    # Truncate long source names
    sources = [s[:50] + '...' if len(str(s)) > 50 else str(s) for s in top_sources["Source"]]
    sentiments = top_sources["Mean Sentiment"].values
    
    bars = ax.barh(range(len(sources)), sentiments, color=color, alpha=0.7, edgecolor='white')
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources, fontsize=9)
    ax.set_xlabel('Mean Sentiment Score', fontsize=11)
    ax.set_title(f'Sentiment by Source (Top {len(sources)})', fontsize=12)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label = f"{width:.3f}"
        ax.text(width, bar.get_y() + bar.get_height()/2, f" {label}", 
               va='center', ha='left' if width >= 0 else 'right', fontsize=8)
    
    if HAS_SEABORN:
        sns.despine(ax=ax)
    
    if fig is not None:
        plt.tight_layout()
        if filename is not None:
            save_plot(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close()
    
    return fig


def plot_sentiment_heatmap(
    result: dict,
    filename: str = None,
    dpi: int = 600,
    show: bool = True,
    figsize: tuple = (8, 6),
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """
    Plot correlation heatmap for sentiment variables.
    
    Parameters
    ----------
    result : dict
        Result from analyze_sentiment_advanced.
    filename : str, optional
        Save path for the figure.
    dpi : int
        Figure resolution.
    show : bool
        Display the figure.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for heatmap.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.figure.Figure or None
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.grid(False)
    
    correlations = result.get("correlations")
    
    if correlations is None or len(correlations) == 0:
        ax.text(0.5, 0.5, 'No correlation data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Sentiment Variable Correlations', fontsize=12)
        if fig is not None:
            plt.tight_layout()
            if filename:
                save_plot(filename, dpi=dpi)
            if show:
                plt.show()
            plt.close()
        return fig
    
    # Shorten column names for display
    short_names = {
        'Composite Sentiment': 'Composite',
        'VADER Compound': 'VADER',
        'TextBlob Polarity': 'Polarity',
        'TextBlob Subjectivity': 'Subjectivity',
        'Certainty Score': 'Certainty',
        'Hedging Score': 'Hedging',
        'Cited by': 'Citations',
    }
    
    # Create a copy with shortened names
    corr_display = correlations.copy()
    corr_display.columns = [short_names.get(c, c) for c in corr_display.columns]
    corr_display.index = [short_names.get(i, i) for i in corr_display.index]
    
    if HAS_SEABORN:
        mask = np.triu(np.ones_like(corr_display, dtype=bool), k=1)
        sns.heatmap(corr_display, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                   center=0, vmin=-1, vmax=1, square=True, ax=ax,
                   linewidths=0.5, cbar_kws={'shrink': 0.8},
                   annot_kws={'size': 8})
        # Rotate labels for better fit
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    else:
        im = ax.imshow(corr_display.values, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_display.columns)))
        ax.set_yticks(range(len(corr_display.index)))
        ax.set_xticklabels(corr_display.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(corr_display.index, fontsize=9)
        if fig is not None:
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    ax.set_title('Sentiment Variable Correlations', fontsize=12)
    
    if fig is not None:
        # Use constrained_layout or subplots_adjust for better label handling
        plt.subplots_adjust(bottom=0.25, left=0.2)
        if filename is not None:
            save_plot(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close()
    else:
        # When drawing on provided axes, adjust the figure if possible
        try:
            ax.figure.subplots_adjust(bottom=0.25, left=0.2)
        except:
            pass
    
    return fig


# Export all functions
__all__ = [
    "plot_sentiment_distribution",
    "plot_sentiment_categories",
    "plot_sentiment_temporal",
    "plot_sentiment_certainty",
    "plot_sentiment_by_source",
    "plot_sentiment_heatmap",
]
