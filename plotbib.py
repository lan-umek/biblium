# --- Standard library ---
import os
import math
import itertools
import textwrap
import re
from collections import Counter, defaultdict
from datetime import datetime
import warnings

# --- Typing ---
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# --- Data handling ---
import numpy as np
import pandas as pd

# --- Visualization: Matplotlib & Seaborn ---
import matplotlib.pyplot as plt
from matplotlib import cm, colors, rcParams
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors

import seaborn as sns

# --- Other visualization libraries ---
from adjustText import adjust_text
import squarify

# UpSet plots
try:
    from upsetplot import UpSet, from_indicators
except ImportError:
    UpSet = from_indicators = None

# Venn diagrams
try:
    from venn import venn
except ImportError:
    venn = None

# Word clouds
try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

# Holoviews (optional)
try:
    import holoviews as hv
    from holoviews import opts
except ImportError:
    hv = opts = None

# Plotly (optional interactive plots)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale, get_colorscale
except ImportError:
    px = go = sample_colorscale = get_colorscale = None

# --- Clustering/Distance ---
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, norm, pearsonr
import scipy.cluster.hierarchy as sch

import networkx as nx
from textwrap import wrap

# --- Local utilities ---
import utilsbib
from utilsbib import compute_main_path


def wrap_labels(labels, width=50):
    """Wrap long labels to a given width."""
    return ["\n".join(textwrap.wrap(label, width)) for label in labels]

def infer_color_scheme(values):
    """Infer color scheme based on number of unique values and data type."""
    if values is None:
        return "default"
    values_series = pd.Series(values)
    if not pd.api.types.is_integer_dtype(values_series):
        return "continuous"
    unique_vals = values_series.nunique()
    return "categorical" if unique_vals <= 10 else "continuous"

def get_colors(n, color_scheme="default", values=None, cmap="viridis", default_color="lightblue", categorical_palette=None):
    """Return a list of colors based on the selected scheme."""
    if categorical_palette is None:
        categorical_palette = ["lightblue", "lightgreen", "lightcoral", "khaki", "lightgrey", "plum", "salmon", "skyblue", "palegreen", "gold"]

    if color_scheme == "default" or values is None:
        return [default_color] * n
    elif color_scheme == "categorical":
        palette = sns.color_palette(categorical_palette)
        return [palette[i % len(palette)] for i in range(n)]
    elif color_scheme == "continuous":
        norm = plt.Normalize(min(values), max(values))
        color_map = plt.cm.get_cmap(cmap)
        return [color_map(norm(val)) for val in values]
    else:
        raise ValueError(f"Unknown color_scheme: {color_scheme}")

def save_plot(filename_base, dpi=600):
    """Save plot to PNG, SVG, and PDF with tight layout and given DPI."""
    for ext in ["png", "svg", "pdf"]:
        path = f"{filename_base}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=dpi)

def plot_barh(df, x, y, color_scheme="default", color_by=None, filename="barh_plot", dpi=600, grid=False, wrap_width=50, cmap="viridis", default_color="lightblue", categorical_palette=None, label_fontsize=8, axis_labelsize=None, colorbar_labelsize=None, show=True):
    """
    Create a horizontal bar plot from a DataFrame.
    """
    df = df.sort_values(by=x, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(df))))
    y_labels = wrap_labels(df[y], wrap_width)
    values_for_colors = df[color_by] if color_by else None
    if color_by and color_scheme == "default":
        color_scheme = infer_color_scheme(values_for_colors)
    colors = get_colors(len(df), color_scheme, values=values_for_colors, cmap=cmap, default_color=default_color, categorical_palette=categorical_palette)
    bars = ax.barh(y_labels, df[x], color=colors)
    for bar in bars:
        width = bar.get_width()
        label = f"{int(width)}" if float(width).is_integer() else f"{width:.2f}"
        ax.text(width, bar.get_y() + bar.get_height() / 2, label, va="center", ha="left", fontsize=label_fontsize)
    if axis_labelsize:
        ax.set_xlabel(x, fontsize=axis_labelsize)
        ax.set_ylabel(y, fontsize=axis_labelsize)
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    if color_scheme == "continuous" and values_for_colors is not None:
        norm = plt.Normalize(min(values_for_colors), max(values_for_colors))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        if colorbar_labelsize:
            cbar.set_label(color_by, fontsize=colorbar_labelsize)
        else:
            cbar.set_label(color_by)
    elif color_scheme == "categorical":
        ax.legend(handles=bars, labels=y_labels, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(grid)
    sns.despine()
    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()

def plot_lollipop(df, x, y, color_scheme="default", color_by=None, filename="lollipop_plot", dpi=600, grid=False, wrap_width=50, cmap="viridis", default_color="lightblue", categorical_palette=None, label_fontsize=8, axis_labelsize=None, colorbar_labelsize=None, show=True):
    """
    Create a horizontal lollipop plot from a DataFrame.
    """
    df = df.sort_values(by=x, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(df))))
    y_labels = wrap_labels(df[y], wrap_width)
    values_for_colors = df[color_by] if color_by else None
    if color_by and color_scheme == "default":
        color_scheme = infer_color_scheme(values_for_colors)
    colors = get_colors(len(df), color_scheme, values=values_for_colors, cmap=cmap, default_color=default_color, categorical_palette=categorical_palette)
    ax.hlines(y=range(len(df)), xmin=0, xmax=df[x], color=colors, linewidth=1.5)

    # Scale marker size relative to maximum value
    max_val = df[x].max()
    sizes = 300 * (df[x] / max_val) if max_val > 0 else [50] * len(df)

    ax.scatter(df[x], range(len(df)), color=colors, s=sizes, zorder=3)
    for i, val in enumerate(df[x]):
        label = f"{int(val)}" if float(val).is_integer() else f"{val:.2f}"
        ax.text(val, i, label, va="center", ha="left", fontsize=label_fontsize)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(y_labels)
    if axis_labelsize:
        ax.set_xlabel(x, fontsize=axis_labelsize)
        ax.set_ylabel(y, fontsize=axis_labelsize)
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    if color_scheme == "continuous" and values_for_colors is not None:
        norm = plt.Normalize(min(values_for_colors), max(values_for_colors))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        if colorbar_labelsize:
            cbar.set_label(color_by, fontsize=colorbar_labelsize)
        else:
            cbar.set_label(color_by)
    elif color_scheme == "categorical":
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    ax.grid(grid)
    sns.despine()
    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    
def plot_timeseries(df, x="Year", bar_y="Number of Documents", line_y="Cumulative Citations", cut_year=None,
                    filename="timeseries_plot", dpi=600, axis_labelsize=None, show=True,
                    bar_color=None, line_color=None, xrotation=90, bar_labels=None):
    """
    Plot a time series combining bar and line plots for documents and citations.

    Parameters:
    - df: pandas.DataFrame with 'Year', 'Number of Documents', 'Cumulative Documents', 'Number of Citations', 'Cumulative Citations'
    - x: time axis column, default 'Year'
    - bar_y: variable for bars (default: 'Number of Documents')
    - line_y: variable for line (default: 'Cumulative Citations')
    - cut_year: optional int to group all years before it into a single category
    - filename: base filename for saving
    - dpi: resolution of saved images
    - axis_labelsize: font size for axis labels
    - show: whether to display the plot
    - bar_color: color for bar plot (optional)
    - line_color: color for line plot (optional)
    - xrotation: rotation angle for x-axis labels (default: 90)
    - bar_labels: if True, display values above the bars
    """
    df_copy = df.copy()
    if cut_year is not None:
        before_df = df_copy[df_copy[x] < cut_year].copy()
        after_df = df_copy[df_copy[x] >= cut_year].copy().sort_values(by=x)

        combined = {}
        combined[x] = f"before {cut_year}"
        for col in df.columns:
            if col == x:
                continue
            if "Cumulative" in col:
                combined[col] = before_df[col].max()  # last cumulative value
            else:
                combined[col] = before_df[col].sum()  # aggregate non-cumulative values

        before_df = pd.DataFrame([combined])
        df_plot = pd.concat([before_df, after_df], ignore_index=True)
    else:
        df_plot = df_copy.sort_values(by=x)

    fig, ax1 = plt.subplots(figsize=(12, 6))


    if bar_y:
        bar_color = bar_color or "lightblue"
        bars = ax1.bar(df_plot[x].astype(str), df_plot[bar_y], color=bar_color, label=bar_y)
        if bar_labels:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, height, f"{int(round(height))}" if height % 1 == 0 else f"{height:.2f}", ha="center", va="bottom", fontsize=8)
        ax1.set_ylabel(bar_y, fontsize=axis_labelsize)
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    if line_y:
        line_color = line_color or "black"
        ax2 = ax1.twinx()
        ax2.plot(df_plot[x].astype(str), df_plot[line_y], color=line_color, marker="o", linewidth=2, label=line_y)
        ax2.set_ylabel(line_y, fontsize=axis_labelsize)
        ax2.ticklabel_format(style="plain", axis="y")

    ax1.set_xlabel(x, fontsize=axis_labelsize)
    ax1.set_xticks(range(len(df_plot)))
    ax1.set_xticklabels(df_plot[x].astype(str), rotation=xrotation, ha="right")

    ax1.grid(False)
    if line_y:
        ax2.grid(False)

    sns.despine()
    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()

def plot_heatmap(df, filename="heatmap", dpi=600, show=True, normalized=False, cmap="viridis",
                 axis_labelsize=None, cbar_label=None, wrap_width=50, square_cells=None,
                 symmetric_option=None, label_fontsize=10, tick_labelsize=10, cbar_labelsize=10,
                 xlabel=None, ylabel=None):
    """
    Plot a heatmap showing the relationship between two categories, with optional colorbar label. Optionally enforce square aspect ratio.

    Parameters:
    - df: pandas.DataFrame with counts or normalized values
    - filename: base filename for saving (default: 'heatmap')
    - dpi: resolution of saved images (default: 600)
    - show: whether to display the plot
    - normalized: if True, format numbers with 2 decimals; if False, use integers
    - cmap: color map to use (default: 'viridis')
    - axis_labelsize: font size for axis labels (optional)
    - cbar_label: label for the colorbar (default: None)
    - wrap_width: max characters per line for wrapped labels (0 disables wrapping, default: 50)
    - square_cells: if True, enforce square aspect; if None, auto-detect for square matrices (default: None)
    - symmetric_option: if 'mask', mask upper triangle; if 'highlight', draw diagonal; if None, do nothing
    - label_fontsize: font size for heatmap cell labels (default: 10)
    - tick_labelsize: font size for axis tick labels (default: 10)
    - cbar_labelsize: font size for colorbar label (default: 10)
    - xlabel: custom label for x-axis (default: None)
    - ylabel: custom label for y-axis (default: None)
    """


    fig, ax = plt.subplots(figsize=(10, 8))
    

    fmt = ".2f" if normalized else ".0f"

    if wrap_width > 0:
        df.columns = ["\n".join(wrap(str(col), wrap_width)) for col in df.columns]
        df.index = ["\n".join(wrap(str(idx), wrap_width)) for idx in df.index]

    auto_square = square_cells if square_cells is not None else df.shape[0] == df.shape[1]
    mask = None
    if symmetric_option == "mask" and df.shape[0] == df.shape[1] and (df.columns == df.index).all():
        mask = np.triu(np.ones_like(df.values, dtype=bool))

    sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, cbar=True, ax=ax,
                annot_kws={"fontsize": label_fontsize},
                cbar_kws={"label": cbar_label, "format": None} if cbar_label else {},
                square=auto_square, mask=mask)

    if symmetric_option == "highlight" and df.shape[0] == df.shape[1] and (df.columns == df.index).all():
        for i in range(len(df)):
            for j in range(i + 1):
                if i == j:
                    edgecolor = "red"
                    linewidth = 1.5
                else:
                    continue
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor=edgecolor, lw=linewidth))
    ax.set_xlabel(xlabel if xlabel is not None else df.columns.name or "", fontsize=axis_labelsize)
    ax.set_ylabel(ylabel if ylabel is not None else df.index.name or "", fontsize=axis_labelsize)
    ax.tick_params(axis='both', labelsize=tick_labelsize)
    if ax.collections and hasattr(ax.collections[0], 'colorbar') and ax.collections[0].colorbar:
        ax.collections[0].colorbar.ax.tick_params(labelsize=cbar_labelsize)
        if cbar_label:
            ax.collections[0].colorbar.set_label(cbar_label, fontsize=cbar_labelsize)
    ax.set_ylabel(df.index.name or "", fontsize=axis_labelsize)
    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_clustermap(df, filename="clustermap", dpi=600, normalized=False, cmap="viridis",
                    wrap_width=50, square_cells=None, symmetric_option=None,
                    axis_labelsize=None, label_fontsize=10, tick_labelsize=10, cbar_labelsize=10,
                    xlabel=None, ylabel=None, cbar_label=None, figsize=(10, 10),
                    method="average", metric="euclidean", show=True):
    """
    Plot a clustermap from a DataFrame with optional symmetric masking, axis label wrapping,
    colorbar styling, and clustering customization.

    Parameters:
    - df: pandas DataFrame (for full functionality square and symmetric)
    - filename: base filename for saving the plots (default: 'clustermap')
    - dpi: resolution of saved plots (default: 600)
    - normalized: if True, uses float formatting; otherwise, integers (default: False)
    - cmap: colormap for heatmap (default: 'viridis')
    - wrap_width: character limit before wrapping axis labels (0 disables wrapping)
    - square_cells: placeholder for future cell shape control (currently unused)
    - symmetric_option: 'mask' to show lower triangle only, 'highlight' to outline diagonal, or None
    - axis_labelsize: font size for axis labels
    - label_fontsize: font size for annotations in cells
    - tick_labelsize: font size for axis tick labels
    - cbar_labelsize: font size for colorbar label and ticks
    - xlabel, ylabel: axis labels (overrides index/column name if set)
    - cbar_label: label for the colorbar (default: None)
    - figsize: figure size (default: (10, 10))
    - method: linkage algorithm for clustering (e.g. 'average', 'single', etc.)
    - metric: distance function for clustering (e.g. 'euclidean', 'cityblock')
    - show: whether to display the plot after saving (default: True)
    """

    # Optionally wrap long axis labels
    if wrap_width > 0:
        df.columns = ["\n".join(wrap(str(col), wrap_width)) for col in df.columns]
        df.index = ["\n".join(wrap(str(idx), wrap_width)) for idx in df.index]

    fmt = ".2f" if normalized else ".0f"

    # Create the clustermap with clustering result to reorder matrix
    mask = None

    if df.shape[0] == df.shape[1] and (df.columns == df.index).all():
        # Compute linkage on symmetric matrix
        dist = pdist(df.values)
        linkage_matrix = linkage(dist, method=method, metric=metric)
        order = leaves_list(linkage_matrix)
        df = df.iloc[order, :].iloc[:, order]

    if symmetric_option == "mask" and df.shape[0] == df.shape[1] and (df.columns == df.index).all():
        df = df.copy()
        for i in range(df.shape[0]):
            for j in range(i + 1, df.shape[1]):
                df.iloc[j, i] = df.iloc[i, j]
        mask = np.triu(np.ones(df.shape), k=1).astype(bool)

    g = sns.clustermap(df, method=method, metric=metric, mask=mask, cmap=cmap,
                    annot=True, fmt=fmt, annot_kws={"fontsize": label_fontsize},
                    figsize=figsize, cbar_pos=(0.84, 0.33, 0.015, 0.32))

    # Axis labels and ticks
    g.ax_heatmap.set_xlabel(xlabel if xlabel is not None else df.columns.name or "", fontsize=axis_labelsize)
    g.ax_heatmap.set_ylabel(ylabel if ylabel is not None else df.index.name or "", fontsize=axis_labelsize)
    g.ax_heatmap.tick_params(axis="both", labelsize=tick_labelsize)

    # Colorbar label and ticks
    if g.cax and cbar_label:
        g.cax.set_ylabel(cbar_label, fontsize=cbar_labelsize)
        g.cax.tick_params(labelsize=cbar_labelsize)

    # Highlight diagonal if requested
    if symmetric_option == "highlight" and df.shape[0] == df.shape[1] and (df.columns == df.index).all():
        for i in range(len(df)):
            g.ax_heatmap.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="red", lw=1.5))

    # Save and optionally display
    plt.tight_layout()
    g.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")
    g.savefig(f"{filename}.svg", dpi=dpi, bbox_inches="tight")
    g.savefig(f"{filename}.pdf", dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_histogram(df, column, filename="histogram", bins=30, color="lightblue", dpi=600, show=True,
                   log_scale=False, log_y=False,
                   xlabel=None, ylabel="Frequency", title=None, fontsize=10, figsize=(8, 6),
                   fit_curve=False, curve_color="darkred", fit_normal=False):
    """
    Plot a histogram from a specified column in a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data
    - column: name of the column to plot
    - filename: base filename for saving the plot (default: 'histogram')
    - bins: number of bins in the histogram (default: 30)
    - color: fill color of the bars (default: 'lightblue')
    - dpi: resolution of saved plots (default: 600)
    - show: whether to display the plot after saving (default: True)
    - xlabel: label for the x-axis (default: column name)
    - ylabel: label for the y-axis (default: 'Frequency')
    - title: title of the plot (default: None)
    - fontsize: font size for labels and title (default: 10)
    - figsize: figure size (default: (8, 6))
    - fit_curve: if True, overlays a fitted normal distribution (default: False)
    - curve_color: color of the fitted curve (default: 'darkred')
    - fit_normal: if True, overlay a normal distribution curve (default: False)
    - log_scale: if True, use log scale on the x-axis (default: False)
    - log_y: if True, use log scale on the y-axis (default: False)
    """


    plt.figure(figsize=figsize)
    data = df[column].dropna()
    plt.hist(data, bins=bins, color=color, edgecolor="black", density=fit_curve, log=log_y)

    if fit_curve:
        
        x_range = np.linspace(data.min(), data.max(), 200)
        sns.kdeplot(data, color=curve_color, linewidth=2, clip=(data.min(), data.max()))

    if fit_normal:
        
        mu, std = norm.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(data.min(), data.max(), 200)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, color="black", linestyle="--", linewidth=1.5)
    if log_scale:
        plt.xscale("log")
    plt.xlabel(xlabel if xlabel else column, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    
    if show:
        plt.show()
    plt.close()


def plot_pairplot(df, columns=None, hue=None, filename="pairplot", dpi=600, show=True,
                  diag_kind="auto", palette="Set2", plot_kws=None, height=2.5):
    """
    Plot a pairplot (scatterplot matrix) from selected columns of a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to include (default: all numeric columns)
    - hue: optional column name for color encoding (categorical)
    - filename: base filename to save the plot (default: 'pairplot')
    - dpi: resolution of saved plots (default: 600)
    - show: whether to display the plot (default: True)
    - diag_kind: 'kde', 'hist', or 'auto' for diagonal plots
    - palette: color palette to use (default: 'Set2')
    - plot_kws: dictionary of keyword arguments for scatter plots
    - height: size of each subplot (default: 2.5)
    """

    data = df[columns] if columns else df.select_dtypes(include=["number"])
    g = sns.pairplot(data, hue=hue, diag_kind=diag_kind, palette=palette,
                     plot_kws=plot_kws or {}, height=height)
    g.fig.set_dpi(dpi)
    g.fig.tight_layout()
    g.savefig(f"{filename}.png", dpi=dpi)
    g.savefig(f"{filename}.svg", dpi=dpi)
    g.savefig(f"{filename}.pdf", dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    
def plot_wordcloud(df, filename="wordcloud", dpi=600, show=True,
                   background_color="white",
                   cbar_location="bottom", cbar_labelsize=10, cbar_ticksize=10,
                   item_col=None, size_col="Number of documents", color_by=None,
                   colormap="viridis", figsize=(10, 8), mask_image=None,
                   prefer_horizontal=1.0, top_n=None, top_n_by=None,
                   scale_func=None, layout_mode="archimedean"):
    """
    Plot a wordcloud from a DataFrame, using a numeric column to size the words
    and an optional column to define color groups.

    Parameters:
    - df: pandas DataFrame with at least one text and one numeric column
    - filename: base filename to save the plot (default: 'wordcloud')
    - dpi: resolution of saved image (default: 600)
    - show: whether to display the plot (default: True)
    - item_col: column with items to visualize (default: first column)
    - size_col: numeric column controlling word size (default: 'Number of documents')
    - color_by: column used to color words (categorical or numeric)
    - colormap: colormap used when color_by is numeric (default: 'viridis')
    - figsize: figure size (default: (10, 8))
    - mask_image: optional grayscale image for shaping the cloud (default: None)
    - prefer_horizontal: proportion of words drawn horizontally (0 to 1, default: 1.0)
    - cbar_location: 'bottom' (default), 'right', or 'none' to position/hide the colorbar
    - cbar_labelsize: font size for the colorbar label (default: 10)
    - cbar_ticksize: font size for the colorbar ticks (default: 10)
    - top_n: optional number of top items to include in the cloud (default: None)
    - scale_func: optional function to transform size values (e.g., np.log1p)
    - layout_mode: placement mode, e.g., 'archimedean', 'rectangular' (default: 'archimedean')
    - top_n_by: column to rank items by (default: None = same as size_col)
    - background_color: background color of the wordcloud (default: 'white')
    """
    


    item_col = item_col or df.columns[0]
    top_n_by = top_n_by or size_col

    if top_n is not None:
        df = df.sort_values(by=top_n_by, ascending=False).head(top_n)

    items = df[item_col].astype(str)
    sizes = df[size_col].astype(float)
    if scale_func is not None:
        sizes = scale_func(sizes)

    if color_by is None:
        color_map = {item: "gray" for item in items}
    else:
        color_values = df[color_by]
        if pd.api.types.is_numeric_dtype(color_values):
            norm = mcolors.Normalize(vmin=color_values.min(), vmax=color_values.max())
            cmap = cm.get_cmap(colormap)
            color_map = {item: mcolors.to_hex(cmap(norm(val))) for item, val in zip(items, color_values)}
            colorbar_type = "continuous"
        else:
            categories = pd.Categorical(color_values)
            palette = plt.get_cmap("tab10")
            colors = [mcolors.to_hex(palette(i)) for i in range(len(categories.categories))]
            color_map = {item: colors[i] for item, i in zip(items, categories.codes)}
            legend_labels = dict(zip(categories.categories, colors))
            colorbar_type = "categorical"

    frequencies = dict(zip(items, sizes))

    def color_func(word, **kwargs):
        return color_map.get(word, "gray")

    wc = WordCloud(width=1000, height=800, background_color=background_color,
                   prefer_horizontal=prefer_horizontal, random_state=42,
                   mask=mask_image, 
                   color_func=color_func).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    if color_by and colorbar_type == "categorical":
        from matplotlib.patches import Patch
        handles = [Patch(color=color, label=label) for label, color in legend_labels.items()]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(handles)))

    elif color_by and colorbar_type == "continuous":
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        if cbar_location != "none":
            orientation = "horizontal" if cbar_location == "bottom" else "vertical"
            cb = fig.colorbar(sm, ax=ax, orientation=orientation, fraction=0.046, pad=0.04)
            cb.set_label(color_by, fontsize=cbar_labelsize)
            cb.ax.tick_params(labelsize=cbar_ticksize)

    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    
    
def plot_treemap(df, filename="treemap", item_col=None, size_col="Number of documents", color_by=None,
                 cmap="viridis", figsize=(10, 6), dpi=600, show=True,
                 sort=True, ascending=False, label_fontsize=10, min_label_size_ratio=0.01,
                 wrap_width=15, show_frame=False, min_fontsize=6, max_fontsize=28, log_scale=False):
    """
    Plot a treemap based on a DataFrame with item sizes and optional color mapping.

    Parameters:
    - df: pandas DataFrame containing the data
    - filename: base filename for saved plots (default: 'treemap')
    - item_col: name of the column with item labels (default: first column)
    - size_col: name of the column that controls the area size of each item
    - color_by: optional column for color mapping (categorical or numerical)
    - cmap: colormap name used for continuous color scale (default: 'viridis')
    - figsize: figure size (default: (10, 6))
    - dpi: resolution of output image (default: 600)
    - show: whether to display the plot after saving
    - sort: whether to sort items by size (default: True)
    - ascending: sorting order (default: False for descending)
    - label_fontsize: font size of item labels inside treemap (default: 10)
    - min_label_size_ratio: minimum size proportion to draw label (default: 0.01)
    - wrap_width: character width for wrapped labels (default: 15)
    - show_frame: whether to show frames around boxes (default: False)
    - min_fontsize: minimum font size for labels (default: 6)
    - max_fontsize: maximum font size for labels (default: 20)
    - log_scale: whether to apply log scaling to size values for font size and box area (default: False)
    """


    item_col = item_col or df.columns[0]
    if sort:
        df = df.sort_values(by=size_col, ascending=ascending)
    sizes = df[size_col]
    labels = ["\n".join(textwrap.wrap(str(label), width=wrap_width)) if size >= sizes.sum() * min_label_size_ratio else ""
              for label, size in zip(df[item_col], sizes)]

    def infer_color_scheme(values):
        if pd.api.types.is_numeric_dtype(values):
            return "continuous"
        else:
            return "categorical"

    if color_by is None:
        color_scheme = "default"
        colors = ["lightblue"] * len(df)
        colorbar_type = None

    color_scheme = infer_color_scheme(df[color_by]) if color_by else "default"

    if color_scheme == "default":
        colors = ["lightblue"] * len(df)
        colorbar_type = None
    else:
        values = df[color_by]
        if pd.api.types.is_numeric_dtype(values):
            norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
            cmap = cm.get_cmap(cmap)
            colors = [cmap(norm(val)) for val in values]
            colorbar_type = "continuous"
        else:
            categories = pd.Categorical(values)
            palette = plt.get_cmap("tab10")
            colormap = [palette(i) for i in range(len(categories.categories))]
            colors = [colormap[i] for i in categories.codes]
            legend_labels = dict(zip(categories.categories, colormap))
            colorbar_type = "categorical"

    fig, ax = plt.subplots(figsize=figsize)
    if log_scale:
        scaled_sizes = np.log1p(sizes)
    else:
        scaled_sizes = sizes
    scaled_fonts = [max(min_fontsize, min(max_fontsize, label_fontsize * (s / max(scaled_sizes)))) for s in scaled_sizes]

    normed_sizes = squarify.normalize_sizes(scaled_sizes, 100, 100)
    boxes = squarify.squarify(normed_sizes, 0, 0, 100, 100)
    drawn_boxes = squarify.plot(sizes=scaled_sizes, label=["" for _ in labels], color=colors, alpha=0.8, ax=ax,
                                 text_kwargs={}, bar_kwargs={"linewidth": 0.5 if show_frame else 0, "edgecolor": "black"})

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for label, box, fontsize in zip(labels, boxes, scaled_fonts):
        if label:
            x = box['x'] + box['dx'] / 2
            y = box['y'] + box['dy'] / 2
            ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, clip_on=True)
    ax.axis("off")

    if color_by and colorbar_type == "categorical":       
        handles = [Patch(color=color, label=str(label)) for label, color in legend_labels.items()]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(handles)))
    elif color_by and colorbar_type == "continuous":
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, label=color_by)

    plt.tight_layout()
    save_plot(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_boxplot(
    df, value_column, group_by=None, group_matrix=None, min_group_size=5, max_groups=None,
    figsize=(10, 6), title=None, x_label_size=14, y_label_size=14, tick_label_size=12,
    value_label=None, filename_base=None, dpi=600, group_order_user=None, order_by_size=False,
    show_counts=False, return_summary=False, label_angle=90, stat_test=False, wrap_width=30,
    show=True, group_colors=None  # renamed and used for palette
):
    """
    Plot a boxplot of a numerical column grouped by either a column in df or a binary group matrix.

    ...
    group_colors : dict, optional
        Dictionary mapping group names to specific colors.
    """

    if group_by is not None and group_matrix is not None:
        raise ValueError("Specify only one of group_by or group_matrix.")

    # Prepare data
    if group_by is not None:
        group_sizes = df[group_by].value_counts()
        valid_groups = group_sizes[group_sizes >= min_group_size].index.tolist()
        if max_groups is not None:
            valid_groups = valid_groups[:max_groups]
        plot_data = df[df[group_by].isin(valid_groups)]
        group_order = group_order_user if group_order_user is not None else plot_data[group_by].value_counts().index.tolist()
        counts = plot_data[group_by].value_counts() if show_counts else None
        x = group_by
    elif group_matrix is not None:
        melted = []
        for group_name in group_matrix.columns:
            indices = group_matrix.index[group_matrix[group_name] == 1]
            values = df.loc[indices, value_column]
            if values.isna().all():
                continue
            values = values.dropna()
            if len(values) >= min_group_size:
                melted.append(pd.DataFrame({value_column: values, 'Group': group_name}))
        if not melted:
            raise ValueError("No groups met the minimum size requirement.")
        plot_data = pd.concat(melted)
        if max_groups is not None:
            group_counts = plot_data['Group'].value_counts()
            valid_groups = group_counts.head(max_groups).index
            plot_data = plot_data[plot_data['Group'].isin(valid_groups)]
        if group_order_user is not None:
            group_order = [g for g in group_order_user if g in plot_data['Group'].unique()]
        elif order_by_size:
            group_order = plot_data['Group'].value_counts().index.tolist()
        else:
            group_order = [col for col in group_matrix.columns if col in plot_data['Group'].unique()]
        counts = plot_data['Group'].value_counts() if show_counts else None
        x = 'Group'
    else:
        raise ValueError("One of group_by or group_matrix must be specified.")

    # Set colors
    if group_colors:
        palette = {g: group_colors.get(g, '#cccccc') for g in group_order}
    else:
        palette = get_colors(len(group_order), color_scheme="categorical")

    # Plotting
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=plot_data,
        x=x,
        y=value_column,
        hue=x,            # Fix for deprecation
        order=group_order,
        palette=palette,
        legend=False      # Fix for deprecation
    )

    plt.xlabel(group_by if group_by is not None else "Group", fontsize=x_label_size)
    plt.ylabel(value_label if value_label is not None else value_column, fontsize=y_label_size)
    if show_counts and counts is not None:
        group_order_labels = [f"{g} (n={counts[g]})" for g in group_order]
    else:
        group_order_labels = group_order
    plt.xticks(
        ticks=range(len(group_order)),
        labels=wrap_labels(group_order_labels, width=wrap_width),
        rotation=label_angle,
        ha='right',
        fontsize=tick_label_size
    )
    plt.yticks(fontsize=tick_label_size)

    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title("")

    plt.grid(False)
    plt.tight_layout()

    # Statistical test (Kruskal-Wallis)
    if stat_test and len(group_order) > 1:
        groups = [plot_data[plot_data[x] == grp][value_column].dropna() for grp in group_order]
        stat, pval = kruskal(*groups)
        plt.figtext(0.99, 0.01, f"Kruskal-Wallis p = {pval:.3g}", horizontalalignment='right', fontsize=12)

    if filename_base:
        save_plot(filename_base, dpi=dpi)

    if show:
        plt.show()

    if return_summary:
        return plot_data.groupby(x)[value_column].describe()



    

def plot_violinplot(
    df, value_column, group_by=None, group_matrix=None, min_group_size=5, max_groups=None,
    figsize=(10, 6), title=None, x_label_size=14, y_label_size=14, tick_label_size=12,
    value_label=None, filename_base=None, dpi=600, group_order_user=None, order_by_size=False,
    show_counts=False, return_summary=False, label_angle=90, stat_test=False, wrap_width=30,
    show=True, group_colors=None
):
    """
    Plot a violin plot of a numerical column grouped by either a column in df or a binary group matrix.

    ...
    group_colors : dict, optional
        Dictionary mapping group names to specific colors.
    """
    if group_by is not None and group_matrix is not None:
        raise ValueError("Specify only one of group_by or group_matrix.")

    if group_by is not None:
        group_sizes = df[group_by].value_counts()
        valid_groups = group_sizes[group_sizes >= min_group_size].index.tolist()
        if max_groups is not None:
            valid_groups = valid_groups[:max_groups]
        plot_data = df[df[group_by].isin(valid_groups)]
        group_order = group_order_user if group_order_user is not None else plot_data[group_by].value_counts().index.tolist()
        counts = plot_data[group_by].value_counts() if show_counts else None
        x = group_by
    elif group_matrix is not None:
        melted = []
        for group_name in group_matrix.columns:
            indices = group_matrix.index[group_matrix[group_name] == 1]
            values = df.loc[indices, value_column]
            if values.isna().all():
                continue
            values = values.dropna()
            if len(values) >= min_group_size:
                melted.append(pd.DataFrame({value_column: values, 'Group': group_name}))
        if not melted:
            raise ValueError("No groups met the minimum size requirement.")
        plot_data = pd.concat(melted)
        if max_groups is not None:
            group_counts = plot_data['Group'].value_counts()
            valid_groups = group_counts.head(max_groups).index
            plot_data = plot_data[plot_data['Group'].isin(valid_groups)]
        if group_order_user is not None:
            group_order = [g for g in group_order_user if g in plot_data['Group'].unique()]
        elif order_by_size:
            group_order = plot_data['Group'].value_counts().index.tolist()
        else:
            group_order = [col for col in group_matrix.columns if col in plot_data['Group'].unique()]
        counts = plot_data['Group'].value_counts() if show_counts else None
        x = 'Group'
    else:
        raise ValueError("One of group_by or group_matrix must be specified.")

    # Set colors
    if group_colors:
        palette = {g: group_colors.get(g, '#cccccc') for g in group_order}
    else:
        palette = get_colors(len(group_order), color_scheme="categorical")

    plt.figure(figsize=figsize)
    sns.violinplot(
        data=plot_data,
        x=x,
        y=value_column,
        hue=x,              # <- Fix for deprecation
        order=group_order,
        palette=palette,
        legend=False        # <- Fix for deprecation
    )

    plt.xlabel(group_by if group_by is not None else "Group", fontsize=x_label_size)
    plt.ylabel(value_label if value_label is not None else value_column, fontsize=y_label_size)
    if show_counts and counts is not None:
        group_order_labels = [f"{g} (n={counts[g]})" for g in group_order]
    else:
        group_order_labels = group_order
    plt.xticks(
        ticks=range(len(group_order)),
        labels=wrap_labels(group_order_labels, width=wrap_width),
        rotation=label_angle,
        ha='right',
        fontsize=tick_label_size
    )
    plt.yticks(fontsize=tick_label_size)

    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title("")

    plt.grid(False)
    plt.tight_layout()

    if stat_test and len(group_order) > 1:
        groups = [plot_data[plot_data[x] == grp][value_column].dropna() for grp in group_order]
        stat, pval = kruskal(*groups)
        plt.figtext(0.99, 0.01, f"Kruskal-Wallis p = {pval:.3g}", horizontalalignment='right', fontsize=12)

    if filename_base:
        save_plot(filename_base, dpi=dpi)

    if show:
        plt.show()

    if return_summary:
        return plot_data.groupby(x)[value_column].describe()





def plot_group_distributions_aligned(df, numerical_cols, group_matrix, bins=30, alpha=0.7, 
                                     save=False, filename_prefix="group_dist", dpi=600, 
                                     show_grid=False, group_colors={}):
    """
    Plot histograms of numerical variables for each group defined in a binary matrix,
    with one subplot per group, aligned on a shared x-axis.

    Parameters:
    df (pd.DataFrame): DataFrame containing the numerical data.
    numerical_cols (list of str): List of numerical column names to plot.
    group_matrix (pd.DataFrame): Binary matrix where each column is a group with 0/1 values.
    bins (int): Number of bins for histograms (default: 30).
    alpha (float): Transparency for histogram fill (default: 0.7).
    save (bool): Whether to save the plot to file.
    filename_prefix (str): Prefix for saved plot filenames (default: 'group_dist').
    dpi (int): DPI resolution for saving plots (default: 600).
    show_grid (bool): Whether to show gridlines (default: False).
    group_colors (dict): Optional dict mapping group names to colors (default: {}).
    """
    default_color = "skyblue"

    for col in numerical_cols:
        num_groups = len(group_matrix.columns)
        fig, axes = plt.subplots(num_groups, 1, figsize=(8, 3 * num_groups), sharex=True)
        if num_groups == 1:
            axes = [axes]

        global_min = df[col][np.isfinite(df[col])].min()
        global_max = df[col][np.isfinite(df[col])].max()
        bin_edges = np.linspace(global_min, global_max, bins + 1)

        for ax, group in zip(axes, group_matrix.columns):
            mask = group_matrix[group] == 1
            data = df.loc[mask, col]
            data = data[np.isfinite(data)]

            color = group_colors.get(group, default_color)

            ax.hist(data, bins=bin_edges, alpha=alpha, color=color, edgecolor="black", density=True)
            ax.set_title(f"{col} - {group}")
            ax.set_ylabel("Density")
            if show_grid:
                ax.grid(True, linestyle="--", alpha=0.6)

        axes[-1].set_xlabel(col)
        plt.tight_layout()

        if save:
            save_plot(f"{filename_prefix}_{col}", dpi=dpi)

        plt.show()



def plot_group_venn(group_matrix: pd.DataFrame, title: str = None, filename: str = None, dpi: int = 600, include_totals: bool = True, show: bool = True, save_results: bool = True, group_color: dict = None, alpha: float = 0.5, **kwargs):
    """
    Plots a Venn diagram for 2–6 groups using the 'venn' package.
    Adds a compact legend in the top-right corner with matched colors, transparency, and outlines.

    Parameters:
    - group_matrix: pd.DataFrame
        Binary matrix (0/1), rows = items, columns = groups.
    - title: str
        Title of the plot.
    - filename: str or None
        If given, saves plot using save_plot(filename, dpi).
    - dpi: int
        Resolution of saved figures.
    - include_totals: bool
        If True, appends (n=...) to group labels.
    - show: bool
        If True, displays the plot.
    - save_results: bool
        If True, calls save_plot().
    - group_color: dict or None
        Optional dictionary mapping group names to colors.
    - alpha: float
        Transparency level for region fills.
    - **kwargs: dict
        Passed directly to venn(), e.g., cmap.

    Requirements:
    - venn (https://pypi.org/project/venn/)
    """


    sets_raw = {col: set(group_matrix.index[group_matrix[col] == 1]) for col in group_matrix.columns}

    if include_totals:
        label_map = {col: f"{col} (n={len(val)})" for col, val in sets_raw.items()}
        labeled_sets = {label_map[col]: sets_raw[col] for col in group_matrix.columns}
    else:
        label_map = {col: col for col in sets_raw}
        labeled_sets = sets_raw

    # Group names and final labels
    group_names = list(group_matrix.columns)
    label_names = [label_map[g] for g in group_names]

    # Assign colors
    if group_color:
        color_cycle = [group_color.get(g, "gray") for g in group_names]
    else:
        color_iter = itertools.cycle(rcParams["axes.prop_cycle"].by_key()["color"])
        color_cycle = [next(color_iter) for _ in group_names]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    venn(labeled_sets, ax=ax, **kwargs)

    # Apply color, transparency, and edge styling
    patch_handles = []
    for label, color, patch in zip(label_names, color_cycle, ax.patches):
        if patch is not None:
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)
            patch_handles.append(Patch(facecolor=color, edgecolor="black", label=label, alpha=alpha))

    ax.legend(
        handles=patch_handles,
        title="Groups",
        loc="upper right",
        fontsize="small",
        title_fontsize="small",
        frameon=True,
        borderpad=0.5,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.5
    )

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if filename and save_results:
        save_plot(filename, dpi=dpi)

    if show:
        plt.show()


def plot_group_upset(group_matrix: pd.DataFrame, title: str = None, filename: str = None, dpi: int = 600, show: bool = True, save_results: bool = True, group_color: dict = None, **kwargs):
    """
    Plots an UpSet diagram showing set intersections based on a binary group membership matrix.

    Parameters:
    - group_matrix: pd.DataFrame
        Binary matrix (0/1), rows = items (e.g., documents), columns = group names.
    - title: str
        Title for the figure.
    - filename: str or None
        Base name for saving PNG, SVG, and PDF via save_plot.
    - dpi: int
        Resolution of saved figures.
    - show: bool
        Whether to display the figure.
    - save_results: bool
        Whether to save the figure using save_plot.
    - group_color: dict or None
        Optional dictionary mapping group names to specific colors.
    - **kwargs: dict
        Additional keyword arguments passed to the UpSet constructor.

    Requirements:
    - upsetplot (https://github.com/jnothman/UpSetPlot)
    """
    try:
        import upsetplot
        import itertools
    except ImportError:
        print("The 'upsetplot' package is required for this function. Install it via 'pip install upsetplot'.")
        return

    # Prepare the UpSet data
    data = from_indicators(group_matrix.columns.tolist(), group_matrix.astype(bool))

    # Color logic
    group_names = list(group_matrix.columns)
    if group_color:
        color_cycle = [group_color.get(g, "gray") for g in group_names]
    else:
        color_iter = itertools.cycle(rcParams["axes.prop_cycle"].by_key()["color"])
        color_cycle = [next(color_iter) for _ in group_names]

    # Default UpSet kwargs (overridable by user)
    default_kwargs = {
        "show_counts": True,
        "show_percentages": False,
        "sort_categories_by": "cardinality",
        "intersection_plot_elements": 20
    }
    default_kwargs.update(kwargs)

    # Plot
    fig = plt.figure(figsize=(9, 6))
    upset = UpSet(data, **default_kwargs)
    upset.plot(fig=fig)

    # Apply colors to group totals bar chart (first subplot)
    ax_cat_totals = fig.axes[0]
    for bar, color in zip(ax_cat_totals.patches, color_cycle):
        bar.set_facecolor(color)

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if filename and save_results:
        save_plot(filename, dpi=dpi)

    if show:
        plt.show()


def plot_group_heatmap(group_matrix: pd.DataFrame,
                       methods: list = ["jaccard"],
                       title: str = None,
                       filename: str = "group_heatmap",
                       dpi: int = 600,
                       group_color: dict = None,
                       color_ticks: bool = False,
                       show: bool = True,
                       save_results: bool = True,
                       save_csv: bool = False,
                       **kwargs):
    """
    Computes and plots group × group heatmaps using various similarity/distance measures.

    Parameters:
    - group_matrix: pd.DataFrame
        Binary matrix (0/1) with rows = items and columns = group names.
    - methods: list of str
        Measures to compute. Supported: 'jaccard', 'count', 'sokal-michener', 'simple-matching', 'rogers-tanimoto'.
    - title: str
        Title prefix for the plots.
    - filename: str
        Base filename for saving output images and CSVs.
    - dpi: int
        Resolution for saved images.
    - group_color: dict
        Optional group-to-color mapping.
    - color_ticks: bool
        If True, apply group_color to tick labels.
    - show: bool
        Whether to display the plot.
    - save_results: bool
        Whether to save output image files.
    - save_csv: bool
        Whether to save the computed matrix to a CSV file.
    - **kwargs:
        Passed through to plot_heatmap, e.g. cmap, label_fontsize, wrap_width, symmetric_option, etc.
    """
    matrices = utilsbib.compute_group_similarity_matrices(group_matrix, methods)
    labels = {
        "jaccard": "Jaccard Index",
        "count": "Shared Items",
        "sokal-michener": "Sokal-Michener",
        "simple-matching": "Simple Matching",
        "rogers-tanimoto": "Rogers-Tanimoto"
    }

    for method, mat in matrices.items():
        normalized = method != "count"
        cbar_label = labels[method]
        fname = f"{filename}_{method}"
        ttl = f"{title or 'Group Similarity'} ({method})"

        plot_heatmap(df=mat,
                     filename=fname,
                     dpi=dpi,
                     show=show,
                     normalized=normalized,
                     cbar_label=cbar_label,
                     xlabel="",
                     ylabel="",
                     **kwargs)

        if save_csv:
            csv_filename = f"{fname}.csv"
            mat.to_csv(csv_filename)

        if color_ticks and group_color:
            ax = plt.gca()
            for label in ax.get_xticklabels():
                label.set_color(group_color.get(label.get_text(), "black"))
            for label in ax.get_yticklabels():
                label.set_color(group_color.get(label.get_text(), "black"))
            plt.draw()

# STILL NEEDS TO BE FIEXD
def plot_group_chord(matrix: pd.DataFrame, threshold: float = 0.0, group_color: dict = None, title: str = None, filename: str = None, dpi: int = 600, show: bool = True):
    """
    Plots a chord diagram from a group × group similarity matrix.

    Parameters:
    - matrix: pd.DataFrame
        A symmetric similarity matrix (e.g., from compute_group_similarity_matrices).
    - threshold: float
        Minimum value to include a connection (default: 0.0).
    - group_color: dict
        Optional color dictionary for each group.
    - title: str
        Optional title for the figure.
    - filename: str
        Base filename to save the diagram.
    - dpi: int
        Resolution for saving the figure.
    - show: bool
        Whether to show the plot.
    """


    hv.extension("matplotlib")

    # Ensure all labels are strings
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    if group_color:
        group_color = {str(k): v for k, v in group_color.items()}

    # Extract upper triangle (no self or duplicate edges)
    links = []
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            if j <= i:
                continue
            weight = matrix.iloc[i, j]
            if weight >= threshold:
                links.append((row, col, weight))

    if not links:
        print("No links to plot above threshold.")
        return None

    # Create dataset for chord with explicit kdims and vdims
    chord_data = hv.Dataset(pd.DataFrame(links, columns=["source", "target", "value"]),
                            kdims=["source", "target"],
                            vdims=["value"])
    chord = hv.Chord(chord_data)

    # Configure options
    chord_opts = dict(
        edge_color="source",
        node_color="index",
        show_legend=False
    )
    if group_color:
        chord_opts["cmap"] = list(group_color.values())

    chord = chord.select(value=(threshold, None)).opts(opts.Chord(**chord_opts))

    # Plot using Holoviews and capture the figure
    fig = hv.render(chord, backend="matplotlib")
    fig.set_size_inches(8, 8)

    # Add manual labels
    ax = fig.axes[0]
    labels = matrix.columns.tolist()
    num_labels = len(labels)
    radius = 1.2
    angle_step = 2 * np.pi / num_labels
    for i, label in enumerate(labels):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        ha = "left" if np.cos(angle) > 0 else "right"
        va = "bottom" if np.sin(angle) > 0 else "top"
        color = group_color.get(label, "black") if group_color else "black"
        ax.text(x, y, label, ha=ha, va=va, fontsize=9, color=color, rotation=np.degrees(angle), rotation_mode="anchor")

    if title:
        ax.set_title(title)

    if filename:
        fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(f"{filename}.svg", bbox_inches="tight")
        fig.savefig(f"{filename}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return chord


# dodaj MDS plot za skupine


"""
def compute_group_similarity_matrices(group_matrix: pd.DataFrame, methods: list = ["jaccard"]) -> dict:
    ...


def plot_group_heatmap(group_matrix: pd.DataFrame,
                       methods: list = ["jaccard"],
                       title: str = None,
                       filename: str = "group_heatmap",
                       dpi: int = 600,
                       group_color: dict = None,
                       color_ticks: bool = False,
                       show: bool = True,
                       save_results: bool = True,
                       save_csv: bool = False,
                       **kwargs):
    ...
"""

def plot_group_chord(matrix: pd.DataFrame, threshold: float = 0.0, group_color: dict = None, title: str = None, filename: str = None, dpi: int = 600, show: bool = True):
    pass

def plot_group_dendrogram(group_matrix: pd.DataFrame, method: str = "average", metric: str = "euclidean", title: str = None, filename: str = None, dpi: int = 600, show: bool = True):
    """
    Plots a dendrogram (hierarchical clustering) from a binary group membership matrix.

    Parameters:
    - group_matrix: pd.DataFrame
        A binary indicator matrix (0/1), rows = items, columns = group names.
    - method: str
        Linkage method (e.g., "average", "complete", "single").
    - metric: str
        Distance metric (e.g., "euclidean", "jaccard", etc.).
    - title: str
        Plot title.
    - filename: str
        If given, saves the dendrogram to this base filename (png, svg, pdf).
    - dpi: int
        Resolution for saved image.
    - show: bool
        Whether to display the plot.
    """


    # Ensure labels are strings
    group_matrix.columns = group_matrix.columns.astype(str)

    # Compute pairwise distances between columns (groups)
    dist_matrix = pdist(group_matrix.T, metric=metric)
    linkage = sch.linkage(dist_matrix, method=method)

    fig, ax = plt.subplots(figsize=(10, 6))
    sch.dendrogram(linkage, labels=group_matrix.columns.tolist(), leaf_rotation=90, leaf_font_size=10, ax=ax)

    ax.set_ylabel("Distance")
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if filename:
        def save_plot(filename_base, dpi=600):
            """Save plot to PNG, SVG, and PDF with tight layout and given DPI."""
            for ext in ["png", "svg", "pdf"]:
                path = f"{filename_base}.{ext}"
                plt.savefig(path, bbox_inches="tight", dpi=dpi)

        save_plot(filename, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)



def plot_top_items_by_group(df: pd.DataFrame,
                             top_n: int = 5,
                             value_column_pattern: str = "Number of documents",
                             title: str = None,
                             filename: str = None,
                             dpi: int = 600,
                             group_color: dict = None,
                             show_values: bool = True,
                             reverse_order: bool = False,
                             show: bool = True):
    """
    Plots a horizontal bar chart showing top N items per group.

    Parameters:
    - df: pd.DataFrame
        DataFrame with item names in first column and one or more group-specific value columns.
    - top_n: int
        Number of top items to show per group (default: 5).
    - value_column_pattern: str
        Pattern prefix of value columns to plot (default: "Number of documents").
    - title: str
        Title of the plot.
    - filename: str
        Base filename to save the figure (PNG, SVG, PDF).
    - dpi: int
        Save resolution.
    - group_color: dict
        Optional color mapping for groups.
    - show_values: bool
        If True, annotate bars with their values.
    - reverse_order: bool
        If True, reverse the sort order of bars.
    - show: bool
        Whether to display the figure.
    """


    item_col = df.columns[0]
    value_cols = [col for col in df.columns if col.startswith(value_column_pattern)]

    records = []
    for col in value_cols:
        group = col.split("(")[-1].rstrip(")").strip()
        temp_df = df[[item_col, col]].copy()
        temp_df.columns = ["Item", "Value"]
        temp_df["Group"] = group
        temp_df = temp_df[temp_df["Value"] > 0]
        top_items = temp_df.sort_values("Value", ascending=False).head(top_n)

        # Handle possible ties
        min_val = top_items["Value"].min()
        all_top = temp_df[temp_df["Value"] >= min_val]
        all_top = all_top.copy()
        all_top["RankGroup"] = group
        records.append(all_top)

    plot_df = pd.concat(records)
    plot_df = plot_df.sort_values(by=["RankGroup", "Value"], ascending=[True, not reverse_order])

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(plot_df) + 1))

    # Assign default group colors if none provided
    if group_color is None:
        palette = itertools.cycle(cm.tab10.colors)
        unique_groups = plot_df["Group"].unique()
        group_color = {group: next(palette) for group in unique_groups}

    colors = plot_df["Group"].map(group_color)

    counts = Counter(plot_df["Item"])
    labels = []
    seen = Counter()
    for item in plot_df["Item"]:
        label = item
        if counts[item] > 1:
            label += " " * seen[item]  # add space padding for repeated items
            seen[item] += 1
        labels.append(label)
    bars = ax.barh(labels, plot_df["Value"], color=colors)

    if show_values:
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f"{width:.0f}", va="center", ha="left", fontsize=9)

    ax.set_xlabel(value_column_pattern)
    if title:
        ax.set_title(title)

    # Add legend
    handles = [plt.Line2D([0], [0], color=group_color[g], lw=6) for g in group_color]
    ax.legend(handles, group_color.keys(), title="Groups")

    plt.tight_layout()

    if filename:
        def save_plot(filename_base, dpi=600):
            """Save plot to PNG, SVG, and PDF with tight layout and given DPI."""
            for ext in ["png", "svg", "pdf"]:
                path = f"{filename_base}.{ext}"
                plt.savefig(path, bbox_inches="tight", dpi=dpi)
        save_plot(filename, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)
    
code_to_coords = utilsbib.code_to_coords

def save_plotly_choropleth_map(
    df,
    value_col,
    filename_prefix=None,
    iso_col="ISO-3",
    width=1000,
    height=600,
    scale=2,
    title=None,
    projection="natural earth",
    scope="world",
    font_size=12,
    title_font_size=16,
    dark_mode=False,
    hover_name=None,
    hover_data=None,
    colormap="Viridis",
    links_df=None,
    link_weight_col="weight",
    link_color="red",
    link_opacity=0.5,
    timeout_seconds=30,
    continent_col="Continent",
    **kwargs
):
    """
    Generate and optionally save a choropleth map in PDF, PNG, SVG, and HTML formats using Plotly,
    with optional inter-country collaboration lines, using fig.to_image() to avoid blocking.
    If a column specifying continent exists and a scope is provided (other than "world"),
    filter the DataFrame to that continent before plotting.

    Args:
        df (pd.DataFrame): DataFrame containing country data.
        value_col (str): Column name with values to visualize.
        filename_prefix (str or None): Prefix for saved file names. If None, the plot is shown but not saved.
        iso_col (str): Column name with ISO-3 country codes.
        width (int): Width of the saved image.
        height (int): Height of the saved image.
        scale (int): Scale factor for image resolution.
        title (str): Optional title for the map.
        projection (str): Map projection (e.g., "natural earth", "equirectangular").
        scope (str): Geographic scope to focus the map (e.g., "world", "europe", "asia").
        font_size (int): Font size for general map text.
        title_font_size (int): Font size for the map title.
        dark_mode (bool): Use dark template if True.
        hover_name (str): Column to use as hover label.
        hover_data (list): List of columns to display on hover.
        colormap (str): Colormap name for continuous data.
        links_df (pd.DataFrame): DataFrame with "source", "target", and weight column for collaboration links.
        link_weight_col (str): Column name representing collaboration strength.
        link_color (str): Color of the collaboration lines.
        link_opacity (float): Opacity of the collaboration lines.
        timeout_seconds (int): Max seconds to wait per image format.
        continent_col (str): Name of the continent column (default: "continent").
    **kwargs: Additional keyword arguments for px.choropleth.
    """


    template = "plotly_dark" if dark_mode else "plotly"

    # Check for continent_col and filter
    col_match = None
    for col in df.columns:
        if col.lower() == continent_col.lower():
            col_match = col
            break

    if col_match and scope.lower() != "world":
        matching = df[col_match].astype(str).str.lower() == scope.lower()
        if matching.any():
            df = df[matching]
        else:
            warnings.warn(
                f"Scope '{scope}' provided but no matching values found in '{col_match}' column. Proceeding without filtering."
            )

    fig = px.choropleth(
        df,
        locations=iso_col,
        color=value_col,
        locationmode="ISO-3",
        color_continuous_scale=colormap,
        title=title,
        hover_name=hover_name,
        hover_data=hover_data,
        projection=projection,
        scope=scope,
        template=template,
        **kwargs
    )

    # Collaboration links (assumes code_to_coords dict is in global scope)
    if links_df is not None:
        for _, row in links_df.iterrows():
            src, tgt = row["source"], row["target"]
            if src not in code_to_coords or tgt not in code_to_coords:
                continue
            lat0, lon0 = code_to_coords[src]["latitude"], code_to_coords[src]["longitude"]
            lat1, lon1 = code_to_coords[tgt]["latitude"], code_to_coords[tgt]["longitude"]
            fig.add_trace(
                go.Scattergeo(
                    lon=[lon0, lon1],
                    lat=[lat0, lat1],
                    mode="lines",
                    line=dict(width=row[link_weight_col], color=link_color),
                    opacity=link_opacity,
                    showlegend=False
                )
            )

    fig.update_layout(
        font=dict(size=font_size),
        title_font=dict(size=title_font_size),
        coloraxis_colorbar=dict(
            title=dict(text=value_col, font=dict(size=font_size + 2)),
            ticks="outside",
            ticklen=5,
            tickcolor="#000",
            tickfont=dict(size=font_size),
        )
    )

    # Save or show
    if filename_prefix:
        for fmt in ("pdf", "png", "svg"):
            try:
                img_bytes = fig.to_image(
                    format=fmt,
                    width=width,
                    height=height,
                    scale=scale,
                    engine="kaleido"
                )
                with open(f"{filename_prefix}.{fmt}", "wb") as f:
                    f.write(img_bytes)
            except Exception as e:
                print(f"Error saving {fmt.upper()}: {e}")

        try:
            fig.write_html(f"{filename_prefix}.html")
        except Exception as e:
            print(f"Error saving HTML: {e}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.show()

    return fig

# Country collaboration plots

def plot_top_country_pairs(matrix_df, top_n=20, figsize=(10, 6), filename_base=None):
    """
    Plots a horizontal barplot of the top N collaborating country pairs, including ties at the cutoff.

    Parameters:
    matrix_df (pd.DataFrame): Symmetric collaboration matrix.
    top_n (int): Minimum number of top collaborating pairs to plot (ties at the cutoff are included).
    figsize (tuple): Size of the figure in inches.
    filename_base (str or None): If provided, saves the plot as PNG, SVG, and PDF using this base name.
    """
    if matrix_df.empty:
        print("Empty matrix: barplot not generated.")
        return

    pair_data = []
    for i in matrix_df.index:
        for j in matrix_df.columns:
            if i < j:
                count = matrix_df.loc[i, j]
                if count > 0:
                    pair_data.append((f"{i} – {j}", count))

    if not pair_data:
        print("No collaboration pairs found: barplot not generated.")
        return

    # Sort and find cutoff for ties
    pair_data_sorted = sorted(pair_data, key=lambda x: x[1], reverse=True)
    if len(pair_data_sorted) > top_n:
        cutoff_value = pair_data_sorted[top_n - 1][1]
        top_pairs = [pair for pair in pair_data_sorted if pair[1] >= cutoff_value]
    else:
        top_pairs = pair_data_sorted

    labels, values = zip(*top_pairs)

    plt.figure(figsize=figsize)
    plt.barh(labels, values)
    plt.xlabel("Collaboration Count")
    plt.title(f"Top Country Collaborations (≥ Top {top_n} with ties)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base)

    plt.show()


# production over time

def plot_group_dot_grid(df,
                        result_type="wide",
                        value_column="Number of documents",
                        color_column=None,
                        top_n=10,
                        year_column="Year",
                        group_column="Group",
                        figsize=(12, 8),
                        cmap="viridis",
                        output_path=None,
                        dpi=300,
                        max_dot_size=600,
                        font_size=14,
                        wrap_labels=True,
                        ylabel_label="Group",
                        year_span=None):
    """
    Plot a dot grid showing group activity over time with dot size = value and color = additional metric.

    Parameters
    ----------
    df : pd.DataFrame
        Output of aggregate_bibliometrics_by_group_and_year (wide or long format).
    result_type : str, default "wide"
        Type of df provided: "wide" or "long".
    value_column : str
        Column to determine dot size (usually "Number of documents").
    color_column : str or None
        Optional column to control dot color (e.g., citations, funding).
    top_n : int
        Show top N groups based on total of value_column.
    year_column : str
        Column indicating time axis.
    group_column : str
        Column indicating grouping (one row per group).
    figsize : tuple
        Size of the figure.
    cmap : str
        Matplotlib colormap name.
    output_path : str or None
        If provided, save the figure to this path.
    dpi : int
        Resolution for saved figure.
    max_dot_size : int
        Maximum dot area size in points^2.
    font_size : int
        Font size for labels.
    wrap_labels : bool
        If True, wrap long y-axis labels.
    ylabel_label : str
        Label for the y-axis.
    year_span : tuple or None
        Optional (start_year, end_year) to define time span. If None, use actual range based on non-zero values.

    Returns
    -------
    None
    """
    if result_type == "long":
        if not {"Metric", "Value"}.issubset(df.columns):
            raise ValueError("Expected columns 'Metric' and 'Value' in long format.")
        df = df.pivot_table(index=[year_column, group_column], columns="Metric", values="Value").reset_index()

    # Determine year span from non-zero values if not provided
    if year_span is None:
        nonzero_df = df[df[value_column] > 0]
        min_year = int(nonzero_df[year_column].min())
        max_year = int(nonzero_df[year_column].max())
        year_span = (min_year, max_year)

    # Apply year filtering early
    df = df[(df[year_column] >= year_span[0]) & (df[year_column] <= year_span[1])]

    # Filter top N groups by value_column
    group_totals = df.groupby(group_column)[value_column].sum().nlargest(top_n)
    df = df[df[group_column].isin(group_totals.index)]

    # Map groups to vertical positions (reverse order)
    group_order = group_totals.index.tolist()[::-1]
    group_pos = {name: i for i, name in enumerate(group_order)}
    df["_y"] = df[group_column].map(group_pos)

    # Scale dot sizes
    max_value = df[value_column].max()
    df["_dot_size"] = df[value_column] / max_value * max_dot_size

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize color if needed
    norm = None
    sm = None
    if color_column:
        norm = plt.Normalize(df[color_column].min(), df[color_column].max())
        cmap_instance = cm.get_cmap(cmap)
        sm = plt.cm.ScalarMappable(cmap=cmap_instance, norm=norm)
        sm.set_array([])
        color_values = df[color_column]
    else:
        color_values = "tab:blue"

    # Plot lines and points
    for group, group_df in df.groupby(group_column):
        x = group_df[year_column]
        y = group_df["_y"]
        size = group_df["_dot_size"]
        color = (group_df[color_column] if color_column else "tab:blue")
        ax.plot(x, y, color="black", linewidth=0.5, zorder=1)
        ax.scatter(x, y, s=size, c=(cmap_instance(norm(color)) if color_column else color), cmap=cmap,
                   norm=norm, edgecolors="black", zorder=2)

    # Y ticks and labels
    y_labels = group_order
    if wrap_labels:
        y_labels = ["\n".join(textwrap.wrap(label, 20)) for label in y_labels]
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(y_labels, fontsize=font_size)
    ax.set_ylabel(ylabel_label, fontsize=font_size)

    # Set x ticks and labels from year_span
    years = list(range(year_span[0], year_span[1] + 1))
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=font_size, rotation=90)
    ax.set_xlabel("Year", fontsize=font_size)

    # Optional colorbar
    if sm is not None:
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(color_column, fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi)
    plt.show()


def plot_item_timelines(
    df: pd.DataFrame,
    min_docs: int = 3,
    regex_filter: str = None,
    top_n_year: int = 3,
    color_by: str = None,
    item_col: str = "Item",
    figsize: tuple = (10, 6),
    dpi: int = 600,
    filename: str = None,
    title: str = None,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    median_rounding: str = None,  # Options: None, "floor", "ceil"
    color_scheme: str = "auto"  # Options: "auto", "viridis", "lightblue"
):
    """
    Plot item timelines based on their Q1-median-Q3 year range and document count.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least the following columns:
        item_col (default: "Item"), "Q1 year", "Median year", "Q3 year", "Number of documents".
        Optionally, a `color_by` column (e.g., "Cited by") can be used for coloring.
    min_docs : int, default=5
        Minimum number of documents for an item to be included.
    regex_filter : str, optional
        A regular expression string to filter item names. If None, no regex filtering is applied.
    top_n : int, default=3
        Number of top items (by document count) to display per median year.
    color_by : str, optional
        Column name used for dot coloring (e.g., "Cited by").
    item_col : str, default="Item"
        Column name for the item/category labels.
    figsize : tuple, default=(10, 6)
        Figure size in inches.
    dpi : int, default=300
        Resolution of the saved plot.
    filename : str, optional
        If provided, saves the plot with this base name to PNG, SVG, and PDF.
    title : str, optional
        Title of the plot. If None, no title is shown.
    title_fontsize : int, default=14
        Font size for the plot title.
    label_fontsize : int, default=12
        Font size for axis labels and colorbar label.
    tick_fontsize : int, default=10
        Font size for axis tick labels.
    median_rounding : str, optional
        How to round "Median year". Options: None, "floor", "ceil".
    color_scheme : str, default="auto"
        If "lightblue", all dots are blue. If "viridis", coloring is by `color_by` column.
        "auto" uses lightblue if no `color_by`, otherwise viridis.

    Returns
    -------
    None
    """
    # Filtering
    filtered = df[df["Number of documents"] >= min_docs].copy()
    if regex_filter:
        filtered = filtered[filtered[item_col].str.contains(regex_filter, flags=re.IGNORECASE, regex=True)]

    # Optional rounding of median year
    if median_rounding == "floor":
        filtered["Median year"] = np.floor(filtered["Median year"]).astype(int)
    elif median_rounding == "ceil":
        filtered["Median year"] = np.ceil(filtered["Median year"]).astype(int)

    # Group by median year and select top N
    grouped = (
        filtered.sort_values(["Median year", "Number of documents"], ascending=[True, False])
        .groupby("Median year")
        .head(top_n_year)
    )

    # Sort for plotting
    grouped = grouped.sort_values(["Median year", "Number of documents"], ascending=[True, False])
    grouped = grouped.reset_index(drop=True)
    grouped["y_pos"] = range(len(grouped))

    # Determine colors
    use_colorbar = False
    if color_scheme == "lightblue" or (color_scheme == "auto" and not color_by):
        colors = ["lightblue"] * len(grouped)
    elif color_scheme == "viridis" or (color_scheme == "auto" and color_by and color_by in grouped.columns):
        norm = mcolors.Normalize(vmin=grouped[color_by].min(), vmax=grouped[color_by].max())
        cmap = cm.viridis
        colors = cmap(norm(grouped[color_by].values))
        use_colorbar = True
    else:
        colors = ["lightblue"] * len(grouped)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)

    for i, row in grouped.iterrows():
        if pd.notnull(row.get("Q1 year")) and pd.notnull(row.get("Q3 year")):
            ax.plot([row["Q1 year"], row["Q3 year"]], [row["y_pos"]] * 2, color="gray", linewidth=1)
        ax.scatter(row["Median year"], row["y_pos"],
                   s=row["Number of documents"] * 10,
                   color=colors[i],
                   edgecolor="black")

    ax.set_yticks(grouped["y_pos"])
    ax.set_yticklabels(grouped[item_col], fontsize=tick_fontsize)
    ax.set_xlabel("Year", fontsize=label_fontsize)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    if use_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(color_by, fontsize=label_fontsize)

    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.grid(False)
    plt.tight_layout()

    if filename:
        save_plot(filename, dpi=dpi)
    plt.show()

def plot_topic_words_bar(topic_df, top_n=10, col_wrap=3, width=12, height=8, show_labels=True, color_column=None, palette_discrete="lightblue", palette_continuous="viridis", save_path=None, dpi=600):
    """
    Plot bar charts of top words per topic based on their weights.

    Args:
        topic_df (pd.DataFrame): DataFrame with columns ["Topic", "Word", "Weight"] and optionally a color_column.
        top_n (int): Number of top words to show per topic.
        col_wrap (int): Number of columns per row in the FacetGrid.
        width (int): Width of the entire figure.
        height (int): Height of each subplot.
        show_labels (bool): Whether to display weight labels on bars (default: True).
        color_column (str, optional): Column in topic_df used to color bars.
        palette_discrete (str or dict): Color or palette for discrete values (default: "lightblue").
        palette_continuous (str): Colormap for continuous values (default: "viridis").
        save_path (str, optional): If provided, the base path to save the figure in PNG, SVG, and PDF.
        dpi (int): Dots per inch for saving the figure (default: 600).

    Returns:
        None: Displays and optionally saves the plot.
    """


    def save_plot(filename_base, dpi=600):
        """Save plot to PNG, SVG, and PDF with tight layout and given DPI."""
        for ext in ["png", "svg", "pdf"]:
            path = f"{filename_base}.{ext}"
            plt.savefig(path, bbox_inches="tight", dpi=dpi)

    # Ensure topic sorting is consistent
    topic_df["Topic"] = pd.Categorical(
        topic_df["Topic"], 
        categories=sorted(topic_df["Topic"].unique(), key=lambda x: int(x.split()[-1]))
    )

    # Select top N words for each topic
    top_words = topic_df.groupby("Topic", group_keys=False).apply(
        lambda x: x.nlargest(top_n, "Weight")
    ).reset_index(drop=True)

    # Determine grid layout
    n_topics = topic_df["Topic"].nunique()
    used_cols = min(col_wrap, n_topics)
    n_rows = math.ceil(n_topics / used_cols)

    # Create barplot without hue, add colors manually
    g = sns.catplot(
        data=top_words,
        x="Weight",
        y="Word",
        col="Topic",
        kind="bar",
        col_wrap=col_wrap,
        height=height / col_wrap,
        aspect=1.0,
        sharex=False,
        sharey=False,
        color=palette_discrete if color_column is None else None
    )

    # Apply custom coloring
    if color_column and color_column in top_words.columns:
        sample_values = top_words[color_column].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(sample_values)

        if is_numeric:
            norm = mcolors.Normalize(vmin=sample_values.min(), vmax=sample_values.max())
            cmap = cm.get_cmap(palette_continuous)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            for ax in g.axes.flatten():
                for bar in ax.patches:
                    label = ax.get_yticklabels()[int(bar.get_y() + bar.get_height() / 2)].get_text()
                    value = top_words[top_words["Word"] == label][color_column].values[0]
                    bar.set_facecolor(cmap(norm(value)))

            # Move colorbar to the side without overlapping
            cbar_ax = g.fig.add_axes([0.92, 0.3, 0.02, 0.4])
            g.fig.colorbar(sm, cax=cbar_ax, label=color_column)

        else:
            color_dict = sns.color_palette(palette_discrete, n_colors=top_words[color_column].nunique())
            color_map = dict(zip(top_words[color_column].unique(), color_dict))

            for ax in g.axes.flatten():
                for bar in ax.patches:
                    label = ax.get_yticklabels()[int(bar.get_y() + bar.get_height() / 2)].get_text()
                    value = top_words[top_words["Word"] == label][color_column].values[0]
                    bar.set_facecolor(color_map.get(value, palette_discrete))

    g.set_titles("{col_name}")
    g.set_axis_labels("Weight", "Word")

    # Add padding for the colorbar/legend
    g.fig.set_size_inches(used_cols * 3.5, n_rows * 3.2)

    # Add labels if requested
    if show_labels:
        for ax in g.axes.flatten():
            for p in ax.patches:
                width = p.get_width()
                ax.text(width + 0.01, p.get_y() + p.get_height() / 2,
                        f"{width:.2f}", va="center")

    g.tight_layout(pad=1.0, rect=[0, 0, 0.9, 1])

    # Save if requested
    if save_path:
        save_plot(save_path, dpi=dpi)

    plt.show()


def plot_topic_distribution(
    df,
    value_column=None,
    palette="viridis",
    save_path=None,
    dpi=600,
    title=None,
    xlabel="Topic",
    ylabel="Number of Documents",
    show_labels=False,
    fontdict_title=None,
    fontdict_labels=None
):
    """
    Plot distribution of documents across topics.

    Args:
        df (pd.DataFrame): DataFrame with a 'Topic' column and optionally a value_column.
        value_column (str, optional): If provided, colors bars by average of this column per topic.
        palette (str): Name of the colormap to use if value_column is given (default: 'viridis').
        save_path (str, optional): If provided, saves the figure to this base path.
        dpi (int): Dots per inch for saving the figure (default: 600).
        title (str, optional): Title of the plot.
        xlabel (str): X-axis label (default: 'Topic').
        ylabel (str): Y-axis label (default: 'Number of Documents').
        show_labels (bool): Whether to show value labels above bars.
        fontdict_title (dict, optional): Font properties for title.
        fontdict_labels (dict, optional): Font properties for axis labels.

    Returns:
        None
    """


    topic_counts = df['Topic'].value_counts().sort_index()
    topic_order = topic_counts.index
    colors = "lightblue"

    fig, ax = plt.subplots(figsize=(8, 6))

    if value_column and value_column in df.columns:
        topic_means = df.groupby("Topic")[value_column].mean().reindex(topic_order)
        norm = mcolors.Normalize(vmin=topic_means.min(), vmax=topic_means.max())
        cmap = cm.get_cmap(palette)
        colors = [cmap(norm(val)) for val in topic_means]

        bars = ax.bar(topic_order, topic_counts.values, color=colors)

        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(f"Average {value_column}")
    else:
        bars = ax.bar(topic_order, topic_counts.values, color=colors)

    ax.set_xlabel(xlabel, fontdict=fontdict_labels)
    ax.set_ylabel(ylabel, fontdict=fontdict_labels)
    if title is not None:
        ax.set_title(title, fontdict=fontdict_title)

    if show_labels:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{int(height)}", ha="center", va="bottom")

    fig.tight_layout()

    if save_path:
        for ext in ["png", "svg", "pdf"]:
            path = f"{save_path}.{ext}"
            plt.savefig(path, bbox_inches="tight", dpi=dpi)

    plt.show()
    
def plot_topic_word_heatmap(topic_df, top_n=10, cmap="viridis", figsize=(12, 8), title="Topic-Word Weights Heatmap", save_path=None, dpi=600, colorbar_label="Weight", fontsize=10, title_fontsize=12, rotate=True, square=True):
    """
    Plot a heatmap of topic-word weights.

    Args:
        topic_df (pd.DataFrame): DataFrame with columns ["Topic", "Word", "Weight"].
        top_n (int): Number of top words to show per topic.
        cmap (str): Matplotlib colormap name (default: 'viridis').
        figsize (tuple): Figure size in inches (default: (12, 8)).
        title (str): Title of the plot.
        save_path (str, optional): If provided, saves the figure to this base path.
        dpi (int): Dots per inch for saving the figure (default: 600).
        colorbar_label (str): Label for the colorbar (default: 'Weight').
        fontsize (int): Font size for tick labels.
        title_fontsize (int): Font size for the plot title.
        rotate (bool): If True, rotate heatmap 90 degrees (topics on y-axis). Default is True.
        square (bool): Whether to draw square cells (default: True).

    Returns:
        None
    """


    # Select top N words per topic
    top_words = topic_df.groupby("Topic", group_keys=False).apply(
        lambda x: x.nlargest(top_n, "Weight")
    ).reset_index(drop=True)

    # Pivot table (rows: topics, columns: words, values: weights)
    heatmap_data = top_words.pivot(index="Topic", columns="Word", values="Weight").fillna(0)
    if not rotate:
        heatmap_data = heatmap_data.transpose()

    # Plot heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(heatmap_data, cmap=cmap, linewidths=0.5, linecolor="gray", cbar_kws={"label": colorbar_label}, square=square)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Words" if rotate else "Topics", fontsize=fontsize)
    ax.set_ylabel("Topics" if rotate else "Words", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()

    if save_path:
        for ext in ["png", "svg", "pdf"]:
            path = f"{save_path}.{ext}"
            plt.savefig(path, bbox_inches="tight", dpi=dpi)

    plt.show()
    
# Plotting of bibliometric laws

def plot_lotka_distribution(lotka_df, title="Lotka's Law - Author Productivity", filename_base=None, dpi=600, 
                             observed_color="blue", expected_color="orange"):
    """
    Plot observed vs expected author productivity under Lotka's Law.

    Parameters:
        lotka_df (pd.DataFrame): Output of compute_lotka_distribution.
        title (str): Title for the plot.
        filename_base (str, optional): Base filename to save the plot without extension.
        dpi (int): Dots per inch for saving the plot.
        observed_color (str): Color for the observed data points and line.
        expected_color (str): Color for the expected (Lotka) line.
    """
    plt.figure(figsize=(8, 6))
    plt.loglog(lotka_df["n_pubs"], lotka_df["n_authors"], "o-", label="Observed", color=observed_color)
    plt.loglog(lotka_df["n_pubs"], lotka_df["expected_n_authors"], "s--", label="Expected (Lotka)", color=expected_color)

    plt.xlabel("Number of Publications (n)", fontsize=12)
    plt.ylabel("Number of Authors", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()

def plot_bradford_distribution(source_counts, title="Bradford's Law - Source Scattering", filename_base=None, dpi=600, color="blue", show_grid=False):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(source_counts) + 1), source_counts["Cumulative_Percentage"], marker="o", color=color)
    plt.xlabel("Ranked Sources", fontsize=12)
    plt.ylabel("Cumulative % of Documents", fontsize=12)
    plt.title(title, fontsize=14)
    if show_grid:
        plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()

def plot_bradford_zones(source_counts, title="Bradford's Law - Zones", filename_base=None, dpi=600, colors=None, annotate_core=True, show_labels="zone1", label_rotation=90, alt_label_col="Abbreviated Source Title", max_label_length=30, show_grid=False):
    fig, ax = plt.subplots(figsize=(12, 6))
    zone_count = source_counts["Zone"].max()
    if colors is None:
        colors = ["#c6dbef", "#9ecae1", "#6baed6"][:zone_count]

    start = 0
    tick_labels = []
    tick_positions = []

    for z in range(1, zone_count + 1):
        group = source_counts[source_counts["Zone"] == z].copy()
        x = list(range(start + 1, start + len(group) + 1))
        ax.fill_between(x, group["Document_Count"], color=colors[z - 1], step="mid", label=f"Zone {z}")

        if show_labels == "all" or (show_labels == "zone1" and z == 1):
            if alt_label_col and alt_label_col in group.columns:
                labels = group[alt_label_col].fillna(group["Source"])
            else:
                labels = group["Source"].apply(lambda s: s if len(s) <= max_label_length else s[:max_label_length-3] + "...")

            tick_labels.extend(labels.tolist())
            tick_positions.extend(x)

        start += len(group)

    ax.set_xlabel("Source Rank", fontsize=12)
    ax.set_ylabel("Documents", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yscale("linear")
    ax.set_xscale("log")
    if show_grid:
        ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    if tick_labels:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=label_rotation, ha="right", fontsize=9)

    if annotate_core:
        ax.text(2, max(source_counts["Document_Count"]) * 0.9, "Core Sources", fontsize=12, alpha=0.6)

    plt.tight_layout()
    if filename_base:
        save_plot(filename_base, dpi)
    plt.show()

def compute_zipf_distribution_from_counts(df, word_col=0, count_col=1):
    """
    Compute Zipf's Law distribution given a DataFrame with word/item counts.
    
    Parameters:
        df (pd.DataFrame): DataFrame where one column is words/items and another is counts.
        word_col (int or str): Column name or index for words/items.
        count_col (int or str): Column name or index for counts.
        
    Returns:
        pd.DataFrame: DataFrame with 'Word', 'Frequency', and 'Rank'.
    """
    # Convert column indices to names if integers
    if isinstance(word_col, int):
        word_col = df.columns[word_col]
    if isinstance(count_col, int):
        count_col = df.columns[count_col]
    
    zipf_df = df[[word_col, count_col]].copy()
    zipf_df.columns = ["Word", "Frequency"]
    zipf_df = zipf_df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
    zipf_df["Rank"] = np.arange(1, len(zipf_df) + 1)
    return zipf_df


def plot_zipf_distribution(zipf_df, title="Zipf's Law - Word Frequencies", filename_base=None, dpi=600, color="blue", show_grid=False, top_n_labels=10):
    """
    Plot Zipf's Law distribution: frequency vs rank on a log-log scale.

    Parameters:
        zipf_df (pd.DataFrame): Output of compute_zipf_distribution_from_counts.
        title (str): Title of the plot.
        filename_base (str, optional): Base filename to save the plot.
        dpi (int): Dots per inch for saving.
        color (str): Color of the curve.
        show_grid (bool): Whether to show grid.
        top_n_labels (int): Number of top labels to display.
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(zipf_df["Rank"], zipf_df["Frequency"], marker="o", linestyle="-", color=color)
    plt.xlabel("Rank", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(title, fontsize=14)

    if top_n_labels > 0:
        for i in range(min(top_n_labels, len(zipf_df))):
            x = zipf_df.loc[i, "Rank"]
            y = zipf_df.loc[i, "Frequency"]
            word = zipf_df.loc[i, "Word"]
            plt.text(x, y, word, fontsize=8, ha="left", va="bottom")

    if show_grid:
        plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()


def plot_prices_law(author_counts, title="Price's Law - Core Author Contribution", filename_base=None, dpi=600, color_core="red", color_tail="gray", show_grid=False):
    """
    Plot cumulative document contribution by authors, highlighting Price's core group.

    Parameters:
        author_counts (pd.Series): Series with authors as index and document counts as values.
        title (str): Plot title.
        filename_base (str): Base path to save the plot.
        dpi (int): Dots per inch for saving.
        color_core (str): Color for core authors.
        color_tail (str): Color for remaining authors.
        show_grid (bool): Whether to show grid.
    """
    sorted_counts = author_counts.sort_values(ascending=False).reset_index(drop=True)
    cumulative_docs = sorted_counts.cumsum() / sorted_counts.sum() * 100
    core_size = int(np.sqrt(len(sorted_counts)))

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_docs.index[:core_size], cumulative_docs.iloc[:core_size], color=color_core, label="Core Authors")
    plt.plot(cumulative_docs.index[core_size:], cumulative_docs.iloc[core_size:], color=color_tail, label="Other Authors")

    plt.axvline(core_size, color="black", linestyle="--", linewidth=1, label=f"sqrt(N) = {core_size}")
    plt.xlabel("Author Rank", fontsize=12)
    plt.ylabel("Cumulative % of Documents", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    if show_grid:
        plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()
    
def plot_pareto_principle(counts, top_percentage=20, title="Pareto Principle Analysis", filename_base=None, dpi=600, color_curve="blue", color_threshold="red", show_grid=False):
    """
    Plot cumulative contribution curve highlighting the Pareto threshold.

    Parameters:
        counts (pd.Series): Series of items and their counts.
        top_percentage (float): Percentage of top items (default 20).
        title (str): Title for the plot.
        filename_base (str): Base filename to save the plot.
        dpi (int): Resolution for saved plots.
        color_curve (str): Line color.
        color_threshold (str): Threshold line color.
        show_grid (bool): Whether to show grid.
    """
    sorted_counts = counts.sort_values(ascending=False).reset_index(drop=True)
    cumulative_contribution = sorted_counts.cumsum() / sorted_counts.sum() * 100

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_contribution.index, cumulative_contribution.values, color=color_curve)

    threshold_index = int(np.ceil(top_percentage / 100 * len(cumulative_contribution)))
    plt.axvline(threshold_index, color=color_threshold, linestyle="--", label=f"Top {top_percentage}%")

    plt.xlabel("Item Rank", fontsize=12)
    plt.ylabel("Cumulative % of Contribution", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()

    if show_grid:
        plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)

    plt.show()
    
# Citations per year (prestavi nekoliko višje)

def plot_average_citations_per_year(
    df,
    year_col="Year",
    avg_col="Average Citations per Document",
    doc_count_col=None,  # optional column for secondary axis
    plot_type="line",
    color="black",
    secondary_color="lightblue",
    title="Average Citations per Document by Year",
    xlabel="Year",
    ylabel="Average Citations per Document",
    ylabel_secondary="Number of Documents",
    fontsize_title=14,
    fontsize_labels=12,
    marker="o",
    linewidth=2,
    xtick_rotation=90,
    wrap_xticks=False,
    wrap_width=10,
    filename_base=None,
    show=True
):
    """
    Plot the average citations per document by year as a line or bar chart,
    with optional secondary y-axis and x-tick label wrapping.

    Parameters:
        df (pd.DataFrame): DataFrame containing the year and citation data.
        year_col (str): Column name for year.
        avg_col (str): Column name for average citations.
        doc_count_col (str or None): Optional column name for document counts (for secondary y-axis).
        plot_type (str): "line" (default) or "bar".
        color (str): Color for main plot.
        secondary_color (str): Color for secondary axis plot (if enabled).
        title (str): Title of the plot.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        ylabel_secondary (str): Y-axis label for secondary axis.
        fontsize_title (int): Title font size.
        fontsize_labels (int): Axis labels font size.
        marker (str): Marker for line plot.
        linewidth (int or float): Line width for line plot.
        xtick_rotation (int): Rotation angle for x-tick labels.
        wrap_xticks (bool): Whether to wrap long tick labels.
        wrap_width (int): Max width of each wrapped line.
        filename_base (str or None): Base filename for saving.
        show (bool): Whether to display the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = df[year_col]
    y = df[avg_col]

    # Plot main data
    if plot_type == "bar":
        ax1.bar(x, y, color=color)
    elif plot_type == "line":
        ax1.plot(x, y, marker=marker, color=color, linewidth=linewidth)
    else:
        raise ValueError("plot_type must be either 'line' or 'bar'.")

    ax1.set_title(title, fontsize=fontsize_title)
    ax1.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax1.set_ylabel(ylabel, fontsize=fontsize_labels)
    ax1.tick_params(axis="y", labelcolor=color)

    # Handle optional wrapping of x-tick labels
    if wrap_xticks:
        labels = [textwrap.fill(str(label), wrap_width) for label in x]
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=xtick_rotation)
    else:
        plt.xticks(rotation=xtick_rotation)

    # Optional secondary axis
    if doc_count_col is not None and doc_count_col in df.columns:
        y2 = df[doc_count_col]
        ax2 = ax1.twinx()
        ax2.plot(x, y2, color=secondary_color, linestyle="--", marker="s")
        ax2.set_ylabel(ylabel_secondary, fontsize=fontsize_labels)
        ax2.tick_params(axis="y", labelcolor=secondary_color)

    if filename_base:
        save_plot(filename_base)

    if show:
        plt.show()
    else:
        plt.close()
        
# factor analysis plotting functions
# use conceptual_structure_analysis function from utilsbib

def plot_word_map(embeddings: np.ndarray, terms: list, labels: np.ndarray,
                  figsize: tuple = (10, 8), title: str = "Word Map",
                  filename_base: str = None, dpi: int = 600,
                  cmap: str = 'tab10', marker_size: int = 50,
                  term_fontsize: int = 8, title_fontsize: int = 12,
                  axis_label_fontsize: int = 10, tick_label_fontsize: int = 8,
                  xlabel: str = 'Dim 1', ylabel: str = 'Dim 2',
                  show_legend: bool = True
) -> None:
    """
    Scatter plot of term embeddings colored by cluster labels.

    Colors correspond to cluster assignments; a legend (if enabled) maps colors to cluster IDs.
    Any terms whose labels overlap exactly will still only show one annotation—use different DR or jitter to separate.

    Parameters
    ----------
    show_legend : bool
        Whether to display a legend for cluster colors.
    cmap : str
        Matplotlib colormap for cluster coloring.
    marker_size : int
        Size of scatter markers.
    term_fontsize : int
        Font size for term annotations.
    title_fontsize : int
        Font size for the title.
    axis_label_fontsize : int
        Font size for axis labels.
    tick_label_fontsize : int
        Font size for tick labels.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.

    Examples
    --------
    >>> result = conceptual_structure_analysis(df)
    >>> plot_word_map(
    ...     result['term_embeddings'], result['terms'], result['term_labels'],
    ...     marker_size=100, cmap='viridis', show_legend=True
    >>> )
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    n_pts = embeddings.shape[0]
    # Auto-fill missing term labels if terms list is too short
    if len(terms) < n_pts:
        fallback = [f"cluster_{lbl}" for lbl in labels[len(terms):]]
        terms = list(terms) + fallback
    # Validate lengths
    if len(terms) != n_pts or len(labels) != n_pts:
        raise ValueError(
            f"plot_word_map: embeddings ({n_pts}) must match len(terms) ({len(terms)}) and len(labels) ({len(labels)})"
        )
    # Ensure 2D for scatter
    if embeddings.ndim == 1:
        x = embeddings; y = np.zeros_like(x)
        embeddings = np.vstack((x, y)).T
    elif embeddings.ndim == 2 and embeddings.shape[1] == 1:
        x = embeddings[:, 0]; y = np.zeros_like(x)
        embeddings = np.vstack((x, y)).T

    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=labels, cmap=cmap, s=marker_size
    )
    for i, term in enumerate(terms):
        plt.text(
            embeddings[i, 0], embeddings[i, 1], term,
            fontsize=term_fontsize
        )
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=axis_label_fontsize)
    plt.ylabel(ylabel, fontsize=axis_label_fontsize)
    plt.xticks(fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.grid(False)
    if show_legend:
        handles, label_values = scatter.legend_elements()
        legend = plt.legend(
            handles, label_values,
            title="Cluster",
            fontsize=tick_label_fontsize
        )
        legend.get_title().set_fontsize(axis_label_fontsize)
    if filename_base:
        save_plot(filename_base, dpi)
    plt.show()


def plot_topic_dendrogram(
    embeddings: np.ndarray,
    terms: list,
    method: str = "ward",
    figsize: tuple = (10, 8),
    title: str = "Topic Dendrogram",
    filename_base: str = None,
    dpi: int = 600,
    xlabel: str = "Terms",
    ylabel: str = "Distance",
    title_fontsize: int = 12,
    axis_label_fontsize: int = 10,
    tick_label_fontsize: int = 8,
    leaf_label_fontsize: int = 8
) -> None:
    """
    Hierarchical clustering dendrogram of term embeddings.
    If filename_base is provided, the plot is saved (PNG, SVG, PDF).

    Parameters
    ----------
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title_fontsize : int
        Font size for the title.
    axis_label_fontsize : int
        Font size for axis labels.
    tick_label_fontsize : int
        Font size for axis tick labels.
    leaf_label_fontsize : int
        Font size for the leaf labels (term names).

    Examples
    --------
    >>> result = conceptual_structure_analysis(df)
    >>> plot_topic_dendrogram(
    ...     result['term_embeddings'], result['terms'],
    ...     xlabel='Component 1', ylabel='Component 2',
    ...     title_fontsize=14, axis_label_fontsize=12,
    ...     tick_label_fontsize=10, leaf_label_fontsize=9
    >>> )
    """


    emb = np.atleast_2d(embeddings)
    n_pts = emb.shape[0]
    # Auto-fill missing term labels
    if len(terms) < n_pts:
        fallback = [f"cluster_{i}" for i in range(len(terms), n_pts)]
        terms = list(terms) + fallback
    # Validate
    if len(terms) != n_pts:
        raise ValueError(
            f"plot_topic_dendrogram: need one label per point, got {len(terms)} labels for {n_pts} points"
        )

    Z = linkage(emb, method=method)
    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=terms,
        leaf_rotation=90,
        leaf_font_size=leaf_label_fontsize
    )
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=axis_label_fontsize)
    plt.ylabel(ylabel, fontsize=axis_label_fontsize)
    plt.xticks(fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi)
    plt.show()

# spectroscopy plotting

def plot_reference_spectrogram(spectrogram_df, title="Spectroscopy of Science", save_path=None, group_by_decade=False):
    """
    Plots the spectrogram of cited years or decades.

    Args:
        spectrogram_df (pd.DataFrame): DataFrame with index as years and citation counts.
        title (str): Plot title.
        save_path (str or None): If provided, saves the plot using save_plot().
        group_by_decade (bool): If True, groups citations by decade.
    """
    df = spectrogram_df.copy()
    if group_by_decade:
        df["Decade"] = (df.index // 10) * 10
        df = df.groupby("Decade")["Citations"].sum().reset_index()
        x = df["Decade"]
        y = df["Citations"]
        xlabel = "Cited Decade"
    else:
        x = df.index
        y = df["Citations"]
        xlabel = "Cited Year"

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Citations")
    plt.grid(False)
    if save_path:
        save_plot(save_path)
    plt.show()



def plot_reference_spectrogram(spectrogram_df, title="Spectroscopy of Science", save_path=None, group_by_decade=False, show_grid=False, fontsize=12):
    """
    Plots the spectrogram of cited years or decades.

    Args:
        spectrogram_df (pd.DataFrame): DataFrame with index as years and citation counts.
        title (str): Plot title.
        save_path (str or None): If provided, saves the plot using save_plot().
        group_by_decade (bool): If True, groups citations by decade.
    """
    df = spectrogram_df.copy()
    if group_by_decade:
        df["Decade"] = (df.index // 10) * 10
        df = df.groupby("Decade")["Citations"].sum().reset_index()
        x = df["Decade"]
        y = df["Citations"]
        xlabel = "Cited Decade"
    else:
        x = df.index
        y = df["Citations"]
        xlabel = "Cited Year"

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=2)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel("Number of Citations", fontsize=fontsize)
    plt.grid(show_grid)
    if save_path:
        save_plot(save_path)
    plt.show()
    
def plot_reference_correlation(plot_df, xlabel=None, ylabel=None, title="Reference Correlation", show_corr=True, save_path=None, show_grid=False, fontsize=12):
    """
    Plots a scatterplot using two columns from a DataFrame with optional correlation display.

    Args:
        df (pd.DataFrame): DataFrame with exactly two columns: x and y.
        xlabel (str or None): Label for x-axis (default is column name).
        ylabel (str or None): Label for y-axis (default is column name).
        title (str): Plot title.
        show_corr (bool): Whether to display Pearson correlation in title.
        save_path (str or None): If provided, saves the plot.
    """
    x = plot_df.iloc[:, 0]
    y = plot_df.iloc[:, 1]
    xlabel = xlabel or plot_df.columns[0]
    ylabel = ylabel or plot_df.columns[1]

    corr_text = ""
    if show_corr:
        r, _ = pearsonr(x, y)
        corr_text = f" (r = {r:.2f})"

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5)
    min_val, max_val = min(min(x), min(y)), max(max(x), max(y))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(f"{title}{corr_text}", fontsize=fontsize)
    plt.grid(show_grid)
    if save_path:
        save_plot(save_path)
    plt.show()
    
# Scientific production by group

def plot_stacked_production_by_group(
    stats: pd.DataFrame,
    group_colors: dict[str, str] | None = None,
    filename_base: str = None,
    figsize: tuple = (10, 6),
    cut_year: int | None = None,
    year_span: tuple[int, int] | None = None,
    citation_mode: str = "group",
    font_size: int = 12,
    xlabel: str = "Year",
    ylabel: str = "Number of documents",
    citation_label: str = "Cumulative Citations",
    legend_title: str = "Group"
) -> plt.Axes:
    """
    Plot a stacked bar chart of document counts by group per year,
    with optional aggregation for years before a cutoff and cumulative citation lines.

    Parameters
    ----------
    stats : pd.DataFrame
        Wide-format statistics from get_scientific_production_by_group.
    group_colors : dict of str, optional
        Mapping from group name to hex color code.
    filename_base : str, optional
        Base filename for saving the plot (without extension).
    figsize : tuple, default (10,6)
        Size of the figure in inches.
    cut_year : int, optional
        Year before which data is aggregated into 'Before {cut_year}'.
    year_span : tuple of int, optional
        (start_year, end_year) to restrict the x-axis.
    citation_mode : {'group', 'together', None}, default 'group'
        Whether to plot cumulative citations per group, all together, or not at all.
    font_size : int, default 12
        Base font size for labels and ticks.
    xlabel : str, default 'Year'
        Label for the x-axis.
    ylabel : str, default 'Number of documents'
        Label for the y-axis.
    citation_label : str, default 'Cumulative Citations'
        Label for the citation y-axis.
    legend_title : str, default 'Group'
        Title for the legend.

    Returns
    -------
    matplotlib.axes.Axes
        The primary axis of the stacked bar chart.
    """
    if "Year" not in stats.columns:
        raise ValueError("The stats DataFrame must contain a 'Year' column.")
    data = stats.copy()
    cols = data.columns.tolist()
    doc_cols = [c for c in cols if c.startswith("Number of documents ")]
    cit_cols = [c for c in cols if c.startswith("Cumulative Citations ")]
    data["Year"] = data["Year"].astype(object)
    if cut_year is not None:
        label = f"Before {cut_year}"
        mask_pre = pd.to_numeric(data["Year"], errors="coerce") < cut_year
        data.loc[mask_pre, "Year"] = label
        agg_dict = {c: 'sum' for c in doc_cols}
        agg_dict.update({c: 'max' for c in cit_cols})
        data = data.groupby("Year", as_index=False).agg(agg_dict)
    if year_span is not None:
        numeric = pd.to_numeric(data["Year"], errors="coerce")
        mask = numeric.between(year_span[0], year_span[1])
        data = data[mask | data["Year"].astype(str).str.startswith("Before")]
    data["Year"] = data["Year"].astype(str)
    unique_years = []
    for y in data["Year"]:
        if y not in unique_years:
            unique_years.append(y)
    before = [y for y in unique_years if y.startswith("Before")]
    nums = sorted([y for y in unique_years if not y.startswith("Before")], key=lambda x: int(x))
    x_order = before + nums
    groups = [c.replace("Number of documents ", "") for c in doc_cols]
    if group_colors is None:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.65)
    bottom = np.zeros(len(x_order))
    for grp, col in zip(groups, doc_cols):
        vals = data.set_index("Year")[col].reindex(x_order).fillna(0).values
        ax.bar(x_order, vals, bottom=bottom, label=grp, color=group_colors.get(grp))
        bottom += vals
    if citation_mode in ("group", "together") and cit_cols:
        ax2 = ax.twinx()
        if citation_mode == "group":
            for grp, col in zip(groups, cit_cols):
                vals2 = data.set_index("Year")[col].reindex(x_order).fillna(0).values
                ax2.plot(x_order, vals2, linestyle="--", marker="o", color=group_colors.get(grp))
        else:
            total_vals = pd.Series(data[cit_cols].sum(axis=1).values, index=data["Year"]).reindex(x_order).fillna(0).values
            ax2.plot(x_order, total_vals, linestyle="--", marker="o", color="black")
        ax2.set_ylabel(citation_label, fontsize=font_size)
        ax2.tick_params(axis='y', labelsize=font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size, rotation=90)
    ax.tick_params(axis='y', labelsize=font_size)
    ax.legend(title=legend_title, fontsize=font_size, title_fontsize=font_size,
              loc="upper left", bbox_to_anchor=(1.15, 1), borderaxespad=0)
    plt.tight_layout()
    if filename_base:
        save_plot(filename_base)
    return ax

# Plotting of differences between two frequency distribvutions

def plot_count_differences(df: pd.DataFrame,
                            orientation: str = "auto",
                            title: str = "Relative Differences in Item Counts",
                            xlabel: str = "Percentage Point Difference (pp)",
                            ylabel: str = "Item",
                            color_pos: str = "skyblue",
                            color_neg: str = "lightcoral",
                            alpha_cap: float = 0.8,
                            grid: bool = False,
                            show_zero_line: bool = True,
                            annotate: bool = True,
                            rotation: int = 45,
                            label_offset: float = 0.1,
                            margin_ratio: float = 0.1,
                            figsize: tuple = (10, 6)) -> None:
    """
    Plot percentage point differences using horizontal or vertical bars.
    """
    df_plot = df.copy()

    if orientation == "auto":
        try:
            years = pd.to_numeric(df_plot.index, errors="coerce")
            is_time_series = years.notna().all() and years.is_monotonic_increasing
            orientation = "vertical" if is_time_series else "horizontal"
        except:
            orientation = "horizontal"

    df_plot["Color"] = np.where(df_plot["PP_Diff"] >= 0, color_pos, color_neg)

    if orientation == "horizontal":
        df_plot = df_plot.sort_values("PP_Diff", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    if orientation == "horizontal":
        bars = ax.barh(df_plot.index.astype(str), df_plot["PP_Diff"],
                       color=df_plot["Color"], alpha=alpha_cap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        x_vals = df_plot["PP_Diff"]
        x_margin = (x_vals.max() - x_vals.min()) * margin_ratio
        ax.set_xlim(x_vals.min() - x_margin, x_vals.max() + x_margin)

        if annotate:
            for bar in bars:
                value = bar.get_width()
                offset = label_offset if value > 0 else -label_offset
                ax.text(value + offset, bar.get_y() + bar.get_height() / 2,
                        f"{value:.1f} pp", va="center",
                        ha="left" if value > 0 else "right")

    else:
        bars = ax.bar(df_plot.index.astype(str), df_plot["PP_Diff"],
                      color=df_plot["Color"], alpha=alpha_cap)
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
        plt.xticks(rotation=rotation, ha="right")
        y_vals = df_plot["PP_Diff"]
        y_margin = (y_vals.max() - y_vals.min()) * margin_ratio
        ax.set_ylim(y_vals.min() - y_margin, y_vals.max() + y_margin)

        if annotate:
            for bar in bars:
                value = bar.get_height()
                offset = label_offset if value > 0 else -label_offset
                ax.text(bar.get_x() + bar.get_width() / 2, value + offset,
                        f"{value:.1f} pp", ha="center",
                        va="bottom" if value > 0 else "top")

    if show_zero_line:
        if orientation == "vertical":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        else:
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    if grid:
        ax.grid(True, linestyle=":", linewidth=0.5)
    else:
        ax.grid(False)

    ax.set_title(title)
    plt.tight_layout()
    



def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    size_col: Optional[str] = None,
    color_col: Optional[str] = None,
    marker_col: Optional[str] = None,
    marker_sequence: Optional[List[str]] = None,
    label_col: Optional[str] = None,
    error_x: Optional[Union[str, Sequence[float]]] = None,
    error_y: Optional[Union[str, Sequence[float]]] = None,
    dropna: bool = True,
    fig_size: Tuple[float, float] = (8, 6),
    style_sheet: Optional[str] = None,
    grid: Union[bool, Dict[str, Any]] = False,
    alpha: float = 1.0,
    edge_color: Optional[str] = None,
    edge_width: float = 0.0,
    zorder: int = 2,
    equal_aspect: bool = False,
    tick_params: Optional[Dict[str, Any]] = None,
    formatter: Optional[Callable] = None,
    x_scale: str = "log",
    y_scale: str = "log",
    log_base: Tuple[int, int] = (10, 10),
    size_scale: str = "linear",
    max_size: float = 300,
    colormap: str = "viridis",
    lines: Optional[List[Tuple[float, float]]] = None,
    symmetric_lines: bool = True,
    mean_line: bool = False,
    median_line: bool = False,
    identity_line: bool = False,
    mean_marker: bool = False,
    adjust_kwargs: Optional[Dict[str, Any]] = None,
    highlight_points: Optional[Union[Sequence[int], Sequence[bool]]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    legend: bool = True,
    legend_kwargs: Optional[Dict[str, Any]] = None,
    filename: str = "scatter",
    dpi: int = 600,
    show: bool = True,
    **kwargs,
) -> None:
    df_plot = df.copy()
    cols = [c for c in [x, y, size_col, color_col, marker_col, label_col, error_x, error_y] if c]
    if dropna and cols:
        df_plot.dropna(subset=cols, inplace=True)
    if style_sheet:
        plt.style.use(style_sheet)
    fig, ax = plt.subplots(figsize=fig_size)
    if grid:
        ax.grid(**grid) if isinstance(grid, dict) else ax.grid(True)
    if x_scale == "log":
        ax.set_xscale("log", base=log_base[0])
    else:
        ax.set_xscale(x_scale)
    if y_scale == "log":
        ax.set_yscale("log", base=log_base[1])
    else:
        ax.set_yscale(y_scale)
    n = len(df_plot)
    if size_col:
        raw = df_plot[size_col]
        if size_scale == "log":
            raw = np.log(raw.clip(lower=np.nextafter(0, 1)))
        max_raw = raw.max()
        sizes = (raw / max_raw) * max_size
    else:
        sizes = np.full(n, max_size)
    use_colorbar = False
    if color_col and pd.api.types.is_numeric_dtype(df_plot[color_col]):
        norm_c = Normalize(vmin=df_plot[color_col].min(), vmax=df_plot[color_col].max())
        color_vals = norm_c(df_plot[color_col])
        use_colorbar = True
    elif color_col:
        color_vals = df_plot[color_col]
    else:
        color_vals = None
    base_kwargs = {"s": sizes, "cmap": colormap, "alpha": alpha, "edgecolors": edge_color, "linewidths": edge_width, "zorder": zorder}
    if color_vals is not None:
        base_kwargs["c"] = color_vals
    marker_handles = []
    if marker_col:
        default_markers = ["o","s","^","D","v","<",">","p","*","X"]
        markers = marker_sequence or default_markers
        for i, grp in enumerate(df_plot[marker_col].unique()):
            mask = df_plot[marker_col] == grp
            kwargs_grp = base_kwargs.copy()
            kwargs_grp.update({"s": sizes[mask], **({"c": color_vals[mask]} if color_vals is not None else {})})
            m = markers[i % len(markers)]
            ax.scatter(df_plot.loc[mask, x], df_plot.loc[mask, y], marker=m, **kwargs_grp, **kwargs)
            marker_handles.append((m, grp))
    else:
        ax.scatter(df_plot[x], df_plot[y], **base_kwargs, **kwargs)
    if error_x or error_y:
        ax.errorbar(df_plot[x], df_plot[y], xerr=(df_plot[error_x] if error_x else None), yerr=(df_plot[error_y] if error_y else None), fmt="none", alpha=alpha*0.5)
    if legend:
        size_flag = size_col is not None
        marker_flag = marker_col is not None
        if size_flag and not marker_flag:
            vals = [df_plot[size_col].min(), df_plot[size_col].median(), df_plot[size_col].max()]
            sizes_leg = [(np.log(v) if size_scale == "log" else v) / (raw.max()) * max_size for v in vals]
            handles = [Line2D([], [], linestyle="", marker="o", markersize=np.sqrt(s), color="black", alpha=alpha) for s in sizes_leg]
            labs = [str(int(v)) if float(v).is_integer() else f"{v:.2f}" for v in vals]
            ax.legend(handles, labs, title=size_col, loc="lower right", frameon=True, edgecolor="black")
        elif marker_flag and not size_flag:
            handles = [Line2D([], [], linestyle="", marker=marker_handles[i][0], color="black", markersize=6) for i in range(len(marker_handles))]
            labs = [grp for _, grp in marker_handles]
            ax.legend(handles, labs, title=marker_col, loc="lower right", frameon=True, edgecolor="black")
        else:
            if size_flag:
                vals = [df_plot[size_col].min(), df_plot[size_col].median(), df_plot[size_col].max()]
                sizes_leg = [(np.log(v) if size_scale == "log" else v) / (raw.max()) * max_size for v in vals]
                handles = [Line2D([], [], linestyle="", marker="o", markersize=np.sqrt(s), color="black", alpha=alpha) for s in sizes_leg]
                labs = [str(int(v)) if float(v).is_integer() else f"{v:.2f}" for v in vals]
                fig.legend(handles, labs, title=size_col, loc="lower center", bbox_to_anchor=(0.25, -0.02), ncol=len(handles), frameon=True, edgecolor="black")
            if marker_flag:
                handles = [Line2D([], [], linestyle="", marker=m, color="black", markersize=6) for m, _ in marker_handles]
                labs = [grp for _, grp in marker_handles]
                fig.legend(handles, labs, title=marker_col, loc="lower center", bbox_to_anchor=(0.75, -0.02), ncol=len(handles), frameon=True, edgecolor="black")
        if color_col and use_colorbar:
            sm = ScalarMappable(norm=norm_c, cmap=colormap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label(color_col)
    xmin, xmax = ax.get_xlim()
    xs = np.linspace(xmin, xmax, 200)
    if mean_line:
        mx, my = df_plot[x].mean(), df_plot[y].mean()
        ax.axvline(mx, linestyle="--", color="gray", zorder=1)
        ax.axhline(my, linestyle="--", color="gray", zorder=1)
    if median_line:
        mdx, mdy = df_plot[x].median(), df_plot[y].median()
        ax.axvline(mdx, linestyle=":", color="gray", zorder=1)
        ax.axhline(mdy, linestyle=":", color="gray", zorder=1)
    if identity_line:
        ax.plot(xs, xs, linestyle="-", color="black", zorder=1)
    if lines:
        for slope, intercept in lines:
            ax.plot(xs, slope*xs + intercept, linestyle="-", color="gray", zorder=1)
            if symmetric_lines:
                ax.plot(xs, (xs - intercept)/slope, linestyle="--", color="gray", zorder=1)
    if mean_marker and mean_line:
        mx, my = df_plot[x].mean(), df_plot[y].mean()
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        y_offset = (ymax - ymin) * 0.02
        ax.text(mx, ymin, f"$\mu_x={mx:.2f}$", color="black", fontsize=kwargs.get("label_font_size",6), ha="center", va="bottom", zorder=3)
        ax.text(xmin, my + y_offset, f"$\mu_y={my:.2f}$", color="black", fontsize=kwargs.get("label_font_size",6), ha="left", va="bottom", zorder=3)
    if highlight_points is not None:
        hp = df_plot.loc[highlight_points]
        ax.scatter(hp[x], hp[y], s=kwargs.get("mean_marker_size",6), color="red", zorder=4)
    if label_col:
        texts = [ax.text(row[x], row[y], str(row[label_col]), fontsize=kwargs.get("label_font_size",6), zorder=5) for _, row in df_plot.iterrows()]
        adjust_text(texts, ax=ax, **(adjust_kwargs or {}))
    if title:
        ax.set_title(title)
    if subtitle:
        ax.set_title(subtitle, fontsize=kwargs.get("label_font_size",6), style="italic")
    if caption:
        fig.text(0.5, -0.3, caption, ha="center", fontsize=kwargs.get("label_font_size",6))
    ax.set_xlabel(x_label or x)
    ax.set_ylabel(y_label or y)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if tick_params:
        ax.tick_params(**tick_params)
    if formatter:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.18, right=0.85)
    save_plot(filename, dpi)
    if show:
        plt.show()
    plt.close(fig)


def prepare_for_sankey(
    dataframes: List[pd.DataFrame],
    top_n: Union[int, Sequence[int]] = 10,
    label_maps: Optional[Dict[str, str]] = None,
    color_series: Optional[pd.Series] = None,
    color_func: Callable[[pd.Series], float] = np.mean,
    all_pairs: bool = False
) -> Tuple[pd.DataFrame, List[str], List[float], List[int], List[int]]:
    """
    Prepare link and node information for a Sankey diagram from multiple binary indicator DataFrames.

    Args:
        dataframes: A list of binary indicator DataFrames (docs x concepts). Column names are used directly as node labels.
        top_n: Number of top columns to select per DataFrame. If an int, applies to all; if a sequence, applies respectively.
        label_maps: Optional mapping to rename labels before plotting (keys should match column names).
        color_series: Optional pd.Series indexed like rows of dataframes, used to compute node colors.
        color_func: Function to aggregate values in color_series for each node (default: np.mean).
        all_pairs: If True, compute connections for all unique pairs; otherwise, only between consecutive fields.

    Returns:
        links: DataFrame with columns ["source", "target", "value"] for Sankey links.
        labels: List of node labels in order.
        color_values: List of aggregated color values for each node.
        selected_counts: List of number of columns selected per field.
        group_ids: List mapping each node index to its originating field index.
    """
    # Normalize top_n
    if top_n is None:
        selected_counts = [df.shape[1] for df in dataframes]
    elif isinstance(top_n, int):
        selected_counts = [top_n] * len(dataframes)
    else:
        selected_counts = list(top_n)

    selected_fields = [df.iloc[:, :n] for df, n in zip(dataframes, selected_counts)]

    # Compute color values per node
    color_values: List[float] = []
    if color_series is not None:
        for df in selected_fields:
            for col in df.columns:
                mask = df[col] == 1
                values = color_series.loc[mask].dropna()
                color_values.append(color_func(values) if not values.empty else np.nan)

    # Build node labels and group assignments
    raw_labels = list(itertools.chain.from_iterable(df.columns.tolist() for df in selected_fields))
    group_ids = list(itertools.chain.from_iterable(
        [i] * df.shape[1] for i, df in enumerate(selected_fields)
    ))

    # Map label to node index
    label_to_idx = {label: idx for idx, label in enumerate(raw_labels)}

    # Compute link pairs
    link_frames = []
    pairs = (
        itertools.combinations(range(len(selected_fields)), 2)
        if all_pairs
        else zip(range(len(selected_fields) - 1), range(1, len(selected_fields)))
    )
    for i, j in pairs:
        counts = selected_fields[i].T.dot(selected_fields[j])
        links = counts.stack().reset_index()
        links.columns = ["source_label", "target_label", "value"]
        link_frames.append(links)

    links_df = pd.concat(link_frames, ignore_index=True)[["source_label", "target_label", "value"]]

    # Rename labels if mapping provided
    rename = (lambda x: label_maps.get(x, x)) if label_maps else (lambda x: x)

    # Map to indices
    links_df = pd.DataFrame({
        "source": links_df["source_label"].map(label_to_idx),
        "target": links_df["target_label"].map(label_to_idx),
        "value": links_df["value"]
    })

    final_labels = [rename(label) for label in raw_labels]

    return links_df, final_labels, color_values, selected_counts, group_ids


def plot_sankey(
    links_df: pd.DataFrame,
    labels: List[str],
    color_values: Optional[List[float]] = None,
    group_ids: Optional[List[int]] = None,
    field_names: Optional[List[str]] = None,
    save_png: Optional[str] = None,
    save_html: Optional[str] = None,
    colorscale: Union[str, List] = "Viridis",
    colorbar_title: str = ""
) -> go.Figure:
    """
    Create and optionally save a Sankey diagram with colorbar and field annotations.

    Args:
        links_df: DataFrame with columns [source, target, value].
        labels: List of node labels.
        color_values: Numeric list for node colors.
        group_ids: Field index for each node (for x-axis placement).
        field_names: Names of each field to annotate.
        save_png: File path to save the figure as a PNG.
        save_html: File path to save the figure as HTML.
        colorscale: Plotly colorscale name or list.
        colorbar_title: Title for the colorbar.

    Returns:
        Plotly Figure object.
    """


    # Build Sankey node/link dicts
    link = dict(
        source=links_df["source"].tolist(),
        target=links_df["target"].tolist(),
        value=links_df["value"].tolist()
    )
    node = dict(label=labels, pad=15, thickness=20)

    # Prepare node colors if provided
    vmin = vmax = None
    if color_values:
        vals = np.array(color_values, dtype=float)
        # normalize
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            norm = (vals - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(vals)
        # sample colors
        cs = get_colorscale(colorscale)
        node_colors = sample_colorscale(cs, norm.tolist())
        node["color"] = node_colors

    # Position nodes by group
    if group_ids and field_names:
        k = len(field_names)
        node["x"] = [gid / (k - 1) for gid in group_ids]

    # Create figure
    fig = go.Figure(go.Sankey(link=link, node=node))

    # Add field annotations
    if field_names:
        annotations = []
        k = len(field_names)
        for idx, name in enumerate(field_names):
            annotations.append(dict(
                x=idx/(k-1), y=1.05,
                xref="paper", yref="paper",
                text=name, showarrow=False,
                font=dict(size=14)
            ))
        fig.update_layout(annotations=annotations)

    # Add a separate invisible scatter for the colorbar
    if color_values:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                colorscale=colorscale,
                cmin=vmin, cmax=vmax,
                color=[vmin],
                showscale=True,
                colorbar=dict(title=colorbar_title)
            ),
            showlegend=False
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        plot_bgcolor="white",
        xaxis=dict(visible=False, showgrid=False),
        yaxis=dict(visible=False, showgrid=False)
    )
    # Save if requested
    if save_html:
        fig.write_html(save_html)
    if save_png:
        fig.write_image(save_png)
    return fig


def k_fields_plot(
    field_dfs: Dict[str, pd.DataFrame],
    df_main: pd.DataFrame,
    fields: List[str] = ["keywords", "sources"],
    customs: Dict[str, pd.DataFrame] = {},
    top_n: Union[int, List[int]] = 10,
    color_option: str = "Average year",
    save_png: Optional[str] = None,
    save_html: Optional[str] = None,
    label_maps: Optional[Dict[str, str]] = None
) -> go.Figure:
    """
    Prepare and plot a k-field Sankey diagram with colorbar and field names.
    """
    # collect DataFrames
    dfs: List[pd.DataFrame] = []
    for fld in fields:
        if fld in customs:
            dfs.append(customs[fld])
        elif fld in field_dfs:
            dfs.append(field_dfs[fld])
        else:
            raise KeyError(f"No data for field '{fld}'")
    # normalize top_n
    if isinstance(top_n, int):
        top_n_list = [top_n] * len(dfs)
    else:
        top_n_list = list(top_n)
    # choose color series/function
    color_map = {
        "Average year": ("Year", np.mean),
        "Citations per document": ("Cited by", np.mean)
    }
    if color_option in color_map:
        col_name, color_func = color_map[color_option]
        color_series = df_main[col_name]
    else:
        color_series, color_func = None, np.mean
    # prepare data
    links_df, labels, color_values, counts, group_ids = prepare_for_sankey(
        dataframes=dfs,
        top_n=top_n_list,
        label_maps=label_maps,
        color_series=color_series,
        color_func=color_func,
        all_pairs=False
    )
    # plot
    fig = plot_sankey(
        links_df=links_df,
        labels=labels,
        color_values=color_values,
        group_ids=group_ids,
        field_names=fields,
        save_png=save_png,
        save_html=save_html,
        colorscale="Viridis",
        colorbar_title=color_option
    )
    return fig


"""
# Sample usage
df_authors = pd.DataFrame({"A": [1, 0, 1], "B": [0, 1, 1]})
df_keywords = pd.DataFrame({"X": [1, 1, 0], "Y": [0, 1, 1]})
df_countries = pd.DataFrame({"US": [1, 0, 1], "UK": [0, 1, 0]})
df_main = pd.DataFrame({"Year": [2019, 2020, 2021], "Cited by": [5, 10, 3]})
field_dfs = {"authors": df_authors, "keywords": df_keywords, "countries": df_countries}
fig = k_fields_plot(
    field_dfs=field_dfs,
    df_main=df_main,
    fields=["authors", "keywords", "countries"],
    top_n=2,
    color_option=None,
    save_png="sankey.png",
    save_html="sankey.html",
    label_maps={"A": "Alice", "X": "Xterm"}
)
print("Plots saved: sankey.png, sankey.html")
"""

# Plotting of networks


def plot_network(
    G,
    partition_attr=None,
    color_attr=None,
    size_attr=None,
    layout="spring",
    cmap_name_continuous="viridis",
    cmap_name_discrete="tab10",
    size_scale=300,
    default_node_color="blue",
    default_node_size=300,
    label_fontsize=12,
    edge_width=1.0,
    edge_alpha=1.0,
    log_scale=True,
    curved_edges=True,
    edge_curve_rad=0.1,
    fix_max_size=True,
    node_alpha=0.7,
    node_shape="o",
    show_colorbar=True,
    show_frame=False,
    pos=None,
    layout_kwargs=None,
    min_edge_width=0.5,
    max_edge_width=5.0,
    adjust_labels=True,
    largest_component=True,
    filename=None,
    **kwargs
):
    """
    Plot a network graph with optional attributes for node partitioning, coloring, sizing, and edge curvature.

    Parameters
    ----------
    G : networkx.Graph
        The network graph to plot.
    partition_attr : str, optional
        Node attribute for grouping into communities (for coloring).
    color_attr : str, optional
        Node attribute for coloring by a scalar value.
    size_attr : str, optional
        Node attribute for node size.
    layout : str, optional
        Graph layout: 'spring', 'circular', 'kamada_kawai', 'shell'.
    cmap_name_continuous : str, optional
        Colormap name for continuous color attributes.
    cmap_name_discrete : str, optional
        Colormap name for discrete partitions.
    size_scale : float, optional
        Max node size after scaling.
    default_node_color : color, optional
        Default node color if no attribute is provided.
    default_node_size : float, optional
        Default node size if no attribute is provided.
    label_fontsize : float, optional
        Base label font size.
    edge_width : float, optional
        Default edge width if edge weights are not present.
    edge_alpha : float, optional
        Edge transparency.
    log_scale : bool, optional
        Whether to use log scaling for node sizes.
    curved_edges : bool, optional
        Whether to plot all edges as curved.
    edge_curve_rad : float, optional
        Curve radius for curved edges.
    fix_max_size : bool, optional
        Normalize node sizes to size_scale.
    node_alpha : float, optional
        Node transparency.
    node_shape : str, optional
        Node shape for matplotlib.
    show_colorbar : bool, optional
        Whether to show a colorbar (if color_attr is used).
    show_frame : bool, optional
        Whether to show plot frame and axes.
    pos : dict, optional
        Node positions (for layout caching).
    layout_kwargs : dict, optional
        Additional arguments for the layout function.
    min_edge_width : float, optional
        Minimum edge width after normalization.
    max_edge_width : float, optional
        Maximum edge width after normalization.
    adjust_labels : bool, optional
        Use adjustText for label collision avoidance.
    largest_component : bool, optional
        Whether to restrict plotting to the largest connected component (default True).
    filename : str, optional
        Base filename to save plot in PNG, SVG, and PDF.
    **kwargs : dict
        Additional arguments passed to draw_networkx_nodes.

    Returns
    -------
    pos : dict
        Node positions used for the plot.
    """


    if largest_component:
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            largest = max(components, key=len)
            G = G.subgraph(largest).copy()

    if layout_kwargs is None:
        layout_kwargs = {}
    ax = plt.gca()
    layouts = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "shell": nx.shell_layout
    }

    if not hasattr(G, "_pos_cache"):
        G._pos_cache = {}

    if pos is None:
        if layout in G._pos_cache:
            pos = G._pos_cache[layout]
        else:
            pos = layouts.get(layout, nx.spring_layout)(G, **layout_kwargs)
            G._pos_cache[layout] = pos

    nodes = list(G.nodes())

    # NODE COLORS
    node_colors = {}
    sm = None
    if partition_attr:
        comms = {}
        key = partition_attr if partition_attr.startswith("partition_") else f"partition_{partition_attr}"
        for n, data in G.nodes(data=True):
            cid = data.get(key, 0)
            comms.setdefault(cid, []).append(n)
        cmap = cm.get_cmap(cmap_name_discrete, len(comms))
        for idx, members in enumerate(comms.values()):
            col = cmap(idx)
            for n in members:
                node_colors[n] = col
    elif color_attr:
        vals = [G.nodes[n].get(color_attr, 0) for n in nodes]
        norm = mcolors.Normalize(vmin=min(vals), vmax=max(vals))
        cmap = cm.get_cmap(cmap_name_continuous)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        for n in nodes:
            node_colors[n] = cmap(norm(G.nodes[n].get(color_attr, 0)))
    else:
        for n in nodes:
            node_colors[n] = default_node_color

    # NODE SIZES
    raw = []
    for n in nodes:
        if size_attr:
            raw.append(G.nodes[n].get(size_attr, default_node_size))
        else:
            if G.has_edge(n, n):
                data = G.get_edge_data(n, n)
                if isinstance(data, dict) and not any(isinstance(v, dict) for v in data.values()):
                    weight = data.get("weight", 1)
                else:
                    weight = sum(attr.get("weight", 1) for attr in data.values())
                raw.append(weight)
            else:
                raw.append(default_node_size)
    vals = [(math.log(v + 1) if log_scale else v) for v in raw]
    if fix_max_size:
        max_val = max(vals) if vals else 1
        sizes = [(v / max_val) * size_scale for v in vals]
    else:
        sizes = [v * size_scale for v in vals]

    # DRAW NODES
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_shape=node_shape,
        node_color=[node_colors[n] for n in nodes],
        node_size=sizes,
        alpha=node_alpha,
        linewidths=0,
        edgecolors="none",
        ax=ax,
        **kwargs
    )

    # EDGE PREP + DRAW
    def normalize_weights(weights):
        if not weights:
            return []
        min_w, max_w = min(weights), max(weights)
        if max_w == min_w:
            return [(min_edge_width + max_edge_width) / 2] * len(weights)
        return [
            min_edge_width + (w - min_w) / (max_w - min_w) * (max_edge_width - min_edge_width)
            for w in weights
        ]

    edgelist = []
    edge_colors = []
    edge_weights = []
    for u, v in G.edges():
        if u == v:
            continue
        c1 = mcolors.to_rgba(node_colors.get(u, default_node_color))
        c2 = mcolors.to_rgba(node_colors.get(v, default_node_color))
        avg_color = tuple((c1[i] + c2[i]) / 2 for i in range(4))
        weight = G[u][v].get("weight", 1)
        edgelist.append((u, v))
        edge_colors.append(avg_color)
        edge_weights.append(weight)
    edge_widths = normalize_weights(edge_weights)

    edge_kwargs = dict(
        G=G,
        pos=pos,
        edgelist=edgelist,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=edge_alpha,
        ax=ax
    )
    if curved_edges:
        edge_kwargs["connectionstyle"] = f"arc3,rad={edge_curve_rad}"
        edge_kwargs["arrows"] = True

    nx.draw_networkx_edges(**edge_kwargs)

    # LABELS
    texts = []
    if sizes:
        max_size = max(sizes)
    else:
        max_size = label_fontsize
    for idx, n in enumerate(nodes):
        x, y = pos[n]
        fs = label_fontsize * (0.5 + 0.5 * (sizes[idx] / max_size)) if max_size > 0 else label_fontsize
        txt = ax.text(x, y, str(n), fontsize=fs, ha="center", va="center", zorder=5)
        texts.append(txt)

    if adjust_labels:
        adjust_text(
            texts,
            only_move={'points': 'y', 'text': 'xy'},
            expand_text=(1.2, 1.2),
            expand_points=(1.2, 1.2),
            force_text=(0.75, 1.0),
            autoalign='xy'
        )

    if color_attr and show_colorbar and sm is not None:
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(color_attr)

    if not show_frame:
        ax.set_frame_on(False)
        ax.axis("off")

    if filename:
        save_plot(filename)
        
    plt.clf()

    return pos






def plot_degree_distribution(G, log_log=False, ax=None, **kwargs):
    """
    Plot the degree distribution histogram.

    Parameters
    ----------
    G : networkx.Graph
    log_log : bool, optional
        If True, use log-log scale.
    ax : matplotlib.axes.Axes, optional
    kwargs : passed to plt.hist

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()
    degrees = [d for _, d in G.degree()]
    ax.hist(degrees, **kwargs)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    if log_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    return ax

# Plotting of citation network and main path


def plot_citation_network(
    G: nx.DiGraph,
    size_dict: Optional[Dict[str, float]] = None,
    color_dict: Optional[Dict[str, float]] = None,
    label_dict: Optional[Dict[str, str]] = None,
    cmap: str = 'viridis',
    arrow_size: int = 10,
    font_size: int = 8,
    node_size_factor: float = 100,
    sqrt_sizes: bool = False,
    edge_width: float = 0.5,
    layout: str = 'kamada_kawai',
    highlight_main_path: bool = False,
    main_path: Optional[List[str]] = None,
    main_path_color: str = 'crimson',
    main_path_width: float = 2.5,
    main_path_style: str = 'solid',
    filename: Optional[str] = None
) -> None:
    """
    Visualize a directed citation network with optional node sizing, coloring,
    labeling, layout choices, and main path highlighting.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph representing the citation network.
    size_dict : dict[str, float], optional
        Dictionary mapping node IDs to size values. If None, in-degree is used.
    color_dict : dict[str, float], optional
        Dictionary mapping node IDs to scalar values for coloring.
    label_dict : dict[str, str], optional
        Dictionary mapping node IDs to label strings.
    cmap : str, default='viridis'
        Matplotlib colormap name used for node colors.
    arrow_size : int, default=10
        Size of the arrowheads on directed edges.
    font_size : int, default=8
        Font size for node labels.
    node_size_factor : float, default=100
        Scaling factor applied to node sizes.
    sqrt_sizes : bool, default=False
        Whether to apply square root scaling to node sizes.
    edge_width : float, default=0.5
        Width of network edges.
    layout : str, default='kamada_kawai'
        Layout algorithm for node positions. Options: 'kamada_kawai', 'spring', 'circular'.
    highlight_main_path : bool, default=False
        If True, highlights the main citation path in the network.
    main_path : list[str], optional
        Optional list of node IDs representing the main citation path. If None and
        `highlight_main_path` is True, the path is computed via `utilsbib.compute_main_path`.
    main_path_color : str, default='crimson'
        Color used to highlight edges along the main path.
    main_path_width : float, default=2.5
        Width of the highlighted main path edges.
    main_path_style : str, default='solid'
        Line style for the main path edges (e.g., 'solid', 'dashed').
    filename : str, optional
        Base filename (without extension) to save the plot. If None, the plot is shown interactively.

    Returns
    -------
    None
    """
    if layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        raise ValueError(f'Unknown layout: {layout}')

    if not isinstance(size_dict, dict):
        size_dict = None
    if size_dict is None:
        size_dict = dict(G.in_degree())

    node_sizes = [size_dict.get(node, 1) * node_size_factor for node in G.nodes()]
    if sqrt_sizes:
        node_sizes = [np.sqrt(s) for s in node_sizes]

    if color_dict is not None:
        node_colors = [color_dict.get(node, 0.5) for node in G.nodes()]
    else:
        node_colors = 'lightblue'

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.get_cmap(cmap))
    nx.draw_networkx_edges(G, pos, width=edge_width, arrows=True, arrowstyle='-|>', arrowsize=arrow_size)

    if label_dict is None:
        label_dict = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=font_size)

    if highlight_main_path:
        if main_path is None:

            main_path = utilsbib.compute_main_path(G)
        main_path_edges = list(zip(main_path[:-1], main_path[1:]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=main_path_edges,
            width=main_path_width,
            edge_color=main_path_color,
            style=main_path_style,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=arrow_size
        )

    if color_dict is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(color_dict.values()), vmax=max(color_dict.values())))
        sm.set_array([])
        plt.colorbar(sm)

    plt.axis('off')
    plt.tight_layout()
    if filename:
        save_plot(filename)
    else:
        plt.show()


def plot_main_path(
    G: nx.DiGraph,
    path: list[str] | None = None,
    size_dict: dict[str, float] | None = None,
    color_dict: dict[str, float] | None = None,
    label_map: dict[str, str] | None = None,
    cmap = plt.cm.viridis,
    edge_color: str = "red",
    edge_width: float = 2.0,
    arrow_size: int = 10,
    font_size: int = 10,
    layout: str = "kamada_kawai",
    filename: Optional[str] = None
) -> None:
    """
    Plot only the main citation path from a directed citation graph,
    with adjustable visuals for nodes, edges, and labels.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph representing the full citation network.
    path : list[str], optional
        List of node IDs forming the main citation path. If None, computed using
        `utilsbib.compute_main_path`.
    size_dict : dict[str, float], optional
        Dictionary mapping node IDs to node sizes.
    color_dict : dict[str, float], optional
        Dictionary mapping node IDs to scalar values for coloring.
    label_map : dict[str, str], optional
        Dictionary mapping node IDs to label strings.
    cmap : matplotlib colormap, default=plt.cm.viridis
        Colormap to use when `color_dict` is provided.
    edge_color : str, default="red"
        Color of the edges along the main path.
    edge_width : float, default=2.0
        Width of the edges along the main path.
    arrow_size : int, default=10
        Size of the arrowheads on the main path.
    font_size : int, default=10
        Font size for node labels.
    layout : str, default="kamada_kawai"
        Layout algorithm to position nodes. Options include: "kamada_kawai", "spring", "circular", "shell".
    filename : str, optional
        Base filename (without extension) to save the plot. If None, the plot is shown interactively.

    Returns
    -------
    None
    """
    if path is None:
        path = utilsbib.compute_main_path(G)
    subG = nx.DiGraph()
    subG.add_nodes_from(path)
    subG.add_edges_from(zip(path, path[1:]))

    layout_funcs = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
        "shell": nx.shell_layout
    }
    pos = layout_funcs.get(layout, nx.kamada_kawai_layout)(subG)

    sizes = ([size_dict.get(n, 300) for n in subG.nodes()] if size_dict else
             [300 + 200 * G.in_degree(n) for n in subG.nodes()])

    if color_dict:
        vals = [color_dict.get(n, 0) for n in subG.nodes()]
        norm = plt.Normalize(vmin=min(vals), vmax=max(vals))
        colors = [cmap(norm(v)) for v in vals]
        numeric = True
    else:
        colors = "orange"
        numeric = False

    labels = ({n: label_map.get(n, n) for n in subG.nodes()} if label_map else
              {n: n for n in subG.nodes()})

    plt.figure(figsize=(8, 6))
    nx.draw(
        subG, pos,
        labels=labels,
        node_size=sizes,
        node_color=colors,
        font_size=font_size,
        arrowsize=arrow_size,
        edge_color=edge_color,
        width=edge_width
    )
    if numeric:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label="Color Value")

    plt.title("Main Citation Path")
    plt.tight_layout()
    if filename:
        save_plot(filename)
    else:
        plt.show()


# historiograph

def layout_historiograph(G):
    """Compute a chronological layout: x = year, y = randomly jittered within each year."""
    

    year_nodes = defaultdict(list)
    for node, attrs in G.nodes(data=True):
        year = attrs.get("year")
        if year is not None:
            year_nodes[year].append(node)

    pos = {}
    for year, nodes in year_nodes.items():
        for i, node in enumerate(sorted(nodes)):
            jitter = np.random.uniform(-0.5, 0.5)
            pos[node] = (year, jitter)

    return pos


def plot_historiograph(
    G,
    pos,
    figsize=(12, 8),
    size_attr=None,
    min_indegree=None,
    min_citations=100,
    min_year=None,
    max_year=None,
    save_as=None,
    dpi=600,
):
    """Draw the historiograph using matplotlib, excluding isolated nodes, loops, and applying filters."""
    plt.figure(figsize=figsize)

    def node_passes_filters(n, d):
        if min_year and d.get("year") < min_year:
            return False
        if max_year and d.get("year") > max_year:
            return False
        if min_citations and d.get("Cited by", 0) < min_citations:
            return False
        if min_indegree and G.in_degree(n) < min_indegree:
            return False
        return True

    filtered_nodes = [n for n, d in G.nodes(data=True) if node_passes_filters(n, d)]
    filtered_graph = G.subgraph(filtered_nodes).copy()
    filtered_graph.remove_edges_from(nx.selfloop_edges(filtered_graph))
    connected_nodes = [n for n in filtered_graph.nodes if filtered_graph.degree(n) > 0]
    subgraph = filtered_graph.subgraph(connected_nodes).copy()

    if size_attr:
        sizes = [subgraph.nodes[n].get(size_attr, 3) * 10 for n in subgraph.nodes]
    else:
        sizes = [300 for _ in subgraph.nodes]

    sub_pos = {k: v for k, v in pos.items() if k in subgraph.nodes}

    nx.draw(subgraph, sub_pos, with_labels=False, arrows=True, node_size=sizes, node_color="lightblue")

    labels = {k: v for k, v in nx.get_node_attributes(subgraph, "title").items()}
    texts = [plt.text(sub_pos[n][0], sub_pos[n][1], labels[n], fontsize=8, ha="center", va="center") for n in labels]
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

    plt.title("Historiograph")
    plt.xlabel("Publication Year")
    plt.axis("off")
    plt.tight_layout()

    if save_as:
        save_plot(save_as, dpi=dpi)

    plt.show()

# specific networks

def plot_country_collab_network(matrix_df, threshold=1, figsize=(12, 12), layout_func="spring", filename_base=None):
    """
    Plots a network graph of country collaborations above a threshold.

    Parameters:
    matrix_df (pd.DataFrame): Symmetric collaboration matrix.
    threshold (int): Minimum collaboration count to include an edge.
    figsize (tuple): Size of the figure in inches.
    layout_func (callable): NetworkX layout function (e.g., nx.spring_layout).
    filename_base (str or None): If provided, saves the plot as PNG, SVG, and PDF using this base name.
    """
    if matrix_df.empty:
        print("Empty matrix: network not generated.")
        return

    G = nx.Graph()
    for i in matrix_df.index:
        for j in matrix_df.columns:
            weight = matrix_df.loc[i, j]
            if i != j and weight >= threshold:
                G.add_edge(i, j, weight=weight)

    layout = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
        "shell": nx.shell_layout
    }[layout_func]

    if len(G.nodes) == 0:
        print("No edges above threshold: network not generated.")
        return

    pos = layout(G)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, width=[w * 0.1 for w in edge_weights])
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Country Collaboration Network")
    plt.axis("off")
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base)

    plt.show()

# specific heatmap

def plot_country_collab_heatmap(matrix_df, top_n=50, figsize=(12, 10), cmap="Blues", annotate=False, filename_base=None):
    """
    Plots a heatmap of the country collaboration matrix, optionally limited to the top N countries by total collaboration.

    Parameters:
    matrix_df (pd.DataFrame): Symmetric collaboration matrix.
    top_n (int): Number of top countries (by total collaborations) to include.
    figsize (tuple): Size of the figure in inches.
    cmap (str): Colormap for heatmap shading.
    annotate (bool): Whether to show collaboration counts in each cell.
    filename_base (str or None): If provided, saves the plot as PNG, SVG, and PDF using this base name.
    """
    if matrix_df.empty:
        print("Empty matrix: heatmap not generated.")
        return

    # Compute total collaborations and select top N countries
    totals = matrix_df.sum(axis=1) + matrix_df.sum(axis=0)
    top_countries = totals.sort_values(ascending=False).head(top_n).index
    matrix_top = matrix_df.loc[top_countries, top_countries]

    plt.figure(figsize=figsize)
    sns.heatmap(matrix_top, cmap=cmap, square=True, annot=annotate, fmt="d", 
                cbar_kws={"label": "Collaboration Count"})
    plt.title("Country Collaboration Matrix (Top {} Countries)".format(top_n))
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base)

    plt.show()


# Thematic map and evolution

def save_sankey(diagram, filename_base, formats=("png", "svg", "pdf", "html")):
    """
    Save a Plotly Sankey diagram to multiple formats.

    Parameters
    ----------
    diagram : plotly.graph_objects.Figure
        Sankey diagram figure.
    filename_base : str
        Base filename without extension.
    formats : tuple of str, optional
        File formats to save (png, svg, pdf, html).
    """
    for ext in formats:
        path = f"{filename_base}.{ext}"
        if ext == "html":
            diagram.write_html(path)
        else:
            diagram.write_image(path)


def plot_thematic_map(
    G,
    partition_attr,
    max_dot_size=200,
    quadrant_labels=False,
    items_per_cluster=3,
    cmap_name="viridis",
    figsize=(8, 6),
    max_clusters=None,
    min_cluster_size=5,
    include_cluster_label=False,
    color_df=None,
    color_col=None,
    save_plot_base=None,
    dpi=600,
    ax=None,
    item_sep="\n"
):
    """
    Plot thematic map of clusters with axis labels, no tick values,
    optional spaced quadrant labels, and non-overlapping cluster annotations.
    Optionally save figure to files using save_plot.

    Parameters
    ----------
    G : networkx.Graph
    partition_attr : str
        Node attribute name (without or with "partition_" prefix).
    max_dot_size : float, optional
        Size of the largest cluster marker.
    quadrant_labels : dict or None, optional
        Mapping quadrant keys ("NE","NW","SW","SE") to labels;
        if None, quadrant labels are not shown.
    items_per_cluster : int, optional
        Number of top nodes (by degree centrality) to list per cluster in plot.
    cmap_name : str, optional
        Colormap for clusters.
    figsize : tuple, optional
        Figure size when creating new figure.
    max_clusters : int, optional
        Limit to this many largest clusters.
    min_cluster_size : int, optional
        Exclude clusters smaller than this size.
    include_cluster_label : bool, optional
        Whether to prefix each label with the cluster ID.
    color_df : pandas.DataFrame, optional
        DataFrame with cluster IDs and a color column.
    color_col : str, optional
        Column name in color_df for coloring clusters.
    save_plot_base : str, optional
        Base filename (without extension) for saving the plot; if None, plot is not auto-saved.
    dpi : int, optional
        Resolution in dots per inch for saving.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot onto. If None, a new figure and axis are created.
    item_sep : str, optional
        Separator string between item labels.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    key = partition_attr if partition_attr.startswith("partition_") else f"partition_{partition_attr}"
    metrics = utilsbib.compute_cluster_metrics(G, key)
    cids = [c for c in metrics if metrics[c]["size"] >= min_cluster_size]
    if max_clusters and len(cids) > max_clusters:
        cids = sorted(cids, key=lambda c: metrics[c]["size"], reverse=True)[:max_clusters]

    densities = [metrics[c]["density"] for c in cids]
    centrals = [metrics[c]["avg_degree_centrality"] for c in cids]
    sizes_raw = [metrics[c]["size"] for c in cids]
    max_raw = max(sizes_raw) if sizes_raw else 1
    sizes = [(s / max_raw) * max_dot_size for s in sizes_raw]

    if color_df is not None and color_col is not None:
        df_col = color_df.set_index(key)
        vals = [df_col.loc[c, color_col] if c in df_col.index else 0 for c in cids]
        norm = colors.Normalize(vmin=min(vals), vmax=max(vals))
        cmap = cm.get_cmap(cmap_name)
        colors_list = [cmap(norm(v)) for v in vals]
    else:
        colors_list = ["lightgrey"] * len(cids)

    deg_c = nx.degree_centrality(G)
    text_objs = []
    for i, cid in enumerate(cids):
        ax.scatter(densities[i], centrals[i], s=sizes[i], color=colors_list[i], alpha=0.7)
        nodes = [n for n, d in G.nodes(data=True) if d.get(key) == cid]
        top_nodes = sorted(nodes, key=lambda n: deg_c.get(n, 0), reverse=True)[:items_per_cluster]
        labels = [str(n) for n in top_nodes]
        if include_cluster_label:
            labels.insert(0, str(cid))
        txt = ax.text(densities[i], centrals[i], item_sep.join(labels), fontsize=8)
        text_objs.append(txt)

    x_mid = sum(densities) / len(densities) if densities else 0
    y_mid = sum(centrals) / len(centrals) if centrals else 0
    ax.axvline(x_mid, color="grey", lw=0.8, linestyle="--")
    ax.axhline(y_mid, color="grey", lw=0.8, linestyle="--")
    adjust_text(text_objs, ax=ax, only_move={"points": "y", "texts": "y"})

    if quadrant_labels:
        ql = {"NE": "Motor Themes", "NW": "Niche Themes",
                           "SW": "Emerging or Declining Themes",
                           "SE": "Basic Themes"}
    else:
        ql = {}

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    dx = (x_max - x_min) * 0.1
    dy = (y_max - y_min) * 0.1
    ax.text(x_max - dx, y_max - dy, ql.get("NE", ""), ha="right", va="top", fontsize=9, color="grey")
    ax.text(x_min + dx, y_max - dy, ql.get("NW", ""), ha="left", va="top", fontsize=9, color="grey")
    ax.text(x_min + dx, y_min + dy, ql.get("SW", ""), ha="left", va="bottom", fontsize=9, color="grey")
    ax.text(x_max - dx, y_min + dy, ql.get("SE", ""), ha="right", va="bottom", fontsize=9, color="grey")

    ax.set_xlabel("Density")
    ax.set_ylabel("Centrality")
    ax.set_xticks([])
    ax.set_yticks([])

    if save_plot_base:
        save_plot(save_plot_base, dpi=dpi)

    return ax


def plot_thematic_evolution(
    graphs,
    partition_attr,
    time_labels=None,
    map_kwargs=None,
    sankey_kwargs=None,
    save_maps_base=None,
    map_formats=("png", "svg", "pdf"),
    save_sankey_base=None,
    sankey_formats=("png", "svg", "pdf", "html"),
    top_k=2,
    item_sep="\n"
):
    """
    Plot a sequence of thematic maps and a Sankey diagram showing cluster evolution across time.

    Instead of showing cluster IDs and timepoint labels, display top_k nodes per cluster.
    Optionally save maps and Sankey diagram to files.

    Parameters
    ----------
    graphs : list of networkx.Graph
        Sequence of graphs representing different time points.
    partition_attr : str
        Node attribute name for cluster assignment.
    time_labels : list of str, optional
        Labels for each time point; not displayed when top_k is used.
    map_kwargs : dict, optional
        Keyword args passed to plot_thematic_map.
    sankey_kwargs : dict, optional
        Keyword args for plotly Sankey layout.
    save_maps_base : str, optional
        Base filename (without extension) for saving thematic maps; if None, maps are not auto-saved.
    map_formats : tuple of str, optional
        File formats to save thematic maps (png, svg, pdf).
    save_sankey_base : str, optional
        Base filename (without extension) to save Sankey diagram; if None, Sankey is not auto-saved.
    sankey_formats : tuple of str, optional
        Formats to save Sankey (png, svg, pdf, html).
    top_k : int, optional
        Number of top nodes (by degree centrality) per cluster to display.
    item_sep : str, optional
        Separator string between items in Sankey node labels.

    Returns
    -------
    tuple
        (matplotlib.figure.Figure, plotly.graph_objects.Figure)
    """
    if map_kwargs is None:
        map_kwargs = {}
    if sankey_kwargs is None:
        sankey_kwargs = {}
    n = len(graphs)
    fig_maps, axes = plt.subplots(
        1, n, figsize=(n * map_kwargs.get("figsize", (8, 6))[0], map_kwargs.get("figsize", (8, 6))[1])
    )
    if n == 1:
        axes = [axes]

    # plot thematic maps without titles
    for ax, G in zip(axes, graphs):
        plot_thematic_map(
            G,
            partition_attr,
            quadrant_labels=None,
            ax=ax,
            item_sep=item_sep,
            **map_kwargs
        )

    plt.tight_layout()

    # auto-save thematic maps
    if save_maps_base:
        for ext in map_formats:
            path = f"{save_maps_base}.{ext}"
            fig_maps.savefig(path, bbox_inches="tight")

    # prepare Sankey data with top_k labels
    clusters_list = []
    degc_list = []
    for G in graphs:
        key = partition_attr if partition_attr.startswith("partition_") else f"partition_{partition_attr}"
        clust = {}
        deg_c = nx.degree_centrality(G)
        for node, d in G.nodes(data=True):
            cid = d.get(key)
            clust.setdefault(cid, set()).add(node)
        clusters_list.append(clust)
        degc_list.append(deg_c)

    node_labels = []
    offsets = []
    cum = 0
    for clust, deg_c in zip(clusters_list, degc_list):
        offsets.append(cum)
        for cid, nodes in clust.items():
            sorted_nodes = sorted(nodes, key=lambda n: deg_c.get(n, 0), reverse=True)[:top_k]
            label = item_sep.join(str(n) for n in sorted_nodes)
            node_labels.append(label)
        cum += len(clust)

    sources, targets, values = [], [], []
    for t in range(n - 1):
        src_cids = list(clusters_list[t].keys())
        tgt_cids = list(clusters_list[t + 1].keys())
        for i, src in enumerate(src_cids):
            for j, tgt in enumerate(tgt_cids):
                val = len(clusters_list[t][src] & clusters_list[t + 1][tgt])
                if val > 0:
                    sources.append(offsets[t] + i)
                    targets.append(offsets[t + 1] + j)
                    values.append(val)

    sankey_node = dict(label=node_labels)
    sankey_link = dict(source=sources, target=targets, value=values)
    fig_sankey = go.Figure(data=[go.Sankey(node=sankey_node, link=sankey_link)], **sankey_kwargs)
    fig_sankey.update_layout(title_text="Thematic Evolution Sankey", font_size=10)

    # auto-save Sankey diagram
    if save_sankey_base:
        save_sankey(fig_sankey, save_sankey_base, formats=sankey_formats)

    return fig_maps, fig_sankey


# plotting of relationships

def plot_correspondence_analysis(
    row_coords: pd.DataFrame,
    col_coords: pd.DataFrame,
    explained_inertia: list,
    df_relation: pd.DataFrame,
    figsize=(8, 6),
    annotate=True,
    alpha=0.8,
    size_scale=300,
    use_size: bool = True,
    filename_base: str = None,
    dpi: int = 600,
    row_label_name: str = "Rows",
    col_label_name: str = "Columns",
    title: str = "Correspondence Analysis with Frequencies"
):
    """
    Plot 2D correspondence analysis with optional scaling by frequency and label customization.

    Parameters:
        row_coords (pd.DataFrame): Row coordinates in CA space.
        col_coords (pd.DataFrame): Column coordinates in CA space.
        explained_inertia (list): Variance explained by each axis.
        df_relation (pd.DataFrame): Original contingency table (used for frequencies).
        figsize (tuple): Size of the figure.
        annotate (bool): Whether to annotate points with labels.
        alpha (float): Point transparency.
        size_scale (float): Scaling factor for point sizes (if used).
        use_size (bool): Whether to scale point size by marginal frequency.
        filename_base (str): If provided, saves plot to PNG, SVG, and PDF.
        dpi (int): Resolution for saved figures.
        row_label_name (str): Legend name for row group.
        col_label_name (str): Legend name for column group.
        title (str): Plot title. If None, no title is shown.
    """
    fig, ax = plt.subplots(figsize=figsize)

    row_freq = df_relation.sum(axis=1)
    col_freq = df_relation.sum(axis=0)

    row_sizes = (row_freq.loc[row_coords.index] / row_freq.max() * size_scale) if use_size else size_scale
    col_sizes = (col_freq.loc[col_coords.index] / col_freq.max() * size_scale) if use_size else size_scale

    # Plot rows
    ax.scatter(
        row_coords.iloc[:, 0], row_coords.iloc[:, 1],
        c='tab:blue', s=row_sizes, alpha=alpha, label=row_label_name
    )

    # Plot columns
    ax.scatter(
        col_coords.iloc[:, 0], col_coords.iloc[:, 1],
        c='tab:red', s=col_sizes, alpha=alpha, marker='^', label=col_label_name
    )

    # Annotate
    if annotate:
        texts = []
        for label, (x, y) in row_coords.iloc[:, :2].iterrows():
            texts.append(ax.text(x, y, label, fontsize=8, color='tab:blue'))
        for label, (x, y) in col_coords.iloc[:, :2].iterrows():
            texts.append(ax.text(x, y, label, fontsize=8, color='tab:red'))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    # Axes and layout
    ax.set_xlabel(f"Dimension 1 ({explained_inertia[0]*100:.1f}%)")
    ax.set_ylabel(f"Dimension 2 ({explained_inertia[1]*100:.1f}%)")
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(False)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi=dpi)

    plt.show()
    

def save_plot(filename_base, dpi=600):
    """
    Save current matplotlib figure to PNG, SVG, and PDF with tight layout.

    Parameters:
        filename_base (str): Path without file extension.
        dpi (int): Resolution of the saved figures.
    """
    for ext in ["png", "svg", "pdf"]:
        path = f"{filename_base}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=dpi)


def plot_residual_heatmap(
    residuals_df: pd.DataFrame,
    center: float = 0.0,
    cmap: str = "coolwarm",
    figsize=(10, 8),
    annotate: bool = False,
    square: bool = True,
    filename_base: str = None,
    dpi: int = 600,
    title: str = "Standardized Pearson Residuals",
    row_label: str = None,
    col_label: str = None, **kwargs
):
    """
    Plot a heatmap of Pearson residuals with optional customization.

    Parameters:
        residuals_df (pd.DataFrame): DataFrame of standardized residuals.
        center (float): Value at center of colormap. Typically 0.
        cmap (str): Seaborn/matplotlib colormap.
        figsize (tuple): Size of figure.
        annotate (bool): Whether to annotate heatmap cells with values.
        square (bool): Whether to enforce square aspect ratio for cells.
        filename_base (str): If provided, saves plot to PNG, SVG, PDF.
        dpi (int): Resolution for saved images.
        title (str): Title of the plot. Use None to omit.
        row_label (str): Label for y-axis. If None, uses DataFrame index name.
        col_label (str): Label for x-axis. If None, uses DataFrame column name.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        residuals_df,
        center=center,
        cmap=cmap,
        annot=annotate,
        fmt=".2f" if annotate else "",
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        square=square,
        cbar_kws={"label": "Residual"},
        **kwargs
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel(col_label or residuals_df.columns.name or "Columns")
    ax.set_ylabel(row_label or residuals_df.index.name or "Rows")
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi=dpi)

    plt.show()

def plot_bipartite_network(
    B: nx.Graph,
    row_nodes: list,
    col_nodes: list,
    node_size_scale: float = 200,
    edge_alpha: float = 0.3,
    same_size: bool = False,
    weight_threshold: float = 0,
    show_edge_weights: bool = False,
    edge_width_scale: float = 1.0,
    figsize=(10, 8),
    title: str = None,
    filename_base: str = None,
    dpi: int = 600,
    row_label_name: str = "Rows",
    col_label_name: str = "Columns"
):
    """
    Visualize a bipartite network with label adjustment, thresholding, and edge weight rendering.

    Parameters:
        B (nx.Graph): Bipartite graph.
        row_nodes (list): Row-type nodes.
        col_nodes (list): Column-type nodes.
        node_size_scale (float): Scaling factor for node size (by degree).
        edge_alpha (float): Edge transparency.
        same_size (bool): If True, all nodes have the same size.
        weight_threshold (float): Minimum edge weight to include.
        show_edge_weights (bool): If True, edge width is scaled by weight.
        edge_width_scale (float): Factor to scale edge width (default 1.0).
        figsize (tuple): Size of the figure.
        title (str): Optional plot title.
        filename_base (str): If provided, saves to PNG/SVG/PDF.
        dpi (int): DPI for saved files.
        row_label_name (str): Legend label for row nodes.
        col_label_name (str): Legend label for column nodes.
    """

    # Filter edges
    edges_to_plot = [
        (u, v) for u, v, d in B.edges(data=True)
        if d.get("weight", 1) >= weight_threshold
    ]
    filtered_nodes = set(u for u, v in edges_to_plot) | set(v for u, v in edges_to_plot)
    B_sub = B.subgraph(filtered_nodes).copy()

    pos = nx.spring_layout(B_sub, seed=42, k=0.15)
    degrees = dict(B_sub.degree())

    row_sizes = [node_size_scale if same_size else degrees[n] * node_size_scale for n in row_nodes if n in B_sub]
    col_sizes = [node_size_scale if same_size else degrees[n] * node_size_scale for n in col_nodes if n in B_sub]

    fig, ax = plt.subplots(figsize=figsize)

    # Nodes
    nx.draw_networkx_nodes(B_sub, pos, nodelist=[n for n in row_nodes if n in B_sub],
                           node_color="tab:blue", node_size=row_sizes, alpha=0.8, label=row_label_name)
    nx.draw_networkx_nodes(B_sub, pos, nodelist=[n for n in col_nodes if n in B_sub],
                           node_color="tab:red", node_shape="s", node_size=col_sizes, alpha=0.8, label=col_label_name)

    # Edge weights
    edge_weights = [B_sub[u][v].get("weight", 1) for u, v in edges_to_plot]
    if show_edge_weights:
        scaled_widths = [w * edge_width_scale for w in edge_weights]
    else:
        scaled_widths = 1

    nx.draw_networkx_edges(B_sub, pos, edgelist=edges_to_plot, width=scaled_widths, alpha=edge_alpha, edge_color="gray")

    # Node labels
    texts = [ax.text(pos[n][0], pos[n][1], n, fontsize=8) for n in B_sub.nodes]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=row_label_name, markerfacecolor="tab:blue", markersize=8),
        plt.Line2D([0], [0], marker="s", color="w", label=col_label_name, markerfacecolor="tab:red", markersize=8)
    ]
    ax.legend(handles=handles, fontsize=8)

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi=dpi)

    plt.show()
    
def plot_top_n_pairs(
    sorted_pairs_df: pd.DataFrame,
    metric_column: str = "Residual",
    top_n: int = 20,
    size_column: str = None,
    size_scale: float = 90,
    color_map: str = None,
    center_color: float = 0.0,
    figsize=(10, 6),
    title: str = "Top-N Row-Column Associations",
    x_label: str = "Row",
    y_label: str = "Column",
    filename_base: str = None,
    dpi: int = 600,
    show_colorbar: bool = True,
    show_guides: bool = False,
    sign: str = "positive"  # "positive", "negative", "all"
):
    """
    Scatter plot of top-N (row, column) pairs by the given metric,
    with customizable axis labels, only showing involved items, and optional grid guides.

    Parameters:
        sorted_pairs_df (pd.DataFrame): Should contain 'Row', 'Column', and metric_column.
        metric_column (str): Column name for coloring (e.g., 'Residual' or 'LogRatio').
        top_n (int): Number of strongest associations to plot.
        size_column (str): Optional column name for dot size scaling.
        size_scale (float): Factor for marker size scaling (default 90).
        color_map (str): Colormap for points. If None, chosen by sign.
        center_color (float): Value to center colormap at (typically 0).
        figsize (tuple): Figure size.
        title (str): Plot title.
        x_label (str): Custom label for x-axis.
        y_label (str): Custom label for y-axis.
        filename_base (str): If set, saves PNG, SVG, PDF.
        dpi (int): Resolution for saving.
        show_colorbar (bool): Whether to show a colorbar legend.
        show_guides (bool): If True, show guide lines from dot to axes.
        sign (str): "positive" (default), "negative", or "all" residuals.
    """
    # Filter by sign if required
    df_top = sorted_pairs_df.copy()
    if sign == "positive":
        df_top = df_top[df_top[metric_column] > 0]
    elif sign == "negative":
        df_top = df_top[df_top[metric_column] < 0]
    # Take top N by abs(metric)
    df_top["abs_metric"] = df_top[metric_column].abs()
    df_top = df_top.sort_values(by="abs_metric", ascending=False).head(top_n)

    # Only show axis ticks for used labels
    used_x = sorted(df_top["Row"].astype(str).unique())
    used_y = sorted(df_top["Column"].astype(str).unique())
    x_lut = {label: i for i, label in enumerate(used_x)}
    y_lut = {label: i for i, label in enumerate(used_y)}
    x_coords = df_top["Row"].astype(str).map(x_lut)
    y_coords = df_top["Column"].astype(str).map(y_lut)

    # Size
    if size_column and size_column in df_top.columns:
        sizes = df_top[size_column] / df_top[size_column].max() * size_scale
    else:
        sizes = size_scale

    # Adaptive color limits
    if sign in ["positive", "negative"]:
        vmin = df_top[metric_column].min()
        vmax = df_top[metric_column].max()
    else:
        abs_max = df_top[metric_column].abs().max()
        vmin, vmax = -abs_max, abs_max

    # Smart colormap selection
    if color_map is None:
        if sign == "positive":
            color_map = "Reds"
        elif sign == "negative":
            color_map = "Blues"
        else:
            color_map = "coolwarm"

    fig, ax = plt.subplots(figsize=figsize)

    # Draw guide lines if requested
    if show_guides:
        for xi, yi in zip(x_coords, y_coords):
            ax.axvline(x=xi, color="lightgray", linestyle="--", lw=0.7, zorder=1)
            ax.axhline(y=yi, color="lightgray", linestyle="--", lw=0.7, zorder=1)

    # Scatter plot
    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=df_top[metric_column],
        cmap=color_map,
        s=sizes,
        vmin=vmin,
        vmax=vmax,
        edgecolor="k",
        alpha=0.85,
        zorder=2
    )
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(metric_column)

    # Set axis ticks and labels for used items only
    ax.set_xticks(range(len(used_x)))
    ax.set_xticklabels(used_x, rotation=90)
    ax.set_yticks(range(len(used_y)))
    ax.set_yticklabels(used_y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    if filename_base:
        save_plot(filename_base, dpi=dpi)
    plt.show()