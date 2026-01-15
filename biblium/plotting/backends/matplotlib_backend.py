# -*- coding: utf-8 -*-
"""
Matplotlib Backend Implementation.

Provides the Matplotlib implementation of the PlotBackend interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from biblium.plotting.backends.base import PlotBackend
from biblium.plotting.config import PlotConfig


class MatplotlibBackend(PlotBackend):
    """
    Matplotlib implementation of PlotBackend.
    
    This backend uses matplotlib for all plotting operations.
    """
    
    name = "matplotlib"
    
    def __init__(self, config: Optional[PlotConfig] = None):
        super().__init__(config)
        
        # Import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib
        
        self.plt = plt
        self.matplotlib = matplotlib
    
    def _create_figure(self, config: PlotConfig) -> Tuple[Any, Any]:
        """Create figure and axes with config settings."""
        # Convert pixels to inches (assuming 100 dpi for screen)
        width_inches = config.width / config.dpi
        height_inches = config.height / config.dpi
        
        fig, ax = self.plt.subplots(
            figsize=(width_inches, height_inches),
            facecolor=config.background_color,
        )
        
        ax.set_facecolor(config.plot_background_color)
        
        return fig, ax
    
    def _apply_config(self, ax: Any, config: PlotConfig) -> None:
        """Apply config settings to axes."""
        # Title
        if config.title:
            ax.set_title(
                config.title,
                fontsize=config.title_font_size,
                fontweight=config.title_font_weight,
                color=config.title_color,
                loc=config.title_position,
            )
        
        # Axis labels
        if config.xlabel:
            ax.set_xlabel(
                config.xlabel,
                fontsize=config.xlabel_font_size,
                color=config.label_color,
            )
        
        if config.ylabel:
            ax.set_ylabel(
                config.ylabel,
                fontsize=config.ylabel_font_size,
                color=config.label_color,
            )
        
        # Axis limits
        if config.xlim:
            ax.set_xlim(config.xlim)
        if config.ylim:
            ax.set_ylim(config.ylim)
        
        # Axis scale
        if config.xscale != "linear":
            ax.set_xscale(config.xscale)
        if config.yscale != "linear":
            ax.set_yscale(config.yscale)
        
        # Invert axes
        if config.invert_xaxis:
            ax.invert_xaxis()
        if config.invert_yaxis:
            ax.invert_yaxis()
        
        # Tick settings
        ax.tick_params(
            axis="x",
            labelsize=config.xtick_font_size,
            labelcolor=config.tick_color,
            rotation=config.xtick_rotation,
        )
        ax.tick_params(
            axis="y",
            labelsize=config.ytick_font_size,
            labelcolor=config.tick_color,
            rotation=config.ytick_rotation,
        )
        
        if not config.show_xticks:
            ax.set_xticks([])
        if not config.show_yticks:
            ax.set_yticks([])
        
        # Grid
        if config.show_grid:
            linestyle = {"solid": "-", "dashed": "--", "dotted": ":"}
            ax.grid(
                True,
                color=config.grid_color,
                alpha=config.grid_alpha,
                linestyle=linestyle.get(config.grid_style, "--"),
                linewidth=config.grid_width,
                axis=config.grid_axis,
            )
        else:
            ax.grid(False)
        
        # Spines
        ax.spines["top"].set_visible(config.show_top_spine)
        ax.spines["right"].set_visible(config.show_right_spine)
        ax.spines["bottom"].set_visible(config.show_bottom_spine)
        ax.spines["left"].set_visible(config.show_left_spine)
        
        for spine in ax.spines.values():
            spine.set_color(config.spine_color)
            spine.set_linewidth(config.spine_width)
    
    def _add_legend(self, ax: Any, config: PlotConfig) -> None:
        """Add legend with config settings."""
        if not config.show_legend:
            return
        
        # Map position names
        loc_map = {
            "outside right": "center left",
            "outside top": "lower center",
        }
        loc = loc_map.get(config.legend_position, config.legend_position)
        
        bbox = None
        if config.legend_position == "outside right":
            bbox = (1.02, 0.5)
        elif config.legend_position == "outside top":
            bbox = (0.5, 1.02)
        
        legend = ax.legend(
            loc=loc,
            fontsize=config.legend_font_size,
            title=config.legend_title,
            frameon=config.legend_frameon,
            ncol=config.legend_ncol,
            bbox_to_anchor=bbox,
        )
        
        if config.legend_title and legend:
            legend.get_title().set_fontsize(config.legend_title_font_size)
    
    def bar(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a vertical bar chart."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        # Use single color for all bars
        color = self._get_colors(1, config)[0]
        
        bars = ax.bar(
            data[x],
            data[y],
            width=config.bar_width,
            color=color,
            alpha=config.alpha,
            edgecolor=config.edge_color,
            linewidth=config.edge_width,
        )
        
        # Add value labels
        if config.show_values:
            for bar in bars:
                height = bar.get_height()
                if config.value_position == "outside":
                    va = "bottom"
                    y_pos = height
                elif config.value_position == "inside":
                    va = "top"
                    y_pos = height * 0.95
                else:  # center
                    va = "center"
                    y_pos = height / 2
                
                ax.annotate(
                    config.value_format.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    ha="center",
                    va=va,
                    fontsize=config.value_font_size,
                    color=config.label_color,
                )
        
        # Apply configuration
        self._apply_config(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    def barh(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a horizontal bar chart with top items at the top."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        # Reverse the data so top items appear at the top
        data = data.iloc[::-1].reset_index(drop=True)
        
        # Use single color for all bars
        color = self._get_colors(1, config)[0]
        
        bars = ax.barh(
            data[y],
            data[x],
            height=config.bar_width,
            color=color,
            alpha=config.alpha,
            edgecolor=config.edge_color,
            linewidth=config.edge_width,
        )
        
        # Add value labels
        if config.show_values:
            for bar in bars:
                width = bar.get_width()
                if config.value_position == "outside":
                    ha = "left"
                    x_pos = width * 1.01
                elif config.value_position == "inside":
                    ha = "right"
                    x_pos = width * 0.95
                else:  # center
                    ha = "center"
                    x_pos = width / 2
                
                ax.annotate(
                    config.value_format.format(width),
                    xy=(x_pos, bar.get_y() + bar.get_height() / 2),
                    ha=ha,
                    va="center",
                    fontsize=config.value_font_size,
                    color=config.label_color,
                )
        
        self._apply_config(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    def line(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a line chart."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        # Handle single or multiple y columns
        y_cols = [y] if isinstance(y, str) else y
        colors = self._get_colors(len(y_cols), config)
        
        linestyle_map = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": "-.",
        }
        linestyle = linestyle_map.get(config.line_style, "-")
        
        for i, col in enumerate(y_cols):
            line_kwargs = {
                "color": colors[i],
                "linewidth": config.line_width,
                "linestyle": linestyle,
                "alpha": config.alpha,
                "label": col,
            }
            
            if config.show_markers:
                line_kwargs["marker"] = config.marker_style
                line_kwargs["markersize"] = config.marker_size
            
            ax.plot(data[x], data[col], **line_kwargs)
            
            if config.fill_area:
                ax.fill_between(
                    data[x],
                    data[col],
                    alpha=config.fill_alpha,
                    color=colors[i],
                )
        
        self._apply_config(ax, config)
        
        if len(y_cols) > 1:
            self._add_legend(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    def scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        size: Optional[Union[str, float]] = None,
        color: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a scatter plot."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        # Determine sizes
        if size is None:
            sizes = config.scatter_size
        elif isinstance(size, str):
            # Map column to size range
            s_min, s_max = config.scatter_size_range
            s_data = data[size]
            sizes = s_min + (s_data - s_data.min()) / (s_data.max() - s_data.min()) * (s_max - s_min)
        else:
            sizes = size
        
        # Determine colors
        if color is None:
            c = self._get_colors(1, config)[0]
            scatter = ax.scatter(
                data[x],
                data[y],
                s=sizes,
                c=c,
                alpha=config.alpha,
                edgecolor=config.edge_color,
                linewidth=config.edge_width,
            )
        else:
            scatter = ax.scatter(
                data[x],
                data[y],
                s=sizes,
                c=data[color],
                cmap=config.colormap,
                alpha=config.alpha,
                edgecolor=config.edge_color,
                linewidth=config.edge_width,
            )
            
            if config.colorbar:
                cbar = self.plt.colorbar(scatter, ax=ax)
                if config.colorbar_label:
                    cbar.set_label(config.colorbar_label)
        
        self._apply_config(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    def heatmap(
        self,
        data: pd.DataFrame,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a heatmap."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        # Plot heatmap
        im = ax.imshow(
            data.values,
            cmap=config.colormap,
            aspect="auto",
        )
        
        # Set ticks
        ax.set_xticks(range(len(data.columns)))
        ax.set_yticks(range(len(data.index)))
        ax.set_xticklabels(data.columns, rotation=config.xtick_rotation)
        ax.set_yticklabels(data.index)
        
        # Add annotations
        if config.annotate:
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    value = data.iloc[i, j]
                    text = config.annotation_format.format(value)
                    ax.text(
                        j, i, text,
                        ha="center",
                        va="center",
                        fontsize=config.annotation_font_size,
                        color="white" if value > data.values.mean() else "black",
                    )
        
        # Colorbar
        if config.colorbar:
            cbar = self.plt.colorbar(im, ax=ax)
            if config.colorbar_label:
                cbar.set_label(config.colorbar_label)
        
        self._apply_config(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    def network(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a network graph."""
        import networkx as nx
        
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for _, row in nodes.iterrows():
            G.add_node(row["id"], **row.to_dict())
        
        # Add edges
        for _, row in edges.iterrows():
            weight = row.get("weight", 1)
            G.add_edge(row["source"], row["target"], weight=weight)
        
        # Layout
        layout_funcs = {
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "random": nx.random_layout,
        }
        pos = layout_funcs.get(config.layout, nx.spring_layout)(G)
        
        # Node sizes
        if config.node_size_by and config.node_size_by in nodes.columns:
            node_sizes = [
                nodes[nodes["id"] == n][config.node_size_by].values[0] * 10
                for n in G.nodes()
            ]
        else:
            node_sizes = config.node_size
        
        # Node colors
        if config.node_color_by and config.node_color_by in nodes.columns:
            node_colors = [
                nodes[nodes["id"] == n][config.node_color_by].values[0]
                for n in G.nodes()
            ]
        else:
            node_colors = self._get_colors(len(G.nodes()), config)
        
        # Edge widths
        if config.edge_width_by:
            edge_weights = [G[u][v].get(config.edge_width_by, 1) for u, v in G.edges()]
            max_weight = max(edge_weights) if edge_weights else 1
            edge_widths = [3 * w / max_weight for w in edge_weights]
        else:
            edge_widths = 1
        
        # Draw network
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_widths,
            alpha=config.edge_alpha,
        )
        
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=config.alpha,
        )
        
        if config.show_labels:
            nx.draw_networkx_labels(
                G, pos, ax=ax,
                font_size=config.label_font_size,
            )
        
        ax.axis("off")
        
        if config.title:
            ax.set_title(
                config.title,
                fontsize=config.title_font_size,
                fontweight=config.title_font_weight,
            )
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # PHASE 2: DISTRIBUTION PLOTS
    # =========================================================================
    
    def histogram(
        self,
        data: pd.DataFrame,
        x: str,
        bins: int = 30,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a histogram."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        color = self._get_colors(1, config)[0]
        
        ax.hist(
            data[x].dropna(),
            bins=bins,
            color=color,
            alpha=config.alpha,
            edgecolor=config.edge_color if config.edge_color != "none" else "white",
            linewidth=config.edge_width if config.edge_width > 0 else 0.5,
        )
        
        self._apply_config(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    def boxplot(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: str = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a box plot."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        colors = self._get_colors(10, config)
        
        if x is not None:
            # Grouped boxplot
            groups = data[x].unique()
            box_data = [data[data[x] == g][y].dropna() for g in groups]
            
            bp = ax.boxplot(
                box_data,
                patch_artist=True,
                labels=groups,
            )
            
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(config.alpha)
        else:
            # Single boxplot
            bp = ax.boxplot(
                [data[y].dropna()],
                patch_artist=True,
            )
            bp["boxes"][0].set_facecolor(colors[0])
            bp["boxes"][0].set_alpha(config.alpha)
        
        self._apply_config(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    def violinplot(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: str = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a violin plot."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        colors = self._get_colors(10, config)
        
        if x is not None:
            groups = data[x].unique()
            violin_data = [data[data[x] == g][y].dropna().values for g in groups]
            positions = range(1, len(groups) + 1)
            
            parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
            
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(config.alpha)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(groups)
        else:
            parts = ax.violinplot([data[y].dropna().values], showmeans=True, showmedians=True)
            parts["bodies"][0].set_facecolor(colors[0])
            parts["bodies"][0].set_alpha(config.alpha)
        
        self._apply_config(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # PHASE 2: AREA PLOTS
    # =========================================================================
    
    def area(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        stacked: bool = False,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create an area chart."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        y_cols = [y] if isinstance(y, str) else y
        colors = self._get_colors(len(y_cols), config)
        
        if stacked:
            ax.stackplot(
                data[x],
                [data[col] for col in y_cols],
                labels=y_cols,
                colors=colors,
                alpha=config.alpha,
            )
        else:
            for i, col in enumerate(y_cols):
                ax.fill_between(
                    data[x],
                    data[col],
                    alpha=config.fill_alpha,
                    color=colors[i],
                    label=col,
                )
                ax.plot(data[x], data[col], color=colors[i], linewidth=config.line_width)
        
        self._apply_config(ax, config)
        
        if len(y_cols) > 1:
            self._add_legend(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # PHASE 2: GROUPED/STACKED BAR PLOTS
    # =========================================================================
    
    def grouped_bar(
        self,
        data: pd.DataFrame,
        x: str,
        y: List[str],
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a grouped bar chart."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        n_groups = len(data)
        n_bars = len(y)
        colors = self._get_colors(n_bars, config)
        
        bar_width = config.bar_width / n_bars
        positions = np.arange(n_groups)
        
        for i, col in enumerate(y):
            offset = (i - n_bars / 2 + 0.5) * bar_width
            ax.bar(
                positions + offset,
                data[col],
                bar_width,
                label=col,
                color=colors[i],
                alpha=config.alpha,
            )
        
        ax.set_xticks(positions)
        ax.set_xticklabels(data[x])
        
        self._apply_config(ax, config)
        self._add_legend(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    def stacked_bar(
        self,
        data: pd.DataFrame,
        x: str,
        y: List[str],
        horizontal: bool = False,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a stacked bar chart."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        colors = self._get_colors(len(y), config)
        
        if horizontal:
            bottom = np.zeros(len(data))
            for i, col in enumerate(y):
                ax.barh(
                    data[x],
                    data[col],
                    left=bottom,
                    label=col,
                    color=colors[i],
                    alpha=config.alpha,
                    height=config.bar_width,
                )
                bottom += data[col].values
        else:
            bottom = np.zeros(len(data))
            for i, col in enumerate(y):
                ax.bar(
                    data[x],
                    data[col],
                    bottom=bottom,
                    label=col,
                    color=colors[i],
                    alpha=config.alpha,
                    width=config.bar_width,
                )
                bottom += data[col].values
        
        self._apply_config(ax, config)
        self._add_legend(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # PHASE 2: COMPARISON PLOTS
    # =========================================================================
    
    def dumbbell(
        self,
        data: pd.DataFrame,
        y: str,
        x_start: str,
        x_end: str,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a dumbbell chart."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        colors = self._get_colors(2, config)
        
        y_positions = range(len(data))
        
        # Draw lines
        for i, (_, row) in enumerate(data.iterrows()):
            ax.plot(
                [row[x_start], row[x_end]],
                [i, i],
                color="gray",
                linewidth=config.line_width,
                alpha=0.5,
            )
        
        # Draw start points
        ax.scatter(
            data[x_start],
            y_positions,
            color=colors[0],
            s=config.marker_size * 20,
            zorder=5,
            label=x_start,
        )
        
        # Draw end points
        ax.scatter(
            data[x_end],
            y_positions,
            color=colors[1],
            s=config.marker_size * 20,
            zorder=5,
            label=x_end,
        )
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(data[y])
        
        self._apply_config(ax, config)
        self._add_legend(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    def lollipop(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        horizontal: bool = True,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a lollipop chart."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        color = self._get_colors(1, config)[0]
        
        if horizontal:
            positions = range(len(data))
            
            # Draw stems
            ax.hlines(
                y=positions,
                xmin=0,
                xmax=data[x],
                color=color,
                linewidth=config.line_width,
                alpha=0.7,
            )
            
            # Draw dots
            ax.scatter(
                data[x],
                positions,
                color=color,
                s=config.marker_size * 15,
                zorder=5,
            )
            
            ax.set_yticks(positions)
            ax.set_yticklabels(data[y])
        else:
            # Vertical lollipop
            ax.vlines(
                x=data[y],
                ymin=0,
                ymax=data[x],
                color=color,
                linewidth=config.line_width,
                alpha=0.7,
            )
            
            ax.scatter(
                data[y],
                data[x],
                color=color,
                s=config.marker_size * 15,
                zorder=5,
            )
        
        self._apply_config(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # PHASE 2: BUBBLE CHART
    # =========================================================================
    
    def bubble(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        size: str,
        color: Optional[str] = None,
        label: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a bubble chart."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        # Scale sizes
        s_min, s_max = config.scatter_size_range
        s_data = data[size]
        sizes = s_min + (s_data - s_data.min()) / (s_data.max() - s_data.min() + 1e-10) * (s_max - s_min)
        
        # Colors
        if color:
            scatter = ax.scatter(
                data[x],
                data[y],
                s=sizes,
                c=data[color],
                cmap=config.colormap,
                alpha=config.alpha,
                edgecolor="white",
                linewidth=0.5,
            )
            if config.colorbar:
                cbar = self.plt.colorbar(scatter, ax=ax)
                if config.colorbar_label:
                    cbar.set_label(config.colorbar_label)
        else:
            colors = self._get_colors(1, config)[0]
            ax.scatter(
                data[x],
                data[y],
                s=sizes,
                c=colors,
                alpha=config.alpha,
                edgecolor="white",
                linewidth=0.5,
            )
        
        # Labels
        if label and config.show_labels:
            for _, row in data.iterrows():
                ax.annotate(
                    str(row[label])[:15],
                    (row[x], row[y]),
                    fontsize=config.label_font_size,
                    ha="center",
                    va="bottom",
                )
        
        self._apply_config(ax, config)
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # PHASE 2: DONUT CHART
    # =========================================================================
    
    def donut(
        self,
        data: pd.DataFrame,
        values: str,
        labels: str,
        hole_size: float = 0.4,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a donut chart."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        colors = self._get_colors(len(data), config)
        
        wedges, texts, autotexts = ax.pie(
            data[values],
            labels=data[labels] if config.show_labels else None,
            colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
            pctdistance=0.75,
            startangle=90,
            wedgeprops=dict(width=1 - hole_size, edgecolor="white"),
        )
        
        # Center circle for donut effect
        centre_circle = self.plt.Circle((0, 0), hole_size, fc=config.background_color)
        ax.add_patch(centre_circle)
        
        ax.axis("equal")
        
        if config.title:
            ax.set_title(
                config.title,
                fontsize=config.title_font_size,
                fontweight=config.title_font_weight,
            )
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # PHASE 2: TREEMAP
    # =========================================================================
    
    def treemap(
        self,
        data: pd.DataFrame,
        values: str,
        labels: str,
        parents: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a treemap."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        try:
            import squarify
        except ImportError:
            raise ImportError("squarify required for treemaps: pip install squarify")
        
        colors = self._get_colors(len(data), config)
        
        # Normalize values
        norm_values = data[values] / data[values].sum() * 100
        
        squarify.plot(
            sizes=norm_values,
            label=data[labels] if config.show_labels else None,
            color=colors,
            alpha=config.alpha,
            ax=ax,
            text_kwargs={"fontsize": config.label_font_size},
        )
        
        ax.axis("off")
        
        if config.title:
            ax.set_title(
                config.title,
                fontsize=config.title_font_size,
                fontweight=config.title_font_weight,
            )
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # PHASE 2: WORDCLOUD
    # =========================================================================
    
    def wordcloud(
        self,
        data: pd.DataFrame,
        text: str,
        weight: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a word cloud."""
        config = self._merge_config(config)
        fig, ax = self._create_figure(config)
        
        try:
            from wordcloud import WordCloud
        except ImportError:
            raise ImportError("wordcloud required: pip install wordcloud")
        
        # Build frequency dict
        if weight:
            freq = dict(zip(data[text], data[weight]))
        else:
            freq = dict(zip(data[text], [1] * len(data)))
        
        wc = WordCloud(
            width=config.width,
            height=config.height,
            background_color=config.background_color,
            colormap=config.colormap,
            max_words=100,
        ).generate_from_frequencies(freq)
        
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        
        if config.title:
            ax.set_title(
                config.title,
                fontsize=config.title_font_size,
                fontweight=config.title_font_weight,
            )
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # PHASE 2: RADAR CHART
    # =========================================================================
    
    def radar(
        self,
        data: pd.DataFrame,
        categories: List[str],
        values: Union[str, List[str]],
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a radar/spider chart."""
        config = self._merge_config(config)
        
        # Number of variables
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = self.plt.subplots(
            figsize=(config.width / config.dpi, config.height / config.dpi),
            subplot_kw=dict(polar=True),
        )
        
        value_cols = [values] if isinstance(values, str) else values
        colors = self._get_colors(len(value_cols), config)
        
        for i, col in enumerate(value_cols):
            if col in data.columns:
                vals = data[col].values.tolist()
            else:
                # Assume col is a row index
                vals = data.loc[col, categories].values.tolist()
            
            vals += vals[:1]  # Complete the circle
            
            ax.plot(angles, vals, color=colors[i], linewidth=config.line_width, label=col)
            ax.fill(angles, vals, color=colors[i], alpha=config.fill_alpha)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=config.xtick_font_size)
        
        if config.title:
            ax.set_title(
                config.title,
                fontsize=config.title_font_size,
                fontweight=config.title_font_weight,
            )
        
        if len(value_cols) > 1 and config.show_legend:
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        
        if config.tight_layout:
            self.plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # SAVE AND SHOW
    # =========================================================================
    
    def save(
        self,
        figure: Any,
        filename: str,
        config: Optional[PlotConfig] = None,
    ) -> List[str]:
        """Save figure to file(s)."""
        config = config or self.config
        
        saved_files = []
        for fmt in config.save_format:
            path = f"{filename}.{fmt}"
            figure.savefig(
                path,
                dpi=config.export_dpi,
                bbox_inches="tight" if config.tight_layout else None,
                transparent=config.transparent,
                facecolor=figure.get_facecolor(),
            )
            saved_files.append(path)
        
        print(f"Plot saved to {filename}.{config.save_format[0]} (and {', '.join(config.save_format[1:])})")
        
        return saved_files
    
    def show(self, figure: Any) -> None:
        """Display figure."""
        self.plt.show()
