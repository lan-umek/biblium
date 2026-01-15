# -*- coding: utf-8 -*-
"""
Bokeh Backend Implementation.

Provides the Bokeh implementation of the PlotBackend interface.
Creates interactive, web-based visualizations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from biblium.plotting.backends.base import PlotBackend
from biblium.plotting.config import PlotConfig


class BokehBackend(PlotBackend):
    """
    Bokeh implementation of PlotBackend.
    
    This backend uses Bokeh for interactive plotting.
    Requires: pip install bokeh
    """
    
    name = "bokeh"
    
    def __init__(self, config: Optional[PlotConfig] = None):
        super().__init__(config)
        
        try:
            from bokeh.plotting import figure, output_file, save, show
            from bokeh.models import (
                ColumnDataSource, HoverTool, ColorBar, LinearColorMapper,
                LabelSet, Title, Legend, LegendItem,
            )
            from bokeh.palettes import Viridis256, Turbo256, YlOrRd, RdYlBu, Category10
            from bokeh.transform import linear_cmap, factor_cmap
            from bokeh.io import export_png, export_svgs
            from bokeh.layouts import column
            
            self.bokeh_plotting = __import__("bokeh.plotting", fromlist=["figure"])
            self.bokeh_models = __import__("bokeh.models", fromlist=["ColumnDataSource"])
            self.bokeh_palettes = __import__("bokeh.palettes", fromlist=["Viridis256"])
            self.bokeh_transform = __import__("bokeh.transform", fromlist=["linear_cmap"])
            self.bokeh_io = __import__("bokeh.io", fromlist=["export_png"])
            
            self._available = True
            
        except ImportError:
            self._available = False
            print("Bokeh not installed. Install with: pip install bokeh")
    
    def _check_available(self):
        """Check if Bokeh is available."""
        if not self._available:
            raise ImportError("Bokeh is required for this backend. Install with: pip install bokeh")
    
    def _get_tools(self, config: PlotConfig) -> str:
        """Get tool string based on config."""
        tools = []
        if config.enable_pan:
            tools.append("pan")
        if config.enable_zoom:
            tools.append("wheel_zoom")
            tools.append("box_zoom")
        if config.enable_save:
            tools.append("save")
        if config.enable_reset:
            tools.append("reset")
        if config.show_tooltips:
            tools.append("hover")
        
        return ",".join(tools) if tools else ""
    
    def _get_palette(self, config: PlotConfig, n: int = 256) -> List[str]:
        """Get Bokeh palette from config colormap."""
        from bokeh.palettes import all_palettes, Viridis256, Turbo256
        
        cmap = config.colormap
        
        # Map common matplotlib names to Bokeh palettes
        cmap_map = {
            "viridis": Viridis256,
            "turbo": Turbo256,
        }
        
        if cmap in cmap_map:
            palette = cmap_map[cmap]
        elif cmap in all_palettes:
            # Get largest available
            sizes = list(all_palettes[cmap].keys())
            size = max(s for s in sizes if s <= n) if sizes else max(sizes)
            palette = all_palettes[cmap][size]
        else:
            palette = Viridis256
        
        return list(palette)
    
    def _create_figure(self, config: PlotConfig, **kwargs) -> Any:
        """Create Bokeh figure with config settings."""
        self._check_available()
        
        from bokeh.plotting import figure
        
        fig_kwargs = {
            "width": config.width,
            "height": config.height,
            "background_fill_color": config.plot_background_color,
            "border_fill_color": config.background_color,
            "tools": self._get_tools(config),
            "toolbar_location": "above" if config.interactive else None,
        }
        
        if config.title:
            fig_kwargs["title"] = config.title
        
        if config.xscale == "log":
            fig_kwargs["x_axis_type"] = "log"
        if config.yscale == "log":
            fig_kwargs["y_axis_type"] = "log"
        
        fig_kwargs.update(kwargs)
        
        p = figure(**fig_kwargs)
        
        return p
    
    def _apply_config(self, p: Any, config: PlotConfig) -> None:
        """Apply config settings to figure."""
        # Title styling
        if p.title:
            p.title.text_font_size = f"{config.title_font_size}pt"
            p.title.text_color = config.title_color
            p.title.text_font_style = "bold" if config.title_font_weight == "bold" else "normal"
            p.title.align = config.title_position
        
        # Axis labels
        if config.xlabel:
            p.xaxis.axis_label = config.xlabel
            p.xaxis.axis_label_text_font_size = f"{config.xlabel_font_size}pt"
            p.xaxis.axis_label_text_color = config.label_color
        
        if config.ylabel:
            p.yaxis.axis_label = config.ylabel
            p.yaxis.axis_label_text_font_size = f"{config.ylabel_font_size}pt"
            p.yaxis.axis_label_text_color = config.label_color
        
        # Tick styling
        p.xaxis.major_label_text_font_size = f"{config.xtick_font_size}pt"
        p.yaxis.major_label_text_font_size = f"{config.ytick_font_size}pt"
        p.xaxis.major_label_text_color = config.tick_color
        p.yaxis.major_label_text_color = config.tick_color
        
        if config.xtick_rotation:
            p.xaxis.major_label_orientation = np.radians(config.xtick_rotation)
        
        # Grid
        if config.show_grid:
            p.xgrid.grid_line_color = config.grid_color
            p.xgrid.grid_line_alpha = config.grid_alpha
            p.ygrid.grid_line_color = config.grid_color
            p.ygrid.grid_line_alpha = config.grid_alpha
            
            if config.grid_style == "dashed":
                p.xgrid.grid_line_dash = "dashed"
                p.ygrid.grid_line_dash = "dashed"
            elif config.grid_style == "dotted":
                p.xgrid.grid_line_dash = "dotted"
                p.ygrid.grid_line_dash = "dotted"
        else:
            p.xgrid.grid_line_color = None
            p.ygrid.grid_line_color = None
        
        # Axis limits
        if config.xlim:
            p.x_range.start, p.x_range.end = config.xlim
        if config.ylim:
            p.y_range.start, p.y_range.end = config.ylim
        
        # Outline
        p.outline_line_color = config.spine_color if config.show_bottom_spine else None
    
    def bar(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a vertical bar chart."""
        self._check_available()
        
        from bokeh.models import ColumnDataSource, HoverTool, LabelSet
        
        config = self._merge_config(config)
        
        # Create figure with categorical x-axis
        p = self._create_figure(
            config,
            x_range=list(data[x].astype(str)),
        )
        
        # Prepare data source with single color
        color = self._get_colors(1, config)[0]
        source_data = {
            x: list(data[x].astype(str)),
            y: list(data[y]),
            "color": [color] * len(data),
        }
        
        source = ColumnDataSource(data=source_data)
        
        # Create bars
        p.vbar(
            x=x,
            top=y,
            width=config.bar_width,
            source=source,
            color="color",
            alpha=config.alpha,
            line_color=config.edge_color if config.edge_color != "none" else None,
            line_width=config.edge_width,
        )
        
        # Add value labels
        if config.show_values:
            labels = LabelSet(
                x=x,
                y=y,
                text=y,
                level="glyph",
                source=source,
                text_font_size=f"{config.value_font_size}pt",
                text_align="center",
                y_offset=5,
            )
            p.add_layout(labels)
        
        # Add hover tool
        if config.show_tooltips:
            hover = p.select(type=HoverTool)
            if hover:
                hover[0].tooltips = [(x, f"@{x}"), (y, f"@{y}{{0,0}}")]
        
        self._apply_config(p, config)
        
        return p
    
    def barh(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a horizontal bar chart with top items at the top."""
        self._check_available()
        
        from bokeh.models import ColumnDataSource, HoverTool
        
        config = self._merge_config(config)
        
        # Reverse data so top items appear at the top
        data = data.iloc[::-1].reset_index(drop=True)
        
        # Create figure with categorical y-axis (reversed order)
        p = self._create_figure(
            config,
            y_range=list(data[y].astype(str)),
        )
        
        # Prepare data source with single color
        color = self._get_colors(1, config)[0]
        source_data = {
            x: list(data[x]),
            y: list(data[y].astype(str)),
            "color": [color] * len(data),
        }
        
        source = ColumnDataSource(data=source_data)
        
        # Create horizontal bars
        p.hbar(
            y=y,
            right=x,
            height=config.bar_width,
            source=source,
            color="color",
            alpha=config.alpha,
            line_color=config.edge_color if config.edge_color != "none" else None,
            line_width=config.edge_width,
        )
        
        # Add hover
        if config.show_tooltips:
            hover = p.select(type=HoverTool)
            if hover:
                hover[0].tooltips = [(y, f"@{y}"), (x, f"@{x}{{0,0}}")]
        
        self._apply_config(p, config)
        
        return p
    
    def line(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a line chart."""
        self._check_available()
        
        from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
        
        config = self._merge_config(config)
        p = self._create_figure(config)
        
        y_cols = [y] if isinstance(y, str) else y
        colors = self._get_colors(len(y_cols), config)
        
        # Line dash mapping
        dash_map = {
            "solid": "solid",
            "dashed": "dashed",
            "dotted": "dotted",
            "dashdot": "dashdot",
        }
        line_dash = dash_map.get(config.line_style, "solid")
        
        legend_items = []
        
        for i, col in enumerate(y_cols):
            source = ColumnDataSource(data={x: data[x], col: data[col]})
            
            line = p.line(
                x=x,
                y=col,
                source=source,
                color=colors[i],
                line_width=config.line_width,
                line_dash=line_dash,
                alpha=config.alpha,
            )
            
            if config.show_markers:
                p.circle(
                    x=x,
                    y=col,
                    source=source,
                    color=colors[i],
                    size=config.marker_size,
                    alpha=config.alpha,
                )
            
            if config.fill_area:
                p.varea(
                    x=x,
                    y1=0,
                    y2=col,
                    source=source,
                    fill_color=colors[i],
                    fill_alpha=config.fill_alpha,
                )
            
            legend_items.append(LegendItem(label=col, renderers=[line]))
        
        # Add legend for multiple lines
        if len(y_cols) > 1 and config.show_legend:
            legend = Legend(items=legend_items)
            p.add_layout(legend, "right")
        
        # Hover
        if config.show_tooltips:
            hover = p.select(type=HoverTool)
            if hover:
                tooltips = [(x, f"@{x}")]
                for col in y_cols:
                    tooltips.append((col, f"@{col}{{0,0.00}}"))
                hover[0].tooltips = tooltips
        
        self._apply_config(p, config)
        
        return p
    
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
        self._check_available()
        
        from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
        from bokeh.transform import linear_cmap
        
        config = self._merge_config(config)
        p = self._create_figure(config)
        
        source_data = {x: data[x], y: data[y]}
        
        # Size mapping
        if size is None:
            source_data["size"] = [config.scatter_size] * len(data)
        elif isinstance(size, str):
            s_min, s_max = config.scatter_size_range
            s_data = data[size]
            normalized = (s_data - s_data.min()) / (s_data.max() - s_data.min())
            source_data["size"] = s_min + normalized * (s_max - s_min)
            source_data[size] = data[size]
        else:
            source_data["size"] = [size] * len(data)
        
        # Color mapping
        if color is None:
            source_data["color"] = [self._get_colors(1, config)[0]] * len(data)
            color_arg = "color"
        else:
            source_data[color] = data[color]
            palette = self._get_palette(config)
            mapper = linear_cmap(
                field_name=color,
                palette=palette,
                low=data[color].min(),
                high=data[color].max(),
            )
            color_arg = mapper
        
        source = ColumnDataSource(data=source_data)
        
        scatter = p.circle(
            x=x,
            y=y,
            size="size",
            source=source,
            color=color_arg,
            alpha=config.alpha,
            line_color=config.edge_color if config.edge_color != "none" else None,
            line_width=config.edge_width,
        )
        
        # Add colorbar
        if color and config.colorbar:
            from bokeh.models import ColorBar
            color_bar = ColorBar(
                color_mapper=mapper["transform"],
                title=config.colorbar_label or color,
            )
            p.add_layout(color_bar, "right")
        
        # Hover
        if config.show_tooltips:
            tooltips = [(x, f"@{x}{{0,0.00}}"), (y, f"@{y}{{0,0.00}}")]
            if isinstance(size, str):
                tooltips.append((size, f"@{size}{{0,0.00}}"))
            if color:
                tooltips.append((color, f"@{color}{{0,0.00}}"))
            
            hover = HoverTool(tooltips=tooltips)
            p.add_tools(hover)
        
        self._apply_config(p, config)
        
        return p
    
    def heatmap(
        self,
        data: pd.DataFrame,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a heatmap."""
        self._check_available()
        
        from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper, LabelSet
        
        config = self._merge_config(config)
        
        # Prepare data
        x_labels = list(data.columns.astype(str))
        y_labels = list(data.index.astype(str))
        
        # Create mesh
        x_coords = []
        y_coords = []
        values = []
        
        for i, row_label in enumerate(y_labels):
            for j, col_label in enumerate(x_labels):
                x_coords.append(col_label)
                y_coords.append(row_label)
                values.append(data.iloc[i, j])
        
        source = ColumnDataSource(data={
            "x": x_coords,
            "y": y_coords,
            "value": values,
        })
        
        # Create figure
        p = self._create_figure(
            config,
            x_range=x_labels,
            y_range=y_labels,
        )
        
        # Color mapping
        palette = self._get_palette(config)
        mapper = LinearColorMapper(
            palette=palette,
            low=min(values),
            high=max(values),
        )
        
        # Draw rectangles
        p.rect(
            x="x",
            y="y",
            width=1,
            height=1,
            source=source,
            fill_color={"field": "value", "transform": mapper},
            line_color=None,
        )
        
        # Annotations
        if config.annotate:
            labels = LabelSet(
                x="x",
                y="y",
                text="value",
                source=source,
                text_font_size=f"{config.annotation_font_size}pt",
                text_align="center",
                text_baseline="middle",
            )
            p.add_layout(labels)
        
        # Colorbar
        if config.colorbar:
            color_bar = ColorBar(
                color_mapper=mapper,
                title=config.colorbar_label,
            )
            p.add_layout(color_bar, "right")
        
        self._apply_config(p, config)
        
        return p
    
    def network(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a network graph."""
        self._check_available()
        
        from bokeh.models import ColumnDataSource, HoverTool, LabelSet
        from bokeh.models import Circle, MultiLine
        import networkx as nx
        
        config = self._merge_config(config)
        
        # Create NetworkX graph for layout
        G = nx.Graph()
        for _, row in nodes.iterrows():
            G.add_node(row["id"])
        for _, row in edges.iterrows():
            G.add_edge(row["source"], row["target"], weight=row.get("weight", 1))
        
        # Layout
        layout_funcs = {
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "random": nx.random_layout,
        }
        pos = layout_funcs.get(config.layout, nx.spring_layout)(G)
        
        # Create figure
        p = self._create_figure(config)
        p.axis.visible = False
        p.grid.visible = False
        
        # Prepare node data
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_ids = list(G.nodes())
        
        # Node sizes
        if config.node_size_by and config.node_size_by in nodes.columns:
            node_sizes = []
            for n in G.nodes():
                val = nodes[nodes["id"] == n][config.node_size_by].values
                node_sizes.append(val[0] * 0.5 if len(val) > 0 else config.node_size)
        else:
            node_sizes = [config.node_size / 10] * len(G.nodes())
        
        colors = self._get_colors(len(G.nodes()), config)
        
        node_source = ColumnDataSource(data={
            "x": node_x,
            "y": node_y,
            "id": node_ids,
            "size": node_sizes,
            "color": colors,
        })
        
        # Draw edges
        edge_x = []
        edge_y = []
        for u, v in G.edges():
            edge_x.append([pos[u][0], pos[v][0]])
            edge_y.append([pos[u][1], pos[v][1]])
        
        edge_source = ColumnDataSource(data={"xs": edge_x, "ys": edge_y})
        
        p.multi_line(
            "xs", "ys",
            source=edge_source,
            line_color="gray",
            line_alpha=config.edge_alpha,
            line_width=1,
        )
        
        # Draw nodes
        p.circle(
            "x", "y",
            source=node_source,
            size="size",
            color="color",
            alpha=config.alpha,
        )
        
        # Labels
        if config.show_labels:
            labels = LabelSet(
                x="x",
                y="y",
                text="id",
                source=node_source,
                text_font_size=f"{config.label_font_size}pt",
                text_align="center",
                y_offset=8,
            )
            p.add_layout(labels)
        
        # Hover
        if config.show_tooltips:
            hover = HoverTool(tooltips=[("ID", "@id")])
            p.add_tools(hover)
        
        self._apply_config(p, config)
        
        return p
    
    def save(
        self,
        figure: Any,
        filename: str,
        config: Optional[PlotConfig] = None,
    ) -> List[str]:
        """Save figure to file(s)."""
        self._check_available()
        
        from bokeh.io import export_png, export_svgs, output_file, save
        
        config = config or self.config
        saved_files = []
        
        for fmt in config.save_format:
            path = f"{filename}.{fmt}"
            
            if fmt == "html":
                output_file(path)
                save(figure)
                saved_files.append(path)
            elif fmt == "png":
                try:
                    export_png(figure, filename=path)
                    saved_files.append(path)
                except Exception as e:
                    print(f"PNG export requires selenium and geckodriver: {e}")
            elif fmt == "svg":
                try:
                    figure.output_backend = "svg"
                    export_svgs(figure, filename=path)
                    saved_files.append(path)
                except Exception as e:
                    print(f"SVG export requires selenium and geckodriver: {e}")
        
        if saved_files:
            print(f"Plot saved to {saved_files[0]}" + 
                  (f" (and {', '.join(saved_files[1:])})" if len(saved_files) > 1 else ""))
        
        return saved_files
    
    def show(self, figure: Any) -> None:
        """Display figure."""
        self._check_available()
        
        from bokeh.io import show
        show(figure)
    
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
        self._check_available()
        
        from bokeh.models import ColumnDataSource
        
        config = self._merge_config(config)
        
        # Compute histogram
        hist, edges = np.histogram(data[x].dropna(), bins=bins)
        
        source = ColumnDataSource(data={
            "top": hist,
            "left": edges[:-1],
            "right": edges[1:],
        })
        
        p = self._create_figure(config)
        
        color = self._get_colors(1, config)[0]
        
        p.quad(
            top="top",
            bottom=0,
            left="left",
            right="right",
            source=source,
            fill_color=color,
            fill_alpha=config.alpha,
            line_color="white",
        )
        
        self._apply_config(p, config)
        
        return p
    
    def boxplot(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: str = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a box plot (simplified version for Bokeh)."""
        self._check_available()
        
        config = self._merge_config(config)
        colors = self._get_colors(10, config)
        
        if x is not None:
            groups = list(data[x].unique())
            p = self._create_figure(config, x_range=groups)
            
            for i, group in enumerate(groups):
                group_data = data[data[x] == group][y].dropna()
                
                q1 = group_data.quantile(0.25)
                q2 = group_data.quantile(0.5)
                q3 = group_data.quantile(0.75)
                iqr = q3 - q1
                upper = min(q3 + 1.5 * iqr, group_data.max())
                lower = max(q1 - 1.5 * iqr, group_data.min())
                
                # Box
                p.vbar(x=group, width=0.7, bottom=q1, top=q3, 
                       fill_color=colors[i % len(colors)], fill_alpha=config.alpha, line_color="black")
                
                # Median line
                p.segment(x0=i - 0.35 + 0.5, y0=q2, x1=i + 0.35 + 0.5, y1=q2, 
                         line_color="black", line_width=2)
                
                # Whiskers
                p.segment(x0=group, y0=upper, x1=group, y1=q3, line_color="black")
                p.segment(x0=group, y0=lower, x1=group, y1=q1, line_color="black")
        else:
            p = self._create_figure(config)
            group_data = data[y].dropna()
            
            q1 = group_data.quantile(0.25)
            q2 = group_data.quantile(0.5)
            q3 = group_data.quantile(0.75)
            
            p.vbar(x=0, width=0.7, bottom=q1, top=q3,
                   fill_color=colors[0], fill_alpha=config.alpha, line_color="black")
        
        self._apply_config(p, config)
        
        return p
    
    def violinplot(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: str = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Create a violin plot (approximated with mirrored histograms)."""
        # Violin plots are complex in Bokeh; fallback to boxplot
        return self.boxplot(data, x, y, config, **kwargs)
    
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
        self._check_available()
        
        from bokeh.models import ColumnDataSource
        
        config = self._merge_config(config)
        p = self._create_figure(config)
        
        y_cols = [y] if isinstance(y, str) else y
        colors = self._get_colors(len(y_cols), config)
        
        if stacked:
            # Use varea_stack for stacked areas
            source = ColumnDataSource(data)
            p.varea_stack(
                y_cols,
                x=x,
                source=source,
                color=colors[:len(y_cols)],
                alpha=config.alpha,
            )
        else:
            for i, col in enumerate(y_cols):
                source = ColumnDataSource({x: data[x], col: data[col]})
                p.varea(
                    x=x,
                    y1=0,
                    y2=col,
                    source=source,
                    fill_color=colors[i],
                    fill_alpha=config.fill_alpha,
                )
                p.line(x=x, y=col, source=source, color=colors[i], line_width=config.line_width)
        
        self._apply_config(p, config)
        
        return p
    
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
        self._check_available()
        
        from bokeh.models import ColumnDataSource, FactorRange
        
        config = self._merge_config(config)
        colors = self._get_colors(len(y), config)
        
        # Create grouped x-axis
        x_vals = [(str(cat), col) for cat in data[x] for col in y]
        
        # Flatten values
        values = []
        for _, row in data.iterrows():
            for col in y:
                values.append(row[col])
        
        source = ColumnDataSource(data={
            "x": x_vals,
            "value": values,
            "color": [colors[y.index(x_val[1])] for x_val in x_vals],
        })
        
        p = self._create_figure(config, x_range=FactorRange(*x_vals))
        
        p.vbar(
            x="x",
            top="value",
            width=0.8,
            source=source,
            color="color",
            alpha=config.alpha,
        )
        
        p.xaxis.major_label_orientation = 0.8
        
        self._apply_config(p, config)
        
        return p
    
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
        self._check_available()
        
        from bokeh.models import ColumnDataSource
        
        config = self._merge_config(config)
        colors = self._get_colors(len(y), config)
        
        source = ColumnDataSource(data)
        
        if horizontal:
            categories = list(data[x].astype(str))
            p = self._create_figure(config, y_range=categories)
            p.hbar_stack(
                y,
                y=x,
                height=0.8,
                source=source,
                color=colors[:len(y)],
                alpha=config.alpha,
            )
        else:
            categories = list(data[x].astype(str))
            p = self._create_figure(config, x_range=categories)
            p.vbar_stack(
                y,
                x=x,
                width=0.8,
                source=source,
                color=colors[:len(y)],
                alpha=config.alpha,
            )
        
        self._apply_config(p, config)
        
        return p
    
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
        self._check_available()
        
        from bokeh.models import ColumnDataSource
        
        config = self._merge_config(config)
        colors = self._get_colors(2, config)
        
        categories = list(data[y].astype(str))
        p = self._create_figure(config, y_range=categories)
        
        source = ColumnDataSource(data)
        
        # Lines
        p.segment(
            x0=x_start, y0=y, x1=x_end, y1=y,
            source=source,
            line_color="gray",
            line_width=config.line_width,
        )
        
        # Start points
        p.circle(
            x=x_start, y=y,
            source=source,
            size=config.marker_size,
            color=colors[0],
        )
        
        # End points
        p.circle(
            x=x_end, y=y,
            source=source,
            size=config.marker_size,
            color=colors[1],
        )
        
        self._apply_config(p, config)
        
        return p
    
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
        self._check_available()
        
        from bokeh.models import ColumnDataSource
        
        config = self._merge_config(config)
        color = self._get_colors(1, config)[0]
        
        source = ColumnDataSource(data)
        
        if horizontal:
            categories = list(data[y].astype(str))
            p = self._create_figure(config, y_range=categories)
            
            # Stems
            p.segment(
                x0=0, y0=y, x1=x, y1=y,
                source=source,
                line_color=color,
                line_width=config.line_width,
            )
            
            # Dots
            p.circle(
                x=x, y=y,
                source=source,
                size=config.marker_size,
                color=color,
            )
        else:
            categories = list(data[y].astype(str))
            p = self._create_figure(config, x_range=categories)
            
            p.segment(
                x0=y, y0=0, x1=y, y1=x,
                source=source,
                line_color=color,
                line_width=config.line_width,
            )
            
            p.circle(
                x=y, y=x,
                source=source,
                size=config.marker_size,
                color=color,
            )
        
        self._apply_config(p, config)
        
        return p
    
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
        # Use scatter with size mapping
        return self.scatter(data, x, y, size=size, color=color, config=config, **kwargs)
    
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
        self._check_available()
        
        from bokeh.models import ColumnDataSource
        from bokeh.transform import cumsum
        import math
        
        config = self._merge_config(config)
        colors = self._get_colors(len(data), config)
        
        # Calculate angles
        total = data[values].sum()
        df = data.copy()
        df["angle"] = df[values] / total * 2 * math.pi
        df["color"] = colors[:len(df)]
        
        source = ColumnDataSource(df)
        
        p = self._create_figure(config)
        
        p.annular_wedge(
            x=0, y=0,
            inner_radius=hole_size,
            outer_radius=1,
            start_angle=cumsum("angle", include_zero=True),
            end_angle=cumsum("angle"),
            fill_color="color",
            fill_alpha=config.alpha,
            line_color="white",
            source=source,
        )
        
        p.axis.visible = False
        p.grid.visible = False
        
        self._apply_config(p, config)
        
        return p
    
    # =========================================================================
    # PHASE 2: TREEMAP (Simplified - Bokeh doesn't have native treemap)
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
        """Create a treemap (simplified using rectangles)."""
        self._check_available()
        
        from bokeh.models import ColumnDataSource, LabelSet
        
        config = self._merge_config(config)
        colors = self._get_colors(len(data), config)
        
        # Simple layout: arrange in rows
        total = data[values].sum()
        df = data.copy().sort_values(values, ascending=False)
        df["pct"] = df[values] / total
        
        # Calculate positions (simple grid layout)
        n = len(df)
        cols = int(np.ceil(np.sqrt(n)))
        
        x_coords = []
        y_coords = []
        widths = []
        heights = []
        
        for i in range(n):
            row = i // cols
            col = i % cols
            x_coords.append(col)
            y_coords.append(-row)
            widths.append(0.9)
            heights.append(0.9)
        
        df["x"] = x_coords
        df["y"] = y_coords
        df["width"] = widths
        df["height"] = heights
        df["color"] = colors[:n]
        
        source = ColumnDataSource(df)
        
        p = self._create_figure(config)
        
        p.rect(
            x="x", y="y",
            width="width", height="height",
            source=source,
            fill_color="color",
            fill_alpha=config.alpha,
            line_color="white",
        )
        
        if config.show_labels:
            label_set = LabelSet(
                x="x", y="y",
                text=labels,
                source=source,
                text_align="center",
                text_baseline="middle",
                text_font_size=f"{config.label_font_size}pt",
            )
            p.add_layout(label_set)
        
        p.axis.visible = False
        p.grid.visible = False
        
        self._apply_config(p, config)
        
        return p
    
    # =========================================================================
    # PHASE 2: WORDCLOUD (Not natively supported in Bokeh)
    # =========================================================================
    
    def wordcloud(
        self,
        data: pd.DataFrame,
        text: str,
        weight: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Word cloud is not natively supported in Bokeh. Returns a text-based visualization."""
        self._check_available()
        
        from bokeh.models import ColumnDataSource, LabelSet
        
        config = self._merge_config(config)
        
        df = data.copy()
        if weight:
            df = df.sort_values(weight, ascending=False).head(50)
            max_w = df[weight].max()
            df["size"] = 8 + (df[weight] / max_w) * 24
        else:
            df["size"] = 12
        
        # Random positions
        np.random.seed(42)
        df["x"] = np.random.uniform(0, 10, len(df))
        df["y"] = np.random.uniform(0, 10, len(df))
        
        colors = self._get_colors(len(df), config)
        df["color"] = colors[:len(df)]
        
        source = ColumnDataSource(df)
        
        p = self._create_figure(config)
        
        labels = LabelSet(
            x="x", y="y",
            text=text,
            source=source,
            text_font_size={"field": "size", "transform": None},
            text_color="color",
            text_align="center",
        )
        p.add_layout(labels)
        
        p.axis.visible = False
        p.grid.visible = False
        
        self._apply_config(p, config)
        
        return p
    
    # =========================================================================
    # PHASE 2: RADAR CHART (Not natively supported in Bokeh)
    # =========================================================================
    
    def radar(
        self,
        data: pd.DataFrame,
        categories: List[str],
        values: Union[str, List[str]],
        config: Optional[PlotConfig] = None,
        **kwargs,
    ) -> Any:
        """Radar chart approximated with line plot in polar-like arrangement."""
        self._check_available()
        
        from bokeh.models import ColumnDataSource
        import math
        
        config = self._merge_config(config)
        
        n = len(categories)
        angles = [i * 2 * math.pi / n for i in range(n)]
        angles.append(angles[0])  # Close the shape
        
        value_cols = [values] if isinstance(values, str) else values
        colors = self._get_colors(len(value_cols), config)
        
        p = self._create_figure(config)
        
        for i, col in enumerate(value_cols):
            if col in data.columns:
                vals = data[col].values.tolist()
            else:
                vals = data.loc[col, categories].values.tolist()
            vals.append(vals[0])
            
            # Convert polar to cartesian
            x_coords = [v * math.cos(a) for v, a in zip(vals, angles)]
            y_coords = [v * math.sin(a) for v, a in zip(vals, angles)]
            
            p.patch(
                x_coords, y_coords,
                fill_color=colors[i],
                fill_alpha=config.fill_alpha,
                line_color=colors[i],
                line_width=config.line_width,
            )
        
        p.axis.visible = False
        p.grid.visible = False
        
        self._apply_config(p, config)
        
        return p
