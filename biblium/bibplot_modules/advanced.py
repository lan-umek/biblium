# -*- coding: utf-8 -*-
"""
Advanced Visualizations Module for Biblium.

Provides additional visualization methods including:
- Animated choropleth maps (geographic evolution over time)
- Interactive network graphs
- Streamgraphs for topic evolution
- Bump charts for ranking changes
- Alluvial/Sankey diagrams for flow visualization

Design Principles:
- No gridlines by default
- Consistent color schemes (using biblium.colors)
- No pie charts
- Save using save_plot() to PNG, SVG, PDF
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import numpy as np


class AdvancedVisualizationsMixin:
    """Mixin class providing advanced visualization methods for BiblioStats/BiblioAnalysis."""
    
    # =========================================================================
    # ANIMATED CHOROPLETH MAP
    # =========================================================================
    
    def _prepare_geographic_data(
        self,
        column: str,
        year_column: str = "Year",
        separator: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        cumulative: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare geographic count data over time.
        
        Parameters
        ----------
        column : str
            Column containing country/location data.
        year_column : str
            Column containing years.
        separator : str, optional
            Separator for multi-value cells.
        min_year, max_year : int, optional
            Year range limits.
        cumulative : bool
            Whether to compute cumulative counts.
            
        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with Year, Country, Count columns.
        """
        df = self.df.copy()
        
        if separator is None:
            separator = getattr(self, "default_separator", "; ")
        
        # Get year range
        years = df[year_column].dropna().astype(int)
        if min_year is None:
            min_year = years.min()
        if max_year is None:
            max_year = years.max()
        
        all_years = list(range(min_year, max_year + 1))
        
        # Build yearly counts
        yearly_data = []
        running_totals = {}
        
        for year in all_years:
            year_df = df[df[year_column] == year]
            
            # Count countries for this year
            items = year_df[column].dropna()
            
            from collections import Counter
            year_counts = Counter()
            
            for val in items:
                if isinstance(val, str):
                    if separator and separator in val:
                        for item in val.split(separator):
                            item = item.strip()
                            if item:
                                year_counts[item] += 1
                    else:
                        year_counts[val.strip()] += 1
            
            # Update running totals if cumulative
            if cumulative:
                for country, count in year_counts.items():
                    running_totals[country] = running_totals.get(country, 0) + count
                
                for country, total in running_totals.items():
                    yearly_data.append({
                        "Year": year,
                        "Country": country,
                        "Count": total,
                    })
            else:
                for country, count in year_counts.items():
                    yearly_data.append({
                        "Year": year,
                        "Country": country,
                        "Count": count,
                    })
        
        return pd.DataFrame(yearly_data)
    
    def create_animated_choropleth(
        self,
        column: str = None,
        year_column: str = "Year",
        separator: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        cumulative: bool = True,
        title: Optional[str] = None,
        cmap: str = "YlOrRd",
        figsize: Tuple[int, int] = (14, 8),
        duration: float = 10.0,
        fps: int = 10,
        output_path: Optional[str] = None,
        output_format: Literal["mp4", "gif"] = "gif",
        show_colorbar: bool = True,
        projection: str = "natural_earth",
    ) -> Any:
        """
        Create an animated choropleth map showing geographic distribution over time.
        
        Parameters
        ----------
        column : str, optional
            Column containing country data. Auto-detects if None.
        year_column : str
            Column containing years.
        separator : str, optional
            Separator for multi-value cells.
        min_year, max_year : int, optional
            Year range limits.
        cumulative : bool
            Show cumulative counts (True) or yearly counts (False).
        title : str, optional
            Map title.
        cmap : str
            Matplotlib colormap (default: YlOrRd).
        figsize : tuple
            Figure size.
        duration : float
            Animation duration in seconds.
        fps : int
            Frames per second.
        output_path : str, optional
            Path to save animation.
        output_format : str
            Output format: "mp4" or "gif".
        show_colorbar : bool
            Whether to show colorbar.
        projection : str
            Map projection type.
            
        Returns
        -------
        matplotlib.animation.FuncAnimation or str
            Animation object or path to saved file.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        # Auto-detect country column
        if column is None:
            country_cols = ["Countries", "Country", "authorships.countries", 
                          "CA Country", "Corresponding Author Country"]
            for col in country_cols:
                if hasattr(self, "_get_column"):
                    found = self._get_column(col, required=False)
                    if found:
                        column = found
                        break
                elif col in self.df.columns:
                    column = col
                    break
            
            if column is None:
                raise ValueError("Could not auto-detect country column. Please specify.")
        
        # Prepare data
        geo_df = self._prepare_geographic_data(
            column=column,
            year_column=year_column,
            separator=separator,
            min_year=min_year,
            max_year=max_year,
            cumulative=cumulative,
        )
        
        if geo_df.empty:
            raise ValueError("No geographic data found")
        
        years = sorted(geo_df["Year"].unique())
        
        # Try to import geopandas for proper map
        try:
            import geopandas as gpd
            HAS_GEOPANDAS = True
            
            # Load world map
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            world = world[world["name"] != "Antarctica"]
        except ImportError:
            HAS_GEOPANDAS = False
            print("Note: Install geopandas for proper map rendering. Using simple visualization.")
        
        # Set up figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get global max for consistent color scale
        global_max = geo_df["Count"].max()
        norm = Normalize(vmin=0, vmax=global_max)
        
        # Generate title
        if title is None:
            count_type = "Cumulative" if cumulative else "Annual"
            title = f"{count_type} Publications by Country"
        
        def animate(frame_idx):
            ax.clear()
            
            year = years[frame_idx]
            year_data = geo_df[geo_df["Year"] == year].copy()
            
            if HAS_GEOPANDAS:
                # Merge with world map
                # Need to map country names/codes
                merged = world.copy()
                
                # Create country name mapping
                country_counts = year_data.set_index("Country")["Count"].to_dict()
                
                # Map by name or ISO code
                merged["count"] = merged["name"].map(country_counts).fillna(0)
                merged["count"] = merged["count"] + merged["iso_a3"].map(country_counts).fillna(0)
                
                # Plot
                merged.plot(
                    column="count",
                    ax=ax,
                    cmap=cmap,
                    legend=False,
                    edgecolor="white",
                    linewidth=0.3,
                    vmin=0,
                    vmax=global_max,
                    missing_kwds={"color": "lightgray", "edgecolor": "white", "linewidth": 0.3},
                )
            else:
                # Simple bar representation without geopandas
                top_countries = year_data.nlargest(15, "Count")
                y_pos = range(len(top_countries))
                
                from biblium.colors import get_colors
                colors = get_colors(len(top_countries))
                
                ax.barh(y_pos, top_countries["Count"], color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_countries["Country"])
                ax.set_xlabel("Publications")
                ax.invert_yaxis()
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            
            # Title with year
            ax.set_title(f"{title}\n{year}", fontsize=14, fontweight="bold")
            
            # Remove axes for map
            if HAS_GEOPANDAS:
                ax.axis("off")
            
            # No grid
            ax.grid(False)
        
        # Add colorbar
        if show_colorbar and HAS_GEOPANDAS:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label("Publications", fontsize=10)
        
        # Calculate interval
        total_frames = len(years)
        interval = (duration * 1000) / total_frames
        
        # Create animation
        anim = FuncAnimation(
            fig,
            animate,
            frames=total_frames,
            interval=interval,
            blit=False,
            repeat=False,
        )
        
        # Save or return
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            if output_format == "mp4":
                try:
                    from matplotlib.animation import FFMpegWriter
                    writer = FFMpegWriter(fps=fps)
                    anim.save(output_path, writer=writer, dpi=150)
                except Exception as e:
                    print(f"MP4 export failed: {e}. Trying GIF...")
                    output_path = output_path.replace(".mp4", ".gif")
                    anim.save(output_path, writer="pillow", fps=fps, dpi=100)
            else:
                anim.save(output_path, writer="pillow", fps=fps, dpi=100)
            
            plt.close(fig)
            print(f"Animation saved to: {output_path}")
            return output_path
        
        return anim
    
    # =========================================================================
    # STREAMGRAPH
    # =========================================================================
    
    def plot_streamgraph(
        self,
        column: str,
        year_column: str = "Year",
        separator: Optional[str] = None,
        top_n: int = 10,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        title: Optional[str] = None,
        colors: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 7),
        baseline: Literal["zero", "sym", "wiggle", "weighted_wiggle"] = "wiggle",
        alpha: float = 0.8,
        filename: Optional[str] = None,
        dpi: int = 600,
    ) -> None:
        """
        Create a streamgraph showing topic/keyword evolution over time.
        
        Parameters
        ----------
        column : str
            Column containing items to track (e.g., keywords).
        year_column : str
            Column containing years.
        separator : str, optional
            Separator for multi-value cells.
        top_n : int
            Number of top items to show.
        min_year, max_year : int, optional
            Year range limits.
        title : str, optional
            Plot title.
        colors : list, optional
            Custom colors for streams.
        figsize : tuple
            Figure size.
        baseline : str
            Baseline type: "zero", "sym", "wiggle", "weighted_wiggle".
        alpha : float
            Transparency of streams.
        filename : str, optional
            Base filename for saving.
        dpi : int
            Resolution for saving.
        """
        import matplotlib.pyplot as plt
        from biblium.plotbib import save_plot
        from biblium.colors import get_colors
        
        if separator is None:
            separator = getattr(self, "default_separator", "; ")
        
        df = self.df.copy()
        
        # Get year range
        years = df[year_column].dropna().astype(int)
        if min_year is None:
            min_year = years.min()
        if max_year is None:
            max_year = years.max()
        
        all_years = list(range(min_year, max_year + 1))
        
        # Count items per year
        from collections import Counter
        yearly_counts = {year: Counter() for year in all_years}
        
        for _, row in df.iterrows():
            year = row.get(year_column)
            if pd.isna(year) or int(year) < min_year or int(year) > max_year:
                continue
            
            year = int(year)
            items = row.get(column)
            
            if pd.isna(items):
                continue
            
            if isinstance(items, str):
                if separator and separator in items:
                    for item in items.split(separator):
                        item = item.strip()
                        if item:
                            yearly_counts[year][item] += 1
                else:
                    yearly_counts[year][items.strip()] += 1
        
        # Get top N items overall
        total_counts = Counter()
        for year_counter in yearly_counts.values():
            total_counts.update(year_counter)
        
        top_items = [item for item, _ in total_counts.most_common(top_n)]
        
        # Build data matrix
        data = np.zeros((len(top_items), len(all_years)))
        for j, year in enumerate(all_years):
            for i, item in enumerate(top_items):
                data[i, j] = yearly_counts[year].get(item, 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate colors
        if colors is None:
            colors = get_colors(len(top_items))
        
        # Plot streamgraph
        ax.stackplot(
            all_years,
            data,
            labels=top_items,
            colors=colors,
            alpha=alpha,
            baseline=baseline,
        )
        
        # Styling
        ax.set_xlim(min_year, max_year)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Publications", fontsize=12)
        
        if title is None:
            title = f"Evolution of Top {top_n} {column}"
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        # Legend outside
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
        
        # No grid
        ax.grid(False)
        
        # Remove top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        plt.tight_layout()
        
        # Save
        if filename:
            save_plot(filename, dpi=dpi)
        elif self.res_folder:
            save_plot(os.path.join(self.res_folder, "figures", "streamgraph"), dpi=dpi)
    
    # =========================================================================
    # BUMP CHART
    # =========================================================================
    
    def plot_bump_chart(
        self,
        column: str,
        year_column: str = "Year",
        separator: Optional[str] = None,
        top_n: int = 10,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        title: Optional[str] = None,
        colors: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 8),
        marker_size: int = 100,
        line_width: float = 2.5,
        show_labels: bool = True,
        filename: Optional[str] = None,
        dpi: int = 600,
    ) -> None:
        """
        Create a bump chart showing ranking changes over time.
        
        Parameters
        ----------
        column : str
            Column containing items to rank.
        year_column : str
            Column containing years.
        separator : str, optional
            Separator for multi-value cells.
        top_n : int
            Number of top items to track.
        min_year, max_year : int, optional
            Year range.
        title : str, optional
            Plot title.
        colors : list, optional
            Custom colors.
        figsize : tuple
            Figure size.
        marker_size : int
            Size of rank markers.
        line_width : float
            Width of connecting lines.
        show_labels : bool
            Show item labels.
        filename : str, optional
            Base filename for saving.
        dpi : int
            Resolution.
        """
        import matplotlib.pyplot as plt
        from biblium.plotbib import save_plot
        from biblium.colors import get_colors
        
        if separator is None:
            separator = getattr(self, "default_separator", "; ")
        
        df = self.df.copy()
        
        # Get year range
        years = df[year_column].dropna().astype(int)
        if min_year is None:
            min_year = years.min()
        if max_year is None:
            max_year = years.max()
        
        all_years = list(range(min_year, max_year + 1))
        
        # Count items per year
        from collections import Counter
        yearly_counts = {year: Counter() for year in all_years}
        
        for _, row in df.iterrows():
            year = row.get(year_column)
            if pd.isna(year) or int(year) < min_year or int(year) > max_year:
                continue
            
            year = int(year)
            items = row.get(column)
            
            if pd.isna(items):
                continue
            
            if isinstance(items, str):
                if separator and separator in items:
                    for item in items.split(separator):
                        item = item.strip()
                        if item:
                            yearly_counts[year][item] += 1
                else:
                    yearly_counts[year][items.strip()] += 1
        
        # Get items that appear in top N at least once
        all_top_items = set()
        for year in all_years:
            top_in_year = [item for item, _ in yearly_counts[year].most_common(top_n)]
            all_top_items.update(top_in_year)
        
        # Limit to actual top N based on total
        total_counts = Counter()
        for yc in yearly_counts.values():
            total_counts.update(yc)
        
        top_items = [item for item, _ in total_counts.most_common(top_n) if item in all_top_items][:top_n]
        
        # Compute rankings per year
        rankings = {item: [] for item in top_items}
        
        for year in all_years:
            sorted_items = [item for item, _ in yearly_counts[year].most_common()]
            
            for item in top_items:
                if item in sorted_items:
                    rank = sorted_items.index(item) + 1
                    rankings[item].append(min(rank, top_n + 1))  # Cap at top_n + 1
                else:
                    rankings[item].append(top_n + 1)  # Not in top
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate colors
        if colors is None:
            colors = get_colors(len(top_items))
        
        # Plot lines and markers
        for i, item in enumerate(top_items):
            ranks = rankings[item]
            color = colors[i % len(colors)]
            
            # Plot line
            ax.plot(all_years, ranks, color=color, linewidth=line_width, alpha=0.7)
            
            # Plot markers
            ax.scatter(all_years, ranks, color=color, s=marker_size, zorder=5)
            
            # Add label at the end
            if show_labels:
                # Truncate long labels
                label = item[:25] + "..." if len(item) > 25 else item
                ax.annotate(
                    label,
                    (all_years[-1], ranks[-1]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=color,
                )
        
        # Styling
        ax.set_xlim(min_year - 0.5, max_year + 2)
        ax.set_ylim(top_n + 1.5, 0.5)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Rank", fontsize=12)
        ax.set_yticks(range(1, top_n + 1))
        
        if title is None:
            title = f"Ranking Changes: Top {top_n} {column}"
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        # No grid
        ax.grid(False)
        
        # Light horizontal lines for ranks
        for rank in range(1, top_n + 1):
            ax.axhline(y=rank, color="lightgray", linewidth=0.5, zorder=0)
        
        # Remove top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        plt.tight_layout()
        
        # Save
        if filename:
            save_plot(filename, dpi=dpi)
        elif self.res_folder:
            save_plot(os.path.join(self.res_folder, "figures", "bump_chart"), dpi=dpi)
    
    # =========================================================================
    # INTERACTIVE NETWORK (HTML export)
    # =========================================================================
    
    def create_interactive_network(
        self,
        column: str,
        separator: Optional[str] = None,
        top_n: int = 50,
        min_edge_weight: int = 2,
        title: Optional[str] = None,
        output_path: Optional[str] = None,
        width: str = "100%",
        height: str = "800px",
        bgcolor: str = "#ffffff",
        font_color: str = "#333333",
    ) -> Optional[str]:
        """
        Create an interactive network visualization (HTML).
        
        Requires pyvis: pip install pyvis
        
        Parameters
        ----------
        column : str
            Column containing items for co-occurrence.
        separator : str, optional
            Separator for multi-value cells.
        top_n : int
            Number of top items to include.
        min_edge_weight : int
            Minimum co-occurrence for edge.
        title : str, optional
            Network title.
        output_path : str, optional
            Path to save HTML file.
        width, height : str
            Network dimensions.
        bgcolor : str
            Background color.
        font_color : str
            Font color.
            
        Returns
        -------
        str or None
            Path to saved HTML file.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            print("pyvis required for interactive networks: pip install pyvis")
            return None
        
        from collections import Counter
        from itertools import combinations
        
        if separator is None:
            separator = getattr(self, "default_separator", "; ")
        
        df = self.df.copy()
        
        # Count items
        item_counts = Counter()
        for val in df[column].dropna():
            if isinstance(val, str):
                if separator and separator in val:
                    items = [x.strip() for x in val.split(separator) if x.strip()]
                else:
                    items = [val.strip()]
                item_counts.update(items)
        
        # Get top items
        top_items = set([item for item, _ in item_counts.most_common(top_n)])
        
        # Count co-occurrences
        cooccurrences = Counter()
        for val in df[column].dropna():
            if isinstance(val, str):
                if separator and separator in val:
                    items = [x.strip() for x in val.split(separator) if x.strip()]
                else:
                    items = [val.strip()]
                
                # Filter to top items
                items = [i for i in items if i in top_items]
                
                # Count pairs
                for pair in combinations(sorted(set(items)), 2):
                    cooccurrences[pair] += 1
        
        # Create network
        net = Network(
            height=height,
            width=width,
            bgcolor=bgcolor,
            font_color=font_color,
            directed=False,
        )
        
        # Add nodes
        from biblium.colors import get_colors
        colors = get_colors(len(top_items))
        
        for i, item in enumerate(top_items):
            size = 10 + (item_counts[item] / max(item_counts.values())) * 30
            net.add_node(
                item,
                label=item[:30],
                size=size,
                color=colors[i % len(colors)],
                title=f"{item}: {item_counts[item]} occurrences",
            )
        
        # Add edges
        for (item1, item2), weight in cooccurrences.items():
            if weight >= min_edge_weight:
                net.add_edge(item1, item2, value=weight, title=f"Co-occurs {weight} times")
        
        # Set title
        if title:
            net.heading = title
        
        # Physics settings
        net.set_options("""
        var options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "solver": "forceAtlas2Based"
            }
        }
        """)
        
        # Save
        if output_path is None:
            if self.res_folder:
                output_path = os.path.join(self.res_folder, "figures", "interactive_network.html")
            else:
                output_path = "interactive_network.html"
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        net.save_graph(output_path)
        print(f"Interactive network saved to: {output_path}")
        
        return output_path
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def animated_country_map(
        self,
        cumulative: bool = True,
        **kwargs,
    ) -> Any:
        """
        Create animated choropleth for countries.
        
        Parameters
        ----------
        cumulative : bool
            Show cumulative (True) or annual (False) counts.
        **kwargs
            Additional arguments for create_animated_choropleth.
        """
        # Try to find country column
        country_cols = ["Countries", "Country", "authorships.countries", "CA Country"]
        
        column = None
        for col in country_cols:
            if hasattr(self, "_get_column"):
                found = self._get_column(col, required=False)
                if found:
                    column = found
                    break
            elif col in self.df.columns:
                column = col
                break
        
        if column is None:
            raise ValueError("No country column found")
        
        return self.create_animated_choropleth(
            column=column,
            cumulative=cumulative,
            **kwargs,
        )
    
    def keyword_streamgraph(
        self,
        top_n: int = 10,
        **kwargs,
    ) -> None:
        """
        Create streamgraph for author keywords.
        
        Parameters
        ----------
        top_n : int
            Number of keywords to show.
        **kwargs
            Additional arguments for plot_streamgraph.
        """
        kw_col = self._get_column("Author Keywords", required=False) if hasattr(self, "_get_column") else "Author Keywords"
        
        if kw_col is None or kw_col not in self.df.columns:
            raise ValueError("Author Keywords column not found")
        
        self.plot_streamgraph(
            column=kw_col,
            top_n=top_n,
            title=f"Evolution of Top {top_n} Keywords",
            **kwargs,
        )
    
    def source_bump_chart(
        self,
        top_n: int = 10,
        **kwargs,
    ) -> None:
        """
        Create bump chart for sources/journals.
        
        Parameters
        ----------
        top_n : int
            Number of sources to track.
        **kwargs
            Additional arguments for plot_bump_chart.
        """
        source_col = self._get_column("Source title", required=False) if hasattr(self, "_get_column") else "Source title"
        
        if source_col is None or source_col not in self.df.columns:
            raise ValueError("Source title column not found")
        
        self.plot_bump_chart(
            column=source_col,
            top_n=top_n,
            title=f"Journal Ranking Changes: Top {top_n}",
            **kwargs,
        )
    
    def keyword_network_interactive(
        self,
        top_n: int = 50,
        **kwargs,
    ) -> Optional[str]:
        """
        Create interactive keyword co-occurrence network.
        
        Parameters
        ----------
        top_n : int
            Number of keywords to include.
        **kwargs
            Additional arguments for create_interactive_network.
        """
        kw_col = self._get_column("Author Keywords", required=False) if hasattr(self, "_get_column") else "Author Keywords"
        
        if kw_col is None or kw_col not in self.df.columns:
            raise ValueError("Author Keywords column not found")
        
        return self.create_interactive_network(
            column=kw_col,
            top_n=top_n,
            title="Keyword Co-occurrence Network",
            **kwargs,
        )
