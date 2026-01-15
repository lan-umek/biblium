# -*- coding: utf-8 -*-
"""
Race Bar Animation Module for Biblium.

Provides animated bar chart races showing cumulative counts over time
for various bibliometric entities (sources, authors, keywords, etc.).

Features:
- Cumulative count animations
- Customizable colors, speed, and display options
- Export to MP4, GIF, or HTML
- Support for sources, authors, keywords, affiliations, countries, etc.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import pandas as pd
import numpy as np


class RaceBarMixin:
    """Mixin class providing race bar animation methods for BiblioStats/BiblioAnalysis."""
    
    def _prepare_race_data(
        self,
        column: str,
        year_column: str = "Year",
        separator: Optional[str] = None,
        top_n: int = 15,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Prepare cumulative count data for race bar animation.
        
        Parameters
        ----------
        column : str
            Column containing items to count (e.g., "Source title", "Authors").
        year_column : str
            Column containing publication years.
        separator : str, optional
            Separator for splitting multi-value cells. If None, uses default.
        top_n : int
            Number of top items to track in animation.
        min_year : int, optional
            Start year for animation.
        max_year : int, optional
            End year for animation.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with years as index, items as columns, cumulative counts as values.
        """
        df = self.df.copy()
        
        # Get year range
        if year_column not in df.columns:
            raise ValueError(f"Year column '{year_column}' not found in data")
        
        years = df[year_column].dropna().astype(int)
        if min_year is None:
            min_year = years.min()
        if max_year is None:
            max_year = years.max()
        
        all_years = list(range(min_year, max_year + 1))
        
        # Get separator
        if separator is None:
            separator = getattr(self, "default_separator", "; ")
        
        # Check if column exists
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        # Build yearly counts
        yearly_counts = {}
        
        for year in all_years:
            year_df = df[df[year_column] == year]
            
            # Count items for this year
            items = year_df[column].dropna()
            
            # Split if separator present
            all_items = []
            for val in items:
                if isinstance(val, str):
                    if separator and separator in val:
                        all_items.extend([x.strip() for x in val.split(separator) if x.strip()])
                    else:
                        all_items.append(val.strip())
            
            # Count occurrences
            from collections import Counter
            counts = Counter(all_items)
            yearly_counts[year] = counts
        
        # Build cumulative DataFrame
        all_items_set = set()
        for counts in yearly_counts.values():
            all_items_set.update(counts.keys())
        
        # Create cumulative counts matrix
        cumulative_data = {}
        running_totals = {item: 0 for item in all_items_set}
        
        for year in all_years:
            for item in all_items_set:
                running_totals[item] += yearly_counts[year].get(item, 0)
            cumulative_data[year] = running_totals.copy()
        
        # Convert to DataFrame
        race_df = pd.DataFrame(cumulative_data).T
        race_df.index.name = "Year"
        
        # Get top N items by final count
        final_counts = race_df.iloc[-1].sort_values(ascending=False)
        top_items = final_counts.head(top_n).index.tolist()
        
        # Filter to top items only
        race_df = race_df[top_items]
        
        return race_df
    
    def create_race_bar_animation(
        self,
        column: str,
        year_column: str = "Year",
        separator: Optional[str] = None,
        top_n: int = 15,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        title: Optional[str] = None,
        xlabel: str = "Cumulative Count",
        colors: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        duration: float = 15.0,
        fps: int = 30,
        output_path: Optional[str] = None,
        output_format: Literal["mp4", "gif", "html"] = "mp4",
        show_value_labels: bool = True,
        bar_height: float = 0.8,
        interpolate_frames: int = 10,
        dark_mode: bool = False,
    ) -> Any:
        """
        Create an animated race bar chart showing cumulative counts over time.
        
        Parameters
        ----------
        column : str
            Column containing items to count (e.g., "Source title", "Authors").
        year_column : str
            Column containing publication years.
        separator : str, optional
            Separator for splitting multi-value cells.
        top_n : int
            Number of top items to show (default: 15).
        min_year : int, optional
            Start year for animation.
        max_year : int, optional
            End year for animation.
        title : str, optional
            Animation title. Auto-generated if None.
        xlabel : str
            X-axis label.
        colors : list, optional
            List of colors for bars. Auto-generated if None.
        figsize : tuple
            Figure size (width, height).
        duration : float
            Total animation duration in seconds.
        fps : int
            Frames per second.
        output_path : str, optional
            Path to save animation. If None, returns the animation object.
        output_format : str
            Output format: "mp4", "gif", or "html".
        show_value_labels : bool
            Whether to show count values on bars.
        bar_height : float
            Height of bars (0-1).
        interpolate_frames : int
            Number of interpolated frames between years for smoother animation.
        dark_mode : bool
            Use dark theme.
            
        Returns
        -------
        matplotlib.animation.FuncAnimation or str
            Animation object or path to saved file.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib import rcParams
        
        # Prepare data
        race_df = self._prepare_race_data(
            column=column,
            year_column=year_column,
            separator=separator,
            top_n=top_n,
            min_year=min_year,
            max_year=max_year,
        )
        
        years = race_df.index.tolist()
        items = race_df.columns.tolist()
        n_items = len(items)
        
        # Generate colors
        if colors is None:
            from biblium.colors import get_colors
            colors = get_colors(n_items)
        
        # Create color mapping
        color_map = {item: colors[i % len(colors)] for i, item in enumerate(items)}
        
        # Generate title
        if title is None:
            title = f"Cumulative {column} Over Time"
        
        # Set up figure
        if dark_mode:
            plt.style.use("dark_background")
            bg_color = "#1a1a2e"
            text_color = "white"
        else:
            plt.style.use("default")
            bg_color = "white"
            text_color = "black"
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        # Calculate total frames
        n_years = len(years)
        total_frames = (n_years - 1) * interpolate_frames + 1
        
        # Interpolate data for smooth animation
        def get_frame_data(frame_idx):
            """Get data for a specific frame with interpolation."""
            year_idx = frame_idx // interpolate_frames
            frame_within_year = frame_idx % interpolate_frames
            
            if year_idx >= n_years - 1:
                return race_df.iloc[-1], years[-1]
            
            # Interpolate between years
            current_data = race_df.iloc[year_idx]
            next_data = race_df.iloc[year_idx + 1]
            
            t = frame_within_year / interpolate_frames
            interpolated = current_data + (next_data - current_data) * t
            
            # Interpolate year for display
            current_year = years[year_idx]
            next_year = years[year_idx + 1]
            display_year = current_year + (next_year - current_year) * t
            
            return interpolated, display_year
        
        def animate(frame_idx):
            """Animation function for each frame."""
            ax.clear()
            
            # Get interpolated data
            data, display_year = get_frame_data(frame_idx)
            
            # Sort by value for this frame
            sorted_data = data.sort_values(ascending=True)
            
            # Create horizontal bars
            y_pos = np.arange(len(sorted_data))
            bar_colors = [color_map[item] for item in sorted_data.index]
            
            bars = ax.barh(
                y_pos, 
                sorted_data.values, 
                height=bar_height,
                color=bar_colors,
                edgecolor="none",
            )
            
            # Add value labels
            if show_value_labels:
                for i, (val, item) in enumerate(zip(sorted_data.values, sorted_data.index)):
                    # Value at end of bar
                    ax.text(
                        val + max(sorted_data.values) * 0.01,
                        i,
                        f"{int(val):,}",
                        va="center",
                        ha="left",
                        fontsize=10,
                        color=text_color,
                        fontweight="bold",
                    )
                    # Item name inside bar
                    ax.text(
                        max(sorted_data.values) * 0.01,
                        i,
                        item[:40] + "..." if len(item) > 40 else item,
                        va="center",
                        ha="left",
                        fontsize=9,
                        color="white" if not dark_mode else "white",
                        fontweight="bold",
                    )
            
            # Year display (large, in corner)
            ax.text(
                0.95, 0.15,
                f"{int(display_year)}",
                transform=ax.transAxes,
                fontsize=60,
                ha="right",
                va="bottom",
                color=text_color,
                alpha=0.3,
                fontweight="bold",
            )
            
            # Styling
            ax.set_xlim(0, race_df.values.max() * 1.15)
            ax.set_ylim(-0.5, n_items - 0.5)
            ax.set_yticks([])
            ax.set_xlabel(xlabel, fontsize=12, color=text_color)
            ax.set_title(title, fontsize=16, fontweight="bold", color=text_color, pad=20)
            
            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            ax.tick_params(colors=text_color)
            
            return bars
        
        # Calculate interval for desired duration
        interval = (duration * 1000) / total_frames  # ms per frame
        
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
            # Ensure directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            if output_format == "mp4":
                try:
                    from matplotlib.animation import FFMpegWriter
                    writer = FFMpegWriter(fps=fps, metadata={"title": title})
                    anim.save(output_path, writer=writer, dpi=150)
                except Exception as e:
                    print(f"MP4 export failed (ffmpeg may not be installed): {e}")
                    print("Trying GIF export instead...")
                    output_path = output_path.replace(".mp4", ".gif")
                    anim.save(output_path, writer="pillow", fps=fps, dpi=100)
            elif output_format == "gif":
                anim.save(output_path, writer="pillow", fps=fps, dpi=100)
            elif output_format == "html":
                from matplotlib.animation import HTMLWriter
                anim.save(output_path, writer="html", fps=fps)
            
            plt.close(fig)
            print(f"Animation saved to: {output_path}")
            return output_path
        
        return anim
    
    # Convenience methods for common entity types
    
    def race_bar_sources(
        self,
        top_n: int = 15,
        title: str = "Top Sources by Cumulative Publications",
        **kwargs,
    ) -> Any:
        """
        Create race bar animation for sources/journals.
        
        Parameters
        ----------
        top_n : int
            Number of top sources to show.
        title : str
            Animation title.
        **kwargs
            Additional arguments passed to create_race_bar_animation.
            
        Returns
        -------
        Animation object or saved file path.
        """
        source_col = self._get_column("Source title", required=False)
        if source_col is None:
            raise ValueError("Source title column not found in data")
        
        return self.create_race_bar_animation(
            column=source_col,
            top_n=top_n,
            title=title,
            separator=None,  # Sources typically not multi-valued
            **kwargs,
        )
    
    def race_bar_authors(
        self,
        top_n: int = 15,
        title: str = "Top Authors by Cumulative Publications",
        **kwargs,
    ) -> Any:
        """
        Create race bar animation for authors.
        
        Parameters
        ----------
        top_n : int
            Number of top authors to show.
        title : str
            Animation title.
        **kwargs
            Additional arguments passed to create_race_bar_animation.
            
        Returns
        -------
        Animation object or saved file path.
        """
        author_col = self._get_column("Authors", required=False)
        if author_col is None:
            raise ValueError("Authors column not found in data")
        
        return self.create_race_bar_animation(
            column=author_col,
            top_n=top_n,
            title=title,
            **kwargs,
        )
    
    def race_bar_author_keywords(
        self,
        top_n: int = 15,
        title: str = "Top Author Keywords by Cumulative Frequency",
        **kwargs,
    ) -> Any:
        """
        Create race bar animation for author keywords.
        
        Parameters
        ----------
        top_n : int
            Number of top keywords to show.
        title : str
            Animation title.
        **kwargs
            Additional arguments passed to create_race_bar_animation.
            
        Returns
        -------
        Animation object or saved file path.
        """
        kw_col = self._get_column("Author Keywords", required=False)
        if kw_col is None:
            raise ValueError("Author Keywords column not found in data")
        
        return self.create_race_bar_animation(
            column=kw_col,
            top_n=top_n,
            title=title,
            **kwargs,
        )
    
    def race_bar_index_keywords(
        self,
        top_n: int = 15,
        title: str = "Top Index Keywords by Cumulative Frequency",
        **kwargs,
    ) -> Any:
        """
        Create race bar animation for index keywords.
        
        Parameters
        ----------
        top_n : int
            Number of top keywords to show.
        title : str
            Animation title.
        **kwargs
            Additional arguments passed to create_race_bar_animation.
            
        Returns
        -------
        Animation object or saved file path.
        """
        kw_col = self._get_column("Index Keywords", required=False)
        if kw_col is None:
            raise ValueError("Index Keywords column not found in data")
        
        return self.create_race_bar_animation(
            column=kw_col,
            top_n=top_n,
            title=title,
            **kwargs,
        )
    
    def race_bar_countries(
        self,
        top_n: int = 15,
        title: str = "Top Countries by Cumulative Publications",
        **kwargs,
    ) -> Any:
        """
        Create race bar animation for countries.
        
        Parameters
        ----------
        top_n : int
            Number of top countries to show.
        title : str
            Animation title.
        **kwargs
            Additional arguments passed to create_race_bar_animation.
            
        Returns
        -------
        Animation object or saved file path.
        """
        # Try different possible country column names
        country_cols = ["Countries", "Country", "Affiliations Country", "authorships.countries"]
        
        country_col = None
        for col in country_cols:
            found = self._get_column(col, required=False)
            if found:
                country_col = found
                break
        
        if country_col is None:
            raise ValueError("Country column not found in data")
        
        return self.create_race_bar_animation(
            column=country_col,
            top_n=top_n,
            title=title,
            **kwargs,
        )
    
    def race_bar_affiliations(
        self,
        top_n: int = 15,
        title: str = "Top Affiliations by Cumulative Publications",
        **kwargs,
    ) -> Any:
        """
        Create race bar animation for affiliations/institutions.
        
        Parameters
        ----------
        top_n : int
            Number of top affiliations to show.
        title : str
            Animation title.
        **kwargs
            Additional arguments passed to create_race_bar_animation.
            
        Returns
        -------
        Animation object or saved file path.
        """
        aff_col = self._get_column("Affiliations", required=False)
        if aff_col is None:
            raise ValueError("Affiliations column not found in data")
        
        return self.create_race_bar_animation(
            column=aff_col,
            top_n=top_n,
            title=title,
            **kwargs,
        )


# Standalone function for use without class
def create_race_bar_from_dataframe(
    df: pd.DataFrame,
    column: str,
    year_column: str = "Year",
    separator: str = "; ",
    top_n: int = 15,
    **kwargs,
) -> Any:
    """
    Create a race bar animation from a DataFrame.
    
    This is a standalone function that doesn't require a BiblioStats instance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with bibliometric data.
    column : str
        Column containing items to count.
    year_column : str
        Column containing publication years.
    separator : str
        Separator for multi-value cells.
    top_n : int
        Number of top items to show.
    **kwargs
        Additional arguments passed to animation creation.
        
    Returns
    -------
    Animation object or saved file path.
    """
    # Create a minimal wrapper class
    class MinimalWrapper:
        def __init__(self, dataframe, sep):
            self.df = dataframe
            self.default_separator = sep
        
        def _get_column(self, name, required=True):
            # Simple column lookup
            if name in self.df.columns:
                return name
            # Try case-insensitive
            for col in self.df.columns:
                if col.lower() == name.lower():
                    return col
            return None
    
    # Add mixin methods
    wrapper = MinimalWrapper(df, separator)
    wrapper._prepare_race_data = RaceBarMixin._prepare_race_data.__get__(wrapper, MinimalWrapper)
    wrapper.create_race_bar_animation = RaceBarMixin.create_race_bar_animation.__get__(wrapper, MinimalWrapper)
    
    return wrapper.create_race_bar_animation(
        column=column,
        year_column=year_column,
        separator=separator,
        top_n=top_n,
        **kwargs,
    )
