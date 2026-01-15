# -*- coding: utf-8 -*-
"""
Geographic plotting functions - country maps, collaboration maps.

This module contains methods for:
- Country performance maps
- Country collaboration maps
- Country collaboration networks
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd


class GeographicPlotsMixin:
    """Mixin class providing geographic plotting methods."""
    
    def plot_ca_countries_map(
        self,
        x: str = "Number of documents",
        filename_prefix: str = "country performance map",
        **kwargs: Any,
    ) -> None:
        """
        Plot corresponding author countries on a world map.

        Parameters
        ----------
        x : str
            Metric to display (column from ca_country_counts_df).
        filename_prefix : str
            Prefix for saved files.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import plotbib
        
        # Ensure country counts exist
        if not hasattr(self, "ca_country_counts_df"):
            self.count_ca_countries()
        
        if self.ca_country_counts_df is None or self.ca_country_counts_df.empty:
            return
        
        plotbib.plot_choropleth(
            self.ca_country_counts_df,
            location_col="Country",
            value_col=x,
            title=f"Corresponding Author Countries - {x}",
            **kwargs,
        )
        
        if self.res_folder is not None:
            self._save_plot(f"{filename_prefix}_{x.replace(' ', '_')}")

    def plot_all_countries_map(
        self,
        x: str = "Number of documents",
        filename_prefix: str = "country collaboration map",
        **kwargs: Any,
    ) -> None:
        """
        Plot all countries (from affiliations) on a world map.

        Parameters
        ----------
        x : str
            Metric to display.
        filename_prefix : str
            Prefix for saved files.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import plotbib
        
        # Ensure all countries are counted
        if not hasattr(self, "all_countries_counts_df"):
            self.count_all_countries()
        
        if self.all_countries_counts_df is None or self.all_countries_counts_df.empty:
            return
        
        plotbib.plot_choropleth(
            self.all_countries_counts_df,
            location_col="Country",
            value_col=x,
            title=f"All Countries (Affiliations) - {x}",
            **kwargs,
        )
        
        if self.res_folder is not None:
            self._save_plot(f"{filename_prefix}_{x.replace(' ', '_')}")

    def plot_country_collaboration(
        self,
        top_n_pairs: int = 20,
        connect_threshold: int = 1,
        top_n_countries: int = 20,
        annotate_heatmap: bool = True,
        figsizes: Optional[Dict[str, Tuple[int, int]]] = None,
        filename: str = "country collaboration",
        **kwargs: Any,
    ) -> None:
        """
        Plot country collaboration analysis with multiple visualizations.

        Creates three plots:
        1. Top collaborating country pairs (bar chart)
        2. Collaboration network
        3. Collaboration heatmap

        Parameters
        ----------
        top_n_pairs : int
            Number of top country pairs to show.
        connect_threshold : int
            Minimum connections for network edges.
        top_n_countries : int
            Number of countries for heatmap.
        annotate_heatmap : bool
            Whether to annotate heatmap cells.
        figsizes : dict, optional
            Figure sizes for each plot type.
        filename : str
            Base filename for saving.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import plotbib, utilsbib
        import matplotlib.pyplot as plt
        
        if figsizes is None:
            figsizes = {
                "pairs": (10, 6),
                "network": (12, 12),
                "heatmap": (12, 10),
            }
        
        # Get country collaboration data
        if not hasattr(self, "country_collaboration_matrix"):
            self.get_country_collaboration()
        
        collab_matrix = getattr(self, "country_collaboration_matrix", None)
        if collab_matrix is None or collab_matrix.empty:
            return
        
        # 1. Top pairs bar chart
        pairs_df = utilsbib.build_links_from_matrix(collab_matrix)
        pairs_df = pairs_df.head(top_n_pairs)
        
        fig1, ax1 = plt.subplots(figsize=figsizes["pairs"])
        ax1.barh(
            range(len(pairs_df)),
            pairs_df["Weight"],
            color="steelblue"
        )
        ax1.set_yticks(range(len(pairs_df)))
        ax1.set_yticklabels([f"{r['Source']} - {r['Target']}" for _, r in pairs_df.iterrows()])
        ax1.set_xlabel("Collaborations")
        ax1.set_title(f"Top {top_n_pairs} Country Collaboration Pairs")
        ax1.invert_yaxis()
        plt.tight_layout()
        
        if self.res_folder is not None:
            self._save_plot(f"{filename}_pairs")
        
        # 2. Network visualization
        import networkx as nx
        
        G = nx.Graph()
        for _, row in pairs_df.iterrows():
            if row["Weight"] >= connect_threshold:
                G.add_edge(row["Source"], row["Target"], weight=row["Weight"])
        
        if G.number_of_nodes() > 0:
            fig2, ax2 = plt.subplots(figsize=figsizes["network"])
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw edges with width based on weight
            edges = G.edges(data=True)
            weights = [e[2].get("weight", 1) for e in edges]
            max_weight = max(weights) if weights else 1
            widths = [3 * w / max_weight for w in weights]
            
            nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, ax=ax2)
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue", ax=ax2)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax2)
            
            ax2.set_title("Country Collaboration Network")
            ax2.axis("off")
            plt.tight_layout()
            
            if self.res_folder is not None:
                self._save_plot(f"{filename}_network")
        
        # 3. Heatmap
        top_countries = collab_matrix.sum().nlargest(top_n_countries).index.tolist()
        heatmap_data = collab_matrix.loc[top_countries, top_countries]
        
        fig3, ax3 = plt.subplots(figsize=figsizes["heatmap"])
        
        plotbib.plot_heatmap(
            heatmap_data,
            ax=ax3,
            annotate=annotate_heatmap,
            title="Country Collaboration Matrix",
            **kwargs,
        )
        
        if self.res_folder is not None:
            self._save_plot(f"{filename}_heatmap")
