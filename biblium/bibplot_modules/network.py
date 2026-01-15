# -*- coding: utf-8 -*-
"""
Network plotting functions - co-occurrence, citation, collaboration networks.

This module contains methods for:
- Co-occurrence networks (keywords, concepts)
- Citation networks
- Co-authorship networks
- Historiograph visualization
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np


class NetworkPlotsMixin:
    """Mixin class providing network plotting methods."""
    
    def plot_cooccurrence(
        self,
        items: str = "keywords",
        top_n: int = 50,
        min_edge_weight: int = 2,
        layout: str = "spring",
        node_size_col: str = "Number of documents",
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 12),
        **kwargs: Any,
    ) -> None:
        """
        Plot co-occurrence network for items.

        Parameters
        ----------
        items : str
            Type of items: "keywords", "authors", "sources", etc.
        top_n : int
            Number of top items to include.
        min_edge_weight : int
            Minimum co-occurrence count for edges.
        layout : str
            Network layout algorithm.
        node_size_col : str
            Column for node sizing.
        filename : str, optional
            Filename for saving.
        figsize : tuple
            Figure size.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import plotbib, utilsbib
        from biblium.utilsbib_modules.network import build_cooccurrence_matrix, matrix_to_network
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Map items to columns
        items_map = {
            "keywords": (["Processed Author Keywords", "Author Keywords"], "Keyword"),
            "author_keywords": (["Processed Author Keywords", "Author Keywords"], "Keyword"),
            "index_keywords": (["Processed Index Keywords", "Index Keywords"], "Keyword"),
            "authors": (["Authors", "Author full names"], "Author"),
            "affiliations": (["Affiliations"], "Affiliation"),
        }
        
        if items not in items_map:
            raise ValueError(f"Unknown items type: {items}")
        
        col_candidates, label = items_map[items]
        
        # Find column
        col = None
        for c in col_candidates:
            if c in self.df.columns:
                col = c
                break
        
        if col is None:
            print(f"No suitable column found for {items}")
            return
        
        # Get top items
        count_method = f"count_{items}"
        counts_attr = f"{items}_counts_df"
        
        if hasattr(self, count_method) and not hasattr(self, counts_attr):
            getattr(self, count_method)()
        
        counts_df = getattr(self, counts_attr, None)
        if counts_df is None:
            return
        
        # Get top item names
        item_col = counts_df.columns[0]
        top_items = counts_df[item_col].head(top_n).tolist()
        
        # Build co-occurrence matrix
        cooc_matrix = build_cooccurrence_matrix(
            self.df,
            col,
            sep=getattr(self, "default_separator", "; "),
            min_count=min_edge_weight,
        )
        
        # Filter to top items
        common_items = [i for i in top_items if i in cooc_matrix.index]
        if len(common_items) < 2:
            print("Not enough items for network")
            return
        
        cooc_matrix = cooc_matrix.loc[common_items, common_items]
        
        # Build network
        G = matrix_to_network(cooc_matrix, min_weight=min_edge_weight)
        
        if G.number_of_nodes() == 0:
            print("Empty network")
            return
        
        # Get node sizes from counts
        size_map = dict(zip(counts_df[item_col], counts_df.get(node_size_col, [100] * len(counts_df))))
        node_sizes = [size_map.get(n, 100) * 10 for n in G.nodes()]
        
        # Layout
        layout_funcs = {
            "spring": nx.spring_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "circular": nx.circular_layout,
            "shell": nx.shell_layout,
        }
        layout_func = layout_funcs.get(layout, nx.spring_layout)
        pos = layout_func(G)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw edges
        edges = G.edges(data=True)
        weights = [e[2].get("weight", 1) for e in edges]
        max_weight = max(weights) if weights else 1
        widths = [2 * w / max_weight for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4, ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color="lightblue",
            alpha=0.8,
            ax=ax,
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title(f"{items.replace('_', ' ').title()} Co-occurrence Network")
        ax.axis("off")
        plt.tight_layout()
        
        if filename and self.res_folder:
            self._save_plot(filename)
        
        # Store network
        self.cooccurrence_network = G

    def plot_keyword_cooccurrence_network(
        self,
        which: str = "author",
        top_n: int = 50,
        min_edge_weight: int = 2,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot keyword co-occurrence network.

        Parameters
        ----------
        which : {"author", "index", "both"}
            Which keywords to use.
        top_n : int
            Number of top keywords.
        min_edge_weight : int
            Minimum edge weight.
        filename : str, optional
            Filename for saving.
        **kwargs :
            Additional plot arguments.
        """
        items = f"{which}_keywords" if which != "both" else "keywords"
        self.plot_cooccurrence(
            items=items,
            top_n=top_n,
            min_edge_weight=min_edge_weight,
            filename=filename or f"{which}_keyword_cooccurrence",
            **kwargs,
        )

    def plot_co_authorship_network(
        self,
        top_n: int = 50,
        min_collaborations: int = 2,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot co-authorship network.

        Parameters
        ----------
        top_n : int
            Number of top authors.
        min_collaborations : int
            Minimum collaborations for edges.
        filename : str, optional
            Filename for saving.
        **kwargs :
            Additional plot arguments.
        """
        self.plot_cooccurrence(
            items="authors",
            top_n=top_n,
            min_edge_weight=min_collaborations,
            filename=filename or "co_authorship_network",
            **kwargs,
        )

    def plot_citation_network(
        self,
        top_n: int = 50,
        min_citations: int = 1,
        layout: str = "spring",
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 14),
        **kwargs: Any,
    ) -> None:
        """
        Plot citation network.

        Parameters
        ----------
        top_n : int
            Number of top-cited documents.
        min_citations : int
            Minimum citation count for inclusion.
        layout : str
            Network layout.
        filename : str, optional
            Filename for saving.
        figsize : tuple
            Figure size.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import utilsbib
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Build citation network
        if self.db in ["open alex", "openalex", "oa"]:
            G = utilsbib.build_openalex_citation_network(
                self.df,
                min_citations=min_citations,
            )
        else:
            G = utilsbib.build_citation_network(
                self.df,
                min_citations=min_citations,
            )
        
        if G is None or G.number_of_nodes() == 0:
            print("Could not build citation network")
            return
        
        # Keep only top nodes by in-degree
        in_degrees = dict(G.in_degree())
        top_nodes = sorted(in_degrees, key=in_degrees.get, reverse=True)[:top_n]
        G = G.subgraph(top_nodes).copy()
        
        # Layout
        layout_funcs = {
            "spring": nx.spring_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "circular": nx.circular_layout,
        }
        layout_func = layout_funcs.get(layout, nx.spring_layout)
        pos = layout_func(G)
        
        # Node sizes by in-degree
        in_deg = dict(G.in_degree())
        node_sizes = [100 + in_deg.get(n, 0) * 50 for n in G.nodes()]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, ax=ax)
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color="lightcoral",
            alpha=0.8,
            ax=ax,
        )
        
        # Labels for top nodes only
        top_labels = {n: n[:30] + "..." if len(str(n)) > 30 else str(n) 
                     for n in list(G.nodes())[:20]}
        nx.draw_networkx_labels(G, pos, labels=top_labels, font_size=7, ax=ax)
        
        ax.set_title("Citation Network")
        ax.axis("off")
        plt.tight_layout()
        
        if filename and self.res_folder:
            self._save_plot(filename)
        
        self.citation_network = G

    def plot_historiograph(
        self,
        top_n: int = 30,
        min_citations: int = 5,
        figsize: Tuple[int, int] = (14, 10),
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot historiograph showing citation flow over time.

        Parameters
        ----------
        top_n : int
            Number of top documents.
        min_citations : int
            Minimum citations for inclusion.
        figsize : tuple
            Figure size.
        filename : str, optional
            Filename for saving.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import utilsbib
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Build historiograph
        G = utilsbib.build_historiograph(
            self.df,
            min_citations=min_citations,
            top_n=top_n,
        )
        
        if G is None or G.number_of_nodes() == 0:
            print("Could not build historiograph")
            return
        
        # Position nodes by year (x) and citation count (y)
        pos = {}
        for node in G.nodes():
            year = G.nodes[node].get("year", 2000)
            citations = G.nodes[node].get("citations", 0)
            pos[node] = (year, citations)
        
        # Normalize positions
        years = [p[0] for p in pos.values()]
        min_year, max_year = min(years), max(years)
        
        for node in pos:
            x, y = pos[node]
            pos[node] = ((x - min_year) / max(1, max_year - min_year), y)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        nx.draw_networkx_edges(G, pos, alpha=0.4, arrows=True, ax=ax)
        
        node_sizes = [100 + G.nodes[n].get("citations", 0) * 20 for n in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color="lightgreen",
            alpha=0.8,
            ax=ax,
        )
        
        ax.set_xlabel("Time (normalized)")
        ax.set_ylabel("Citations")
        ax.set_title("Historiograph - Citation Flow Over Time")
        
        plt.tight_layout()
        
        if filename and self.res_folder:
            self._save_plot(filename)
        
        self.historiograph = G
