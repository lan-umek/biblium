# -*- coding: utf-8 -*-
"""
Thematic plotting functions - word clouds, topic maps, dendrograms.

This module contains methods for:
- Word clouds
- Thematic maps
- Topic dendrograms
- Topic visualizations
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np


class ThematicPlotsMixin:
    """Mixin class providing thematic plotting methods."""
    
    def visualize_text(
        self,
        items: str = "keywords",
        kind: str = "cloud",
        x: str = "Number of documents",
        filename: str = "wordcloud",
        top_n: int = 100,
        figsize: Tuple[int, int] = (12, 8),
        **kwargs: Any,
    ) -> None:
        """
        Visualize text items as word cloud or treemap.

        Parameters
        ----------
        items : str
            Type of items: "keywords", "abstract", "title", etc.
        kind : {"cloud", "treemap"}
            Visualization type.
        x : str
            Metric column for sizing.
        filename : str
            Filename for saving.
        top_n : int
            Number of top items.
        figsize : tuple
            Figure size.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import plotbib
        import matplotlib.pyplot as plt
        
        # Get data
        items_map = {
            "keywords": "author_keywords_counts_df",
            "author_keywords": "author_keywords_counts_df",
            "index_keywords": "index_keywords_counts_df",
            "abstract": "words_abs_counts_df",
            "title": "words_tit_counts_df",
        }
        
        attr_name = items_map.get(items)
        if attr_name is None:
            raise ValueError(f"Unknown items type: {items}")
        
        # Ensure data exists
        if not hasattr(self, attr_name):
            count_method = f"count_{items}" if items != "abstract" else "count_ngrams_abstract"
            if items == "title":
                count_method = "count_ngrams_title"
            if hasattr(self, count_method):
                getattr(self, count_method)()
        
        df = getattr(self, attr_name, None)
        if df is None or df.empty:
            return
        
        df = df.head(top_n)
        
        if kind == "cloud":
            # Word cloud
            try:
                from wordcloud import WordCloud
            except ImportError:
                print("wordcloud package not installed")
                return
            
            # Build frequency dict
            item_col = df.columns[0]
            freq_dict = dict(zip(df[item_col], df[x]))
            
            wc = WordCloud(
                width=figsize[0] * 100,
                height=figsize[1] * 100,
                background_color="white",
                **kwargs,
            ).generate_from_frequencies(freq_dict)
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{items.replace('_', ' ').title()} Word Cloud")
            
        elif kind == "treemap":
            # Treemap
            try:
                import squarify
            except ImportError:
                print("squarify package not installed")
                return
            
            item_col = df.columns[0]
            sizes = df[x].values
            labels = df[item_col].values
            
            fig, ax = plt.subplots(figsize=figsize)
            squarify.plot(
                sizes=sizes,
                label=labels,
                alpha=0.8,
                ax=ax,
            )
            ax.axis("off")
            ax.set_title(f"{items.replace('_', ' ').title()} Treemap")
        
        plt.tight_layout()
        
        if self.res_folder:
            self._save_plot(filename)

    def plot_thematic_map(
        self,
        items: str = "keywords",
        top_n: int = 100,
        n_clusters: int = 5,
        partition_method: str = "louvain",
        figsize: Tuple[int, int] = (12, 10),
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot thematic map based on co-occurrence clustering.

        Parameters
        ----------
        items : str
            Type of items to analyze.
        top_n : int
            Number of top items.
        n_clusters : int
            Number of clusters (if applicable).
        partition_method : str
            Clustering method.
        figsize : tuple
            Figure size.
        filename : str, optional
            Filename for saving.
        **kwargs :
            Additional plot arguments.
        """
        from biblium import utilsbib
        from biblium.utilsbib_modules.network import (
            build_cooccurrence_matrix, 
            matrix_to_network,
            add_partitions,
            compute_centralities,
        )
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Get item column
        items_map = {
            "keywords": (["Processed Author Keywords", "Author Keywords"], "author_keywords_counts_df"),
            "author_keywords": (["Processed Author Keywords", "Author Keywords"], "author_keywords_counts_df"),
        }
        
        if items not in items_map:
            raise ValueError(f"Unknown items type: {items}")
        
        col_candidates, counts_attr = items_map[items]
        
        col = None
        for c in col_candidates:
            if c in self.df.columns:
                col = c
                break
        
        if col is None:
            return
        
        # Get top items
        if not hasattr(self, counts_attr):
            self.count_author_keywords()
        
        counts_df = getattr(self, counts_attr)
        item_col = counts_df.columns[0]
        top_items = counts_df[item_col].head(top_n).tolist()
        
        # Build co-occurrence network
        cooc_matrix = build_cooccurrence_matrix(
            self.df, col,
            sep=getattr(self, "default_separator", "; "),
        )
        
        common_items = [i for i in top_items if i in cooc_matrix.index]
        cooc_matrix = cooc_matrix.loc[common_items, common_items]
        
        G = matrix_to_network(cooc_matrix, min_weight=1)
        
        if G.number_of_nodes() < 2:
            return
        
        # Add partitions
        G = add_partitions(G, methods=[partition_method])
        
        # Compute centralities
        centralities = compute_centralities(G)
        
        # Calculate density and centrality per cluster
        partition_attr = f"partition_{partition_method}"
        clusters = {}
        for node in G.nodes():
            cluster = G.nodes[node].get(partition_attr, 0)
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(node)
        
        # Calculate cluster metrics
        cluster_data = []
        for cluster_id, nodes in clusters.items():
            subgraph = G.subgraph(nodes)
            
            # Density (internal cohesion)
            density = nx.density(subgraph) if len(nodes) > 1 else 0
            
            # Centrality (external connections)
            external_edges = sum(
                1 for n in nodes 
                for neighbor in G.neighbors(n) 
                if neighbor not in nodes
            )
            centrality = external_edges / len(nodes) if nodes else 0
            
            # Size
            size = len(nodes)
            
            cluster_data.append({
                "cluster": cluster_id,
                "density": density,
                "centrality": centrality,
                "size": size,
                "nodes": nodes,
            })
        
        cluster_df = pd.DataFrame(cluster_data)
        
        # Plot strategic diagram
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_df)))
        
        for idx, row in cluster_df.iterrows():
            ax.scatter(
                row["centrality"],
                row["density"],
                s=row["size"] * 100,
                c=[colors[idx]],
                alpha=0.7,
                label=f"Cluster {row['cluster']} ({row['size']} items)",
            )
            
            # Add top keyword labels
            top_keywords = row["nodes"][:3]
            label = "\n".join(top_keywords)
            ax.annotate(
                label,
                (row["centrality"], row["density"]),
                fontsize=8,
                ha="center",
            )
        
        # Add quadrant lines
        ax.axhline(cluster_df["density"].median(), color="gray", linestyle="--", alpha=0.5)
        ax.axvline(cluster_df["centrality"].median(), color="gray", linestyle="--", alpha=0.5)
        
        ax.set_xlabel("Centrality (External Links)")
        ax.set_ylabel("Density (Internal Cohesion)")
        ax.set_title("Thematic Map - Strategic Diagram")
        ax.legend(loc="best", fontsize=8)
        
        plt.tight_layout()
        
        if filename and self.res_folder:
            self._save_plot(filename)
        
        self.thematic_clusters = cluster_df

    def plot_topic_dendrogram(
        self,
        items: str = "keywords",
        top_n: int = 50,
        method: str = "ward",
        figsize: Tuple[int, int] = (12, 10),
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot hierarchical clustering dendrogram for topics.

        Parameters
        ----------
        items : str
            Type of items to cluster.
        top_n : int
            Number of top items.
        method : str
            Linkage method for clustering.
        figsize : tuple
            Figure size.
        filename : str, optional
            Filename for saving.
        **kwargs :
            Additional plot arguments.
        """
        from biblium.utilsbib_modules.network import build_cooccurrence_matrix
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        import matplotlib.pyplot as plt
        
        # Get item column
        items_map = {
            "keywords": ["Processed Author Keywords", "Author Keywords"],
            "author_keywords": ["Processed Author Keywords", "Author Keywords"],
        }
        
        col_candidates = items_map.get(items)
        if not col_candidates:
            return
        
        col = None
        for c in col_candidates:
            if c in self.df.columns:
                col = c
                break
        
        if col is None:
            return
        
        # Get top items
        counts_attr = f"{items}_counts_df"
        if not hasattr(self, counts_attr):
            getattr(self, f"count_{items}")()
        
        counts_df = getattr(self, counts_attr)
        item_col = counts_df.columns[0]
        top_items = counts_df[item_col].head(top_n).tolist()
        
        # Build co-occurrence matrix
        cooc_matrix = build_cooccurrence_matrix(
            self.df, col,
            sep=getattr(self, "default_separator", "; "),
        )
        
        common_items = [i for i in top_items if i in cooc_matrix.index]
        cooc_matrix = cooc_matrix.loc[common_items, common_items]
        
        if len(common_items) < 2:
            return
        
        # Convert to distance matrix
        # Normalize by max and invert
        max_val = cooc_matrix.values.max()
        dist_matrix = 1 - (cooc_matrix.values / max_val)
        np.fill_diagonal(dist_matrix, 0)
        
        # Hierarchical clustering
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method=method)
        
        # Plot dendrogram
        fig, ax = plt.subplots(figsize=figsize)
        
        dendrogram(
            Z,
            labels=common_items,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax,
        )
        
        ax.set_title(f"{items.replace('_', ' ').title()} Dendrogram")
        ax.set_xlabel("Items")
        ax.set_ylabel("Distance")
        
        plt.tight_layout()
        
        if filename and self.res_folder:
            self._save_plot(filename)
