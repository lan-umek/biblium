# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 22:21:17 2025

@author: Lan
"""

from bibstats import BiblioStats
from bibgroup import BiblioGroup
import plotbib, utilsbib
import numpy as np
import os
import logging
import pandas as pd
from typing import Optional, Dict, List



mapping = {
    "sources": {
        "stats_attr": "sources_stats_df",
        "getter": "get_sources_stats",
        "label": "Source"
    },
    "author keywords": {
        "stats_attr": "author_keywords_stats_df",
        "getter": "get_author_keywords_stats",
        "label": "Keyword",
        "coocurence network": "ak_cooccurrence_network",
        "get coocurrence": "get_author_keyword_cooccurrence"
    },
    "countries": {
        "stats_attr": "ca_countries_stats_df",
        "getter": "get_ca_countries_stats",
        "label": "Country"
    },
    "words from abstract": {
        "stats_attr": "ngrams_abstract_stats_df",
        "getter": "get_ngrams_abstract_stats",
        "label": "Word - Phrase",
        "coocurence network": "ngrams_abstract_cooccurrence_network",
        "get coocurrence": "get_ngrams_abstract_cooccurrence"
    },
    # add more item types here...
    "fields": {
        "stats_attr": "fields_stats_df",
        "getter": "get_fields_stats",
        "label": "Field"
    },
    "authors":
        {"coocurence network": "co_authorship_network",
         "get coocurrence": "get_coauthorship"},
    "references":
        {"coocurence network": "co_citation_network",
         "get coocurrence": "get_co_citations"},
    "all countries": {
        "stats_attr": "all_countries_stats_df",
        "getter": "get_all_countries_stats",
        "label": "Country",
        "coocurence network": "all_countries_cooccurrence_network",
        "get coocurrence": "get_country_collaboration_network"
    },
        
        
}

mapping["words abstract"] = mapping["words from abstract"]

co_mapping = {"ak": {"net_attr": "ak_cooccurrence_network", "getter": "get_author_keyword_cooccurrence"}}

counts_mapping = {
    "words from abstract": {
        "counts_attr": "words_abs_counts_df",
        "getter": "count_ngrams_abstract"},
    "sources": {
        "counts_attr": "sources_counts_df",
        "getter": "count_sources",
        "label": "Source"
    },
    "countries": {
        "counts_attr": "ca_countries_counts_df",
        "getter": "count_ca_countries",
        "label": "Country"
    }
    }


binary_indicators = {"affiliations": "binary_affiliations_df",
                     "all countries": "binary_all_countries_df",
                     "author keywords": "binary_author_keywords_df",
                     "authors" : "binary_authors_df",
                     "countries": "binary_ca_countries_df",
                     "document types": "binary_document_types_df",
                     "index keywords" : "binary_index_keywords_df",
                     "references": "binary_references_df",
                     "sources" : "binary_sources_df",
                     "abstract": "binary_words_abs_df",
                     "title" : "binary_words_tit_df"}

class BiblioPlot(BiblioStats):
    
    
    def plot_average_citations_per_year(self, filename_base="average citations per document", **kwargs):
        grouped = utilsbib.compute_average_citations_per_year(self.df)
        if filename_base is not None:
            filename_base = os.path.join(self.res_folder, "plots", filename_base)
            plotbib.plot_average_citations_per_year(grouped, **kwargs)

    
    def dist_plots(self, grouping_vars=["Source title", "Document Type"],
               numeric_vars=["Year", "Cited by"], max_groups=5,
               order_by_size=True, plot_type="box", **kwargs):
        """
        Generate box or violin plots for combinations of numeric and grouping variables.
    
        Parameters
        ----------
        grouping_vars : list of str
            Categorical variables to use for grouping.
        numeric_vars : list of str
            Numerical variables to be plotted.
        max_groups : int
            Maximum number of groups to show per plot.
        order_by_size : bool
            Whether to order groups by size.
        plot_type : str
            Either 'box' or 'violin'.
        **kwargs : dict
            Additional keyword arguments passed to the plot function.
        """
        plot_func = {
            "box": plotbib.plot_boxplot,
            "violin": plotbib.plot_violinplot
        }.get(plot_type)
    
        if plot_func is None:
            raise ValueError("plot_type must be 'box' or 'violin'")
    
        for group_var in grouping_vars:
            for numeric_var in numeric_vars:
                filename_base = os.path.join(
                    self.res_folder, "plots", f"{numeric_var} by {group_var}_{plot_type}"
                )
                plot_func(
                    self.df,
                    value_column=numeric_var,
                    group_by=group_var,
                    max_groups=max_groups,
                    order_by_size=order_by_size,
                    filename_base=filename_base,
                    dpi=self.dpi,
                    **kwargs
                )
                
    
    def plot_scientific_production(self, filename="scientific production", **kwargs):
        filename = os.path.join(self.res_folder, "plots", filename)
        if not hasattr(self, "production_df"):
            self.get_production()
        plotbib.plot_timeseries(self.production_df, filename=filename, dpi=self.dpi, **kwargs)

    def plot_reference_spectrogram(self, save_path="spectrogram", **kwargs):
        if not hasattr(self, "spectrogram_df"):
            self.compute_reference_spectrogram()
        save_path = os.path.join(self.res_folder, "plots", save_path)
        plotbib.plot_reference_spectrogram(self.spectrogram_df, save_path=save_path, **kwargs)

    def plot_ca_coutries_map(self, x="Number of documents", filename_prefix="country pefromance map", **kwargs):
        if not hasattr(self, "ca_country_counts_df"):
            self.count_ca_countries()
        if filename_prefix is not None:
            filename_prefix = os.path.join(self.res_folder, "plots", filename_prefix)

        plotbib.save_plotly_choropleth_map(self.ca_country_counts_df, x, filename_prefix=filename_prefix, 
                                           colormap=self.cmap, **kwargs)

    def plot_all_countries_map(self, x="Number of documents", filename_prefix="country collaboration map", **kwargs):
        
        plotbib.save_plotly_choropleth_map(self.all_countries_counts_df, x, filename_prefix=filename_prefix, 
                                           colormap=self.cmap, links_df=self.countries_links_df, **kwargs)
        
        

    # Bibliographic laws
    
    def lotka_law(self, author_col="Authors", separator="; ", filename_base="lotka law", **kwargs):
        self.lotka_df = utilsbib.compute_lotka_distribution(self.df, author_col=author_col, separator=separator)
        self.lotka_stats_df = utilsbib.evaluate_lotka_fit(self.lotka_df)
        if filename_base is not None:
            filename_base = os.path.join(self.res_folder, "plots", filename_base)
        plotbib.plot_lotka_distribution(self.lotka_df, filename_base=filename_base, dpi=self.dpi, **kwargs)
   
    def bradford_law(self, source_col="Source title", zone_count=3, lowercase=False, filename_base="bradford law", **kwargs):
        self.bradford_df = utilsbib.compute_bradford_distribution(self.df, source_col=source_col, zone_count=zone_count, lowercase=lowercase)
        self.bradford_stats_df = utilsbib.evaluate_bradford_fit(self.bradford_df, zone_count=zone_count)
        
        F1_KEYS = {"color", "show_grid"}
        F2_KEYS = {"colors", "annotate_core", "show_labels", "label_rotation", "alt_label_col", "max_label_length", "show_grid"}
        kw1 = {k: kwargs[k] for k in F1_KEYS if k in kwargs}
        kw2 = {k: kwargs[k] for k in F2_KEYS if k in kwargs}
        
        if filename_base is not None:
            f1 = os.path.join(self.res_folder, "plots", filename_base + " plot")
            f2 = os.path.join(self.res_folder, "plots", filename_base + " zones")
           
            plotbib.plot_bradford_distribution(self.bradford_df, title="Bradford's Law - Source Scattering", filename_base=f1, dpi=self.dpi, **kw1)
            plotbib.plot_bradford_zones(self.bradford_df, title="Bradford's Law - Zones", filename_base=f2, dpi=self.dpi, **kw2)

    def zipf_law(self, df_counts=None, items="words from abstract", **kwargs):
        
        if df_counts is None:
            d = counts_mapping[items]
            if not hasattr(self, d["counts_attr"]):
                getattr(self, d["getter"])(**kwargs)
            df_counts = getattr(self, d["counts_attr"])
        
        self.zipf_df = utilsbib.compute_zipf_distribution_from_counts(df_counts)
        self.zipf_stats = utilsbib.evaluate_zipf_fit(self.zipf_df)



    # Performance plots

    import logging

    def plot_top_items(self, items, x="Number of documents", y=None, kind="barh", **kwargs):
        """
        items : str
           Which stats to plot; must be a key in the internal `mapping`.
        x : str, default="Number of documents"
            Name of the column to use for the x‐axis values.
        y : str or None, optional
            Name of the column to use for the y‐axis values (required for scatter).
        kind : str, {"barh", "lollipop", "scatter"}, default="barh"
    
        Optional kwargs for scatter size_col, color_col, label_col    
    
        """
        
        if items not in mapping:
            raise ValueError(f"Unknown item type: {items!r}")
    
        cfg = mapping[items]
        stats_attr = cfg["stats_attr"]
        get_stats = cfg["getter"]
    
        if not hasattr(self, stats_attr):
            getattr(self, get_stats)()
        df = getattr(self, stats_attr)
    
        label = cfg["label"]
        base_fn = f"top_{items}_plot"
        filename = f"{base_fn} {kind}"
        plot_path = os.path.join(self.res_folder, "plots", filename)
    
        fn = {"barh": plotbib.plot_barh, 
              "lollipop": plotbib.plot_lollipop,
              "scatter": plotbib.plot_scatter}[kind]
        
        if kind in ["barh", "lollipop"]:
            fn(df, x, label, filename=plot_path, dpi=self.dpi, cmap=self.cmap,
               default_color=self.default_color, **kwargs)
        elif kind == "scatter":
            fn(df, x, y, colormap=self.cmap, filename=filename, dpi=self.dpi, show=True, **kwargs)
    
    def plot_top_items_multi(self, items, x="Number of documents", y=None, kind="barh", **kwargs):
        """
        Loop over multiple item-types and plot each one in turn.
    
        Parameters
        ----------
        items : Iterable[str]
            A sequence of stats keys; each must be defined in the mapping used by `plot_top_items`.
        x : str, default="Number of documents"
            Column name to use for the x-axis values in each plot.
        kind : str, {"barh", "lollipop"}, default="barh"
            Plot style to apply for each item.
        **kwargs
            Additional keyword arguments passed through to `plot_top_items`.
    
        Notes
        -----
        If plotting fails for any individual item, the error will be logged
        and the loop will continue with the next item.
        """
        for item in items:
            try:
                self.plot_top_items(item, x=x, y=y, kind=kind, **kwargs)
            except Exception as e:
                logging.error(f"Failed to plot top items for {item!r}: {e}", exc_info=True)
           
    
    def visualize_text(self, items, kind="cloud", x="Number of documents", filename="wordcloud", top_n=20, **kwargs):
        
        if items not in mapping:
            raise ValueError(f"Unknown item type: {items!r}")
    
        fn = {"cloud": plotbib.plot_wordcloud, "treemap": plotbib.plot_treemap}[kind]
    
        cfg = mapping[items]
        stats_attr = cfg["stats_attr"]
        get_stats = cfg["getter"]
    
        if not hasattr(self, stats_attr):
            getattr(self, get_stats)(top_n=top_n)
        df = getattr(self, stats_attr).head(top_n)
        
        filename = os.path.join(self.res_folder, "plots", filename + "_" + kind + "_" + items)
        if kind == "cloud":
            fn(df, filename=filename, dpi=self.dpi, colormap=self.cmap, **kwargs)
        else:
            fn(df, filename=filename, dpi=self.dpi, cmap=self.cmap, **kwargs)
                
    def visualize_text_multi(self, items, kind="cloud", x="Number of documents", filename="wordcloud", top_n=20, **kwargs):
        for item in items:
            try:
                self.visualize_text(item, kind=kind, x=x, filename=filename, top_n=top_n, **kwargs)
            except Exception as e:
                logging.error(f"Failed to plot top items for {item!r}: {e}", exc_info=True)
                
    def plot_thematic_map(self, G=None, items="ak", recompute=False, partition_attr="walktrap", max_dot_size=200, 
                          quadrant_labels=False, items_per_cluster=3,
                          cmap_name="viridis",  figsize=(8, 6),  max_clusters=None,
                          min_cluster_size=5, include_cluster_label=False,
                          color_df=None, color_col=None, save_plot_base="thematic map",
                          ax=None, item_sep="\n", **kwargs):
        
        
        
        if G is None:
            d = co_mapping[items]
            if not hasattr(self, d["net_attr"]) or recompute:
                getattr(self, d["getter"])(**kwargs)
            G = getattr(self, d["net_attr"])

        if save_plot_base is not None:
            save_plot_base = os.path.join(self.res_folder, "plots", partition_attr + "_" + save_plot_base)

        plotbib.plot_thematic_map(G, partition_attr, max_dot_size=max_dot_size, 
                              quadrant_labels=quadrant_labels, items_per_cluster=items_per_cluster,
                              cmap_name=self.cmap, figsize=figsize, max_clusters=max_clusters,
                              min_cluster_size=min_cluster_size, include_cluster_label=include_cluster_label,
                              color_df=color_df, color_col=color_col, save_plot_base=save_plot_base,
                              dpi=self.dpi, ax=ax, item_sep=item_sep)

    def plot_word_map(self, figsize: tuple = (10, 8), title: str = "Word Map",
                      filename_base: str = "factorial word map", marker_size: int = 50,
                      term_fontsize: int = 8, title_fontsize: int = 12,
                      axis_label_fontsize: int = 10, tick_label_fontsize: int = 8,
                      xlabel: str = 'Dim 1', ylabel: str = 'Dim 2',
                      show_legend: bool = True, **kwargs):
        if not hasattr(self, "conceptual_structure_d"):
            self.conceptual_structure_analysis(**kwargs)
            
        filename_base = os.path.join(self.res_folder, "plots", filename_base)    
        plotbib.plot_word_map(embeddings=self.conceptual_structure_d["term_embeddings"],
                              terms=self.conceptual_structure_d["terms"],
                              labels=self.conceptual_structure_d["term_labels"],
                          figsize=figsize, title=title, filename_base=filename_base,
                          dpi=self.dpi, cmap=self.cmap, marker_size=marker_size,
                          term_fontsize=term_fontsize, title_fontsize=title_fontsize,
                          axis_label_fontsize=axis_label_fontsize, 
                          tick_label_fontsize=tick_label_fontsize,
                          xlabel=xlabel, ylabel=ylabel,
                          show_legend=show_legend, **kwargs)
    
    def plot_topic_dendrogram(self,
        method: str = "ward",
        figsize: tuple = (10, 8),
        title: str = "Topic Dendrogram",
        filename_base: str = "topic dendrogram",
        xlabel: str = "Terms",
        ylabel: str = "Distance",
        title_fontsize: int = 12,
        axis_label_fontsize: int = 10,
        tick_label_fontsize: int = 8,
        leaf_label_fontsize: int = 8,
        **kwargs):
    
        if not hasattr(self, "conceptual_structure_d"):
            self.conceptual_structure_analysis(**kwargs)
        
        terms=self.conceptual_structure_d["terms"]
        print(terms)
        
        filename_base = os.path.join(self.res_folder, "plots", filename_base)
        plotbib.plot_topic_dendrogram(embeddings=self.conceptual_structure_d["term_embeddings"],
                              terms=self.conceptual_structure_d["terms"],
                              method=method, figsize=figsize,title=title,
                              filename_base=filename_base, dpi=self.dpi,
                              xlabel=xlabel, ylabel=ylabel, title_fontsize=title_fontsize,
                              axis_label_fontsize=axis_label_fontsize, 
                              tick_label_fontsize=tick_label_fontsize,
                              leaf_label_fontsize=leaf_label_fontsize)
        
    def plot_trend_topics(self, df: pd.DataFrame = None,
                          items: str = "Author Keywords", 
                          min_docs: int = 3,
                          regex_filter: str = None,
                          top_n_year: int = 3,
                          color_by: str = "Total citations",
                          item_col: str = "Item",
                          figsize: tuple = (10, 6),
                          filename: str = "trend topics",
                          title: str = None,
                          title_fontsize: int = 14,
                          label_fontsize: int = 12,
                          tick_fontsize: int = 10,
                          median_rounding: str = None,
                          override: bool = True,
                          **kwargs):
        
        d = mapping[items.lower()]
        if (df is None) or override:
            if not hasattr(self, d["stats_attr"]):

                getattr(self, d["getter"])(**kwargs)
            df = getattr(self, d["stats_attr"])
            print(df)
            item_col = d["label"]
        
        filename = os.path.join(self.res_folder, "plots", filename + "_" + items) 
        
        plotbib.plot_item_timelines(df, min_docs=min_docs, regex_filter=regex_filter, top_n_year=top_n_year,
            color_by=color_by, item_col=item_col, figsize=figsize,
            dpi=self.dpi, filename=filename, title=title,
            title_fontsize=title_fontsize, label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize, median_rounding=median_rounding,
            color_scheme=self.cmap)
        
        
    def plot_coocurence_network(self, items,
                                    partition_attr="walktrap",
                                    color_attr="Average year",
                                    size_attr="H-index",
                                    layout="spring",
                                    size_scale=300,
                                    default_node_color="blue",
                                    default_node_size=300,
                                    label_fontsize=8,
                                    edge_width=1.0,
                                    edge_alpha=1.0,
                                    log_scale=True,
                                    curved_edges=False,
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
                                    filename="coocurrence",
                                    **kwargs):
        """
        items je na primer 'author keywords'
        """
        
        d = mapping[items]
        if not hasattr(self, d["coocurence network"]):
            getattr(self, d["get coocurrence"])()
        G = getattr(self, d["coocurence network"])
        
        if filename is not None:
            filename = os.path.join(self.res_folder, "plots", filename + " of " + items)
        
        if color_attr is not None:
            filename0 = filename + " overlay" + color_attr
            plotbib.plot_network(G, color_attr="Average year",
                                            size_attr=size_attr,
                                            layout=layout,
                                            cmap_name_continuous=self.cmap,
                                            size_scale=size_scale,
                                            default_node_color=self.default_color,
                                            default_node_size=default_node_size,
                                            label_fontsize=label_fontsize,
                                            edge_width=edge_width,
                                            edge_alpha=edge_alpha,
                                            log_scale=log_scale,
                                            curved_edges=curved_edges,
                                            edge_curve_rad=edge_curve_rad,
                                            fix_max_size=fix_max_size,
                                            node_alpha=node_alpha,
                                            node_shape=node_shape,
                                            show_colorbar=show_colorbar,
                                            show_frame=show_frame,
                                            pos=pos,
                                            layout_kwargs=layout_kwargs,
                                            min_edge_width=min_edge_width,
                                            max_edge_width=max_edge_width,
                                            adjust_labels=adjust_labels,
                                            largest_component=largest_component,
                                            filename = filename0,
                                            **kwargs)
            
        
        filename1 = filename + " network " +  partition_attr           
        plotbib.plot_network(G, partition_attr=partition_attr,
                                        size_attr=size_attr,
                                        layout=layout,
                                        cmap_name_discrete=self.cmap_disc,
                                        size_scale=size_scale,
                                        default_node_color=self.default_color,
                                        default_node_size=default_node_size,
                                        label_fontsize=label_fontsize,
                                        edge_width=edge_width,
                                        edge_alpha=edge_alpha,
                                        log_scale=log_scale,
                                        curved_edges=curved_edges,
                                        edge_curve_rad=edge_curve_rad,
                                        fix_max_size=fix_max_size,
                                        node_alpha=node_alpha,
                                        node_shape=node_shape,
                                        show_colorbar=show_colorbar,
                                        show_frame=show_frame,
                                        pos=pos,
                                        layout_kwargs=layout_kwargs,
                                        min_edge_width=min_edge_width,
                                        max_edge_width=max_edge_width,
                                        adjust_labels=adjust_labels,
                                        largest_component=largest_component,
                                        filename = filename1,
                                        **kwargs)
        
    def plot_citation_network(
        self,
        size_dict: Optional[Dict[str, float]] = None,
        color_dict: Optional[Dict[str, float]] = None,
        label_dict: Optional[Dict[str, str]] = None,
        cmap: str = "viridis",
        arrow_size: int = 10,
        font_size: int = 8,
        node_size_factor: float = 100,
        sqrt_sizes: bool = False,
        edge_width: float = 0.5,
        layout: str = "kamada_kawai",
        main_path_color: str = "crimson",
        main_path_width: float = 2.5,
        main_path_style: str = "solid",
        filename: Optional[str] = None,
        plot_mode: str = "just path"  # options: "network", "with_path", "all", "just path"
    ) -> None:
        """
        Plot a citation network and optionally highlight or isolate the main path.
    
        Parameters:
            size_dict: Optional dictionary mapping node IDs to sizes.
            color_dict: Optional dictionary mapping node IDs to color values.
            label_dict: Optional dictionary mapping node IDs to custom labels.
            cmap: Matplotlib colormap name.
            arrow_size: Arrowhead size for directed edges.
            font_size: Font size for node labels.
            node_size_factor: Scaling multiplier for node sizes.
            sqrt_sizes: Whether to apply square root scaling to node sizes.
            edge_width: Width of edges.
            layout: Layout algorithm ("kamada_kawai", "spring", etc.).
            main_path_color: Color to highlight the main path.
            main_path_width: Width of edges along the main path.
            main_path_style: Line style for the main path.
            filename: File path to save the figure(s); applies to all plots.
            plot_mode: Which plots to generate:
                - "network": citation network only
                - "with_path": citation network with main path highlighted
                - "all": both versions and a separate plot of the main path
        """
        G = self.citation_network_documents
        path = self.citation_main_path
        cmap_obj = self.cmap
    
        filename = os.path.join(self.res_folder, "plots", "citation network")
        if plot_mode in {"network", "all"}:
            plotbib.plot_citation_network(
                G=G,
                size_dict=size_dict,
                color_dict=color_dict,
                label_dict=label_dict,
                cmap=cmap,
                arrow_size=arrow_size,
                font_size=font_size,
                node_size_factor=node_size_factor,
                sqrt_sizes=sqrt_sizes,
                edge_width=edge_width,
                layout=layout,
                highlight_main_path=False,
                filename=filename 
            )
    
        if plot_mode in {"with_path", "all"}:
            plotbib.plot_citation_network(
                G=G,
                size_dict=size_dict,
                color_dict=color_dict,
                label_dict=label_dict,
                cmap=cmap,
                arrow_size=arrow_size,
                font_size=font_size,
                node_size_factor=node_size_factor,
                sqrt_sizes=sqrt_sizes,
                edge_width=edge_width,
                layout=layout,
                highlight_main_path=True,
                main_path=path,
                main_path_color=main_path_color,
                main_path_width=main_path_width,
                main_path_style=main_path_style,
                filename=filename + "with main path"
            )
    
        if plot_mode in {"all", "just path"}:
            plotbib.plot_main_path(
                G=G,
                path=path,
                size_dict=size_dict,
                color_dict=color_dict,
                label_map=label_dict,
                cmap=cmap_obj,
                edge_color=main_path_color,
                edge_width=main_path_width,
                arrow_size=arrow_size,
                font_size=font_size,
                layout=layout,
                filename=filename + "main path"
            )


    def plot_country_collaboration(self, top_n_pairs=20, connect_threshold=1, top_n_countries=20, annotate_heatmap=True, figsizes={"pairs": (10,6), "network": (12,12), "heatmap": (12,10)}, filename="country collaboration", **kwargs):
        
        filename = os.path.join(self.res_folder, "plots", filename)
        plotbib.plot_top_country_pairs(self.country_collab_matrix, top_n=top_n_pairs, figsize=figsizes["pairs"], filename_base=filename + "top pairs")
        plotbib.plot_country_collab_network(self.country_collab_matrix, threshold=connect_threshold, figsize=figsizes["network"], layout_func="spring", filename_base=filename + "network")
        plotbib.plot_country_collab_heatmap(self.country_collab_matrix, top_n=top_n_countries, figsize=figsizes["heatmap"], cmap=self.cmap, annotate=annotate_heatmap, filename_base=filename + "heatmap")
        plotbib.plot_country_collab_heatmap(self.country_collab_matrix_norm, top_n=top_n_countries, figsize=figsizes["heatmap"], cmap=self.cmap, annotate=annotate_heatmap, filename_base=filename + "heatmap normalized")
        #self.plot_coocurence_network("all countries", **kwargs)
                    
    def plot_historiograph(self, figsize=(12, 8), size_attr=None, min_indegree=None,
                              min_citations=100, min_year=None, max_year=None, filename="historiograph", **kwargs):
        if not hasattr(self, "historiograph"):
            self.build_historiograph(**kwargs)
        G = self.historiograph
        pos = plotbib.layout_historiograph(G)
        
        filename = os.path.join(self.res_folder, "plots", filename)
        plotbib.plot_historiograph(G, pos, figsize=figsize, size_attr=size_attr,
                                   min_indegree=min_indegree, min_citations=min_citations,
                                   min_year=min_year, max_year=max_year, save_as=filename,
                                   dpi=self.dpi)
        
    # plotting of relations

    def plot_correspondence_analysis(self, concept1, concept2, 
                                     figsize=(8, 6), 
                                     annotate=True, 
                                     alpha=0.8, 
                                     size_scale=300, 
                                     use_size: bool = True,  
                                     filename_base: str = "correspondence analysis", 
                                     title: str = "Correspondence Analysis with Frequencies"):
        try:
            R = self.relations[concept1][concept2]
        except:
            print("Relation no yet computed. See documentation how to compute it")
            return None
        
        filename_base = os.path.join(self.res_folder, "plots", filename_base + "_" + concept1 + "_" + concept2)
        plotbib.plot_correspondence_analysis(R.ca_row_coords,
                                             R.ca_col_coords,
                                             R.ca_explained_inertia,
                                             R.rm,
                                             row_label_name=R.concept1,
                                             col_label_name=R.concept2,
                                             figsize=figsize,
                                             annotate=annotate,
                                             alpha=alpha,
                                             use_size=use_size,
                                             title=title,
                                             filename_base=filename_base,
                                             dpi=self.dpi)    
        
    def plot_residual_heatmap(self, concept1, concept2, 
                              kind="chi2",
                              center: float = 0.0,
                              cmap: str = "coolwarm",
                              figsize=(10, 8),
                              annotate: bool = False,
                              square: bool = True,
                              filename_base: str = "heatmap of residuals",
                              title: str = "Standardized Pearson Residuals",
                              **kwargs):
        try:
            R = self.relations[concept1][concept2]
        except:
            print("Relation no yet computed. See documentation how to compute it")
            return None
        filename_base = os.path.join(self.res_folder, "plots", filename_base + "_" + concept1 + "_" + concept2)
        if kind == "chi2":
            df=R.chi2_residuals_df
        elif kind == "log-ratio":
            df=R.log_ratio_df
            if title == "Standardized Pearson Residuals":
                title = "Log Ratio Heatmap (Observed / Expected)"
        
        plotbib.plot_residual_heatmap(df,
                                      center=center,
                                      cmap=cmap, # do not change to self.cmap
                                      figsize=figsize,
                                      annotate=annotate,
                                      square=square,
                                      filename_base=filename_base,
                                      title=title,
                                      dpi=self.dpi,
                                      row_label=R.concept1,
                                      col_label=R.concept2)
    
    def plot_bipartite_network(self, concept1, concept2, 
                               node_size_scale: float = 200,
                               edge_alpha: float = 0.3,
                               same_size: bool = False,
                               weight_threshold: float = 0,
                               show_edge_weights: bool = False,
                               edge_width_scale: float = 1.0,
                               figsize=(10, 8),
                               title: str = None,
                               filename_base: str = "bipartite netwkor",
                               dpi: int = 600,
                               row_label_name: str = "Rows",
                               col_label_name: str = "Columns"):
        try:
            R = self.relations[concept1][concept2]
        except:
            print("Relation no yet computed. See documentation how to compute it")
            return None
        
        filename_base = os.path.join(self.res_folder, "plots", filename_base + "_" + concept1 + "_" + concept2)
        plotbib.plot_bipartite_network(R.bipartite_graph,
                                       R.bipartite_row_projection.nodes, 
                                       R.bipartite_column_projection.nodes,
                                       node_size_scale=node_size_scale,
                                       edge_alpha=edge_alpha,
                                       same_size=same_size,
                                       show_edge_weights=show_edge_weights,
                                       edge_width_scale=edge_width_scale,
                                       figsize=figsize,
                                       title=title,
                                       filename_base=filename_base,
                                       dpi=self.dpi,
                                       row_label_name=concept1,
                                       col_label_name=concept2)


    def plot_top_n_pairs(self, concept1, concept2,
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
                         filename_base: str = "top n pairs",
                         dpi: int = 600,
                         show_colorbar: bool = True,
                         show_guides: bool = False,
                         sign: str = "positive"  # "positive", "negative", "all"
                         ):
        try:
            R = self.relations[concept1][concept2]
        except:
            print("Relation no yet computed. See documentation how to compute it")
            return None
        
        filename_base = os.path.join(self.res_folder, "plots", filename_base + concept1 + "_" + concept2)
        plotbib.plot_top_n_pairs(R.chi2_sorted_residuals,
                                 metric_column=metric_column,
                                 top_n=top_n,
                                 size_column=size_column,
                                 size_scale=size_scale,
                                 color_map=color_map,
                                 center_color=center_color,
                                 figsize=figsize,
                                 title=title,
                                 x_label=concept1,
                                 y_label=concept2,
                                 filename_base=filename_base,
                                 dpi=self.dpi,
                                 show_colorbar=show_colorbar,
                                 show_guides=show_guides,
                                 sign=sign)
    
    
class BiblioGroupPlot(BiblioGroup):
    
    def plot_group_overlapping(self, plot_types=["venn", "upset", "heatmap","dendrogram"],
                               title=None, filename="overapping", show=True,
                               include_totals_venn=True,
                               alpha_venn=0.5,
                               methods_heatmap=["jaccard"],
                               color_ticks_heatmap=False,
                               save_csv_heatmap=True,
                               threshold_chord=0.0,
                               method_dendrogram="average",
                               metric_dendrogram="euclidean",
                               **kwargs):

        filename = os.path.join(self.res_folder, "plots", filename + "_")

        for plot_type in plot_types:
                    
            if plot_type == "venn":
                plotbib.plot_group_venn(self.group_matrix, title=title, filename=filename+"venn", dpi=self.dpi, include_totals=include_totals_venn, show=show, save_results=True, group_color=self.group_colors, alpha=alpha_venn, **kwargs)
            if plot_type == "upset":
                plotbib.plot_group_upset(self.group_matrix, title=title, filename=filename+"upset", dpi=self.dpi, show=show, save_results=True, group_color=self.group_colors, **kwargs)
            if plot_type == "heatmap":
                plotbib.plot_group_heatmap(self.group_matrix, methods=methods_heatmap, title=title, filename=filename+"heatmap", dpi=self.dpi, group_color=self.group_colors, color_ticks=color_ticks_heatmap, show=show, save_results=True, save_csv=save_csv_heatmap, **kwargs)
            if plot_type == "chord": # to be fixed
                plotbib.plot_group_chord(self.group_matrix, threshold=threshold_chord, group_color=self.group_colors, title=title, filename=filename+"chord", dpi=self.dpi, show=show)
            if plot_type == "dendrogram":
                plotbib.plot_group_dendrogram(self.group_matrix, method=method_dendrogram, metric=metric_dendrogram, title=title, filename=filename+"dendrogram", dpi=self.dpi, show=show)


    def plot_top_items(self, items=["sources", "countries"], top_n=5, value_column_pattern="Number of documents",
                       title=None, filename="top", show_values=True,
                       reverse_order=False, show=True):
        
        filename = os.path.join(self.res_folder, "plots", filename + "_")
        
        for item in items:
            d = counts_mapping[item]
            if not hasattr(self, d["counts_attr"]):
                getattr(self, d["getter"])()
            df = getattr(self, d["counts_attr"])
            
            plotbib.plot_top_items_by_group(df, top_n=top_n,
                                 value_column_pattern=value_column_pattern,
                                 title=title,
                                 filename=filename+item,
                                 dpi=self.dpi,
                                 group_color=self.group_colors,
                                 show_values=show_values,
                                 reverse_order=reverse_order,
                                 show=show)
            
    def plot_c_vars_across_groups(self, numerical_cols=["Year", "Cited by", "Sentiment Score", "Interdisciplinarity"],
                                  plot_types=["histogram", "violin plot", "boxplot"], file_name="group comparison",
                                  group_colors=True, bins=30, alpha=0.7, show_grid=False, **kwargs):
        numerical_cols = [c for c in numerical_cols if c in self.df.columns]
        group_colors = self.group_colors if group_colors else  {}
        
        if file_name is not None:
            file_name = os.path.join(self.res_folder, "plots", file_name)
            save = True
        else:
            save=False
        
        for plot_type in plot_types:
            if plot_type == "histogram":
                plotbib.plot_group_distributions_aligned(self.df, numerical_cols, self.group_matrix,
                                                         bins=bins, alpha=alpha, 
                                                     save=save, filename_prefix=file_name+"_histogram", dpi=self.dpi, 
                                                     show_grid=show_grid, group_colors=group_colors)
            if plot_type in ["boxplot", "box"]:
                for value_column in numerical_cols:
                    plotbib.plot_boxplot(self.df, value_column, group_matrix=self.group_matrix, filename_base=file_name+"_boxplot", dpi=self.dpi, group_colors=group_colors, **kwargs)
            if plot_type in ["violin plot", "violin"]:
                for value_column in numerical_cols:
                    plotbib.plot_violinplot(self.df, value_column, group_matrix=self.group_matrix, filename_base=file_name+"_violin", dpi=self.dpi, group_colors=group_colors, **kwargs)
    

    def plot_stacked_production_by_group(self, filename_base="production by group",
                                         figsize=(10,6), cut_year=None, year_span=None,
                                         font_size=12, xlabel="Year", ylabel="Number of documents",
                                         citation_mode="group",
                                         citation_label="Cumulative Citations", legend_title="Group",
                                         **kwargs):
        
        filename_base = os.path.join(self.res_folder, "plots", filename_base)
        if not hasattr(self, "production_df"):
            self.get_scientific_production(**kwargs)
        
        
        plotbib.plot_stacked_production_by_group(self.production_df,
           group_colors=self.group_colors,
           filename_base=filename_base,
           figsize=figsize,
           cut_year=cut_year,
           year_span=year_span,
           citation_mode=citation_mode,
           font_size=font_size,
           xlabel=xlabel,
           ylabel=ylabel,
           citation_label=citation_label,
           legend_title=legend_title)




        
