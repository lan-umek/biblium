# -*- coding: utf-8 -*-
"""Paneli 2.19: diakrone mreže, časovnica klastrov, structural variation,
mehki klastri dokumentov, ekstrakcija izrazov."""
from __future__ import annotations

import os
import tempfile

import pandas as pd

from biblium.gui.panels.simple import SimpleAnalysisPanel, Opt

KW_COLS = ["Author Keywords", "Index Keywords", "Extracted Terms"]


class DiachronicNetworkPanel(SimpleAnalysisPanel):
    title = "Diachronic Networks"
    icon = "🎞️"
    description = "Co-word networks per period with a fixed node set and layout; static grid and animated GIF (bibliometrix 5.1 analogue)."
    OPTIONS = [
        Opt("field", "Keyword column", "columns", "Author Keywords", values=KW_COLS),
        Opt("n_slices", "Periods", "spin", 3, from_=2, to=8),
        Opt("cumulative", "Cumulative periods", "check", True),
        Opt("top_n", "Top N keywords", "spin", 40, from_=10, to=300),
        Opt("min_count", "Min co-occurrence", "spin", 2, from_=1, to=50),
        Opt("animate", "Write animated GIF", "check", True),
        Opt("seed", "Seed", "spin", 42, from_=0, to=99999),
    ]
    INFO_TEXT = """
DIACHRONIC NETWORKS
───────────────────
The node set is fixed to the top-N keywords of the whole corpus and the layout
is computed once on the union graph, so the eye can follow how communities
form and merge across periods. The GIF is written next to the results
(res_folder/plots) or in a temporary folder when no results folder is set.
"""

    def compute(self, bib, o):
        from biblium import network_dynamics as ND
        graphs, labels = bib.diachronic_networks(field=o["field"], n_slices=o["n_slices"], cumulative=o["cumulative"],
                                                 top_n=o["top_n"], min_count=o["min_count"], seed=o["seed"],
                                                 animate=False)
        if not graphs:
            raise ValueError("No period networks could be built (too few keywords/co-occurrences).")
        rows = [{"Period": l, "Nodes": g.number_of_nodes(), "Edges": g.number_of_edges(),
                 "Communities": len({d.get("partition_louvain") for _, d in g.nodes(data=True)})}
                for g, l in zip(graphs, labels)]
        msg = None
        if o["animate"]:
            folder = os.path.join(bib.res_folder, "plots") if getattr(bib, "res_folder", None) else tempfile.gettempdir()
            os.makedirs(folder, exist_ok=True)
            path = ND.animate_diachronic_network(graphs, labels, os.path.join(folder, "diachronic network.gif"),
                                                 seed=o["seed"])
            bib.diachronic_animation_path = path
            msg = f"Animated GIF written to: {path}"

        def grid(fig):
            n = len(graphs)
            ncols = min(3, n)
            nrows = (n + ncols - 1) // ncols
            import networkx as nx
            import matplotlib as mpl
            union = nx.compose_all(graphs)
            pos = nx.spring_layout(union, seed=o["seed"], weight="weight")
            cmap = mpl.colormaps["tab10"]
            for i, (g, l) in enumerate(zip(graphs, labels)):
                ax = fig.add_subplot(nrows, ncols, i + 1)
                parts = [d.get("partition_louvain", 0) for _, d in g.nodes(data=True)]
                sizes = [30 + 6 * d.get("freq", 1) for _, d in g.nodes(data=True)]
                nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, width=0.6)
                nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color=[cmap(p % 10) for p in parts], alpha=0.9)
                nx.draw_networkx_labels(g, pos, ax=ax, font_size=5)
                ax.set_title(l, fontsize=9); ax.axis("off")
        return {"summary": [("Periods", len(graphs)), ("Nodes", graphs[-1].number_of_nodes())], "message": msg,
                "tables": {"Periods": pd.DataFrame(rows)}, "plots": {"Grid": grid}}


class ClusterTimelinePanel(SimpleAnalysisPanel):
    title = "Cluster Timeline"
    icon = "📅"
    description = "CiteSpace-style timeline view: keyword clusters as horizontal lanes, nodes placed at their (mean) year."
    OPTIONS = [
        Opt("field", "Keyword column", "columns", "Author Keywords", values=KW_COLS),
        Opt("top_n", "Top N keywords", "spin", 60, from_=10, to=400),
        Opt("min_count", "Min co-occurrence", "spin", 2, from_=1, to=50),
        Opt("year_stat", "Node year", "combo", "mean", values=["mean", "first", "median"]),
        Opt("top_labels", "Labels per cluster", "spin", 5, from_=1, to=20),
        Opt("min_cluster_size", "Min cluster size", "spin", 3, from_=1, to=50),
        Opt("seed", "Seed", "spin", 42, from_=0, to=99999),
    ]

    def compute(self, bib, o):
        from biblium import network_dynamics as ND
        graphs, _ = bib.diachronic_networks(field=o["field"], n_slices=1, cumulative=True, top_n=o["top_n"],
                                            min_count=o["min_count"], seed=o["seed"])
        if not graphs:
            raise ValueError("No network could be built.")
        G = graphs[-1]
        table = bib.cluster_timeline(field=o["field"], G=G, year_stat=o["year_stat"], top_labels=o["top_labels"],
                                     min_cluster_size=o["min_cluster_size"])
        if table.empty:
            raise ValueError("No cluster reaches the minimum size; lower 'Min cluster size' or 'Min co-occurrence'.")
        labels = bib.cluster_timeline_labels
        lab_df = pd.DataFrame([{"Cluster": k, "Label": v, "Size": int((table["Cluster"] == k).sum())} for k, v in labels.items()])
        return {"summary": [("Clusters", len(labels)), ("Nodes", len(table))],
                "tables": {"Clusters": lab_df, "Nodes": table},
                "plots": {"Timeline": lambda fig: ND.plot_cluster_timeline(table, labels, G=G, ax=fig.add_subplot(111))}}


class StructuralVariationPanel(SimpleAnalysisPanel):
    title = "Structural Variation"
    icon = "🧬"
    description = "Chen (2012): how much each document changes the reference co-citation network (modularity change, cluster linkage, centrality divergence)."
    OPTIONS = [
        Opt("min_ref_citations", "Min citations of a reference", "spin", 3, from_=2, to=100),
        Opt("top_n_refs", "Top N references", "spin", 150, from_=20, to=1000),
        Opt("baseline_window", "Baseline window (years, 0 = all)", "spin", 0, from_=0, to=50),
        Opt("metric", "Plot metric", "combo", "Modularity change rate", values=["Modularity change rate", "Cluster linkage", "Centrality divergence"]),
        Opt("seed", "Seed", "spin", 42, from_=0, to=99999),
    ]
    INFO_TEXT = """
STRUCTURAL VARIATION ANALYSIS
─────────────────────────────
For each document, its reference pairs are added to the co-citation network of
all earlier documents. Modularity change rate (ΔM %) measures how much the
community structure is weakened; cluster linkage is the share of the
document's reference pairs that bridge different baseline clusters; centrality
divergence is the KL divergence of betweenness centralities before/after.
Documents with high values are candidates for transformative, boundary-spanning
work.

Chen, C. (2012). Predictive effects of structural variation on citation counts.
JASIST, 63(3), 431–449.
"""

    def compute(self, bib, o):
        from biblium import network_dynamics as ND
        bw = o["baseline_window"] or None
        df = bib.structural_variation(min_ref_citations=o["min_ref_citations"], top_n_refs=o["top_n_refs"],
                                      baseline_window=bw, seed=o["seed"])
        if df.empty:
            raise ValueError("No documents could be scored (reference network too sparse). Lower the thresholds.")
        top = df.sort_values(o["metric"], ascending=False).head(15)
        return {"summary": [("Documents scored", len(df)), ("Mean ΔM %", f"{df['Modularity change rate'].mean():.2f}"),
                            ("Mean cluster linkage", f"{df['Cluster linkage'].mean():.2f}")],
                "tables": {"All documents": df, f"Top by {o['metric']}": top},
                "plots": {"Metric vs citations": lambda fig: ND.plot_structural_variation(df, metric=o["metric"], ax=fig.add_subplot(111))}}


class FuzzyClustersPanel(SimpleAnalysisPanel):
    title = "Fuzzy Doc Clusters"
    icon = "🌫️"
    description = "Fuzzy assignment of documents to co-word clusters (share of a document's keywords in each cluster)."
    OPTIONS = [
        Opt("field", "Keyword column", "columns", "Author Keywords", values=KW_COLS),
        Opt("top_n", "Top N keywords", "spin", 60, from_=10, to=400),
        Opt("min_count", "Min co-occurrence", "spin", 2, from_=1, to=50),
        Opt("min_membership", "Min membership for listing", "spin", 50, from_=0, to=100),
        Opt("seed", "Seed", "spin", 42, from_=0, to=99999),
    ]

    def compute(self, bib, o):
        from biblium import network_dynamics as ND
        graphs, _ = bib.diachronic_networks(field=o["field"], n_slices=1, cumulative=True, top_n=o["top_n"],
                                            min_count=o["min_count"], seed=o["seed"])
        if not graphs:
            raise ValueError("No network could be built.")
        G = graphs[-1]
        mem = bib.fuzzy_clusters(field=o["field"], G=G)
        assigned = mem.dropna(subset=["Main cluster"])
        counts = assigned["Main cluster"].value_counts().rename_axis("Cluster").reset_index(name="Documents")
        cols = [c for c in ("Title", "Year", "Cited by") if c in bib.df.columns]
        out = pd.concat([bib.df[cols], mem], axis=1)
        out = out.dropna(subset=["Main cluster"]).sort_values("Main membership", ascending=False)

        def draw(fig):
            ax = fig.add_subplot(111)
            ax.hist(assigned["Main membership"], bins=20, color="#3b82f6")
            ax.set_xlabel("Main-cluster membership"); ax.set_ylabel("Documents")
        return {"summary": [("Docs in network", len(assigned)), ("Clusters", len(counts)),
                            ("Mean membership", f"{assigned['Main membership'].mean():.2f}")],
                "tables": {"Cluster sizes": counts, "Documents": out}, "plots": {"Membership": draw}}


class TermExtractionPanel(SimpleAnalysisPanel):
    title = "Term Extraction"
    icon = "🔤"
    description = "VOSviewer-style term extraction from titles/abstracts with relevance scores; adds an 'Extracted Terms' column for mapping."
    OPTIONS = [
        Opt("use_abstract", "Use abstracts", "check", True),
        Opt("max_len", "Max phrase length", "spin", 3, from_=1, to=5),
        Opt("min_occurrences", "Min occurrences", "spin", 10, from_=2, to=500),
        Opt("share", "Keep most relevant (%)", "spin", 60, from_=10, to=100),
        Opt("add_column", "Add 'Extracted Terms' column", "check", True),
    ]
    INFO_TEXT = """
TERM EXTRACTION WITH RELEVANCE SCORE
────────────────────────────────────
Noun phrases are approximated by splitting sentences at stop words (VOSviewer
uses a POS tagger). A term's relevance is the Kullback–Leibler divergence
between its co-occurrence profile and the overall term distribution: generic
terms co-occur with everything (low relevance), specific terms with few (high).
By default the 60 % most relevant terms are kept (VOSviewer default).

van Eck, N. J., & Waltman, L. (2011). Text mining and visualization using
VOSviewer. ISSI Newsletter, 7(3), 50–54.
"""

    def compute(self, bib, o):
        from biblium import term_extraction as TE
        cols = ("Title", "Abstract") if o["use_abstract"] else ("Title",)
        scores = bib.extract_terms(columns=cols, max_len=o["max_len"], min_occurrences=o["min_occurrences"],
                                   share=o["share"] / 100.0, add_column=o["add_column"])
        return {"summary": [("Terms kept", len(scores)), ("Documents", len(bib.df))],
                "tables": {"Terms": scores, "Co-occurrence": bib.term_cooccurrence_df.reset_index().head(200)},
                "plots": {"Relevance": lambda fig: TE.plot_term_relevance(scores, ax=fig.add_subplot(111))}}
