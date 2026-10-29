# -*- coding: utf-8 -*-
"""
Regression tests for ``biblium.network_dynamics`` (new in 2.19).

Small synthetic corpora with hand-computed expectations; deterministic and
hermetic (the only file written is a GIF in a pytest temporary directory).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from biblium import network_dynamics as ND


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kw_df():
    # item frequencies in the full corpus: A=6, B=6, C=4, D=4, E=4, F=2, G=1
    return pd.DataFrame({
        "Year": [2000, 2000, 2001, 2001, 2002, 2002, 2003, 2003, 2004, 2004, 2005, 2005],
        "Author Keywords": ["A; B", "A; B", "A; C", "B; C", "A; B; C", "D; E",
                            "D; E", "D; E; F", "A; D", "B; E", "F; G", "A; B; C"],
    })


@pytest.fixture
def two_cluster_graph():
    G = nx.Graph()
    G.add_edges_from([("a", "b", {"weight": 2}), ("b", "c", {"weight": 1}), ("a", "c", {"weight": 1}),
                      ("d", "e", {"weight": 3})])
    for n in "abc":
        G.nodes[n]["partition_louvain"] = 0
    for n in "de":
        G.nodes[n]["partition_louvain"] = 1
    return G


# ---------------------------------------------------------------------------
# Period graphs
# ---------------------------------------------------------------------------

class TestPeriodGraphs:
    PERIODS = [(2000, 2001), (2002, 2003), (2004, 2005)]

    def test_quantile_periods(self):
        years = pd.Series([2000, 2001, 2001, 2002, 2003])
        assert ND._quantile_periods(years, 2) == [(2000, 2001), (2002, 2003)]
        with pytest.raises(ValueError):
            ND._quantile_periods(pd.Series([2000, 2000]), 2)

    def test_stable_node_set_top_n(self, kw_df):
        graphs, labels = ND.period_graphs(kw_df, periods=self.PERIODS, top_n=3,
                                          min_count=1, min_weight=0)
        assert labels == ["2000-2001", "2002-2003", "2004-2005"]
        for G in graphs:
            assert set(G.nodes()) <= {"A", "B", "C"}
            assert all("freq" in d and "partition_louvain" in d for _, d in G.nodes(data=True))
        assert set(graphs[0].nodes()) == {"A", "B", "C"}

    def test_cumulative_counts_monotone(self, kw_df):
        graphs, _ = ND.period_graphs(kw_df, periods=self.PERIODS, top_n=10, cumulative=True,
                                     min_count=1, min_weight=0)
        assert [G.nodes["A"]["freq"] for G in graphs] == [3, 4, 6]
        assert [G["A"]["B"]["weight"] for G in graphs] == [2, 3, 4]
        # node sets grow monotonically in cumulative mode
        for g1, g2 in zip(graphs, graphs[1:]):
            assert set(g1.nodes()) <= set(g2.nodes())
        assert set(graphs[-1].nodes()) == set("ABCDEFG")

    def test_non_cumulative(self, kw_df):
        graphs, _ = ND.period_graphs(kw_df, periods=self.PERIODS, top_n=10, cumulative=False,
                                     min_count=1, min_weight=0)
        assert [G["A"]["B"]["weight"] for G in graphs] == [2, 1, 1]
        assert graphs[0].nodes["A"]["freq"] == 3 and graphs[1].nodes["A"]["freq"] == 1
        assert "D" not in graphs[0]

    def test_min_count_filters_edges(self, kw_df):
        graphs, _ = ND.period_graphs(kw_df, periods=self.PERIODS, top_n=10, cumulative=True,
                                     min_count=3, min_weight=0)
        G = graphs[-1]
        assert G.has_edge("A", "B") and G.has_edge("D", "E")   # 4 and 3 co-occurrences
        assert G.has_edge("A", "C")                             # 3 -> kept
        assert not G.has_edge("B", "E") and not G.has_edge("A", "D")   # 1 each -> dropped

    def test_louvain_partition_numbering(self):
        G = nx.Graph()
        G.add_edges_from([("x", "y"), ("y", "z"), ("x", "z"), ("a", "b"), ("b", "c"), ("a", "c")])
        part = ND._louvain_partition(G, seed=1)
        assert part["a"] == part["b"] == part["c"] == 0   # tie in size -> smallest label first
        assert part["x"] == part["y"] == part["z"] == 1
        assert ND._louvain_partition(nx.empty_graph(3)) == {0: 0, 1: 1, 2: 2}

    def test_deterministic(self, kw_df):
        g1, _ = ND.period_graphs(kw_df, periods=self.PERIODS, min_count=1, min_weight=0)
        g2, _ = ND.period_graphs(kw_df, periods=self.PERIODS, min_count=1, min_weight=0)
        for a, b in zip(g1, g2):
            assert dict(a.nodes(data=True)) == dict(b.nodes(data=True))


# ---------------------------------------------------------------------------
# Animation / grid
# ---------------------------------------------------------------------------

class TestAnimation:
    def test_animate_writes_gif(self, kw_df, tmp_path):
        graphs, labels = ND.period_graphs(kw_df, periods=[(2000, 2002), (2003, 2005)],
                                          min_count=1, min_weight=0)
        out = tmp_path / "anim.gif"
        path = ND.animate_diachronic_network(graphs, labels, str(out), fps=2, figsize=(3, 3), dpi=40)
        assert path == str(out) and out.exists() and os.path.getsize(out) > 0
        with open(out, "rb") as fh:
            assert fh.read(6) in (b"GIF87a", b"GIF89a")

    def test_grid(self, kw_df):
        graphs, labels = ND.period_graphs(kw_df, periods=[(2000, 2002), (2003, 2005)],
                                          min_count=1, min_weight=0)
        fig, axes = ND.plot_diachronic_grid(graphs, labels, ncols=3, figsize_per_panel=(2, 2))
        assert len(axes) == 2
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_union_layout_is_fixed(self, kw_df):
        graphs, _ = ND.period_graphs(kw_df, periods=[(2000, 2002), (2003, 2005)],
                                     min_count=1, min_weight=0)
        pos1 = ND._fixed_layout(graphs, "union", 42)
        pos2 = ND._fixed_layout(graphs, "union", 42)
        assert set(pos1) == set("ABCDEFG")
        assert all(np.allclose(pos1[n], pos2[n]) for n in pos1)


# ---------------------------------------------------------------------------
# Fuzzy membership
# ---------------------------------------------------------------------------

class TestFuzzyMembership:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "Title": ["t0", "t1", "t2", "t3"],
            "Year": [2000, 2001, 2002, 2003],
            "Cited by": [5, 1, 0, 2],
            "Author Keywords": ["a; b; d", "d; e; zzz", "zzz", "a; d"],
        })

    def test_rows_sum_to_one_and_main_cluster(self, df, two_cluster_graph):
        m = ND.fuzzy_cluster_membership(df, two_cluster_graph)
        assert list(m.columns[:2]) == ["Cluster 0", "Cluster 1"]
        assert m.loc[0, ["Cluster 0", "Cluster 1"]].tolist() == pytest.approx([2 / 3, 1 / 3])
        assert m.loc[1, ["Cluster 0", "Cluster 1"]].tolist() == pytest.approx([0.0, 1.0])
        sums = m.loc[[0, 1, 3], ["Cluster 0", "Cluster 1"]].sum(axis=1)
        assert sums.tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert m.loc[0, "Main cluster"] == 0 and m.loc[1, "Main cluster"] == 1
        assert m.loc[3, "Main cluster"] == 0          # tie 0.5/0.5 -> smallest id
        assert m.loc[0, "Main membership"] == pytest.approx(2 / 3)
        assert m["N keywords in network"].tolist() == [3, 2, 0, 2]

    def test_entropy_and_nan(self, df, two_cluster_graph):
        m = ND.fuzzy_cluster_membership(df, two_cluster_graph)
        assert m.loc[0, "Entropy"] == pytest.approx(-(2 / 3 * np.log(2 / 3) + 1 / 3 * np.log(1 / 3)))
        assert m.loc[1, "Entropy"] == pytest.approx(0.0)
        assert m.loc[3, "Entropy"] == pytest.approx(np.log(2))
        assert np.isnan(m.loc[2, "Cluster 0"]) and np.isnan(m.loc[2, "Main cluster"])
        assert np.isnan(m.loc[2, "Entropy"])

    def test_raw_counts(self, df, two_cluster_graph):
        m = ND.fuzzy_cluster_membership(df, two_cluster_graph, normalize=False)
        assert m.loc[0, ["Cluster 0", "Cluster 1"]].tolist() == [2, 1]
        assert m.loc[0, "Main membership"] == pytest.approx(2 / 3)   # shares regardless

    def test_cluster_document_table(self, df, two_cluster_graph):
        m = ND.fuzzy_cluster_membership(df, two_cluster_graph)
        t = ND.cluster_document_table(m, df, 1, min_membership=0.5)
        assert list(t.index) == [1, 3]
        assert t["Membership"].tolist() == [1.0, 0.5]
        assert list(t.columns) == ["Title", "Year", "Cited by", "Membership", "Main cluster"]
        with pytest.raises(ValueError):
            ND.cluster_document_table(m, df, 9)


# ---------------------------------------------------------------------------
# Cluster timeline
# ---------------------------------------------------------------------------

class TestClusterTimeline:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "Year": [2000, 2002, 2004, 2001],
            "Author Keywords": ["a; b; d", "a; d", "a; c; e", "b"],
        })

    def test_years_and_freq(self, df, two_cluster_graph):
        tab, labels = ND.cluster_timeline(two_cluster_graph, df, top_labels=1)
        t = tab.set_index("Node")
        assert t.loc["a", "Freq"] == 3 and t.loc["b", "Freq"] == 2 and t.loc["e", "Freq"] == 1
        assert t.loc["a", "Year"] == pytest.approx(2002.0)      # mean(2000, 2002, 2004)
        assert t.loc["b", "Year"] == pytest.approx(2000.5)
        assert t.loc["d", "Year"] == pytest.approx(2001.0)
        assert t.loc["a", "Label"] and t.loc["d", "Label"] and not t.loc["b", "Label"]
        assert labels == {0: "a; b; c", 1: "d; e"}
        assert t.loc["a", "Cluster rank"] == 0 and t.loc["d", "Cluster rank"] == 1
        assert tab["Node"].tolist() == ["a", "b", "c", "d", "e"]   # by cluster size, then Freq

    def test_first_and_median(self, df, two_cluster_graph):
        tab, _ = ND.cluster_timeline(two_cluster_graph, df, year_stat="first")
        assert tab.set_index("Node").loc["a", "Year"] == 2000
        tab, _ = ND.cluster_timeline(two_cluster_graph, df, year_stat="median")
        assert tab.set_index("Node").loc["b", "Year"] == 2000.5
        with pytest.raises(ValueError):
            ND.cluster_timeline(two_cluster_graph, df, year_stat="max")

    def test_plot(self, df, two_cluster_graph):
        tab, labels = ND.cluster_timeline(two_cluster_graph, df)
        fig, ax = ND.plot_cluster_timeline(tab, labels, G=two_cluster_graph)
        assert len(ax.get_yticklabels()) == 2
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Structural variation
# ---------------------------------------------------------------------------

def _refs(*names):
    return "; ".join(f"{n}, Title, Journal, 1, pp. 1-2, (1990)" for n in names)


class TestStructuralVariation:
    @pytest.fixture
    def df(self):
        # baseline (2000): two disconnected triangles {R1,R2,R3} and {R4,R5,R6}
        rows = [{"Year": 2000, "Title": f"base{i}", "Cited by": 1, "References": _refs("R1", "R2", "R3")}
                for i in range(3)]
        rows += [{"Year": 2000, "Title": f"base{i + 3}", "Cited by": 1, "References": _refs("R4", "R5", "R6")}
                 for i in range(3)]
        rows += [
            {"Year": 2001, "Title": "bridge", "Cited by": 50, "References": _refs("R1", "R4")},
            {"Year": 2001, "Title": "within", "Cited by": 3, "References": _refs("R1", "R2")},
            {"Year": 2001, "Title": "mixed", "Cited by": 10, "References": _refs("R1", "R2", "R4")},
            {"Year": 2001, "Title": "single", "Cited by": 0, "References": _refs("R1", "R9")},
        ]
        return pd.DataFrame(rows)

    def test_metrics(self, df):
        out = ND.structural_variation(df, min_ref_citations=3, top_n_refs=6).set_index("Title")
        # the first year has no baseline, 'single' has only one reference in the network
        assert set(out.index) == {"bridge", "within", "mixed"}
        assert (out["Year"] == 2001).all()
        assert out.loc["bridge", "N refs in network"] == 2 and out.loc["mixed", "N refs in network"] == 3
        # cluster linkage
        assert out.loc["bridge", "Cluster linkage"] == 1.0
        assert out.loc["within", "Cluster linkage"] == 0.0
        assert out.loc["mixed", "Cluster linkage"] == pytest.approx(2 / 3)
        assert out["Cluster linkage"].idxmax() == "bridge"
        # modularity: M0 = 0.5; bridge -> 2 * (9/19 - (19/38)^2) = 0.4474 -> -10.5 %
        assert out.loc["bridge", "Modularity change rate"] == pytest.approx((2 * (9 / 19 - 0.25) - 0.5) / 0.5 * 100)
        assert out.loc["bridge", "Modularity change rate"] < 0
        # within: 10/19 - (20/38)^2 + 9/19 - (18/38)^2 (only a slight imbalance effect)
        m1 = 10 / 19 - (20 / 38) ** 2 + 9 / 19 - (18 / 38) ** 2
        assert out.loc["within", "Modularity change rate"] == pytest.approx((m1 - 0.5) / 0.5 * 100)
        assert out.loc["bridge", "Modularity change rate"] < out.loc["within", "Modularity change rate"]
        # mixed: two inter-cluster edges + one intra -> m = 21, internal 10 / 9, degrees 22 / 20
        m_mixed = 10 / 21 - (22 / 42) ** 2 + 9 / 21 - (20 / 42) ** 2
        assert out.loc["mixed", "Modularity change rate"] == pytest.approx((m_mixed - 0.5) / 0.5 * 100)
        assert out["Modularity change rate"].idxmin() == "mixed"
        # centrality divergence: betweenness unchanged for 'within', bridge creates central nodes
        assert out.loc["within", "Centrality divergence"] == pytest.approx(0.0, abs=1e-6)
        assert out.loc["bridge", "Centrality divergence"] > out.loc["within", "Centrality divergence"]
        assert out.loc["bridge", "Cited by"] == 50

    def test_baseline_window_and_limits(self, df):
        out = ND.structural_variation(df, min_ref_citations=3, top_n_refs=6, max_docs_per_year=1)
        assert out["Title"].tolist() == ["bridge"]
        # window of 0 years back -> no baseline documents -> nothing processed
        out = ND.structural_variation(df, min_ref_citations=3, top_n_refs=6, baseline_window=0)
        assert out.empty
        out = ND.structural_variation(df, min_ref_citations=3, top_n_refs=6, baseline_window=1)
        assert len(out) == 3

    def test_approx_k_and_missing_columns(self, df):
        out = ND.structural_variation(df, top_n_refs=6, approx_k=3)
        assert len(out) == 3
        d2 = df.drop(columns=["Title", "Cited by"])
        out = ND.structural_variation(d2, top_n_refs=6)
        assert "Title" not in out.columns and "Cited by" not in out.columns and len(out) == 3
        assert ND.structural_variation(pd.DataFrame({"Year": [2000]})).empty

    def test_kl_divergence(self):
        assert ND._kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)
        assert ND._kl_divergence([1.0, 0.0], [0.5, 0.5]) > 0

    def test_plot(self, df):
        out = ND.structural_variation(df, top_n_refs=6)
        fig, ax = ND.plot_structural_variation(out, top_n_annotate=2)
        assert len(ax.texts) == 2
        import matplotlib.pyplot as plt
        plt.close(fig)
