# -*- coding: utf-8 -*-
"""
Tests for ``biblium.dual_map`` (new in 2.19).

All checks use small fake ASJC tables passed explicitly; expectations are
hand-computed. One smoke test loads the bundled Excel files.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from biblium import dual_map as D


@pytest.fixture
def meta():
    return pd.DataFrame({
        "code": [1000, 1700, 1702, 3300, 3309, 2200, 2700],
        "Field": ["Multidisciplinary", "General Computer Science", "Artificial Intelligence",
                  "General Social Sciences", "Library and Information Sciences",
                  "General Engineering", "General Medicine"],
        "Area": ["Multidisciplinary", "Computer Science", "Computer Science",
                 "Social Sciences", "Social Sciences", "Engineering", "Medicine"],
        "Science": ["Multidisciplinary", "Physical Sciences", "Physical Sciences",
                    "Social Sciences", "Social Sciences", "Physical Sciences", "Health Sciences"],
    })


@pytest.fixture
def asjc_map():
    m = pd.DataFrame({
        "Source title": ["Scientometrics", "Journal of Informetrics", "Nature",
                         "Engineering Letters", "Lancet", "AI Review"],
        "ASJC": [[3309, 1700], [3309, 1702], [1000], [2200], [2700], [1702]],
    })
    m["key"] = m["Source title"].str.lower()
    return m


@pytest.fixture
def df():
    # doc 0 (Scientometrics: Social Sciences + Computer Science) cites
    #   Nature (Multi), Engineering Letters (Engineering)
    # doc 1 (Nature: Multidisciplinary) cites Lancet (Medicine), an unknown journal,
    #   and a reference without a recognisable journal
    return pd.DataFrame({
        "Source title": ["Scientometrics", "Nature"],
        "References": [
            "Smith, J., Paper A, Nature, 500, pp. 1-5, (2010); "
            "Doe, A., Paper B, Engineering Letters, 3, 2, pp. 10-20, (2011)",
            "Roe, B., Paper C, Lancet, 1, pp. 2-3, (2012); "
            "Poe, C., Paper D, Unknown Journal, 2, pp. 4-5, (2013); "
            "Only two parts, (2014)",
        ],
    })


class TestMapping:
    def test_source_to_areas_levels(self, asjc_map, meta):
        assert D.source_to_areas("Scientometrics", asjc_map, meta) == ["Social Sciences", "Computer Science"]
        assert D.source_to_areas("scientometrics ", asjc_map, meta, "Field") == [
            "Library and Information Sciences", "General Computer Science"]
        assert D.source_to_areas("SCIENTOMETRICS", asjc_map, meta, "Science") == [
            "Social Sciences", "Physical Sciences"]
        # distinct labels only: two codes in the same area collapse to one
        assert D.source_to_areas("Journal of Informetrics", asjc_map, meta) == ["Social Sciences", "Computer Science"]

    def test_unknown_and_empty(self, asjc_map, meta):
        assert D.source_to_areas("No Such Journal", asjc_map, meta) == []
        assert D.source_to_areas(None, asjc_map, meta) == []
        assert D.source_to_areas(np.nan, asjc_map, meta) == []

    def test_bad_level(self, asjc_map, meta):
        with pytest.raises(ValueError):
            D.source_to_areas("Nature", asjc_map, meta, level="Galaxy")

    def test_parse_codes(self):
        assert D._parse_codes("2210; 2211; 3104;") == [2210, 2211, 3104]
        assert D._parse_codes("2210; 2210") == [2210]
        assert D._parse_codes(np.nan) == []

    def test_document_and_reference_areas(self, df, asjc_map, meta):
        da = D.document_areas(df, asjc_map=asjc_map, meta=meta)
        assert da.tolist() == [["Social Sciences", "Computer Science"], ["Multidisciplinary"]]
        ra = D.reference_areas(df, asjc_map=asjc_map, meta=meta)
        assert ra.tolist() == [[["Multidisciplinary"], ["Engineering"]], [["Medicine"]]]

    def test_custom_extractor(self, df, asjc_map, meta):
        ra = D.reference_areas(df, asjc_map=asjc_map, meta=meta, extractor=lambda r: "Lancet")
        assert ra.tolist() == [[["Medicine"]] * 2, [["Medicine"]] * 3]


class TestFlows:
    def test_fractional_counting(self, df, asjc_map, meta):
        flows, st = D.dual_map_flows(df, asjc_map=asjc_map, meta=meta, min_flow=0)
        mat = D.dual_map_matrix(flows)
        # doc 0: 2 citing areas, each reference 1 area -> 1/2 per pair
        assert mat.loc["Social Sciences", "Multidisciplinary"] == pytest.approx(0.5)
        assert mat.loc["Computer Science", "Engineering"] == pytest.approx(0.5)
        assert mat.loc["Multidisciplinary", "Medicine"] == pytest.approx(1.0)
        assert flows["Flow"].sum() == pytest.approx(3.0)  # 3 mapped references
        assert st["n_docs_total"] == 2 and st["n_docs_mapped"] == 2
        assert st["n_refs_total"] == 5 and st["n_refs_mapped"] == 3
        assert st["coverage_docs"] == pytest.approx(1.0)
        assert st["coverage_refs"] == pytest.approx(3 / 5)
        assert st["self_discipline_share"] == pytest.approx(0.0)
        assert st["n_citing_areas"] == 3 and st["n_cited_areas"] == 3
        assert st["citing_totals"]["Social Sciences"] == pytest.approx(1.0)
        assert st["cited_totals"]["Medicine"] == pytest.approx(1.0)

    def test_full_counting(self, df, asjc_map, meta):
        flows, st = D.dual_map_flows(df, asjc_map=asjc_map, meta=meta, fractional=False)
        mat = D.dual_map_matrix(flows)
        assert mat.loc["Social Sciences", "Multidisciplinary"] == pytest.approx(1.0)
        assert mat.loc["Computer Science", "Engineering"] == pytest.approx(1.0)
        assert flows["Flow"].sum() == pytest.approx(5.0)  # 2*2 + 1
        assert st["total_flow"] == pytest.approx(5.0)

    def test_min_flow_filters_table_not_stats(self, df, asjc_map, meta):
        flows, st = D.dual_map_flows(df, asjc_map=asjc_map, meta=meta, min_flow=1)
        assert len(flows) == 1 and flows.loc[0, "Cited area"] == "Medicine"
        assert st["total_flow"] == pytest.approx(3.0)

    def test_self_share_and_summary(self, asjc_map, meta):
        d = pd.DataFrame({
            "Source title": ["Lancet", "Lancet"],
            "References": [
                "A, B, T, Lancet, 1, pp. 1-2, (2000); C, D, T, Lancet, 2, pp. 1-2, (2001); "
                "E, F, T, Nature, 3, pp. 1-2, (2002)",
                "G, H, T, Nature, 4, pp. 1-2, (2003)",
            ],
        })
        flows, st = D.dual_map_flows(d, asjc_map=asjc_map, meta=meta)
        assert st["self_discipline_share"] == pytest.approx(0.5)
        s = D.dual_map_summary(flows).set_index("Citing area")
        assert s.loc["Medicine", "Outflow"] == pytest.approx(4.0)
        assert s.loc["Medicine", "N cited areas"] == 2
        assert s.loc["Medicine", "Self share"] == pytest.approx(0.5)
        assert s.loc["Medicine", "Entropy"] == pytest.approx(1.0)  # two equal destinations
        assert s.loc["Medicine", "Top cited area"] in ("Medicine", "Multidisciplinary")

    def test_empty_corpus(self, asjc_map, meta):
        d = pd.DataFrame({"Source title": ["Unknown"], "References": [np.nan]})
        flows, st = D.dual_map_flows(d, asjc_map=asjc_map, meta=meta)
        assert flows.empty and st["n_docs_mapped"] == 0 and np.isnan(st["self_discipline_share"])
        assert D.dual_map_summary(flows).empty
        assert D.dual_map_matrix(flows).empty


class TestPlot:
    def test_plot_renders(self, df, asjc_map, meta, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        flows, st = D.dual_map_flows(df, asjc_map=asjc_map, meta=meta, min_flow=0)
        out = tmp_path / "dual.png"
        fig, ax = D.plot_dual_map(flows, st, f_name=str(out), dpi=50)
        assert out.exists() and out.stat().st_size > 0
        assert len(ax.patches) > 0
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_empty(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        flows = pd.DataFrame(columns=["Citing area", "Cited area", "Flow"])
        fig, ax = D.plot_dual_map(flows)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_stack_positions_no_overlap(self):
        pos = D._stack_positions([3, 1, 1], 1.0, 0.05)
        assert len(pos) == 3
        for (y0, h0), (y1, h1) in zip(pos, pos[1:]):
            assert y1 + h1 <= y0 + 1e-9  # next bar ends below the current one
        assert pos[0][1] == pytest.approx(3 * 0.9 / 5)


class TestBundledFiles:
    def test_real_tables_map_scientometrics(self):
        if not (os.path.exists(D.DEFAULT_MAP_PATH) and os.path.exists(D.DEFAULT_META_PATH)):
            pytest.skip("bundled ASJC files not available")
        m, t = D.load_asjc_tables()
        assert {"Source title", "ASJC", "key"} <= set(m.columns)
        assert {"code", "Field", "Area", "Science"} <= set(t.columns)
        areas = D.source_to_areas("Scientometrics", m, t)
        assert areas and "Social Sciences" in areas
        assert D.source_to_areas("Scientometrics", m, t, "Science")
