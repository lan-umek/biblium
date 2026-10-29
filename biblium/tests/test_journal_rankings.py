# -*- coding: utf-8 -*-
"""Tests for ``biblium.journal_rankings`` (new in 2.19): key normalisation,
SCImago-style CSV loading, ISSN-before-title matching, summaries, filtering
and plotting. Hermetic (temporary files only)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from biblium import journal_rankings as J


SCIMAGO_CSV = (
    "Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index;Categories\n"
    '1;1;Science of the Total Environment;journal;00489697, 18791026;2,123;Q1;300;"Environmental Engineering (Q1); Pollution (Q1)"\n'
    "2;2;Journal of Business Research;journal;01482963;2,000;1;200;Marketing (Q1)\n"
    "3;3;The Minor Journal;journal;9999-999X;0,15;q4;5;Other (Q4)\n"
    "4;4;No ISSN Journal;journal;-;0,10;3;2;Other (Q3)\n"
)


class TestNormalisation:
    @pytest.mark.parametrize("raw, exp", [
        ("1234-567X", "1234567X"),
        ("1234567x", "1234567X"),
        ("12345678", "12345678"),
        (" 1234-5678 ", "12345678"),
        ("1234-5678, 8765-4321", "12345678"),
        ("00489697; 18791026", "00489697"),
        (1389130, "01389130"),          # leading zero lost by numeric conversion
        (1389130.0, "01389130"),
        ("1234567", "01234567"),
        ("123456789", None),
        ("abc", None),
        ("", None),
        (None, None),
        (np.nan, None),
    ])
    def test_normalize_issn(self, raw, exp):
        assert J.normalize_issn(raw) == exp

    def test_split_issns(self):
        assert J.split_issns("1234-5678, 8765-432X; 1234-5678 / bad") == ["12345678", "8765432X"]
        assert J.split_issns(None) == [] and J.split_issns(["1234-5678"]) == ["12345678"]

    @pytest.mark.parametrize("raw, exp", [
        ("The Journal of Marketing & Sales: Vol. II", "journal of marketing and sales vol ii"),
        ("  Science   of the Total-Environment ", "science of the total environment"),
        (None, ""),
    ])
    def test_normalize_title(self, raw, exp):
        assert J.normalize_title(raw) == exp

    @pytest.mark.parametrize("raw, exp", [("1", "Q1"), ("Q1", "Q1"), ("q3", "Q3"), (2, "Q2"),
                                          ("2.0", "Q2"), ("A*", "A*"), (None, None), ("", None)])
    def test_normalize_quartile(self, raw, exp):
        assert J._normalize_quartile(raw) == exp


class TestLoad:
    def test_scimago_csv(self, tmp_path):
        p = tmp_path / "scimagojr.csv"
        p.write_text(SCIMAGO_CSV, encoding="utf-8")
        r = J.load_ranking_list(p)
        assert list(r.columns) == ["Title", "Title norm", "ISSN", "Rank", "Quartile", "Category", "Score"]
        assert len(r) == 4
        assert r.loc[0, "ISSN"] == ["00489697", "18791026"]
        assert r.loc[0, "Quartile"] == "Q1" and r.loc[1, "Quartile"] == "Q1" and r.loc[2, "Quartile"] == "Q4"
        assert r.loc[0, "Category"] == "Environmental Engineering (Q1); Pollution (Q1)"
        assert r["Rank"].tolist() == [1, 2, 3, 4]
        assert r.loc[0, "Score"] == pytest.approx(2.123)   # decimal comma
        assert r.loc[2, "Title norm"] == "minor journal"
        assert r.loc[3, "ISSN"] == []

    def test_generic_dataframe_with_named_columns(self):
        df = pd.DataFrame({"Journal name": ["A", "B"], "ABS": ["4*", "2"], "Field": ["Mgmt", "Econ"]})
        r = J.load_ranking_list(df, title_col="Journal name", rank_col="ABS", category_col="Field")
        assert r["Rank"].tolist() == ["4*", 2]
        assert r["Quartile"].isna().all() and r["ISSN"].tolist() == [[], []]

    def test_xlsx_and_comma_csv(self, tmp_path):
        df = pd.DataFrame({"Title": ["X"], "ISSN": ["1111-1111"], "Quartile": ["2"]})
        p = tmp_path / "list.csv"
        df.to_csv(p, index=False)
        r = J.load_ranking_list(p)
        assert r.loc[0, "ISSN"] == ["11111111"] and r.loc[0, "Quartile"] == "Q2"
        x = tmp_path / "list.xlsx"
        df.to_excel(x, index=False)
        r2 = J.load_ranking_list(x)
        assert r2.loc[0, "Title"] == "X"

    def test_errors(self):
        with pytest.raises(ValueError):
            J.load_ranking_list(pd.DataFrame({"foo": [1]}))
        with pytest.raises(ValueError):
            J.load_ranking_list(pd.DataFrame({"Title": ["a"]}), rank_col="nope")


@pytest.fixture
def ranking(tmp_path):
    p = tmp_path / "scimagojr.csv"
    p.write_text(SCIMAGO_CSV, encoding="utf-8")
    return J.load_ranking_list(p)


@pytest.fixture
def docs():
    return pd.DataFrame({
        "Source title": [
            "Science of the Total Environment",   # by ISSN (second ISSN)
            "Journal of Business Research",       # no ISSN → title match
            "Minor Journal",                      # title match (leading 'The' dropped in ranking)
            "Journal of Business Research",       # ISSN says STOTEN → ISSN wins over title
            "Unknown Journal",
            "No ISSN Journal",
        ],
        "ISSN": ["18791026", np.nan, "", "00489697", "55555555", None],
        "Year": [2020, 2020, 2021, 2021, 2021, 2020],
        "Cited by": [10, 30, 4, 8, 0, np.nan],
    })


class TestMatch:
    def test_precedence_and_columns(self, docs, ranking):
        m = J.match_sources(docs, ranking)
        assert m["Ranking match"].tolist() == ["issn", "title", "title", "issn", None, "title"]
        assert m["Journal Quartile"].tolist() == ["Q1", "Q1", "Q4", "Q1", None, "Q3"]
        assert m["Journal Rank"].tolist()[:4] == [1, 2, 3, 1]
        assert np.isnan(m.loc[4, "Journal Rank"])
        assert m.loc[3, "Journal Category"].startswith("Environmental")
        assert m.loc[0, "Journal Score"] == pytest.approx(2.123)
        assert len(m.columns) == len(docs.columns) + 5
        assert "Journal Rank" not in docs.columns  # input not mutated

    def test_without_issn_column(self, docs, ranking):
        m = J.match_sources(docs.drop(columns=["ISSN"]), ranking)
        assert m["Ranking match"].tolist() == ["title", "title", "title", "title", None, "title"]
        assert m.loc[3, "Journal Rank"] == 2


class TestSummaries:
    def test_ranking_summary(self, docs, ranking):
        s = J.ranking_summary(J.match_sources(docs, ranking)).set_index("Quartile")
        assert list(s.index) == ["Q1", "Q3", "Q4", "Unranked"]
        assert s.loc["Q1", "Docs"] == 3 and s.loc["Q1", "% Docs"] == pytest.approx(50.0)
        assert s.loc["Q1", "Total citations"] == 48 and s.loc["Q1", "Mean citations"] == pytest.approx(16.0)
        # year means (NaN → 0): 2020: (10+30+0)/3 = 40/3; 2021: (4+8+0)/3 = 4
        # Q1 docs: 10/(40/3) = 0.75, 30/(40/3) = 2.25, 8/4 = 2 → mean 5/3
        assert s.loc["Q1", "MNCS"] == pytest.approx(5 / 3)
        assert s.loc["Q4", "MNCS"] == pytest.approx(1.0)
        assert s.loc["Unranked", "Docs"] == 1 and s.loc["Unranked", "MNCS"] == pytest.approx(0.0)
        assert np.isnan(s.loc["Q3", "Mean citations"]) and s.loc["Q3", "Total citations"] == 0

    def test_ranking_by_year(self, docs, ranking):
        t = J.ranking_by_year(J.match_sources(docs, ranking))
        assert list(t.columns) == ["Q1", "Q3", "Q4", "Unranked"]
        assert list(t.index) == [2020, 2021]
        assert t.loc[2020].tolist() == [2, 1, 0, 0]
        assert t.loc[2021].tolist() == [1, 0, 1, 1]

    def test_filter(self, docs, ranking):
        m = J.match_sources(docs, ranking)
        assert len(J.filter_by_ranking(m, quartiles=["Q1"])) == 3
        assert len(J.filter_by_ranking(m, quartiles=["q1", "4"])) == 4
        assert len(J.filter_by_ranking(m, quartiles=["Q1"], keep_unranked=True)) == 4
        assert len(J.filter_by_ranking(m, max_rank=2)) == 3   # ranks 1, 2, 1
        assert len(J.filter_by_ranking(m, categories=["marketing"])) == 1
        assert len(J.filter_by_ranking(m, quartiles=["Q1"], categories=["pollution"])) == 2

    def test_plot(self, docs, ranking, tmp_path):
        t = J.ranking_by_year(J.match_sources(docs, ranking))
        out = tmp_path / "rank.png"
        fig, ax = J.plot_ranking_distribution(t, f_name=str(out), dpi=50)
        assert out.exists()
        assert [t_.get_text() for t_ in ax.get_legend().get_texts()] == ["Q1", "Q3", "Q4", "Unranked"]
        matplotlib.pyplot.close(fig)
        fig, ax = J.plot_ranking_distribution(t, stacked=False, percent=True)
        assert ax.get_ylabel().startswith("Share")
        matplotlib.pyplot.close(fig)
