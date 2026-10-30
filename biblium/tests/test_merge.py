# -*- coding: utf-8 -*-
"""Tests for ``biblium.merge`` (combining records from several databases, 2.19)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biblium import merge as M


def _scopus():
    return pd.DataFrame({
        "Title": ["A paper", "B paper", "C paper", "E paper"],
        "DOI": ["10.1/a", "10.1/B", None, "10.1/e"],
        "Cited by": [5, 1, 2, 7],
        "Author Keywords": ["x; y", "z", "q", "e"],
        "Abstract": ["short", "longer abstract", None, "e"],
        "Only Scopus": [1, 2, 3, 4],
        "Year": [2001, 2002, 2003, 2005],
    })


def _openalex():
    return pd.DataFrame({
        "Title": ["A Paper!", "D paper", "c paper", "E paper"],
        "DOI": ["https://doi.org/10.1/A", "10.1/d", None, "doi:10.1/E"],
        "Cited by": [9, 3, 0, 4],
        "Author Keywords": ["y|w", "k", "q", "e2"],
        "Abstract": ["a much longer abstract here", None, "abs", None],
        "Year": [2001, 2004, 2003, 2005],
    })


class TestNormalisation:
    def test_doi(self):
        assert M.normalize_doi("https://doi.org/10.1/A ") == "10.1/a"
        assert M.normalize_doi("DOI:10.1/x.") == "10.1/x"
        assert M.normalize_doi("not a doi") is None
        assert M.normalize_doi(None) is None and M.normalize_doi(float("nan")) is None

    def test_title(self):
        assert M.normalize_title("A Paper!") == "a paper"
        assert M.normalize_title("  <i>Title</i>, with: punctuation ") == "title with punctuation"
        assert M.normalize_title(None) is None

    def test_harmonize_openalex_separators_and_types(self):
        df = pd.DataFrame({"Author Keywords": ["a|b | c"], "DOI": ["https://doi.org/10.1/Z"], "Year": ["2001"], "Cited by": ["3"]})
        h = M.harmonize(df, "openalex")
        assert h.loc[0, "Author Keywords"] == "a; b; c"
        assert h.loc[0, "DOI"] == "10.1/z"
        assert h.loc[0, "Year"] == 2001 and h.loc[0, "Cited by"] == 3
        assert h.loc[0, "Source DB"] == "openalex"


class TestCombine:
    def test_doi_union(self):
        m, r = M.combine_databases([(_scopus(), "scopus"), (_openalex(), "openalex")], key="doi", columns="union")
        assert r["n_input"] == {"scopus": 4, "openalex": 4}
        # A and E matched by DOI; C/c not (no DOI) -> 6 records
        assert r["n_output"] == 6 and r["n_removed"] == 2 and r["n_without_key"] == 2
        a = m[m["Title"] == "A paper"].iloc[0]
        assert a["Source DB"] == "scopus; openalex" and a["N sources"] == 2
        assert a["Cited by"] == 9 and a["Cited by (scopus)"] == 5 and a["Cited by (openalex)"] == 9
        assert a["Author Keywords"] == "x; y; w"             # union, scopus first
        assert a["Abstract"] == "a much longer abstract here"  # longest
        assert a["Only Scopus"] == 1                         # union keeps source-specific columns
        d = m[m["Title"] == "D paper"].iloc[0]
        assert pd.isna(d["Only Scopus"]) and d["Source DB"] == "openalex"
        assert "Only Scopus" in r["columns_kept"] and r["columns_dropped"] == []

    def test_intersection_drops_source_specific_columns(self):
        m, r = M.combine_databases([(_scopus(), "scopus"), (_openalex(), "openalex")], key="doi", columns="intersection")
        assert "Only Scopus" not in m.columns and r["columns_dropped"] == ["Only Scopus"]
        assert "Source DB" in m.columns and "Merge key" in m.columns

    def test_doi_title_fallback(self):
        m, r = M.combine_databases([(_scopus(), "scopus"), (_openalex(), "openalex")], key="doi_title")
        assert r["n_output"] == 5
        c = m[m["Title"] == "C paper"].iloc[0]
        assert c["Match type"] == "title" and c["N sources"] == 2 and c["Abstract"] == "abs"

    def test_title_only(self):
        m, r = M.combine_databases([(_scopus(), "scopus"), (_openalex(), "openalex")], key="title")
        assert r["n_output"] == 5 and set(m["Match type"]) == {"title"}

    def test_priority_decides_ordinary_fields(self):
        a = pd.DataFrame({"Title": ["T"], "DOI": ["10.1/p"], "Document Type": ["Article"]})
        b = pd.DataFrame({"Title": ["T"], "DOI": ["10.1/p"], "Document Type": ["journal-article"]})
        m, _ = M.combine_databases([(a, "scopus"), (b, "openalex")])
        assert m.loc[0, "Document Type"] == "Article"
        m2, _ = M.combine_databases([(a, "scopus"), (b, "openalex")], priority=["openalex", "scopus"])
        assert m2.loc[0, "Document Type"] == "journal-article"

    def test_within_source_duplicates_and_overlap_tables(self):
        a = pd.DataFrame({"Title": ["T", "T dup"], "DOI": ["10.1/p", "10.1/P"], "Cited by": [1, 2]})
        b = pd.DataFrame({"Title": ["T"], "DOI": ["10.1/p"], "Cited by": [3]})
        m, r = M.combine_databases([(a, "scopus"), (b, "wos")])
        assert len(m) == 1 and r["n_within_source_duplicates"] == 1 and m.loc[0, "Cited by"] == 3
        ov = r["overlap"]
        assert ov.loc[0, "Sources"] == "scopus; wos" and ov.loc[0, "Records"] == 1
        pw = r["pairwise"]
        assert pw.loc["scopus", "wos"] == 1 and pw.loc["scopus", "scopus"] == 1

    def test_column_coverage(self):
        m, r = M.combine_databases([(_scopus(), "scopus"), (_openalex(), "openalex")], key="doi", columns="union")
        cov = r["column_coverage"].set_index("Column")
        assert cov.loc["Only Scopus", "Non-missing"] == 4 and cov.loc["Source DB", "Coverage %"] == 100.0

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            M.combine_databases([(_scopus(), "scopus")], key="isbn")
        with pytest.raises(ValueError):
            M.combine_databases([(_scopus(), "scopus")], columns="left")

    def test_single_source_dedups_itself(self):
        a = pd.DataFrame({"Title": ["T", "T"], "DOI": ["10.1/p", "10.1/p"], "Cited by": [1, 5]})
        m, r = M.combine_databases([(a, "scopus")])
        assert len(m) == 1 and m.loc[0, "Cited by"] == 5 and m.loc[0, "N sources"] == 1

    def test_plot(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        _, r = M.combine_databases([(_scopus(), "scopus"), (_openalex(), "openalex")])
        fig, ax = M.plot_source_overlap(r["overlap"], f_name=str(tmp_path / "ov.png"))
        assert (tmp_path / "ov.png").exists()


class TestBiblioStatsIntegration:
    def test_from_databases_and_merge_with(self):
        pytest.importorskip("biblium.bibstats")
        from biblium.bibstats import BiblioStats
        bib = BiblioStats.from_databases([(_scopus(), "scopus"), (_openalex(), "openalex")], key="doi_title",
                                         columns="intersection", verbose=False)
        assert bib.n == 5 and bib.source_dbs == ["scopus", "openalex"]
        assert "Source DB" in bib.df.columns and bib.merge_report["n_removed"] == 3
        txt = bib.methods_text("en")
        assert "Scopus, OpenAlex" in txt and "duplicates were removed" in txt
        bib2 = BiblioStats(df=_scopus(), db="scopus")
        rep = bib2.merge_with([(_openalex(), "openalex")], key="doi")
        assert bib2.n == 6 and rep["n_removed"] == 2
