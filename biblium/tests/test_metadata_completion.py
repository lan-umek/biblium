# -*- coding: utf-8 -*-
"""Tests for ``biblium.metadata_completion`` (new in 2.19). Fully hermetic:
API responses are injected as small fake JSON dicts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biblium import metadata_completion as M


OA_WORK = {
    "id": "https://openalex.org/W1",
    "title": "A study of things",
    "publication_year": 2020,
    "primary_location": {"source": {"display_name": "Journal of Tests", "issn_l": "1234-567X"}},
    "authorships": [
        {"author": {"display_name": "Mohammad Hossein Shiri"}},
        {"author": {"display_name": "Lan Umek"}},
        {"author": {"display_name": "Jean-Pierre Dupont"}},
    ],
    "abstract_inverted_index": {"world": [1], "Hello": [0], "again": [2], "hello": [3]},
    "cited_by_count": 5,
    "type": "article",
    "biblio": {"volume": "3", "issue": "2", "first_page": "10", "last_page": "20"},
    "open_access": {"is_oa": True, "oa_status": "gold"},
    "language": "en",
    "keywords": [{"display_name": "Bibliometrics"}, {"keyword": "Testing"}],
}

CR_MSG = {
    "title": ["Another <i>study</i>"],
    "issued": {"date-parts": [[2021, 3, 1]]},
    "container-title": ["Journal of Crossref"],
    "author": [{"given": "Lan", "family": "Umek"}, {"given": "Ana Maria", "family": "Kovač"},
               {"name": "Consortium X"}],
    "ISSN": ["8765-4321", "1111-2222"],
    "is-referenced-by-count": 7,
    "type": "journal-article",
    "volume": "12",
    "journal-issue": {"issue": "4"},
    "page": "100-110",
    "abstract": "<jats:p>Abstract: Some text.</jats:p>",
    "language": "sl",
}


class TestHelpers:
    @pytest.mark.parametrize("raw, exp", [
        ("10.1000/ABC", "10.1000/abc"),
        ("https://doi.org/10.1000/abc ", "10.1000/abc"),
        ("http://dx.doi.org/10.1/x", "10.1/x"),
        ("doi:10.1/x", "10.1/x"),
        ("DOI: 10.1/x", "10.1/x"),
        ("not a doi", None),
        (None, None),
        (np.nan, None),
        ("", None),
    ])
    def test_normalize_doi(self, raw, exp):
        assert M.normalize_doi(raw) == exp

    def test_abstract_from_inverted_index(self):
        assert M._abstract_from_inverted_index({"b": [1], "a": [0, 2]}) == "a b a"
        assert M._abstract_from_inverted_index(None) is None
        assert M._abstract_from_inverted_index({}) is None


class TestAudit:
    def test_percentages_and_fillable(self):
        df = pd.DataFrame({
            "Title": ["a", "", "  ", None],
            "Year": [2000, np.nan, 2002, 2003],
            "EID": ["1", "2", "3", "4"],
        })
        a = M.audit_missing(df).set_index("Field")
        assert a.loc["Title", "N missing"] == 3 and a.loc["Title", "% missing"] == pytest.approx(75.0)
        assert a.loc["Year", "N missing"] == 1 and a.loc["Year", "% missing"] == pytest.approx(25.0)
        assert a.loc["EID", "N missing"] == 0
        assert bool(a.loc["Title", "Fillable"]) is True and bool(a.loc["EID", "Fillable"]) is False
        assert list(a.index) == ["Title", "Year", "EID"]  # sorted by % missing desc

    def test_fields_subset_and_absent_field(self):
        df = pd.DataFrame({"Title": ["a", "b"]})
        a = M.audit_missing(df, fields=["Abstract", "Title"]).set_index("Field")
        assert a.loc["Abstract", "% missing"] == 100.0 and a.loc["Title", "% missing"] == 0.0

    def test_empty_frame(self):
        a = M.audit_missing(pd.DataFrame({"Title": []}))
        assert len(a) == 1 and np.isnan(a.loc[0, "% missing"])


class TestMapping:
    def test_openalex(self):
        f = M.openalex_to_fields(OA_WORK)
        assert f["Title"] == "A study of things"
        assert f["Year"] == 2020
        assert f["Source title"] == "Journal of Tests"
        assert f["ISSN"] == "1234567X"
        assert f["Authors"] == "Shiri, M.H.; Umek, L.; Dupont, J.-P."
        assert f["Abstract"] == "Hello world again hello"
        assert f["Cited by"] == 5
        assert f["Document Type"] == "Article"
        assert (f["Volume"], f["Issue"], f["Page start"], f["Page end"]) == ("3", "2", "10", "20")
        assert f["Open Access"] == "All Open Access; Gold Open Access"
        assert f["Language of Original Document"] == "English"
        assert f["Author Keywords"] == "Bibliometrics; Testing"
        assert set(f) <= M.FILLABLE_FIELDS

    def test_openalex_sparse(self):
        f = M.openalex_to_fields({"id": "W2", "open_access": {"is_oa": False}, "authorships": []})
        assert f == {}
        assert M.openalex_to_fields(None) == {}

    def test_crossref(self):
        f = M.crossref_to_fields(CR_MSG)
        assert f["Title"] == "Another study"
        assert f["Year"] == 2021
        assert f["Source title"] == "Journal of Crossref"
        assert f["Authors"] == "Umek, L.; Kovač, A.M.; Consortium X"
        assert f["ISSN"] == "87654321"
        assert f["Cited by"] == 7
        assert f["Document Type"] == "Article"
        assert (f["Volume"], f["Issue"], f["Page start"], f["Page end"]) == ("12", "4", "100", "110")
        assert f["Abstract"] == "Some text."
        assert f["Language of Original Document"] == "Slovenian"
        assert "Open Access" not in f
        assert set(f) <= M.FILLABLE_FIELDS

    def test_crossref_year_fallback(self):
        f = M.crossref_to_fields({"issued": {"date-parts": [[None]]}, "created": {"date-parts": [[2019]]}})
        assert f["Year"] == 2019


@pytest.fixture
def frame():
    return pd.DataFrame({
        "DOI": ["10.1/a", "10.1/b", None, "10.1/d"],
        "Title": ["T1", None, None, "T4"],
        "Year": [2000.0, np.nan, np.nan, 2004.0],
        "Abstract": ["", np.nan, "x", "Has abstract"],
    })


def _fake_oa(doi):
    return OA_WORK if doi == "10.1/a" else None


def _fake_cr(doi):
    return CR_MSG if doi in ("10.1/b", "10.1/d") else None


class TestComplete:
    def test_fill_only_missing(self, frame):
        orig = frame.copy()
        out, prov = M.complete_metadata(frame, fetch_openalex=_fake_oa, fetch_crossref=_fake_cr)
        pd.testing.assert_frame_equal(frame, orig)                      # input untouched
        assert out.loc[0, "Title"] == "T1" and out.loc[0, "Year"] == 2000  # existing kept
        assert out.loc[0, "Abstract"] == "Hello world again hello"      # empty string filled
        assert out.loc[1, "Title"] == "Another study" and out.loc[1, "Year"] == 2021
        assert out.loc[1, "Abstract"] == "Some text."
        assert pd.isna(out.loc[2, "Title"])                              # no DOI → untouched
        assert out.loc[3, "Abstract"] == "Has abstract"
        assert out.loc[0, "Metadata provenance"] == "Abstract:openalex"
        assert out.loc[1, "Metadata provenance"] == "Abstract:crossref; Title:crossref; Year:crossref"
        assert pd.isna(out.loc[2, "Metadata provenance"]) and pd.isna(out.loc[3, "Metadata provenance"])
        assert list(prov.columns) == M.PROVENANCE_COLUMNS
        assert len(prov) == 4
        r = prov[(prov["Row index"] == 1) & (prov["Field"] == "Year")].iloc[0]
        assert pd.isna(r["Old value"]) and r["New value"] == 2021 and r["Source"] == "crossref" and r["DOI"] == "10.1/b"

    def test_source_order_and_fallback(self, frame):
        calls = []

        def oa(doi):
            calls.append(("oa", doi))
            return _fake_oa(doi)

        def cr(doi):
            calls.append(("cr", doi))
            return _fake_cr(doi)

        out, prov = M.complete_metadata(frame, fields=["Title", "Year", "Abstract"],
                                        fetch_openalex=oa, fetch_crossref=cr)
        # row 0: openalex fills everything missing → crossref not called for it
        assert ("cr", "10.1/a") not in calls and ("oa", "10.1/a") in calls
        # row 1: openalex returns None → crossref used
        assert ("oa", "10.1/b") in calls and ("cr", "10.1/b") in calls
        # row 3: nothing missing → no call at all
        assert all(d != "10.1/d" for _, d in calls)
        assert set(prov["Source"]) == {"openalex", "crossref"}

    def test_crossref_only(self, frame):
        out, prov = M.complete_metadata(frame, sources=("crossref",), fetch_openalex=_fake_oa,
                                        fetch_crossref=_fake_cr)
        assert pd.isna(out.loc[0, "Abstract"]) or out.loc[0, "Abstract"] == ""
        assert set(prov["Source"]) == {"crossref"}

    def test_overwrite(self, frame):
        out, prov = M.complete_metadata(frame, fields=["Title", "Year"], overwrite=True,
                                        fetch_openalex=_fake_oa, fetch_crossref=_fake_cr)
        assert out.loc[0, "Title"] == "A study of things" and out.loc[0, "Year"] == 2020
        r = prov[(prov["Row index"] == 0) & (prov["Field"] == "Title")].iloc[0]
        assert r["Old value"] == "T1" and r["New value"] == "A study of things"
        # identical value is not logged as a change
        assert out.loc[3, "Title"] == "Another study"
        assert len(prov[(prov["Row index"] == 3) & (prov["Field"] == "Year")]) == 1  # 2004 → 2021

    def test_new_column_created_and_max_rows(self, frame):
        out, prov = M.complete_metadata(frame, fields=["Source title"], max_rows=1,
                                        fetch_openalex=_fake_oa, fetch_crossref=_fake_cr)
        assert "Source title" in out.columns
        assert out.loc[0, "Source title"] == "Journal of Tests"
        assert pd.isna(out.loc[1, "Source title"])
        assert len(prov) == 1

    def test_undo_roundtrip(self, frame):
        out, prov = M.complete_metadata(frame, fetch_openalex=_fake_oa, fetch_crossref=_fake_cr)
        back = M.undo_completion(out, prov)
        for c in ("Title", "Year", "Abstract"):
            pd.testing.assert_series_equal(back[c], frame[c], check_dtype=False)
        assert back["Metadata provenance"].isna().all()

    def test_undo_partial_keeps_other_tags(self, frame):
        out, prov = M.complete_metadata(frame, fetch_openalex=_fake_oa, fetch_crossref=_fake_cr)
        back = M.undo_completion(out, prov[prov["Field"] == "Year"])
        assert pd.isna(back.loc[1, "Year"]) and back.loc[1, "Title"] == "Another study"
        assert back.loc[1, "Metadata provenance"] == "Abstract:crossref; Title:crossref"

    def test_summary(self, frame):
        _, prov = M.complete_metadata(frame, fetch_openalex=_fake_oa, fetch_crossref=_fake_cr)
        s = M.completion_summary(prov)
        assert s.loc["Abstract", "openalex"] == 1 and s.loc["Abstract", "crossref"] == 1
        assert s.loc["Title", "Total"] == 1
        assert M.completion_summary(prov.iloc[0:0]).empty

    def test_no_doi_column(self):
        df = pd.DataFrame({"Title": [None]})
        out, prov = M.complete_metadata(df, fetch_openalex=_fake_oa, fetch_crossref=_fake_cr)
        assert len(prov) == 0 and "Metadata provenance" in out.columns

    def test_unknown_source(self, frame):
        with pytest.raises(ValueError):
            M.complete_metadata(frame, sources=("scopus",), fetch_openalex=_fake_oa, fetch_crossref=_fake_cr)
