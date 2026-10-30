# -*- coding: utf-8 -*-
"""Tests for ``biblium.datasets`` (domain demo datasets fetched from OpenAlex, 2.19.1)."""
from __future__ import annotations

import gzip
import json
import os

import pandas as pd
import pytest

from biblium import datasets as D


def _fake_work(i: int) -> dict:
    return {
        "id": f"https://openalex.org/W{i}",
        "doi": f"https://doi.org/10.1000/w{i}",
        "title": f"Work number {i}",
        "display_name": f"Work number {i}",
        "publication_year": 2015 + (i % 5),
        "publication_date": f"{2015 + (i % 5)}-01-01",
        "language": "en", "type": "article", "type_crossref": "journal-article",
        "indexed_in": ["crossref", "openalex"],
        "cited_by_count": 10 * i, "fwci": 1.5, "is_retracted": False,
        "referenced_works_count": 2,
        "referenced_works": [f"https://openalex.org/W{i + 100}", f"https://openalex.org/W{i + 200}"],
        "ids": {"openalex": f"https://openalex.org/W{i}", "doi": f"https://doi.org/10.1000/w{i}"},
        "primary_location": {"is_oa": True, "landing_page_url": "http://x",
                             "source": {"id": "https://openalex.org/S1", "display_name": "Public Administration Review",
                                        "issn_l": "0033-3352", "issn": ["0033-3352", "1540-6210"], "type": "journal"}},
        "open_access": {"is_oa": True, "oa_status": "gold"},
        "biblio": {"volume": "1", "issue": "2", "first_page": "3", "last_page": "9"},
        "primary_topic": {"id": "T1", "display_name": "E-government", "score": 0.9,
                          "subfield": {"display_name": "Public Administration"},
                          "field": {"display_name": "Social Sciences"}, "domain": {"display_name": "Social Sciences"}},
        "topics": [{"id": "T1", "display_name": "E-government", "score": 0.9,
                    "subfield": {"display_name": "Public Administration"},
                    "field": {"display_name": "Social Sciences"}, "domain": {"display_name": "Social Sciences"}},
                   {"id": "T2", "display_name": "Digital transformation", "score": 0.5,
                    "subfield": {"display_name": "Information Systems"},
                    "field": {"display_name": "Computer Science"}, "domain": {"display_name": "Physical Sciences"}}],
        "keywords": [{"id": "k1", "display_name": "E-government", "score": 0.7},
                     {"id": "k2", "display_name": "Trust", "score": 0.4}],
        "concepts": [{"id": "c1", "display_name": "Political science", "level": 0, "score": 0.8}],
        "authorships": [
            {"author_position": "first", "author": {"id": "A1", "display_name": "Ana Novak", "orcid": None},
             "raw_author_name": "Novak, Ana", "countries": ["SI"], "is_corresponding": True,
             "raw_affiliation_strings": ["University of Ljubljana, Faculty of Public Administration"],
             "institutions": [{"display_name": "University of Ljubljana", "country_code": "SI"}]},
            {"author_position": "last", "author": {"id": "A2", "display_name": "John Doe", "orcid": "0000-0001"},
             "raw_author_name": "Doe, John", "countries": ["US"], "is_corresponding": False,
             "raw_affiliation_strings": ["Harvard University"],
             "institutions": [{"display_name": "Harvard University", "country_code": "US"}]},
        ],
        "grants": [{"funder": "F1", "funder_display_name": "ARIS", "award_id": "P5-0093"}],
        "counts_by_year": [{"year": 2024, "cited_by_count": 3}, {"year": 2023, "cited_by_count": 7}],
        "abstract_inverted_index": {"Digital": [0], "government": [1, 4], "improves": [2], "local": [3]},
    }


def _fake_fetch_json_factory(n_total=7):
    """Returns a fake ``fetch_json`` that pages through ``n_total`` works (page size from URL)."""
    from urllib.parse import parse_qs, urlparse
    calls = []

    def fetch_json(url):
        calls.append(url)
        q = parse_qs(urlparse(url).query)
        per = int(q.get("per_page", q.get("per-page"))[0])
        if "sample" in q:  # random sampling uses page numbers
            start = (int(q["page"][0]) - 1) * per
            nxt = None
        else:
            cursor = q["cursor"][0]
            start = 0 if cursor == "*" else int(cursor)
            nxt = str(start + per) if start + per < n_total else None
        res = [_fake_work(i) for i in range(start, min(start + per, n_total))]
        return {"meta": {"next_cursor": nxt, "count": n_total}, "results": res}

    fetch_json.calls = calls
    return fetch_json


class TestCatalogue:
    def test_catalogue_is_consistent(self):
        assert len(D.DATASETS) >= 40
        assert "Public administration" in D.DOMAINS and "Medicine" in D.DOMAINS and "Law" in D.DOMAINS
        names = [s.name for s in D.DATASETS.values()]
        assert len(names) == len(set(names))
        for s in D.DATASETS.values():
            assert s.kind in ("topic", "journal", "country")
            fs = s.filter_string()
            assert "type:article" in fs and "publication_year:" in fs
            if s.kind == "journal":
                assert "primary_location.source.issn:" in fs and s.search is None
                assert "has_abstract" not in fs  # publishers often do not deposit abstracts
            else:
                assert s.search and "has_abstract:true" in fs
        tab = D.list_datasets("Law")
        assert set(tab["domain"]) == {"Law"} and {"name", "title", "bundled"} <= set(tab.columns)

    def test_journal_counts(self):
        assert sum(s.kind == "journal" for s in D.DATASETS.values()) >= 10
        pa = [s for s in D.DATASETS.values() if s.domain == "Public administration"]
        assert len(pa) >= 8


class TestFlatten:
    def test_flatten_work_layout(self):
        rec = D.flatten_work(_fake_work(1))
        assert rec["authorships.raw_author_name"] == "Novak, Ana|Doe, John"
        assert rec["authorships.countries"] == "SI|US"
        assert rec["keywords.display_name"] == "E-government|Trust"
        assert rec["topics.display_name"] == "E-government|Digital transformation"
        assert rec["referenced_works"].count("|") == 1
        assert rec["abstract"] == "Digital government improves local government"
        assert rec["primary_location.source.display_name"] == "Public Administration Review"
        assert rec["counts_by_year.year"] == "2024|2023"
        assert rec["grants.funder_display_name"] == "ARIS"
        assert rec["cited_by_count"] == 10

    def test_flatten_handles_missing_blocks(self):
        rec = D.flatten_work({"id": "W0", "title": "bare"})
        assert rec["abstract"] is None and rec["authorships.raw_author_name"] is None
        assert rec["primary_location.source.display_name"] is None


class TestFetch:
    def test_fetch_works_paginates(self):
        spec = D.DATASETS["pa_egovernment"]
        fj = _fake_fetch_json_factory(n_total=7)
        works = D.fetch_works(spec, email="a@b.si", n=5, per_page=2, fetch_json=fj, delay=0)
        assert len(works) == 5 and len(fj.calls) == 3
        assert "mailto=a%40b.si" in fj.calls[0] and "search=" in fj.calls[0]
        assert "filter=type%3Aarticle" in fj.calls[0] and "sort=cited_by_count%3Adesc" in fj.calls[0]

    def test_fetch_works_sort_options(self):
        spec = D.DATASETS["pa_egovernment"]
        fj = _fake_fetch_json_factory(n_total=7)
        D.fetch_works(spec, n=3, fetch_json=fj, delay=0, sort="newest")
        assert "sort=publication_date%3Adesc" in fj.calls[-1]
        D.fetch_works(spec, n=3, fetch_json=fj, delay=0, sort="Oldest first")
        assert "sort=publication_date%3Aasc" in fj.calls[-1]
        works = D.fetch_works(spec, n=5, per_page=2, fetch_json=fj, delay=0, sort="random", seed=7)
        assert len(works) == 5 and "sample=5" in fj.calls[-1] and "seed=7" in fj.calls[-1] and "page=3" in fj.calls[-1]
        # relevance without a search string falls back to citations (journal specs have no search)
        D.fetch_works(D.DATASETS["j_jpart"], n=2, fetch_json=fj, delay=0, sort="relevance")
        assert "sort=cited_by_count%3Adesc" in fj.calls[-1]

    def test_fetch_dataset_round_trip(self, tmp_path):
        fj = _fake_fetch_json_factory(n_total=6)
        path = D.fetch_dataset("pa_egovernment", email="a@b.si", out_dir=str(tmp_path), n=6, fetch_json=fj,
                               verbose=False)
        assert path.endswith("pa_egovernment.csv.gz") and os.path.exists(path)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            header = fh.readline()
        assert "authorships.raw_author_name" in header and "abstract" in header
        # skip when exists
        assert D.fetch_dataset("pa_egovernment", out_dir=str(tmp_path), fetch_json=fj, verbose=False) == path
        # read through the standard OpenAlex reader -> canonical column names
        df = D.load_dataset_df("pa_egovernment", data_dir=str(tmp_path))
        for col in ["Title", "Authors", "Year", "Cited by", "Abstract", "Author Keywords", "DOI", "Source title",
                    "References"]:
            assert col in df.columns, col
        assert len(df) == 6 and df["Cited by"].max() == 50
        assert df.loc[0, "Authors"] == "Novak, Ana|Doe, John"

    def test_load_dataset_bibliostats(self, tmp_path):
        pytest.importorskip("biblium.bibstats")
        fj = _fake_fetch_json_factory(n_total=6)
        D.fetch_dataset("j_jpart", out_dir=str(tmp_path), n=6, fetch_json=fj, verbose=False)
        bib = D.load_dataset("j_jpart", data_dir=str(tmp_path), verbose=False)
        assert bib.n == 6 and bib.dataset_name == "j_jpart" and bib.dataset_spec.kind == "journal"
        assert bib.db == "oa"
        with pytest.raises(FileNotFoundError):
            D.load_dataset_df("pa_npm_reform", data_dir=str(tmp_path))
        with pytest.raises(KeyError):
            D.load_dataset_df("nope", data_dir=str(tmp_path))

    def test_fetch_all_reports_failures(self, tmp_path, capsys):
        def bad(url):
            raise RuntimeError("offline")
        out = D.fetch_all(out_dir=str(tmp_path), names=["pa_egovernment"], fetch_json=bad, verbose=False)
        assert out == {} and "[fail] pa_egovernment" in capsys.readouterr().out

    def test_cli_list(self, capsys):
        assert D._main(["--list", "--domain", "Mathematics"]) == 0
        out = capsys.readouterr().out
        assert "math_optimal_transport" in out and "pa_egovernment" not in out


def test_methods_text_mentions_dataset(tmp_path):
    pytest.importorskip("biblium.bibstats")
    fj = _fake_fetch_json_factory(n_total=6)
    D.fetch_dataset("pa_egovernment", out_dir=str(tmp_path), n=6, fetch_json=fj, verbose=False)
    bib = D.load_dataset("pa_egovernment", data_dir=str(tmp_path))
    en, sl = bib.methods_text("en"), bib.methods_text("sl")
    assert "pa_egovernment" in en and "OpenAlex API" in en and "search string" in en
    assert "pa_egovernment" in sl and "Public administration" in sl


def test_sort_helpers():
    from biblium.openalex_api import resolve_sort, paging_params, sort_description, SORT_OPTIONS, SORT_LABELS
    assert set(SORT_OPTIONS) == set(SORT_LABELS)
    assert resolve_sort(None) == "cited_by_count:desc"
    assert resolve_sort("Newest first") == "publication_date:desc" and resolve_sort("least_cited") == "cited_by_count:asc"
    assert resolve_sort("relevance", has_search=True) == "relevance_score:desc"
    assert resolve_sort("relevance", has_search=False) == "cited_by_count:desc"
    assert resolve_sort("fwci:desc") == "fwci:desc" and resolve_sort("random") == "random"
    assert paging_params("random", 500, 0, None, 1) == {"sample": 500, "seed": 42, "per_page": 200, "page": 1}
    assert paging_params("publication_date:asc", 50, 0, None, 1) == {"sort": "publication_date:asc", "per_page": 50, "cursor": "*"}
    assert "random" in sort_description("Random sample") and "naključni" in sort_description("random", "sl")
