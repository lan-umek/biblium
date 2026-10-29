# -*- coding: utf-8 -*-
"""
Tests for ``biblium.snowball`` (new in 2.19).

A tiny fake OpenAlex graph is served through an injected ``fetch_json``;
no network access. Expectations are hand-computed.
"""
from __future__ import annotations

import re
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pytest

from biblium import snowball as S


# ---------------------------------------------------------------------------
# Fake OpenAlex
# ---------------------------------------------------------------------------

def _work(wid, year, refs=(), doi=None, title=None, source="Journal X", cited=3, authors=("Jane Q. Smith", "Bob Doe")):
    return {
        "id": f"https://openalex.org/{wid}",
        "doi": f"https://doi.org/{doi}" if doi else None,
        "display_name": title or f"Title {wid}",
        "publication_year": year,
        "cited_by_count": cited,
        "type": "article",
        "primary_location": {"source": {"display_name": source}},
        "authorships": [{"author": {"display_name": a}} for a in authors],
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
    }


# Graph:
#   seeds S1, S2
#   backward: R1 cited by S1 and S2; R2 cited by S1 only (old: 1990)
#   forward:  C1 cites S1 and S2; C2 cites S2 only
#   round 2:  R3 cited by R1; C3 cites C1
GRAPH = {
    "S1": _work("S1", 2015, refs=["R1", "R2"], doi="10.1000/s1"),
    "S2": _work("S2", 2016, refs=["R1"], doi="10.1000/s2"),
    "R1": _work("R1", 2010, refs=["R3"]),
    "R2": _work("R2", 1990),
    "R3": _work("R3", 2005),
    "C1": _work("C1", 2019, refs=["S1", "S2"]),
    "C2": _work("C2", 2020, refs=["S2"]),
    "C3": _work("C3", 2021, refs=["C1"]),
}
DOIS = {"10.1000/s1": "S1", "10.1000/s2": "S2"}


class FakeFetcher:
    """Maps OpenAlex URLs to dicts; records the URLs requested."""

    def __init__(self, graph=GRAPH, page_size=None, fail=()):
        self.graph = graph
        self.page_size = page_size  # force pagination of citing lists
        self.fail = set(fail)
        self.calls = []

    def _citing(self, wid):
        return sorted(k for k, w in self.graph.items()
                      if f"https://openalex.org/{wid}" in w["referenced_works"])

    def __call__(self, url):
        self.calls.append(url)
        url = unquote(url)
        m = re.match(r"https://api\.openalex\.org/works/https://doi\.org/(.+)$", url)
        if m:
            wid = DOIS.get(m.group(1))
            return self.graph.get(wid) if wid and wid not in self.fail else None
        m = re.match(r"https://api\.openalex\.org/works/(W?[A-Z]\d+)$", url)
        if m:
            wid = m.group(1)
            return None if wid in self.fail else self.graph.get(wid)
        m = re.match(r"https://api\.openalex\.org/works\?filter=cites:([A-Z]\d+)&per-page=(\d+)&cursor=(.+)$", url)
        if m:
            wid, cursor = m.group(1), m.group(3)
            ids = self._citing(wid)
            if self.page_size:
                start = 0 if cursor == "*" else int(cursor)
                chunk = ids[start:start + self.page_size]
                nxt = str(start + self.page_size) if start + self.page_size < len(ids) else None
            else:
                chunk, nxt = ids, None
            return {"results": [self.graph[i] for i in chunk], "meta": {"next_cursor": nxt}}
        m = re.match(r"https://api\.openalex\.org/works\?filter=openalex:([^&]+)&per-page=(\d+)$", url)
        if m:
            ids = m.group(1).split("|")
            return {"results": [self.graph[i] for i in ids if i in self.graph and i not in self.fail]}
        raise AssertionError(f"unexpected url {url}")


@pytest.fixture
def fetch():
    return FakeFetcher()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_normalize_openalex_id(self):
        assert S.normalize_openalex_id("https://openalex.org/W123") == "W123"
        assert S.normalize_openalex_id("W123") == "W123"
        assert S.normalize_openalex_id("w123") == "W123"
        assert S.normalize_openalex_id("https://openalex.org/works/W7") == "W7"
        assert S.normalize_openalex_id("10.1000/abc") is None
        assert S.normalize_openalex_id(None) is None
        assert S.normalize_openalex_id(np.nan) is None
        assert S.normalize_openalex_id("") is None

    def test_resolve_seeds_ids_and_dois(self, fetch):
        works = S.resolve_seeds(["S1", "https://doi.org/10.1000/s2", "doi:10.1000/s1", "nonsense", "W999"], fetch)
        assert [S.normalize_openalex_id(w["id"]) for w in works] == ["S1", "S2"]  # deduplicated, failures skipped
        assert any(u.endswith("/works/https://doi.org/10.1000/s2") for u in fetch.calls)
        assert any(u.endswith("/works/S1") for u in fetch.calls)

    def test_fetch_references(self, fetch):
        assert S.fetch_references("S1", fetch) == ["R1", "R2"]
        assert S.fetch_references(GRAPH["S2"], fetch) == ["R1"]
        assert S.fetch_references("W999", fetch) == []

    def test_fetch_citing_with_pagination(self):
        f = FakeFetcher(page_size=1)
        res = S.fetch_citing("S2", f)
        assert sorted(S.normalize_openalex_id(w["id"]) for w in res) == ["C1", "C2"]
        pages = [u for u in f.calls if "cites:S2" in u]
        assert len(pages) == 2 and "cursor=*" in pages[0] and "cursor=1" in pages[1]
        # max_pages cap
        f2 = FakeFetcher(page_size=1)
        assert len(S.fetch_citing("S2", f2, max_pages=1)) == 1

    def test_fetch_works_batch(self, fetch):
        res = S.fetch_works_batch(["R1", "https://openalex.org/C2", "R1", "W999"], fetch, batch=1)
        assert [S.normalize_openalex_id(w["id"]) for w in res] == ["R1", "C2"]
        assert sum("filter=openalex:" in u for u in fetch.calls) == 3

    def test_default_fetch_json_mailto(self):
        assert S._with_mailto("https://x/y", "a@b.c") == "https://x/y?mailto=a@b.c"
        assert S._with_mailto("https://x/y?q=1", "a@b.c") == "https://x/y?q=1&mailto=a@b.c"
        assert S._with_mailto("https://x/y", None) == "https://x/y"


# ---------------------------------------------------------------------------
# Snowball
# ---------------------------------------------------------------------------

class TestSnowball:
    def test_min_links_two(self, fetch):
        df, st = S.snowball(["S1", "S2"], min_links=2, fetch_json=fetch)
        d = df.set_index("OpenAlex ID")
        assert sorted(d.index) == ["C1", "R1", "S1", "S2"]
        assert d.loc["R1", "Direction"] == "backward" and d.loc["R1", "Links to seed set"] == 2
        assert d.loc["C1", "Direction"] == "forward" and d.loc["C1", "Links to seed set"] == 2
        assert (d.loc[["S1", "S2"], "Round"] == 0).all() and (d.loc[["S1", "S2"], "Direction"] == "seed").all()
        assert (d.loc[["R1", "C1"], "Round"] == 1).all()
        assert st["n_seeds_resolved"] == 2 and st["n_total"] == 4
        assert st["per_round"][0]["n_candidates"] == 4 and st["per_round"][0]["n_added"] == 2
        assert st["per_round"][0]["backward"] == 1 and st["per_round"][0]["forward"] == 1

    def test_min_links_one(self, fetch):
        df, st = S.snowball(["S1", "S2"], min_links=1, fetch_json=fetch)
        d = df.set_index("OpenAlex ID")
        assert sorted(d.index) == ["C1", "C2", "R1", "R2", "S1", "S2"]
        assert d.loc["R2", "Links to seed set"] == 1 and d.loc["C2", "Links to seed set"] == 1
        assert list(df.columns) == S.SNOWBALL_COLUMNS
        assert d.loc["S1", "DOI"] == "10.1000/s1"
        assert d.loc["S1", "Authors"] == "Smith, J.Q.; Doe, B."
        assert d.loc["S1", "Source title"] == "Journal X"

    def test_direction_backward_only(self, fetch):
        df, _ = S.snowball(["S1", "S2"], direction="backward", fetch_json=fetch)
        assert sorted(df["OpenAlex ID"]) == ["R1", "R2", "S1", "S2"]
        assert not any("cites:" in u for u in fetch.calls)

    def test_direction_forward_only(self, fetch):
        df, _ = S.snowball(["S1", "S2"], direction="forward", fetch_json=fetch)
        assert sorted(df["OpenAlex ID"]) == ["C1", "C2", "S1", "S2"]

    def test_rounds_two_extends(self, fetch):
        df, st = S.snowball(["S1", "S2"], rounds=2, min_links=1, fetch_json=fetch)
        d = df.set_index("OpenAlex ID")
        assert sorted(d.index) == ["C1", "C2", "C3", "R1", "R2", "R3", "S1", "S2"]
        assert d.loc["R3", "Round"] == 2 and d.loc["R3", "Direction"] == "backward"
        assert d.loc["C3", "Round"] == 2 and d.loc["C3", "Direction"] == "forward"
        assert len(st["per_round"]) == 2 and st["per_round"][1]["n_added"] == 2
        # seeds are never re-added in round 2 even though C1 cites them
        assert d.loc["S1", "Round"] == 0 and df["OpenAlex ID"].is_unique

    def test_year_filter(self, fetch):
        df, st = S.snowball(["S1", "S2"], min_links=1, fetch_json=fetch, year_min=2000, year_max=2019)
        ids = sorted(df["OpenAlex ID"])
        assert "R2" not in ids and "C2" not in ids  # 1990 and 2020 dropped
        assert ids == ["C1", "R1", "S1", "S2"]
        assert st["per_round"][0]["n_year_dropped"] == 2

    def test_max_candidates(self, fetch):
        df, _ = S.snowball(["S1", "S2"], min_links=1, max_candidates=2, fetch_json=fetch)
        # the two best-linked candidates (2 links each) are kept
        assert sorted(df["OpenAlex ID"]) == ["C1", "R1", "S1", "S2"]

    def test_unresolvable_seeds(self, fetch):
        df, st = S.snowball(["W999"], fetch_json=fetch)
        assert df.empty and st["n_seeds_resolved"] == 0 and st["per_round"] == []

    def test_bad_arguments(self, fetch):
        with pytest.raises(ValueError):
            S.snowball(["S1"], direction="sideways", fetch_json=fetch)
        with pytest.raises(ValueError):
            S.snowball(["S1"], rounds=0, fetch_json=fetch)

    def test_summary_and_plot(self, fetch, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df, _ = S.snowball(["S1", "S2"], rounds=2, min_links=1, fetch_json=fetch)
        tab = S.expansion_summary(df)
        assert tab.loc[0, "seed"] == 2 and tab.loc[0, "Total"] == 2
        assert tab.loc[1, "backward"] == 2 and tab.loc[1, "forward"] == 2
        assert tab.loc[2, "Total"] == 2
        out = tmp_path / "exp.png"
        fig, ax = S.plot_expansion(df, f_name=str(out), dpi=50)
        assert out.exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Mapping to biblium columns
# ---------------------------------------------------------------------------

class TestWorksToBiblium:
    def test_mapping(self):
        w = {
            "id": "https://openalex.org/W42",
            "doi": "https://doi.org/10.1234/abc",
            "display_name": "A study",
            "publication_year": 2021,
            "cited_by_count": 7,
            "type": "article",
            "primary_location": {"source": {"display_name": "Scientometrics"}},
            "authorships": [
                {"author": {"display_name": "Jane Q. Smith"}},
                {"author": {"display_name": "Novak, Ana Maria"}},
                {"author": {"display_name": "Plato"}},
            ],
            "abstract_inverted_index": {"world": [1], "Hello": [0], "again": [2, 3]},
            "keywords": [{"display_name": "Bibliometrics"}, {"display_name": "Scopus"}],
            "concepts": [{"display_name": "Should not be used"}],
            "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
        }
        df = S.works_to_biblium_df([w])
        r = df.iloc[0]
        assert r["Title"] == "A study" and r["Year"] == 2021 and r["Cited by"] == 7
        assert r["DOI"] == "10.1234/abc"
        assert r["Source title"] == "Scientometrics"
        assert r["Authors"] == "Smith, J.Q.; Novak, A.M.; Plato"
        assert r["Abstract"] == "Hello world again again"
        assert r["Author Keywords"] == "Bibliometrics; Scopus"
        assert r["Document Type"] == "article"
        assert r["OpenAlex ID"] == "W42"
        assert r["References"] == "W1; W2"

    def test_mapping_fallbacks(self):
        w = {"id": "https://openalex.org/W5", "title": "T", "publication_date": "1999-05-01",
             "concepts": [{"display_name": "Physics"}]}
        r = S.works_to_biblium_df([w]).iloc[0]
        assert r["Title"] == "T" and r["Year"] == 1999 and r["Cited by"] == 0
        assert r["Author Keywords"] == "Physics" and r["Abstract"] is None
        assert r["References"] is None and r["Source title"] is None and r["Authors"] == ""

    def test_empty(self):
        df = S.works_to_biblium_df([])
        assert df.empty and "OpenAlex ID" in df.columns
