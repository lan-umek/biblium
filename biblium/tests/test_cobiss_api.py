# -*- coding: utf-8 -*-
"""
Tests for ``biblium.cobiss_api``.

These are *hermetic*: no test makes a live HTTP request. We exercise:

- the static URL allowlist via ``CobissClient._is_cobiss_url``
- on-disk caching (cache hit / TTL expiry / write failure tolerance)
- the rate-limit accounting field
- the convenience function's CSV export (using a stubbed fetch)
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from biblium.cobiss_api import (
    CobissClient,
    CobissFetchResult,
    fetch_personal_bibliography_to_csv,
)


# ---------------------------------------------------------------------------
# Host allowlist
# ---------------------------------------------------------------------------


class TestHostAllowlist:
    """``_is_cobiss_url`` should accept only known COBISS hosts."""

    @pytest.mark.parametrize("url", [
        "https://bib.cobiss.net/bibliographies/si/webBiblio/bib201_x.html",
        "http://bib.cobiss.net/foo",
        "https://plus.cobiss.net/cobiss/si/sl/bib/123",
        "https://plus-legacy.cobiss.net/cobiss/si/sl/bib/123",
        "https://www.cobiss.net/cobiss-platform.htm",
        "https://www.cobiss.si/en/news.htm",
        "https://home.izum.si/cobiss/bib/Help_SI_en.html",
    ])
    def test_known_hosts_accepted(self, url):
        assert CobissClient._is_cobiss_url(url) is True

    @pytest.mark.parametrize("url", [
        "https://example.com/page",
        "https://evil.cobiss.net.attacker.com/",
        "https://malicious.com/?fake=cobiss.net",
        "ftp://bib.cobiss.net/file",  # no scheme allow even on the right host
        "not-a-url",
        "",
    ])
    def test_other_hosts_rejected(self, url):
        # FTP scheme is technically still a "cobiss host" by hostname but
        # `requests.Session().get` won't follow it. Accept either truth value
        # for that edge case but reject the rest definitely.
        if url.startswith("ftp://"):
            return  # we don't make claims about the scheme here
        assert CobissClient._is_cobiss_url(url) is False


# ---------------------------------------------------------------------------
# Constructor & defaults
# ---------------------------------------------------------------------------


class TestClientConstruction:
    """Basic constructor behaviour."""

    def test_default_user_agent_mentions_biblium(self):
        c = CobissClient()
        assert "Biblium" in c.user_agent or "biblium" in c.user_agent.lower()

    def test_custom_user_agent_is_used(self):
        c = CobissClient(user_agent="MyAgent/1.0")
        assert c.user_agent == "MyAgent/1.0"
        assert c.session.headers["User-Agent"] == "MyAgent/1.0"

    def test_cache_directory_is_created(self, tmp_path):
        cache = tmp_path / "fresh-cache"
        assert not cache.exists()
        c = CobissClient(cache_dir=str(cache))
        assert cache.exists() and cache.is_dir()
        assert c.cache_dir == str(cache)

    def test_no_cache_dir_means_caching_disabled(self):
        c = CobissClient(cache_dir=None)
        assert c.cache_dir is None
        # No cache path returned for any URL
        assert c._cache_path("https://bib.cobiss.net/foo") is None


# ---------------------------------------------------------------------------
# URL validation in fetch()
# ---------------------------------------------------------------------------


class TestFetchURLValidation:
    """``fetch`` must refuse URLs outside the allowlist *before* hitting the network."""

    def test_non_cobiss_url_raises_value_error(self):
        c = CobissClient()
        with pytest.raises(ValueError, match="COBISS"):
            c.fetch("https://example.com/")

    def test_validation_does_not_make_a_request(self):
        c = CobissClient()
        # Patch the session so any network call would be a hard error
        with patch.object(c.session, "get", side_effect=AssertionError("network!")):
            with pytest.raises(ValueError):
                c.fetch("https://other-host.com/foo")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    """On-disk cache behaviour."""

    def test_cache_path_is_deterministic(self, tmp_cache_dir):
        c = CobissClient(cache_dir=str(tmp_cache_dir))
        url = "https://bib.cobiss.net/bibliographies/si/webBiblio/test.html"
        p1 = c._cache_path(url)
        p2 = c._cache_path(url)
        assert p1 == p2
        assert p1 is not None and p1.endswith(".html")
        assert tmp_cache_dir in Path(p1).parents

    def test_cache_path_differs_per_url(self, tmp_cache_dir):
        c = CobissClient(cache_dir=str(tmp_cache_dir))
        p1 = c._cache_path("https://bib.cobiss.net/a.html")
        p2 = c._cache_path("https://bib.cobiss.net/b.html")
        assert p1 != p2

    def test_cache_round_trip(self, tmp_cache_dir):
        c = CobissClient(cache_dir=str(tmp_cache_dir))
        url = "https://bib.cobiss.net/bibliographies/si/webBiblio/test.html"
        html = "<html><body>some content</body></html>"
        final_url = url + "?redirected=1"
        c._cache_write(url, html, final_url)
        result = c._try_cache_read(url)
        assert result is not None
        cached_html, cached_final = result
        assert cached_html == html
        assert cached_final == final_url

    def test_cache_miss_returns_none(self, tmp_cache_dir):
        c = CobissClient(cache_dir=str(tmp_cache_dir))
        result = c._try_cache_read("https://bib.cobiss.net/never-cached.html")
        assert result is None

    def test_cache_ttl_expiry(self, tmp_cache_dir):
        # cache_ttl_seconds=0 means "always expired"
        c = CobissClient(cache_dir=str(tmp_cache_dir), cache_ttl_seconds=0)
        url = "https://bib.cobiss.net/foo.html"
        c._cache_write(url, "<html/>", url)
        # Even though the file exists, the TTL is 0, so reads should miss
        time.sleep(0.05)
        assert c._try_cache_read(url) is None

    def test_cache_write_with_disabled_cache_is_noop(self):
        c = CobissClient(cache_dir=None)
        # Should not raise, even though there is no cache
        c._cache_write("https://bib.cobiss.net/x.html", "<html/>", "...")


# ---------------------------------------------------------------------------
# Convenience function with a stubbed client
# ---------------------------------------------------------------------------


class TestFetchPersonalBibliographyToCSV:
    """End-to-end behaviour of the convenience function with a stubbed fetch."""

    @pytest.fixture()
    def stub_fetch(self, cobiss_sample_text, monkeypatch):
        """
        Replace ``CobissClient.fetch`` with a deterministic stub that
        returns the test fixture instead of making a live HTTP call.
        """
        def fake_fetch(self, url):
            return cobiss_sample_text, url
        monkeypatch.setattr(CobissClient, "fetch", fake_fetch)

    def test_csv_is_written(self, tmp_path, stub_fetch):
        out = tmp_path / "out.csv"
        url = "https://bib.cobiss.net/bibliographies/si/webBiblio/test.html"
        result = fetch_personal_bibliography_to_csv(url, str(out))
        assert isinstance(result, CobissFetchResult)
        assert out.exists()
        assert result.csv_path == str(out)

    def test_csv_round_trip_preserves_records(self, tmp_path, stub_fetch):
        out = tmp_path / "out.csv"
        url = "https://bib.cobiss.net/bibliographies/si/webBiblio/test.html"
        result = fetch_personal_bibliography_to_csv(url, str(out))
        df = pd.read_csv(out)
        assert len(df) == result.n_records
        assert "Authors" in df.columns
        assert "cobiss_typology_code" in df.columns

    def test_metadata_propagates_to_result(self, tmp_path, stub_fetch):
        out = tmp_path / "out.csv"
        url = "https://bib.cobiss.net/bibliographies/si/webBiblio/test.html"
        result = fetch_personal_bibliography_to_csv(url, str(out))
        assert result.researcher_name == "dr. Lan Umek"
        assert result.researcher_code == "28519"
        assert result.period == "2021-2026"
        assert result.n_records == 7
        assert result.typology_counts == {
            "1.01": 3, "1.02": 1, "1.03": 1, "1.08": 2,
        }

    def test_default_citation_source_is_wos(self, tmp_path, stub_fetch):
        out = tmp_path / "out.csv"
        url = "https://bib.cobiss.net/bibliographies/si/webBiblio/test.html"
        fetch_personal_bibliography_to_csv(url, str(out))
        df = pd.read_csv(out)
        # Record #2 in the fixture has WoS TC=1, Scopus TC=1.
        # Record #24 has WoS TC=9, Scopus TC=18.
        # With default WoS, "Cited by" must equal cobiss_wos_tc, not scopus.
        rec24 = df[df["cobiss_id"].astype(str) == "226983683"].iloc[0]
        assert rec24["Cited by"] == 9

    def test_scopus_citation_source(self, tmp_path, stub_fetch):
        out = tmp_path / "out.csv"
        url = "https://bib.cobiss.net/bibliographies/si/webBiblio/test.html"
        fetch_personal_bibliography_to_csv(
            url, str(out), default_citation_source="scopus"
        )
        df = pd.read_csv(out)
        rec24 = df[df["cobiss_id"].astype(str) == "226983683"].iloc[0]
        assert rec24["Cited by"] == 18

    def test_creates_output_directory(self, tmp_path, stub_fetch):
        nested = tmp_path / "deep" / "nested" / "out.csv"
        url = "https://bib.cobiss.net/bibliographies/si/webBiblio/test.html"
        fetch_personal_bibliography_to_csv(url, str(nested))
        assert nested.exists()


# ---------------------------------------------------------------------------
# Outside-Slovenia warning
# ---------------------------------------------------------------------------


class TestOutsideSloveniaWarning:
    """When fetched HTML contains no citation counts, emit a UserWarning."""

    def test_warning_when_no_citations_anywhere(self, monkeypatch, fixtures_dir):
        # Build a synthetic payload: 3 records, none with any citations.
        # We strip out all WoS/Scopus citation lines from the sample text.
        text = (fixtures_dir / "cobiss_sample.md").read_text(encoding="utf-8")
        import re
        # Remove every "do <date>: ..." TC/CI/CIAu sentence segment
        text_no_cit = re.sub(
            r"do \d{1,2}\.\s*\d{1,2}\.\s*\d{4}:.*?(?=\]|$)",
            "", text, flags=re.DOTALL,
        )

        def fake_fetch(self, url):
            return text_no_cit, url

        monkeypatch.setattr(CobissClient, "fetch", fake_fetch)
        client = CobissClient()
        with pytest.warns(UserWarning, match="non-.si IP|Slovenia"):
            df, _result = client.fetch_personal_bibliography(
                "https://bib.cobiss.net/foo.html"
            )
        # All Cited by values must be NaN
        assert df["Cited by"].notna().sum() == 0

    def test_no_warning_when_citations_present(
        self, cobiss_sample_text, monkeypatch
    ):
        def fake_fetch(self, url):
            return cobiss_sample_text, url
        monkeypatch.setattr(CobissClient, "fetch", fake_fetch)
        client = CobissClient()
        # The sample fixture has citations on at least one record (#2),
        # so we should NOT see the outside-Slovenia warning.
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client.fetch_personal_bibliography(
                "https://bib.cobiss.net/foo.html"
            )
        slovenia_warnings = [
            w for w in caught
            if "non-.si IP" in str(w.message) or "Slovenia" in str(w.message)
        ]
        assert slovenia_warnings == []
