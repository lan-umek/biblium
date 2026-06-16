# -*- coding: utf-8 -*-
"""
Tests for ``biblium.utilsbib_modules.cobiss_url``.

These cover URL classification (CRIS / CONOR / pre-rendered HTML /
unknown) and the rewriting helper that adds ``format=`` and
``citation=`` parameters without dropping the user's own.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlparse

import pytest

from biblium.utilsbib_modules.cobiss_url import (
    classify_cobiss_url,
    is_xml_format,
    prepare_request_url,
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    """``classify_cobiss_url`` recognises CRIS, CONOR and pre-rendered HTML."""

    def test_cris_direct_basic(self):
        info = classify_cobiss_url(
            "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519"
        )
        assert info.kind == "cris_direct"
        assert info.code == "28519"
        assert info.country == "si"
        assert info.lang == "eng"
        assert info.supports_format_param is True

    def test_cris_direct_with_query(self):
        info = classify_cobiss_url(
            "https://bib.cobiss.net/biblioweb/direct/si/slv/cris/05490"
            "?fromyear=2015&toyear=2018"
        )
        assert info.kind == "cris_direct"
        assert info.code == "05490"
        assert info.lang == "slv"

    def test_conor_direct(self):
        info = classify_cobiss_url(
            "https://bib.cobiss.net/biblioweb/direct/si/eng/conor/54909283"
        )
        assert info.kind == "conor_direct"
        assert info.code == "54909283"
        assert info.country == "si"
        assert info.lang == "eng"
        assert info.supports_format_param is True

    def test_prerendered_html_with_date_and_time(self):
        # The real-world filename has both date and time underscore parts:
        #   bib201_<YYYYMMDD>_<HHMMSS>_<code>.html
        info = classify_cobiss_url(
            "https://bib.cobiss.net/bibliographies/si/webBiblio/"
            "bib201_20260430_111142_28519.html"
        )
        assert info.kind == "prerendered_html"
        assert info.code == "28519"
        assert info.country == "si"
        assert info.lang is None
        assert info.supports_format_param is False

    def test_unknown_url(self):
        info = classify_cobiss_url("https://example.com/some/path")
        assert info.kind == "unknown"
        assert info.code is None
        assert info.supports_format_param is False

    def test_malformed_url_does_not_crash(self):
        # A bare string with weird characters must not raise
        info = classify_cobiss_url("not-a-url")
        assert info.kind == "unknown"

    def test_macedonian_country(self):
        # COBISS is multi-country; "mk" should also classify as direct
        info = classify_cobiss_url(
            "https://bib.cobiss.net/biblioweb/direct/mk/eng/cris/12345"
        )
        assert info.kind == "cris_direct"
        assert info.country == "mk"


# ---------------------------------------------------------------------------
# Request preparation
# ---------------------------------------------------------------------------


def _query_dict(url: str) -> dict:
    """Helper: parse the query string into a {key: value} dict."""
    return dict(parse_qsl(urlparse(url).query, keep_blank_values=True))


class TestPrepareRequestURL:
    """``prepare_request_url`` rewrites direct links and leaves others alone."""

    def test_default_adds_format_and_citation(self):
        out = prepare_request_url(
            "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519"
        )
        q = _query_dict(out)
        assert q.get("format") == "X"        # XML default
        assert q.get("citation") == "true"

    def test_preserves_user_supplied_params(self):
        out = prepare_request_url(
            "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519"
            "?fromyear=2021&toyear=2026"
        )
        q = _query_dict(out)
        assert q.get("fromyear") == "2021"
        assert q.get("toyear") == "2026"
        assert q.get("format") == "X"
        assert q.get("citation") == "true"

    def test_user_format_wins_when_auto(self):
        # If user already specified format=H, "auto" mode must not override it
        out = prepare_request_url(
            "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519?format=H",
            format="auto",
        )
        q = _query_dict(out)
        assert q.get("format") == "H"

    def test_explicit_format_overrides_user(self):
        # Explicit format="xml" must overwrite the user's format=H
        out = prepare_request_url(
            "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519?format=H",
            format="xml",
        )
        q = _query_dict(out)
        assert q.get("format") == "X"

    def test_citation_false_strips_existing(self):
        out = prepare_request_url(
            "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519"
            "?citation=true",
            citation=False,
        )
        q = _query_dict(out)
        assert "citation" not in q

    def test_citation_none_leaves_existing_alone(self):
        out = prepare_request_url(
            "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519"
            "?citation=false",
            citation=None,
        )
        q = _query_dict(out)
        # The user's explicit citation=false survives
        assert q.get("citation") == "false"

    def test_prerendered_html_unchanged(self):
        url = (
            "https://bib.cobiss.net/bibliographies/si/webBiblio/"
            "bib201_20260430_111142_28519.html"
        )
        # No matter what we ask for, pre-rendered HTML URLs are untouched
        for kw in ({}, {"format": "xml"}, {"format": "html"},
                   {"citation": False}):
            assert prepare_request_url(url, **kw) == url

    def test_unknown_url_unchanged(self):
        url = "https://example.com/foo?bar=1"
        assert prepare_request_url(url) == url

    def test_format_letters_match_documentation(self):
        # Spot-check each documented format code
        for fmt_name, expected_letter in [
            ("xml", "X"),
            ("html", "H"),
            ("latex", "L"),
            ("pdf", "P"),
            ("txt", "T"),
        ]:
            out = prepare_request_url(
                "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519",
                format=fmt_name,
            )
            assert _query_dict(out).get("format") == expected_letter

    def test_invalid_format_raises_keyerror(self):
        with pytest.raises(KeyError):
            prepare_request_url(
                "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519",
                format="json",  # not in _FORMAT_LETTER
            )

    def test_does_not_duplicate_format(self):
        # Calling twice with default settings must not produce format=X&format=X
        once = prepare_request_url(
            "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519"
        )
        twice = prepare_request_url(once)
        # The number of "format=" occurrences must be exactly one
        assert twice.count("format=") == 1
        assert twice.count("citation=") == 1


# ---------------------------------------------------------------------------
# is_xml_format helper
# ---------------------------------------------------------------------------


class TestIsXmlFormat:
    """``is_xml_format`` detects ``?format=X`` (case-insensitive)."""

    @pytest.mark.parametrize("url,expected", [
        ("https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519?format=X",
            True),
        ("https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519?format=x",
            True),
        ("https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519?format=H",
            False),
        ("https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519?format=T",
            False),
        ("https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519",
            False),
        ("not-a-url", False),
    ])
    def test_recognises_xml_format(self, url, expected):
        assert is_xml_format(url) is expected
