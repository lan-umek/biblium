# -*- coding: utf-8 -*-
"""
Integration tests for the URL-routing layer in
``biblium.cobiss_api.CobissClient.fetch_personal_bibliography``.

These verify that:
- CRIS / CONOR direct links get rewritten to request XML by default
- pre-rendered HTML URLs are NEVER rewritten
- ``prefer_format`` lets the user override the choice
- when XML parsing is not yet implemented, ``"auto"`` mode silently
  falls back to HTML
- when the user explicitly requests ``"xml"``, the
  ``NotImplementedError`` from the XML parser propagates

All tests are *hermetic*: ``CobissClient.fetch`` is patched to return
canned responses instead of making real HTTP calls.
"""

from __future__ import annotations

import warnings
from urllib.parse import parse_qsl, urlparse

import pandas as pd
import pytest

from biblium.cobiss_api import CobissClient
from biblium.utilsbib_modules import cobiss_xml_parser as xp
from biblium.utilsbib_modules.cobiss_parser import ParsedCobissRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _query(url: str) -> dict:
    return dict(parse_qsl(urlparse(url).query, keep_blank_values=True))


@pytest.fixture()
def captured_url(monkeypatch, cobiss_sample_text):
    """
    Patch ``CobissClient.fetch`` so it captures the URL it would have
    fetched and returns the HTML sample fixture as the response body.
    Returns a list that the patched function appends to.
    """
    captured: list[str] = []

    def fake_fetch(self, url):
        captured.append(url)
        return cobiss_sample_text, url

    monkeypatch.setattr(CobissClient, "fetch", fake_fetch)
    return captured


# ---------------------------------------------------------------------------
# CRIS / CONOR direct links get format=X by default
# ---------------------------------------------------------------------------


class TestRoutingForDirectLinks:
    """Direct links are rewritten with ``format=X&citation=true`` by default."""

    @pytest.mark.parametrize("url", [
        "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519",
        "https://bib.cobiss.net/biblioweb/direct/si/slv/cris/28519?fromyear=2021",
        "https://bib.cobiss.net/biblioweb/direct/si/eng/conor/54909283",
    ])
    def test_default_requests_xml(self, captured_url, url):
        client = CobissClient()
        # Stub _dispatch_parsing so the test does not require XML parsing
        # to be implemented; we only care that the URL was rewritten.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client.fetch_personal_bibliography(url)

        assert len(captured_url) == 1
        rewritten = captured_url[0]
        q = _query(rewritten)
        assert q.get("format") == "X"
        assert q.get("citation") == "true"

    def test_user_query_params_preserved(self, captured_url):
        client = CobissClient()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client.fetch_personal_bibliography(
                "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519"
                "?fromyear=2021&toyear=2026"
            )
        rewritten = captured_url[0]
        q = _query(rewritten)
        assert q.get("fromyear") == "2021"
        assert q.get("toyear") == "2026"
        assert q.get("format") == "X"

    def test_explicit_html_skips_xml(self, captured_url):
        client = CobissClient()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client.fetch_personal_bibliography(
                "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519",
                prefer_format="html",
            )
        q = _query(captured_url[0])
        assert q.get("format") == "H"


# ---------------------------------------------------------------------------
# Pre-rendered HTML URLs are never rewritten
# ---------------------------------------------------------------------------


class TestPreRenderedHTMLNotRewritten:
    """The static HTML report URLs must reach the server verbatim."""

    @pytest.mark.parametrize("prefer_format", ["auto", "xml", "html"])
    def test_url_is_passed_through(self, captured_url, prefer_format):
        url = (
            "https://bib.cobiss.net/bibliographies/si/webBiblio/"
            "bib201_20260430_111142_28519.html"
        )
        client = CobissClient()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client.fetch_personal_bibliography(
                url, prefer_format=prefer_format,
            )
        assert captured_url[0] == url


# ---------------------------------------------------------------------------
# XML parsing fallback to HTML
# ---------------------------------------------------------------------------


class TestXMLFallback:
    """
    Behaviour when the URL says XML (``format=X``) but the XML parser
    is not yet implemented for record-level extraction.
    """

    def test_auto_falls_back_to_html_when_xml_not_implemented(
        self, monkeypatch, cobiss_sample_text
    ):
        captured: list[str] = []

        # The fetch returns the *HTML* sample even though the URL asked
        # for format=X (some endpoints quietly ignore unknown format codes).
        def fake_fetch(self, url):
            captured.append(url)
            return cobiss_sample_text, url

        monkeypatch.setattr(CobissClient, "fetch", fake_fetch)

        client = CobissClient()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df, result = client.fetch_personal_bibliography(
                "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519",
                prefer_format="auto",
            )

        # With auto, the failure to parse XML is silent and we fall back
        # to HTML, producing real records.
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert result.format_used == "html"

    def test_explicit_xml_propagates_not_implemented(
        self, monkeypatch
    ):
        # When the user asks for XML explicitly, the NotImplementedError
        # from the XML parser must surface (no silent fallback).
        # We use an XML response that contains a unit candidate so the
        # placeholder _parse_unit() is reached.
        xml_with_unit = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Bibliography code="28519">'
            '  <Name>Test</Name>'
            '  <bibUnit><title>fake</title></bibUnit>'
            '</Bibliography>'
        )

        def fake_fetch(self, url):
            return xml_with_unit, url

        monkeypatch.setattr(CobissClient, "fetch", fake_fetch)

        client = CobissClient()
        with pytest.raises(NotImplementedError):
            client.fetch_personal_bibliography(
                "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519",
                prefer_format="xml",
            )

    def test_xml_parsing_runs_when_parser_is_complete(
        self, monkeypatch
    ):
        """
        When ``_parse_unit`` is filled in (here we monkey-patch it),
        the XML path runs to completion and produces records. This
        is the test that will *automatically pass* once the XML
        parser is wired up against a real sample.
        """
        xml_with_unit = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Bibliography code="28519" period="2024">'
            '  <Name>Test User</Name>'
            '  <bibUnit><title>my title</title></bibUnit>'
            '</Bibliography>'
        )

        def fake_fetch(self, url):
            return xml_with_unit, url

        def fake_parse_unit(elem):
            t = elem.find("title")
            if t is None:
                return None
            return ParsedCobissRecord(
                Title=t.text,
                cobiss_typology_code="1.01",
            )

        monkeypatch.setattr(CobissClient, "fetch", fake_fetch)
        monkeypatch.setattr(xp, "_parse_unit", fake_parse_unit)

        client = CobissClient()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df, result = client.fetch_personal_bibliography(
                "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519",
                prefer_format="xml",
            )
        assert len(df) == 1
        assert df.iloc[0]["Title"] == "my title"
        assert result.format_used == "xml"
        assert result.researcher_code == "28519"
