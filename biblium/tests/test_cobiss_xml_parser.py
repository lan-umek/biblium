# -*- coding: utf-8 -*-
"""
Tests for ``biblium.utilsbib_modules.cobiss_xml_parser``.

The XML parser is currently a *skeleton*: document-level metadata
extraction is implemented; per-record extraction raises
``NotImplementedError`` until we have a real ``format=X`` sample
to validate against.

These tests cover:
- top-level ``<Bibliography>`` attribute extraction
- ``<Name>``, ``<OrgName>``, ``<Title>`` text extraction
- empty-bibliography warning
- explicit ``NotImplementedError`` when unit candidates are present
- monkey-patchable ``_parse_unit`` hook so client code can drive
  the parser end-to-end in tests
"""

from __future__ import annotations

import warnings

import pytest

from biblium.utilsbib_modules import cobiss_xml_parser as xp
from biblium.utilsbib_modules.cobiss_parser import ParsedCobissRecord


# Minimal XML payloads
_EMPTY_BIB = """<?xml version="1.0" encoding="UTF-8"?>
<Bibliography biblioType="personal" code="28519" period="2021-2026">
    <Name>dr. Lan Umek</Name>
    <OrgName>Univerza v Ljubljani, Fakulteta za upravo</OrgName>
    <Title>Osebna bibliografija za obdobje 2021-2026</Title>
</Bibliography>
"""

_BIB_WITH_FAKE_UNIT = """<?xml version="1.0" encoding="UTF-8"?>
<Bibliography code="28519" period="2021-2026">
    <Name>dr. Lan Umek</Name>
    <bibUnit><title>fake unit</title></bibUnit>
</Bibliography>
"""

_BIB_WITH_BOM = "\ufeff" + _EMPTY_BIB

_BIB_WITH_NAMESPACE = """<?xml version="1.0" encoding="UTF-8"?>
<Bibliography xmlns="http://home.izum.si/cobiss/xml" code="42" period="2024">
    <Name>Test Person</Name>
</Bibliography>
"""


# ---------------------------------------------------------------------------
# Document-level metadata
# ---------------------------------------------------------------------------


class TestDocumentMetadata:
    """Top-level ``<Bibliography>`` attributes and children."""

    def test_extracts_code_and_period(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, meta = xp.parse_cobiss_xml(_EMPTY_BIB)
        assert meta.researcher_code == "28519"
        assert meta.period == "2021-2026"

    def test_extracts_name_orgname_title(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, meta = xp.parse_cobiss_xml(_EMPTY_BIB)
        assert meta.researcher_name == "dr. Lan Umek"
        assert "Fakulteta za upravo" in (meta.org_name or "")
        assert meta.title == "Osebna bibliografija za obdobje 2021-2026"

    def test_extracts_biblio_type_attribute(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, meta = xp.parse_cobiss_xml(_EMPTY_BIB)
        assert meta.biblio_type == "personal"

    def test_handles_xml_with_bom(self):
        # UTF-8 BOM at the start must not break ElementTree
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, meta = xp.parse_cobiss_xml(_BIB_WITH_BOM)
        assert meta.researcher_code == "28519"

    def test_handles_xml_with_default_namespace(self):
        # The published schema has a target namespace; the parser must
        # tolerate either namespaced or non-namespaced elements.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, meta = xp.parse_cobiss_xml(_BIB_WITH_NAMESPACE)
        assert meta.researcher_code == "42"
        assert meta.researcher_name == "Test Person"


# ---------------------------------------------------------------------------
# Empty-bibliography behaviour
# ---------------------------------------------------------------------------


class TestEmptyBibliography:
    """A bibliography with no unit elements warns but does not raise."""

    def test_empty_bibliography_does_not_raise(self):
        # Should complete without exceptions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            records, meta = xp.parse_cobiss_xml(_EMPTY_BIB)
        assert records == []
        assert meta.n_records == 0

    def test_empty_bibliography_warns(self):
        with pytest.warns(UserWarning, match="No bibliographic-unit"):
            xp.parse_cobiss_xml(_EMPTY_BIB)


# ---------------------------------------------------------------------------
# Not-yet-implemented record-level parsing
# ---------------------------------------------------------------------------


class TestRecordParsingPlaceholder:
    """When unit candidates are found, the placeholder raises clearly."""

    def test_raises_on_record_candidate(self):
        # The XML contains a <bibUnit>, which is in our heuristic set ->
        # _parse_unit() runs and raises NotImplementedError.
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            xp.parse_cobiss_xml(_BIB_WITH_FAKE_UNIT)


# ---------------------------------------------------------------------------
# Monkey-patchable _parse_unit (used by integration tests later)
# ---------------------------------------------------------------------------


class TestMonkeyPatchableParseUnit:
    """``_parse_unit`` can be replaced to drive the parser end-to-end in tests."""

    def test_custom_parse_unit_runs_to_completion(self, monkeypatch):
        # Replace _parse_unit with a trivial implementation that returns
        # a ParsedCobissRecord with just the title set.
        def fake_parse_unit(elem):
            title_elem = elem.find("title")
            if title_elem is None:
                return None
            return ParsedCobissRecord(
                Title=title_elem.text,
                cobiss_typology_code="1.01",
            )
        monkeypatch.setattr(xp, "_parse_unit", fake_parse_unit)

        records, meta = xp.parse_cobiss_xml(_BIB_WITH_FAKE_UNIT)
        assert len(records) == 1
        assert records[0].Title == "fake unit"
        assert meta.n_records == 1


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicSurfaceMatchesHTMLParser:
    """
    The XML and HTML parsers expose the same public surface so callers
    can swap between them transparently.
    """

    def test_returns_records_metadata_tuple(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = xp.parse_cobiss_xml(_EMPTY_BIB)
        # parse_cobiss_html returns (List[ParsedCobissRecord], metadata)
        # parse_cobiss_xml must do the same
        assert isinstance(result, tuple) and len(result) == 2
        records, meta = result
        assert isinstance(records, list)
        # Metadata object must have the fields downstream code reads
        for attr in ("researcher_name", "researcher_code", "period",
                     "n_records", "n_records_per_typology"):
            assert hasattr(meta, attr), attr
