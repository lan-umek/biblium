# -*- coding: utf-8 -*-
"""
Tests for ``biblium.utilsbib_modules.cobiss_typology``.

Covers:
- presence and shape of the TYPOLOGY table
- bilingual label lookup
- mapping to canonical biblium Document Type categories
- reverse lookup from labels back to codes
"""

from __future__ import annotations

import pytest

from biblium.utilsbib_modules.cobiss_typology import (
    TYPOLOGY,
    code_from_label,
    typology_label,
    typology_to_document_type,
)


class TestTypologyTable:
    """Basic invariants of the TYPOLOGY mapping."""

    def test_table_is_non_empty(self):
        assert len(TYPOLOGY) > 0

    def test_all_codes_match_dotted_format(self):
        # All codes follow "<digit>.<two-digits>" (1.01, 2.20, 3.10, ...)
        for code in TYPOLOGY:
            parts = code.split(".")
            assert len(parts) == 2, f"bad code shape: {code!r}"
            assert parts[0].isdigit() and parts[1].isdigit()

    def test_each_entry_has_three_fields(self):
        for code, value in TYPOLOGY.items():
            assert isinstance(value, tuple) and len(value) == 3, code
            label_sl, label_en, doc_type = value
            assert isinstance(label_sl, str) and label_sl
            assert isinstance(label_en, str) and label_en
            assert isinstance(doc_type, str) and doc_type

    def test_known_codes_present(self):
        # Spot-check the most important codes
        for code in ("1.01", "1.02", "1.03", "1.08", "1.16", "2.01"):
            assert code in TYPOLOGY


class TestTypologyLabel:
    """Behaviour of ``typology_label``."""

    @pytest.mark.parametrize("code,expected_en", [
        ("1.01", "Original Scientific Article"),
        ("1.02", "Review Article"),
        ("1.08", "Published Scientific Conference Contribution"),
        ("2.01", "Scientific Monograph"),
    ])
    def test_english_labels(self, code, expected_en):
        assert typology_label(code, "en") == expected_en

    def test_default_language_is_english(self):
        # No `lang` argument given -> should match the explicit "en" form.
        assert typology_label("1.01") == typology_label("1.01", "en")

    @pytest.mark.parametrize("code,expected_sl", [
        ("1.01", "Izvirni znanstveni članek"),
        ("1.02", "Pregledni znanstveni članek"),
    ])
    def test_slovenian_labels(self, code, expected_sl):
        assert typology_label(code, "sl") == expected_sl

    def test_unknown_code_returns_none(self):
        assert typology_label("9.99") is None
        assert typology_label("not-a-code") is None

    def test_whitespace_is_stripped(self):
        assert typology_label("  1.01  ") == typology_label("1.01")


class TestDocumentTypeMapping:
    """Behaviour of ``typology_to_document_type``."""

    @pytest.mark.parametrize("code,expected", [
        ("1.01", "Article"),       # original scientific article
        ("1.02", "Review"),        # review article
        ("1.03", "Article"),       # other scientific article
        ("1.06", "Conference Paper"),
        ("1.08", "Conference Paper"),
        ("1.13", "Conference Paper"),
        ("1.16", "Book Chapter"),
        ("1.17", "Book Chapter"),
        ("1.20", "Editorial"),     # preface
        ("2.01", "Book"),          # scientific monograph
        ("2.06", "Book"),          # encyclopaedia
        ("2.20", "Other"),         # research data
        ("3.15", "Other"),         # conference talk without a printed version
    ])
    def test_canonical_categories(self, code, expected):
        assert typology_to_document_type(code) == expected

    def test_unknown_code_falls_back_to_other(self):
        assert typology_to_document_type("9.99") == "Other"

    def test_only_canonical_categories_are_used(self):
        """
        ``Document Type`` values must match the existing biblium
        Scopus-reader convention so ``BiblioStats(db='cobiss')`` and
        ``BiblioStats(db='scopus')`` produce comparable category labels.
        """
        allowed = {
            "Article", "Review", "Conference Paper",
            "Book", "Book Chapter", "Editorial", "Other",
        }
        for code, (_, _, doc_type) in TYPOLOGY.items():
            assert doc_type in allowed, (code, doc_type)


class TestReverseLookup:
    """``code_from_label`` should recover the code from either language."""

    def test_slovenian_label(self):
        assert code_from_label("Izvirni znanstveni članek") == "1.01"

    def test_english_label(self):
        assert code_from_label("Original Scientific Article") == "1.01"

    def test_case_insensitive(self):
        assert code_from_label("ORIGINAL SCIENTIFIC ARTICLE") == "1.01"
        assert code_from_label("izvirni znanstveni članek") == "1.01"

    def test_missing_diacritics_does_not_match(self):
        # Reverse lookup is exact (case-insensitive) against the canonical
        # tables; without diacritics there is no match -> None.
        assert code_from_label("Izvirni znanstveni clanek") is None

    def test_unknown_label(self):
        assert code_from_label("Some made-up label") is None
