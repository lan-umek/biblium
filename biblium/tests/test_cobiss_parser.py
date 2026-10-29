# -*- coding: utf-8 -*-
"""
Tests for ``biblium.utilsbib_modules.cobiss_parser``.

Validates field extraction from real-shape sample fixtures, with a focus on
edge cases that emerged during 2.16 development:

- multi-word surnames (``PAJTLER ROŠAR``, ``LAVRIČ GROZNIK``)
- apostrophes in surnames (``DELL'ARTI``)
- ``et al.`` markers in mid-list and end-of-list
- role tags (``(avtor, korespondenčni avtor)``)
- conference / book chapter pattern (``V: *Container*``)
- citation counts in Slovenian-language sentences
- partially-citable records (only WoS, only Scopus, neither)
"""

from __future__ import annotations

import pandas as pd
import pytest

from biblium.utilsbib_modules.cobiss_parser import (
    parse_cobiss_html,
    records_to_dataframe,
)


# ---------------------------------------------------------------------------
# Document-level metadata
# ---------------------------------------------------------------------------


class TestDocumentMetadata:
    """Researcher name, code, period, and the per-typology counts."""

    def test_researcher_name_and_code(self, cobiss_sample_text):
        _, meta = parse_cobiss_html(cobiss_sample_text)
        assert meta.researcher_name == "dr. Lan Umek"
        assert meta.researcher_code == "28519"

    def test_period(self, cobiss_sample_text):
        _, meta = parse_cobiss_html(cobiss_sample_text)
        assert meta.period == "2021-2026"

    def test_record_counts_per_typology(self, cobiss_sample_text):
        _, meta = parse_cobiss_html(cobiss_sample_text)
        assert meta.n_records_per_typology == {
            "1.01": 3, "1.02": 1, "1.03": 1, "1.08": 2,
        }

    def test_total_record_count_matches_sum(self, cobiss_sample_text):
        records, meta = parse_cobiss_html(cobiss_sample_text)
        assert meta.n_records == sum(meta.n_records_per_typology.values())
        assert meta.n_records == len(records)


# ---------------------------------------------------------------------------
# Author parsing
# ---------------------------------------------------------------------------


class TestAuthorParsing:
    """Author extraction across various edge cases."""

    @pytest.fixture()
    def records_by_index(self, cobiss_full_sample_text):
        records, _ = parse_cobiss_html(cobiss_full_sample_text)
        return {r.record_index: r for r in records}

    def test_simple_three_authors(self, records_by_index):
        # #2: STANIMIROVIĆ, Dalibor, UMEK, Lan, RAVŠELJ, Dejan
        assert records_by_index[2].Authors == (
            "Stanimirović, Dalibor; Umek, Lan; Ravšelj, Dejan"
        )

    def test_compound_first_word_surname(self, records_by_index):
        # #1: KOZJEK, ZOREC KLEMENČIČ (two-word surname), UMEK
        assert records_by_index[1].Authors == (
            "Kozjek, Tatjana; Zorec Klemenčič, Uroška; Umek, Lan"
        )

    def test_et_al_in_middle_of_list(self, records_by_index):
        # #5: 6 listed authors then "et al."
        a = records_by_index[5].Authors
        assert a is not None
        # All six listed authors must be present, and the et al. tail too
        assert "Ravšelj, Dejan" in a
        assert "Aristovnik, Aleksander" in a
        assert "et al." in a
        # The Brezovar entry must come *before* Aristovnik and after Umek
        assert a.index("Brezovar, Nejc") > a.index("Umek, Lan")
        assert a.index("Brezovar, Nejc") < a.index("Aristovnik, Aleksander")

    def test_role_tag_is_stripped(self, records_by_index):
        # #8: LONGO, Alja (avtor, korespondenčni avtor), HUDLER, Petra, ...
        a = records_by_index[8].Authors
        assert a is not None
        assert a.startswith("Longo, Alja")
        assert "(avtor" not in a  # role tag must not leak into output
        assert "korespond" not in a.lower()
        # Six total authors expected
        assert a.count(";") == 5

    def test_apostrophe_in_surname(self, records_by_index):
        # #19: DELL'ARTI must survive
        a = records_by_index[19].Authors
        assert a is not None
        assert "Dell'Arti, Laura" in a

    def test_long_author_list(self, records_by_index):
        # #19 has 12 authors -> 11 semicolons
        a = records_by_index[19].Authors
        assert a is not None
        assert a.count(";") == 11

    def test_multiword_surnames(self, records_by_index):
        # #19: PAJTLER ROŠAR, LAVRIČ GROZNIK, GLOBOČNIK PETROVIČ, VIDOVIĆ VALENTINČIČ
        a = records_by_index[19].Authors
        assert a is not None
        assert "Pajtler Rošar, Ana" in a
        assert "Lavrič Groznik, Alenka" in a
        assert "Globočnik Petrovič, Mojca" in a
        assert "Vidović Valentinčič, Nataša" in a


# ---------------------------------------------------------------------------
# Title and source extraction
# ---------------------------------------------------------------------------


class TestTitleAndSource:
    """Title and journal/conference name extraction."""

    @pytest.fixture()
    def records_by_index(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        return {r.record_index: r for r in records}

    def test_journal_article_title(self, records_by_index):
        assert records_by_index[1].Title == (
            "Volunteer motivation in firefighting organisations : "
            "a case of the Slovenian Firefighters Association"
        )

    def test_journal_article_source(self, records_by_index):
        assert records_by_index[1].Source_title == "Fire"
        # Conference must be empty for journal articles
        assert records_by_index[1].Conference is None

    def test_conference_paper_separates_container(self, records_by_index):
        # #28: Conference paper -> Conference is set, Source_title is None
        rec = records_by_index[28]
        assert rec.Conference == (
            "The 33rd NISPAcee Annual Conference, "
            "Bratislava, Slovakia, May 22-24, 2025"
        )
        assert rec.Source_title is None
        # The actual title of the contribution must NOT be a single letter "V"
        assert rec.Title is not None
        assert len(rec.Title) > 10
        assert rec.Title.startswith("Bibliometric analysis")


# ---------------------------------------------------------------------------
# Numeric / structured fields
# ---------------------------------------------------------------------------


class TestStructuredFields:
    """Year, volume, issue, pages, identifiers."""

    @pytest.fixture()
    def records_by_index(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        return {r.record_index: r for r in records}

    def test_year_is_int(self, records_by_index):
        assert records_by_index[1].Year == 2025
        assert isinstance(records_by_index[1].Year, int)

    @pytest.mark.parametrize("idx,vol,iss,pages,art", [
        (1, "8", "6", "1-17", "220"),
        (2, "13", "22", "1-22", "2979"),
        (24, "6", "3", "1-25", None),
    ])
    def test_volume_issue_pages(self, records_by_index, idx, vol, iss, pages, art):
        r = records_by_index[idx]
        assert r.Volume == vol
        assert r.Issue == iss
        assert r.Pages == pages
        assert r.Article_no == art

    def test_no_issue_means_none_not_garbage(self, records_by_index):
        # #27 has no "iss." or "no." in its volume statement;
        # it must NOT pick up the "št. citatov" Slovenian-language phrase.
        assert records_by_index[27].Issue is None

    def test_page_start_and_end(self, records_by_index):
        r = records_by_index[1]
        assert r.Page_start == "1"
        assert r.Page_end == "17"

    def test_identifiers(self, records_by_index):
        r = records_by_index[2]
        assert r.ISSN == "2227-9032"
        assert r.DOI == "10.3390/healthcare13222979"
        assert r.cobiss_id == "257977347"

    def test_doi_org_prefix_is_stripped(self, records_by_index):
        # #1: "DOI: [doi.org/10.3390/fire8060220](...)" -- the bare prefix must go
        assert records_by_index[1].DOI == "10.3390/fire8060220"


# ---------------------------------------------------------------------------
# Citation counts (WoS / Scopus / Cited by)
# ---------------------------------------------------------------------------


class TestCitations:
    """Web of Science / Scopus citation counts."""

    @pytest.fixture()
    def records_by_index(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        return {r.record_index: r for r in records}

    def test_wos_and_scopus_both_present(self, records_by_index):
        r = records_by_index[2]
        assert r.cobiss_wos_tc == 1
        assert r.cobiss_wos_ci == 1
        assert r.cobiss_wos_ciau == pytest.approx(0.33)
        assert r.cobiss_scopus_tc == 1

    def test_cited_by_defaults_to_wos(self, records_by_index):
        # #24 has WoS TC=9, Scopus TC=18 -> Cited by must be 9 (default WoS)
        assert records_by_index[24].Cited_by == 9
        assert records_by_index[24].cobiss_wos_tc == 9
        assert records_by_index[24].cobiss_scopus_tc == 18

    def test_cited_by_can_be_switched_to_scopus(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(
            cobiss_sample_text, default_citation_source="scopus"
        )
        idx = {r.record_index: r for r in records}
        assert idx[24].Cited_by == 18  # Scopus value, not 9

    def test_no_citations_yields_none(self, records_by_index):
        # #1 has only [JCR, SNIP] markers -> no TC anywhere
        r = records_by_index[1]
        assert r.Cited_by is None
        assert r.cobiss_wos_tc is None
        assert r.cobiss_scopus_tc is None

    def test_citation_date_captured(self, records_by_index):
        r = records_by_index[2]
        assert r.cobiss_wos_date is not None
        assert "2026" in r.cobiss_wos_date

    def test_only_scopus_link_no_counts_yields_none(self, cobiss_full_sample_text):
        # In the full sample, record #17 has a bare [Scopus](...) link
        # with no "do <date>: ... TC: ..." follow-up -> no counts.
        records, _ = parse_cobiss_html(cobiss_full_sample_text)
        idx = {r.record_index: r for r in records}
        if 17 in idx:  # only check if the fixture includes this record
            r = idx[17]
            assert r.cobiss_scopus_tc is None
            assert r.cobiss_wos_tc is None


# ---------------------------------------------------------------------------
# Open-access and repository flags
# ---------------------------------------------------------------------------


class TestOpenAccessAndRepoFlags:
    """Detection of open-access marker and repository links."""

    @pytest.fixture()
    def records_by_index(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        return {r.record_index: r for r in records}

    def test_open_access_detected(self, records_by_index):
        # #2 has "Odprti dostop" marker -> Open Access = True
        assert records_by_index[2].Open_Access is True
        assert records_by_index[2].cobiss_open_access is True

    def test_no_open_access_when_marker_absent(self, records_by_index):
        # #1: only [JCR, SNIP] markers, no "Odprti dostop"
        assert records_by_index[1].Open_Access is False

    def test_repository_link_flags(self, records_by_index):
        r = records_by_index[2]
        assert r.cobiss_RUL is True       # has [RUL](...) link
        assert r.cobiss_dCOBISS is True   # has [dCOBISS](...) link

    def test_dlib_link_flag(self, records_by_index):
        # #3 has dLib.si link
        r = records_by_index[3]
        assert r.cobiss_dLib is True


# ---------------------------------------------------------------------------
# Funding and awards
# ---------------------------------------------------------------------------


class TestFundingAndAwards:
    """Extraction of funding (`projekt:` lines) and awards."""

    def test_funding_extracted_with_funder(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        idx = {r.record_index: r for r in records}
        f = idx[2].Funding
        assert f is not None
        assert "P5-0093-2019" in f
        assert "funder:" in f

    def test_award_extracted(self, cobiss_full_sample_text):
        records, _ = parse_cobiss_html(cobiss_full_sample_text)
        idx = {r.record_index: r for r in records}
        # #11 has "nagrada: Priznanje Fakultete za upravo..."
        # #14 has "nagrada: Odlični v znanosti 2022 (ARRS)"
        if 14 in idx:
            assert idx[14].Awards is not None
            assert "Odlični v znanosti" in idx[14].Awards


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------


class TestDataFrameConversion:
    """``records_to_dataframe`` produces biblium-shaped output."""

    def test_returns_dataframe(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        df = records_to_dataframe(records)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(records)

    def test_canonical_columns_present(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        df = records_to_dataframe(records)
        for col in (
            "Authors", "Title", "Year", "Source title",
            "Volume", "Issue", "Pages", "Document Type",
            "Cited by", "Open Access", "DOI", "ISSN",
        ):
            assert col in df.columns, f"missing column: {col}"

    def test_cobiss_specific_columns_present(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        df = records_to_dataframe(records)
        for col in (
            "cobiss_id",
            "cobiss_typology_code",
            "cobiss_typology_label_sl",
            "cobiss_typology_label_en",
            "cobiss_wos_tc",
            "cobiss_scopus_tc",
        ):
            assert col in df.columns

    def test_columns_renamed_with_spaces(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        df = records_to_dataframe(records)
        # Underscored field names from the dataclass should not leak
        assert "Source_title" not in df.columns
        assert "Page_start" not in df.columns
        assert "Document_Type" not in df.columns

    def test_empty_input(self):
        df = records_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_raw_text_dropped_by_default(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        df = records_to_dataframe(records)
        assert "raw_text" not in df.columns

    def test_raw_text_kept_when_requested(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        df = records_to_dataframe(records, drop_raw_text=False)
        assert "raw_text" in df.columns


# ---------------------------------------------------------------------------
# Source detection (HTML vs text vs path)
# ---------------------------------------------------------------------------


class TestSourceDetection:
    """``parse_cobiss_html`` accepts text, HTML, and file paths."""

    def test_string_input(self, cobiss_sample_text):
        records, _ = parse_cobiss_html(cobiss_sample_text)
        assert len(records) > 0

    def test_path_input(self, fixtures_dir):
        records, _ = parse_cobiss_html(str(fixtures_dir / "cobiss_sample.md"))
        assert len(records) > 0

    def test_html_with_html_tag_is_extracted(self):
        # Bare-bones HTML fragment around a record
        html = """
        <html><body>
        <h1>dr. Test User [99999]</h1>
        <h2>Osebna bibliografija za obdobje 2020-2024</h2>
        <h3>ČLANKI IN DRUGI SESTAVNI DELI</h3>
        <h4>1.01 Izvirni znanstveni članek</h4>
        <p>1.</p>
        <p>SMITH, John, JONES, Mary. A test article. <i>Test Journal</i>.
        2024, vol. 1, iss. 2, str. 5-10. ISSN 1234-5678.
        DOI: <a href="https://dx.doi.org/10.1234/test">10.1234/test</a>.
        [COBISS.SI-ID 12345]</p>
        </body></html>
        """
        records, meta = parse_cobiss_html(html, is_html=True)
        # We just check that parsing didn't crash and produced *something*
        assert meta.researcher_code == "99999"
