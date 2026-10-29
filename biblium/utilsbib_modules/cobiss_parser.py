# -*- coding: utf-8 -*-
"""
biblium.utilsbib_modules.cobiss_parser
======================================

Parser for COBISS+ personal bibliography reports.

Background
----------
COBISS+ ("Slovenian national library system") publishes personal
bibliographies as static HTML pages at URLs of the form:

    https://bib.cobiss.net/bibliographies/si/webBiblio/bib201_<timestamp>_<code>.html

These pages list a researcher's entire publication record, organised
into hierarchical sections by the official COBISS *Typology of
Documents/Works* (e.g. ``1.01 Izvirni znanstveni članek`` -> Original
Scientific Article). Each entry follows the **ISO citation format**
with embedded metadata: COBISS.SI-ID, DOI, ISSN, links to JCR/SNIP and
Web of Science / Scopus citation counts (visible only from a Slovenian
IP).

Public components
-----------------
ParsedCobissRecord
    Dataclass holding the canonical fields extracted from a single
    bibliographic record (Authors, Title, Year, Source title, Volume,
    Issue, Pages, ISSN, DOI, Cited by, ...). One per row in the output
    DataFrame / CSV.

parse_cobiss_html
    Top-level parser. Accepts raw HTML (typical case), pre-extracted
    plain text, or a path to either, and returns
    ``(records, metadata)``.

records_to_dataframe
    Convert a list of ``ParsedCobissRecord`` to a biblium-shaped
    DataFrame ready to be written to CSV.

Author: Lan Umek
Version: 2.16.0
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field, asdict
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

import pandas as pd

from biblium.utilsbib_modules.cobiss_typology import (
    typology_label,
    typology_to_document_type,
)

try:
    from bs4 import BeautifulSoup  # type: ignore
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# =============================================================================
# DATA CLASS
# =============================================================================


@dataclass
class ParsedCobissRecord:
    """
    Canonical bibliographic record parsed from a COBISS+ personal
    bibliography page.

    Field naming follows the biblium convention used elsewhere in the
    package (compatible with ``read_scopus_csv`` output):

    - ``Authors`` is the full author list as a single string with
      ``"; "`` separator (matches Scopus convention).
    - ``Year`` is an integer; missing values are stored as ``None``.
    - ``Cited by`` is the number of citations from the user's preferred
      source (``"wos"`` by default, can be set to ``"scopus"``).
    - ``Document Type`` is the broad category (``"Article"``,
      ``"Review"``, ``"Conference Paper"``, ``"Book"``, ``"Book Chapter"``,
      ``"Editorial"``, ``"Other"``) derived from ``cobiss_typology_code``.

    Cobiss-specific fields are prefixed with ``cobiss_``.
    """
    # Core scopus-compatible fields
    Authors: Optional[str] = None
    Title: Optional[str] = None
    Year: Optional[int] = None
    Source_title: Optional[str] = None
    Volume: Optional[str] = None
    Issue: Optional[str] = None
    Page_start: Optional[str] = None
    Page_end: Optional[str] = None
    Pages: Optional[str] = None
    Article_no: Optional[str] = None
    ISSN: Optional[str] = None
    ISBN: Optional[str] = None
    DOI: Optional[str] = None
    Document_Type: Optional[str] = None
    Cited_by: Optional[int] = None
    Open_Access: bool = False
    Editors: Optional[str] = None
    Publisher: Optional[str] = None
    Conference: Optional[str] = None
    URL: Optional[str] = None
    # Cobiss-specific
    cobiss_id: Optional[str] = None
    cobiss_typology_code: Optional[str] = None
    cobiss_typology_label_sl: Optional[str] = None
    cobiss_typology_label_en: Optional[str] = None
    # Citation breakdown (so the user can later choose the source)
    cobiss_wos_tc: Optional[int] = None       # total citations (WoS)
    cobiss_wos_ci: Optional[int] = None       # pure citations (CI)
    cobiss_wos_ciau: Optional[float] = None   # CI per author
    cobiss_wos_date: Optional[str] = None     # snapshot date as raw string
    cobiss_scopus_tc: Optional[int] = None
    cobiss_scopus_ci: Optional[int] = None
    cobiss_scopus_ciau: Optional[float] = None
    cobiss_scopus_date: Optional[str] = None
    # Funding & awards (semicolon-joined strings)
    Funding: Optional[str] = None
    Awards: Optional[str] = None
    # Repository links flags (not values, to keep CSV slim)
    cobiss_open_access: bool = False
    cobiss_dCOBISS: bool = False
    cobiss_RUL: bool = False
    cobiss_dLib: bool = False
    # Bookkeeping
    record_index: Optional[int] = None    # ordinal in the listing (1, 2, ...)
    raw_text: Optional[str] = None        # full original text, for debugging


# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Markdown link [text](url) and bare <url>
_RE_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_RE_BARE_URL = re.compile(r"<(https?://[^>]+)>")

# Typology section heading, e.g. "#### 1.01 Izvirni znanstveni članek" or
# (when stripped of markdown) "1.01 Izvirni znanstveni članek".
_RE_TYPOLOGY_HEADER = re.compile(
    r"^\s*#{0,6}\s*(?P<code>[123]\.\d{2})\s+(?P<label>[^\n]+?)\s*$",
    re.MULTILINE,
)

# Record number: digit(s) then "." on its own line
_RE_RECORD_START = re.compile(r"^\s*(\d{1,4})\.\s*$", re.MULTILINE)

# Author cluster: SURNAME (uppercase, may include accents/spaces), then comma+space, then
# proper-case first name. Sequence joined by ", ". Stop at first ". " followed by capital.
# We capture the leading run of CAPS-words greedily.
_RE_COBISS_ID = re.compile(r"COBISS\.SI-ID\s*\[?(\d+)\]?")
_RE_DOI = re.compile(
    r"DOI:\s*\[?(?:doi\.org/|https?://(?:dx\.)?doi\.org/)?([^\]\s)]+)",
    re.IGNORECASE,
)
_RE_ISSN = re.compile(r"\bISSN\s+([\dX]{4}-[\dX]{4})\b", re.IGNORECASE)
_RE_ISBN = re.compile(r"\bISBN\s+([\d\-X]+)\b", re.IGNORECASE)

# Volume / issue / pages / article number (Slovenian-style "vol., iss., str.")
_RE_VOLUME = re.compile(
    r"\b(?:vol\.|letn\.)\s*(?P<vol>[\w\d\-]+)", re.IGNORECASE
)
_RE_ISSUE = re.compile(
    # Match issue keywords followed only by digit/slash patterns: "iss. 22",
    # "iss. 1/2", "no. 4", "št. 6", "br. 12". Never match "no. of", "št. citatov",
    # etc. (only word characters that look like proper issue identifiers).
    r"\b(?:iss\.|št\.|br\.|no\.)\s*(?P<iss>\d+(?:[/\-]\d+)?)",
    re.IGNORECASE,
)
_RE_PAGES = re.compile(
    r"\bstr\.\s*(?P<p1>[\w\d]+)(?:\s*[-–]\s*(?P<p2>[\w\d]+))?", re.IGNORECASE
)
_RE_ARTICLE_NO = re.compile(
    r"\[article no\.?\]\s*(?P<art>[\w\d\-]+)", re.IGNORECASE
)

# Year: take first standalone 4-digit year (between 1900 and current+1)
_RE_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Citation counts (one per source). "do <date>: ... TC: X, ... CI: Y, ... CIAu: Z"
# Both WoS and Scopus follow the same Slovenian-language template.
def _make_citation_re(label: str) -> re.Pattern[str]:
    """Build a regex that captures TC/CI/CIAu following ``label``."""
    # The "do <date>:" prefix is optional (older records may show only "Scopus")
    return re.compile(
        rf"\b{label}\b"
        r"(?:[^\[\]]*?do\s+(?P<date>\d{1,2}\.\s*\d{1,2}\.\s*\d{4}))?"
        r"[^\[\]]*?\(TC\):\s*(?P<tc>\d+)"
        r"[^\[\]]*?\(CI\):\s*(?P<ci>\d+)"
        r"[^\[\]]*?\(CIAu\):\s*(?P<ciau>[\d.]+)",
        re.IGNORECASE | re.DOTALL,
    )


_RE_WOS_CITATIONS = _make_citation_re("WoS")
_RE_SCOPUS_CITATIONS = _make_citation_re("Scopus")

# Funding line pattern (Slovenian "projekt: ... ; financer: ..."); one per line.
_RE_FUNDING_LINE = re.compile(
    r"projekt:\s*(?P<project>[^;\n]+?)(?:;\s*financer:\s*(?P<funder>[^\n]+))?$",
    re.IGNORECASE | re.MULTILINE,
)
_RE_AWARD_LINE = re.compile(r"nagrada:\s*(?P<award>[^\n]+)", re.IGNORECASE)

# Authors run: a sequence of UPPERCASE-NAME, FirstName tokens separated by comma+space,
# possibly with role tags in parentheses like "(avtor, korespondenčni avtor)".
# We anchor at the start of the record body and stop when we hit a sentence-ending
# period followed by a capitalised word that does not look like an author name
# (i.e., heuristic; we treat everything up to the first "." that ends a sentence
# of authors).
#
# More robust: match all "SURNAME, FirstName" pairs and join them.
_RE_AUTHOR_PAIR = re.compile(
    # surname can be ALL CAPS, hyphenated, with diacritics
    r"\b([A-ZČĆŽŠĐÁÉÍÓÚÝÀÈÌÒÙÜÖÄ][A-ZČĆŽŠĐÁÉÍÓÚÝÀÈÌÒÙÜÖÄ\-' ]{1,40}?),\s*"
    # first name has at least one capital + lower
    r"([A-ZČĆŽŠĐ][a-zčćžšđáéíóúýàèìòùüöäA-ZČĆŽŠĐ\-' ]+?)"
    # role tag in parens (optional)
    r"(?:\s*\([^)]+\))?"
    # delimiter to next author or end
    r"(?=,\s*[A-ZČĆŽŠĐ]|\.|;)"
)

# "et al." marker
_RE_ET_AL = re.compile(r"\bet\s+al\.?", re.IGNORECASE)


# =============================================================================
# HELPERS
# =============================================================================


def _strip_markdown_links(text: str) -> str:
    """Replace ``[label](url)`` with ``label``, and ``<url>`` with the URL itself."""
    text = _RE_MARKDOWN_LINK.sub(lambda m: m.group(1), text)
    text = _RE_BARE_URL.sub(lambda m: m.group(1), text)
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse soft line breaks but keep paragraph breaks (double newlines)."""
    # Convert single newlines into spaces but keep double-newlines as paragraph
    # separators (these mark the boundary between record-body and funding lines).
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _html_to_text(html: str) -> str:
    """
    Convert raw HTML to a flattened text representation that is regex-friendly.

    We aim for output similar in shape to the markdown that COBISS+ would
    produce so the same parser works in both cases:

    - section headers prefixed with their level (``# `` for H1, etc.)
    - block-level elements separated by blank lines
    - inline anchors flattened to ``text (url)`` form
    - whitespace normalised
    """
    if not BS4_AVAILABLE:
        raise ImportError(
            "BeautifulSoup4 is required to parse raw HTML. "
            "Install with: pip install beautifulsoup4"
        )
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Replace anchors with "<text>(<url>)" so DOI/Cobiss links survive get_text()
    for a in soup.find_all("a"):
        href = a.get("href", "")
        label = a.get_text(strip=True)
        if href:
            a.replace_with(f"[{label}]({href})")
        else:
            a.replace_with(label)

    # Italic formatting (used for source titles) -- wrap in *...*
    for em in soup.find_all(["i", "em"]):
        em.replace_with(f"*{em.get_text(strip=True)}*")

    # Headings
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            h.insert_before("\n\n" + "#" * level + " ")
            h.insert_after("\n\n")

    # Paragraphs and list items: ensure newlines around them
    for tag in soup.find_all(["p", "li", "div", "br"]):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text(separator=" ")
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalise_pages(p1: Optional[str], p2: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (page_start, page_end, page_range_string)."""
    if p1 is None:
        return None, None, None
    if p2 is None:
        return p1, None, p1
    return p1, p2, f"{p1}-{p2}"


def _safe_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _safe_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# =============================================================================
# RECORD PARSER
# =============================================================================


def _parse_authors(text: str) -> Tuple[Optional[str], bool]:
    """
    Extract authors from the *leading* portion of a record's body.

    Rules:
    - Authors live in the first ~600 chars before the title.
    - Each author is "SURNAME, FirstName" with the surname in caps.
    - The list ends at the first ". " that introduces the title (the
      character following ". " is uppercase but is not another caps surname,
      i.e. the next token is mixed-case like "Volunteer").
    - "et al." may appear and is preserved as a flag in the output.

    Returns ``(authors_string, has_et_al)`` with authors joined by ``"; "``,
    or ``(None, False)`` when the leading authors cannot be parsed.
    """
    # Only look at the first 600 chars; the title-then-rest of the record
    # follows very quickly after the author list.
    head = text[:600]

    # Find the boundary where authors end. An author entry has the shape
    # "SURNAME, Firstname" with the surname in (mostly) caps. Authors are
    # separated by ", " and the *list* ends with ". ". Detect that by
    # finding the first ". " whose next non-space char is a capital letter
    # followed by lowercase (i.e. a normal English word starting the title),
    # not by another all-caps surname.
    #
    # We do this by walking through the leading caps-name pattern.
    boundary = None
    pos = 0
    surname_pat = re.compile(
        r"\s*([A-ZČĆŽŠĐÁÉÍÓÚÝÀÈÌÒÙÜÖÄ][A-ZČĆŽŠĐÁÉÍÓÚÝÀÈÌÒÙÜÖÄ\-' ]{1,40}?),\s*"
        # First name: starts with capital + lower, then keep eating word chars,
        # spaces, hyphens, apostrophes -- as long as the name continues.
        # Stop at the first ", " (next author), "." (end of authors), ";" or "(".
        r"([A-ZČĆŽŠĐ][a-zčćžšđáéíóúýàèìòùüöä][\wčćžšđáéíóúýàèìòùüöäA-ZČĆŽŠĐ\-' ]*[\wčćžšđáéíóúýàèìòùüöä])"
        r"(?:\s*\([^)]+\))?"
    )
    pairs: List[Tuple[str, str]] = []
    while pos < len(head):
        m = surname_pat.match(head, pos)
        if not m:
            # Maybe "et al." marker?
            et_m = re.match(r"\s*et\s+al\.?", head[pos:], re.IGNORECASE)
            if et_m and pairs:
                pos += et_m.end()
                # Author list ends with et al. -- look for following ". "
                end_m = re.match(r"\s*[,;.]?\s*", head[pos:])
                if end_m:
                    pos += end_m.end()
                boundary = pos
                # has_et_al
                authors = "; ".join(f"{s.strip().title()}, {f.strip()}" for s, f in pairs)
                authors += "; et al."
                return authors, True
            break
        pairs.append((m.group(1), m.group(2)))
        pos = m.end()
        # After a successful match, expect either ", " (next author) or ". " (end of list).
        # ", " -> continue. Anything else (".", end-of-line, etc.) -> end of authors.
        sep_m = re.match(r"\s*,\s*", head[pos:])
        if sep_m:
            # Commas continue ONLY if the next chunk starts with another caps surname.
            # Peek ahead: the next surname must start with two consecutive caps letters.
            after_comma = head[pos + sep_m.end():]
            if re.match(
                r"[A-ZČĆŽŠĐÁÉÍÓÚÝÀÈÌÒÙÜÖÄ]{2,}",
                after_comma,
            ) or re.match(r"et\s+al", after_comma, re.IGNORECASE):
                pos += sep_m.end()
                continue
            # Otherwise, authors end here.
            boundary = pos
            break
        # Period or other punctuation -> author list ends
        boundary = pos
        break

    if not pairs:
        return None, False

    has_et_al = bool(_RE_ET_AL.search(head[: (boundary or len(head))]))
    formatted = "; ".join(f"{s.strip().title()}, {f.strip()}" for s, f in pairs)
    if has_et_al:
        formatted += "; et al."
    return formatted, has_et_al


def _parse_title_and_source(record_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract the article title, the source (journal name), and the conference
    container (if any).

    For journal articles the structure is:
        ``AUTHORS. Title : subtitle. *Source*. Year, vol. X ...``
    For conference papers the structure is:
        ``AUTHORS. Title : subtitle. V: *Container*. Place: Publisher ...``
    For book chapters with editors:
        ``AUTHORS. Title. V: EDITOR (ur.). *Container*. ...``

    Returns ``(title, source_title, conference)``.
    """
    # All italicised spans -- the FIRST is the proper source title.
    italics = list(re.finditer(r"\*([^*]+)\*", record_text))
    if not italics:
        return None, None, None

    # Use first italic span as the canonical source name
    src_match = italics[0]
    source = src_match.group(1).strip().rstrip(".")

    # Locate the title: text between authors-end and the source italic.
    pre_source = record_text[: src_match.start()]

    # Special pattern: "V: *Container*" (book/conference contribution).
    # In that case the *first* italic is the container, and the title sits before " V: ".
    is_in_container = bool(re.search(r"\bV:\s*$", pre_source.strip()) or
                           re.search(r"\.\s*V:\s*\*", record_text[: src_match.end() + 1]))
    if is_in_container:
        # Cut off everything from " V:" onward in pre_source
        cut = re.search(r"\.\s*V:\s*", pre_source)
        if cut:
            pre_source = pre_source[: cut.start() + 1]
        conference = source  # the italicised name *is* the conference / book
        # No journal source for conference papers / chapters
        # (Let downstream logic decide whether to copy `Conference` into
        # `Source title` for compatibility with existing biblium readers.)
        source = None
    else:
        conference = None

    # Find the title: last "sentence" inside pre_source that follows the author block.
    # Author block ends at the first ". " whose next char is the title's leading capital.
    # Heuristic: skip the first ". " that follows a capitalised name, then take the rest
    # up to the source italic.
    #
    # We look for ". " followed by a Title-ish chunk, then ". " (the period before the
    # source). Implementation: find the rightmost ". " that is at least 5 chars before
    # the source italic and use everything after it.
    pre_source_stripped = pre_source.rstrip().rstrip(".")
    # Strip the leading author run by removing everything up to the FIRST sentence
    # boundary that is followed by a non-author capital (i.e. a regular Title word
    # like "Volunteer", not "STANIMIROVIĆ").
    # A simple rule: find the first ". " whose next non-space char starts with capital
    # then lowercase letter, and take everything after.
    title = None
    title_search = re.search(
        r"\.\s+([A-ZČĆŽŠĐ][a-zčćžšđáéíóúýàèìòùüöä][^*]*?)\s*$",
        pre_source_stripped,
        re.DOTALL,
    )
    if title_search:
        title = title_search.group(1).strip()
    else:
        # Fallback: take the last 200 chars before the source italic and clean
        candidate = pre_source_stripped[-300:].strip()
        # Drop leading author-ish portion if any
        candidate = re.sub(
            r"^.*?[A-ZČĆŽŠĐÁÉÍÓÚÝ]{2,}[^.]*?\.\s+",
            "",
            candidate,
            count=1,
        )
        title = candidate.strip() or None

    if title:
        title = title.replace("\n", " ")
        title = re.sub(r"\s+", " ", title).strip(" .:;,")

    return title, source, conference


def _parse_one_record(
    text: str,
    record_index: int,
    typology_code: Optional[str],
    default_citation_source: Literal["wos", "scopus"] = "wos",
) -> ParsedCobissRecord:
    """
    Parse the body of one bibliographic record.
    """
    # Strip markdown link syntax for the regex pass; keep urls separate
    body = _strip_markdown_links(text)
    body = _collapse_whitespace(body)

    rec = ParsedCobissRecord(
        record_index=record_index,
        cobiss_typology_code=typology_code,
        raw_text=text.strip(),
    )
    if typology_code:
        rec.cobiss_typology_label_sl = typology_label(typology_code, "sl")
        rec.cobiss_typology_label_en = typology_label(typology_code, "en")
        rec.Document_Type = typology_to_document_type(typology_code)

    # ---- Authors / title / source ----
    authors, _ = _parse_authors(body)
    rec.Authors = authors

    title, source, conference = _parse_title_and_source(body)
    rec.Title = title
    rec.Source_title = source
    rec.Conference = conference

    # ---- Year ----
    # Year is best taken from the part *after* the source title (publication date).
    src_match = re.search(r"\*([^*]+)\*", body)
    after_source = body[src_match.end():] if src_match else body
    year_m = _RE_YEAR.search(after_source)
    if year_m:
        rec.Year = int(year_m.group(1))

    # ---- Volume / issue / pages / article number ----
    vol_m = _RE_VOLUME.search(body)
    if vol_m:
        rec.Volume = vol_m.group("vol")

    iss_m = _RE_ISSUE.search(body)
    if iss_m:
        rec.Issue = iss_m.group("iss")

    pg_m = _RE_PAGES.search(body)
    if pg_m:
        ps, pe, pr = _normalise_pages(pg_m.group("p1"), pg_m.group("p2"))
        rec.Page_start, rec.Page_end, rec.Pages = ps, pe, pr

    art_m = _RE_ARTICLE_NO.search(body)
    if art_m:
        rec.Article_no = art_m.group("art")

    # ---- Identifiers ----
    issn_m = _RE_ISSN.search(body)
    if issn_m:
        rec.ISSN = issn_m.group(1)

    isbn_m = _RE_ISBN.search(body)
    if isbn_m:
        rec.ISBN = isbn_m.group(1)

    doi_m = _RE_DOI.search(body)
    if doi_m:
        doi = doi_m.group(1).rstrip(").,;").strip()
        # Strip accidental "doi.org/" prefix
        doi = re.sub(r"^doi\.org/", "", doi, flags=re.IGNORECASE)
        rec.DOI = doi

    cob_m = _RE_COBISS_ID.search(body)
    if cob_m:
        rec.cobiss_id = cob_m.group(1)

    # ---- Repository / open-access flags ----
    rec.cobiss_open_access = bool(re.search(r"Odprti dostop", body, re.IGNORECASE))
    rec.Open_Access = rec.cobiss_open_access
    rec.cobiss_dCOBISS = bool(re.search(r"\bdCOBISS\b", body))
    rec.cobiss_RUL = bool(re.search(r"\bRUL\b", body))
    rec.cobiss_dLib = bool(re.search(r"dLib\.si", body, re.IGNORECASE))

    # ---- Citations (WoS / Scopus) ----
    wos_m = _RE_WOS_CITATIONS.search(body)
    if wos_m:
        rec.cobiss_wos_tc = _safe_int(wos_m.group("tc"))
        rec.cobiss_wos_ci = _safe_int(wos_m.group("ci"))
        rec.cobiss_wos_ciau = _safe_float(wos_m.group("ciau"))
        rec.cobiss_wos_date = wos_m.group("date")

    sco_m = _RE_SCOPUS_CITATIONS.search(body)
    if sco_m:
        rec.cobiss_scopus_tc = _safe_int(sco_m.group("tc"))
        rec.cobiss_scopus_ci = _safe_int(sco_m.group("ci"))
        rec.cobiss_scopus_ciau = _safe_float(sco_m.group("ciau"))
        rec.cobiss_scopus_date = sco_m.group("date")

    # Cited by = preferred source's TC. None if unavailable.
    if default_citation_source == "scopus":
        rec.Cited_by = rec.cobiss_scopus_tc
    else:
        rec.Cited_by = rec.cobiss_wos_tc

    # ---- Funding & awards ----
    funds = []
    for m in _RE_FUNDING_LINE.finditer(text):
        proj = m.group("project").strip()
        funder = (m.group("funder") or "").strip()
        if funder:
            funds.append(f"{proj} (funder: {funder})")
        else:
            funds.append(proj)
    if funds:
        rec.Funding = "; ".join(funds)

    awards = [m.group("award").strip() for m in _RE_AWARD_LINE.finditer(text)]
    if awards:
        rec.Awards = "; ".join(awards)

    # ---- First URL (canonical landing page or DOI link) ----
    url_match = re.search(r"https?://[^\s)>\]]+", text)
    if url_match:
        rec.URL = url_match.group(0).rstrip(").,;")

    return rec


# =============================================================================
# DOCUMENT-LEVEL PARSER
# =============================================================================


@dataclass
class _ParseMetadata:
    """Metadata about the parsed document (researcher, period, ...)."""
    researcher_name: Optional[str] = None
    researcher_code: Optional[str] = None
    period: Optional[str] = None
    n_records: int = 0
    n_records_per_typology: Dict[str, int] = field(default_factory=dict)


def _detect_typology_blocks(text: str) -> List[Tuple[str, int, int]]:
    """
    Find all typology section headers in the document.

    Returns a list of ``(code, start_pos, end_pos_of_header)`` tuples,
    sorted by position. ``end_pos_of_header`` is the offset where the
    record bodies start.
    """
    blocks: List[Tuple[str, int, int]] = []
    for m in _RE_TYPOLOGY_HEADER.finditer(text):
        code = m.group("code").strip()
        # Sanity check: must be a known typology code
        if typology_label(code) is None:
            continue
        blocks.append((code, m.start(), m.end()))
    return blocks


def _split_into_records(
    section_text: str,
    typology_code: str,
    default_citation_source: Literal["wos", "scopus"],
) -> List[ParsedCobissRecord]:
    """
    Split a typology section into individual records and parse each.

    Records are separated by lines containing only an ordinal like ``"42."``.
    """
    matches = list(_RE_RECORD_START.finditer(section_text))
    out: List[ParsedCobissRecord] = []
    for i, m in enumerate(matches):
        record_index = int(m.group(1))
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        body = section_text[body_start:body_end].strip()
        if not body:
            continue
        try:
            rec = _parse_one_record(
                body, record_index, typology_code, default_citation_source
            )
            out.append(rec)
        except Exception as exc:  # never let a single record break the whole import
            warnings.warn(
                f"Could not parse record {record_index} (typology {typology_code}): "
                f"{type(exc).__name__}: {exc}",
                UserWarning, stacklevel=2,
            )
    return out


# =============================================================================
# PUBLIC ENTRY POINTS
# =============================================================================


def parse_cobiss_html(
    source: str,
    *,
    default_citation_source: Literal["wos", "scopus"] = "wos",
    is_path: Optional[bool] = None,
    is_html: Optional[bool] = None,
) -> Tuple[List[ParsedCobissRecord], _ParseMetadata]:
    """
    Parse a COBISS+ personal bibliography document.

    Parameters
    ----------
    source : str
        One of:
        - raw HTML string,
        - pre-extracted plain text / markdown string,
        - or a path to a local file containing either.
    default_citation_source : {"wos", "scopus"}, default "wos"
        Which citation source to copy into the canonical ``Cited by``
        column. Both raw counts are kept in ``cobiss_wos_*`` and
        ``cobiss_scopus_*`` regardless of this choice.
    is_path : bool, optional
        Force interpretation of ``source`` as a file path. By default,
        the function auto-detects this when the string is short and
        points to an existing file.
    is_html : bool, optional
        Force HTML parsing (as opposed to text/markdown). By default,
        we look for ``<html`` or ``<body`` substrings.

    Returns
    -------
    records : list of ParsedCobissRecord
    metadata : _ParseMetadata
    """
    # Resolve source -> text payload
    if is_path is None:
        is_path = (
            isinstance(source, str)
            and len(source) < 4096
            and "\n" not in source
            and os.path.exists(source)
        )
    if is_path:
        with open(source, encoding="utf-8", errors="replace") as fh:
            payload = fh.read()
    else:
        payload = source

    if is_html is None:
        is_html = bool(re.search(r"<html|<body|<!DOCTYPE", payload, re.IGNORECASE))

    if is_html:
        text = _html_to_text(payload)
    else:
        text = payload

    # ---- Document metadata: researcher name & code, period ----
    meta = _ParseMetadata()

    # Heading like "# dr. Lan Umek [28519]"
    name_m = re.search(
        r"^\s*#?\s*(.+?)\s*\[(\d+)\]\s*$", text.split("\n", 1)[0], re.MULTILINE
    )
    if not name_m:
        # Search anywhere in the first 500 chars
        head = text[:500]
        name_m = re.search(r"^([^\n\[]{2,80})\s*\[(\d+)\]", head, re.MULTILINE)
    if name_m:
        meta.researcher_name = name_m.group(1).strip().lstrip("# ").strip()
        meta.researcher_code = name_m.group(2)

    period_m = re.search(
        r"Osebna bibliografija za obdobje\s+([\d\-]+)", text
    )
    if period_m:
        meta.period = period_m.group(1)

    # ---- Records ----
    blocks = _detect_typology_blocks(text)
    records: List[ParsedCobissRecord] = []
    for i, (code, _start, header_end) in enumerate(blocks):
        section_end = blocks[i + 1][1] if i + 1 < len(blocks) else len(text)
        section_text = text[header_end:section_end]
        section_records = _split_into_records(
            section_text, code, default_citation_source
        )
        records.extend(section_records)
        meta.n_records_per_typology[code] = len(section_records)

    meta.n_records = len(records)

    # Sort by record_index to preserve listing order
    records.sort(key=lambda r: (r.record_index or 10**9))
    return records, meta


def records_to_dataframe(
    records: List[ParsedCobissRecord],
    *,
    drop_raw_text: bool = True,
) -> pd.DataFrame:
    """
    Convert parsed records to a biblium-compatible DataFrame.

    Column ordering mirrors the Scopus reader output where possible
    (``Authors``, ``Title``, ``Year``, ``Source title``, ``Volume``,
    ``Issue``, ``Page start``, ``Page end``, ``Pages``, ``Article no``,
    ``ISSN``, ``ISBN``, ``DOI``, ``Document Type``, ``Cited by``,
    ``Open Access``, ...) followed by COBISS-specific fields.
    """
    if not records:
        return pd.DataFrame()

    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows)

    if drop_raw_text and "raw_text" in df.columns:
        df = df.drop(columns=["raw_text"])

    # Rename underscore field names to the spaced forms biblium uses elsewhere
    rename_map = {
        "Source_title": "Source title",
        "Page_start": "Page start",
        "Page_end": "Page end",
        "Article_no": "Article no",
        "Document_Type": "Document Type",
        "Cited_by": "Cited by",
        "Open_Access": "Open Access",
    }
    df = df.rename(columns=rename_map)

    # Preferred column order
    preferred = [
        "Authors", "Title", "Year", "Source title", "Volume", "Issue",
        "Page start", "Page end", "Pages", "Article no",
        "ISSN", "ISBN", "DOI", "Document Type", "Cited by", "Open Access",
        "Editors", "Publisher", "Conference", "URL",
        "Funding", "Awards",
        "cobiss_id", "cobiss_typology_code",
        "cobiss_typology_label_sl", "cobiss_typology_label_en",
        "cobiss_wos_tc", "cobiss_wos_ci", "cobiss_wos_ciau", "cobiss_wos_date",
        "cobiss_scopus_tc", "cobiss_scopus_ci", "cobiss_scopus_ciau",
        "cobiss_scopus_date",
        "cobiss_open_access", "cobiss_dCOBISS", "cobiss_RUL", "cobiss_dLib",
        "record_index",
    ]
    cols = [c for c in preferred if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]


__all__ = [
    "ParsedCobissRecord",
    "parse_cobiss_html",
    "records_to_dataframe",
]
