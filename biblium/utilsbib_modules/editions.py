# -*- coding: utf-8 -*-
"""
biblium.utilsbib_modules.editions
=================================

Confidence-tiered attribution of a citing document's raw reference
strings to specific editions of a multi-edition target work
(e.g. Pollitt & Bouckaert, Public Management Reform, 1st-4th eds).

Motivation
----------
Bibliometric edition-analysis pipelines commonly assign a citing
document to a cited edition using only the year that appears in the
Scopus reference string and a year-window table. That is fragile:
year windows overlap real reprints, later reprints of earlier editions,
and years missing from the reference string. The functions here provide
a hierarchy of signals (ISBN -> unique subtitle -> title+year ->
publisher+year -> inferred year window) and record which tier fired,
so downstream analyses can restrict to high-confidence attributions.

Public functions
----------------
attribute_cited_work_editions
    Attribute the reference strings of a single citing document to
    one or more editions of a target work, with a confidence tier.

build_editions_spec_pmr
    Ready-to-use `editions_spec` dictionary for Pollitt & Bouckaert
    Public Management Reform (1st ed. 2000, 2nd ed. 2004,
    3rd ed. 2011, 4th ed. 2017).

Author: Lan Umek
Version: 2.16.0 (merged into 2.19)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Ordered from strongest to weakest evidence.
_TIER_ORDER: Tuple[str, ...] = (
    "exact_isbn",
    "exact_subtitle",
    "high_title_year",
    "good_publisher_year",
    "inferred_year_window",
    "none",
)
_TIER_RANK: Dict[str, int] = {t: i for i, t in enumerate(_TIER_ORDER)}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _extract_years(segment: str, year_regex: str = r"(\d{4})") -> List[int]:
    """Return all plausible publication-year integers in ``segment``."""
    years: List[int] = []
    for m in re.finditer(year_regex, segment):
        try:
            y = int(m.group(1))
        except (ValueError, IndexError):
            continue
        if 1900 <= y <= 2100:
            years.append(y)
    return years


def _matches_any(segment: str, patterns: Optional[Sequence[str]]) -> bool:
    if not patterns:
        return False
    for p in patterns:
        if not p:
            continue
        try:
            if re.search(p, segment, flags=re.IGNORECASE):
                return True
        except re.error:
            # Fall back to plain substring if the pattern is not a valid regex.
            if p.lower() in segment.lower():
                return True
    return False


def _isbn_matches(segment: str, isbn_patterns: Optional[Sequence[str]]) -> bool:
    if not isbn_patterns:
        return False
    seg_norm = re.sub(r"[^0-9Xx]", "", segment)
    for p in isbn_patterns:
        p_norm = re.sub(r"[^0-9Xx]", "", str(p))
        if p_norm and p_norm in seg_norm:
            return True
    return False


def _year_in_range(years: Sequence[int],
                   target_year: Optional[int],
                   tolerance: int = 1) -> bool:
    if target_year is None or not years:
        return False
    return any(abs(y - target_year) <= tolerance for y in years)


def _year_in_window(years: Sequence[int],
                    window: Optional[Tuple[int, int]]) -> bool:
    if window is None or not years:
        return False
    lo, hi = window
    return any(lo <= y <= hi for y in years)


def _best_tier(current: str, candidate: str) -> str:
    """Return the higher-priority (lower-rank) of the two tiers."""
    return candidate if _TIER_RANK[candidate] < _TIER_RANK[current] else current


# ---------------------------------------------------------------------------
# Segment-level tier decision
# ---------------------------------------------------------------------------
def _classify_segment(
    segment: str,
    editions_spec: Dict[str, Dict[str, Any]],
    year_regex: str,
) -> Tuple[Optional[str], str]:
    """
    Decide the best (edition_key, tier) for a single reference segment.

    Returns (None, "none") if no tier fired for any edition.
    """
    years = _extract_years(segment, year_regex)

    best: Tuple[Optional[str], str] = (None, "none")

    for ed_key, spec in editions_spec.items():
        title_rx = spec.get("title_regex")
        author_rx = spec.get("author_regex")

        has_title = bool(title_rx) and bool(re.search(title_rx, segment,
                                                      flags=re.IGNORECASE))
        has_author = bool(author_rx) and bool(re.search(author_rx, segment,
                                                        flags=re.IGNORECASE))

        # Tier 1: ISBN
        if _isbn_matches(segment, spec.get("isbn_patterns")):
            cand = (ed_key, "exact_isbn")
            if _TIER_RANK[cand[1]] < _TIER_RANK[best[1]]:
                best = cand
                continue  # cannot do better for this edition

        # Tier 2: subtitle unique to this edition
        if _matches_any(segment, spec.get("subtitle_patterns")):
            cand = (ed_key, "exact_subtitle")
            if _TIER_RANK[cand[1]] < _TIER_RANK[best[1]]:
                best = cand
                continue

        # For tiers 3-5 we need at least a title or author hit to bind the
        # segment to *this work at all*.
        if not (has_title or has_author):
            continue

        # Tier 3: title regex + edition year (+/-1)
        if has_title and _year_in_range(years, spec.get("year")):
            cand = (ed_key, "high_title_year")
            if _TIER_RANK[cand[1]] < _TIER_RANK[best[1]]:
                best = cand
                continue

        # Tier 4: publisher name + edition year
        if (_matches_any(segment, spec.get("publisher_patterns"))
                and _year_in_range(years, spec.get("year"))):
            cand = (ed_key, "good_publisher_year")
            if _TIER_RANK[cand[1]] < _TIER_RANK[best[1]]:
                best = cand
                continue

        # Tier 5: year falls inside the between-editions window
        if has_title and _year_in_window(years, spec.get("year_range")):
            cand = (ed_key, "inferred_year_window")
            if _TIER_RANK[cand[1]] < _TIER_RANK[best[1]]:
                best = cand
                continue

    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def attribute_cited_work_editions(
    ref_strings: Iterable[str],
    editions_spec: Dict[str, Dict[str, Any]],
    year_regex: str = r"(\d{4})",
) -> Dict[str, Any]:
    """
    Attribute one citing document's raw reference strings to editions
    of a target work.

    Parameters
    ----------
    ref_strings : iterable of str
        Raw reference segments for one citing document (as returned by
        splitting a Scopus ``References`` field on ``;``).
    editions_spec : dict
        Mapping ``{edition_key: {...}}`` where each inner dict may contain:

        - ``"year"``: int, the publication year of the edition
        - ``"isbn_patterns"``: list[str], ISBN substrings (digits & X only)
        - ``"subtitle_patterns"``: list[str], subtitle regex patterns unique
          to this edition
        - ``"title_regex"``: str, regex identifying the work's title
        - ``"author_regex"``: str, regex identifying the (co-)author name
        - ``"publisher_patterns"``: list[str], regex patterns for the
          publisher name
        - ``"year_range"``: tuple(int, int), inclusive window for the
          inferred-year fallback tier

    year_regex : str, default ``r"(\\d{4})"``
        Regex whose first capture group extracts a year from a segment.

    Returns
    -------
    dict
        ``{
            "editions_attributed": list[str],   # unique keys of editions hit
            "confidence": str,                  # highest-tier match overall
            "segments": [
                {"segment": raw_str, "edition": key, "tier": tier_str},
                ...
            ],
        }``

        The ``confidence`` field is the strongest tier fired across all
        segments; it is ``"none"`` when no segment matched any edition
        signal.
    """
    if editions_spec is None:
        editions_spec = {}

    seg_details: List[Dict[str, Any]] = []
    editions_hit: List[str] = []
    best_tier_overall = "none"

    for raw in ref_strings:
        if raw is None:
            continue
        seg = str(raw).strip()
        if not seg:
            continue
        ed_key, tier = _classify_segment(seg, editions_spec, year_regex)
        if ed_key is not None:
            seg_details.append({"segment": seg, "edition": ed_key, "tier": tier})
            if ed_key not in editions_hit:
                editions_hit.append(ed_key)
            best_tier_overall = _best_tier(best_tier_overall, tier)

    return {
        "editions_attributed": editions_hit,
        "confidence": best_tier_overall,
        "segments": seg_details,
    }


def build_editions_spec_pmr() -> Dict[str, Dict[str, Any]]:
    """
    Ready-to-use editions_spec for Pollitt & Bouckaert,
    *Public Management Reform* (Oxford University Press).

    Editions
    --------
    - ``"1"``  1st ed. (2000), subtitle "A Comparative Analysis"
    - ``"2"``  2nd ed. (2004), same subtitle "A Comparative Analysis"
    - ``"3"``  3rd ed. (2011), unique subtitle
      "New Public Management, Governance, and the Neo-Weberian State"
    - ``"4"``  4th ed. (2017), unique subtitle "Into the Age of Austerity"

    Users may extend each entry (add ISBNs, tighten regexes, etc.).
    """
    title_rx = r"public\s+management\s+reform"
    author_rx = r"pollitt"
    publisher_pats = [
        r"oxford\s+univ", r"\boup\b", r"oxford\s+university\s+press",
    ]
    return {
        "1": {
            "year": 2000,
            "isbn_patterns": [],
            # 1st ed. shares the subtitle with 2nd; distinguish via year.
            "subtitle_patterns": [],
            "title_regex": title_rx,
            "author_regex": author_rx,
            "publisher_patterns": publisher_pats,
            "year_range": (1999, 2003),
        },
        "2": {
            "year": 2004,
            "isbn_patterns": [],
            "subtitle_patterns": [],
            "title_regex": title_rx,
            "author_regex": author_rx,
            "publisher_patterns": publisher_pats,
            "year_range": (2004, 2010),
        },
        "3": {
            "year": 2011,
            "isbn_patterns": [],
            "subtitle_patterns": [
                r"new\s+public\s+management,?\s+governance",
                r"neo[-\s]?weberian",
            ],
            "title_regex": title_rx,
            "author_regex": author_rx,
            "publisher_patterns": publisher_pats,
            "year_range": (2011, 2016),
        },
        "4": {
            "year": 2017,
            "isbn_patterns": [],
            "subtitle_patterns": [
                r"into\s+the\s+age\s+of\s+austerity",
            ],
            "title_regex": title_rx,
            "author_regex": author_rx,
            "publisher_patterns": publisher_pats,
            "year_range": (2017, 2030),
        },
    }


__all__ = [
    "attribute_cited_work_editions",
    "build_editions_spec_pmr",
]
