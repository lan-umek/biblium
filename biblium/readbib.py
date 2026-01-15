# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 12:18:25 2025

@author: Lan.Umek
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

fd = os.path.dirname(__file__)
# REFACTORED: Used os.path.join
df0 = pd.read_excel(os.path.join(fd, "additional files", "variable names.xlsx"), sheet_name="names")

"""Helpers for reading Web of Science (WoS) exports in various formats (Excel, text, BibTeX)."""
# WOS

# Regular expression to detect WoS tag lines in the .txt export
_TAG_LINE_RE = re.compile(r'^([A-Z0-9]{2,3})\s+(.*)$')

def create_name_mapper(
    df: pd.DataFrame,
    key_column: str,
) -> Callable[[Any], Any]:
    """
    Create a mapper function that maps values from a specified column to the 'name' column.

    Args:
        df (pd.DataFrame): DataFrame containing a 'name' column and the key_column.
        key_column (str): The name of the column whose values you want to map.

    Returns:
        Callable[[Any], Any]: A function that takes a value (from key_column) and returns the corresponding
                              'name' value from the same row, or the original value if not found.
    """
    # Build mapping from key_column values to name values
    mapping = dict(zip(df[key_column], df["name"]))

    def mapper(
        value: Any,
    ) -> Any:
        """
        Map a single value to its corresponding 'name', or return it unchanged if missing.

        Args:
            value (Any): A value to look up in key_column.

        Returns:
            Any: The mapped 'name' value, or the original value if not found.
        """
        return mapping.get(value, value)

    return mapper

"""Helpers for reading OpenAlex CSV exports and mapping columns using a shared variable-name table."""
# OPEN ALEX

"""Helpers for reading OpenAlex CSV exports and mapping columns using a shared variable-name table."""
# Open Alex

def read_oa_csv(
    filepath: str,
    mapping_column: str = 'open alex',
):
    """
    Read an OpenAlex CSV export and return a DataFrame with standardized column names.
    
    Parameters
    ----------
    filepath : str
        Path to the OpenAlex CSV file.
    mapping_column : str, optional
        Column in variable names mapping file to use for renaming.
        Default is 'open alex'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.
    """
    # Load the Excel file (auto-detect engine)
    df = pd.read_csv(filepath, dtype=str, decimal=".", na_values=["", "NA", "NaN", "n/a", "-", "?"])
    # Drop any columns that are entirely missing
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c], errors="raise")
            except Exception:
                # If any value can't be parsed, keep original column
                pass

    return df


def read_oa_xlsx(
    filepath: str,
    mapping_column: str = 'open alex',
) -> pd.DataFrame:
    """
    Read an OpenAlex Excel (.xlsx) export and return a DataFrame
    with columns mapped to standardized names.
    
    OpenAlex Excel exports typically include columns like:
    - id, doi, title, display_name
    - publication_year, publication_date
    - type, cited_by_count
    - authorships, primary_location, host_venue
    - concepts, topics, keywords
    - abstract_inverted_index (may need special handling)
    - referenced_works, related_works
    
    Parameters
    ----------
    filepath : str
        Path to the OpenAlex Excel file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names. If provided,
        a mapper is constructed via create_name_mapper(df0, mapping_column)
        and applied to rename columns. Default is 'open alex'.

    Returns
    -------
    pd.DataFrame
        DataFrame with OpenAlex columns, no all-NA columns,
        potentially with renamed columns and numeric conversions applied.
    """
    # Load the Excel file (auto-detect engine based on extension)
    df = pd.read_excel(filepath, dtype=str)
    # Drop any columns that are entirely missing
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    # Attempt to convert numeric columns
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c], errors="raise")
            except Exception:
                # If any value can't be parsed, keep original column
                pass

    return df

"""Utilities for resolving OpenAlex links/IDs via the OpenAlex API and normalizing payloads."""
#  additional reading from OpenAlex

"""
Utilities to parse OpenAlex links, detect entity type (work/author/source/institution/concept/publisher/funder),
fetch their metadata from the OpenAlex API, and aggregate results into a pandas DataFrame.

Supported input forms (case-insensitive):
- "https://openalex.org/W4285719527" (short)
- "https://openalex.org/works/W4285719527" (with entity path)
- "https://api.openalex.org/works/W4285719527" (API URL)
- Bare IDs like "W4285719527" also work.

Notes:
- Map of ID prefixes -> entity endpoints:
  W→works, A→authors, S→sources, I→institutions, C→concepts, P→publishers, F→funders.
- The function extracts a compact, type-specific summary plus common fields (id, display_name/title, url).
- Set `mailto` if you have a contact email to use OpenAlex's polite pool (recommended).
"""

_OPENALEX_PREFIX_TO_ENDPOINT = {
    "W": "works",
    "A": "authors",
    "S": "sources",       # venues/journals/conferences/repositories
    "I": "institutions",
    "C": "concepts",
    "P": "publishers",
    "F": "funders",
}

def _extract_openalex_key(
    s: str,
) -> Tuple[str, str]:
    """
    Return (prefix_letter, full_key) from a string that contains an OpenAlex link or ID.
    Raises ValueError if not recognized.

    Examples
    --------
    _extract_openalex_key("https://openalex.org/W4285719527") -> ("W", "W4285719527")
    _extract_openalex_key("https://openalex.org/works/W2741809807") -> ("W", "W2741809807")
    _extract_openalex_key("https://api.openalex.org/authors/A1969205037") -> ("A", "A1969205037")
    _extract_openalex_key("S1234567") -> ("S", "S1234567")
    """
    if not isinstance(s, str) or not s.strip():
        raise ValueError("Input must be a non-empty string.")
    s = s.strip()

    m = re.search(r"/([WASICPF]\d{3,})\b", s, flags=re.IGNORECASE)
    if m:
        key = m.group(1).upper()
        prefix = key[0]
        if prefix in _OPENALEX_PREFIX_TO_ENDPOINT:
            return prefix, key

    m2 = re.fullmatch(r"([WASICPF]\d{3,})", s.strip(), flags=re.IGNORECASE)
    if m2:
        key = m2.group(1).upper()
        prefix = key[0]
        if prefix in _OPENALEX_PREFIX_TO_ENDPOINT:
            return prefix, key

    raise ValueError(f"Not a recognizable OpenAlex link or ID: {s!r}")

def _build_api_url(
    prefix: str,
    key: str,
) -> str:
    """
    Build the OpenAlex API URL for a given prefix and key.
    """
    endpoint = _OPENALEX_PREFIX_TO_ENDPOINT[prefix]
    return f"https://api.openalex.org/{endpoint}/{key}"

def _get_json(
    url: str,
    mailto: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    GET a JSON resource with simple retry and optional polite pool mailto.
    Retries on HTTP 429/5xx with exponential backoff.
    """
    params = {}
    if mailto:
        params["mailto"] = mailto

    delay = 1.0
    for attempt in range(1, max_retries + 1):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            time.sleep(delay)
            delay *= 2
            continue
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"OpenAlex request failed ({r.status_code}) for {url}: {detail}")

# ---------- Helpers to flatten and enrich ----------

def _invert_abstract_index(
    inv: Optional[Dict[str, List[int]]],
) -> Optional[str]:
    """
    Reconstruct the abstract text from OpenAlex 'abstract_inverted_index'.
    Returns None if not present.
    """
    if not inv or not isinstance(inv, dict):
        return None
    # Find max index to size list
    max_pos = 0
    for positions in inv.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    words = [None] * (max_pos + 1)
    for word, positions in inv.items():
        for p in positions:
            if 0 <= p < len(words):
                words[p] = word
    return " ".join(w for w in words if isinstance(w, str))

def _semijoin(
    values: Optional[List[Any]],
) -> Optional[str]:
    """
    Join a list into a semicolon-separated string, JSON-dumping any dicts for readability.
    Returns None for falsy inputs.
    """
    if not values:
        return None
    out = []
    for v in values:
        if isinstance(v, (dict, list)):
            out.append(json.dumps(v, ensure_ascii=False))
        else:
            out.append(str(v))
    return "; ".join(out)

def _pluck(
    d: Dict[str, Any],
    keys: List[str],
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Copy selected keys from dict d. Optionally add a prefix to output keys.
    Missing keys map to None.
    """
    out = {}
    for k in keys:
        out[(prefix + k) if prefix else k] = d.get(k)
    return out

def _safe_lower_id(
    url_like: Optional[str],
    prefix: str,
) -> Optional[str]:
    """
    Normalize an external ID URL by removing a known URL prefix, returning the bare ID.
    """
    if not url_like or not isinstance(url_like, str):
        return None
    if url_like.startswith(prefix):
        return url_like[len(prefix):]
    return url_like

# ---------- Normalizers: compact (original) & rich (new) ----------

def _normalize_entity_compact(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Your original compact normalization with light highlights; kept for backward compatibility.
    """
    entity_id = payload.get("id")
    openalex_key = entity_id.rsplit("/", 1)[-1] if isinstance(entity_id, str) else None
    entity_type = payload.get("type") or payload.get("entity_type")
    obj = {
        "openalex_id": entity_id,
        "openalex_key": openalex_key,
        "api_url": payload.get("api_url"),
        "display_name": payload.get("display_name") or payload.get("title"),
        "entity_type_hint": entity_type,
    }

    if openalex_key and openalex_key.startswith("W"):
        obj.update({
            "kind": "work",
            "title": payload.get("title"),
            "publication_year": payload.get("publication_year"),
            "doi": (payload.get("doi") or "").lower() if payload.get("doi") else None,
            "host_venue": (payload.get("host_venue") or {}).get("display_name"),
            "host_venue_id": (payload.get("host_venue") or {}).get("id"),
            "type": payload.get("type"),
            "open_access": (payload.get("open_access") or {}).get("is_oa"),
            "cited_by_count": payload.get("cited_by_count"),
            "authorships": "; ".join(a["author"]["display_name"] for a in payload.get("authorships", []) if a.get("author")),
        })
    elif openalex_key and openalex_key.startswith("A"):
        inst = payload.get("last_known_institution") or {}
        obj.update({
            "kind": "author",
            "orcid": _safe_lower_id(payload.get("orcid"), "https://orcid.org/") if payload.get("orcid") else None,
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
            "last_known_institution": inst.get("display_name"),
            "last_known_institution_id": inst.get("id"),
            "h_index": payload.get("summary_stats", {}).get("h_index"),
        })
    elif openalex_key and openalex_key.startswith("S"):
        host_org = payload.get("host_organization") or {}
        obj.update({
            "kind": "source",
            "issn_l": payload.get("issn_l"),
            "issn": "; ".join(payload.get("issn", []) or []),
            "type": payload.get("type"),
            "publisher": (payload.get("host_organization_name") or host_org) if isinstance(host_org, str) else payload.get("host_organization_name"),
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
            "is_in_doaj": payload.get("is_in_doaj"),
        })
    elif openalex_key and openalex_key.startswith("I"):
        ror = payload.get("ror")
        obj.update({
            "kind": "institution",
            "ror": _safe_lower_id(ror, "https://ror.org/") if isinstance(ror, str) else None,
            "country_code": payload.get("country_code"),
            "type": payload.get("type"),
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
        })
    elif openalex_key and openalex_key.startswith("C"):
        obj.update({
            "kind": "concept",
            "wikidata": _safe_lower_id(payload.get("wikidata"), "https://www.wikidata.org/wiki/") if payload.get("wikidata") else None,
            "level": payload.get("level"),
            "description": payload.get("description"),
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
        })
    elif openalex_key and openalex_key.startswith("P"):
        obj.update({
            "kind": "publisher",
            "alternate_titles": "; ".join(payload.get("alternate_titles", []) or []),
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
        })
    elif openalex_key and openalex_key.startswith("F"):
        obj.update({
            "kind": "funder",
            "ror": _safe_lower_id(payload.get("ror"), "https://ror.org/") if payload.get("ror") else None,
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
        })
    else:
        obj["kind"] = "unknown"

    if not obj.get("openalex_id") and openalex_key:
        obj["openalex_id"] = f"https://openalex.org/{openalex_key}"
    return obj

def _normalize_entity_rich(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Rich normalization: flatten top-level keys and expand the most useful nested structures.
    Designed to be "tidy-ish" for pandas while still human-readable.

    Notes
    -----
    * For works, reconstructs the abstract; flattens host_venue, locations, best_oa_location,
      primary_location, authorships (+ institutions), concepts and topics, IDs, biblio, and counts_by_year.
    * For other entities, surfaces common metadata such as country/geo, IDs, and summary stats.
    * List-valued fields are rendered as semicolon-separated strings when appropriate; complex
      objects are JSON-encoded inside those strings to avoid lossy flattening.
    """
    out: Dict[str, Any] = {}

    # Common root
    out.update({
        "openalex_id": payload.get("id"),
        "openalex_key": payload.get("id", "").rsplit("/", 1)[-1] if payload.get("id") else None,
        "api_url": payload.get("api_url"),
        "display_name": payload.get("display_name") or payload.get("title"),
        "entity_type_hint": payload.get("type") or payload.get("entity_type"),
        "updated_date": payload.get("updated_date"),
        "created_date": payload.get("created_date"),
    })

    key = out.get("openalex_key") or ""
    prefix = key[:1]

    # ---------- Works ----------
    if prefix == "W":
        # Simple scalar copies
        out.update(_pluck(payload, [
            "title", "original_title", "publication_year", "publication_date",
            "type", "type_crossref", "language", "cited_by_count", "is_paratext"
        ], prefix=None))

        # IDs block
        ids = payload.get("ids", {}) or {}
        out.update({
            "doi": (payload.get("doi") or "").lower() if payload.get("doi") else None,
            "doi_id": _safe_lower_id(ids.get("doi"), "https://doi.org/") if ids.get("doi") else None,
            "pmid": _safe_lower_id(ids.get("pmid"), "https://pubmed.ncbi.nlm.nih.gov/") if ids.get("pmid") else None,
            "pmcid": _safe_lower_id(ids.get("pmcid"), "https://www.ncbi.nlm.nih.gov/pmc/articles/") if ids.get("pmcid") else None,
            "arxiv_id": _safe_lower_id(ids.get("arxiv"), "https://arxiv.org/abs/") if ids.get("arxiv") else None,
            "mag_id": ids.get("mag"),
        })

        # Bibliographic fields
        biblio = payload.get("biblio", {}) or {}
        out.update(_pluck(biblio, ["volume", "issue", "first_page", "last_page"], prefix="biblio_"))

        # Host venue and OA
        host = payload.get("host_venue", {}) or {}
        out.update({
            "host_venue_id": host.get("id"),
            "host_venue_display_name": host.get("display_name"),
            "host_venue_type": host.get("type"),
            "host_venue_publisher": host.get("publisher"),
            "host_venue_is_oa": host.get("is_oa"),
            "host_venue_issn": _semijoin(host.get("issn")),
            "host_venue_url": host.get("url"),
        })

        oa = payload.get("open_access", {}) or {}
        out.update({
            "oa_is_oa": oa.get("is_oa"),
            "oa_oa_status": oa.get("oa_status"),
            "oa_oa_url": oa.get("oa_url"),
        })

        best_oa = payload.get("best_oa_location", {}) or {}
        out.update({
            "best_oa_location_source": best_oa.get("source", {}).get("display_name") if isinstance(best_oa.get("source"), dict) else None,
            "best_oa_location_landing_page_url": best_oa.get("landing_page_url"),
            "best_oa_location_pdf_url": best_oa.get("pdf_url"),
            "best_oa_location_version": best_oa.get("version"),
        })

        prim = payload.get("primary_location", {}) or {}
        out.update({
            "primary_location_source": prim.get("source", {}).get("display_name") if isinstance(prim.get("source"), dict) else None,
            "primary_location_landing_page_url": prim.get("landing_page_url"),
            "primary_location_pdf_url": prim.get("pdf_url"),
            "primary_location_version": prim.get("version"),
        })

        # All locations (JSON rows joined)
        locations = payload.get("locations", []) or []
        out["locations_all"] = _semijoin(locations)

        # Authorships (+ institutions)
        auths = payload.get("authorships", []) or []
        out["authors_all"] = _semijoin([
            {
                "author_id": a.get("author", {}).get("id"),
                "author_key": (a.get("author", {}).get("id") or "").rsplit("/", 1)[-1] if a.get("author", {}).get("id") else None,
                "author_name": a.get("author", {}).get("display_name"),
                "author_position": a.get("author_position"),
                "raw_affiliation_string": a.get("raw_affiliation_string"),
                "institutions": [
                    {
                        "institution_id": ins.get("id"),
                        "institution_key": (ins.get("id") or "").rsplit("/", 1)[-1] if ins.get("id") else None,
                        "institution_name": ins.get("display_name"),
                        "ror": _safe_lower_id(ins.get("ror"), "https://ror.org/") if ins.get("ror") else None,
                        "country_code": ins.get("country_code")
                    } for ins in a.get("institutions", []) or []
                ],
            } for a in auths
        ])

        # Concepts and topics
        concepts = payload.get("concepts", []) or []
        out["concepts_all"] = _semijoin([
            {"id": c.get("id"), "key": (c.get("id") or "").rsplit("/", 1)[-1] if c.get("id") else None,
             "display_name": c.get("display_name"), "level": c.get("level"), "score": c.get("score")}
            for c in concepts
        ])

        topics = payload.get("topics", []) or []
        # Newer OpenAlex "topics" structure (domain/field/subfield hierarchy)
        out["topics_all"] = _semijoin([
            {
                "id": t.get("id"),
                "key": (t.get("id") or "").rsplit("/", 1)[-1] if t.get("id") else None,
                "display_name": t.get("display_name"),
                "score": t.get("score"),
                "subfield": (t.get("subfield", {}) or {}).get("display_name"),
                "field": (t.get("field", {}) or {}).get("display_name"),
                "domain": (t.get("domain", {}) or {}).get("display_name"),
            } for t in topics
        ])

        # Related/referenced works
        out["referenced_works"] = _semijoin(payload.get("referenced_works"))
        out["related_works"] = _semijoin(payload.get("related_works"))

        # Abstract reconstruction
        abstract = _invert_abstract_index(payload.get("abstract_inverted_index"))
        out["abstract"] = abstract

        # Counts by year (JSON rows joined)
        cby = payload.get("counts_by_year", []) or []
        out["counts_by_year"] = _semijoin(cby)

        # Primary topic (if present)
        pt = payload.get("primary_topic") or {}
        out.update({
            "primary_topic_id": pt.get("id"),
            "primary_topic_display_name": pt.get("display_name"),
            "primary_topic_score": pt.get("score"),
        })

    # ---------- Authors ----------
    elif prefix == "A":
        out.update(_pluck(payload, [
            "works_count", "cited_by_count", "display_name_alternatives",
            "last_known_institution", "summary_stats"
        ]))
        inst = payload.get("last_known_institution") or {}
        out.update({
            "orcid": _safe_lower_id(payload.get("orcid"), "https://orcid.org/") if payload.get("orcid") else None,
            "last_known_institution_id": inst.get("id") if isinstance(inst, dict) else None,
            "last_known_institution_name": inst.get("display_name") if isinstance(inst, dict) else None,
            "last_known_institution_ror": _safe_lower_id(inst.get("ror"), "https://ror.org/") if isinstance(inst, dict) and inst.get("ror") else None,
            "h_index": (payload.get("summary_stats") or {}).get("h_index"),
        })

    # ---------- Sources (venues) ----------
    elif prefix == "S":
        out.update(_pluck(payload, [
            "type", "is_in_doaj", "works_count", "cited_by_count",
            "homepage_url", "abbrev", "societies"
        ]))
        out["issn_l"] = payload.get("issn_l")
        out["issn"] = _semijoin(payload.get("issn"))
        host_org = payload.get("host_organization") or {}
        out["host_organization_id"] = host_org.get("id") if isinstance(host_org, dict) else None
        out["host_organization_name"] = host_org.get("display_name") if isinstance(host_org, dict) else payload.get("host_organization_name")

    # ---------- Institutions ----------
    elif prefix == "I":
        out.update(_pluck(payload, [
            "country_code", "type", "works_count", "cited_by_count", "geo", "homepage_url"
        ]))
        out["ror"] = _safe_lower_id(payload.get("ror"), "https://ror.org/") if payload.get("ror") else None

    # ---------- Concepts ----------
    elif prefix == "C":
        out.update(_pluck(payload, [
            "level", "description", "wikidata", "works_count", "cited_by_count"
        ]))
        out["wikidata"] = _safe_lower_id(out.get("wikidata"), "https://www.wikidata.org/wiki/") if out.get("wikidata") else None

    # ---------- Publishers ----------
    elif prefix == "P":
        out.update(_pluck(payload, [
            "alternate_titles", "works_count", "cited_by_count", "homepage_url"
        ]))
        out["alternate_titles"] = _semijoin(payload.get("alternate_titles"))

    # ---------- Funders ----------
    elif prefix == "F":
        out.update(_pluck(payload, [
            "works_count", "cited_by_count", "display_name_alternatives", "homepage_url"
        ]))
        out["ror"] = _safe_lower_id(payload.get("ror"), "https://ror.org/") if payload.get("ror") else None

    # Fallback: ensure human URL
    if not out.get("openalex_id") and key:
        out["openalex_id"] = f"https://openalex.org/{key}"

    return out

def _normalize_entity_web(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize an OpenAlex payload to match what's prominently visible on the
    OpenAlex web page (UI), keeping a concise, human-facing subset.

    Fix
    ---
    Adds the "source" field for Works (and aliases) that mirrors the UI label.
    Falls back from host_venue to primary_location.source if needed.

    Returns
    -------
    dict
        UI-focused normalized fields.
    """
    out: Dict[str, Any] = {
        "openalex_id": payload.get("id"),
        "openalex_key": (payload.get("id") or "").rsplit("/", 1)[-1] if payload.get("id") else None,
        "api_url": payload.get("api_url"),
        "display_name": payload.get("display_name") or payload.get("title"),
        "updated_date": payload.get("updated_date"),
    }
    key = out.get("openalex_key") or ""
    prefix = key[:1]

    def _semijoin(
        vals: Optional[List[Any]],
    ) -> Optional[str]:
        """Join two DataFrames on key column, keeping only matching rows."""
        if not vals:
            return None
        return "; ".join(str(v) for v in vals)

    def _safe_lower(
        url_like: Optional[str],
        prefix_: str,
    ) -> Optional[str]:
        """Safely lowercase and strip a string, removing URL prefix if present."""
        if not url_like or not isinstance(url_like, str):
            return None
        return url_like[len(prefix_):] if url_like.startswith(prefix_) else url_like

    # ---------- Works (W) ----------
    if prefix == "W":
        out.update({
            "title": payload.get("title"),
            "publication_year": payload.get("publication_year"),
            "publication_date": payload.get("publication_date"),
            "type": payload.get("type"),
            "language": payload.get("language"),
            "cited_by_count": payload.get("cited_by_count"),
        })

        # IDs (as in UI)
        ids = payload.get("ids", {}) or {}
        out.update({
            "doi": _safe_lower(ids.get("doi"), "https://doi.org/") if ids.get("doi") else (payload.get("doi") or None),
            "pmid": _safe_lower(ids.get("pmid"), "https://pubmed.ncbi.nlm.nih.gov/") if ids.get("pmid") else None,
            "pmcid": _safe_lower(ids.get("pmcid"), "https://www.ncbi.nlm.nih.gov/pmc/articles/") if ids.get("pmcid") else None,
            "arxiv_id": _safe_lower(ids.get("arxiv"), "https://arxiv.org/abs/") if ids.get("arxiv") else None,
        })

        # Venue + pages (UI shows this as "Source" + biblio)
        host = payload.get("host_venue", {}) or {}
        prim = payload.get("primary_location", {}) or {}
        prim_src = prim.get("source") or {}

        # Prefer host_venue (journal/venue) like the UI; fall back to primary_location.source
        source_name = host.get("display_name") or prim_src.get("display_name")
        source_id = host.get("id") or prim_src.get("id")
        source_type = host.get("type") or prim_src.get("type")
        host_venue_url = host.get("url")

        biblio = payload.get("biblio", {}) or {}
        out.update({
            "host_venue_display_name": host.get("display_name"),
            "host_venue_id": host.get("id"),
            "host_venue_url": host_venue_url,
            # UI-aligned aliases:
            "source": source_name,            # <-- new: what the web page labels as "Source"
            "source_id": source_id,
            "source_type": source_type,
            # biblio shown near Source
            "biblio_volume": biblio.get("volume"),
            "biblio_issue": biblio.get("issue"),
            "biblio_first_page": biblio.get("first_page"),
            "biblio_last_page": biblio.get("last_page"),
        })

        # Open access (UI summary)
        oa = payload.get("open_access", {}) or {}
        best_oa = payload.get("best_oa_location", {}) or {}
        out.update({
            "oa_is_oa": oa.get("is_oa"),
            "oa_status": oa.get("oa_status"),
            "best_oa_landing_page_url": best_oa.get("landing_page_url"),
            "best_oa_pdf_url": best_oa.get("pdf_url"),
        })

        # Authors (names only)
        authorships = payload.get("authorships", []) or []
        out["authors"] = _semijoin([
            (a.get("author") or {}).get("display_name")
            for a in authorships
            if (a.get("author") or {}).get("display_name")
        ])

        # Abstract (if present)
        inv = payload.get("abstract_inverted_index")
        if inv and isinstance(inv, dict):
            max_pos = max((max(v) for v in inv.values() if v), default=-1)
            words = [None] * (max_pos + 1)
            for w, pos_list in inv.items():
                for p in pos_list:
                    if 0 <= p < len(words):
                        words[p] = w
            out["abstract"] = " ".join(filter(None, words))
        else:
            out["abstract"] = None

        # Concepts/Topics (names only)
        concepts = payload.get("concepts", []) or []
        topics = payload.get("topics", []) or []
        out["concepts"] = _semijoin([c.get("display_name") for c in concepts if c.get("display_name")])
        out["topics"] = _semijoin([t.get("display_name") for t in topics if t.get("display_name")])

        # Link counts (sections in UI)
        out["referenced_works_count"] = len(payload.get("referenced_works") or [])
        out["related_works_count"] = len(payload.get("related_works") or [])

        out["kind"] = "work"

    # ---------- Authors (A) ----------
    elif prefix == "A":
        inst = payload.get("last_known_institution") or {}
        out.update({
            "kind": "author",
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
            "h_index": (payload.get("summary_stats") or {}).get("h_index"),
            "last_known_institution_name": inst.get("display_name") if isinstance(inst, dict) else None,
            "last_known_institution_id": inst.get("id") if isinstance(inst, dict) else None,
        })

    # ---------- Sources (S) ----------
    elif prefix == "S":
        out.update({
            "kind": "source",
            "type": payload.get("type"),
            "issn_l": payload.get("issn_l"),
            "issn": _semijoin(payload.get("issn")),
            "is_in_doaj": payload.get("is_in_doaj"),
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
            "homepage_url": payload.get("homepage_url"),
            "publisher": payload.get("host_organization_name"),
        })

    # ---------- Institutions (I) ----------
    elif prefix == "I":
        out.update({
            "kind": "institution",
            "ror": _safe_lower(payload.get("ror"), "https://ror.org/") if payload.get("ror") else None,
            "country_code": payload.get("country_code"),
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
            "homepage_url": payload.get("homepage_url"),
        })

    # ---------- Concepts (C) ----------
    elif prefix == "C":
        out.update({
            "kind": "concept",
            "level": payload.get("level"),
            "description": payload.get("description"),
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
            "wikidata": _safe_lower(payload.get("wikidata"), "https://www.wikidata.org/wiki/") if payload.get("wikidata") else None,
        })

    # ---------- Publishers (P) ----------
    elif prefix == "P":
        out.update({
            "kind": "publisher",
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
            "homepage_url": payload.get("homepage_url"),
        })

    # ---------- Funders (F) ----------
    elif prefix == "F":
        out.update({
            "kind": "funder",
            "ror": _safe_lower(payload.get("ror"), "https://ror.org/") if payload.get("ror") else None,
            "works_count": payload.get("works_count"),
            "cited_by_count": payload.get("cited_by_count"),
            "homepage_url": payload.get("homepage_url"),
        })

    else:
        out["kind"] = "unknown"

    return out

# ---------- Public API ----------

def get_openalex_entity(
    s: str,
    mailto: Optional[str] = None,
    mode: str = 'web',
    include_raw: bool = False,
) -> Dict[str, Any]:
    """
    Parse an OpenAlex link/ID, fetch its API JSON, and return a normalized dict.

    Parameters
    ----------
    s : str
        A string containing an OpenAlex link or bare ID.
    mailto : str, optional
        Contact email to enable OpenAlex's polite pool (recommended).
    mode : {"web", "compact", "rich"}, default "web"
        "web": UI-focused subset (now includes the "source" field for works).
        "compact": earlier concise schema.
        "rich": expanded flattening (many nested fields).
    include_raw : bool, default False
        If True, include the unmodified API payload under "_raw".

    Returns
    -------
    dict
        Normalized entity fields according to the chosen mode.
    """
    prefix, key = _extract_openalex_key(s)
    url = _build_api_url(prefix, key)
    payload = _get_json(url, mailto=mailto)

    if mode == "web":
        out = _normalize_entity_web(payload)
    elif mode == "rich":
        out = _normalize_entity_rich(payload)
    else:
        out = _normalize_entity_compact(payload)

    if include_raw:
        out["_raw"] = payload
    return out

def harvest_openalex_links_to_df(
    links: Iterable[str],
    mailto: Optional[str] = None,
    mode: str = 'web',
    include_raw: bool = False,
) -> pd.DataFrame:
    """
    Fetch OpenAlex entities for links/IDs and return a tidy DataFrame.

    Update
    ------
    Ensures the UI-aligned "source" (plus "source_id", "source_type") is present
    for Works, and included among preferred columns. Also keeps the older
    host_venue_* fields for compatibility.

    Returns
    -------
    pandas.DataFrame
        One row per unique entity (by key), preserving first-seen order.
    """
    seen: set[str] = set()
    rows: List[Dict[str, Any]] = []

    for s in links:
        try:
            prefix, key = _extract_openalex_key(s)
            if key in seen:
                continue
            seen.add(key)

            row = get_openalex_entity(s, mailto=mailto, mode=mode, include_raw=include_raw)
            row["input"] = s
            row.setdefault("kind", {"W": "work", "A": "author", "S": "source", "I": "institution", "C": "concept", "P": "publisher", "F": "funder"}.get(prefix, "unknown"))
            rows.append(row)

        except Exception as e:
            rows.append({
                "input": s,
                "openalex_key": None,
                "kind": "error",
                "error": str(e),
            })

    preferred = [
        "input",
        # universal
        "kind", "display_name", "title",
        "openalex_id", "openalex_key", "api_url",
        # Work UI (now includes "source" alias)
        "publication_year", "publication_date", "type", "language",
        "cited_by_count",
        "authors",
        "abstract",
        # Source (UI label) + compatibility fields
        "source", "source_id", "source_type",
        "host_venue_display_name", "host_venue_id", "host_venue_url",
        # biblio
        "biblio_volume", "biblio_issue", "biblio_first_page", "biblio_last_page",
        # OA
        "oa_is_oa", "oa_status", "best_oa_landing_page_url", "best_oa_pdf_url",
        # IDs
        "doi", "pmid", "pmcid", "arxiv_id",
        # chips
        "concepts", "topics",
        # related counts
        "referenced_works_count", "related_works_count",
        # author/source/institution summaries
        "h_index", "last_known_institution_name", "last_known_institution_id",
        "issn_l", "issn", "is_in_doaj", "publisher", "homepage_url",
        "country_code", "ror", "wikidata",
        "works_count",
        # admin & errors
        "updated_date", "error",
    ]

    all_keys = set().union(*(row.keys() for row in rows)) if rows else set()
    columns = preferred + [k for k in all_keys if k not in preferred]

    df_out = pd.DataFrame(rows, columns=columns)
    return df_out.dropna(axis=1, how="all")

def add_short_and_full_labels(
    df_out: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add "Short Label" and "Label" to df_out using OpenAlex-style fields.

    Rules
    -----
    - Authors are in column "authors", separated by ";".
    - First author is formatted as "Surname F." (only first given-name initial).
    - If there are multiple authors, append " et al.".
    - Short Label: "<AuthorFmt> — <source> (<year>)"
    - Label: Short Label + ": <first 5 words of title>"
    """
    import re

    def format_first_author(
        auths: str,
    ) -> tuple[str, int]:
        """Format author string to extract first author in standard format."""
        # split by ";" and clean
        s = "" if pd.isna(auths) else str(auths)
        parts = [a.strip() for a in s.split(";") if a.strip()]
        n = len(parts)
        if n == 0:
            return "", 0
        a = parts[0]
        tokens = [t for t in re.split(r"\s+", a) if t]
        if not tokens:
            return "", n
        surname = tokens[0]
        initial = (tokens[1][0].upper() + ".") if len(tokens) > 1 else ""
        return (f"{surname} {initial}".strip(), n)

    first_fmt, counts = zip(*df_out.get("authors", pd.Series([""]*len(df_out))).map(format_first_author))
    first_fmt = pd.Series(first_fmt, index=df_out.index)
    counts = pd.Series(counts, index=df_out.index)

    author_disp = first_fmt.where(counts <= 1, first_fmt.str.cat(pd.Series([" et al."]*len(df_out), index=df_out.index)))

    src = df_out.get("source", pd.Series([""]*len(df_out))).fillna("").astype(str).str.strip()
    yr = df_out.get("publication_year", pd.Series([""]*len(df_out)))
    yr_str = yr.apply(lambda v: str(int(v)) if pd.notna(v) and str(v).strip() != "" else "")

    base = author_disp.combine(src, lambda a, s: f"{a}, {s}" if a and s else (a or s))
    short = base.combine(yr_str, lambda b, y: f"{b} ({y})" if y else b)

    title5 = df_out.get("title", pd.Series([""]*len(df_out))).fillna("").str.split().str[:5].str.join(" ")
    label = short.combine(title5, lambda b, t: f"{b}: {t}" if t else b)

    df_out["Short Label"] = short
    df_out["Label"] = label
    return df_out

"""Helpers for reading Web of Science (WoS) exports in various formats (Excel, text, BibTeX)."""
# WOS

def read_wos_xls(
    filepath: str,
    mapping_column: str = 'wos',
) -> pd.DataFrame:
    """
    Read a Web of Science Excel (.xls or .xlsx) export and return a DataFrame
    containing the raw columns from the file, with any all-NA columns dropped,
    optionally remapped based on a mapping column.

    Parameters
    ----------
    filepath : str
        Path to the WoS Excel file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names. If provided,
        a mapper is constructed via create_name_mapper(df0, mapping_column)
        and applied to rename columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw WoS columns, no all-NA columns,
        potentially with renamed columns.
    """
    # Load the Excel file (auto-detect engine)
    df = pd.read_excel(filepath, dtype=str)
    # Drop any columns that are entirely missing
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df

def read_wos_txt(
    filepath: str,
    mapping_column: str = 'wos-abb',
) -> pd.DataFrame:
    """
    Read a Web of Science plain-text export (.txt) and return a DataFrame
    with raw field tags as columns, parsing records separated by blank lines,
    skipping the first two header lines, dropping any all-NA columns,
    and optionally remapping based on a mapping column.

    Parameters
    ----------
    filepath : str
        Path to the WoS .txt file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names. If provided,
        a mapper is constructed via create_name_mapper(df0, mapping_column)
        and applied to rename columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw WoS tags as columns, no all-NA columns,
        potentially with renamed columns.
    """
    records = []
    with open(filepath, encoding='utf-8') as f:
        # Skip first two metadata lines
        next(f, None)
        next(f, None)
        record = {}
        last_tag = None
        for line in f:
            line = line.rstrip('\n')
            # Blank line indicates end of record
            if not line.strip():
                if record:
                    records.append(record)
                    record = {}
                last_tag = None
                continue
            # Match tag lines
            m = _TAG_LINE_RE.match(line)
            if m:
                tag, val = m.groups()
                record[tag] = val
                last_tag = tag
            else:
                # Continuation of previous tag
                if last_tag and last_tag in record:
                    record[last_tag] += ' ' + line.strip()
        # Append last record if present
        if record:
            records.append(record)
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df

def read_wos_bib(
    filepath: str,
    mapping_column: str = 'wos-bib',
) -> pd.DataFrame:
    """
    Read a Web of Science BibTeX export (.bib) and return a DataFrame
    with raw BibTeX fields as columns, all-NA columns dropped,
    and optionally remapped based on a mapping column.

    Parameters
    ----------
    filepath : str
        Path to the WoS .bib file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names. If provided,
        a mapper is constructed via create_name_mapper(df0, mapping_column)
        and applied to rename columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw BibTeX fields as columns, no all-NA columns,
        potentially with renamed columns.
    """
    records = []
    entry = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # New entry on '@'
            if line.startswith('@'):
                if entry:
                    records.append(entry)
                entry = {}
            elif '=' in line:
                key, rest = line.split('=', 1)
                key = key.strip().lower()
                val = rest.strip().rstrip(',').strip('{}').strip()
                entry[key] = val
    # Append last entry
    if entry:
        records.append(entry)
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df

"""Helpers for reading PubMed/MEDLINE exports in text format."""
# PUBMED

# Regular expression to detect PubMed/MEDLINE tag lines
_PUBMED_TAG_LINE_RE = re.compile(r'^([A-Z]{2,4})\s*-\s+(.*)$')

def read_pubmed_txt(
    filepath: str,
    mapping_column: str = 'pubmed',
) -> pd.DataFrame:
    """
    Read a PubMed/MEDLINE plain-text export (.txt or .nbib) and return a DataFrame
    with raw field tags as columns, parsing records separated by blank lines,
    dropping any all-NA columns, and optionally remapping based on a mapping column.
    
    PubMed MEDLINE format uses two-letter field tags like:
    PMID - PubMed ID
    TI   - Title
    AB   - Abstract
    AU   - Author
    AD   - Affiliation
    TA   - Journal abbreviation
    JT   - Full journal title
    DP   - Publication date
    MH   - MeSH terms
    PT   - Publication type
    LA   - Language
    FAU  - Full author name
    etc.

    Parameters
    ----------
    filepath : str
        Path to the PubMed .txt or .nbib file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names. If provided,
        a mapper is constructed via create_name_mapper(df0, mapping_column)
        and applied to rename columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw PubMed tags as columns, no all-NA columns,
        potentially with renamed columns.
    """
    records = []
    with open(filepath, encoding='utf-8') as f:
        record = {}
        last_tag = None
        for line in f:
            line = line.rstrip('\n\r')
            # Blank line indicates end of record
            if not line.strip():
                if record:
                    records.append(record)
                    record = {}
                last_tag = None
                continue
            # Match tag lines (e.g., "PMID- 12345678" or "TI  - Article title")
            m = _PUBMED_TAG_LINE_RE.match(line)
            if m:
                tag, val = m.groups()
                # Multi-value fields like AU, MH are appended with semicolon
                if tag in record:
                    record[tag] += '; ' + val.strip()
                else:
                    record[tag] = val.strip()
                last_tag = tag
            else:
                # Continuation line (starts with spaces)
                if last_tag and last_tag in record and line.startswith('      '):
                    record[last_tag] += ' ' + line.strip()
        # Append last record if present
        if record:
            records.append(record)
    
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df



"""Helpers for reading PubMed CSV and Summary exports."""

def read_pubmed_csv(
    filepath: str,
    mapping_column: str | None = None,
) -> pd.DataFrame:
    """
    Read a PubMed CSV export and return a DataFrame with standardized columns.
    
    PubMed CSV exports include columns like:
    - PMID, Title, Authors, Citation
    - First Author, Journal/Book, Publication Year
    - Create Date, PMCID, NIHMS ID, DOI
    
    Parameters
    ----------
    filepath : str
        Path to the PubMed CSV file.
    mapping_column : str, optional
        Not used for CSV, kept for API consistency.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # Standard column mapping for PubMed CSV
    column_mapping = {
        "PMID": "PMID",
        "Title": "Title",
        "Authors": "Authors",
        "Journal/Book": "Source title",
        "Publication Year": "Year",
        "DOI": "DOI",
        "Citation": "Citation",
        "First Author": "First Author",
        "Create Date": "Create Date",
        "PMCID": "PMCID",
    }
    
    df = df.rename(columns=column_mapping)
    
    # Add missing columns with defaults
    defaults = {
        "Cited by": 0,
        "Document Type": "Article",
        "Abstract": "",
        "Author Keywords": "",
        "Index Keywords": "",
        "Affiliations": "",
        "References": "",
        "Language of Original Document": "English",
        "Open Access": "",
    }
    
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
    
    # Ensure Year is numeric
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    
    return df


def read_pubmed_summary(
    filepath: str,
    mapping_column: str | None = None,
) -> pd.DataFrame:
    """
    Parse PubMed summary text format into a DataFrame.
    
    Format example:
    1: Author A, Author B. Title of article. Journal. Year Mon;Vol(Issue):Pages. 
    doi: 10.xxx/xxx. PMID: 12345678; PMCID: PMC12345678.
    
    Parameters
    ----------
    filepath : str
        Path to the PubMed summary text file.
    mapping_column : str, optional
        Not used, kept for API consistency.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with extracted fields.
    """
    import re
    
    records = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by record number pattern (number followed by colon at start)
    # Handle both newline-separated and continuous entries
    entries = re.split(r'\n(?=\d+:)', content)
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        record = {}
        
        # Remove record number prefix
        entry = re.sub(r'^\d+:\s*', '', entry)
        
        # Join multi-line entries
        entry = ' '.join(entry.split())
        
        # Extract PMID
        pmid_match = re.search(r'PMID:\s*(\d+)', entry)
        if pmid_match:
            record['PMID'] = pmid_match.group(1)
        
        # Extract PMCID
        pmcid_match = re.search(r'PMCID:\s*(PMC\d+)', entry)
        if pmcid_match:
            record['PMCID'] = pmcid_match.group(1)
        
        # Extract DOI
        doi_match = re.search(r'doi:\s*([^\s.]+\.[^\s]+?)(?:\.|;|\s+PMID)', entry, re.IGNORECASE)
        if doi_match:
            record['DOI'] = doi_match.group(1).rstrip('.')
        
        # Extract year (look for 4-digit year pattern after a period)
        year_match = re.search(r'\.\s*(\d{4})\s', entry)
        if year_match:
            record['Year'] = int(year_match.group(1))
        
        # Extract authors and title
        # Pattern: "Authors. Title. Journal. Year..."
        parts = entry.split('. ')
        if len(parts) >= 2:
            # First part is usually authors
            authors_part = parts[0]
            if re.search(r'[A-Z][a-z]+\s+[A-Z]{1,3}[,;]?', authors_part):
                record['Authors'] = authors_part
            
            # Second part is usually title
            if len(parts) >= 2:
                record['Title'] = parts[1]
        
        # Extract journal - typically before the year
        journal_match = re.search(r'\.\s+([A-Z][a-zA-Z\s&]+)\.\s*\d{4}', entry)
        if journal_match:
            record['Source title'] = journal_match.group(1).strip()
        
        if record.get('PMID'):  # Only add if we got at least PMID
            records.append(record)
    
    df = pd.DataFrame(records)
    
    # Add missing columns with defaults
    defaults = {
        "Cited by": 0,
        "Document Type": "Article",
        "Abstract": "",
        "Author Keywords": "",
        "Index Keywords": "",
        "Affiliations": "",
        "References": "",
        "Language of Original Document": "English",
        "Open Access": "",
    }
    
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
    
    return df


def read_pubmed_pmid_list(
    filepath: str,
    mapping_column: str | None = None,
) -> pd.DataFrame:
    """
    Parse PubMed PMID list format (one PMID per line).
    
    Note: This format only contains PMIDs, no other metadata.
    For full analysis, use CSV or MEDLINE format exports.
    
    Parameters
    ----------
    filepath : str
        Path to the PMID list file.
    mapping_column : str, optional
        Not used, kept for API consistency.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with PMID column and default values for other fields.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        pmids = [line.strip() for line in f if line.strip().isdigit()]
    
    df = pd.DataFrame({'PMID': pmids})
    
    # Add minimal columns with defaults
    df["Title"] = ""
    df["Authors"] = ""
    df["Source title"] = ""
    df["Year"] = None
    df["Cited by"] = 0
    df["Document Type"] = "Article"
    
    return df


"""Helpers for reading Dimensions exports in CSV and Excel formats."""
# DIMENSIONS

def read_dimensions_csv(
    filepath: str,
    mapping_column: str = 'dimensions-csv',
) -> pd.DataFrame:
    """
    Read a Dimensions CSV export (Export for bibliometric mapping) and return a DataFrame.
    
    Dimensions CSV exports typically include columns like:
    - Publication ID, DOI, Title, Abstract
    - Authors, Authors (Raw Affiliation), Corresponding Authors
    - Source Title, Publisher, PubYear, Publication Date
    - Publication Type, Open Access
    - Times Cited, Recent Citations, Relative Citation Ratio
    - Fields of Research (ANZSRC 2020), Sustainable Development Goals
    - Cited References (in bibliometric mapping export)

    Parameters
    ----------
    filepath : str
        Path to the Dimensions CSV file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Dimensions columns, potentially with renamed columns.
    """
    df = pd.read_csv(filepath, dtype=str, na_values=["", "NA", "NaN", "n/a", "-", "?"])
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    # Convert numeric columns
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c], errors="raise")
            except Exception:
                pass

    return df


def read_dimensions_xlsx(
    filepath: str,
    mapping_column: str = 'dimensions-xlsx',
) -> pd.DataFrame:
    """
    Read a Dimensions Excel export and return a DataFrame.
    
    Dimensions Excel exports typically include metadata columns like:
    - Publication ID, DOI, Title, Abstract
    - Authors, Affiliations
    - Source Title, Publisher, PubYear
    - Publication Type, Document Type
    - Times Cited, Keywords
    - Fields of Research
    
    Note: The Excel format may contain different metadata than the CSV
    (bibliometric mapping) export. Excel typically includes abstracts
    and keywords but may not include cited references.

    Parameters
    ----------
    filepath : str
        Path to the Dimensions Excel file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Dimensions columns, potentially with renamed columns.
    """
    df = pd.read_excel(filepath, dtype=str)
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


"""Helpers for reading Lens.org exports in CSV and JSON formats."""
# LENS

def read_lens_csv(
    filepath: str,
    mapping_column: str = 'lens',
) -> pd.DataFrame:
    """
    Read a Lens.org CSV export and return a DataFrame.
    
    Lens.org CSV exports typically include columns like:
    - Lens ID, Title, Abstract
    - Publication Year, Publication Date
    - Source Title, Publisher, ISSN
    - Authors, Authors (Raw Affiliation)
    - Document Type, Publication Type
    - Scholarly Citations Count, Patent Citations Count
    - Open Access Status, Open Access Colour
    - Fields of Study, MeSH Terms, Keywords
    - DOI, PMID, PMCID
    - Funding, References, Citing Works

    Parameters
    ----------
    filepath : str
        Path to the Lens.org CSV file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Lens columns, potentially with renamed columns.
    """
    df = pd.read_csv(filepath, dtype=str, na_values=["", "NA", "NaN", "n/a", "-", "?"])
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    # Convert numeric columns
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c], errors="raise")
            except Exception:
                pass

    return df


def read_lens_json(
    filepath: str,
    mapping_column: str = 'lens-json',
) -> pd.DataFrame:
    """
    Read a Lens.org JSON or JSON Lines export and return a DataFrame.
    
    Lens.org JSON exports contain the same fields as CSV but in 
    a structured JSON format, which may preserve nested data better.

    Parameters
    ----------
    filepath : str
        Path to the Lens.org JSON or JSONL file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Lens columns, potentially with renamed columns.
    """
    # Try to detect if it's JSON Lines or regular JSON
    records = []
    with open(filepath, encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # Regular JSON array
            data = json.load(f)
            if isinstance(data, list):
                records = data
            else:
                records = [data]
        else:
            # JSON Lines format
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    # Flatten nested structures for common fields
    flat_records = []
    for rec in records:
        flat_rec = {}
        for key, value in rec.items():
            if isinstance(value, list):
                # Join lists with semicolon
                if all(isinstance(v, str) for v in value):
                    flat_rec[key] = '; '.join(value)
                elif all(isinstance(v, dict) for v in value):
                    # For list of dicts, try to extract display names
                    names = []
                    for v in value:
                        if 'display_name' in v:
                            names.append(v['display_name'])
                        elif 'name' in v:
                            names.append(v['name'])
                        else:
                            names.append(json.dumps(v, ensure_ascii=False))
                    flat_rec[key] = '; '.join(names)
                else:
                    flat_rec[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                # Flatten common nested dicts
                if 'display_name' in value:
                    flat_rec[key] = value['display_name']
                else:
                    flat_rec[key] = json.dumps(value, ensure_ascii=False)
            else:
                flat_rec[key] = value
        flat_records.append(flat_rec)
    
    df = pd.DataFrame(flat_records)
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


"""Helpers for reading Cochrane Library exports in RIS and CSV formats."""
# COCHRANE

# RIS tag patterns
_RIS_TAG_RE = re.compile(r'^([A-Z][A-Z0-9])\s+-\s+(.*)$')

def read_cochrane_ris(
    filepath: str,
    mapping_column: str = 'cochrane-ris',
) -> pd.DataFrame:
    """
    Read a Cochrane Library RIS export and return a DataFrame.
    
    Cochrane RIS exports use standard RIS tags:
    TY - Type of reference
    TI - Title
    AU - Author
    AB - Abstract
    PY - Publication year
    JO/JF - Journal name
    VL - Volume
    IS - Issue
    SP - Start page
    EP - End page
    DO - DOI
    UR - URL
    KW - Keywords
    ER - End of reference

    Parameters
    ----------
    filepath : str
        Path to the Cochrane RIS file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with RIS tags as columns, potentially with renamed columns.
    """
    records = []
    with open(filepath, encoding='utf-8') as f:
        record = {}
        last_tag = None
        for line in f:
            line = line.rstrip('\n\r')
            # End of record marker
            if line.startswith('ER  -') or line.startswith('ER-'):
                if record:
                    records.append(record)
                    record = {}
                last_tag = None
                continue
            # Match tag lines
            m = _RIS_TAG_RE.match(line)
            if m:
                tag, val = m.groups()
                # Multi-value fields like AU, KW are appended with semicolon
                if tag in record:
                    record[tag] += '; ' + val.strip()
                else:
                    record[tag] = val.strip()
                last_tag = tag
            elif line.strip() and last_tag:
                # Continuation line
                record[last_tag] += ' ' + line.strip()
        # Append last record if present
        if record:
            records.append(record)
    
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


def read_cochrane_csv(
    filepath: str,
    mapping_column: str = 'cochrane-csv',
) -> pd.DataFrame:
    """
    Read a Cochrane Library CSV export and return a DataFrame.
    
    Cochrane CSV exports may include columns like:
    - Title, Authors, Abstract
    - Source, Year, Volume, Issue, Pages
    - DOI, URL, Cochrane ID
    - Publication Type

    Parameters
    ----------
    filepath : str
        Path to the Cochrane CSV file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Cochrane columns, potentially with renamed columns.
    """
    df = pd.read_csv(filepath, dtype=str, na_values=["", "NA", "NaN", "n/a", "-", "?"])
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


# =============================================================================
# GENERIC FORMAT READERS (RIS, BibTeX, EndNote XML, Zotero RDF)
# =============================================================================

"""Generic RIS format reader - works with most databases."""

def read_ris(
    filepath: str,
    mapping_column: str = 'ris',
) -> pd.DataFrame:
    """
    Read a generic RIS (Research Information Systems) format file and return a DataFrame.
    
    RIS is a standardized tag format used by many databases including:
    Scopus, EBSCO, ProQuest, JSTOR, PsycINFO, ERIC, EconLit, and many others.
    
    Standard RIS tags:
    TY  - Type of reference
    TI  - Title (primary)
    T1  - Title (alternative)
    AU  - Author
    A1  - Author (alternative)
    PY  - Publication year
    Y1  - Year (alternative)
    AB  - Abstract
    N2  - Abstract (alternative)
    JO  - Journal name
    JF  - Journal full name
    JA  - Journal abbreviation
    VL  - Volume
    IS  - Issue
    SP  - Start page
    EP  - End page
    DO  - DOI
    UR  - URL
    KW  - Keywords
    SN  - ISSN/ISBN
    PB  - Publisher
    LA  - Language
    ER  - End of reference

    Parameters
    ----------
    filepath : str
        Path to the RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with RIS tags as columns.
    """
    records = []
    with open(filepath, encoding='utf-8', errors='replace') as f:
        record = {}
        last_tag = None
        for line in f:
            line = line.rstrip('\n\r')
            # End of record marker
            if line.startswith('ER  -') or line.startswith('ER-'):
                if record:
                    records.append(record)
                    record = {}
                last_tag = None
                continue
            # Match tag lines (e.g., "AU  - Smith, John")
            m = _RIS_TAG_RE.match(line)
            if m:
                tag, val = m.groups()
                val = val.strip()
                # Multi-value fields are appended with semicolon
                if tag in record:
                    record[tag] += '; ' + val
                else:
                    record[tag] = val
                last_tag = tag
            elif line.strip() and last_tag:
                # Continuation line
                record[last_tag] += ' ' + line.strip()
        # Append last record if present
        if record:
            records.append(record)
    
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


"""Generic BibTeX format reader - works with many databases and tools."""

# BibTeX parsing patterns
_BIBTEX_ENTRY_RE = re.compile(r'@(\w+)\s*\{\s*([^,]*),', re.IGNORECASE)
_BIBTEX_FIELD_RE = re.compile(r'^\s*(\w+)\s*=\s*(.*)$')

def read_bibtex(
    filepath: str,
    mapping_column: str = 'bibtex',
) -> pd.DataFrame:
    """
    Read a generic BibTeX format file and return a DataFrame.
    
    BibTeX is a standard bibliography format used by:
    LaTeX, Google Scholar, DBLP, arXiv, Scopus, IEEE Xplore, and many others.
    
    Common BibTeX fields:
    author      - Authors
    title       - Title
    journal     - Journal name
    booktitle   - Conference/book title
    year        - Publication year
    volume      - Volume number
    number      - Issue number
    pages       - Page numbers
    doi         - Digital Object Identifier
    url         - URL
    abstract    - Abstract
    keywords    - Keywords
    publisher   - Publisher
    issn        - ISSN
    isbn        - ISBN

    Parameters
    ----------
    filepath : str
        Path to the BibTeX file (.bib).
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with BibTeX fields as columns.
    """
    records = []
    
    with open(filepath, encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Split into entries
    entries = re.split(r'(?=@\w+\s*\{)', content)
    
    for entry in entries:
        entry = entry.strip()
        if not entry or not entry.startswith('@'):
            continue
        
        record = {}
        
        # Extract entry type and citation key
        entry_match = _BIBTEX_ENTRY_RE.match(entry)
        if entry_match:
            record['_entry_type'] = entry_match.group(1).lower()
            record['_citation_key'] = entry_match.group(2).strip()
        
        # Extract fields - handle multi-line values with braces or quotes
        # Remove the entry header
        field_text = re.sub(r'^@\w+\s*\{[^,]*,', '', entry, count=1)
        
        # Parse fields
        current_field = None
        current_value = []
        brace_depth = 0
        in_quotes = False
        
        lines = field_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line == '}':
                continue
            
            # Check if this is a new field
            field_match = _BIBTEX_FIELD_RE.match(line)
            
            if field_match and brace_depth == 0 and not in_quotes:
                # Save previous field
                if current_field:
                    value = ' '.join(current_value).strip()
                    # Clean up the value
                    value = value.rstrip(',').strip()
                    # Remove outer braces or quotes
                    if (value.startswith('{') and value.endswith('}')) or \
                       (value.startswith('"') and value.endswith('"')):
                        value = value[1:-1]
                    # Clean up LaTeX artifacts
                    value = re.sub(r'\{|\}', '', value)
                    value = value.strip()
                    if value:
                        record[current_field.lower()] = value
                
                current_field = field_match.group(1)
                current_value = [field_match.group(2)]
                
                # Count braces/quotes in the value part
                val_part = field_match.group(2)
                brace_depth = val_part.count('{') - val_part.count('}')
                if val_part.count('"') % 2 == 1:
                    in_quotes = True
            else:
                # Continuation of previous field
                current_value.append(line)
                brace_depth += line.count('{') - line.count('}')
                if line.count('"') % 2 == 1:
                    in_quotes = not in_quotes
        
        # Save last field
        if current_field:
            value = ' '.join(current_value).strip()
            value = value.rstrip(',').strip()
            if (value.startswith('{') and value.endswith('}')) or \
               (value.startswith('"') and value.endswith('"')):
                value = value[1:-1]
            value = re.sub(r'\{|\}', '', value)
            value = value.strip()
            if value:
                record[current_field.lower()] = value
        
        if record and (record.get('title') or record.get('author')):
            records.append(record)
    
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


"""EndNote XML format reader."""

def read_endnote_xml(
    filepath: str,
    mapping_column: str = 'endnote-xml',
) -> pd.DataFrame:
    """
    Read an EndNote XML export file and return a DataFrame.
    
    EndNote XML includes fields like:
    - title, secondary-title
    - authors/author
    - year, pub-dates
    - periodical/full-title
    - volume, number, pages
    - abstract
    - keywords/keyword
    - urls/related-urls
    - electronic-resource-num (DOI)

    Parameters
    ----------
    filepath : str
        Path to the EndNote XML file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with EndNote fields as columns.
    """
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("xml.etree.ElementTree is required for EndNote XML parsing")
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    records = []
    
    # Find all record elements (handle different XML structures)
    record_elements = root.findall('.//record') or root.findall('.//Record')
    
    for rec_elem in record_elements:
        record = {}
        
        # Helper to get text from element
        def get_text(parent, tag):
            elem = parent.find(f'.//{tag}')
            if elem is not None and elem.text:
                return elem.text.strip()
            return None
        
        # Helper to get all texts from multiple elements
        def get_all_texts(parent, tag):
            elems = parent.findall(f'.//{tag}')
            texts = [e.text.strip() for e in elems if e is not None and e.text]
            return '; '.join(texts) if texts else None
        
        # Extract common fields
        record['title'] = get_text(rec_elem, 'title')
        record['secondary_title'] = get_text(rec_elem, 'secondary-title')
        record['authors'] = get_all_texts(rec_elem, 'author')
        record['year'] = get_text(rec_elem, 'year')
        record['journal'] = get_text(rec_elem, 'full-title') or get_text(rec_elem, 'secondary-title')
        record['volume'] = get_text(rec_elem, 'volume')
        record['issue'] = get_text(rec_elem, 'number')
        record['pages'] = get_text(rec_elem, 'pages')
        record['abstract'] = get_text(rec_elem, 'abstract')
        record['keywords'] = get_all_texts(rec_elem, 'keyword')
        record['doi'] = get_text(rec_elem, 'electronic-resource-num')
        record['url'] = get_text(rec_elem, 'url')
        record['isbn'] = get_text(rec_elem, 'isbn')
        record['issn'] = get_text(rec_elem, 'issn')
        record['publisher'] = get_text(rec_elem, 'publisher')
        record['ref_type'] = get_text(rec_elem, 'ref-type')
        
        # Remove None values
        record = {k: v for k, v in record.items() if v is not None}
        
        if record:
            records.append(record)
    
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


"""Zotero RDF format reader."""

def read_zotero_rdf(
    filepath: str,
    mapping_column: str = 'zotero-rdf',
) -> pd.DataFrame:
    """
    Read a Zotero RDF export file and return a DataFrame.
    
    Zotero RDF uses Dublin Core and other namespaces for metadata.

    Parameters
    ----------
    filepath : str
        Path to the Zotero RDF file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Zotero fields as columns.
    """
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("xml.etree.ElementTree is required for Zotero RDF parsing")
    
    # Define namespaces
    namespaces = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'dcterms': 'http://purl.org/dc/terms/',
        'bib': 'http://purl.org/net/biblio#',
        'z': 'http://www.zotero.org/namespaces/export#',
        'foaf': 'http://xmlns.com/foaf/0.1/',
        'prism': 'http://prismstandard.org/namespaces/1.2/basic/',
    }
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    records = []
    
    # Find all item types (Article, Book, etc.)
    for item_type in ['Article', 'Book', 'BookSection', 'ConferencePaper', 'Report', 'Thesis', 'Document']:
        for ns_prefix in ['bib:', 'z:']:
            try:
                items = root.findall(f'.//{ns_prefix}{item_type}', namespaces)
            except:
                items = []
            
            for item in items:
                record = {}
                
                # Extract fields using various namespace prefixes
                def find_text(tag_variants):
                    for tag in tag_variants:
                        try:
                            elem = item.find(tag, namespaces)
                            if elem is not None and elem.text:
                                return elem.text.strip()
                        except:
                            continue
                    return None
                
                record['title'] = find_text(['dc:title', 'dcterms:title'])
                record['abstract'] = find_text(['dc:description', 'dcterms:abstract'])
                record['year'] = find_text(['dc:date', 'dcterms:date', 'prism:publicationDate'])
                record['journal'] = find_text(['prism:publicationName', 'dc:source'])
                record['volume'] = find_text(['prism:volume'])
                record['issue'] = find_text(['prism:number'])
                record['pages'] = find_text(['bib:pages', 'prism:pageRange'])
                record['doi'] = find_text(['dc:identifier', 'prism:doi'])
                record['url'] = find_text(['dc:source', 'prism:url'])
                record['publisher'] = find_text(['dc:publisher', 'dcterms:publisher'])
                record['language'] = find_text(['dc:language'])
                record['item_type'] = item_type
                
                # Get authors
                authors = []
                for creator in item.findall('.//foaf:Person', namespaces):
                    name = creator.find('foaf:name', namespaces)
                    if name is not None and name.text:
                        authors.append(name.text.strip())
                if authors:
                    record['authors'] = '; '.join(authors)
                
                # Get keywords/tags
                keywords = []
                for tag in item.findall('.//dc:subject', namespaces):
                    if tag.text:
                        keywords.append(tag.text.strip())
                if keywords:
                    record['keywords'] = '; '.join(keywords)
                
                # Remove None values
                record = {k: v for k, v in record.items() if v is not None}
                
                if record and record.get('title'):
                    records.append(record)
    
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


# =============================================================================
# SCOPUS ADDITIONAL FORMATS (BibTeX, RIS)
# =============================================================================

"""Scopus BibTeX and RIS format readers."""

def read_scopus_bib(
    filepath: str,
    mapping_column: str = 'scopus-bib',
) -> pd.DataFrame:
    """
    Read a Scopus BibTeX export file and return a DataFrame.
    
    Scopus BibTeX exports include fields like:
    author, title, journal, year, volume, number, pages, doi,
    abstract, keywords, affiliations, document_type, source, etc.

    Parameters
    ----------
    filepath : str
        Path to the Scopus BibTeX file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Scopus BibTeX fields as columns.
    """
    # Use the generic BibTeX reader with Scopus-specific mapping
    df = read_bibtex(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_scopus_ris(
    filepath: str,
    mapping_column: str = 'scopus-ris',
) -> pd.DataFrame:
    """
    Read a Scopus RIS export file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the Scopus RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Scopus RIS fields as columns.
    """
    # Use the generic RIS reader with Scopus-specific mapping
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


# =============================================================================
# IEEE XPLORE
# =============================================================================

"""IEEE Xplore format readers."""

def read_ieee_csv(
    filepath: str,
    mapping_column: str = 'ieee-csv',
) -> pd.DataFrame:
    """
    Read an IEEE Xplore CSV export file and return a DataFrame.
    
    IEEE Xplore CSV exports include columns like:
    - Document Title, Authors, Author Affiliations
    - Publication Title, Publication Year, Date Added To Xplore
    - Volume, Issue, Start Page, End Page
    - Abstract, DOI, ISSN, ISBN
    - Article Citation Count, Reference Count
    - Publisher, Document Identifier

    Parameters
    ----------
    filepath : str
        Path to the IEEE Xplore CSV file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with IEEE fields as columns.
    """
    df = pd.read_csv(filepath, dtype=str, na_values=["", "NA", "NaN", "n/a", "-", "?"])
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    # Convert numeric columns
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c], errors="raise")
            except Exception:
                pass

    return df


def read_ieee_bib(
    filepath: str,
    mapping_column: str = 'ieee-bib',
) -> pd.DataFrame:
    """
    Read an IEEE Xplore BibTeX export file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the IEEE Xplore BibTeX file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with IEEE BibTeX fields as columns.
    """
    df = read_bibtex(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_ieee_ris(
    filepath: str,
    mapping_column: str = 'ieee-ris',
) -> pd.DataFrame:
    """
    Read an IEEE Xplore RIS export file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the IEEE Xplore RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with IEEE RIS fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


# =============================================================================
# PROQUEST
# =============================================================================

"""ProQuest format readers."""

def read_proquest_ris(
    filepath: str,
    mapping_column: str = 'proquest-ris',
) -> pd.DataFrame:
    """
    Read a ProQuest RIS export file and return a DataFrame.
    
    ProQuest exports dissertations, theses, and scholarly journals in RIS format.

    Parameters
    ----------
    filepath : str
        Path to the ProQuest RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with ProQuest fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_proquest_csv(
    filepath: str,
    mapping_column: str = 'proquest-csv',
) -> pd.DataFrame:
    """
    Read a ProQuest CSV export file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the ProQuest CSV file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with ProQuest fields as columns.
    """
    df = pd.read_csv(filepath, dtype=str, na_values=["", "NA", "NaN", "n/a", "-", "?"])
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


# =============================================================================
# EBSCO
# =============================================================================

"""EBSCO format reader (CINAHL, Business Source, Academic Search, etc.)."""

def read_ebsco_ris(
    filepath: str,
    mapping_column: str = 'ebsco-ris',
) -> pd.DataFrame:
    """
    Read an EBSCO RIS export file and return a DataFrame.
    
    Works with EBSCO databases including:
    - CINAHL (nursing and allied health)
    - Business Source Complete
    - Academic Search Complete
    - PsycINFO
    - And many others

    Parameters
    ----------
    filepath : str
        Path to the EBSCO RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with EBSCO fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


# =============================================================================
# DBLP (Computer Science Bibliography)
# =============================================================================

"""DBLP format readers."""

def read_dblp_bib(
    filepath: str,
    mapping_column: str = 'dblp-bib',
) -> pd.DataFrame:
    """
    Read a DBLP BibTeX export file and return a DataFrame.
    
    DBLP is the computer science bibliography database.
    DBLP BibTeX exports include: author, title, journal/booktitle,
    year, volume, pages, doi, url, etc.

    Parameters
    ----------
    filepath : str
        Path to the DBLP BibTeX file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with DBLP fields as columns.
    """
    df = read_bibtex(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_dblp_xml(
    filepath: str,
    mapping_column: str = 'dblp-xml',
) -> pd.DataFrame:
    """
    Read a DBLP XML export file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the DBLP XML file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with DBLP fields as columns.
    """
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("xml.etree.ElementTree is required for DBLP XML parsing")
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    records = []
    
    # DBLP uses various element types for different publication types
    pub_types = ['article', 'inproceedings', 'proceedings', 'book', 'incollection', 
                 'phdthesis', 'mastersthesis', 'www']
    
    for pub_type in pub_types:
        for item in root.findall(f'.//{pub_type}'):
            record = {'_pub_type': pub_type}
            
            # Get key attribute
            if 'key' in item.attrib:
                record['dblp_key'] = item.attrib['key']
            
            # Extract fields
            for field in ['title', 'year', 'journal', 'booktitle', 'volume', 
                         'number', 'pages', 'publisher', 'school', 'ee', 'url']:
                elem = item.find(field)
                if elem is not None and elem.text:
                    record[field] = elem.text.strip()
            
            # Get all authors
            authors = []
            for author in item.findall('author'):
                if author.text:
                    authors.append(author.text.strip())
            if authors:
                record['authors'] = '; '.join(authors)
            
            # ee often contains DOI
            if 'ee' in record and 'doi.org' in record.get('ee', ''):
                record['doi'] = record['ee'].split('doi.org/')[-1]
            
            if record.get('title') or record.get('authors'):
                records.append(record)
    
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


# =============================================================================
# ARXIV
# =============================================================================

"""arXiv format readers."""

def read_arxiv_bib(
    filepath: str,
    mapping_column: str = 'arxiv-bib',
) -> pd.DataFrame:
    """
    Read an arXiv BibTeX export file and return a DataFrame.
    
    arXiv BibTeX exports include: author, title, eprint (arXiv ID),
    archivePrefix, primaryClass, abstract, year, etc.

    Parameters
    ----------
    filepath : str
        Path to the arXiv BibTeX file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with arXiv fields as columns.
    """
    df = read_bibtex(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


# =============================================================================
# SEMANTIC SCHOLAR
# =============================================================================

"""Semantic Scholar format readers."""

def read_semantic_scholar_csv(
    filepath: str,
    mapping_column: str = 'semantic-scholar-csv',
) -> pd.DataFrame:
    """
    Read a Semantic Scholar CSV export file and return a DataFrame.
    
    Semantic Scholar exports include:
    - paperId, title, abstract, year
    - authors, venue, publicationDate
    - citationCount, referenceCount, influentialCitationCount
    - fieldsOfStudy, s2FieldsOfStudy
    - doi, arxivId, pmid

    Parameters
    ----------
    filepath : str
        Path to the Semantic Scholar CSV file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Semantic Scholar fields as columns.
    """
    df = pd.read_csv(filepath, dtype=str, na_values=["", "NA", "NaN", "n/a", "-", "?"])
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    # Convert numeric columns
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c], errors="raise")
            except Exception:
                pass

    return df


def read_semantic_scholar_json(
    filepath: str,
    mapping_column: str = 'semantic-scholar-json',
) -> pd.DataFrame:
    """
    Read a Semantic Scholar JSON export file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the Semantic Scholar JSON file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Semantic Scholar fields as columns.
    """
    records = []
    with open(filepath, encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            data = json.load(f)
            if isinstance(data, list):
                records = data
            else:
                records = [data]
        else:
            # JSON Lines format
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    # Flatten nested structures
    flat_records = []
    for rec in records:
        flat_rec = {}
        for key, value in rec.items():
            if key == 'authors' and isinstance(value, list):
                # Extract author names
                names = [a.get('name', '') for a in value if isinstance(a, dict)]
                flat_rec['authors'] = '; '.join(names)
            elif key == 'fieldsOfStudy' and isinstance(value, list):
                flat_rec['fieldsOfStudy'] = '; '.join(value)
            elif isinstance(value, list):
                if all(isinstance(v, str) for v in value):
                    flat_rec[key] = '; '.join(value)
                else:
                    flat_rec[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                flat_rec[key] = json.dumps(value, ensure_ascii=False)
            else:
                flat_rec[key] = value
        flat_records.append(flat_rec)
    
    df = pd.DataFrame(flat_records)
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


# =============================================================================
# JSTOR
# =============================================================================

"""JSTOR format reader."""

def read_jstor_ris(
    filepath: str,
    mapping_column: str = 'jstor-ris',
) -> pd.DataFrame:
    """
    Read a JSTOR RIS export file and return a DataFrame.
    
    JSTOR provides access to humanities and social sciences literature.

    Parameters
    ----------
    filepath : str
        Path to the JSTOR RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with JSTOR fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


# =============================================================================
# CROSSREF (via API exports)
# =============================================================================

"""CrossRef JSON format reader."""

def read_crossref_json(
    filepath: str,
    mapping_column: str = 'crossref-json',
) -> pd.DataFrame:
    """
    Read a CrossRef JSON export file and return a DataFrame.
    
    CrossRef JSON includes:
    - DOI, title, container-title (journal)
    - author, published-print/published-online dates
    - volume, issue, page
    - abstract, subject, type
    - is-referenced-by-count (citations)
    - reference (cited references)

    Parameters
    ----------
    filepath : str
        Path to the CrossRef JSON file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with CrossRef fields as columns.
    """
    records = []
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different CrossRef JSON structures
    if isinstance(data, dict):
        if 'message' in data:
            # API response format
            items = data['message'].get('items', [data['message']])
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        items = []
    
    for item in items:
        record = {}
        
        # Basic fields
        record['doi'] = item.get('DOI')
        
        # Title (can be a list)
        title = item.get('title', [])
        if isinstance(title, list) and title:
            record['title'] = title[0]
        elif isinstance(title, str):
            record['title'] = title
        
        # Container title (journal name)
        container = item.get('container-title', [])
        if isinstance(container, list) and container:
            record['journal'] = container[0]
        elif isinstance(container, str):
            record['journal'] = container
        
        # Authors
        authors = item.get('author', [])
        if authors:
            author_names = []
            for auth in authors:
                if isinstance(auth, dict):
                    name = f"{auth.get('family', '')} {auth.get('given', '')}".strip()
                    if name:
                        author_names.append(name)
            record['authors'] = '; '.join(author_names)
        
        # Publication date
        pub_date = item.get('published-print', item.get('published-online', {}))
        if isinstance(pub_date, dict) and 'date-parts' in pub_date:
            date_parts = pub_date['date-parts']
            if date_parts and date_parts[0]:
                record['year'] = str(date_parts[0][0])
        
        # Other fields
        record['volume'] = item.get('volume')
        record['issue'] = item.get('issue')
        record['pages'] = item.get('page')
        record['publisher'] = item.get('publisher')
        record['type'] = item.get('type')
        record['cited_by_count'] = item.get('is-referenced-by-count')
        record['issn'] = '; '.join(item.get('ISSN', [])) if item.get('ISSN') else None
        record['isbn'] = '; '.join(item.get('ISBN', [])) if item.get('ISBN') else None
        
        # Abstract
        abstract = item.get('abstract', '')
        if abstract:
            # Remove JATS XML tags if present
            record['abstract'] = re.sub(r'<[^>]+>', '', abstract)
        
        # Subject
        subjects = item.get('subject', [])
        if subjects:
            record['subjects'] = '; '.join(subjects)
        
        # Remove None values
        record = {k: v for k, v in record.items() if v is not None}
        
        if record.get('title') or record.get('doi'):
            records.append(record)
    
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


# =============================================================================
# ORCID (Author publication lists)
# =============================================================================

"""ORCID JSON format reader."""

def read_orcid_json(
    filepath: str,
    mapping_column: str = 'orcid-json',
) -> pd.DataFrame:
    """
    Read an ORCID JSON export file and return a DataFrame.
    
    ORCID JSON exports contain author publication lists with:
    - title, journal-title, type
    - publication-date (year, month, day)
    - external-ids (DOI, PMID, etc.)
    - contributors

    Parameters
    ----------
    filepath : str
        Path to the ORCID JSON file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with ORCID fields as columns.
    """
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    
    # Handle different ORCID JSON structures
    works = []
    if isinstance(data, dict):
        if 'group' in data:
            # Works summary format
            for group in data['group']:
                if 'work-summary' in group:
                    works.extend(group['work-summary'])
        elif 'bulk' in data:
            # Bulk format
            for item in data['bulk']:
                if 'work' in item:
                    works.append(item['work'])
        elif 'work' in data:
            works.append(data['work'])
    elif isinstance(data, list):
        works = data
    
    for work in works:
        record = {}
        
        # Title
        title_obj = work.get('title', {})
        if isinstance(title_obj, dict):
            title_val = title_obj.get('title', {})
            if isinstance(title_val, dict):
                record['title'] = title_val.get('value')
            else:
                record['title'] = title_val
        
        # Journal title
        journal = work.get('journal-title', {})
        if isinstance(journal, dict):
            record['journal'] = journal.get('value')
        else:
            record['journal'] = journal
        
        # Type
        record['type'] = work.get('type')
        
        # Publication date
        pub_date = work.get('publication-date', {})
        if pub_date:
            year = pub_date.get('year', {})
            if isinstance(year, dict):
                record['year'] = year.get('value')
        
        # External IDs (DOI, PMID, etc.)
        ext_ids = work.get('external-ids', {})
        if ext_ids:
            for ext_id in ext_ids.get('external-id', []):
                id_type = ext_id.get('external-id-type', '').lower()
                id_value = ext_id.get('external-id-value')
                if id_type == 'doi':
                    record['doi'] = id_value
                elif id_type == 'pmid':
                    record['pmid'] = id_value
                elif id_type == 'issn':
                    record['issn'] = id_value
        
        # Remove None values
        record = {k: v for k, v in record.items() if v is not None}
        
        if record.get('title'):
            records.append(record)
    
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


# =============================================================================
# SPECIALIZED DATABASES (PsycINFO, ERIC, EconLit, etc.)
# =============================================================================

"""Specialized database readers - most use RIS format."""

def read_psycinfo_ris(
    filepath: str,
    mapping_column: str = 'psycinfo-ris',
) -> pd.DataFrame:
    """
    Read a PsycINFO (APA PsycInfo) RIS export file and return a DataFrame.
    
    PsycINFO covers psychology and behavioral sciences literature.

    Parameters
    ----------
    filepath : str
        Path to the PsycINFO RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with PsycINFO fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_eric_ris(
    filepath: str,
    mapping_column: str = 'eric-ris',
) -> pd.DataFrame:
    """
    Read an ERIC (Education Resources Information Center) RIS export file.
    
    ERIC covers education research literature.

    Parameters
    ----------
    filepath : str
        Path to the ERIC RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with ERIC fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_eric_csv(
    filepath: str,
    mapping_column: str = 'eric-csv',
) -> pd.DataFrame:
    """
    Read an ERIC CSV export file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the ERIC CSV file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with ERIC fields as columns.
    """
    df = pd.read_csv(filepath, dtype=str, na_values=["", "NA", "NaN", "n/a", "-", "?"])
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


def read_econlit_ris(
    filepath: str,
    mapping_column: str = 'econlit-ris',
) -> pd.DataFrame:
    """
    Read an EconLit RIS export file and return a DataFrame.
    
    EconLit covers economics literature.

    Parameters
    ----------
    filepath : str
        Path to the EconLit RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with EconLit fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_mathscinet_bib(
    filepath: str,
    mapping_column: str = 'mathscinet-bib',
) -> pd.DataFrame:
    """
    Read a MathSciNet BibTeX export file and return a DataFrame.
    
    MathSciNet covers mathematics literature.

    Parameters
    ----------
    filepath : str
        Path to the MathSciNet BibTeX file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with MathSciNet fields as columns.
    """
    df = read_bibtex(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_inspec_ris(
    filepath: str,
    mapping_column: str = 'inspec-ris',
) -> pd.DataFrame:
    """
    Read an Inspec RIS export file and return a DataFrame.
    
    Inspec covers physics, engineering, and computing literature.

    Parameters
    ----------
    filepath : str
        Path to the Inspec RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Inspec fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_georef_ris(
    filepath: str,
    mapping_column: str = 'georef-ris',
) -> pd.DataFrame:
    """
    Read a GeoRef RIS export file and return a DataFrame.
    
    GeoRef covers geosciences literature.

    Parameters
    ----------
    filepath : str
        Path to the GeoRef RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with GeoRef fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_cab_abstracts_ris(
    filepath: str,
    mapping_column: str = 'cab-ris',
) -> pd.DataFrame:
    """
    Read a CAB Abstracts RIS export file and return a DataFrame.
    
    CAB Abstracts covers agriculture and life sciences literature.

    Parameters
    ----------
    filepath : str
        Path to the CAB Abstracts RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with CAB Abstracts fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_cinahl_ris(
    filepath: str,
    mapping_column: str = 'cinahl-ris',
) -> pd.DataFrame:
    """
    Read a CINAHL RIS export file and return a DataFrame.
    
    CINAHL covers nursing and allied health literature.

    Parameters
    ----------
    filepath : str
        Path to the CINAHL RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with CINAHL fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_embase_ris(
    filepath: str,
    mapping_column: str = 'embase-ris',
) -> pd.DataFrame:
    """
    Read an Embase RIS export file and return a DataFrame.
    
    Embase covers biomedical and pharmacological literature.

    Parameters
    ----------
    filepath : str
        Path to the Embase RIS file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Embase fields as columns.
    """
    df = read_ris(filepath, mapping_column=None)
    
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)
    
    return df


def read_embase_csv(
    filepath: str,
    mapping_column: str = 'embase-csv',
) -> pd.DataFrame:
    """
    Read an Embase CSV export file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the Embase CSV file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Embase fields as columns.
    """
    df = pd.read_csv(filepath, dtype=str, na_values=["", "NA", "NaN", "n/a", "-", "?"])
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


# =============================================================================
# GENERIC CONFIGURABLE CSV READER
# =============================================================================

def read_generic_csv(
    filepath: str,
    mapping_column: Optional[str] = None,
    delimiter: str = ',',
    encoding: str = 'utf-8',
) -> pd.DataFrame:
    """
    Read a generic CSV file with configurable options.
    
    This reader can be used for any CSV export by specifying
    the appropriate mapping column for column name standardization.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names.
    delimiter : str, optional
        Field delimiter (default: comma).
    encoding : str, optional
        File encoding (default: utf-8).

    Returns
    -------
    pd.DataFrame
        DataFrame with CSV fields as columns.
    """
    df = pd.read_csv(
        filepath, 
        dtype=str, 
        delimiter=delimiter,
        encoding=encoding,
        na_values=["", "NA", "NaN", "n/a", "-", "?"]
    )
    df.dropna(axis=1, how='all', inplace=True)

    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    # Convert numeric columns
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c], errors="raise")
            except Exception:
                pass

    return df


"""Generic helper to read bibliographic files from different databases."""
# general

def read_bibfile(
    f_name,
    db,
):
    """
    Generic function to read bibliographic files from different databases.
    
    Supported databases and formats:
    - Scopus: csv, xlsx, bib, ris
    - WoS (Web of Science): txt, xls/xlsx, bib
    - OpenAlex: csv, xlsx
    - PubMed: txt, nbib
    - Dimensions: csv, xlsx
    - Lens: csv, json, jsonl
    - Cochrane: ris, csv
    - IEEE Xplore: csv, bib, ris
    - ProQuest: ris, csv
    - EBSCO: ris
    - DBLP: bib, xml
    - arXiv: bib
    - Semantic Scholar: csv, json
    - JSTOR: ris
    - CrossRef: json
    - ORCID: json
    - PsycINFO: ris
    - ERIC: ris, csv
    - EconLit: ris
    - MathSciNet: bib
    - Inspec: ris
    - GeoRef: ris
    - CAB Abstracts: ris
    - CINAHL: ris
    - Embase: ris, csv
    - Generic: ris, bib, csv, endnote-xml, zotero-rdf
    
    Parameters
    ----------
    f_name : str
        Path to the bibliographic file.
    db : str
        Database source (case-insensitive).
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the bibliographic data with standardized column names.
    """
    if f_name is None:
        return pd.DataFrame([])
    
    db_lower = db.lower().strip().replace(' ', '').replace('-', '').replace('_', '')
    f_name_lower = f_name.lower()
    
    # =========================================================================
    # MAJOR DATABASES
    # =========================================================================
    
    # Scopus
    if db_lower == "scopus":
        if ".xlsx" in f_name_lower:
            df = pd.read_excel(f_name)
        elif ".csv" in f_name_lower:
            df = pd.read_csv(f_name)
        elif ".bib" in f_name_lower:
            df = read_scopus_bib(f_name)
        elif ".ris" in f_name_lower:
            df = read_scopus_ris(f_name)
        else:
            raise ValueError(f"Unsupported file format for Scopus: {f_name}")
    
    # Web of Science
    elif db_lower == "wos" or db_lower == "webofscience":
        if ".txt" in f_name_lower:
            df = read_wos_txt(f_name)
        elif ".xls" in f_name_lower:
            df = read_wos_xls(f_name)
        elif ".bib" in f_name_lower:
            df = read_wos_bib(f_name)
        else:
            raise ValueError(f"Unsupported file format for WoS: {f_name}")
    
    # OpenAlex
    elif db_lower in ["openalex", "oa"]:
        if ".csv" in f_name_lower:
            df = read_oa_csv(f_name)
        elif ".xlsx" in f_name_lower:
            df = read_oa_xlsx(f_name)
        else:
            raise ValueError(f"Unsupported file format for OpenAlex: {f_name}")
    
    # PubMed
    elif db_lower == "pubmed":
        if ".csv" in f_name_lower:
            df = read_pubmed_csv(f_name)
        elif ".nbib" in f_name_lower:
            df = read_pubmed_txt(f_name)
        elif ".txt" in f_name_lower:
            # Auto-detect: MEDLINE format vs Summary format vs PMID list
            with open(f_name, 'r', encoding='utf-8') as f_check:
                first_lines = f_check.read(500)
            import re
            if re.search(r'^PMID-\s*\d+', first_lines, re.MULTILINE):
                # MEDLINE format
                df = read_pubmed_txt(f_name)
            elif re.match(r'^\d+\s*$', first_lines.split('\n')[0].strip()):
                # PMID list (just numbers)
                df = read_pubmed_pmid_list(f_name)
            elif re.match(r'^\d+:\s*[A-Z]', first_lines):
                # Summary format
                df = read_pubmed_summary(f_name)
            else:
                # Default to MEDLINE
                df = read_pubmed_txt(f_name)
        else:
            raise ValueError(f"Unsupported file format for PubMed: {f_name}. Use .csv, .txt, or .nbib format.")
    
    # Dimensions
    elif db_lower == "dimensions":
        if ".csv" in f_name_lower:
            df = read_dimensions_csv(f_name)
        elif ".xlsx" in f_name_lower or ".xls" in f_name_lower:
            df = read_dimensions_xlsx(f_name)
        else:
            raise ValueError(f"Unsupported file format for Dimensions: {f_name}")
    
    # Lens.org
    elif db_lower == "lens" or db_lower == "lensorg":
        if ".csv" in f_name_lower:
            df = read_lens_csv(f_name)
        elif ".json" in f_name_lower or ".jsonl" in f_name_lower:
            df = read_lens_json(f_name)
        else:
            raise ValueError(f"Unsupported file format for Lens: {f_name}")
    
    # Cochrane Library
    elif db_lower == "cochrane":
        if ".ris" in f_name_lower:
            df = read_cochrane_ris(f_name)
        elif ".csv" in f_name_lower:
            df = read_cochrane_csv(f_name)
        else:
            raise ValueError(f"Unsupported file format for Cochrane: {f_name}. Use .ris or .csv format.")
    
    # =========================================================================
    # ENGINEERING / COMPUTER SCIENCE
    # =========================================================================
    
    # IEEE Xplore
    elif db_lower == "ieee" or db_lower == "ieeexplore":
        if ".csv" in f_name_lower:
            df = read_ieee_csv(f_name)
        elif ".bib" in f_name_lower:
            df = read_ieee_bib(f_name)
        elif ".ris" in f_name_lower:
            df = read_ieee_ris(f_name)
        else:
            raise ValueError(f"Unsupported file format for IEEE Xplore: {f_name}")
    
    # DBLP
    elif db_lower == "dblp":
        if ".bib" in f_name_lower:
            df = read_dblp_bib(f_name)
        elif ".xml" in f_name_lower:
            df = read_dblp_xml(f_name)
        else:
            raise ValueError(f"Unsupported file format for DBLP: {f_name}")
    
    # arXiv
    elif db_lower == "arxiv":
        if ".bib" in f_name_lower:
            df = read_arxiv_bib(f_name)
        else:
            raise ValueError(f"Unsupported file format for arXiv: {f_name}. Use .bib format.")
    
    # =========================================================================
    # MULTIDISCIPLINARY / AI-POWERED
    # =========================================================================
    
    # Semantic Scholar
    elif db_lower == "semanticscholar" or db_lower == "s2":
        if ".csv" in f_name_lower:
            df = read_semantic_scholar_csv(f_name)
        elif ".json" in f_name_lower or ".jsonl" in f_name_lower:
            df = read_semantic_scholar_json(f_name)
        else:
            raise ValueError(f"Unsupported file format for Semantic Scholar: {f_name}")
    
    # CrossRef
    elif db_lower == "crossref":
        if ".json" in f_name_lower:
            df = read_crossref_json(f_name)
        else:
            raise ValueError(f"Unsupported file format for CrossRef: {f_name}. Use .json format.")
    
    # ORCID
    elif db_lower == "orcid":
        if ".json" in f_name_lower:
            df = read_orcid_json(f_name)
        else:
            raise ValueError(f"Unsupported file format for ORCID: {f_name}. Use .json format.")
    
    # =========================================================================
    # AGGREGATORS / MULTI-DATABASE
    # =========================================================================
    
    # ProQuest
    elif db_lower == "proquest":
        if ".ris" in f_name_lower:
            df = read_proquest_ris(f_name)
        elif ".csv" in f_name_lower:
            df = read_proquest_csv(f_name)
        else:
            raise ValueError(f"Unsupported file format for ProQuest: {f_name}")
    
    # EBSCO
    elif db_lower == "ebsco":
        if ".ris" in f_name_lower:
            df = read_ebsco_ris(f_name)
        else:
            raise ValueError(f"Unsupported file format for EBSCO: {f_name}. Use .ris format.")
    
    # JSTOR
    elif db_lower == "jstor":
        if ".ris" in f_name_lower:
            df = read_jstor_ris(f_name)
        else:
            raise ValueError(f"Unsupported file format for JSTOR: {f_name}. Use .ris format.")
    
    # =========================================================================
    # SPECIALIZED SUBJECT DATABASES
    # =========================================================================
    
    # PsycINFO (Psychology)
    elif db_lower == "psycinfo" or db_lower == "apapsycinfo":
        if ".ris" in f_name_lower:
            df = read_psycinfo_ris(f_name)
        else:
            raise ValueError(f"Unsupported file format for PsycINFO: {f_name}. Use .ris format.")
    
    # ERIC (Education)
    elif db_lower == "eric":
        if ".ris" in f_name_lower:
            df = read_eric_ris(f_name)
        elif ".csv" in f_name_lower:
            df = read_eric_csv(f_name)
        else:
            raise ValueError(f"Unsupported file format for ERIC: {f_name}")
    
    # EconLit (Economics)
    elif db_lower == "econlit":
        if ".ris" in f_name_lower:
            df = read_econlit_ris(f_name)
        else:
            raise ValueError(f"Unsupported file format for EconLit: {f_name}. Use .ris format.")
    
    # MathSciNet (Mathematics)
    elif db_lower == "mathscinet":
        if ".bib" in f_name_lower:
            df = read_mathscinet_bib(f_name)
        else:
            raise ValueError(f"Unsupported file format for MathSciNet: {f_name}. Use .bib format.")
    
    # Inspec (Physics/Engineering/Computing)
    elif db_lower == "inspec":
        if ".ris" in f_name_lower:
            df = read_inspec_ris(f_name)
        else:
            raise ValueError(f"Unsupported file format for Inspec: {f_name}. Use .ris format.")
    
    # GeoRef (Geosciences)
    elif db_lower == "georef":
        if ".ris" in f_name_lower:
            df = read_georef_ris(f_name)
        else:
            raise ValueError(f"Unsupported file format for GeoRef: {f_name}. Use .ris format.")
    
    # CAB Abstracts (Agriculture/Life Sciences)
    elif db_lower == "cababstracts" or db_lower == "cab":
        if ".ris" in f_name_lower:
            df = read_cab_abstracts_ris(f_name)
        else:
            raise ValueError(f"Unsupported file format for CAB Abstracts: {f_name}. Use .ris format.")
    
    # CINAHL (Nursing/Allied Health)
    elif db_lower == "cinahl":
        if ".ris" in f_name_lower:
            df = read_cinahl_ris(f_name)
        else:
            raise ValueError(f"Unsupported file format for CINAHL: {f_name}. Use .ris format.")
    
    # Embase (Biomedical/Pharmacological)
    elif db_lower == "embase":
        if ".ris" in f_name_lower:
            df = read_embase_ris(f_name)
        elif ".csv" in f_name_lower:
            df = read_embase_csv(f_name)
        else:
            raise ValueError(f"Unsupported file format for Embase: {f_name}")
    
    # =========================================================================
    # GENERIC FORMATS (auto-detect by file extension)
    # =========================================================================
    
    # Generic RIS
    elif db_lower == "ris" or db_lower == "generic_ris":
        df = read_ris(f_name)
    
    # Generic BibTeX
    elif db_lower == "bibtex" or db_lower == "bib" or db_lower == "generic_bib":
        df = read_bibtex(f_name)
    
    # EndNote XML
    elif db_lower == "endnote" or db_lower == "endnotexml":
        df = read_endnote_xml(f_name)
    
    # Zotero RDF
    elif db_lower == "zotero" or db_lower == "zoterordf":
        df = read_zotero_rdf(f_name)
    
    # Generic CSV
    elif db_lower == "csv" or db_lower == "generic_csv":
        df = read_generic_csv(f_name)
    
    else:
        # Build list of supported databases for error message
        supported = [
            "scopus", "wos", "openalex/oa", "pubmed", "dimensions", "lens", "cochrane",
            "ieee", "dblp", "arxiv", "semantic_scholar/s2", "crossref", "orcid",
            "proquest", "ebsco", "jstor",
            "psycinfo", "eric", "econlit", "mathscinet", "inspec", "georef", 
            "cab_abstracts/cab", "cinahl", "embase",
            "ris", "bibtex/bib", "endnote", "zotero", "csv"
        ]
        raise ValueError(f"Unsupported database: {db}. Supported: {', '.join(supported)}")
    
    return df


# =============================================================================
# REFERENCE STRING PARSING
# =============================================================================

"""
Reference string parsers for various citation styles.
Extracts structured data from unstructured reference strings.
"""

# Common patterns for reference parsing
_YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
_DOI_PATTERN = re.compile(r'10\.\d{4,}/[^\s]+')
_VOLUME_ISSUE_PATTERN = re.compile(r'(\d+)\s*\((\d+)\)')
_PAGES_PATTERN = re.compile(r'(\d+)\s*[-–]\s*(\d+)')
_PMID_PATTERN = re.compile(r'PMID:\s*(\d+)', re.IGNORECASE)


def parse_reference_string(
    ref_string: str,
    style: str = "auto",
) -> Dict[str, Any]:
    """
    Parse a reference string into structured components.
    
    Attempts to extract authors, year, title, journal, volume, issue, 
    pages, and DOI from an unstructured reference string.
    
    Parameters
    ----------
    ref_string : str
        The reference string to parse.
    style : str
        Citation style hint: "auto", "apa", "vancouver", "chicago", 
        "harvard", "mla". Default "auto" tries to detect.
    
    Returns
    -------
    dict
        Parsed reference components:
        - authors: str (raw author string)
        - author_list: list (split authors)
        - year: int or None
        - title: str or None
        - journal: str or None
        - volume: str or None
        - issue: str or None
        - pages: str or None
        - doi: str or None
        - pmid: str or None
        - raw: str (original string)
        - style_detected: str
    
    Examples
    --------
    >>> parse_reference_string("Smith, J., & Jones, A. (2020). Article title. Journal Name, 10(2), 100-120.")
    {'authors': 'Smith, J., & Jones, A.', 'year': 2020, 'title': 'Article title', ...}
    
    >>> parse_reference_string("Smith J, Jones A. Article title. J Name. 2020;10(2):100-120.")
    {'authors': 'Smith J, Jones A', 'year': 2020, 'title': 'Article title', ...}
    """
    if not ref_string or not isinstance(ref_string, str):
        return {"raw": ref_string, "error": "Invalid input"}
    
    ref_string = ref_string.strip()
    result = {
        "raw": ref_string,
        "authors": None,
        "author_list": [],
        "year": None,
        "title": None,
        "journal": None,
        "volume": None,
        "issue": None,
        "pages": None,
        "doi": None,
        "pmid": None,
        "style_detected": None,
    }
    
    # Extract DOI if present
    doi_match = _DOI_PATTERN.search(ref_string)
    if doi_match:
        result["doi"] = doi_match.group(0).rstrip(".")
    
    # Extract PMID if present
    pmid_match = _PMID_PATTERN.search(ref_string)
    if pmid_match:
        result["pmid"] = pmid_match.group(1)
    
    # Extract year
    year_match = _YEAR_PATTERN.search(ref_string)
    if year_match:
        result["year"] = int(year_match.group(0))
    
    # Extract volume/issue
    vol_issue_match = _VOLUME_ISSUE_PATTERN.search(ref_string)
    if vol_issue_match:
        result["volume"] = vol_issue_match.group(1)
        result["issue"] = vol_issue_match.group(2)
    
    # Extract pages
    pages_match = _PAGES_PATTERN.search(ref_string)
    if pages_match:
        result["pages"] = f"{pages_match.group(1)}-{pages_match.group(2)}"
    
    # Detect style and parse accordingly
    if style == "auto":
        style = _detect_citation_style(ref_string)
    
    result["style_detected"] = style
    
    if style == "apa":
        _parse_apa_reference(ref_string, result)
    elif style == "vancouver":
        _parse_vancouver_reference(ref_string, result)
    elif style == "chicago":
        _parse_chicago_reference(ref_string, result)
    elif style == "harvard":
        _parse_harvard_reference(ref_string, result)
    else:
        # Generic parsing
        _parse_generic_reference(ref_string, result)
    
    return result


def _detect_citation_style(ref_string: str) -> str:
    """Detect citation style from reference string patterns."""
    
    # APA: Authors (Year). Title. Journal, Volume(Issue), Pages.
    if re.search(r'\([12]\d{3}\)\.', ref_string):
        return "apa"
    
    # Vancouver: Authors. Title. Journal. Year;Volume(Issue):Pages.
    if re.search(r'\.\s*[12]\d{3}\s*;', ref_string):
        return "vancouver"
    
    # Chicago: Authors. "Title." Journal Volume, no. Issue (Year): Pages.
    if re.search(r'no\.\s*\d+', ref_string, re.IGNORECASE):
        return "chicago"
    
    # Harvard: Authors (Year) Title, Journal, Volume(Issue), pp. Pages.
    if re.search(r'pp\.\s*\d+', ref_string):
        return "harvard"
    
    return "unknown"


def _parse_apa_reference(ref_string: str, result: Dict):
    """Parse APA-style reference."""
    # Pattern: Authors (Year). Title. Journal, Volume(Issue), Pages.
    
    # Split by year in parentheses
    year_split = re.split(r'\(([12]\d{3})\)\.?\s*', ref_string, maxsplit=1)
    
    if len(year_split) >= 2:
        result["authors"] = year_split[0].strip().rstrip(",").rstrip("&").strip()
        
        # Parse author list
        author_str = result["authors"]
        if ", &" in author_str or ", and" in author_str.lower():
            # Multiple authors
            parts = re.split(r',\s*&\s*|,\s*and\s*', author_str, flags=re.IGNORECASE)
            result["author_list"] = [a.strip() for a in parts if a.strip()]
        else:
            result["author_list"] = [author_str]
        
        if len(year_split) >= 3:
            remaining = year_split[2]
            
            # Title is usually before the first period followed by journal
            title_match = re.match(r'^([^.]+)\.', remaining)
            if title_match:
                result["title"] = title_match.group(1).strip()
                remaining = remaining[title_match.end():]
            
            # Journal is usually italicized or before volume
            journal_match = re.match(r'\s*([^,\d]+?)\s*,?\s*\d', remaining)
            if journal_match:
                result["journal"] = journal_match.group(1).strip().rstrip(",")


def _parse_vancouver_reference(ref_string: str, result: Dict):
    """Parse Vancouver-style reference."""
    # Pattern: Authors. Title. Journal. Year;Volume(Issue):Pages.
    
    # Split by periods
    parts = ref_string.split(".")
    
    if len(parts) >= 2:
        result["authors"] = parts[0].strip()
        result["author_list"] = [a.strip() for a in result["authors"].split(",") if a.strip()]
        
        if len(parts) >= 3:
            result["title"] = parts[1].strip()
        
        if len(parts) >= 4:
            # Journal and year/volume info
            journal_part = parts[2].strip()
            result["journal"] = journal_part


def _parse_chicago_reference(ref_string: str, result: Dict):
    """Parse Chicago-style reference."""
    # Pattern: Authors. "Title." Journal Volume, no. Issue (Year): Pages.
    
    # Extract quoted title
    title_match = re.search(r'"([^"]+)"', ref_string)
    if title_match:
        result["title"] = title_match.group(1)
    
    # Authors before title
    if title_match:
        result["authors"] = ref_string[:title_match.start()].strip().rstrip(".")
        result["author_list"] = [a.strip() for a in result["authors"].split(",") if a.strip()]


def _parse_harvard_reference(ref_string: str, result: Dict):
    """Parse Harvard-style reference."""
    # Pattern: Authors (Year) Title, Journal, Volume(Issue), pp. Pages.
    
    # Similar to APA but with different punctuation
    year_split = re.split(r'\(([12]\d{3})\)\s*', ref_string, maxsplit=1)
    
    if len(year_split) >= 2:
        result["authors"] = year_split[0].strip()
        result["author_list"] = [a.strip() for a in result["authors"].split(",") if a.strip()]
        
        if len(year_split) >= 3:
            remaining = year_split[2]
            # Title before first comma followed by journal
            parts = remaining.split(",", 1)
            if parts:
                result["title"] = parts[0].strip()
                if len(parts) > 1:
                    result["journal"] = parts[1].split(",")[0].strip()


def _parse_generic_reference(ref_string: str, result: Dict):
    """Generic reference parsing when style is unknown."""
    
    # Try to find authors (usually at the beginning, before year or title)
    # Common patterns: ends with period or year
    
    year_pos = ref_string.find(str(result["year"])) if result["year"] else -1
    
    if year_pos > 0:
        # Authors likely before year
        before_year = ref_string[:year_pos].strip()
        # Remove trailing punctuation
        before_year = before_year.rstrip(".(,")
        result["authors"] = before_year
        result["author_list"] = [a.strip() for a in before_year.split(",") if a.strip() and len(a.strip()) > 1]
    
    # Try to find title (often in quotes or after year)
    title_match = re.search(r'[""]([^""]+)[""]', ref_string)
    if title_match:
        result["title"] = title_match.group(1)
    elif result["year"]:
        # Title might be after year
        year_str = str(result["year"])
        year_pos = ref_string.find(year_str)
        if year_pos >= 0:
            after_year = ref_string[year_pos + len(year_str):].strip()
            after_year = after_year.lstrip(".)]: ")
            # First sentence-like segment
            title_match = re.match(r'^([^.]+\.)', after_year)
            if title_match:
                result["title"] = title_match.group(1).rstrip(".")


def parse_reference_list(
    references: List[str],
    style: str = "auto",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Parse a list of reference strings into a DataFrame.
    
    Parameters
    ----------
    references : list
        List of reference strings.
    style : str
        Citation style hint.
    verbose : bool
        Show progress.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with parsed reference components.
    """
    from .utilsbib import progress_bar
    
    results = []
    iterator = progress_bar(references, desc="Parsing references", disable=not verbose)
    
    for ref in iterator:
        parsed = parse_reference_string(ref, style=style)
        results.append(parsed)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    preferred_order = ["authors", "year", "title", "journal", "volume", "issue", 
                       "pages", "doi", "pmid", "style_detected", "raw"]
    existing_cols = [c for c in preferred_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in preferred_order]
    df = df[existing_cols + other_cols]
    
    return df


def extract_references_from_text(
    text: str,
    min_length: int = 30,
) -> List[str]:
    """
    Extract individual references from a text block.
    
    Attempts to split a reference list or bibliography into individual
    reference strings.
    
    Parameters
    ----------
    text : str
        Text containing multiple references.
    min_length : int
        Minimum length for a valid reference.
    
    Returns
    -------
    list
        List of individual reference strings.
    """
    if not text:
        return []
    
    references = []
    
    # Try numbered references first: [1], 1., (1), etc.
    numbered_pattern = re.compile(r'(?:^|\n)\s*(?:\[?\d+\]?\.?|\(\d+\))\s*')
    if numbered_pattern.search(text):
        parts = numbered_pattern.split(text)
        for part in parts:
            part = part.strip()
            if len(part) >= min_length:
                references.append(part)
    
    # If no numbered references, try splitting by double newlines
    elif "\n\n" in text:
        parts = text.split("\n\n")
        for part in parts:
            part = " ".join(part.split())  # Normalize whitespace
            if len(part) >= min_length:
                references.append(part)
    
    # Last resort: split by lines that start with author-like patterns
    else:
        lines = text.split("\n")
        current_ref = []
        
        for line in lines:
            line = line.strip()
            # Check if line starts a new reference (author name pattern)
            if re.match(r'^[A-Z][a-z]+,?\s+[A-Z]\.?', line) and current_ref:
                ref_text = " ".join(current_ref)
                if len(ref_text) >= min_length:
                    references.append(ref_text)
                current_ref = [line]
            else:
                current_ref.append(line)
        
        # Don't forget last reference
        if current_ref:
            ref_text = " ".join(current_ref)
            if len(ref_text) >= min_length:
                references.append(ref_text)
    
    return references