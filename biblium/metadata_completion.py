# -*- coding: utf-8 -*-
"""
biblium.metadata_completion
===========================

Missing-metadata audit and completion from open APIs (OpenAlex, Crossref)
by DOI — a biblium analogue of bibliometrix ``completeMetadata``. New in 2.19.

Design
------
* ``audit_missing``        which columns are incomplete and whether an API can fill them
* ``complete_metadata``    fills *only missing* cells (unless ``overwrite=True``),
                           returns a copy of the frame plus a provenance table
* ``undo_completion``      restores the original values from the provenance table
* ``completion_summary``   Field × Source counts of the fills

Network access is confined to ``fetch_openalex_work`` / ``fetch_crossref_work``.
``complete_metadata`` accepts injectable fetchers (``doi -> dict | None``), so
pipelines and tests can run fully offline. The mapping functions
``openalex_to_fields`` / ``crossref_to_fields`` translate the API JSON into
Scopus-style column names used throughout biblium.

Author: Lan Umek
Version: 2.19.0
"""
from __future__ import annotations

import re
import time
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "FILLABLE_FIELDS",
    "audit_missing",
    "normalize_doi",
    "fetch_openalex_work",
    "fetch_crossref_work",
    "openalex_to_fields",
    "crossref_to_fields",
    "complete_metadata",
    "undo_completion",
    "completion_summary",
]

#: Columns that OpenAlex and/or Crossref can supply.
FILLABLE_FIELDS = {
    "Title", "Year", "Source title", "Authors", "Abstract", "Cited by",
    "Document Type", "ISSN", "Volume", "Issue", "Page start", "Page end",
    "Open Access", "Language of Original Document", "Author Keywords",
}

PROVENANCE_COL = "Metadata provenance"
PROVENANCE_COLUMNS = ["Row index", "DOI", "Field", "Old value", "New value", "Source"]

OPENALEX_API = "https://api.openalex.org/works/https://doi.org/{doi}"
CROSSREF_API = "https://api.crossref.org/works/{doi}"
_TIMEOUT = 20

_DOC_TYPE_MAP = {
    "article": "Article", "journal-article": "Article",
    "review": "Review", "review-article": "Review",
    "book": "Book", "monograph": "Book", "edited-book": "Book", "reference-book": "Book",
    "book-chapter": "Book chapter", "chapter": "Book chapter",
    "proceedings-article": "Conference paper", "proceedings": "Conference paper",
    "paratext": "Editorial", "editorial": "Editorial", "erratum": "Erratum",
    "letter": "Letter", "dissertation": "Thesis", "preprint": "Preprint",
    "dataset": "Data paper", "report": "Report", "other": "Other",
}

_LANG_MAP = {
    "en": "English", "sl": "Slovenian", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "ja": "Japanese",
    "nl": "Dutch", "pl": "Polish", "cs": "Czech", "hr": "Croatian", "sv": "Swedish",
    "tr": "Turkish", "ko": "Korean", "ar": "Arabic", "hu": "Hungarian", "fi": "Finnish",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    try:
        if v is pd.NA or v is pd.NaT:
            return True
    except Exception:  # pragma: no cover
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _missing_mask(s: pd.Series) -> pd.Series:
    """Boolean mask: NaN, None, empty string or whitespace-only."""
    m = s.isna()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        m = m | (s.notna() & s.astype(str).str.strip().eq(""))
    return m


def normalize_doi(s: Any) -> Optional[str]:
    """Lower-case DOI without ``https://doi.org/`` / ``doi:`` prefixes.

    Returns ``None`` for missing or non-DOI values (a DOI starts with ``10.``).
    """
    if _is_missing(s):
        return None
    d = str(s).strip().lower()
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", d)
    d = re.sub(r"^doi:\s*", "", d).strip()
    if not d.startswith("10."):
        return None
    return d


def _initials(given: str) -> str:
    """'Mohammad Hossein' → 'M.H.'; 'J.-P.' → 'J.-P.'."""
    given = given.strip()
    if not given:
        return ""
    parts = re.split(r"[\s]+", given)
    out = []
    for p in parts:
        hy = p.split("-")
        out.append("-".join(h[0].upper() + "." for h in hy if h))
    return "".join(out)


def _name_from_display(display_name: str) -> str:
    """'Hao Wu' → 'Wu, H.'; 'Wu' → 'Wu'; 'Wu, Hao' → 'Wu, H.'."""
    n = (display_name or "").strip()
    if not n:
        return ""
    if "," in n:
        fam, given = [x.strip() for x in n.split(",", 1)]
        ini = _initials(given)
        return f"{fam}, {ini}" if ini else fam
    parts = n.split()
    if len(parts) == 1:
        return parts[0]
    fam = parts[-1]
    ini = _initials(" ".join(parts[:-1]))
    return f"{fam}, {ini}" if ini else fam


def _strip_tags(text: str) -> str:
    t = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", t).strip()


def _abstract_from_inverted_index(idx: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not idx:
        return None
    positions: Dict[int, str] = {}
    for word, pos in idx.items():
        for p in pos:
            positions[int(p)] = word
    if not positions:
        return None
    return " ".join(positions[k] for k in sorted(positions))


def _clean_issn(s: Any) -> Optional[str]:
    if _is_missing(s):
        return None
    v = re.sub(r"[^0-9Xx]", "", str(s)).upper()
    return v if len(v) == 8 else None


def _doc_type(t: Any) -> Optional[str]:
    if _is_missing(t):
        return None
    key = str(t).strip().lower()
    if key in _DOC_TYPE_MAP:
        return _DOC_TYPE_MAP[key]
    return key.replace("-", " ").capitalize()


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_missing(df: pd.DataFrame, fields: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Missing-data audit.

    Returns a frame with columns ``Field``, ``N missing``, ``% missing`` and
    ``Fillable`` (True if an open API can supply the field, see
    ``FILLABLE_FIELDS``), sorted by ``% missing`` descending (ties by name).
    NaN, empty and whitespace-only strings count as missing.

    Parameters
    ----------
    fields : sequence of str, optional
        Columns to audit (default: all columns). Fields not in ``df`` are
        reported as 100 % missing.
    """
    cols = list(df.columns) if fields is None else list(fields)
    n = len(df)
    rows = []
    for c in cols:
        if c in df.columns:
            k = int(_missing_mask(df[c]).sum())
        else:
            k = n
        rows.append({
            "Field": c,
            "N missing": k,
            "% missing": (100.0 * k / n) if n else np.nan,
            "Fillable": c in FILLABLE_FIELDS,
        })
    out = pd.DataFrame(rows, columns=["Field", "N missing", "% missing", "Fillable"])
    out = out.sort_values(["% missing", "Field"], ascending=[False, True], kind="mergesort")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fetching (the only network code in the module)
# ---------------------------------------------------------------------------

def fetch_openalex_work(
    doi: str,
    email: Optional[str] = None,
    session=None,
    delay: float = 0.1,
) -> Optional[dict]:
    """GET one work from OpenAlex by DOI; ``None`` on any error (timeout 20 s)."""
    d = normalize_doi(doi)
    if d is None:
        return None
    try:
        import requests
        url = OPENALEX_API.format(doi=d)
        params = {"mailto": email} if email else None
        if delay:
            time.sleep(delay)
        getter = session.get if session is not None else requests.get
        r = getter(url, params=params, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        js = r.json()
        return js if isinstance(js, dict) and js.get("id") else None
    except Exception:
        return None


def fetch_crossref_work(
    doi: str,
    email: Optional[str] = None,
    session=None,
    delay: float = 0.1,
) -> Optional[dict]:
    """GET one work from Crossref by DOI; returns the ``message`` dict or
    ``None`` on any error (timeout 20 s)."""
    d = normalize_doi(doi)
    if d is None:
        return None
    try:
        import requests
        url = CROSSREF_API.format(doi=d)
        headers = {"User-Agent": f"biblium (mailto:{email})" if email else "biblium"}
        params = {"mailto": email} if email else None
        if delay:
            time.sleep(delay)
        getter = session.get if session is not None else requests.get
        r = getter(url, params=params, headers=headers, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        js = r.json()
        msg = js.get("message") if isinstance(js, dict) else None
        return msg if isinstance(msg, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Mapping API JSON → Scopus-style columns
# ---------------------------------------------------------------------------

def openalex_to_fields(work: dict) -> Dict[str, Any]:
    """Map an OpenAlex work JSON to biblium/Scopus column names.

    Only keys with a non-empty value are returned.
    """
    out: Dict[str, Any] = {}
    if not isinstance(work, dict):
        return out
    g = work.get

    title = g("title") or g("display_name")
    if title:
        out["Title"] = str(title).strip()
    if g("publication_year") is not None:
        out["Year"] = int(g("publication_year"))

    loc = g("primary_location") or {}
    src = (loc.get("source") if isinstance(loc, dict) else None) or {}
    if src.get("display_name"):
        out["Source title"] = str(src["display_name"]).strip()
    issn = _clean_issn(src.get("issn_l")) or next(
        (_clean_issn(x) for x in (src.get("issn") or []) if _clean_issn(x)), None)
    if issn:
        out["ISSN"] = issn

    names = []
    for a in g("authorships") or []:
        auth = (a or {}).get("author") or {}
        dn = auth.get("display_name")
        if dn:
            names.append(_name_from_display(dn))
    if names:
        out["Authors"] = "; ".join(names)

    abstract = _abstract_from_inverted_index(g("abstract_inverted_index"))
    if abstract:
        out["Abstract"] = abstract
    if g("cited_by_count") is not None:
        out["Cited by"] = int(g("cited_by_count"))
    dt = _doc_type(g("type"))
    if dt:
        out["Document Type"] = dt

    bib = g("biblio") or {}
    for api_key, col in (("volume", "Volume"), ("issue", "Issue"),
                         ("first_page", "Page start"), ("last_page", "Page end")):
        if bib.get(api_key):
            out[col] = str(bib[api_key]).strip()

    oa = g("open_access") or {}
    if oa.get("is_oa"):
        status = str(oa.get("oa_status") or "").strip().lower()
        out["Open Access"] = ("All Open Access; " + status.capitalize() + " Open Access") if status and status != "closed" else "All Open Access"

    lang = g("language")
    if lang:
        out["Language of Original Document"] = _LANG_MAP.get(str(lang).lower(), str(lang))

    kws = []
    for k in g("keywords") or []:
        if isinstance(k, dict):
            v = k.get("display_name") or k.get("keyword")
        else:
            v = k
        if v:
            kws.append(str(v).strip())
    if kws:
        out["Author Keywords"] = "; ".join(kws)
    return out


def crossref_to_fields(msg: dict) -> Dict[str, Any]:
    """Map a Crossref ``message`` JSON to biblium/Scopus column names.

    Only keys with a non-empty value are returned.
    """
    out: Dict[str, Any] = {}
    if not isinstance(msg, dict):
        return out
    g = msg.get

    def _first(v):
        if isinstance(v, (list, tuple)):
            return v[0] if v else None
        return v

    title = _first(g("title"))
    if title:
        out["Title"] = _strip_tags(str(title))

    year = None
    for key in ("issued", "published-print", "published-online", "created"):
        dp = (g(key) or {}).get("date-parts") if isinstance(g(key), dict) else None
        if dp and dp[0] and dp[0][0]:
            year = int(dp[0][0])
            break
    if year is not None:
        out["Year"] = year

    ct = _first(g("container-title"))
    if ct:
        out["Source title"] = str(ct).strip()
    issn = next((_clean_issn(x) for x in (g("ISSN") or []) if _clean_issn(x)), None)
    if issn:
        out["ISSN"] = issn

    names = []
    for a in g("author") or []:
        fam = (a or {}).get("family")
        giv = (a or {}).get("given") or ""
        if fam:
            ini = _initials(giv)
            names.append(f"{fam}, {ini}" if ini else fam)
        elif (a or {}).get("name"):
            names.append(str(a["name"]))
    if names:
        out["Authors"] = "; ".join(names)

    if g("abstract"):
        ab = _strip_tags(str(g("abstract")))
        ab = re.sub(r"^(abstract)\s*:?\s*", "", ab, flags=re.I)
        if ab:
            out["Abstract"] = ab
    if g("is-referenced-by-count") is not None:
        out["Cited by"] = int(g("is-referenced-by-count"))
    dt = _doc_type(g("type"))
    if dt:
        out["Document Type"] = dt
    if g("volume"):
        out["Volume"] = str(g("volume")).strip()
    issue = g("issue") or ((g("journal-issue") or {}).get("issue") if isinstance(g("journal-issue"), dict) else None)
    if issue:
        out["Issue"] = str(issue).strip()
    page = g("page")
    if page:
        parts = re.split(r"\s*[-–]\s*", str(page).strip(), maxsplit=1)
        out["Page start"] = parts[0]
        if len(parts) > 1 and parts[1]:
            out["Page end"] = parts[1]
    lang = g("language")
    if lang:
        out["Language of Original Document"] = _LANG_MAP.get(str(lang).lower(), str(lang))
    lic = g("license")
    if isinstance(lic, list) and lic:
        # Crossref carries no OA status; a Creative Commons licence is a proxy
        if any("creativecommons" in str((l or {}).get("URL", "")) for l in lic):
            out["Open Access"] = "All Open Access"
    return out


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

def _value_for_column(df: pd.DataFrame, col: str, value: Any) -> Any:
    """Coerce an API value to the column's dtype where it is numeric."""
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        try:
            return float(value) if pd.api.types.is_float_dtype(df[col]) else int(value)
        except (TypeError, ValueError):
            return value
    return value


def complete_metadata(
    df: pd.DataFrame,
    fields: Optional[Sequence[str]] = None,
    doi_col: str = "DOI",
    sources: Sequence[str] = ("openalex", "crossref"),
    overwrite: bool = False,
    email: Optional[str] = None,
    max_rows: Optional[int] = None,
    fetch_openalex: Optional[Callable[[str], Optional[dict]]] = None,
    fetch_crossref: Optional[Callable[[str], Optional[dict]]] = None,
    delay: float = 0.1,
    progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing metadata cells from OpenAlex / Crossref by DOI.

    For every row with a DOI and at least one missing requested field the
    sources are queried in order; each source fills only the cells that are
    still missing (unless ``overwrite=True``, in which case every non-empty
    API value replaces the current one — the first source that supplies a
    field wins). Every change is logged in the returned provenance table.

    Parameters
    ----------
    fields : sequence of str, optional
        Columns to complete (default: ``FILLABLE_FIELDS`` present in ``df``).
        Requested columns missing from ``df`` are created.
    sources : sequence of {"openalex", "crossref"}
        Query order.
    overwrite : bool
        Replace existing values as well.
    max_rows : int, optional
        Stop after this many rows *with a DOI* have been processed.
    fetch_openalex, fetch_crossref : callable ``doi -> dict | None``, optional
        Injectable fetchers (for caching, testing, custom clients). The
        defaults are ``fetch_openalex_work`` / ``fetch_crossref_work``.
    delay : float
        Seconds to sleep before each default API call.
    progress : bool
        Print a one-line progress counter.

    Returns
    -------
    completed : pd.DataFrame
        Copy of ``df`` with the filled cells and a ``"Metadata provenance"``
        column (``"Abstract:openalex; Year:crossref"`` for changed rows).
    provenance : pd.DataFrame
        Columns ``Row index, DOI, Field, Old value, New value, Source``.
    """
    out = df.copy()
    if fields is None:
        fields = [c for c in df.columns if c in FILLABLE_FIELDS]
    fields = list(fields)
    for c in fields:
        if c not in out.columns:
            out[c] = pd.Series([np.nan] * len(out), index=out.index, dtype=object)
    if PROVENANCE_COL not in out.columns:
        out[PROVENANCE_COL] = pd.Series([np.nan] * len(out), index=out.index, dtype=object)

    fetchers: Dict[str, Callable[[str], Optional[dict]]] = {
        "openalex": fetch_openalex or partial(fetch_openalex_work, email=email, delay=delay),
        "crossref": fetch_crossref or partial(fetch_crossref_work, email=email, delay=delay),
    }
    mappers = {"openalex": openalex_to_fields, "crossref": crossref_to_fields}
    for s in sources:
        if s not in fetchers:
            raise ValueError(f"Unknown source {s!r}; use 'openalex' and/or 'crossref'")

    prov_rows: List[Dict[str, Any]] = []
    if doi_col not in out.columns:
        return out, pd.DataFrame(columns=PROVENANCE_COLUMNS)

    dois = out[doi_col].map(normalize_doi, na_action=None)
    candidates = [i for i in out.index if isinstance(dois.loc[i], str)]
    if max_rows is not None:
        candidates = candidates[: int(max_rows)]
    cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
    n_total = len(candidates)

    for k, idx in enumerate(candidates, 1):
        if progress:
            print(f"\rcomplete_metadata: {k}/{n_total}", end="", flush=True)
        doi = dois.loc[idx]
        missing = {c for c in fields if _is_missing(out.at[idx, c])}
        todo = set(fields) if overwrite else set(missing)
        if not todo:
            continue
        changed: List[str] = []
        for src in sources:
            if not todo:
                break
            key = (src, doi)
            if key not in cache:
                raw = fetchers[src](doi)
                cache[key] = mappers[src](raw) if raw else None
            mapped = cache[key]
            if not mapped:
                continue
            for c in sorted(todo):
                if c not in mapped or _is_missing(mapped[c]):
                    continue
                old = out.at[idx, c]
                new = _value_for_column(out, c, mapped[c])
                if not _is_missing(old) and str(old) == str(new):
                    todo.discard(c)
                    continue
                out.at[idx, c] = new
                prov_rows.append({"Row index": idx, "DOI": doi, "Field": c,
                                  "Old value": old, "New value": new, "Source": src})
                changed.append(f"{c}:{src}")
                todo.discard(c)
        if changed:
            prev = out.at[idx, PROVENANCE_COL]
            tag = "; ".join(changed)
            out.at[idx, PROVENANCE_COL] = tag if _is_missing(prev) else f"{prev}; {tag}"
    if progress and n_total:
        print()

    prov = pd.DataFrame(prov_rows, columns=PROVENANCE_COLUMNS)
    return out, prov


def undo_completion(df: pd.DataFrame, provenance_df: pd.DataFrame) -> pd.DataFrame:
    """Restore the ``Old value`` of every provenance row (returns a copy).

    The matching ``Field:source`` tags are removed from ``"Metadata
    provenance"``; rows left without tags get NaN there.
    """
    out = df.copy()
    if provenance_df is None or len(provenance_df) == 0:
        return out
    for _, r in provenance_df.iterrows():
        idx, col = r["Row index"], r["Field"]
        if idx not in out.index or col not in out.columns:
            continue
        old = r["Old value"]
        out.at[idx, col] = old if isinstance(old, str) or not _is_missing(old) else np.nan
        if PROVENANCE_COL in out.columns:
            prev = out.at[idx, PROVENANCE_COL]
            if not _is_missing(prev):
                tag = f"{col}:{r['Source']}"
                parts = [p.strip() for p in str(prev).split(";") if p.strip() and p.strip() != tag]
                out.at[idx, PROVENANCE_COL] = "; ".join(parts) if parts else np.nan
    return out


def completion_summary(provenance_df: pd.DataFrame) -> pd.DataFrame:
    """Field × Source count table of the fills (with a ``Total`` column)."""
    if provenance_df is None or len(provenance_df) == 0:
        return pd.DataFrame(columns=["Total"]).rename_axis("Field")
    tab = pd.crosstab(provenance_df["Field"], provenance_df["Source"])
    tab["Total"] = tab.sum(axis=1)
    return tab.sort_values("Total", ascending=False)
