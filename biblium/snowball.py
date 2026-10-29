# -*- coding: utf-8 -*-
"""
biblium.snowball
================

Citation-based corpus expansion ("snowballing") through the OpenAlex API
(new in 2.19). Starting from a *root set* of seed works the module gathers
their references (backward snowballing) and the works citing them (forward
snowballing), keeps candidates linked to at least ``min_links`` seeds and
builds a *base set* -- the analogue of CiteSpace's cascading citation
expansion and of the root set -> base set construction in pyBiblioNet.

All HTTP access goes through an injectable ``fetch_json(url) -> dict | None``
callable, so the module can be used offline with a fake fetcher (unit tests
do exactly that). The default fetcher uses ``requests`` with the polite
``mailto`` parameter.

Contents
--------
API helpers
    ``normalize_openalex_id``   "https://openalex.org/W123" -> "W123"
    ``default_fetch_json``      requests-based fetcher (returns None on error)
    ``resolve_seeds``           seeds (OpenAlex IDs or DOIs) -> work JSON
    ``fetch_references``        referenced work ids of a work
    ``fetch_citing``            works citing a work (cursor pagination)
    ``fetch_works_batch``       metadata for many ids in batches

Expansion
    ``snowball``                root set -> base set, multi-round
    ``works_to_biblium_df``     OpenAlex work JSON -> Scopus-like columns
    ``expansion_summary``       Round x Direction counts
    ``plot_expansion``          bar chart of works per round and direction

Author: Lan Umek
Version: 2.19.0
"""
from __future__ import annotations

import math
import re
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "normalize_openalex_id",
    "default_fetch_json",
    "resolve_seeds",
    "fetch_references",
    "fetch_citing",
    "fetch_works_batch",
    "snowball",
    "works_to_biblium_df",
    "expansion_summary",
    "plot_expansion",
]

OPENALEX_BASE = "https://api.openalex.org"
FetchJson = Callable[[str], Optional[dict]]

SNOWBALL_COLUMNS = [
    "OpenAlex ID", "DOI", "Title", "Year", "Source title", "Authors", "Cited by",
    "Type", "Round", "Direction", "Links to seed set",
]

_OA_ID_RE = re.compile(r"^[A-Za-z]\d+$")
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.I)


# ---------------------------------------------------------------------------
# Identifiers and fetching
# ---------------------------------------------------------------------------

def normalize_openalex_id(x: object) -> Optional[str]:
    """
    Normalise an OpenAlex work identifier.

    ``"https://openalex.org/W123"`` and ``"w123"`` both give ``"W123"``.
    Returns None for empty / NaN input or for strings that are not OpenAlex
    ids (e.g. DOIs -- those are handled by :func:`resolve_seeds`).
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = re.sub(r"^https?://(www\.)?openalex\.org/(works/)?", "", s, flags=re.I)
    s = s.strip("/ ")
    if _OA_ID_RE.match(s):
        return s[0].upper() + s[1:]
    return None


def _is_doi(x: str) -> bool:
    return bool(_DOI_RE.search(str(x)))


def _clean_doi(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.I)
    s = re.sub(r"^doi:\s*", "", s, flags=re.I)
    return s.strip()


def _with_mailto(url: str, email: Optional[str]) -> str:
    if not email:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}mailto={email}"


def default_fetch_json(
    url: str,
    email: Optional[str] = None,
    timeout: float = 20,
    delay: float = 0.1,
) -> Optional[dict]:
    """
    Fetch ``url`` with ``requests`` and return the parsed JSON, or None on any
    error (network, HTTP status, decoding). ``delay`` seconds are slept
    *before* every request to stay within the polite rate limit.
    """
    try:
        import requests
    except Exception:  # pragma: no cover
        return None
    try:
        if delay and delay > 0:
            time.sleep(delay)
        r = requests.get(_with_mailto(url, email), timeout=timeout,
                         headers={"User-Agent": "biblium/2.19"})
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _fetcher(fetch_json: Optional[FetchJson], email: Optional[str]) -> FetchJson:
    if fetch_json is not None:
        return fetch_json
    return lambda url: default_fetch_json(url, email=email)


def _work_url(work_id: str) -> str:
    return f"{OPENALEX_BASE}/works/{work_id}"


def _doi_url(doi: str) -> str:
    return f"{OPENALEX_BASE}/works/https://doi.org/{_clean_doi(doi)}"


def resolve_seeds(
    seeds: Sequence[str],
    fetch_json: Optional[FetchJson] = None,
    email: Optional[str] = None,
) -> List[dict]:
    """
    Resolve seed identifiers (OpenAlex ids or DOIs) to OpenAlex work JSON.

    OpenAlex ids are fetched from ``/works/{id}``; DOIs from
    ``/works/https://doi.org/{doi}``. Failures are skipped; duplicates (same
    resolved id) are returned once, in the order of first appearance.
    """
    fj = _fetcher(fetch_json, email)
    out: List[dict] = []
    seen: set = set()
    for s in seeds:
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        oid = normalize_openalex_id(s)
        if oid is not None:
            url = _work_url(oid)
        elif _is_doi(str(s)):
            url = _doi_url(str(s))
        else:
            continue
        w = fj(url)
        if not isinstance(w, dict) or not w.get("id"):
            continue
        wid = normalize_openalex_id(w.get("id"))
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        out.append(w)
    return out


def fetch_references(
    work_id,
    fetch_json: Optional[FetchJson] = None,
    email: Optional[str] = None,
) -> List[str]:
    """
    Referenced work ids (normalised) of a work. ``work_id`` may be an id (the
    work is fetched) or an already fetched work dict.
    """
    if isinstance(work_id, dict):
        w = work_id
    else:
        oid = normalize_openalex_id(work_id)
        if oid is None:
            return []
        w = _fetcher(fetch_json, email)(_work_url(oid))
    if not isinstance(w, dict):
        return []
    refs = w.get("referenced_works") or []
    out: List[str] = []
    for r in refs:
        rid = normalize_openalex_id(r)
        if rid is not None and rid not in out:
            out.append(rid)
    return out


def fetch_citing(
    work_id,
    fetch_json: Optional[FetchJson] = None,
    email: Optional[str] = None,
    per_page: int = 200,
    max_pages: int = 5,
) -> List[dict]:
    """
    Works citing ``work_id`` (``filter=cites:{id}``) with cursor pagination;
    at most ``max_pages`` pages of ``per_page`` results are fetched.
    """
    oid = normalize_openalex_id(work_id.get("id") if isinstance(work_id, dict) else work_id)
    if oid is None:
        return []
    fj = _fetcher(fetch_json, email)
    out: List[dict] = []
    seen: set = set()
    cursor = "*"
    for _ in range(max(int(max_pages), 1)):
        url = f"{OPENALEX_BASE}/works?filter=cites:{oid}&per-page={int(per_page)}&cursor={cursor}"
        page = fj(url)
        if not isinstance(page, dict):
            break
        for w in page.get("results") or []:
            wid = normalize_openalex_id(w.get("id")) if isinstance(w, dict) else None
            if wid and wid not in seen:
                seen.add(wid)
                out.append(w)
        cursor = (page.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
    return out


def fetch_works_batch(
    ids: Iterable[str],
    fetch_json: Optional[FetchJson] = None,
    email: Optional[str] = None,
    batch: int = 50,
) -> List[dict]:
    """
    Metadata of many works through ``filter=openalex:W1|W2|...`` in batches of
    ``batch`` ids (OpenAlex allows up to 50 per filter). Order follows the
    input ids; ids that are not returned are silently missing.
    """
    fj = _fetcher(fetch_json, email)
    norm: List[str] = []
    for x in ids:
        oid = normalize_openalex_id(x)
        if oid and oid not in norm:
            norm.append(oid)
    found: Dict[str, dict] = {}
    step = max(int(batch), 1)
    for i in range(0, len(norm), step):
        chunk = norm[i:i + step]
        url = f"{OPENALEX_BASE}/works?filter=openalex:{'|'.join(chunk)}&per-page={step}"
        page = fj(url)
        if not isinstance(page, dict):
            continue
        for w in page.get("results") or []:
            wid = normalize_openalex_id(w.get("id")) if isinstance(w, dict) else None
            if wid:
                found[wid] = w
    return [found[i] for i in norm if i in found]


# ---------------------------------------------------------------------------
# Work JSON -> rows
# ---------------------------------------------------------------------------

def _author_short(display_name: object) -> str:
    """``"Jane Q. Smith"`` -> ``"Smith, J.Q."``; single tokens unchanged."""
    if display_name is None:
        return ""
    s = str(display_name).strip()
    if not s:
        return ""
    if "," in s:  # already "Surname, Given"
        sur, given = [p.strip() for p in s.split(",", 1)]
        initials = "".join(f"{g[0].upper()}." for g in re.split(r"[\s\-]+", given) if g)
        return f"{sur}, {initials}" if initials else sur
    parts = s.split()
    if len(parts) == 1:
        return parts[0]
    sur = parts[-1]
    initials = "".join(f"{g[0].upper()}." for g in parts[:-1] if g)
    return f"{sur}, {initials}"


def _authors_str(w: dict) -> str:
    names = []
    for a in w.get("authorships") or []:
        if not isinstance(a, dict):
            continue
        au = a.get("author") or {}
        nm = _author_short(au.get("display_name") or a.get("raw_author_name"))
        if nm:
            names.append(nm)
    return "; ".join(names)


def _source_title(w: dict) -> Optional[str]:
    loc = w.get("primary_location") or {}
    src = loc.get("source") if isinstance(loc, dict) else None
    if isinstance(src, dict) and src.get("display_name"):
        return str(src["display_name"])
    hv = w.get("host_venue") or {}
    if isinstance(hv, dict) and hv.get("display_name"):
        return str(hv["display_name"])
    return None


def _abstract(w: dict) -> Optional[str]:
    inv = w.get("abstract_inverted_index")
    if not isinstance(inv, dict) or not inv:
        return w.get("abstract") if isinstance(w.get("abstract"), str) else None
    pos: List[Tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs or []:
            pos.append((int(i), str(word)))
    pos.sort()
    return " ".join(wd for _, wd in pos)


def _keywords(w: dict) -> Optional[str]:
    kws = []
    for k in w.get("keywords") or []:
        if isinstance(k, dict) and k.get("display_name"):
            kws.append(str(k["display_name"]))
        elif isinstance(k, str):
            kws.append(k)
    if not kws:
        for c in w.get("concepts") or []:
            if isinstance(c, dict) and c.get("display_name"):
                kws.append(str(c["display_name"]))
    return "; ".join(kws) if kws else None


def _year(w: dict):
    y = w.get("publication_year")
    if y is None:
        d = w.get("publication_date")
        if isinstance(d, str) and len(d) >= 4 and d[:4].isdigit():
            y = int(d[:4])
    try:
        return int(y) if y is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _work_row(w: dict) -> Dict[str, object]:
    return {
        "OpenAlex ID": normalize_openalex_id(w.get("id")),
        "DOI": _clean_doi(w["doi"]) if w.get("doi") else None,
        "Title": w.get("display_name") or w.get("title"),
        "Year": _year(w),
        "Source title": _source_title(w),
        "Authors": _authors_str(w),
        "Cited by": int(w.get("cited_by_count") or 0),
        "Type": w.get("type"),
    }


def works_to_biblium_df(works: List[dict]) -> pd.DataFrame:
    """
    Map OpenAlex work JSON to a biblium / Scopus-like DataFrame.

    Columns: ``Title``, ``Year``, ``Source title``, ``Authors`` ("Surname,
    I." joined by "; "), ``Cited by``, ``DOI``, ``Abstract`` (rebuilt from
    ``abstract_inverted_index``), ``Author Keywords`` (from ``keywords`` or,
    failing that, ``concepts``), ``Document Type``, ``OpenAlex ID`` and
    ``References`` ("; "-joined referenced work ids).
    """
    cols = ["Title", "Year", "Source title", "Authors", "Cited by", "DOI", "Abstract",
            "Author Keywords", "Document Type", "OpenAlex ID", "References"]
    rows = []
    for w in works:
        if not isinstance(w, dict):
            continue
        r = _work_row(w)
        rows.append({
            "Title": r["Title"],
            "Year": r["Year"],
            "Source title": r["Source title"],
            "Authors": r["Authors"],
            "Cited by": r["Cited by"],
            "DOI": r["DOI"],
            "Abstract": _abstract(w),
            "Author Keywords": _keywords(w),
            "Document Type": r["Type"],
            "OpenAlex ID": r["OpenAlex ID"],
            "References": "; ".join(fetch_references(w)) or None,
        })
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Snowballing
# ---------------------------------------------------------------------------

def _year_ok(w: dict, year_min: Optional[int], year_max: Optional[int]) -> bool:
    if year_min is None and year_max is None:
        return True
    y = _year(w)
    if isinstance(y, float) and math.isnan(y):
        return False
    if year_min is not None and y < year_min:
        return False
    if year_max is not None and y > year_max:
        return False
    return True


def snowball(
    seeds: Sequence[str],
    direction: str = "both",
    rounds: int = 1,
    min_links: int = 1,
    max_candidates: int = 2000,
    fetch_json: Optional[FetchJson] = None,
    email: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    progress: bool = False,
    per_page: int = 200,
    max_pages: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Expand a root set of seed works into a base set by citation links.

    Round *r* takes the current frontier (round 0: the resolved seeds;
    later: the works added in round *r-1*), collects for every frontier work
    its references (``direction`` "backward"), the works citing it
    ("forward") or both, and counts for each candidate the number of distinct
    frontier works it is linked to (``Links to seed set``). Candidates with
    at least ``min_links`` links that are not yet in the set are fetched in
    batches (at most ``max_candidates`` per round, the best linked first)
    and added.

    Parameters
    ----------
    seeds : sequence of str
        OpenAlex ids or DOIs.
    direction : {"backward", "forward", "both"}
    rounds : int
        Number of expansion rounds (1 = classic one-step snowball).
    min_links : int
        Minimum number of frontier works a candidate must be linked to.
    year_min, year_max : int, optional
        Candidates outside the interval are dropped (seeds are kept).
    fetch_json : callable, optional
        ``url -> dict | None``; defaults to :func:`default_fetch_json`.

    Returns
    -------
    df : pd.DataFrame
        Columns ``OpenAlex ID, DOI, Title, Year, Source title, Authors,
        Cited by, Type, Round, Direction, Links to seed set``. Seeds have
        ``Round`` 0 and ``Direction`` "seed"; other rows "backward",
        "forward" or "both" (linked in both directions in the same round).
    stats : dict
        ``n_seeds_requested``, ``n_seeds_resolved``, ``n_total``,
        ``per_round`` (list of dicts with counts per direction and the
        number of candidates seen / kept / dropped by year filter) and
        ``direction``, ``rounds``, ``min_links``.
    """
    if direction not in ("backward", "forward", "both"):
        raise ValueError("direction must be 'backward', 'forward' or 'both'")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    fj = _fetcher(fetch_json, email)
    want_back = direction in ("backward", "both")
    want_fwd = direction in ("forward", "both")

    seed_works = resolve_seeds(seeds, fj)
    rows: List[Dict[str, object]] = []
    in_set: Dict[str, dict] = {}
    for w in seed_works:
        wid = normalize_openalex_id(w.get("id"))
        r = _work_row(w)
        r.update({"Round": 0, "Direction": "seed", "Links to seed set": 0})
        rows.append(r)
        in_set[wid] = w
    frontier: List[dict] = list(seed_works)
    per_round: List[Dict[str, object]] = []

    for rnd in range(1, int(rounds) + 1):
        if not frontier:
            break
        back_links: Dict[str, set] = {}
        fwd_links: Dict[str, set] = {}
        fetched_candidates: Dict[str, dict] = {}
        for w in frontier:
            fid = normalize_openalex_id(w.get("id"))
            if want_back:
                for rid in fetch_references(w, fj):
                    back_links.setdefault(rid, set()).add(fid)
            if want_fwd:
                for cw in fetch_citing(fid, fj, per_page=per_page, max_pages=max_pages):
                    cid = normalize_openalex_id(cw.get("id"))
                    if cid is None:
                        continue
                    fwd_links.setdefault(cid, set()).add(fid)
                    fetched_candidates.setdefault(cid, cw)
            if progress:
                print(f"[snowball] round {rnd}: processed {fid}")

        cand_ids = set(back_links) | set(fwd_links)
        cand_ids -= set(in_set)
        scored = []
        for cid in cand_ids:
            links = len(back_links.get(cid, set()) | fwd_links.get(cid, set()))
            if links >= min_links:
                scored.append((cid, links))
        scored.sort(key=lambda t: (-t[1], t[0]))
        n_seen, n_pass = len(cand_ids), len(scored)
        if max_candidates is not None:
            scored = scored[:int(max_candidates)]

        need = [cid for cid, _ in scored if cid not in fetched_candidates]
        for w in fetch_works_batch(need, fj):
            fetched_candidates[normalize_openalex_id(w.get("id"))] = w

        added: List[dict] = []
        n_year_dropped = 0
        counts = {"backward": 0, "forward": 0, "both": 0}
        for cid, links in scored:
            w = fetched_candidates.get(cid)
            if w is None:
                continue
            if not _year_ok(w, year_min, year_max):
                n_year_dropped += 1
                continue
            is_b, is_f = cid in back_links, cid in fwd_links
            d = "both" if (is_b and is_f) else ("backward" if is_b else "forward")
            r = _work_row(w)
            r.update({"Round": rnd, "Direction": d, "Links to seed set": int(links)})
            rows.append(r)
            in_set[cid] = w
            added.append(w)
            counts[d] += 1
        per_round.append({
            "Round": rnd, "n_candidates": n_seen, "n_passed_min_links": n_pass,
            "n_added": len(added), "n_year_dropped": n_year_dropped, **counts,
        })
        if progress:
            print(f"[snowball] round {rnd}: {len(added)} works added")
        frontier = added

    df = pd.DataFrame(rows, columns=SNOWBALL_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates("OpenAlex ID", keep="first").reset_index(drop=True)
    stats: Dict[str, object] = {
        "direction": direction, "rounds": int(rounds), "min_links": int(min_links),
        "n_seeds_requested": int(len(seeds)), "n_seeds_resolved": int(len(seed_works)),
        "n_total": int(len(df)), "per_round": per_round,
    }
    return df, stats


def expansion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Round x Direction table of work counts (with a ``Total`` column)."""
    if df.empty:
        return pd.DataFrame()
    tab = pd.crosstab(df["Round"], df["Direction"])
    order = [c for c in ("seed", "backward", "forward", "both") if c in tab.columns]
    tab = tab[order + [c for c in tab.columns if c not in order]]
    tab["Total"] = tab.sum(axis=1)
    tab.index.name, tab.columns.name = "Round", "Direction"
    return tab


def _finish(fig, ax, title, f_name, dpi, show):
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if f_name:
        fig.savefig(f_name, dpi=dpi, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig, ax


def plot_expansion(
    df: pd.DataFrame,
    ax=None,
    figsize: Tuple[float, float] = (7, 4),
    colors: Optional[Dict[str, str]] = None,
    title: Optional[str] = "Snowball expansion",
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    """Stacked bars of the number of works per round, split by direction."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    tab = expansion_summary(df)
    default_colors = {"seed": "#4d4d4d", "backward": "#1f77b4", "forward": "#ff7f0e", "both": "#2ca02c"}
    if colors:
        default_colors.update(colors)
    if tab.empty:
        ax.text(0.5, 0.5, "No works", ha="center", va="center", transform=ax.transAxes)
        return _finish(fig, ax, title, f_name, dpi, show)
    cols = [c for c in tab.columns if c != "Total"]
    bottom = np.zeros(len(tab))
    x = np.arange(len(tab))
    for c in cols:
        vals = tab[c].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=c, color=default_colors.get(c))
        bottom += vals
    for xi, tot in zip(x, tab["Total"]):
        ax.text(xi, tot, str(int(tot)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in tab.index])
    ax.set_xlabel("Round")
    ax.set_ylabel("Number of works")
    ax.legend(frameon=False)
    return _finish(fig, ax, title, f_name, dpi, show)
