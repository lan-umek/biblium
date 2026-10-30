# -*- coding: utf-8 -*-
"""
biblium.merge
=============

Combining records from several bibliographic databases (Scopus, Web of
Science, OpenAlex, PubMed, Dimensions, Lens, …) into one de-duplicated
dataset (novo v 2.19).

Workflow
--------
1. ``load_sources`` reads each export with ``readbib.read_bibfile`` (column
   names are already mapped to the canonical, Scopus-like names there).
2. ``harmonize`` unifies what the readers leave database-specific:
   multi-value separators (OpenAlex uses ``|``), DOI spelling, numeric
   ``Year``/``Cited by``; and adds a ``Source DB`` column.
3. ``combine_databases`` matches records across sources — by **DOI**
   (default), by normalised **title**, or DOI with title fallback — merges
   every duplicate group into one record and returns the merged frame plus a
   report (overlap between sources, column coverage).

Two column policies:

- ``columns="union"`` keeps every column that appears in any source (many
  cells are then missing for records that come from a single database);
- ``columns="intersection"`` keeps only the columns present in **all**
  sources (no structural missingness; database-specific fields are dropped).

Merging rules inside a duplicate group (``priority`` order, default
Scopus > WoS > Dimensions > OpenAlex > Lens > PubMed > …):

- ordinary fields: first non-missing value by priority;
- ``UNION_FIELDS`` (keywords, references): union of the values, in order;
- ``LONGEST_FIELDS`` (abstract, affiliations): the longest value;
- ``MAX_FIELDS`` (``Cited by``): the maximum, with per-source copies
  ``Cited by (scopus)`` … when ``keep_source_columns=True``;
- provenance: ``Source DB`` ("scopus; wos"), ``N sources``, ``Merge key``.

Version: 2.19.1
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_PRIORITY", "UNION_FIELDS", "LONGEST_FIELDS", "MAX_FIELDS", "MULTI_VALUE_FIELDS",
    "normalize_doi", "normalize_title", "harmonize", "load_sources", "combine_databases",
    "source_overlap_table", "column_coverage", "plot_source_overlap",
]

DEFAULT_PRIORITY: List[str] = [
    "scopus", "wos", "dimensions", "openalex", "oa", "open alex", "lens", "pubmed",
    "crossref", "semantic scholar", "cobiss",
]

UNION_FIELDS = ["Author Keywords", "Index Keywords", "Keywords", "References"]
LONGEST_FIELDS = ["Abstract", "Affiliations", "Authors with affiliations", "Funding Texts"]
MAX_FIELDS = ["Cited by"]
MULTI_VALUE_FIELDS = [
    "Authors", "Author full names", "Author(s) ID", "Author Keywords", "Index Keywords", "Keywords",
    "Affiliations", "Authors with affiliations", "References", "Funding Details", "Countries",
]

_SEP_BY_DB = {"openalex": "|", "oa": "|", "open alex": "|"}
_DOI_PREFIX = re.compile(r"^(https?://)?(dx\.)?doi\.org/", re.I)


# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------

def _isna(value: Any) -> bool:
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_doi(value: Any) -> Optional[str]:
    """Lower-case DOI without URL prefix / 'doi:' / whitespace; None if empty."""
    if _isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null", "<na>"):
        return None
    s = _DOI_PREFIX.sub("", s)
    s = re.sub(r"^doi:\s*", "", s, flags=re.I).strip().rstrip(".").lower()
    return s if s.startswith("10.") else None


def normalize_title(value: Any) -> Optional[str]:
    """Lower-case alphanumeric title key (punctuation, accents-insensitive spacing removed)."""
    if _isna(value):
        return None
    s = str(value).strip().lower()
    if not s or s in ("nan", "<na>", "none"):
        return None
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s or None


def _db_key(db: str) -> str:
    return str(db).strip().lower().replace("_", " ").replace("-", " ")


def _is_missing(v: Any) -> bool:
    if _isna(v):
        return True
    if isinstance(v, str) and v.strip().lower() in ("", "nan", "<na>", "none"):
        return True
    return False


# ---------------------------------------------------------------------------
# harmonisation of one source
# ---------------------------------------------------------------------------

def harmonize(
    df: pd.DataFrame,
    db: str,
    target_sep: str = "; ",
    doi_col: str = "DOI",
    year_col: str = "Year",
    citations_col: str = "Cited by",
    multi_value_fields: Sequence[str] = MULTI_VALUE_FIELDS,
) -> pd.DataFrame:
    """Return a copy with unified separators, DOI, numeric Year / Cited by and a ``Source DB`` column."""
    out = df.copy()
    dbk = _db_key(db)
    src_sep = _SEP_BY_DB.get(dbk, "; ")
    if src_sep.strip() != target_sep.strip():
        for col in multi_value_fields:
            if col in out.columns:
                s = out[col]
                mask = s.notna()
                out.loc[mask, col] = (
                    s[mask].astype(str).str.split(re.escape(src_sep.strip()))
                    .map(lambda parts: target_sep.join(p.strip() for p in parts if p.strip()))
                )
    if doi_col in out.columns:
        out[doi_col] = out[doi_col].map(normalize_doi)
    if year_col in out.columns:
        out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
    if citations_col in out.columns:
        out[citations_col] = pd.to_numeric(out[citations_col], errors="coerce")
    out["Source DB"] = dbk
    return out


def load_sources(
    sources: Sequence[Union[Tuple[str, str], Tuple[pd.DataFrame, str]]],
    harmonize_sources: bool = True,
) -> List[Tuple[pd.DataFrame, str]]:
    """
    Read / harmonise several sources.

    ``sources`` is a list of ``(path_or_DataFrame, db)`` tuples, e.g.
    ``[("scopus.csv", "scopus"), ("wos.txt", "wos"), (df_oa, "openalex")]``.
    Files are read with ``biblium.readbib.read_bibfile``.
    """
    out = []
    for item, db in sources:
        if isinstance(item, pd.DataFrame):
            df = item
        else:
            from biblium import readbib
            df = readbib.read_bibfile(str(item), db)
        if harmonize_sources:
            df = harmonize(df, db)
        out.append((df, db))
    return out


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------

def _union_join(values: Iterable[Any], sep: str = "; ") -> Any:
    seen, items = set(), []
    for v in values:
        if _is_missing(v):
            continue
        for part in str(v).split(sep.strip()):
            p = part.strip()
            k = p.lower()
            if p and k not in seen:
                seen.add(k)
                items.append(p)
    return sep.join(items) if items else np.nan


def _longest(values: Iterable[Any]) -> Any:
    best = None
    for v in values:
        if _is_missing(v):
            continue
        if best is None or len(str(v)) > len(str(best)):
            best = v
    return best if best is not None else np.nan


def _first_valid(values: Iterable[Any]) -> Any:
    for v in values:
        if not _is_missing(v):
            return v
    return np.nan


def _make_keys(df: pd.DataFrame, key: str, doi_col: str, title_col: str) -> Tuple[pd.Series, pd.Series]:
    """Return (merge key, key type) per row; rows without a usable key get a unique key."""
    doi = df[doi_col].map(normalize_doi) if doi_col in df.columns else pd.Series([None] * len(df), index=df.index)
    title = df[title_col].map(normalize_title) if title_col in df.columns else pd.Series([None] * len(df), index=df.index)
    keys, kinds = [], []
    for i, (d, t) in enumerate(zip(doi, title)):
        # str-dtype Series turn None into NaN, and NaN is truthy -> normalise explicitly
        d = d if isinstance(d, str) and d else None
        t = t if isinstance(t, str) and t else None
        if key == "doi":
            k, kind = (f"doi:{d}", "doi") if d else (None, None)
        elif key == "title":
            k, kind = (f"title:{t}", "title") if t else (None, None)
        else:  # doi_title: DOI first, title as fallback
            if d:
                k, kind = f"doi:{d}", "doi"
            elif t:
                k, kind = f"title:{t}", "title"
            else:
                k, kind = None, None
        if k is None:
            k, kind = f"row:{i}", "none"
        keys.append(k)
        kinds.append(kind)
    return pd.Series(keys, index=df.index), pd.Series(kinds, index=df.index)


def combine_databases(
    sources: Sequence[Union[Tuple[str, str], Tuple[pd.DataFrame, str]]],
    key: str = "doi",
    columns: str = "union",
    priority: Optional[Sequence[str]] = None,
    keep_source_columns: bool = True,
    doi_col: str = "DOI",
    title_col: str = "Title",
    sep: str = "; ",
    harmonize_sources: bool = True,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Combine and de-duplicate records from several databases.

    Parameters
    ----------
    sources : list of (path | DataFrame, db)
    key : {"doi", "title", "doi_title"}
        Matching key: DOI (default), normalised title, or DOI with title fallback
        for records without a DOI.
    columns : {"union", "intersection"}
        Keep all columns of all sources, or only the columns common to all.
    priority : list of db names, optional
        Which source wins for ordinary fields (first = most trusted).
        Defaults to ``DEFAULT_PRIORITY``; unknown databases go last in input order.
    keep_source_columns : bool
        Add ``Cited by (<db>)`` columns with the per-source citation counts.

    Returns
    -------
    merged : pd.DataFrame
        One row per unique record with provenance columns ``Source DB``
        (e.g. "scopus; wos"), ``N sources``, ``Merge key`` and ``Match type``.
    report : dict
        n_input (per source), n_output, n_duplicate_groups, n_removed,
        n_within_source_duplicates, overlap (DataFrame of source combinations),
        pairwise (DataFrame source × source shared records), columns_kept,
        columns_dropped, column_coverage (DataFrame), key, columns policy.
    """
    if key not in ("doi", "title", "doi_title"):
        raise ValueError("key must be 'doi', 'title' or 'doi_title'")
    if columns not in ("union", "intersection"):
        raise ValueError("columns must be 'union' or 'intersection'")

    loaded = load_sources(sources, harmonize_sources=harmonize_sources)
    if not loaded:
        raise ValueError("No sources given")
    dbs = [_db_key(db) for _, db in loaded]

    prio = [_db_key(p) for p in (priority or DEFAULT_PRIORITY)]
    rank = {}
    for i, db in enumerate(dbs):
        rank[db] = prio.index(db) if db in prio else len(prio) + i

    # column policy
    _internal = {"Source DB", "N sources", "Merge key", "Match type"}
    col_sets = [set(df.columns) - _internal for df, _ in loaded]
    all_cols: List[str] = []
    for df, _ in loaded:
        for c in df.columns:
            if c not in all_cols and c not in _internal:
                all_cols.append(c)
    common = set.intersection(*col_sets)
    if columns == "intersection":
        keep_cols = [c for c in all_cols if c in common]
    else:
        keep_cols = all_cols
    dropped = [c for c in all_cols if c not in keep_cols]

    frames = []
    for df, db in loaded:
        f = df.reindex(columns=keep_cols).copy()
        f["Source DB"] = _db_key(db)
        f["_rank"] = rank[_db_key(db)]
        frames.append(f)
    big = pd.concat(frames, ignore_index=True)
    n_input = {db: int(len(df)) for df, db in loaded}

    mkey, mkind = _make_keys(big, key, doi_col, title_col)
    big["Merge key"] = mkey.values
    big["Match type"] = mkind.values
    big["_order"] = np.arange(len(big))
    big = big.sort_values(["Merge key", "_rank", "_order"], kind="stable")

    value_cols = [c for c in keep_cols if c not in ("Source DB",)]
    groups = big.groupby("Merge key", sort=False)
    sizes = groups.size()
    dup_keys = set(sizes[sizes > 1].index)

    # fast path for singletons, explicit merge for duplicate groups
    singles = big[~big["Merge key"].isin(dup_keys)].copy()
    merged_rows: List[Dict[str, Any]] = []
    within_source_dups = 0
    for k in dup_keys:
        g = big[big["Merge key"] == k]
        src_list = list(g["Source DB"])
        within_source_dups += len(src_list) - len(set(src_list))
        rec: Dict[str, Any] = {}
        for c in value_cols:
            vals = g[c].tolist()
            if c in UNION_FIELDS:
                rec[c] = _union_join(vals, sep)
            elif c in LONGEST_FIELDS:
                rec[c] = _longest(vals)
            elif c in MAX_FIELDS:
                nums = pd.to_numeric(pd.Series(vals), errors="coerce")
                rec[c] = float(nums.max()) if nums.notna().any() else np.nan
            else:
                rec[c] = _first_valid(vals)
        ordered_sources = []
        for s in src_list:
            if s not in ordered_sources:
                ordered_sources.append(s)
        rec["Source DB"] = "; ".join(ordered_sources)
        rec["Merge key"] = k
        rec["Match type"] = g["Match type"].iloc[0]
        rec["_order"] = int(g["_order"].min())
        if keep_source_columns:
            for c in MAX_FIELDS:
                if c in value_cols:
                    for s in ordered_sources:
                        sub = pd.to_numeric(g.loc[g["Source DB"] == s, c], errors="coerce")
                        rec[f"{c} ({s})"] = float(sub.max()) if sub.notna().any() else np.nan
        merged_rows.append(rec)

    if keep_source_columns:
        for c in MAX_FIELDS:
            if c in value_cols:
                for s in sorted(set(dbs), key=lambda x: rank[x]):
                    col = f"{c} ({s})"
                    singles[col] = np.where(singles["Source DB"] == s, singles[c], np.nan)
    merged = pd.concat([singles.drop(columns=["_rank"]), pd.DataFrame(merged_rows)], ignore_index=True)
    merged = merged.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    merged["N sources"] = merged["Source DB"].astype(str).str.count(";") + 1

    # tidy column order: data columns, then provenance / per-source columns
    prov = ["Source DB", "N sources", "Merge key", "Match type"]
    per_src = [c for c in merged.columns if any(c.startswith(f"{m} (") for m in MAX_FIELDS)]
    ordered = [c for c in keep_cols if c in merged.columns] + per_src + prov
    merged = merged[[c for c in ordered if c in merged.columns]]
    if "Cited by" in merged.columns:
        merged["Cited by"] = pd.to_numeric(merged["Cited by"], errors="coerce")
    if "Year" in merged.columns:
        merged["Year"] = pd.to_numeric(merged["Year"], errors="coerce").astype("Int64")

    n_output = int(len(merged))
    total_in = int(sum(n_input.values()))
    report: Dict[str, Any] = {
        "key": key, "columns": columns, "priority": [d for d in sorted(set(dbs), key=lambda x: rank[x])],
        "n_input": n_input, "n_input_total": total_in, "n_output": n_output,
        "n_duplicate_groups": int(len(dup_keys)), "n_removed": total_in - n_output,
        "n_within_source_duplicates": int(within_source_dups),
        "n_without_key": int((merged["Match type"] == "none").sum()),
        "overlap": source_overlap_table(merged),
        "pairwise": _pairwise_overlap(merged, dbs),
        "columns_kept": keep_cols, "columns_dropped": dropped,
        "column_coverage": column_coverage(merged),
    }
    if verbose:
        print(f"Combined {total_in} records from {len(dbs)} sources -> {n_output} unique "
              f"({report['n_removed']} duplicates removed; key={key}, columns={columns})")
    return merged, report


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def source_overlap_table(merged: pd.DataFrame, source_col: str = "Source DB") -> pd.DataFrame:
    """Records per combination of sources (an UpSet-style table)."""
    combo = merged[source_col].astype(str).map(lambda s: "; ".join(sorted(p.strip() for p in s.split(";") if p.strip())))
    tab = combo.value_counts().rename_axis("Sources").reset_index(name="Records")
    tab["N sources"] = tab["Sources"].str.count(";") + 1
    tab["Share %"] = 100.0 * tab["Records"] / len(merged) if len(merged) else np.nan
    return tab.sort_values(["N sources", "Records"], ascending=[False, False]).reset_index(drop=True)


def _pairwise_overlap(merged: pd.DataFrame, dbs: Sequence[str]) -> pd.DataFrame:
    names = []
    for d in dbs:
        if d not in names:
            names.append(d)
    sets = {d: set(merged.index[merged["Source DB"].astype(str).str.split(";").map(lambda parts: d in [p.strip() for p in parts])]) for d in names}
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            mat.loc[a, b] = len(sets[a] & sets[b])
    mat.index.name = "Source"
    return mat.astype(int)


def column_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Share of non-missing cells per column (to judge the cost of ``columns='union'``)."""
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        miss = s.isna() | (s.astype(str).str.strip().isin(["", "nan", "None"]))
        rows.append({"Column": c, "Non-missing": int((~miss).sum()), "Coverage %": 100.0 * (~miss).sum() / n if n else np.nan})
    return pd.DataFrame(rows).sort_values("Coverage %", ascending=False).reset_index(drop=True)


def plot_source_overlap(
    overlap: pd.DataFrame,
    title: Optional[str] = "Records by source combination",
    ax=None,
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
    color_single: str = "#94a3b8",
    color_multi: str = "#3b82f6",
):
    """Horizontal bars of the source-combination table (multi-source combinations highlighted)."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(overlap) + 1)))
    else:
        fig = ax.figure
    d = overlap.iloc[::-1]
    colors = [color_multi if n > 1 else color_single for n in d["N sources"]]
    ax.barh(d["Sources"].astype(str), d["Records"], color=colors)
    for i, (v, s) in enumerate(zip(d["Records"], d["Share %"])):
        ax.text(v, i, f" {v} ({s:.0f} %)", va="center", fontsize=8)
    ax.set_xlabel("Records")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if f_name:
        fig.savefig(f_name, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
