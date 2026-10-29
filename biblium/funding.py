# -*- coding: utf-8 -*-
"""
biblium.funding
===============

Funding analysis from the Scopus ``Funding Details`` / ``Funding Texts``
columns. New in 2.19.

Scopus lists one entry per funder, separated by ``;``. Two layouts occur:

* ``"European Commission, EC: 123"``                 (older: grant after colon)
* ``"Chinese Academy of Sciences, CAS, (XDA28020102)"`` (newer: grant in parentheses;
  the same funder is often repeated once more without the grant)
* ``"Universidad de Extremadura, UEx, UNEX"``        (several acronyms, no grant)
* ``"Javna Agencija za Raziskovalno Dejavnost RS, ARRS"`` (no grant)

Contents
--------
``parse_funding_details``   list of ``{funder, acronym, grant}`` dicts
``normalize_funder``        lower-case key with alias map (``FUNDER_ALIASES``)
``add_funding_flags``       Funded / N funders / Funders columns
``funder_counts``           documents and citation impact per funder (full counting)
``funding_by_year``         funded share and citations per year
``funded_vs_unfunded``      citation comparison incl. Mann–Whitney U test
``funder_cooccurrence``     funder × funder co-funding matrix
``funder_keyword_profile``  topics supported by each major funder
``plot_funding_trend``, ``plot_top_funders``

Author: Lan Umek
Version: 2.19.0
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "FUNDER_ALIASES",
    "parse_funding_details",
    "normalize_funder",
    "funder_display_name",
    "add_funding_flags",
    "funder_counts",
    "funding_by_year",
    "funded_vs_unfunded",
    "funder_cooccurrence",
    "funder_keyword_profile",
    "plot_funding_trend",
    "plot_top_funders",
]

#: Alias → canonical key (all lower-case). Keys are matched against the
#: normalised funder name *and* against acronyms.
FUNDER_ALIASES: Dict[str, str] = {
    "nsfc": "national natural science foundation of china",
    "national natural science foundation of china": "national natural science foundation of china",
    "nsf": "national science foundation",
    "national science foundation": "national science foundation",
    "nih": "national institutes of health",
    "national institutes of health": "national institutes of health",
    "ec": "european commission",
    "european commission": "european commission",
    "h2020": "european commission",
    "horizon 2020": "european commission",
    "horizon 2020 framework programme": "european commission",
    "european union": "european commission",
    "arrs": "slovenian research and innovation agency",
    "aris": "slovenian research and innovation agency",
    "javna agencija za raziskovalno dejavnost rs": "slovenian research and innovation agency",
    "javna agencija za raziskovalno dejavnost republike slovenije": "slovenian research and innovation agency",
    "slovenian research agency": "slovenian research and innovation agency",
    "slovenian research and innovation agency": "slovenian research and innovation agency",
    "dfg": "deutsche forschungsgemeinschaft",
    "deutsche forschungsgemeinschaft": "deutsche forschungsgemeinschaft",
    "ukri": "uk research and innovation",
    "uk research and innovation": "uk research and innovation",
}

#: Display names for canonical keys produced by the alias map.
_CANONICAL_DISPLAY: Dict[str, str] = {
    "national natural science foundation of china": "National Natural Science Foundation of China",
    "national science foundation": "National Science Foundation",
    "national institutes of health": "National Institutes of Health",
    "european commission": "European Commission",
    "slovenian research and innovation agency": "Slovenian Research and Innovation Agency",
    "deutsche forschungsgemeinschaft": "Deutsche Forschungsgemeinschaft",
    "uk research and innovation": "UK Research and Innovation",
}

_PAREN_RE = re.compile(r"\(([^()]*)\)\s*$")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    try:
        if v is pd.NA:
            return True
    except Exception:  # pragma: no cover
        pass
    return isinstance(v, str) and v.strip() == ""


def _looks_like_acronym(tok: str) -> bool:
    """Short token without spaces and with a majority of upper-case letters
    (``CAS``, ``UEx``, ``H2020``), but not an ordinary capitalised word
    (``Malaysia``)."""
    t = tok.strip()
    if not t or " " in t or len(t) > 12:
        return False
    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return False
    upper = sum(ch.isupper() for ch in letters)
    return upper / len(letters) >= 0.5


def _parse_entry(entry: str) -> Optional[Dict[str, Optional[str]]]:
    e = entry.strip().strip(".").strip()
    if not e:
        return None
    grant: Optional[str] = None
    # grant in trailing parentheses: "Funder, ACR, (G1, G2)"
    m = _PAREN_RE.search(e)
    if m:
        grant = m.group(1).strip() or None
        e = e[: m.start()].rstrip(" ,;")
    # grant after colon: "Funder, ACR: G1, G2"
    if grant is None and ":" in e:
        left, right = e.split(":", 1)
        if right.strip():
            grant = right.strip()
        e = left.rstrip(" ,;")
    parts = [p.strip() for p in e.split(",") if p.strip()]
    if not parts:
        return None  # e.g. "(2020004)" without a funder name
    funder_parts = [parts[0]]
    acronyms: List[str] = []
    for p in parts[1:]:
        if _looks_like_acronym(p):
            acronyms.append(p)
        elif not acronyms:
            funder_parts.append(p)  # "Ministry of Higher Education, Malaysia"
        else:
            acronyms.append(p)
    funder = ", ".join(funder_parts).strip()
    if not funder:
        return None
    return {"funder": funder, "acronym": acronyms[0] if acronyms else None, "grant": grant}


def parse_funding_details(
    s: Any,
    sep: str = ";",
    merge_duplicates: bool = False,
) -> List[Dict[str, Optional[str]]]:
    """Parse a Scopus ``Funding Details`` cell.

    Returns a list of ``{"funder": str, "acronym": str | None, "grant": str | None}``
    in the order given. Entries that carry only a grant number (no funder
    name) are dropped. With ``merge_duplicates=True`` repeated entries of the
    same funder (common in newer exports: once with, once without the grant)
    are merged into one dict keeping the first acronym / grant found.
    """
    if _is_missing(s):
        return []
    items: List[Dict[str, Optional[str]]] = []
    for entry in str(s).split(sep):
        d = _parse_entry(entry)
        if d is not None:
            items.append(d)
    if merge_duplicates:
        merged: Dict[str, Dict[str, Optional[str]]] = {}
        for d in items:
            key = normalize_funder(d["funder"])
            if key not in merged:
                merged[key] = dict(d)
            else:
                for f in ("acronym", "grant"):
                    if merged[key][f] is None and d[f] is not None:
                        merged[key][f] = d[f]
        items = list(merged.values())
    return items


def normalize_funder(name: Any, acronym: Optional[str] = None) -> str:
    """Lower-case matching key for a funder: whitespace collapsed, trailing
    punctuation removed, aliases from ``FUNDER_ALIASES`` applied (the acronym,
    if given, is tried as an alias too)."""
    if _is_missing(name):
        return ""
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    key = key.rstrip(" .,;:-")
    key = key.replace("the ", "", 1) if key.startswith("the ") else key
    if key in FUNDER_ALIASES:
        return FUNDER_ALIASES[key]
    if acronym:
        a = str(acronym).strip().lower()
        if a in FUNDER_ALIASES:
            return FUNDER_ALIASES[a]
    return key


def funder_display_name(key: str, original: Optional[str] = None) -> str:
    """Display name for a normalised key: canonical name for aliases,
    otherwise the original spelling (or the key, title-cased)."""
    if key in _CANONICAL_DISPLAY:
        return _CANONICAL_DISPLAY[key]
    if original and not _is_missing(original):
        return re.sub(r"\s+", " ", str(original).strip()).rstrip(" .,;:-")
    return key


# ---------------------------------------------------------------------------
# Document-level helpers
# ---------------------------------------------------------------------------

def _funders_long(df: pd.DataFrame, details_col: str, sep: str = ";") -> pd.DataFrame:
    """Long table ``_doc, Funder key, Funder, Acronym, Grant`` — one row per
    distinct funder per document (duplicates within a document collapsed)."""
    rows = []
    if details_col not in df.columns:
        return pd.DataFrame(columns=["_doc", "Funder key", "Funder", "Acronym", "Grant"])
    for idx, val in df[details_col].items():
        for d in parse_funding_details(val, sep=sep, merge_duplicates=True):
            key = normalize_funder(d["funder"], d["acronym"])
            rows.append({"_doc": idx, "Funder key": key,
                         "Funder": funder_display_name(key, d["funder"]),
                         "Acronym": d["acronym"], "Grant": d["grant"]})
    out = pd.DataFrame(rows, columns=["_doc", "Funder key", "Funder", "Acronym", "Grant"])
    return out.drop_duplicates(["_doc", "Funder key"]).reset_index(drop=True)


def _display_names(long: pd.DataFrame) -> Dict[str, str]:
    """Most common display spelling per key."""
    out = {}
    for key, sub in long.groupby("Funder key"):
        out[key] = sub["Funder"].value_counts().index[0]
    return out


def _funded_mask(df: pd.DataFrame, details_col: str, text_col: Optional[str]) -> pd.Series:
    m = pd.Series(False, index=df.index)
    for c in (details_col, text_col):
        if c is not None and c in df.columns:
            s = df[c]
            m |= s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str).str.lower() != "nan")
    return m


def add_funding_flags(
    df: pd.DataFrame,
    details_col: str = "Funding Details",
    text_col: Optional[str] = "Funding Texts",
    sep: str = ";",
) -> pd.DataFrame:
    """Copy of ``df`` with ``Funded`` (bool: details or text non-empty),
    ``N funders`` (distinct normalised funders in the details) and
    ``Funders`` (``"; "``-joined display names)."""
    out = df.copy()
    out["Funded"] = _funded_mask(df, details_col, text_col)
    long = _funders_long(df, details_col, sep)
    names = _display_names(long) if len(long) else {}
    n = long.groupby("_doc").size() if len(long) else pd.Series(dtype=int)
    joined = (long.assign(Name=long["Funder key"].map(names)).groupby("_doc")["Name"]
              .agg(lambda x: "; ".join(x))) if len(long) else pd.Series(dtype=object)
    out["N funders"] = n.reindex(out.index).fillna(0).astype(int)
    out["Funders"] = joined.reindex(out.index)
    return out


# ---------------------------------------------------------------------------
# Citation helpers (same formula as indicators._year_normalised)
# ---------------------------------------------------------------------------

def _ncs(df: pd.DataFrame, citations_col: Optional[str], year_col: Optional[str]) -> pd.Series:
    if citations_col is None or citations_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    cit = pd.to_numeric(df[citations_col], errors="coerce").fillna(0.0)
    if year_col is not None and year_col in df.columns:
        grp = pd.to_numeric(df[year_col], errors="coerce").fillna(-1).astype(int)
    else:
        grp = pd.Series(0, index=df.index)
    mean_by = cit.groupby(grp).transform("mean")
    return pd.Series(np.where(mean_by > 0, cit / mean_by.replace(0, np.nan), np.nan), index=df.index)


def _cit(df: pd.DataFrame, citations_col: Optional[str]) -> pd.Series:
    if citations_col is None or citations_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[citations_col], errors="coerce")


# ---------------------------------------------------------------------------
# Funder statistics
# ---------------------------------------------------------------------------

def funder_counts(
    df: pd.DataFrame,
    details_col: str = "Funding Details",
    citations_col: Optional[str] = "Cited by",
    year_col: Optional[str] = "Year",
    top_n: Optional[int] = 30,
    min_docs: int = 1,
    sep: str = ";",
) -> pd.DataFrame:
    """Documents and citation impact per funder (full counting: a document
    with *k* funders counts once for each of them).

    Columns: ``Funder, Acronym, Docs, % of funded docs, Total citations,
    Mean citations, MNCS, First year, Last year``; sorted by Docs (then
    Total citations) descending. ``% of funded docs`` is relative to the
    number of documents with at least one parsed funder.
    """
    long = _funders_long(df, details_col, sep)
    cols = ["Funder", "Acronym", "Docs", "% of funded docs", "Total citations",
            "Mean citations", "MNCS", "First year", "Last year"]
    if len(long) == 0:
        return pd.DataFrame(columns=cols)
    names = _display_names(long)
    long["_cit"] = _cit(df, citations_col).reindex(long["_doc"]).to_numpy()
    long["_ncs"] = _ncs(df, citations_col, year_col).reindex(long["_doc"]).to_numpy()
    long["_year"] = (pd.to_numeric(df[year_col], errors="coerce").reindex(long["_doc"]).to_numpy()
                     if year_col is not None and year_col in df.columns else np.nan)
    n_funded = long["_doc"].nunique()
    rows = []
    for key, sub in long.groupby("Funder key", sort=False):
        acr = sub["Acronym"].dropna()
        rows.append({
            "Funder": names[key],
            "Acronym": acr.value_counts().index[0] if len(acr) else None,
            "Docs": int(len(sub)),
            "% of funded docs": 100.0 * len(sub) / n_funded,
            "Total citations": float(sub["_cit"].fillna(0).sum()),
            "Mean citations": float(sub["_cit"].mean()) if sub["_cit"].notna().any() else np.nan,
            "MNCS": float(sub["_ncs"].mean()) if sub["_ncs"].notna().any() else np.nan,
            "First year": int(sub["_year"].min()) if sub["_year"].notna().any() else pd.NA,
            "Last year": int(sub["_year"].max()) if sub["_year"].notna().any() else pd.NA,
        })
    out = pd.DataFrame(rows, columns=cols)
    out = out[out["Docs"] >= min_docs]
    out = out.sort_values(["Docs", "Total citations", "Funder"], ascending=[False, False, True],
                          kind="mergesort").reset_index(drop=True)
    if top_n is not None:
        out = out.head(int(top_n))
    return out


def funding_by_year(
    df: pd.DataFrame,
    details_col: str = "Funding Details",
    text_col: Optional[str] = "Funding Texts",
    citations_col: Optional[str] = "Cited by",
    year_col: str = "Year",
) -> pd.DataFrame:
    """Per year: ``N docs, Funded, Funded %, Mean citations funded,
    Mean citations unfunded`` (years as a column, sorted)."""
    funded = _funded_mask(df, details_col, text_col)
    years = pd.to_numeric(df[year_col], errors="coerce")
    cit = _cit(df, citations_col)
    d = pd.DataFrame({"Year": years, "Funded": funded, "cit": cit}).dropna(subset=["Year"])
    d["Year"] = d["Year"].astype(int)
    rows = []
    for y, sub in d.groupby("Year"):
        f, u = sub[sub["Funded"]], sub[~sub["Funded"]]
        rows.append({
            "Year": int(y),
            "N docs": int(len(sub)),
            "Funded": int(len(f)),
            "Funded %": 100.0 * len(f) / len(sub),
            "Mean citations funded": float(f["cit"].mean()) if f["cit"].notna().any() else np.nan,
            "Mean citations unfunded": float(u["cit"].mean()) if u["cit"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows, columns=["Year", "N docs", "Funded", "Funded %",
                                       "Mean citations funded", "Mean citations unfunded"])


def funded_vs_unfunded(
    df: pd.DataFrame,
    details_col: str = "Funding Details",
    text_col: Optional[str] = "Funding Texts",
    citations_col: str = "Cited by",
    year_col: Optional[str] = "Year",
) -> Dict[str, float]:
    """Citation comparison of funded and unfunded documents.

    Keys: ``n_funded, n_unfunded, mean_cit_funded, mean_cit_unfunded,
    median_cit_funded, median_cit_unfunded, mncs_funded, mncs_unfunded,
    mncs_ratio, share_uncited_funded, share_uncited_unfunded,
    mannwhitney_u, mannwhitney_p`` (two-sided; NaN if a group is empty).
    Missing citation counts are treated as 0 (consistent with MNCS).
    """
    funded = _funded_mask(df, details_col, text_col)
    cit = _cit(df, citations_col).fillna(0.0)
    ncs = _ncs(df, citations_col, year_col)
    f, u = cit[funded], cit[~funded]

    def _mean(s):
        return float(s.mean()) if len(s) else np.nan

    def _median(s):
        return float(s.median()) if len(s) else np.nan

    def _uncited(s):
        return float((s == 0).mean()) if len(s) else np.nan

    out: Dict[str, float] = {
        "n_funded": int(len(f)),
        "n_unfunded": int(len(u)),
        "mean_cit_funded": _mean(f),
        "mean_cit_unfunded": _mean(u),
        "median_cit_funded": _median(f),
        "median_cit_unfunded": _median(u),
        "mncs_funded": _mean(ncs[funded].dropna()),
        "mncs_unfunded": _mean(ncs[~funded].dropna()),
        "share_uncited_funded": _uncited(f),
        "share_uncited_unfunded": _uncited(u),
    }
    mu = out["mncs_unfunded"]
    out["mncs_ratio"] = out["mncs_funded"] / mu if mu and not np.isnan(mu) and mu > 0 else np.nan
    if len(f) and len(u):
        from scipy.stats import mannwhitneyu
        try:
            res = mannwhitneyu(f.to_numpy(), u.to_numpy(), alternative="two-sided")
            out["mannwhitney_u"], out["mannwhitney_p"] = float(res.statistic), float(res.pvalue)
        except ValueError:  # e.g. all values identical
            out["mannwhitney_u"], out["mannwhitney_p"] = np.nan, np.nan
    else:
        out["mannwhitney_u"], out["mannwhitney_p"] = np.nan, np.nan
    return out


def funder_cooccurrence(
    df: pd.DataFrame,
    details_col: str = "Funding Details",
    min_docs: int = 2,
    top_n: Optional[int] = 50,
    sep: str = ";",
) -> pd.DataFrame:
    """Symmetric funder × funder co-funding matrix (diagonal = number of
    documents of the funder). Only funders with at least ``min_docs``
    documents are kept, at most the ``top_n`` most frequent."""
    long = _funders_long(df, details_col, sep)
    if len(long) == 0:
        return pd.DataFrame()
    names = _display_names(long)
    counts = long["Funder key"].value_counts()
    keep = counts[counts >= min_docs]
    if top_n is not None:
        keep = keep.head(int(top_n))
    keys = list(keep.index)
    sub = long[long["Funder key"].isin(keys)]
    inc = pd.crosstab(sub["_doc"], sub["Funder key"]).reindex(columns=keys, fill_value=0)
    M = inc.T.dot(inc)
    M.index = [names[k] for k in M.index]
    M.columns = [names[k] for k in M.columns]
    M.index.name = M.columns.name = "Funder"
    return M.astype(int)


def funder_keyword_profile(
    df: pd.DataFrame,
    details_col: str = "Funding Details",
    keyword_col: str = "Author Keywords",
    top_n_funders: int = 10,
    top_n_keywords: int = 10,
    sep: str = ";",
    keyword_sep: Optional[str] = None,
) -> pd.DataFrame:
    """Long table ``Funder, Keyword, Docs``: the ``top_n_keywords`` most
    frequent keywords among the documents of each of the ``top_n_funders``
    largest funders (keywords lower-cased and stripped)."""
    keyword_sep = sep if keyword_sep is None else keyword_sep
    long = _funders_long(df, details_col, sep)
    cols = ["Funder", "Keyword", "Docs"]
    if len(long) == 0 or keyword_col not in df.columns:
        return pd.DataFrame(columns=cols)
    names = _display_names(long)
    top = long["Funder key"].value_counts().head(int(top_n_funders)).index
    kw = df[keyword_col].dropna().astype(str).str.split(keyword_sep).explode().str.strip().str.lower()
    kw = kw[(kw != "") & (kw != "nan")].rename("Keyword").reset_index().rename(columns={"index": "_doc"})
    kw = kw.drop_duplicates(["_doc", "Keyword"])
    rows = []
    for key in top:
        docs = long.loc[long["Funder key"] == key, "_doc"]
        cnt = kw[kw["_doc"].isin(docs)]["Keyword"].value_counts()
        cnt = cnt.sort_index().sort_values(ascending=False, kind="mergesort").head(int(top_n_keywords))
        for k, n in cnt.items():
            rows.append({"Funder": names[key], "Keyword": k, "Docs": int(n)})
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

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


def plot_funding_trend(
    by_year_df: pd.DataFrame,
    title: Optional[str] = "Funded documents per year",
    bar_color: str = "#c7d4e8",
    line_color: str = "#1f3b73",
    ax=None,
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    """Bars = number of documents per year, line (right axis) = funded share
    in %. Input is the table from ``funding_by_year``."""
    import matplotlib.pyplot as plt
    d = by_year_df.sort_values("Year")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    xs = np.arange(len(d))
    ax.bar(xs, d["N docs"].to_numpy(), color=bar_color, label="Documents", width=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(y)) for y in d["Year"]], rotation=90 if len(d) > 12 else 0)
    ax.set_ylabel("Number of documents")
    ax2 = ax.twinx()
    ax2.plot(xs, d["Funded %"].to_numpy(), color=line_color, marker="o", lw=2, label="Funded (%)")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Funded documents (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    return _finish(fig, ax, title, f_name, dpi, show)


def plot_top_funders(
    funder_counts_df: pd.DataFrame,
    top_n: int = 15,
    value_col: str = "Docs",
    title: Optional[str] = "Top funders",
    color: str = "#1f3b73",
    ax=None,
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    """Horizontal bars of the ``top_n`` funders by ``value_col`` (input from
    ``funder_counts``)."""
    import matplotlib.pyplot as plt
    d = funder_counts_df.sort_values(value_col, ascending=False, kind="mergesort").head(int(top_n))
    d = d.iloc[::-1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.35 * len(d) + 1)))
    else:
        fig = ax.figure
    labels = [f"{f} ({a})" if isinstance(a, str) and a else str(f)
              for f, a in zip(d["Funder"], d.get("Acronym", [None] * len(d)))]
    ax.barh(np.arange(len(d)), d[value_col].to_numpy(dtype=float), color=color)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(value_col)
    return _finish(fig, ax, title, f_name, dpi, show)
