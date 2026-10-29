# -*- coding: utf-8 -*-
"""
biblium.journal_rankings
========================

Journal ranking lists (SCImago SJR, JCR, ABS/AJG, ABDC, national lists …)
matched to a bibliographic frame by ISSN or by title. New in 2.19.

Contents
--------
``normalize_issn``, ``split_issns``, ``normalize_title``   key normalisation
``load_ranking_list``          read a CSV/XLSX/DataFrame into a normalised list
``match_sources``              add Journal Rank / Quartile / Category columns
``ranking_summary``            documents and citation impact per quartile
``ranking_by_year``            year × quartile counts
``filter_by_ranking``          keep documents by quartile / rank / category
``plot_ranking_distribution``  stacked bars per year by quartile

The SCImago CSV export (``;``-separated, columns ``Rank, Title, Issn, SJR,
SJR Best Quartile, Categories``) is recognised automatically; any other list
is handled by naming the columns.

Author: Lan Umek
Version: 2.19.0
"""
from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "normalize_issn",
    "split_issns",
    "normalize_title",
    "load_ranking_list",
    "match_sources",
    "ranking_summary",
    "ranking_by_year",
    "filter_by_ranking",
    "plot_ranking_distribution",
    "QUARTILES",
]

QUARTILES = ("Q1", "Q2", "Q3", "Q4")
UNRANKED = "Unranked"

RANK_COL, QUART_COL, CAT_COL, MATCH_COL, SCORE_COL = (
    "Journal Rank", "Journal Quartile", "Journal Category", "Ranking match", "Journal Score")

# Column name candidates for auto-detection (lower-case).
_TITLE_CANDS = ("title", "journal", "journal title", "source title", "journal name", "name", "source")
_ISSN_CANDS = ("issn", "issns", "issn (print)", "print issn", "issn1", "eissn", "e-issn")
_RANK_CANDS = ("rank", "ranking", "ajg rank", "abs rank", "abdc rank", "rating", "grade", "position")
_QUART_CANDS = ("sjr best quartile", "quartile", "best quartile", "jif quartile", "jcr quartile", "q")
_CAT_CANDS = ("categories", "category", "subject", "subject category", "field", "area", "discipline")
_SCORE_CANDS = ("sjr", "jif", "impact factor", "journal impact factor", "score", "if")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _is_missing(v) -> bool:
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


def split_issns(s) -> List[str]:
    """All normalised ISSNs found in a string such as ``"1234-5678, 8765-432X"``.

    Accepts ``,``, ``;``, ``/``, ``|`` and whitespace as separators; invalid
    tokens are dropped; order is preserved and duplicates removed.
    """
    if _is_missing(s):
        return []
    if isinstance(s, (list, tuple, set)):
        toks = [str(x) for x in s]
    elif isinstance(s, (int, float, np.integer, np.floating)):
        toks = [s]
    else:
        toks = re.split(r"[,;/|\s]+", str(s))
    out: List[str] = []
    for t in toks:
        n = normalize_issn(t)
        if n and n not in out:
            out.append(n)
    return out


def normalize_issn(s) -> Optional[str]:
    """8-character upper-case ISSN without hyphen, or ``None``.

    ``"1234-567X"``, ``"1234567x"`` and ``"12345678"`` all give ``"1234567X"`` /
    ``"12345678"``. A string with several ISSNs returns the first valid one.
    """
    if _is_missing(s):
        return None
    if isinstance(s, (int, float, np.integer, np.floating)) and float(s).is_integer():
        s = str(int(s)).zfill(8)  # leading zeros lost by numeric conversion
    txt = str(s).strip()
    # several ISSNs → first one
    if re.search(r"[,;/|\s]", txt):
        for tok in re.split(r"[,;/|\s]+", txt):
            n = normalize_issn(tok)
            if n:
                return n
        return None
    v = re.sub(r"[^0-9Xx]", "", txt).upper()
    if len(v) == 7 and re.fullmatch(r"\d{7}", v):
        v = "0" + v  # leading zero lost by numeric conversion (e.g. 1382 → 01382…)
    if len(v) != 8 or not re.fullmatch(r"\d{7}[0-9X]", v):
        return None
    return v


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_title(s) -> str:
    """Lower-case title key: punctuation removed, spaces collapsed,
    leading ``"the "`` dropped, ``&`` read as ``and``."""
    if _is_missing(s):
        return ""
    t = str(s).lower().replace("&", " and ")
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if t.startswith("the "):
        t = t[4:]
    return t


def _normalize_quartile(v) -> Optional[str]:
    if _is_missing(v):
        return None
    t = str(v).strip().upper()
    m = re.fullmatch(r"Q?\s*([1-4])(\.0)?", t)
    if m:
        return f"Q{m.group(1)}"
    return t if t else None


def _to_rank(v):
    if _is_missing(v):
        return np.nan
    try:
        f = float(str(v).replace(",", "."))
        return int(f) if f.is_integer() else f
    except ValueError:
        return str(v).strip()


def _to_score(v):
    if _is_missing(v):
        return np.nan
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _pick(columns: Sequence[str], cands: Sequence[str]) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in columns}
    for cand in cands:
        if cand in low:
            return low[cand]
    return None


def _read_table(path: str, sep: Optional[str]) -> pd.DataFrame:
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path)
    if sep is None:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            head = fh.readline()
        sep = ";" if head.count(";") > head.count(",") else ("\t" if head.count("\t") > head.count(",") else ",")
    return pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig", engine="python", index_col=False)


def load_ranking_list(
    path_or_df: Union[str, "os.PathLike[str]", pd.DataFrame],
    title_col: Optional[str] = None,
    issn_col: Optional[str] = None,
    rank_col: Optional[str] = None,
    quartile_col: Optional[str] = None,
    category_col: Optional[str] = None,
    sep: Optional[str] = None,
    score_col: Optional[str] = None,
) -> pd.DataFrame:
    """Read a journal ranking list into a normalised frame.

    Columns of the result: ``Title``, ``Title norm``, ``ISSN`` (list of
    normalised ISSNs), ``Rank``, ``Quartile`` (``Q1``..``Q4`` when it looks
    like a quartile), ``Category`` and ``Score`` (e.g. SJR); absent ones are
    NaN. Column names are auto-detected (SCImago: ``Title, Issn, Rank, SJR,
    SJR Best Quartile, Categories``) unless given explicitly. CSV delimiter
    is auto-detected (``;`` for the SCImago export) unless ``sep`` is given.
    """
    if isinstance(path_or_df, pd.DataFrame):
        raw = path_or_df.copy()
    else:
        raw = _read_table(str(path_or_df), sep)
    cols = list(raw.columns)
    title_col = title_col or _pick(cols, _TITLE_CANDS)
    issn_col = issn_col or _pick(cols, _ISSN_CANDS)
    rank_col = rank_col or _pick(cols, _RANK_CANDS)
    quartile_col = quartile_col or _pick(cols, _QUART_CANDS)
    category_col = category_col or _pick(cols, _CAT_CANDS)
    score_col = score_col or _pick(cols, _SCORE_CANDS)
    if title_col is None and issn_col is None:
        raise ValueError("Ranking list needs a title and/or an ISSN column "
                         f"(columns found: {cols})")
    for name, c in (("title_col", title_col), ("issn_col", issn_col), ("rank_col", rank_col),
                    ("quartile_col", quartile_col), ("category_col", category_col), ("score_col", score_col)):
        if c is not None and c not in raw.columns:
            raise ValueError(f"{name}={c!r} not found in the ranking list")

    out = pd.DataFrame(index=raw.index)
    out["Title"] = raw[title_col].astype(str).str.strip() if title_col else np.nan
    out["Title norm"] = out["Title"].map(normalize_title) if title_col else ""
    out["ISSN"] = raw[issn_col].map(split_issns) if issn_col else [[] for _ in range(len(raw))]
    out["Rank"] = raw[rank_col].map(_to_rank) if rank_col else np.nan
    out["Quartile"] = raw[quartile_col].map(_normalize_quartile) if quartile_col else None
    out["Category"] = raw[category_col].astype(str).str.strip().where(raw[category_col].notna()) if category_col else np.nan
    out["Score"] = raw[score_col].map(_to_score) if score_col else np.nan
    out = out.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _build_lookups(ranking: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """ISSN → row position, title-norm → row position (first occurrence wins)."""
    by_issn: Dict[str, int] = {}
    by_title: Dict[str, int] = {}
    issn_lists = ranking["ISSN"].tolist() if "ISSN" in ranking.columns else [[]] * len(ranking)
    titles = ranking["Title norm"].tolist() if "Title norm" in ranking.columns else [""] * len(ranking)
    for pos, (lst, t) in enumerate(zip(issn_lists, titles)):
        for i in (lst if isinstance(lst, (list, tuple)) else split_issns(lst)):
            by_issn.setdefault(i, pos)
        if isinstance(t, str) and t:
            by_title.setdefault(t, pos)
    return by_issn, by_title


def match_sources(
    df: pd.DataFrame,
    ranking: pd.DataFrame,
    source_col: str = "Source title",
    issn_col: Optional[str] = "ISSN",
) -> pd.DataFrame:
    """Attach ranking information to every document.

    Matching is by ISSN first (any ISSN of the document found in the ranking
    row's ISSN list), otherwise by normalised title. Adds the columns
    ``Journal Rank``, ``Journal Quartile``, ``Journal Category``,
    ``Journal Score`` and ``Ranking match`` (``'issn'`` / ``'title'`` / None).

    Parameters
    ----------
    ranking : pd.DataFrame
        Output of ``load_ranking_list``.
    """
    out = df.copy()
    by_issn, by_title = _build_lookups(ranking)

    n = len(out)
    pos = np.full(n, -1, dtype=int)
    how = np.full(n, None, dtype=object)

    if issn_col is not None and issn_col in out.columns and by_issn:
        for k, v in enumerate(out[issn_col].tolist()):
            for i in split_issns(v):
                p = by_issn.get(i)
                if p is not None:
                    pos[k] = p
                    how[k] = "issn"
                    break
    if source_col in out.columns and by_title:
        titles = out[source_col].map(normalize_title).tolist()
        for k in range(n):
            if pos[k] < 0 and titles[k]:
                p = by_title.get(titles[k])
                if p is not None:
                    pos[k] = p
                    how[k] = "title"

    def _col(name, default=np.nan):
        if name not in ranking.columns:
            return [default] * n
        vals = ranking[name].tolist()
        return [vals[p] if p >= 0 else default for p in pos]

    out[RANK_COL] = _col("Rank")
    out[QUART_COL] = pd.Series(_col("Quartile", None), index=out.index, dtype=object)
    out[CAT_COL] = _col("Category")
    out[SCORE_COL] = _col("Score")
    out[MATCH_COL] = pd.Series(list(how), index=out.index, dtype=object)
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _ncs(df: pd.DataFrame, citations_col: str, year_col: Optional[str]) -> pd.Series:
    """Year-normalised citation score: citations / mean citations of the
    publication year (NaN citations count as 0; same formula as
    ``indicators._year_normalised``)."""
    cit = pd.to_numeric(df[citations_col], errors="coerce").fillna(0.0) if citations_col in df.columns \
        else pd.Series(0.0, index=df.index)
    if year_col is not None and year_col in df.columns:
        grp = pd.to_numeric(df[year_col], errors="coerce").fillna(-1).astype(int)
    else:
        grp = pd.Series(0, index=df.index)
    mean_by = cit.groupby(grp).transform("mean")
    return pd.Series(np.where(mean_by > 0, cit / mean_by.replace(0, np.nan), np.nan), index=df.index)


def _quartile_labels(df_matched: pd.DataFrame) -> pd.Series:
    q = df_matched[QUART_COL] if QUART_COL in df_matched.columns else pd.Series(None, index=df_matched.index)
    q = q.map(lambda v: UNRANKED if _is_missing(v) else str(v))
    return q


def _quartile_order(labels: Iterable[str]) -> List[str]:
    labs = set(labels)
    order = [q for q in QUARTILES if q in labs]
    order += sorted(l for l in labs if l not in QUARTILES and l != UNRANKED)
    if UNRANKED in labs:
        order.append(UNRANKED)
    return order


def ranking_summary(
    df_matched: pd.DataFrame,
    citations_col: str = "Cited by",
    year_col: Optional[str] = "Year",
) -> pd.DataFrame:
    """Documents and citation impact per journal quartile (plus ``Unranked``).

    Columns: ``Quartile, Docs, % Docs, Total citations, Mean citations, MNCS``.
    MNCS is the mean of year-normalised citation scores (citations divided by
    the mean citations of all documents published in the same year).
    """
    d = pd.DataFrame(index=df_matched.index)
    d["Quartile"] = _quartile_labels(df_matched)
    d["cit"] = pd.to_numeric(df_matched[citations_col], errors="coerce") if citations_col in df_matched.columns else np.nan
    d["ncs"] = _ncs(df_matched, citations_col, year_col)
    n = len(d)
    rows = []
    for q in _quartile_order(d["Quartile"]):
        sub = d[d["Quartile"] == q]
        rows.append({
            "Quartile": q,
            "Docs": int(len(sub)),
            "% Docs": 100.0 * len(sub) / n if n else np.nan,
            "Total citations": float(sub["cit"].fillna(0).sum()),
            "Mean citations": float(sub["cit"].mean()) if sub["cit"].notna().any() else np.nan,
            "MNCS": float(sub["ncs"].mean()) if sub["ncs"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows, columns=["Quartile", "Docs", "% Docs", "Total citations", "Mean citations", "MNCS"])


def ranking_by_year(df_matched: pd.DataFrame, year_col: str = "Year") -> pd.DataFrame:
    """Year × quartile document counts (years as rows, ``Q1..Q4`` and
    ``Unranked`` as columns)."""
    q = _quartile_labels(df_matched)
    years = pd.to_numeric(df_matched[year_col], errors="coerce")
    ok = years.notna()
    tab = pd.crosstab(years[ok].astype(int), q[ok])
    tab = tab.reindex(columns=_quartile_order(tab.columns), fill_value=0)
    tab.index.name = year_col
    tab.columns.name = "Quartile"
    return tab


def filter_by_ranking(
    df_matched: pd.DataFrame,
    quartiles: Optional[Sequence[str]] = None,
    max_rank: Optional[float] = None,
    categories: Optional[Sequence[str]] = None,
    keep_unranked: bool = False,
) -> pd.DataFrame:
    """Keep documents whose journal satisfies the ranking criteria.

    Parameters
    ----------
    quartiles : e.g. ``("Q1", "Q2")``
    max_rank : keep ``Journal Rank <= max_rank`` (numeric ranks only)
    categories : keep journals whose ``Journal Category`` contains any of the
        given substrings (case-insensitive)
    keep_unranked : also keep documents without a ranking match
    """
    mask = pd.Series(True, index=df_matched.index)
    matched = df_matched[MATCH_COL].notna() if MATCH_COL in df_matched.columns else pd.Series(True, index=df_matched.index)
    if quartiles is not None:
        qs = {_normalize_quartile(q) for q in quartiles}
        mask &= df_matched[QUART_COL].map(_normalize_quartile).isin(qs)
    if max_rank is not None:
        r = pd.to_numeric(df_matched[RANK_COL], errors="coerce")
        mask &= r <= max_rank
    if categories is not None:
        cats = [str(c).lower() for c in categories]
        col = df_matched[CAT_COL].fillna("").astype(str).str.lower()
        mask &= col.map(lambda s: any(c in s for c in cats))
    if keep_unranked:
        mask |= ~matched
    return df_matched[mask].copy()


# ---------------------------------------------------------------------------
# Plot
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


def plot_ranking_distribution(
    by_year_table: pd.DataFrame,
    title: Optional[str] = "Documents by journal quartile",
    cmap: str = "Blues",
    unranked_color: str = "#bdbdbd",
    stacked: bool = True,
    percent: bool = False,
    ax=None,
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    """Bars per year by quartile (``Q1`` darkest). Input is the table from
    ``ranking_by_year``; ``percent=True`` shows shares instead of counts."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    tab = by_year_table.copy()
    if percent:
        tot = tab.sum(axis=1).replace(0, np.nan)
        tab = tab.div(tot, axis=0) * 100.0
    cols = _quartile_order(tab.columns)
    tab = tab[cols]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    q_cols = [c for c in cols if c != UNRANKED]
    shades = mpl.colormaps[cmap](np.linspace(0.9, 0.35, max(len(q_cols), 1)))
    colors = {c: shades[i] for i, c in enumerate(q_cols)}
    colors[UNRANKED] = unranked_color
    xs = np.arange(len(tab))
    k = len(cols)
    width = 0.8 if stacked else 0.8 / max(k, 1)
    bottom = np.zeros(len(tab))
    for i, c in enumerate(cols):
        vals = tab[c].fillna(0).to_numpy(dtype=float)
        if stacked:
            ax.bar(xs, vals, bottom=bottom, color=colors[c], label=c, width=width)
            bottom += vals
        else:
            ax.bar(xs - 0.4 + width * (i + 0.5), vals, color=colors[c], label=c, width=width)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in tab.index], rotation=90 if len(tab) > 12 else 0)
    ax.set_ylabel("Share of documents (%)" if percent else "Number of documents")
    ax.set_xlabel(str(tab.index.name or "Year"))
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _finish(fig, ax, title, f_name, dpi, show)
