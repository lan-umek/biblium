# -*- coding: utf-8 -*-
"""
biblium.dual_map
================

Dual-map overlay of science (new in 2.19) -- an ASJC-based analogue of the
CiteSpace dual-map overlay (Chen & Leydesdorff 2014).

The left panel shows the disciplines of the *citing* corpus: the source
title of every document is mapped to its Scopus ASJC codes and, through the
ASJC code table, to a discipline (``Field`` -- the 4-digit code label,
``Area`` -- the 2-digit subject area, e.g. "Engineering", or ``Science`` --
one of Life / Physical / Health / Social Sciences or Multidisciplinary).
The right panel shows the disciplines of the *cited* references: journal
names are parsed from Scopus reference strings and mapped the same way.
Curved flows connect citing disciplines to cited disciplines; their width is
proportional to the number of (fractionally counted) citations.

Contents
--------
Mapping
    ``load_asjc_tables``     bundled source -> ASJC map and the ASJC code table
    ``source_to_areas``      disciplines of one source title
    ``document_areas``       disciplines of every document (from its source)
    ``reference_areas``      disciplines of every document's references

Flows
    ``dual_map_flows``       citing area x cited area flow table + statistics
    ``dual_map_matrix``      pivot of the flow table
    ``dual_map_summary``     per-citing-area outflow, diversity and self share

Plotting (matplotlib)
    ``plot_dual_map``        two stacked bar columns with Bezier ribbons

All public functions are pure and deterministic. The bundled lookup tables
(``additional files/sources_data_short.xlsx`` and
``additional files/scopus subject area codes.xlsx``) are loaded lazily and
cached; unit tests may pass their own small tables instead.

Author: Lan Umek
Version: 2.19.0
"""
from __future__ import annotations

import functools
import math
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from biblium.indicators import _source_from_scopus_reference

__all__ = [
    "load_asjc_tables",
    "source_to_areas",
    "document_areas",
    "reference_areas",
    "dual_map_flows",
    "dual_map_matrix",
    "dual_map_summary",
    "plot_dual_map",
]

LEVELS = ("Field", "Area", "Science")

_ADD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "additional files")
DEFAULT_MAP_PATH = os.path.join(_ADD_DIR, "sources_data_short.xlsx")
DEFAULT_META_PATH = os.path.join(_ADD_DIR, "scopus subject area codes.xlsx")

_WS_RE = re.compile(r"\s+")
_AMP_RE = re.compile(r"\s*&\s*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_title(title: object) -> str:
    """Normalised lookup key for a source title: lower case, ``&`` -> ``and``,
    collapsed whitespace, no trailing punctuation."""
    if title is None or (isinstance(title, float) and math.isnan(title)):
        return ""
    s = str(title).strip().lower()
    s = _AMP_RE.sub(" and ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip(" .,;:")


def _parse_codes(cell: object) -> List[int]:
    """``"2210; 2211; 3104;"`` -> ``[2210, 2211, 3104]`` (4-digit ints, in order,
    without duplicates)."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    out: List[int] = []
    for tok in re.split(r"[;,\s]+", str(cell)):
        tok = tok.strip()
        if re.fullmatch(r"\d{4}", tok):
            c = int(tok)
            if c not in out:
                out.append(c)
    return out


def _check_level(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return level


def _meta_lookup(meta: pd.DataFrame, level: str) -> Dict[int, str]:
    """Dict code -> label at ``level`` (cached on the frame object by identity
    through a small module-level memo)."""
    key = (id(meta), level)
    hit = _META_MEMO.get(key)
    if hit is not None and hit[0] is meta:
        return hit[1]
    codes = pd.to_numeric(meta["code"], errors="coerce")
    d: Dict[int, str] = {}
    for c, lab in zip(codes, meta[level]):
        if pd.notna(c) and pd.notna(lab):
            d[int(c)] = str(lab)
    _META_MEMO[key] = (meta, d)
    return d


_META_MEMO: Dict[Tuple[int, str], Tuple[pd.DataFrame, Dict[int, str]]] = {}


def _map_lookup(asjc_map: pd.DataFrame) -> Dict[str, List[int]]:
    key = id(asjc_map)
    hit = _MAP_MEMO.get(key)
    if hit is not None and hit[0] is asjc_map:
        return hit[1]
    d: Dict[str, List[int]] = {}
    keys = asjc_map["key"] if "key" in asjc_map.columns else asjc_map["Source title"].map(_normalize_title)
    for k, codes in zip(keys, asjc_map["ASJC"]):
        if k and k not in d:
            d[k] = list(codes) if isinstance(codes, (list, tuple)) else _parse_codes(codes)
    _MAP_MEMO[key] = (asjc_map, d)
    return d


_MAP_MEMO: Dict[int, Tuple[pd.DataFrame, Dict[str, List[int]]]] = {}


# ---------------------------------------------------------------------------
# Loading bundled tables
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _load_asjc_tables_cached(map_path: str, meta_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(map_path)
    title_col, code_col = raw.columns[0], raw.columns[1]
    asjc_map = pd.DataFrame({
        "Source title": raw[title_col].astype(str).str.strip(),
        "ASJC": raw[code_col].map(_parse_codes),
    })
    asjc_map["key"] = asjc_map["Source title"].map(_normalize_title)
    asjc_map = asjc_map[(asjc_map["key"] != "") & (asjc_map["ASJC"].map(len) > 0)]
    asjc_map = asjc_map.drop_duplicates("key").reset_index(drop=True)

    xl = pd.ExcelFile(meta_path)
    sheet = "scopus codes" if "scopus codes" in xl.sheet_names else xl.sheet_names[0]
    meta = pd.read_excel(xl, sheet)
    meta = meta.rename(columns={c: str(c).strip() for c in meta.columns})
    lower = {c.lower(): c for c in meta.columns}
    meta = meta.rename(columns={lower.get("code", "code"): "code"})
    meta["code"] = pd.to_numeric(meta["code"], errors="coerce")
    meta = meta.dropna(subset=["code"]).copy()
    meta["code"] = meta["code"].astype(int)
    for col in ("Field", "Area", "Science"):
        if col not in meta.columns:
            raise ValueError(f"ASJC code table lacks column {col!r}")
        meta[col] = meta[col].astype(str).str.strip()
    meta = meta[["code", "Field", "Area", "Science"]].reset_index(drop=True)
    return asjc_map, meta


def load_asjc_tables(
    map_path: Optional[str] = None,
    meta_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the bundled source -> ASJC map and the ASJC code table.

    Parameters
    ----------
    map_path, meta_path : str, optional
        Paths to the Excel files; defaults to the files shipped in
        ``additional files``. Results are cached per path pair.

    Returns
    -------
    asjc_map : pd.DataFrame
        Columns ``Source title``, ``ASJC`` (list of 4-digit int codes) and
        ``key`` (normalised lower-case title used for lookups).
    meta : pd.DataFrame
        Columns ``code`` (int), ``Field``, ``Area``, ``Science``.
    """
    return _load_asjc_tables_cached(
        os.path.abspath(map_path or DEFAULT_MAP_PATH),
        os.path.abspath(meta_path or DEFAULT_META_PATH),
    )


def _tables(asjc_map: Optional[pd.DataFrame], meta: Optional[pd.DataFrame]):
    if asjc_map is None or meta is None:
        m, t = load_asjc_tables()
        asjc_map = m if asjc_map is None else asjc_map
        meta = t if meta is None else meta
    return asjc_map, meta


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

def _codes_to_areas(codes: Sequence[int], lookup: Dict[int, str]) -> List[str]:
    out: List[str] = []
    for c in codes:
        lab = lookup.get(int(c))
        if lab is not None and lab not in out:
            out.append(lab)
    return out


def source_to_areas(
    source_title: object,
    asjc_map: pd.DataFrame,
    meta: pd.DataFrame,
    level: str = "Area",
) -> List[str]:
    """
    Disciplines of one source title at the chosen level.

    Parameters
    ----------
    source_title : str
        Journal / source name (matched case-insensitively after
        normalisation).
    asjc_map, meta : pd.DataFrame
        Tables as returned by :func:`load_asjc_tables`.
    level : {"Field", "Area", "Science"}

    Returns
    -------
    list of str
        Distinct labels in the order of the source's ASJC codes; empty when
        the source is unknown.
    """
    _check_level(level)
    key = _normalize_title(source_title)
    if not key:
        return []
    codes = _map_lookup(asjc_map).get(key)
    if not codes:
        return []
    return _codes_to_areas(codes, _meta_lookup(meta, level))


def document_areas(
    df: pd.DataFrame,
    source_col: str = "Source title",
    level: str = "Area",
    asjc_map: Optional[pd.DataFrame] = None,
    meta: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Disciplines of every document, derived from its source title.

    Returns
    -------
    pd.Series
        Index as ``df``; each value a (possibly empty) list of distinct
        labels at ``level``.
    """
    _check_level(level)
    asjc_map, meta = _tables(asjc_map, meta)
    if source_col not in df.columns:
        raise ValueError(f"Column {source_col!r} not found")
    vals = [source_to_areas(v, asjc_map, meta, level) for v in df[source_col]]
    return pd.Series(vals, index=df.index, name=f"Citing {level}", dtype=object)


def reference_areas(
    df: pd.DataFrame,
    references_col: str = "References",
    sep: str = ";",
    level: str = "Area",
    asjc_map: Optional[pd.DataFrame] = None,
    meta: Optional[pd.DataFrame] = None,
    extractor: Optional[Callable[[str], Optional[str]]] = None,
) -> pd.Series:
    """
    Disciplines of every document's references.

    For each reference the journal name is extracted with ``extractor``
    (default: the Scopus heuristic ``biblium.indicators._source_from_scopus_reference``)
    and mapped to ASJC labels. One entry (a list of labels) is kept per
    reference whose journal is recognised, so a document's value is a list
    of lists and may contain duplicate journals.

    Returns
    -------
    pd.Series
        Index as ``df``; each value a list of label lists (one per mapped
        reference).
    """
    _check_level(level)
    asjc_map, meta = _tables(asjc_map, meta)
    if references_col not in df.columns:
        raise ValueError(f"Column {references_col!r} not found")
    ext = extractor or _source_from_scopus_reference
    vals: List[List[List[str]]] = []
    for blob in df[references_col]:
        per_doc: List[List[str]] = []
        if not pd.isna(blob):
            for ref in str(blob).split(sep):
                ref = ref.strip()
                if not ref:
                    continue
                src = ext(ref)
                if not src:
                    continue
                areas = source_to_areas(src, asjc_map, meta, level)
                if areas:
                    per_doc.append(areas)
        vals.append(per_doc)
    return pd.Series(vals, index=df.index, name=f"Cited {level}", dtype=object)


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------

def _count_refs(blob: object, sep: str) -> int:
    if pd.isna(blob):
        return 0
    return sum(1 for r in str(blob).split(sep) if r.strip())


def dual_map_flows(
    df: pd.DataFrame,
    level: str = "Area",
    fractional: bool = True,
    min_flow: float = 1,
    source_col: str = "Source title",
    references_col: str = "References",
    sep: str = ";",
    asjc_map: Optional[pd.DataFrame] = None,
    meta: Optional[pd.DataFrame] = None,
    extractor: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Flow table between citing disciplines (sources of the corpus) and cited
    disciplines (journals in the references).

    Every (document, recognised reference) pair is one citation. With
    ``fractional=True`` a document with *a* citing areas and a reference with
    *b* cited areas contributes ``1 / (a * b)`` to each of the ``a * b``
    (citing, cited) pairs, so each citation has total weight 1. With
    ``fractional=False`` every pair receives the full weight 1.

    Parameters
    ----------
    level : {"Field", "Area", "Science"}
    min_flow : float
        Pairs with a smaller total flow are dropped from the table (the
        statistics are computed before dropping).

    Returns
    -------
    flows : pd.DataFrame
        Columns ``Citing area``, ``Cited area``, ``Flow`` sorted by ``Flow``
        descending.
    stats : dict
        ``n_docs_total``, ``n_docs_mapped`` (documents whose source is
        mapped), ``n_refs_total``, ``n_refs_mapped`` (references of mapped
        documents whose journal is mapped), ``coverage_docs``,
        ``coverage_refs``, ``citing_totals`` / ``cited_totals`` (Series of
        outflow / inflow), ``self_discipline_share`` (share of flow with
        citing area == cited area), ``n_citing_areas``, ``n_cited_areas``,
        ``total_flow``, ``level``, ``fractional``.
    """
    _check_level(level)
    asjc_map, meta = _tables(asjc_map, meta)
    citing = document_areas(df, source_col, level, asjc_map, meta)
    cited = reference_areas(df, references_col, sep, level, asjc_map, meta, extractor)

    acc: Dict[Tuple[str, str], float] = {}
    n_docs_total = int(len(df))
    n_docs_mapped = 0
    n_refs_total = 0
    n_refs_mapped = 0
    for idx in df.index:
        n_refs_total += _count_refs(df.at[idx, references_col], sep)
        c_areas = citing.at[idx]
        if not c_areas:
            continue
        n_docs_mapped += 1
        for r_areas in cited.at[idx]:
            n_refs_mapped += 1
            w = 1.0 / (len(c_areas) * len(r_areas)) if fractional else 1.0
            for a in c_areas:
                for b in r_areas:
                    acc[(a, b)] = acc.get((a, b), 0.0) + w

    flows = pd.DataFrame(
        [(a, b, v) for (a, b), v in acc.items()],
        columns=["Citing area", "Cited area", "Flow"],
    )
    if flows.empty:
        flows = pd.DataFrame({"Citing area": pd.Series(dtype=object),
                              "Cited area": pd.Series(dtype=object),
                              "Flow": pd.Series(dtype=float)})
    flows = flows.sort_values(["Flow", "Citing area", "Cited area"],
                              ascending=[False, True, True]).reset_index(drop=True)

    total = float(flows["Flow"].sum())
    citing_totals = flows.groupby("Citing area")["Flow"].sum().sort_values(ascending=False)
    cited_totals = flows.groupby("Cited area")["Flow"].sum().sort_values(ascending=False)
    self_flow = float(flows.loc[flows["Citing area"] == flows["Cited area"], "Flow"].sum())
    stats: Dict[str, object] = {
        "level": level,
        "fractional": bool(fractional),
        "n_docs_total": n_docs_total,
        "n_docs_mapped": n_docs_mapped,
        "n_refs_total": n_refs_total,
        "n_refs_mapped": n_refs_mapped,
        "coverage_docs": n_docs_mapped / n_docs_total if n_docs_total else np.nan,
        "coverage_refs": n_refs_mapped / n_refs_total if n_refs_total else np.nan,
        "total_flow": total,
        "citing_totals": citing_totals,
        "cited_totals": cited_totals,
        "self_discipline_share": self_flow / total if total > 0 else np.nan,
        "n_citing_areas": int(citing_totals.size),
        "n_cited_areas": int(cited_totals.size),
    }
    if min_flow is not None and min_flow > 0:
        flows = flows[flows["Flow"] >= min_flow].reset_index(drop=True)
    return flows, stats


def dual_map_matrix(flows: pd.DataFrame) -> pd.DataFrame:
    """Pivot of the flow table: rows = citing areas, columns = cited areas,
    values = flow (0 where absent), both axes sorted by their totals."""
    if flows.empty:
        return pd.DataFrame(dtype=float)
    mat = flows.pivot_table(index="Citing area", columns="Cited area", values="Flow",
                            aggfunc="sum", fill_value=0.0)
    mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index,
                  mat.sum(axis=0).sort_values(ascending=False).index]
    mat.index.name, mat.columns.name = "Citing area", "Cited area"
    return mat.astype(float)


def dual_map_summary(flows: pd.DataFrame) -> pd.DataFrame:
    """
    Per citing area: ``Outflow`` (total flow), ``Share %`` of all flow,
    ``N cited areas``, ``Top cited area``, ``Top share`` (share of the
    outflow going to the top destination), ``Self share`` (share going to the
    same area) and ``Entropy`` (Shannon entropy, in bits, of the destination
    distribution).
    """
    cols = ["Citing area", "Outflow", "Share %", "N cited areas", "Top cited area",
            "Top share", "Self share", "Entropy"]
    if flows.empty:
        return pd.DataFrame(columns=cols)
    total = float(flows["Flow"].sum())
    rows = []
    for area, g in flows.groupby("Citing area", sort=False):
        g = g[g["Flow"] > 0].sort_values(["Flow", "Cited area"], ascending=[False, True])
        out = float(g["Flow"].sum())
        if out <= 0:
            continue
        p = g["Flow"].to_numpy(dtype=float) / out
        ent = float(-np.sum(p * np.log2(p)))
        self_flow = float(g.loc[g["Cited area"] == area, "Flow"].sum())
        rows.append({
            "Citing area": area,
            "Outflow": out,
            "Share %": 100.0 * out / total if total > 0 else np.nan,
            "N cited areas": int(g["Cited area"].nunique()),
            "Top cited area": str(g["Cited area"].iloc[0]),
            "Top share": float(p[0]),
            "Self share": self_flow / out,
            "Entropy": ent,
        })
    res = pd.DataFrame(rows, columns=cols)
    return res.sort_values(["Outflow", "Citing area"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
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


def _stack_positions(sizes: Sequence[float], total_height: float, gap: float) -> List[Tuple[float, float]]:
    """Vertical (bottom, height) of stacked bars proportional to ``sizes``
    with ``gap`` between them, filling ``total_height`` from the top down."""
    n = len(sizes)
    if n == 0:
        return []
    s = np.asarray(sizes, dtype=float)
    s = np.where(s > 0, s, 0.0)
    avail = max(total_height - gap * (n - 1), 1e-9)
    tot = s.sum()
    heights = (s / tot * avail) if tot > 0 else np.full(n, avail / n)
    heights = np.maximum(heights, 1e-6)
    pos = []
    y = total_height
    for h in heights:
        pos.append((y - h, h))
        y -= h + gap
    return pos


def _ribbon_path(x0: float, y0: float, h0: float, x1: float, y1: float, h1: float):
    """Closed Bezier ribbon from the band [y0, y0+h0] at x0 to [y1, y1+h1] at x1."""
    from matplotlib.path import Path
    xm = (x0 + x1) / 2.0
    verts = [
        (x0, y0), (xm, y0), (xm, y1), (x1, y1),            # lower edge
        (x1, y1 + h1),                                      # up the right side
        (xm, y1 + h1), (xm, y0 + h0), (x0, y0 + h0),        # upper edge back
        (x0, y0),                                           # close
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    return Path(verts, codes)


def plot_dual_map(
    flows: pd.DataFrame,
    stats: Optional[Dict[str, object]] = None,
    top_citing: int = 12,
    top_cited: int = 12,
    ax=None,
    figsize: Tuple[float, float] = (13, 8),
    cmap: str = "tab20",
    min_flow_share: float = 0.01,
    curve_alpha: float = 0.45,
    bar_width: float = 0.06,
    gap: float = 0.012,
    other_label: str = "Other",
    show_labels: bool = True,
    title: Optional[str] = "Dual-map overlay (citing disciplines -> cited disciplines)",
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    """
    Draw the dual-map overlay.

    Left column: citing areas stacked by outflow; right column: cited areas
    stacked by inflow; cubic Bezier ribbons between them with width
    proportional to the flow, coloured by the citing area. Areas beyond
    ``top_citing`` / ``top_cited`` are merged into ``other_label``; ribbons
    whose flow is below ``min_flow_share`` of the total are not drawn (but
    still count towards the bar sizes).

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.patches import PathPatch, Rectangle

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    fl = flows[["Citing area", "Cited area", "Flow"]].copy() if not flows.empty else flows
    if fl.empty or float(fl["Flow"].sum()) <= 0:
        ax.text(0.5, 0.5, "No mapped flows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _finish(fig, ax, title, f_name, dpi, show)

    fl = fl.groupby(["Citing area", "Cited area"], as_index=False)["Flow"].sum()
    total = float(fl["Flow"].sum())

    # collapse small areas into "Other"
    c_tot = fl.groupby("Citing area")["Flow"].sum().sort_values(ascending=False)
    d_tot = fl.groupby("Cited area")["Flow"].sum().sort_values(ascending=False)
    keep_c = list(c_tot.index[:max(1, int(top_citing))])
    keep_d = list(d_tot.index[:max(1, int(top_cited))])
    fl["Citing area"] = fl["Citing area"].where(fl["Citing area"].isin(keep_c), other_label)
    fl["Cited area"] = fl["Cited area"].where(fl["Cited area"].isin(keep_d), other_label)
    fl = fl.groupby(["Citing area", "Cited area"], as_index=False)["Flow"].sum()
    c_tot = fl.groupby("Citing area")["Flow"].sum()
    d_tot = fl.groupby("Cited area")["Flow"].sum()
    # order: by size, "Other" last
    c_order = [a for a in c_tot.sort_values(ascending=False).index if a != other_label]
    if other_label in c_tot.index:
        c_order.append(other_label)
    d_order = [a for a in d_tot.sort_values(ascending=False).index if a != other_label]
    if other_label in d_tot.index:
        d_order.append(other_label)

    x_left, x_right = 0.12, 0.88
    c_pos = dict(zip(c_order, _stack_positions([c_tot[a] for a in c_order], 1.0, gap)))
    d_pos = dict(zip(d_order, _stack_positions([d_tot[a] for a in d_order], 1.0, gap)))

    cm = mpl.colormaps.get_cmap(cmap) if hasattr(mpl, "colormaps") else plt.get_cmap(cmap)
    n_colors = getattr(cm, "N", 20)
    colors = {a: (cm(i % n_colors) if n_colors <= 32 else cm(i / max(len(c_order) - 1, 1)))
              for i, a in enumerate(c_order)}
    if other_label in colors:
        colors[other_label] = (0.6, 0.6, 0.6, 1.0)

    # bars
    for a in c_order:
        y, h = c_pos[a]
        ax.add_patch(Rectangle((x_left - bar_width, y), bar_width, h, facecolor=colors[a],
                               edgecolor="white", lw=0.5))
        if show_labels:
            ax.text(x_left - bar_width - 0.01, y + h / 2,
                    f"{a} ({100 * c_tot[a] / total:.1f}%)", ha="right", va="center", fontsize=8)
    for b in d_order:
        y, h = d_pos[b]
        ax.add_patch(Rectangle((x_right, y), bar_width, h, facecolor=(0.45, 0.45, 0.45, 1.0),
                               edgecolor="white", lw=0.5))
        if show_labels:
            ax.text(x_right + bar_width + 0.01, y + h / 2,
                    f"{b} ({100 * d_tot[b] / total:.1f}%)", ha="left", va="center", fontsize=8)

    # ribbons: consume each bar from the top, ordered by destination/source stack
    c_cursor = {a: c_pos[a][0] + c_pos[a][1] for a in c_order}  # top of each bar
    d_cursor = {b: d_pos[b][0] + d_pos[b][1] for b in d_order}
    d_rank = {b: i for i, b in enumerate(d_order)}
    c_rank = {a: i for i, a in enumerate(c_order)}
    fl["_cr"] = fl["Citing area"].map(c_rank)
    fl["_dr"] = fl["Cited area"].map(d_rank)
    # allocate band positions in a consistent order on both sides
    bands = []
    for _, r in fl.sort_values(["_cr", "_dr"]).iterrows():
        a, b, v = r["Citing area"], r["Cited area"], float(r["Flow"])
        hc = c_pos[a][1] * v / c_tot[a]
        c_cursor[a] -= hc
        bands.append([a, b, v, c_cursor[a], hc, None, None])
    bands_by_dest = sorted(range(len(bands)), key=lambda i: (d_rank[bands[i][1]], c_rank[bands[i][0]]))
    for i in bands_by_dest:
        a, b, v = bands[i][0], bands[i][1], bands[i][2]
        hd = d_pos[b][1] * v / d_tot[b]
        d_cursor[b] -= hd
        bands[i][5], bands[i][6] = d_cursor[b], hd
    for a, b, v, yc, hc, yd, hd in sorted(bands, key=lambda t: -t[2]):
        if v / total < min_flow_share:
            continue
        path = _ribbon_path(x_left, yc, hc, x_right, yd, hd)
        ax.add_patch(PathPatch(path, facecolor=colors[a], edgecolor="none", alpha=curve_alpha))

    ax.set_xlim(-0.25 if show_labels else 0.0, 1.25 if show_labels else 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_axis_off()
    ax.text(x_left - bar_width / 2, 1.03, "Citing disciplines", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    ax.text(x_right + bar_width / 2, 1.03, "Cited disciplines", ha="center", va="bottom",
            fontsize=10, fontweight="bold")

    if title and stats:
        cd, cr = stats.get("coverage_docs"), stats.get("coverage_refs")
        parts = []
        if cd is not None and not (isinstance(cd, float) and math.isnan(cd)):
            parts.append(f"docs mapped {100 * cd:.0f}%")
        if cr is not None and not (isinstance(cr, float) and math.isnan(cr)):
            parts.append(f"refs mapped {100 * cr:.0f}%")
        ss = stats.get("self_discipline_share")
        if ss is not None and not (isinstance(ss, float) and math.isnan(ss)):
            parts.append(f"within-discipline {100 * ss:.0f}%")
        if parts:
            title = f"{title}\n(" + ", ".join(parts) + ")"
    if title:
        ax.set_title(title, pad=22)  # room for the two column headings
        title = None
    return _finish(fig, ax, title, f_name, dpi, show)
