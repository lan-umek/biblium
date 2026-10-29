"""
editions.py — analiza večizdajnih citiranih del (2.18, novo)
============================================================
Orodja za "biografijo knjige": atribucija citirane IZDAJE iz referenčnih
nizov, edition migration (prehajanje citiranja med izdajami skozi čas) in
priprava edicijske matrike skupin za BiblioGroupAnalysis (fingerprints).

Tipična raba (Pollitt & Bouckaert, Public Management Reform):

    from biblium.editions import (
        attribute_cited_editions, edition_migration_table,
        plot_edition_migration, edition_group_matrix,
    )

    att = attribute_cited_editions(
        df, refs_col="References",
        title_regex=r"public\\s+management\\s+reform",
        author_regex=r"pollitt",
        edition_windows={1: (1999, 2003), 2: (2004, 2010),
                         3: (2011, 2016), 4: (2017, 2026)},
    )
    mig = edition_migration_table(att, df, year_col="Year")
    plot_edition_migration(mig, edition_years={1: 2000, 2: 2004,
                                               3: 2011, 4: 2017},
                           filename_base="results/26_editions/migration")

Opombe o metodi
---------------
- Referenčni nizi (Scopus `References`) so ločeni s ";". Segment šteje za
  referenco na ciljno delo, če ustreza `title_regex` ALI (`author_regex`
  in vsa dodatna imena v `coauthor_regexes`).
- Iz segmenta se izluščijo vsa leta 19xx/20xx znotraj `plausible_years`;
  vsako leto se prek `edition_windows` preslika v izdajo. Leta med
  uradnimi izdajami (ponatisi, napačna leta) padejo v okno zadnje
  predhodne izdaje — to je treba v člankih transparentno navesti.
- Dokument lahko citira več izdaj; privzeto se uporabi FRAKCIJSKO
  uteževanje (1/k na izdajo), `fractional=False` pa šteje vsako izdajo
  s težo 1.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "attribute_cited_editions",
    "edition_migration_table",
    "plot_edition_migration",
    "edition_group_matrix",
]


def _split_refs(refs: str) -> List[str]:
    return [r.strip() for r in str(refs).split(";") if r.strip()]


def _extract_years(segment: str, lo: int, hi: int) -> List[int]:
    out = []
    for m in re.finditer(r"\b(19\d{2}|20[0-4]\d)\b", segment):
        y = int(m.group(0))
        if lo <= y <= hi:
            out.append(y)
    return out


def attribute_cited_editions(
    df: pd.DataFrame,
    refs_col: str = "References",
    *,
    title_regex: str,
    author_regex: Optional[str] = None,
    coauthor_regexes: Optional[Sequence[str]] = None,
    edition_windows: Dict[int, Tuple[int, int]],
    plausible_years: Tuple[int, int] = (1990, 2026),
    fractional: bool = True,
) -> pd.DataFrame:
    """
    Za vsak dokument v ``df`` poišče reference na ciljno (večizdajno) delo
    in pripiše citirane izdaje.

    Returns
    -------
    DataFrame, indeksiran kot ``df``, s stolpci:
        n_target_refs   — št. referenčnih segmentov na ciljno delo
        cited_years     — najdena leta (niz "2000,2011")
        editions        — pripisane izdaje (niz "1,3")
        n_editions      — št. različnih izdaj
        w_edition_<k>   — frakcijska (ali polna) utež za izdajo k
        attributable    — bool
    """
    title_rx = re.compile(title_regex, re.IGNORECASE)
    author_rx = re.compile(author_regex, re.IGNORECASE) if author_regex else None
    co_rxs = [re.compile(rx, re.IGNORECASE) for rx in (coauthor_regexes or [])]

    def year_to_ed(y: int) -> Optional[int]:
        for ed, (lo, hi) in edition_windows.items():
            if lo <= y <= hi:
                return ed
        return None

    ed_keys = sorted(edition_windows)
    rows = []
    for idx, refs in df[refs_col].items():
        segs = []
        for s in _split_refs(refs):
            if title_rx.search(s):
                segs.append(s)
            elif author_rx and author_rx.search(s) and all(
                rx.search(s) for rx in co_rxs
            ):
                segs.append(s)
        cited_years: set = set()
        for s in segs:
            for y in _extract_years(s, *plausible_years):
                if year_to_ed(y) is not None:
                    cited_years.add(y)
        eds = sorted({year_to_ed(y) for y in cited_years})
        eds = [e for e in eds if e is not None]
        row = {
            "n_target_refs": len(segs),
            "cited_years": ",".join(map(str, sorted(cited_years))),
            "editions": ",".join(map(str, eds)),
            "n_editions": len(eds),
            "attributable": len(eds) > 0,
        }
        w = (1.0 / len(eds)) if (fractional and eds) else 1.0
        for k in ed_keys:
            row[f"w_edition_{k}"] = w if k in eds else 0.0
        rows.append(row)

    return pd.DataFrame(rows, index=df.index)


def edition_migration_table(
    attribution: pd.DataFrame,
    df: pd.DataFrame,
    year_col: str = "Year",
    *,
    min_year_n: int = 5,
) -> pd.DataFrame:
    """
    Leto citiranja × izdaja: frakcijske vsote in deleži.

    Returns
    -------
    DataFrame z indeksom = leto citiranja in stolpci
    ``n_<k>`` (frakcijski count) ter ``share_<k>`` (delež med
    pripisljivimi) za vsako izdajo k, plus ``n_attributable``.
    """
    w_cols = [c for c in attribution.columns if c.startswith("w_edition_")]
    yrs = pd.to_numeric(df[year_col], errors="coerce")
    tmp = attribution[w_cols].copy()
    tmp["_year"] = yrs.values
    tmp = tmp.dropna(subset=["_year"])
    tmp["_year"] = tmp["_year"].astype(int)
    g = tmp.groupby("_year")[w_cols].sum()
    g.index.name = "citing_year"
    tot = g.sum(axis=1)
    out = pd.DataFrame(index=g.index)
    for c in w_cols:
        k = c.replace("w_edition_", "")
        out[f"n_{k}"] = g[c]
        out[f"share_{k}"] = np.where(tot > 0, g[c] / tot, np.nan)
    out["n_attributable"] = tot
    return out[out["n_attributable"] >= min_year_n]


def plot_edition_migration(
    migration: pd.DataFrame,
    *,
    edition_years: Optional[Dict[int, int]] = None,
    edition_labels: Optional[Dict[int, str]] = None,
    filename_base: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5.5),
    cmap_name: str = "viridis",
    dpi: int = 200,
    title: str = "Edition migration — share of cited editions by citing year",
):
    """
    Stacked-area graf deležev citiranih izdaj po letu citiranja.
    Navpičnice označijo leta izdaj (``edition_years``).
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    share_cols = [c for c in migration.columns if c.startswith("share_")]
    ks = [c.replace("share_", "") for c in share_cols]
    labels = [
        (edition_labels or {}).get(int(k), f"Edition {k}") for k in ks
    ]
    cmap = colormaps.get_cmap(cmap_name)
    colors = [cmap(i / max(len(ks) - 1, 1)) for i in range(len(ks))]

    x = migration.index.to_numpy()
    Y = migration[share_cols].fillna(0).to_numpy().T * 100

    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(x, Y, labels=labels, colors=colors, alpha=0.85)
    if edition_years:
        for ed, y in edition_years.items():
            if x.min() <= y <= x.max():
                ax.axvline(y, color="white", lw=1.2, ls="--", alpha=0.9)
                ax.text(y, 101, str(y), ha="center", va="bottom", fontsize=8)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Citing year")
    ax.set_ylabel("Share of cited editions (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    if filename_base:
        fig.savefig(f"{filename_base}.png", dpi=dpi, bbox_inches="tight")
        try:
            fig.savefig(f"{filename_base}.svg", bbox_inches="tight")
        except Exception:
            pass
    return fig


def edition_group_matrix(
    attribution: pd.DataFrame,
    *,
    edition_labels: Optional[Dict[int, str]] = None,
    binary: bool = True,
) -> pd.DataFrame:
    """
    Iz atribucije zgradi matriko dokument × izdaja (0/1 ali frakcijske
    uteži) — neposredno uporabno kot ``group_matrix`` za edicijske
    "fingerprints" (crosstabi z državami, revijami, koncepti ...).
    """
    w_cols = [c for c in attribution.columns if c.startswith("w_edition_")]
    out = attribution[w_cols].copy()
    if binary:
        out = (out > 0).astype(int)
    out.columns = [
        (edition_labels or {}).get(
            int(c.replace("w_edition_", "")),
            f"Edition {c.replace('w_edition_', '')}",
        )
        for c in w_cols
    ]
    return out
