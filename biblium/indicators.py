# -*- coding: utf-8 -*-
"""
biblium.indicators
==================

Novi bibliometrični indikatorji (novo v 2.19). Modul je namenoma brez
specializiranih odvisnosti (numpy, pandas, scipy, matplotlib) in vse
funkcije so deterministične (vsak naključni proces sprejme ``seed``).

Vsebina
-------
Citiranost
    ``characteristic_scores_and_scales``  CSS razredi (Glänzel & Schubert 1988)
    ``percentile_indicators``             PP(top 10 %), PP(top 1 %), MNCS po entitetah
    ``uncitedness``                       delež necitiranih po letu / skupini
    ``bootstrap_ci``                      bootstrap intervali zaupanja za indikatorje

Reference
    ``reference_ages``                    starosti referenc na dokument
    ``reference_age_profile``             Priceov indeks, mediana starosti, citing half-life
    ``cited_sources``                     "knowledge base": citirane revije v referencah

Struktura produkcije
    ``lorenz_curve``, ``gini_coefficient``, ``theil_index``, ``concentration``
    ``author_careers``                    akademska starost, m-kvocient, lokalni h
    ``fit_heaps_law``                     rast besedišča (Heaps 1978)

Risanje (matplotlib)
    ``plot_lorenz_curve``, ``plot_css_classes``, ``plot_reference_age_profile``,
    ``plot_heaps_law``, ``plot_uncitedness``

Author: Lan Umek
Version: 2.19.0
"""
from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "characteristic_scores_and_scales",
    "percentile_indicators",
    "uncitedness",
    "bootstrap_ci",
    "reference_ages",
    "reference_age_profile",
    "cited_sources",
    "lorenz_curve",
    "gini_coefficient",
    "theil_index",
    "concentration",
    "author_careers",
    "fit_heaps_law",
    "plot_lorenz_curve",
    "plot_css_classes",
    "plot_reference_age_profile",
    "plot_heaps_law",
    "plot_uncitedness",
]

CSS_LABELS = ("Poorly cited", "Fairly cited", "Remarkably cited", "Outstandingly cited")


# ---------------------------------------------------------------------------
# Pomožne funkcije
# ---------------------------------------------------------------------------

def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _explode_items(
    df: pd.DataFrame,
    col: str,
    sep: str = ";",
    strip: bool = True,
) -> pd.DataFrame:
    """Dolga tabela (indeks dokumenta, Item) iz stolpca z ločenimi vrednostmi.

    Prazni in ``nan`` žetoni so odstranjeni; podvojeni žetoni znotraj istega
    dokumenta se štejejo enkrat.
    """
    s = df[col].dropna().astype(str).str.split(sep)
    long = s.explode()
    if strip:
        long = long.str.strip()
    long = long[(long != "") & (long.str.lower() != "nan")]
    out = long.rename("Item").reset_index().rename(columns={"index": "_doc"})
    out = out.drop_duplicates(["_doc", "Item"])
    return out


def _h_index(citations: Iterable[float]) -> int:
    c = sorted((int(x) for x in citations if pd.notna(x)), reverse=True)
    return sum(1 for i, v in enumerate(c) if v >= i + 1)


# ---------------------------------------------------------------------------
# 1) Characteristic Scores and Scales (Glänzel & Schubert 1988)
# ---------------------------------------------------------------------------

def _css_thresholds(values: np.ndarray, n_classes: int = 4) -> List[float]:
    """Iterativni pragovi: b1 = mean(vse), b2 = mean(x > b1), b3 = mean(x > b2) …"""
    thresholds: List[float] = []
    cur = np.asarray(values, dtype=float)
    cur = cur[~np.isnan(cur)]
    for _ in range(n_classes - 1):
        if cur.size == 0:
            break
        m = float(cur.mean())
        thresholds.append(m)
        cur = cur[cur > m]
    return thresholds


def _css_assign(values: np.ndarray, thresholds: Sequence[float], labels: Sequence[str]) -> np.ndarray:
    out = np.full(values.shape, labels[0], dtype=object)
    for i, t in enumerate(thresholds):
        out[values > t] = labels[i + 1]
    out[np.isnan(values)] = None
    return out


def characteristic_scores_and_scales(
    df: pd.DataFrame,
    citations_col: str = "Cited by",
    year_col: Optional[str] = None,
    n_classes: int = 4,
    labels: Sequence[str] = CSS_LABELS,
    min_year_size: int = 10,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Characteristic Scores and Scales (Glänzel & Schubert 1988).

    Pragovi: β1 = povprečje vseh citatov; β2 = povprečje dokumentov nad β1;
    β3 = povprečje dokumentov nad β2. Razredi: ≤β1 "poorly cited",
    (β1, β2] "fairly cited", (β2, β3] "remarkably cited", >β3
    "outstandingly cited". Metoda je brezparametrična in neodvisna od
    porazdelitve, zato je primerna za primerjavo področij in obdobij.

    Parameters
    ----------
    year_col : str, optional
        Če je podan, se pragovi računajo ločeno za vsako leto (leta z manj
        kot ``min_year_size`` dokumenti uporabijo globalne pragove).

    Returns
    -------
    classes : pd.Series
        Razred za vsak dokument (indeks kot ``df``).
    summary : pd.DataFrame
        Stolpci: Scope (leto ali "all"), b1..b{k-1}, N, in število ter delež
        dokumentov po razredih.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    labels = list(labels)[:n_classes]
    if len(labels) < n_classes:
        labels += [f"Class {i + 1}" for i in range(len(labels), n_classes)]

    cit = _to_numeric(df[citations_col]).to_numpy(dtype=float)
    glob_thr = _css_thresholds(cit, n_classes)
    classes = pd.Series(_css_assign(cit, glob_thr, labels), index=df.index, name="CSS Class")

    def _row(scope, thr, vals, cls):
        rec = {"Scope": scope, "N": int(np.sum(~np.isnan(vals)))}
        for i in range(n_classes - 1):
            rec[f"b{i + 1}"] = float(thr[i]) if i < len(thr) else np.nan
        for lab in labels:
            k = int(np.sum(cls == lab))
            rec[f"N {lab}"] = k
            rec[f"% {lab}"] = 100.0 * k / rec["N"] if rec["N"] else np.nan
        return rec

    rows = [_row("all", glob_thr, cit, classes.to_numpy())]

    if year_col is not None and year_col in df.columns:
        years = _to_numeric(df[year_col])
        for y, idx in years.dropna().astype(int).groupby(years.dropna().astype(int)).groups.items():
            vals = _to_numeric(df.loc[idx, citations_col]).to_numpy(dtype=float)
            if np.sum(~np.isnan(vals)) >= min_year_size:
                thr = _css_thresholds(vals, n_classes)
            else:
                thr = glob_thr
            cls = _css_assign(vals, thr, labels)
            classes.loc[idx] = cls
            rows.append(_row(int(y), thr, vals, cls))

    summary = pd.DataFrame(rows)
    classes = classes.astype("category") if classes.notna().any() else classes
    return classes, summary


# ---------------------------------------------------------------------------
# 2) Percentilni indikatorji po entitetah (Leiden-style)
# ---------------------------------------------------------------------------

def _year_normalised(
    df: pd.DataFrame,
    citations_col: str,
    year_col: Optional[str],
) -> pd.DataFrame:
    """Doda stolpce ``_cit``, ``_ncs`` (cit / povp. cit v letu) in ``_pct``
    (percentilni rang znotraj leta, 0-100, višje = bolj citirano)."""
    out = pd.DataFrame(index=df.index)
    out["_cit"] = _to_numeric(df[citations_col]).fillna(0.0)
    if year_col is not None and year_col in df.columns:
        grp = _to_numeric(df[year_col]).fillna(-1).astype(int)
    else:
        grp = pd.Series(0, index=df.index)
    out["_year"] = grp
    mean_by = out.groupby("_year")["_cit"].transform("mean")
    out["_ncs"] = np.where(mean_by > 0, out["_cit"] / mean_by.replace(0, np.nan), np.nan)
    # percentilni rang (average ties), 0..100
    out["_pct"] = out.groupby("_year")["_cit"].rank(pct=True, method="average") * 100.0
    return out


def percentile_indicators(
    df: pd.DataFrame,
    entity_col: str,
    citations_col: str = "Cited by",
    year_col: Optional[str] = "Year",
    sep: str = ";",
    min_docs: int = 5,
    top_n: Optional[int] = 50,
    thresholds: Sequence[float] = (10.0, 1.0),
    sort_by: str = "PP(top 10%)",
) -> pd.DataFrame:
    """
    Percentilni in normalizirani indikatorji citiranosti po entitetah
    (avtorji, revije, države, ustanove …), v slogu Leidenske lestvice.

    Za vsak dokument se znotraj njegovega leta objave izračuna percentilni
    rang citiranosti (povprečni rang pri vezeh, v %); dokument je v
    "top X %", če je njegov rang strogo večji od (100 − X). Pri 10
    dokumentih brez vezi je tako v top 10 % natanko najbolj citirani.
    Za entiteto (polno štetje: vsak dokument šteje vsaki entiteti v celoti):

    - ``P``            število dokumentov,
    - ``TCS``          skupno število citatov,
    - ``MCS``          povprečje citatov,
    - ``MNCS``         povprečje letno normaliziranih citatov (cit / povp. leta),
    - ``PP(top X%)``   delež dokumentov v zgornjih X % leta (v %),
    - ``P(top X%)``    število teh dokumentov,
    - ``h``            h-indeks entitete.

    Parameters
    ----------
    thresholds : sequence of float
        Percentilni pragovi (privzeto 10 in 1 → PP(top 10%), PP(top 1%)).
    min_docs : int
        Entitete z manj dokumenti so izpuščene.
    """
    if entity_col not in df.columns:
        raise ValueError(f"Column {entity_col!r} not found")
    norm = _year_normalised(df, citations_col, year_col)
    long = _explode_items(df, entity_col, sep=sep)
    long = long.join(norm, on="_doc")

    for t in thresholds:
        long[f"_top{t:g}"] = (long["_pct"] > (100.0 - t)).astype(int)

    agg = {
        "P": ("_cit", "size"),
        "TCS": ("_cit", "sum"),
        "MCS": ("_cit", "mean"),
        "MNCS": ("_ncs", "mean"),
        "h": ("_cit", _h_index),
    }
    for t in thresholds:
        agg[f"P(top {t:g}%)"] = (f"_top{t:g}", "sum")
        agg[f"PP(top {t:g}%)"] = (f"_top{t:g}", lambda s: 100.0 * s.mean())
    res = long.groupby("Item").agg(**agg)
    res = res[res["P"] >= min_docs]
    res["TCS"] = res["TCS"].astype(int)
    for t in thresholds:
        res[f"P(top {t:g}%)"] = res[f"P(top {t:g}%)"].astype(int)
    if sort_by not in res.columns:
        sort_by = "P"
    res = res.sort_values([sort_by, "P"], ascending=[False, False])
    if top_n:
        res = res.head(top_n)
    res.index.name = entity_col
    return res.reset_index()


# ---------------------------------------------------------------------------
# 3) Necitiranost
# ---------------------------------------------------------------------------

def uncitedness(
    df: pd.DataFrame,
    citations_col: str = "Cited by",
    year_col: str = "Year",
    group_col: Optional[str] = None,
    sep: str = ";",
    current_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Delež necitiranih dokumentov po letu (in po želji po skupini, npr.
    "Document Type" ali "Source title").

    Vrne stolpce: [group], Year, Age, N, Uncited, Uncited %, Cumulative Uncited %
    (kumulativni delež od najstarejšega leta naprej, kar prikaže "starostno
    pričakovanje": starejši dokumenti so praviloma redkeje necitirani).
    """
    years = _to_numeric(df[year_col])
    cit = _to_numeric(df[citations_col])
    if current_year is None:
        current_year = int(years.max()) if years.notna().any() else pd.Timestamp.today().year
    base = pd.DataFrame({"Year": years, "_uncited": (cit.fillna(0) <= 0).astype(int)}, index=df.index)
    base = base.dropna(subset=["Year"])
    base["Year"] = base["Year"].astype(int)

    if group_col is not None and group_col in df.columns:
        long = _explode_items(df, group_col, sep=sep)
        base = long.join(base, on="_doc", how="inner").rename(columns={"Item": group_col})
        keys = [group_col, "Year"]
    else:
        keys = ["Year"]

    out = base.groupby(keys).agg(N=("_uncited", "size"), Uncited=("_uncited", "sum")).reset_index()
    out["Age"] = current_year - out["Year"]
    out["Uncited %"] = 100.0 * out["Uncited"] / out["N"]
    sort_keys = keys
    out = out.sort_values(sort_keys).reset_index(drop=True)
    grp = out.groupby(group_col) if group_col in keys else None
    if grp is not None:
        out["Cumulative Uncited %"] = 100.0 * grp["Uncited"].cumsum() / grp["N"].cumsum()
    else:
        out["Cumulative Uncited %"] = 100.0 * out["Uncited"].cumsum() / out["N"].cumsum()
    cols = keys + ["Age", "N", "Uncited", "Uncited %", "Cumulative Uncited %"]
    return out[cols]


# ---------------------------------------------------------------------------
# 4) Bootstrap intervali zaupanja
# ---------------------------------------------------------------------------

_INDICATORS: Dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda x: float(np.mean(x)),
    "median": lambda x: float(np.median(x)),
    "h_index": lambda x: float(_h_index(x)),
    "total": lambda x: float(np.sum(x)),
    "uncited_share": lambda x: float(np.mean(x <= 0)),
}


def bootstrap_ci(
    values: Union[Sequence[float], np.ndarray, pd.Series],
    statistic: Union[str, Callable[[np.ndarray], float]] = "mean",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = 42,
    method: str = "percentile",
) -> Dict[str, float]:
    """
    Bootstrap interval zaupanja za poljuben indikator na vektorju vrednosti
    (npr. citati dokumentov ene revije / avtorja / obdobja).

    Parameters
    ----------
    statistic : str or callable
        "mean", "median", "h_index", "total", "uncited_share" ali funkcija
        ``f(np.ndarray) -> float``.
    method : {"percentile", "basic"}
        Percentilni ali osnovni (pivot) interval.

    Returns
    -------
    dict with keys: statistic, estimate, ci_low, ci_high, se, n, n_boot, ci
    """
    x = np.asarray(pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna(), dtype=float)
    func = _INDICATORS[statistic] if isinstance(statistic, str) else statistic
    name = statistic if isinstance(statistic, str) else getattr(statistic, "__name__", "custom")
    n = x.size
    if n == 0:
        return {"statistic": name, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "se": np.nan, "n": 0, "n_boot": n_boot, "ci": ci}
    est = func(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([func(x[row]) for row in idx], dtype=float)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    if method == "basic":
        lo, hi = 2 * est - hi, 2 * est - lo
    elif method != "percentile":
        raise ValueError("method must be 'percentile' or 'basic'")
    return {"statistic": name, "estimate": float(est), "ci_low": float(lo), "ci_high": float(hi),
            "se": float(boots.std(ddof=1)) if n_boot > 1 else np.nan, "n": int(n),
            "n_boot": int(n_boot), "ci": ci}


def bootstrap_ci_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "Cited by",
    statistic: Union[str, Callable] = "mean",
    sep: str = ";",
    min_docs: int = 5,
    top_n: Optional[int] = 20,
    **kwargs,
) -> pd.DataFrame:
    """Bootstrap CI za indikator po skupinah (entitete z ločilom ali leta)."""
    long = _explode_items(df, group_col, sep=sep)
    long["_v"] = _to_numeric(df[value_col]).reindex(long["_doc"]).to_numpy()
    rows = []
    for item, sub in long.groupby("Item"):
        v = sub["_v"].dropna()
        if len(v) < min_docs:
            continue
        r = bootstrap_ci(v, statistic=statistic, **kwargs)
        r[group_col] = item
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("estimate", ascending=False)
    if top_n:
        out = out.head(top_n)
    cols = [group_col, "n", "estimate", "ci_low", "ci_high", "se", "statistic"]
    return out[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5) Starost referenc, Priceov indeks, citing half-life
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\((1[6-9]\d{2}|20\d{2})\)")
_YEAR_RE_LOOSE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")


def _ref_years(ref_blob: str, sep: str, loose: bool) -> List[int]:
    years = []
    for ref in str(ref_blob).split(sep):
        ref = ref.strip()
        if not ref:
            continue
        m = _YEAR_RE.findall(ref)
        if not m and loose:
            m = _YEAR_RE_LOOSE.findall(ref)
        if m:
            years.append(int(m[-1]))  # Scopus: leto je v oklepaju na koncu
    return years


def reference_ages(
    df: pd.DataFrame,
    references_col: str = "References",
    year_col: str = "Year",
    sep: str = ";",
    price_window: int = 5,
    loose_year_match: bool = True,
    max_age: int = 200,
) -> pd.DataFrame:
    """
    Starosti referenc za vsak dokument: leto objave − leto reference.

    Vrne DataFrame (indeks kot ``df``) s stolpci:
    ``N refs``, ``N refs with year``, ``Mean ref age``, ``Median ref age``,
    ``Price index`` (delež referenc starih ≤ ``price_window`` let, v %),
    ``Max ref age``.
    Negativne starosti (reference "iz prihodnosti", npr. in-press) se
    postavijo na 0; starosti > ``max_age`` se izpustijo (napake pri parsanju).
    """
    years = _to_numeric(df[year_col])
    rows = []
    for idx, (ref_blob, y) in enumerate(zip(df[references_col], years)):
        rec = {"N refs": 0, "N refs with year": 0, "Mean ref age": np.nan,
               "Median ref age": np.nan, "Price index": np.nan, "Max ref age": np.nan}
        if pd.isna(ref_blob) or pd.isna(y):
            rows.append(rec)
            continue
        refs = [r for r in str(ref_blob).split(sep) if r.strip()]
        rec["N refs"] = len(refs)
        ry = _ref_years(ref_blob, sep, loose_year_match)
        ages = np.array([int(y) - r for r in ry], dtype=float)
        ages = ages[ages <= max_age]
        ages[ages < 0] = 0
        if ages.size:
            rec["N refs with year"] = int(ages.size)
            rec["Mean ref age"] = float(ages.mean())
            rec["Median ref age"] = float(np.median(ages))
            rec["Price index"] = 100.0 * float(np.mean(ages <= price_window))
            rec["Max ref age"] = float(ages.max())
        rows.append(rec)
    return pd.DataFrame(rows, index=df.index)


def reference_age_profile(
    df: pd.DataFrame,
    references_col: str = "References",
    year_col: str = "Year",
    sep: str = ";",
    price_window: int = 5,
    loose_year_match: bool = True,
    max_age: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Agregirani profil starosti referenc po letu objave in za celoto.

    Returns
    -------
    by_year : pd.DataFrame
        Year, N docs, N refs, Mean ref age, Median ref age, Price index (%),
        Citing half-life (mediana starosti vseh referenc dokumentov tega leta).
    overall : dict
        n_docs, n_refs, mean_ref_age, median_ref_age, price_index,
        citing_half_life (vse reference skupaj, pool).
    """
    years = _to_numeric(df[year_col])
    pooled: Dict[int, List[float]] = {}
    per_doc = reference_ages(df, references_col, year_col, sep, price_window, loose_year_match, max_age)
    for ref_blob, y in zip(df[references_col], years):
        if pd.isna(ref_blob) or pd.isna(y):
            continue
        ry = _ref_years(ref_blob, sep, loose_year_match)
        ages = np.array([int(y) - r for r in ry], dtype=float)
        ages = ages[ages <= max_age]
        ages[ages < 0] = 0
        pooled.setdefault(int(y), []).extend(ages.tolist())

    rows = []
    for y in sorted(pooled):
        a = np.asarray(pooled[y])
        n_docs = int(((years == y) & per_doc["N refs with year"].gt(0)).sum())
        if a.size == 0:
            continue
        rows.append({
            "Year": y, "N docs": n_docs, "N refs": int(a.size),
            "Mean ref age": float(a.mean()), "Median ref age": float(np.median(a)),
            "Price index": 100.0 * float(np.mean(a <= price_window)),
            "Citing half-life": float(np.median(a)),
        })
    by_year = pd.DataFrame(rows)
    all_ages = np.asarray([v for vals in pooled.values() for v in vals], dtype=float)
    overall = {
        "n_docs": int(per_doc["N refs with year"].gt(0).sum()),
        "n_refs": int(all_ages.size),
        "mean_ref_age": float(all_ages.mean()) if all_ages.size else np.nan,
        "median_ref_age": float(np.median(all_ages)) if all_ages.size else np.nan,
        "price_index": 100.0 * float(np.mean(all_ages <= price_window)) if all_ages.size else np.nan,
        "citing_half_life": float(np.median(all_ages)) if all_ages.size else np.nan,
        "price_window": price_window,
    }
    return by_year, overall


# ---------------------------------------------------------------------------
# 6) Citirane revije v referencah ("knowledge base")
# ---------------------------------------------------------------------------

_PAGES_RE = re.compile(r"^(pp?\.\s*)?\d{1,6}\s*[-–]\s*\d{1,6}$")
_NUM_RE = re.compile(r"^\d{1,5}$")
_ISSUE_RE = re.compile(r"^\d{1,4}\s*\(\d{1,4}\)$")


def _source_from_scopus_reference(ref: str) -> Optional[str]:
    """Hevristika za Scopus reference: ``Avtor, I., Naslov, Revija, vol, issue,
    pp. x-y, (leto)`` → vrne "Revija". Vrne None, če ni prepoznavne revije."""
    s = re.sub(r"\(\d{4}\)\s*$", "", ref.strip()).rstrip(", ").strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    # odstrani številske repe (strani, letniki, številke)
    while parts and (_PAGES_RE.match(parts[-1]) or _NUM_RE.match(parts[-1]) or _ISSUE_RE.match(parts[-1])):
        parts.pop()
    if len(parts) < 3:
        return None  # avtor(ji) + naslov + revija so minimum
    cand = parts[-1]
    if len(cand) < 3 or len(cand) > 120:
        return None
    if re.search(r"\b(ed\.|eds\.|pp\.|doi|http|retrieved|accessed|available)\b", cand, re.I):
        return None
    return cand


def cited_sources(
    df: pd.DataFrame,
    references_col: str = "References",
    source_col: Optional[str] = "Source title",
    sep: str = ";",
    top_n: int = 30,
    normalize_case: bool = True,
    extractor: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Najpogosteje citirane revije v referencah (knowledge base področja) in
    njihovo prekrivanje z revijami, v katerih korpus objavlja.

    Returns
    -------
    table : pd.DataFrame
        Cited Source, Citations (vse omembe v referencah), Citing Docs
        (število dokumentov, ki revijo citirajo), Share % (delež vseh
        prepoznanih referenc), Cumulative %, Publishing Docs (dokumenti
        korpusa, objavljeni v tej reviji), In Corpus (bool).
    stats : dict
        n_refs, n_refs_with_source, n_distinct_sources, hhi (Herfindahl),
        top10_share, jaccard_overlap (citirane vs. objavne revije),
        self_reliance (delež referenc na revije, v katerih korpus objavlja).
    """
    ext = extractor or _source_from_scopus_reference
    recs = []
    n_refs = 0
    for idx, blob in df[references_col].items():
        if pd.isna(blob):
            continue
        for ref in str(blob).split(sep):
            ref = ref.strip()
            if not ref:
                continue
            n_refs += 1
            src = ext(ref)
            if src:
                recs.append((idx, src.strip()))
    long = pd.DataFrame(recs, columns=["_doc", "Cited Source"])
    if long.empty:
        return pd.DataFrame(columns=["Cited Source", "Citations", "Citing Docs", "Share %",
                                     "Cumulative %", "Publishing Docs", "In Corpus"]), {
            "n_refs": n_refs, "n_refs_with_source": 0}
    key = long["Cited Source"].str.lower().str.strip() if normalize_case else long["Cited Source"]
    long["_key"] = key
    display = long.groupby("_key")["Cited Source"].agg(lambda s: s.value_counts().index[0])
    tab = long.groupby("_key").agg(Citations=("_doc", "size"), **{"Citing Docs": ("_doc", "nunique")})
    tab["Cited Source"] = display
    tab = tab.sort_values(["Citations", "Citing Docs"], ascending=False)
    total = int(tab["Citations"].sum())
    tab["Share %"] = 100.0 * tab["Citations"] / total
    tab["Cumulative %"] = tab["Share %"].cumsum()

    pub_counts = pd.Series(dtype=int)
    if source_col is not None and source_col in df.columns:
        pc = df[source_col].dropna().astype(str).str.strip()
        pub_counts = pc.groupby(pc.str.lower() if normalize_case else pc).size()
    tab["Publishing Docs"] = tab.index.map(lambda k: int(pub_counts.get(k, 0)))
    tab["In Corpus"] = tab["Publishing Docs"] > 0

    shares = tab["Citations"].to_numpy(dtype=float) / total
    stats = {
        "n_refs": n_refs,
        "n_refs_with_source": total,
        "n_distinct_sources": int(len(tab)),
        "hhi": float(np.sum(shares ** 2)),
        "top10_share": float(100.0 * shares[:10].sum()),
        "self_reliance": float(100.0 * tab.loc[tab["In Corpus"], "Citations"].sum() / total),
    }
    if len(pub_counts):
        a, b = set(tab.index), set(pub_counts.index)
        stats["jaccard_overlap"] = float(len(a & b) / len(a | b)) if (a | b) else np.nan
    out = tab.reset_index(drop=True)[["Cited Source", "Citations", "Citing Docs", "Share %",
                                      "Cumulative %", "Publishing Docs", "In Corpus"]]
    if top_n:
        out = out.head(top_n)
    return out, stats


# ---------------------------------------------------------------------------
# 7) Koncentracija: Lorenz, Gini, Theil
# ---------------------------------------------------------------------------

def lorenz_curve(values: Union[Sequence[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Točke Lorenzove krivulje (kumulativni delež enot, kumulativni delež vsote),
    z začetno točko (0, 0). Vrednosti so urejene naraščajoče."""
    x = np.sort(np.asarray(values, dtype=float))
    x = x[~np.isnan(x)]
    if x.size == 0 or x.sum() <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cum = np.cumsum(x) / x.sum()
    p = np.arange(1, x.size + 1) / x.size
    return np.concatenate([[0.0], p]), np.concatenate([[0.0], cum])


def gini_coefficient(values: Union[Sequence[float], np.ndarray]) -> float:
    """Gini koeficient (0 = popolna enakost, 1 = vsa masa v eni enoti)."""
    x = np.sort(np.asarray(values, dtype=float))
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0 or x.sum() <= 0:
        return np.nan
    i = np.arange(1, n + 1)
    return float((2.0 * np.sum(i * x) / (n * x.sum())) - (n + 1.0) / n)


def theil_index(values: Union[Sequence[float], np.ndarray]) -> float:
    """Theilov T indeks (0 = enakost, max = ln N)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    x = x[x > 0]
    if x.size == 0:
        return np.nan
    mu = x.mean()
    r = x / mu
    return float(np.mean(r * np.log(r)))


def concentration(
    df: pd.DataFrame,
    entity_col: str,
    value_col: Optional[str] = None,
    sep: str = ";",
    top_shares: Sequence[float] = (0.1, 0.2),
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Koncentracija produkcije (ali citatov) po entitetah.

    Če ``value_col`` ni podan, se za vsako entiteto šteje število dokumentov;
    sicer vsota ``value_col`` (npr. "Cited by") po dokumentih entitete.

    Returns
    -------
    counts : pd.DataFrame
        Item, Value (urejeno padajoče), Share %, Cumulative %.
    stats : dict
        n_items, total, gini, theil, hhi, top10_share, top20_share (deleži v %),
        share_of_items_for_half (delež entitet, ki prispeva 50 % vsote, v %),
        lorenz_x, lorenz_y.
    """
    long = _explode_items(df, entity_col, sep=sep)
    if value_col is not None:
        long["_v"] = _to_numeric(df[value_col]).reindex(long["_doc"]).fillna(0).to_numpy()
        counts = long.groupby("Item")["_v"].sum()
    else:
        counts = long.groupby("Item").size().astype(float)
    counts = counts.sort_values(ascending=False)
    total = float(counts.sum())
    tab = counts.rename("Value").reset_index()
    tab["Share %"] = 100.0 * tab["Value"] / total if total else np.nan
    tab["Cumulative %"] = tab["Share %"].cumsum()
    v = counts.to_numpy(dtype=float)
    n = v.size
    lx, ly = lorenz_curve(v)
    stats = {
        "n_items": int(n), "total": total,
        "gini": gini_coefficient(v), "theil": theil_index(v),
        "hhi": float(np.sum((v / total) ** 2)) if total else np.nan,
        "lorenz_x": lx, "lorenz_y": ly,
    }
    for s in top_shares:
        k = max(1, int(np.ceil(s * n)))
        stats[f"top{int(round(100 * s))}_share"] = float(100.0 * v[:k].sum() / total) if total else np.nan
    if total:
        cum = np.cumsum(v) / total
        k_half = int(np.searchsorted(cum, 0.5) + 1)
        stats["share_of_items_for_half"] = float(100.0 * k_half / n)
    else:
        stats["share_of_items_for_half"] = np.nan
    return tab, stats


# ---------------------------------------------------------------------------
# 8) Kariere avtorjev v korpusu
# ---------------------------------------------------------------------------

def author_careers(
    df: pd.DataFrame,
    authors_col: str = "Authors",
    year_col: str = "Year",
    citations_col: str = "Cited by",
    sep: str = ";",
    min_docs: int = 2,
    top_n: Optional[int] = 50,
    current_year: Optional[int] = None,
    sort_by: str = "Docs",
) -> pd.DataFrame:
    """
    Kariere avtorjev, kot jih vidi korpus: prvo in zadnje leto, akademska
    starost (current_year − prvo leto + 1), aktivni razpon, produktivnost na
    aktivno leto, lokalni h-indeks, m-kvocient (h / akademska starost; Hirsch
    2005), delež prvoavtorstev in povprečni citati.

    Opomba: "akademska starost" je merjena znotraj korpusa (prvi dokument v
    zbirki), ne kot dejanski začetek kariere.
    """
    long = _explode_items(df, authors_col, sep=sep)
    years = _to_numeric(df[year_col])
    cit = _to_numeric(df[citations_col]).fillna(0)
    first_author = df[authors_col].dropna().astype(str).str.split(sep).str[0].str.strip()
    long["_year"] = years.reindex(long["_doc"]).to_numpy()
    long["_cit"] = cit.reindex(long["_doc"]).to_numpy()
    long["_first"] = (long["Item"].to_numpy() == first_author.reindex(long["_doc"]).to_numpy()).astype(int)
    long = long.dropna(subset=["_year"])
    if current_year is None:
        current_year = int(years.max()) if years.notna().any() else pd.Timestamp.today().year

    g = long.groupby("Item")
    out = pd.DataFrame({
        "Docs": g.size(),
        "First Year": g["_year"].min().astype(int),
        "Last Year": g["_year"].max().astype(int),
        "Active Years": g["_year"].nunique(),
        "Total Citations": g["_cit"].sum().astype(int),
        "Mean Citations": g["_cit"].mean(),
        "h": g["_cit"].agg(_h_index),
        "First-author Docs": g["_first"].sum().astype(int),
    })
    out["Academic Age"] = current_year - out["First Year"] + 1
    out["Span"] = out["Last Year"] - out["First Year"] + 1
    out["Docs per Active Year"] = out["Docs"] / out["Active Years"]
    out["Docs per Year of Age"] = out["Docs"] / out["Academic Age"]
    out["m-quotient"] = out["h"] / out["Academic Age"]
    out["First-author %"] = 100.0 * out["First-author Docs"] / out["Docs"]
    out = out[out["Docs"] >= min_docs]
    if sort_by not in out.columns:
        sort_by = "Docs"
    out = out.sort_values([sort_by, "Total Citations"], ascending=[False, False])
    if top_n:
        out = out.head(top_n)
    out.index.name = "Author"
    cols = ["Docs", "First Year", "Last Year", "Academic Age", "Span", "Active Years",
            "Docs per Active Year", "Docs per Year of Age", "Total Citations", "Mean Citations",
            "h", "m-quotient", "First-author Docs", "First-author %"]
    return out[cols].reset_index()


# ---------------------------------------------------------------------------
# 9) Heapsov zakon
# ---------------------------------------------------------------------------

def fit_heaps_law(
    df: pd.DataFrame,
    col: str = "Author Keywords",
    sep: str = ";",
    order_col: Optional[str] = "Year",
    normalize_case: bool = True,
    seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Heapsov zakon rasti besedišča: V(n) = K · n^β, kjer je n število
    procesiranih žetonov (ključnih besed/pojavnic) in V(n) število različnih.

    Dokumenti se obdelajo v vrstnem redu ``order_col`` (privzeto leto; vezi
    se premešajo s semenom), ali v naključnem vrstnem redu, če je
    ``order_col=None``. β < 1 pomeni sublinearno rast (besedišče se
    "zapira"); β blizu 1 kaže na neomejeno rast / nestandardizirano izrazje.

    Returns
    -------
    curve : pd.DataFrame
        n (kumulativni žetoni), V (različni žetoni), V_fit.
    params : dict
        K, beta, r2 (na log-log skali), n_tokens, vocabulary.
    """
    rng = np.random.default_rng(seed)
    work = df[[col] + ([order_col] if order_col and order_col in df.columns else [])].copy()
    work = work.dropna(subset=[col])
    work["_rnd"] = rng.random(len(work))
    if order_col and order_col in work.columns:
        work["_ord"] = _to_numeric(work[order_col])
        work = work.sort_values(["_ord", "_rnd"])
    else:
        work = work.sort_values("_rnd")
    seen = set()
    n = 0
    ns, vs = [], []
    for blob in work[col].astype(str):
        toks = [t.strip() for t in blob.split(sep)]
        toks = [t.lower() if normalize_case else t for t in toks if t and t.lower() != "nan"]
        for t in toks:
            n += 1
            seen.add(t)
            ns.append(n)
            vs.append(len(seen))
    curve = pd.DataFrame({"n": ns, "V": vs})
    if len(curve) < 3:
        return curve, {"K": np.nan, "beta": np.nan, "r2": np.nan,
                       "n_tokens": int(n), "vocabulary": len(seen)}
    ln_n, ln_v = np.log(curve["n"].to_numpy(float)), np.log(curve["V"].to_numpy(float))
    beta, ln_k = np.polyfit(ln_n, ln_v, 1)
    pred = ln_k + beta * ln_n
    ss_res = float(np.sum((ln_v - pred) ** 2))
    ss_tot = float(np.sum((ln_v - ln_v.mean()) ** 2))
    curve["V_fit"] = np.exp(pred)
    params = {"K": float(np.exp(ln_k)), "beta": float(beta),
              "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
              "n_tokens": int(n), "vocabulary": int(len(seen))}
    return curve, params


# ---------------------------------------------------------------------------
# Risanje
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


def plot_lorenz_curve(
    stats_or_values,
    label: Optional[str] = None,
    title: Optional[str] = "Lorenz curve",
    color: str = "#1f77b4",
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
    ax=None,
):
    """Lorenzova krivulja iz ``concentration()`` statistik (dict) ali surovih vrednosti."""
    import matplotlib.pyplot as plt
    if isinstance(stats_or_values, dict):
        x, y, g = stats_or_values["lorenz_x"], stats_or_values["lorenz_y"], stats_or_values.get("gini")
    else:
        x, y = lorenz_curve(stats_or_values)
        g = gini_coefficient(stats_or_values)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    lab = label or "Lorenz"
    if g is not None and not np.isnan(g):
        lab = f"{lab} (Gini = {g:.3f})"
    ax.plot(x, y, color=color, lw=2, label=lab)
    ax.fill_between(x, y, x, color=color, alpha=0.12)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Equality")
    ax.set_xlabel("Cumulative share of items")
    ax.set_ylabel("Cumulative share of total")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    return _finish(fig, ax, title, f_name, dpi, show)


def plot_css_classes(
    summary: pd.DataFrame,
    labels: Sequence[str] = CSS_LABELS,
    title: Optional[str] = "Characteristic scores and scales",
    cmap: str = "viridis",
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
    ax=None,
):
    """Naloženi stolpci deležev CSS razredov po letih (iz ``summary``; vrstica
    "all" je izpuščena, če obstajajo leta)."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    s = summary.copy()
    if (s["Scope"] != "all").any():
        s = s[s["Scope"] != "all"]
    s = s.set_index("Scope")
    cols = [f"% {lab}" for lab in labels if f"% {lab}" in s.columns]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    colors = mpl.colormaps[cmap](np.linspace(0.15, 0.9, len(cols)))
    bottom = np.zeros(len(s))
    xs = np.arange(len(s))
    for c, col in zip(colors, cols):
        vals = s[col].fillna(0).to_numpy()
        ax.bar(xs, vals, bottom=bottom, color=c, label=col[2:], width=0.8)
        bottom += vals
    ax.set_xticks(xs); ax.set_xticklabels([str(i) for i in s.index], rotation=90 if len(s) > 12 else 0)
    ax.set_ylabel("Share of documents (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _finish(fig, ax, title, f_name, dpi, show)


def plot_reference_age_profile(
    by_year: pd.DataFrame,
    title: Optional[str] = "Reference age profile",
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
    ax=None,
):
    """Mediana starosti referenc (levo) in Priceov indeks (desno) po letu objave."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    ax.plot(by_year["Year"], by_year["Median ref age"], marker="o", color="#1f77b4", label="Median reference age")
    ax.set_xlabel("Publication year"); ax.set_ylabel("Median reference age (years)")
    ax2 = ax.twinx()
    ax2.plot(by_year["Year"], by_year["Price index"], marker="s", color="#d62728", label="Price index (%)")
    ax2.set_ylabel("Price index (%)")
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="best")
    return _finish(fig, ax, title, f_name, dpi, show)


def plot_heaps_law(
    curve: pd.DataFrame,
    params: Dict[str, float],
    title: Optional[str] = "Heaps' law",
    loglog: bool = True,
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
    ax=None,
):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    else:
        fig = ax.figure
    ax.plot(curve["n"], curve["V"], color="#1f77b4", lw=1.5, label="Observed")
    if "V_fit" in curve.columns:
        ax.plot(curve["n"], curve["V_fit"], color="#d62728", ls="--",
                label=f"Fit: V = {params.get('K', np.nan):.2f}·n^{params.get('beta', np.nan):.3f}")
    if loglog:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Tokens processed (n)"); ax.set_ylabel("Vocabulary size V(n)")
    ax.legend(frameon=False)
    return _finish(fig, ax, title, f_name, dpi, show)


def plot_uncitedness(
    table: pd.DataFrame,
    group_col: Optional[str] = None,
    title: Optional[str] = "Share of uncited documents",
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
    ax=None,
):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    if group_col and group_col in table.columns:
        for g, sub in table.groupby(group_col):
            ax.plot(sub["Year"], sub["Uncited %"], marker="o", ms=3, label=str(g))
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.plot(table["Year"], table["Uncited %"], marker="o", color="#1f77b4")
    ax.set_xlabel("Publication year"); ax.set_ylabel("Uncited documents (%)")
    ax.set_ylim(0, max(5.0, float(np.nanmax(table["Uncited %"])) * 1.1) if len(table) else 100)
    return _finish(fig, ax, title, f_name, dpi, show)
