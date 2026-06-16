# -*- coding: utf-8 -*-
"""
Integrity Audit Addon - paper-mill / research-integrity flags
================================================================

This addon screens a bibliographic corpus for known red flags associated
with paper-mill output and research-integrity issues. Each function is
pure (DataFrame in, DataFrame out); `integrity_audit_report` aggregates
all flags into one papers_with_flags table and writes an XLSX + summary
plot.

Functions
---------
- tortured_phrases_check(df, text_cols, lexicon)
- check_openalex_retracted(df, oa_id_col, is_retracted_col)
- compute_author_velocity_anomalies(df, author_col, year_col, sep, z_threshold)
- compute_coauthor_anomalies(df, author_col, sep, min_cluster_size,
        dense_ratio_threshold)
- missing_institution_check(df, inst_col, country_col)
- compute_self_citation_anomalies(df, ref_col, oa_id_col, author_col,
        sep, threshold)
- integrity_audit_report(df, out_folder, **kwargs)
- plot_integrity_summary(audit_df, out, title)

Design notes
------------
- Honours user memory rules: horizontal bar (single-color NAVY) for
  summary; no grid lines; no truncated labels (wrap_label).
- Sandbox-safe: no external API calls. Crossref / Retraction Watch are
  intentionally omitted (BLOCKED in the sandbox).
- Graceful: missing input columns produce warnings + empty results,
  never exceptions.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _resolve_navy() -> str:
    try:
        from biblium.config import plot_config  # type: ignore

        palette = list(plot_config.categorical_palette)
        if palette:
            return palette[0]
    except Exception:
        pass
    return "#1F3864"


def _wrap(label: Any, width: int = 24) -> str:
    try:
        from biblium.utilsbib_modules.plotting_helpers import wrap_label

        return wrap_label(label, width=width)
    except Exception:
        import textwrap

        s = str(label).replace(" and ", " & ")
        if len(s) <= width:
            return s
        return "\n".join(textwrap.wrap(s, width=width)) or s


# ---------------------------------------------------------------------------
# tortured phrases lexicon
# ---------------------------------------------------------------------------
# Curated from Cabanac, Labbe & Magazinov (2021) "Tortured phrases" and
# subsequent updates. Phrases are lower-cased, regex-safe.
DEFAULT_TORTURED_LEXICON: Dict[str, str] = {
    "bosom peril": "breast cancer",
    "colossal information": "big data",
    "haze figuring": "cloud computing",
    "man-made consciousness": "artificial intelligence",
    "fake neural organization": "artificial neural network",
    "fake neural organizations": "artificial neural networks",
    "profound learning": "deep learning",
    "irregular woodland": "random forest",
    "irregular timberland": "random forest",
    "uphold vector machine": "support vector machine",
    "support vector hardware": "support vector machine",
    "underlying foundations": "roots",
    "creature insight": "swarm intelligence",
    "remote sensor organization": "wireless sensor network",
    "remote sensor organizations": "wireless sensor networks",
    "lung carcinoma": "lung cancer",
    "energy productivity": "energy efficiency",
    "signal-to-clamor proportion": "signal-to-noise ratio",
    "mean square mistake": "mean squared error",
    "mean square blunder": "mean squared error",
    "mean outright mistake": "mean absolute error",
    "false neural organization": "artificial neural network",
    "force lattice": "power grid",
    "force network": "power grid",
    "particular worth deterioration": "singular value decomposition",
    "head part examination": "principal component analysis",
    "ahead of all comers": "first place",
    "fragile lattice": "smart grid",
    "shrewd network": "smart grid",
    "savvy network": "smart grid",
    "shrewd metropolitan": "smart city",
    "shrewd metropolitan area": "smart city",
    "internet of things gadgets": "internet of things devices",
    "informal community": "social network",
    "informal communities": "social networks",
    "Gaussian commotion": "Gaussian noise",
    "tumor microenvironment": "tumour microenvironment",
}


def _compile_phrase_patterns(
    lexicon: Dict[str, str],
) -> Tuple[re.Pattern, Dict[str, str]]:
    """Compile a single combined regex of phrases (word-boundary)."""
    phrases = sorted(lexicon.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(p) for p in phrases) + r")\b",
        flags=re.IGNORECASE,
    )
    return pattern, {p.lower(): m for p, m in lexicon.items()}


def tortured_phrases_check(
    df: pd.DataFrame,
    text_cols: Iterable[str] = ("Title", "Abstract"),
    lexicon: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Scan text columns for known tortured phrases.

    Returns DataFrame with columns: doc_idx, phrase_found, suggested_meaning,
    score (1 + 0.5 * (matches - 1)). Only flagged papers are returned.
    """
    if lexicon is None:
        lexicon = DEFAULT_TORTURED_LEXICON
    pattern, mapping = _compile_phrase_patterns(lexicon)

    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        warnings.warn(
            f"tortured_phrases_check: no text columns found "
            f"(looked for {list(text_cols)})"
        )
        return pd.DataFrame(
            columns=["doc_idx", "phrase_found", "suggested_meaning", "score"]
        )

    rows = []
    for idx in range(len(df)):
        blob_parts = []
        for c in cols:
            v = df.iloc[idx][c]
            if isinstance(v, str):
                blob_parts.append(v)
        if not blob_parts:
            continue
        blob = " || ".join(blob_parts).lower()
        matches = pattern.findall(blob)
        if not matches:
            continue
        for m in set(matches):
            count = sum(1 for x in matches if x.lower() == m.lower())
            rows.append(
                {
                    "doc_idx": idx,
                    "phrase_found": m,
                    "suggested_meaning": mapping.get(m.lower(), ""),
                    "score": 1.0 + 0.5 * (count - 1),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OpenAlex retracted flag
# ---------------------------------------------------------------------------
def check_openalex_retracted(
    df: pd.DataFrame,
    oa_id_col: str = "oa_openalex_id",
    is_retracted_col: str = "oa_is_retracted",
) -> Dict[str, Any]:
    """Check the dataframe's OpenAlex `is_retracted` flag (if present)."""
    summary: Dict[str, Any] = {
        "n_total": len(df),
        "n_with_oa_id": 0,
        "n_with_flag": 0,
        "n_retracted": 0,
        "flagged_idx": [],
    }
    if oa_id_col in df.columns:
        summary["n_with_oa_id"] = int(df[oa_id_col].notna().sum())
    if is_retracted_col not in df.columns:
        warnings.warn(
            f"check_openalex_retracted: column '{is_retracted_col}' "
            f"missing from dataframe; returning empty result. Re-run the "
            f"OpenAlex enrichment after upgrading biblium to populate it."
        )
        summary["mask"] = np.zeros(len(df), dtype=bool)
        return summary
    s = df[is_retracted_col]
    has = s.notna()
    summary["n_with_flag"] = int(has.sum())
    mask = s.fillna(False).astype(bool).to_numpy()
    summary["n_retracted"] = int(mask.sum())
    summary["flagged_idx"] = list(np.where(mask)[0])
    summary["mask"] = mask
    return summary


# ---------------------------------------------------------------------------
# author velocity anomalies
# ---------------------------------------------------------------------------
def _split_authors(value: Any, sep: str) -> List[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [a.strip() for a in value.split(sep) if a.strip()]


def compute_author_velocity_anomalies(
    df: pd.DataFrame,
    author_col: str = "Author full names",
    year_col: str = "Year",
    sep: str = "; ",
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """For each author, flag years with z-score >= threshold on n papers."""
    if author_col not in df.columns or year_col not in df.columns:
        warnings.warn(
            f"author velocity: missing column '{author_col}' or '{year_col}'"
        )
        return pd.DataFrame(
            columns=[
                "doc_idx",
                "author",
                "year",
                "n_papers_year",
                "author_mean",
                "author_std",
                "z_score",
                "flag",
            ]
        )

    # Build {author: {year: [doc_idxs]}}
    author_year_docs: Dict[str, Dict[int, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for idx in range(len(df)):
        authors = _split_authors(df.iloc[idx][author_col], sep)
        yr = df.iloc[idx][year_col]
        try:
            yr_i = int(yr)
        except Exception:
            continue
        for a in authors:
            author_year_docs[a][yr_i].append(idx)

    rows = []
    for author, year_map in author_year_docs.items():
        counts = np.array([len(v) for v in year_map.values()], dtype=float)
        if len(counts) < 3:
            continue
        mu = float(counts.mean())
        sd = float(counts.std(ddof=0))
        if sd == 0:
            continue
        for yr, idxs in year_map.items():
            n = len(idxs)
            z = (n - mu) / sd
            if z >= z_threshold:
                for di in idxs:
                    rows.append(
                        {
                            "doc_idx": di,
                            "author": author,
                            "year": yr,
                            "n_papers_year": n,
                            "author_mean": round(mu, 3),
                            "author_std": round(sd, 3),
                            "z_score": round(z, 3),
                            "flag": True,
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# coauthor anomalies
# ---------------------------------------------------------------------------
def compute_coauthor_anomalies(
    df: pd.DataFrame,
    author_col: str = "Author full names",
    sep: str = "; ",
    min_cluster_size: int = 20,
    dense_ratio_threshold: float = 0.7,
) -> pd.DataFrame:
    """Build co-author graph; report dense isolated components."""
    try:
        import networkx as nx
    except Exception:
        warnings.warn("networkx unavailable; coauthor anomalies skipped")
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "size",
                "n_edges",
                "density",
                "isolation",
                "members",
                "suspect_score",
            ]
        )
    if author_col not in df.columns:
        warnings.warn(f"coauthor anomalies: missing column '{author_col}'")
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "size",
                "n_edges",
                "density",
                "isolation",
                "members",
                "suspect_score",
            ]
        )

    G = nx.Graph()
    for v in df[author_col].dropna():
        authors = _split_authors(v, sep)
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a, b = authors[i], authors[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    rows = []
    for cid, comp in enumerate(nx.connected_components(G)):
        sub = G.subgraph(comp)
        n = sub.number_of_nodes()
        if n < min_cluster_size:
            continue
        m = sub.number_of_edges()
        max_edges = n * (n - 1) / 2
        density = m / max_edges if max_edges else 0.0
        # isolation: 1.0 if component is its own connected piece (always
        # 1.0 here since we iterate over connected_components); kept for
        # API symmetry.
        isolation = 1.0
        suspect = float(density >= dense_ratio_threshold) * density
        members = sorted(sub.nodes())
        rows.append(
            {
                "cluster_id": cid,
                "size": n,
                "n_edges": m,
                "density": round(density, 4),
                "isolation": isolation,
                "members": "; ".join(members[:50])
                + (f" ... (+{len(members) - 50})" if len(members) > 50 else ""),
                "suspect_score": round(suspect, 4),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "suspect_score", ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# missing institution
# ---------------------------------------------------------------------------
def missing_institution_check(
    df: pd.DataFrame,
    inst_col: str = "oa_institutions",
    country_col: str = "oa_institution_countries",
) -> pd.DataFrame:
    """Flag papers without any reported institution or country."""
    n = len(df)
    has_inst = np.zeros(n, dtype=bool)
    has_country = np.zeros(n, dtype=bool)
    if inst_col in df.columns:
        s = df[inst_col].fillna("").astype(str).str.strip()
        has_inst = s.str.len().gt(0).to_numpy()
    if country_col in df.columns:
        s = df[country_col].fillna("").astype(str).str.strip()
        has_country = s.str.len().gt(0).to_numpy()
    flag = ~(has_inst | has_country)
    return pd.DataFrame(
        {
            "doc_idx": np.arange(n),
            "has_inst": has_inst,
            "has_country": has_country,
            "flag": flag,
        }
    )


# ---------------------------------------------------------------------------
# self-citation anomalies
# ---------------------------------------------------------------------------
def compute_self_citation_anomalies(
    df: pd.DataFrame,
    ref_col: str = "oa_referenced_works",
    oa_id_col: str = "oa_openalex_id",
    author_col: str = "Author full names",
    sep: str = "; ",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute (#author-self-references) / (#refs) per paper.

    Heuristic (no per-reference author data available): a reference is
    considered a "self-cite" if its OpenAlex id appears in the corpus AND
    the cited paper shares at least one author with the citing paper.
    """
    n = len(df)
    out_cols = ["doc_idx", "n_refs", "n_self_cites", "self_cit_ratio", "flag"]
    if ref_col not in df.columns or oa_id_col not in df.columns:
        warnings.warn(
            f"self-citation: missing '{ref_col}' or '{oa_id_col}'; "
            f"returning empty result"
        )
        return pd.DataFrame(columns=out_cols)

    # Map OA id -> author set
    oa2authors: Dict[str, set] = {}
    if author_col in df.columns:
        for idx in range(n):
            oa = df.iloc[idx][oa_id_col]
            if not isinstance(oa, str) or not oa:
                continue
            oa2authors[oa] = set(_split_authors(df.iloc[idx][author_col], sep))

    rows = []
    for idx in range(n):
        refs_raw = df.iloc[idx][ref_col]
        if not isinstance(refs_raw, str) or not refs_raw.strip():
            continue
        refs = [r.strip() for r in refs_raw.split("|") if r.strip()]
        n_refs = len(refs)
        if n_refs == 0:
            continue
        my_authors = set(_split_authors(df.iloc[idx].get(author_col, ""), sep))
        if not my_authors:
            continue
        n_self = 0
        for r in refs:
            ref_authors = oa2authors.get(r)
            if ref_authors and my_authors & ref_authors:
                n_self += 1
        ratio = n_self / n_refs if n_refs else 0.0
        rows.append(
            {
                "doc_idx": idx,
                "n_refs": n_refs,
                "n_self_cites": n_self,
                "self_cit_ratio": round(ratio, 4),
                "flag": ratio >= threshold and n_refs >= 5,
            }
        )
    return pd.DataFrame(rows, columns=out_cols)


# ---------------------------------------------------------------------------
# aggregator
# ---------------------------------------------------------------------------
def integrity_audit_report(
    df: pd.DataFrame,
    out_folder: Union[str, Path],
    text_cols: Iterable[str] = ("Title", "Abstract"),
    author_col: str = "Author full names",
    year_col: str = "Year",
    sep: str = "; ",
    z_velocity: float = 3.0,
    self_cit_threshold: float = 0.5,
    coauthor_min_size: int = 20,
    coauthor_density: float = 0.7,
    oa_id_col: str = "oa_openalex_id",
    is_retracted_col: str = "oa_is_retracted",
    inst_col: str = "oa_institutions",
    country_col: str = "oa_institution_countries",
    ref_col: str = "oa_referenced_works",
) -> Dict[str, Any]:
    """Run all integrity checks, aggregate to one papers_with_flags table."""
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    n = len(df)

    results: Dict[str, Any] = {}

    # 1. tortured phrases
    tort_df = tortured_phrases_check(df, text_cols=text_cols)
    flag_tort = np.zeros(n, dtype=bool)
    if not tort_df.empty:
        flag_tort[tort_df["doc_idx"].astype(int).unique()] = True
    results["tortured"] = tort_df

    # 2. retracted
    retr = check_openalex_retracted(
        df, oa_id_col=oa_id_col, is_retracted_col=is_retracted_col
    )
    flag_retracted = retr["mask"]
    results["retracted"] = retr

    # 3. velocity anomalies
    vel_df = compute_author_velocity_anomalies(
        df,
        author_col=author_col,
        year_col=year_col,
        sep=sep,
        z_threshold=z_velocity,
    )
    flag_vel = np.zeros(n, dtype=bool)
    if not vel_df.empty:
        flag_vel[vel_df["doc_idx"].astype(int).unique()] = True
    results["velocity"] = vel_df

    # 4. coauthor anomalies (cluster level; mark papers whose authors
    # belong to a flagged dense cluster).
    coa_df = compute_coauthor_anomalies(
        df,
        author_col=author_col,
        sep=sep,
        min_cluster_size=coauthor_min_size,
        dense_ratio_threshold=coauthor_density,
    )
    flag_coa = np.zeros(n, dtype=bool)
    if not coa_df.empty:
        suspect_authors: set = set()
        for _, r in coa_df.iterrows():
            if r["suspect_score"] > 0:
                # extract before " ... " tail if present
                m_text = str(r["members"]).split(" ... ")[0]
                for a in m_text.split("; "):
                    a = a.strip()
                    if a:
                        suspect_authors.add(a)
        if suspect_authors and author_col in df.columns:
            for idx in range(n):
                authors = set(_split_authors(df.iloc[idx][author_col], sep))
                if authors & suspect_authors:
                    flag_coa[idx] = True
    results["coauthor"] = coa_df

    # 5. missing institution
    inst_df = missing_institution_check(
        df, inst_col=inst_col, country_col=country_col
    )
    flag_inst = inst_df["flag"].to_numpy() if not inst_df.empty else np.zeros(
        n, dtype=bool
    )
    results["missing_inst"] = inst_df

    # 6. self-citation
    sc_df = compute_self_citation_anomalies(
        df,
        ref_col=ref_col,
        oa_id_col=oa_id_col,
        author_col=author_col,
        sep=sep,
        threshold=self_cit_threshold,
    )
    flag_sc = np.zeros(n, dtype=bool)
    if not sc_df.empty:
        flag_sc[sc_df[sc_df["flag"]]["doc_idx"].astype(int).to_numpy()] = True
    results["self_cit"] = sc_df

    # aggregate
    papers_with_flags = pd.DataFrame(
        {
            "doc_idx": np.arange(n),
            "flag_tortured": flag_tort,
            "flag_oa_retracted": flag_retracted,
            "flag_velocity": flag_vel,
            "flag_coauthor_anomaly": flag_coa,
            "flag_no_inst": flag_inst,
            "flag_self_cit": flag_sc,
        }
    )
    flag_cols = [
        "flag_tortured",
        "flag_oa_retracted",
        "flag_velocity",
        "flag_coauthor_anomaly",
        "flag_no_inst",
        "flag_self_cit",
    ]
    papers_with_flags["total_flags"] = papers_with_flags[flag_cols].sum(axis=1)

    # include some descriptive columns
    descriptor_cols = []
    for c in ("Title", "Year", "DOI", "Authors", "Source title"):
        if c in df.columns:
            descriptor_cols.append(c)
    if descriptor_cols:
        descriptor = df[descriptor_cols].reset_index(drop=True)
        descriptor.insert(0, "doc_idx", np.arange(n))
        papers_with_flags = papers_with_flags.merge(
            descriptor, on="doc_idx", how="left"
        )

    papers_with_flags = papers_with_flags.sort_values(
        "total_flags", ascending=False
    ).reset_index(drop=True)

    # write outputs
    xlsx_path = out_folder / "papers_with_flags.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        papers_with_flags.to_excel(xw, sheet_name="papers", index=False)
        if not tort_df.empty:
            tort_df.to_excel(xw, sheet_name="tortured", index=False)
        if not vel_df.empty:
            vel_df.to_excel(xw, sheet_name="velocity", index=False)
        if not coa_df.empty:
            coa_df.to_excel(xw, sheet_name="coauthor_clusters", index=False)
        if not sc_df.empty:
            sc_df.to_excel(xw, sheet_name="self_citation", index=False)
        inst_df.to_excel(xw, sheet_name="missing_inst", index=False)

    results["papers_with_flags"] = papers_with_flags
    results["xlsx"] = xlsx_path
    return results


# ---------------------------------------------------------------------------
# summary plot
# ---------------------------------------------------------------------------
def plot_integrity_summary(
    audit_df: pd.DataFrame,
    out: Union[str, Path],
    title: str = "Integrity audit summary",
    df_for_year: Optional[pd.DataFrame] = None,
    year_col: str = "Year",
) -> None:
    """Two-panel summary plot.

    Top: horizontal bar (NAVY) of #papers flagged per indicator.
    Bottom: bar chart of #papers with >=2 flags per year (if year provided).
    """
    flag_cols = [c for c in audit_df.columns if c.startswith("flag_")]
    counts = audit_df[flag_cols].sum().sort_values(ascending=True)
    labels = [
        _wrap(
            c.replace("flag_", "").replace("_", " "),
            width=22,
        )
        for c in counts.index
    ]
    navy = _resolve_navy()

    if df_for_year is not None and year_col in df_for_year.columns:
        fig, axes = plt.subplots(2, 1, figsize=(10, 9))
        ax_top, ax_bot = axes
    else:
        fig, ax_top = plt.subplots(figsize=(10, 6))
        ax_bot = None

    ax_top.barh(labels, counts.values, color=navy, alpha=0.9)
    for y, v in enumerate(counts.values):
        ax_top.text(
            v,
            y,
            f" {int(v):,}",
            va="center",
            ha="left",
            fontsize=9,
            color="black",
        )
    ax_top.set_title(_wrap(title, width=70), fontsize=12)
    ax_top.set_xlabel("Number of papers flagged")
    ax_top.grid(False)
    for spine in ("top", "right"):
        ax_top.spines[spine].set_visible(False)

    if ax_bot is not None:
        multi = audit_df[audit_df["total_flags"] >= 2]
        if not multi.empty and "doc_idx" in multi.columns:
            yrs = df_for_year.loc[
                multi["doc_idx"].astype(int).values, year_col
            ]
            yrs = pd.to_numeric(yrs, errors="coerce").dropna().astype(int)
            if not yrs.empty:
                per_yr = yrs.value_counts().sort_index()
                ax_bot.bar(
                    per_yr.index.astype(int),
                    per_yr.values,
                    color=navy,
                    alpha=0.9,
                )
                ax_bot.set_title(
                    _wrap(
                        "Papers with >=2 integrity flags by year",
                        width=70,
                    ),
                    fontsize=11,
                )
                ax_bot.set_xlabel("Year")
                ax_bot.set_ylabel("Number of papers")
                ax_bot.grid(False)
                for spine in ("top", "right"):
                    ax_bot.spines[spine].set_visible(False)
            else:
                ax_bot.set_visible(False)
        else:
            ax_bot.set_visible(False)

    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "DEFAULT_TORTURED_LEXICON",
    "tortured_phrases_check",
    "check_openalex_retracted",
    "compute_author_velocity_anomalies",
    "compute_coauthor_anomalies",
    "missing_institution_check",
    "compute_self_citation_anomalies",
    "integrity_audit_report",
    "plot_integrity_summary",
]
