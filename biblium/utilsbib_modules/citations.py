# -*- coding: utf-8 -*-
"""
Citation Analysis Module.

This module provides advanced citation analysis functions:
- Field-normalized citations (FWCI-like metrics)
- Self-citation detection
- Citation velocity and momentum
- DOI-based reference matching
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


# =============================================================================
# FIELD-NORMALIZED CITATIONS
# =============================================================================

def compute_field_normalized_citations(
    df: pd.DataFrame,
    citations_col: str = "Cited by",
    year_col: str = "Year",
    field_col: Optional[str] = None,
    method: str = "mean",
    min_field_size: int = 5,
) -> pd.DataFrame:
    """
    Compute field-normalized citation scores (similar to FWCI).
    
    The Field-Weighted Citation Impact (FWCI) normalizes citations by comparing
    to the average citations of similar documents (same field, same year, same type).
    
    FWCI = actual_citations / expected_citations
    - FWCI > 1: Above average impact
    - FWCI = 1: Average impact  
    - FWCI < 1: Below average impact
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic data with citations.
    citations_col : str
        Column containing citation counts.
    year_col : str
        Column containing publication years.
    field_col : str, optional
        Column containing field/subject classification.
        If None, normalizes by year only.
    method : str
        Normalization method: 'mean' or 'median'.
    min_field_size : int
        Minimum number of documents in a field-year group for normalization.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - 'Expected Citations': Average/median citations for that group
        - 'FNCI': Field-normalized citation impact
        - 'FNCI Percentile': Percentile rank within group
    """
    df = df.copy()
    
    # Ensure numeric citations
    df[citations_col] = pd.to_numeric(df[citations_col], errors="coerce").fillna(0)
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    
    # Define grouping columns
    group_cols = [year_col]
    if field_col and field_col in df.columns:
        group_cols.append(field_col)
    
    # Compute expected citations per group
    if method == "median":
        expected = df.groupby(group_cols)[citations_col].transform("median")
    else:
        expected = df.groupby(group_cols)[citations_col].transform("mean")
    
    # Handle small groups - fall back to year-only
    group_sizes = df.groupby(group_cols)[citations_col].transform("size")
    
    if field_col and field_col in df.columns:
        # For small groups, use year-only average
        year_expected = df.groupby(year_col)[citations_col].transform(method)
        expected = np.where(group_sizes < min_field_size, year_expected, expected)
    
    df["Expected Citations"] = expected
    
    # Compute FNCI (avoid division by zero)
    df["FNCI"] = np.where(
        expected > 0,
        df[citations_col] / expected,
        np.where(df[citations_col] > 0, 2.0, 0.0)  # Above avg if has citations, else 0
    )
    
    # Compute percentile within group
    def compute_percentile(group):
        return group.rank(pct=True) * 100
    
    df["FNCI Percentile"] = df.groupby(group_cols)[citations_col].transform(compute_percentile)
    
    # Round for display
    df["Expected Citations"] = df["Expected Citations"].round(2)
    df["FNCI"] = df["FNCI"].round(3)
    df["FNCI Percentile"] = df["FNCI Percentile"].round(1)
    
    return df


def compute_citation_classes(
    df: pd.DataFrame,
    fnci_col: str = "FNCI",
    percentile_col: str = "FNCI Percentile",
) -> pd.DataFrame:
    """
    Classify documents by citation impact.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with FNCI computed.
    fnci_col : str
        Column with FNCI values.
    percentile_col : str
        Column with percentile values.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'Citation Class' column.
    """
    df = df.copy()
    
    def classify(row):
        fnci = row.get(fnci_col, 0)
        pct = row.get(percentile_col, 50)
        
        if pd.isna(fnci) or pd.isna(pct):
            return "Unknown"
        
        if pct >= 99:
            return "Top 1%"
        elif pct >= 90:
            return "Top 10%"
        elif pct >= 75:
            return "Top 25%"
        elif fnci >= 1:
            return "Above Average"
        elif fnci >= 0.5:
            return "Average"
        else:
            return "Below Average"
    
    df["Citation Class"] = df.apply(classify, axis=1)
    
    return df


def get_citation_summary(
    df: pd.DataFrame,
    citations_col: str = "Cited by",
    fnci_col: str = "FNCI",
) -> Dict[str, Any]:
    """
    Generate summary statistics for citations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with citation data.
    citations_col : str
        Column with citation counts.
    fnci_col : str
        Column with FNCI values.
        
    Returns
    -------
    dict
        Summary statistics.
    """
    citations = pd.to_numeric(df[citations_col], errors="coerce").fillna(0)
    
    summary = {
        "total_citations": int(citations.sum()),
        "mean_citations": round(citations.mean(), 2),
        "median_citations": round(citations.median(), 2),
        "max_citations": int(citations.max()),
        "h_index": _compute_h_index(citations.tolist()),
        "uncited_papers": int((citations == 0).sum()),
        "uncited_percentage": round(100 * (citations == 0).sum() / len(df), 1),
    }
    
    if fnci_col in df.columns:
        fnci = pd.to_numeric(df[fnci_col], errors="coerce")
        summary["mean_fnci"] = round(fnci.mean(), 3)
        summary["papers_above_average"] = int((fnci >= 1).sum())
        summary["top_10_percent"] = int((df.get("FNCI Percentile", pd.Series()) >= 90).sum())
    
    return summary


def _compute_h_index(citations: List[int]) -> int:
    """Compute h-index from citation list."""
    sorted_cites = sorted([int(c) for c in citations if pd.notna(c)], reverse=True)
    h = 0
    for i, c in enumerate(sorted_cites, start=1):
        if c >= i:
            h = i
        else:
            break
    return h


# =============================================================================
# SELF-CITATION DETECTION
# =============================================================================

def detect_self_citations(
    df: pd.DataFrame,
    authors_col: str = "Authors",
    references_col: str = "References",
    separator: str = "; ",
    author_match_threshold: float = 0.85,
) -> pd.DataFrame:
    """
    Detect self-citations in bibliographic data.
    
    A self-citation is when a paper cites another paper by the same author(s).
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic data.
    authors_col : str
        Column containing author names.
    references_col : str
        Column containing references.
    separator : str
        Separator for multi-value fields.
    author_match_threshold : float
        Threshold for fuzzy author matching (0-1).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - 'Self Citations Count': Number of self-citations
        - 'Self Citation Rate': Percentage of references that are self-citations
        - 'Self Citing Authors': Authors involved in self-citations
    """
    df = df.copy()
    
    if references_col not in df.columns:
        df["Self Citations Count"] = 0
        df["Self Citation Rate"] = 0.0
        df["Self Citing Authors"] = ""
        return df
    
    # Build author index: author_name -> set of doc indices
    author_docs = defaultdict(set)
    
    for idx, row in df.iterrows():
        if pd.isna(row.get(authors_col)):
            continue
        
        authors = str(row[authors_col]).split(separator)
        for author in authors:
            author = _normalize_author_name(author)
            if author:
                author_docs[author].add(idx)
    
    # For each document, check references against author index
    self_cite_counts = []
    self_cite_rates = []
    self_cite_authors = []
    
    for idx, row in df.iterrows():
        if pd.isna(row.get(references_col)) or pd.isna(row.get(authors_col)):
            self_cite_counts.append(0)
            self_cite_rates.append(0.0)
            self_cite_authors.append("")
            continue
        
        # Get document authors
        doc_authors = set()
        for author in str(row[authors_col]).split(separator):
            normalized = _normalize_author_name(author)
            if normalized:
                doc_authors.add(normalized)
        
        # Parse references and look for author matches
        refs = str(row[references_col]).split(separator)
        
        self_citations = 0
        involved_authors = set()
        
        for ref in refs:
            ref_authors = _extract_authors_from_reference(ref)
            
            for ref_author in ref_authors:
                ref_author_norm = _normalize_author_name(ref_author)
                
                for doc_author in doc_authors:
                    if _authors_match(doc_author, ref_author_norm, author_match_threshold):
                        self_citations += 1
                        involved_authors.add(doc_author)
                        break
        
        total_refs = len([r for r in refs if r.strip()])
        
        self_cite_counts.append(self_citations)
        self_cite_rates.append(
            round(100 * self_citations / total_refs, 1) if total_refs > 0 else 0.0
        )
        self_cite_authors.append(separator.join(sorted(involved_authors)))
    
    df["Self Citations Count"] = self_cite_counts
    df["Self Citation Rate"] = self_cite_rates
    df["Self Citing Authors"] = self_cite_authors
    
    return df


def _normalize_author_name(name: str) -> str:
    """Normalize author name for comparison."""
    if not name or not isinstance(name, str):
        return ""
    
    name = name.strip().lower()
    
    # Remove accents
    import unicodedata
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )
    
    # Remove punctuation except spaces and hyphens
    name = re.sub(r"[^\w\s-]", "", name)
    name = " ".join(name.split())
    
    return name


def _extract_authors_from_reference(ref: str) -> List[str]:
    """Extract author names from a reference string."""
    if not ref or not isinstance(ref, str):
        return []
    
    ref = ref.strip()
    
    # Try to extract first part (usually authors) before year or title
    # Common patterns: "Author1, Author2 (2020)" or "Author1, Author2. Title..."
    
    # Split on year pattern
    parts = re.split(r'\(\d{4}\)|\b\d{4}\b', ref)
    if parts:
        author_part = parts[0]
    else:
        author_part = ref
    
    # Split on period (end of author section)
    author_part = author_part.split(".")[0]
    
    # Extract individual authors
    # Handle "and", "&", commas
    author_part = re.sub(r'\s+and\s+|\s*&\s*', ', ', author_part, flags=re.IGNORECASE)
    
    authors = []
    for part in author_part.split(","):
        part = part.strip()
        if part and len(part) > 1:
            # Skip single letters (initials only)
            if not re.match(r'^[A-Z]\.?$', part):
                authors.append(part)
    
    return authors[:5]  # Limit to first 5 authors


def _authors_match(author1: str, author2: str, threshold: float = 0.85) -> bool:
    """Check if two author names match."""
    if not author1 or not author2:
        return False
    
    # Exact match
    if author1 == author2:
        return True
    
    # Check last name match
    parts1 = author1.split()
    parts2 = author2.split()
    
    if parts1 and parts2:
        last1 = parts1[-1] if len(parts1[-1]) > 2 else parts1[0]
        last2 = parts2[-1] if len(parts2[-1]) > 2 else parts2[0]
        
        if last1 == last2:
            # Same last name - check initials
            init1 = "".join(p[0] for p in parts1 if p)
            init2 = "".join(p[0] for p in parts2 if p)
            
            if init1 and init2 and init1[0] == init2[0]:
                return True
    
    # Fuzzy match
    if FUZZY_AVAILABLE:
        ratio = fuzz.ratio(author1, author2) / 100.0
        return ratio >= threshold
    
    return False


def get_self_citation_summary(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate summary of self-citation patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with self-citation analysis results.
        
    Returns
    -------
    dict
        Summary statistics.
    """
    if "Self Citations Count" not in df.columns:
        return {"error": "Self-citation analysis not performed"}
    
    sc_count = df["Self Citations Count"]
    sc_rate = df["Self Citation Rate"]
    
    return {
        "total_self_citations": int(sc_count.sum()),
        "papers_with_self_citations": int((sc_count > 0).sum()),
        "percentage_with_self_citations": round(100 * (sc_count > 0).sum() / len(df), 1),
        "mean_self_citation_rate": round(sc_rate.mean(), 2),
        "max_self_citation_rate": round(sc_rate.max(), 1),
        "median_self_citations_per_paper": round(sc_count.median(), 1),
    }


# =============================================================================
# DOI-BASED REFERENCE MATCHING
# =============================================================================

def extract_dois_from_references(
    df: pd.DataFrame,
    references_col: str = "References",
    separator: str = "; ",
) -> pd.DataFrame:
    """
    Extract DOIs from reference strings.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic data.
    references_col : str
        Column containing references.
    separator : str
        Separator for multi-value fields.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'Reference DOIs' column containing extracted DOIs.
    """
    df = df.copy()
    
    if references_col not in df.columns:
        df["Reference DOIs"] = ""
        df["Reference DOI Count"] = 0
        return df
    
    doi_pattern = re.compile(
        r'\b(10\.\d{4,}/[^\s,;\[\]<>\"\']+)',
        re.IGNORECASE
    )
    
    ref_dois = []
    doi_counts = []
    
    for idx, row in df.iterrows():
        if pd.isna(row.get(references_col)):
            ref_dois.append("")
            doi_counts.append(0)
            continue
        
        refs = str(row[references_col])
        matches = doi_pattern.findall(refs)
        
        # Clean DOIs
        cleaned = []
        for doi in matches:
            # Remove trailing punctuation
            doi = re.sub(r'[.,;:)\]]+$', '', doi)
            doi = doi.lower()
            if doi not in cleaned:
                cleaned.append(doi)
        
        ref_dois.append(separator.join(cleaned))
        doi_counts.append(len(cleaned))
    
    df["Reference DOIs"] = ref_dois
    df["Reference DOI Count"] = doi_counts
    
    return df


def match_references_by_doi(
    df: pd.DataFrame,
    doi_col: str = "DOI",
    ref_dois_col: str = "Reference DOIs",
    separator: str = "; ",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match references to documents in the dataset using DOIs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic data with DOIs extracted from references.
    doi_col : str
        Column containing document DOIs.
    ref_dois_col : str
        Column containing reference DOIs.
    separator : str
        Separator for multi-value fields.
        
    Returns
    -------
    tuple
        (df_with_matches, citation_network)
        - df_with_matches: DataFrame with 'Matched References' column
        - citation_network: DataFrame with source-target citation pairs
    """
    df = df.copy()
    
    if doi_col not in df.columns or ref_dois_col not in df.columns:
        df["Matched References"] = 0
        df["Internal Citation Rate"] = 0.0
        return df, pd.DataFrame(columns=["source", "target", "source_idx", "target_idx"])
    
    # Build DOI index
    doi_to_idx = {}
    for idx, row in df.iterrows():
        doi = row.get(doi_col)
        if pd.notna(doi):
            doi_clean = str(doi).lower().strip()
            if doi_clean:
                doi_to_idx[doi_clean] = idx
    
    # Match references
    matched_counts = []
    total_ref_counts = []
    citation_edges = []
    
    for idx, row in df.iterrows():
        if pd.isna(row.get(ref_dois_col)):
            matched_counts.append(0)
            total_ref_counts.append(0)
            continue
        
        ref_dois = str(row[ref_dois_col]).split(separator)
        total_refs = len([d for d in ref_dois if d.strip()])
        
        matches = 0
        for ref_doi in ref_dois:
            ref_doi = ref_doi.strip().lower()
            if ref_doi in doi_to_idx:
                target_idx = doi_to_idx[ref_doi]
                if target_idx != idx:  # Exclude self-references
                    citation_edges.append({
                        "source": row.get(doi_col, ""),
                        "target": ref_doi,
                        "source_idx": idx,
                        "target_idx": target_idx,
                    })
                    matches += 1
        
        matched_counts.append(matches)
        total_ref_counts.append(total_refs)
    
    df["Matched References"] = matched_counts
    df["Internal Citation Rate"] = [
        round(100 * m / t, 1) if t > 0 else 0.0
        for m, t in zip(matched_counts, total_ref_counts)
    ]
    
    citation_network = pd.DataFrame(citation_edges)
    
    return df, citation_network


def build_citation_network_from_dois(
    df: pd.DataFrame,
    doi_col: str = "DOI",
    references_col: str = "References",
    separator: str = "; ",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build a citation network using DOI-based matching.
    
    This is a convenience function that combines DOI extraction and matching.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic data.
    doi_col : str
        Column containing document DOIs.
    references_col : str
        Column containing references.
    separator : str
        Separator for multi-value fields.
        
    Returns
    -------
    tuple
        (citation_network_df, stats)
        - citation_network_df: DataFrame with source-target pairs
        - stats: Dictionary with network statistics
    """
    # Extract DOIs from references
    df = extract_dois_from_references(df, references_col, separator)
    
    # Match references
    df, citation_network = match_references_by_doi(df, doi_col, "Reference DOIs", separator)
    
    # Compute statistics
    stats = {
        "total_documents": len(df),
        "documents_with_doi": int(df[doi_col].notna().sum()),
        "total_references_with_doi": int(df["Reference DOI Count"].sum()),
        "matched_references": int(df["Matched References"].sum()),
        "citation_edges": len(citation_network),
        "mean_internal_citation_rate": round(df["Internal Citation Rate"].mean(), 2),
    }
    
    if len(citation_network) > 0:
        # Add network metrics
        cited_docs = set(citation_network["target_idx"])
        citing_docs = set(citation_network["source_idx"])
        
        stats["documents_cited"] = len(cited_docs)
        stats["documents_citing"] = len(citing_docs)
        stats["isolated_documents"] = len(df) - len(cited_docs | citing_docs)
    
    return citation_network, stats


# =============================================================================
# CITATION VELOCITY
# =============================================================================

def compute_citation_velocity(
    df: pd.DataFrame,
    citations_col: str = "Cited by",
    year_col: str = "Year",
    current_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute citation velocity metrics.
    
    Citation velocity measures how quickly a paper accumulates citations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic data.
    citations_col : str
        Column containing citation counts.
    year_col : str
        Column containing publication years.
    current_year : int, optional
        Current year for age calculation. Defaults to max year in data.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - 'Age': Years since publication
        - 'Citations per Year': Average annual citations
        - 'Citation Velocity Class': Fast/Normal/Slow classification
    """
    df = df.copy()
    
    # Get current year
    if current_year is None:
        current_year = int(df[year_col].max())
    
    # Compute age
    df["Age"] = current_year - pd.to_numeric(df[year_col], errors="coerce")
    df["Age"] = df["Age"].clip(lower=1)  # Minimum 1 year
    
    # Compute citations per year
    citations = pd.to_numeric(df[citations_col], errors="coerce").fillna(0)
    df["Citations per Year"] = (citations / df["Age"]).round(2)
    
    # Classify velocity
    def classify_velocity(row):
        cpy = row.get("Citations per Year", 0)
        age = row.get("Age", 1)
        
        if pd.isna(cpy):
            return "Unknown"
        
        # Thresholds based on age
        if age <= 2:
            if cpy >= 10:
                return "Fast"
            elif cpy >= 3:
                return "Normal"
            else:
                return "Slow"
        elif age <= 5:
            if cpy >= 5:
                return "Fast"
            elif cpy >= 1:
                return "Normal"
            else:
                return "Slow"
        else:
            if cpy >= 3:
                return "Fast"
            elif cpy >= 0.5:
                return "Normal"
            else:
                return "Slow"
    
    df["Citation Velocity Class"] = df.apply(classify_velocity, axis=1)
    
    return df


def compute_local_citations(
    df: pd.DataFrame,
    doi_col: str = "DOI",
    references_col: str = "References",
    cited_by_col: str = "Cited by",
    title_col: str = "Title",
    year_col: str = "Year",
    separator: str = "; ",
    match_by: str = "auto",
    min_title_len: int = 20,
) -> pd.DataFrame:
    """
    Compute Local Citation Score (LCS) — kolikokrat je vsak dokument v zbirki
    citiran s strani drugih dokumentov iste zbirke — in razmerje do Global
    Citation Score (GCS, stolpec `cited_by_col`).

    match_by:
    - "doi":   ujemanje po DOI v referencah
    - "title": ujemanje po normaliziranem naslovu (podniz v referencah);
               upoštevajo se naslovi z vsaj `min_title_len` znaki
    - "auto":  najprej DOI; če reference ne vsebujejo DOI-jev, preklopi na title

    Returns
    -------
    pd.DataFrame
        Kopija df z dodanima stolpcema "Local Citations" in "LCS/GCS".
    """
    def _norm_text(s: pd.Series) -> pd.Series:
        return (
            s.astype(str).str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip()
        )

    lcs = pd.Series(0, index=df.index, dtype=int)
    mode = match_by

    if mode in ("doi", "auto"):
        work = extract_dois_from_references(df, references_col=references_col, separator=separator)
        exploded = (
            work["Reference DOIs"].astype(str).str.split(separator).explode().str.strip()
        )
        exploded = exploded[(exploded != "") & (exploded != "nan")]
        if len(exploded) == 0 and mode == "auto":
            mode = "title"
        else:
            ref_counts = exploded.value_counts()
            if doi_col in df.columns:
                doi_norm = df[doi_col].astype(str).str.strip().str.lower()
                lcs = doi_norm.map(ref_counts).fillna(0).astype(int)
                lcs[doi_norm.isin(["", "nan", "none"])] = 0
            mode = "doi"

    if mode == "title" and title_col in df.columns and references_col in df.columns:
        refs_norm = _norm_text(df[references_col].fillna(""))
        titles_norm = _norm_text(df[title_col].fillna(""))
        for idx, t in titles_norm.items():
            if len(t) < min_title_len:
                continue
            mask = refs_norm.str.contains(t, regex=False)
            mask.loc[idx] = False  # brez samocitiranja lastnega zapisa
            lcs.loc[idx] = int(mask.sum())

    out = df.copy()
    out["Local Citations"] = lcs
    if cited_by_col in out.columns:
        gcs = pd.to_numeric(out[cited_by_col], errors="coerce")
        out["LCS/GCS"] = (lcs / gcs.where(gcs > 0)).round(3)
    return out


def build_author_cocitation_network(
    df: pd.DataFrame,
    refs_col: str = "References",
    *,
    top_n_authors: int = 100,
    min_cocitations: int = 5,
    first_author_only: bool = True,
    exclude_authors: Optional[List[str]] = None,
):
    """
    Author co-citation analysis (ACA) iz referenčnih nizov (2.18, novo).

    Za vsak dokument iz referenc izlušči (prve) avtorje citiranih del;
    dva avtorja sta so-citirana, če se pojavita v referencah istega
    dokumenta. Vrne (G, authors_df, edges_df):
      - G: networkx.Graph z utežmi 'weight' (št. so-citiranj)
      - authors_df: avtor, n_docs (v koliko dokumentih je citiran)
      - edges_df: author1, author2, weight

    Parameters
    ----------
    first_author_only : bool, default True
        Standardna ACA praksa (White & Griffith 1981): upošteva samo
        prvega avtorja vsake reference.
    exclude_authors : list, optional
        Normalizirana imena, ki se izločijo (npr. avtorja analizirane
        knjige, da ne dominirata mreže).
    """
    import networkx as nx
    from itertools import combinations
    from collections import Counter

    excl = {_normalize_author_name(a) for a in (exclude_authors or [])}

    doc_authors: List[set] = []
    freq: Counter = Counter()
    for refs in df[refs_col].fillna(""):
        seen = set()
        for seg in str(refs).split(";"):
            seg = seg.strip()
            if not seg:
                continue
            auths = _extract_authors_from_reference(seg)
            if first_author_only:
                auths = auths[:1]
            for a in auths:
                na = _normalize_author_name(a)
                if na and na not in excl and len(na) > 3:
                    seen.add(na)
        doc_authors.append(seen)
        freq.update(seen)

    top = {a for a, _ in freq.most_common(top_n_authors)}
    pair_counts: Counter = Counter()
    for seen in doc_authors:
        sel = sorted(seen & top)
        for a, b in combinations(sel, 2):
            pair_counts[(a, b)] += 1

    G = nx.Graph()
    for a in top:
        G.add_node(a, n_docs=int(freq[a]))
    edges = []
    for (a, b), w in pair_counts.items():
        if w >= min_cocitations:
            G.add_edge(a, b, weight=int(w))
            edges.append({"author1": a, "author2": b, "weight": int(w)})
    G.remove_nodes_from(list(nx.isolates(G)))

    authors_df = pd.DataFrame(
        [{"author": a, "n_docs": int(freq[a])} for a in top]
    ).sort_values("n_docs", ascending=False).reset_index(drop=True)
    edges_df = pd.DataFrame(edges).sort_values(
        "weight", ascending=False).reset_index(drop=True) if edges else \
        pd.DataFrame(columns=["author1", "author2", "weight"])
    return G, authors_df, edges_df


def compute_network_centralities(G, weight: str = "weight") -> pd.DataFrame:
    """
    Tabela centralnosti za poljubno omrežje (2.18, novo): degree,
    weighted degree, betweenness, eigenvector, PageRank, HITS (za
    usmerjena omrežja) — vse v enem DataFrame, urejeno po PageRank.
    """
    import networkx as nx

    out = pd.DataFrame(index=list(G.nodes()))
    out["degree"] = pd.Series(dict(G.degree()))
    out["weighted_degree"] = pd.Series(dict(G.degree(weight=weight)))
    try:
        out["betweenness"] = pd.Series(
            nx.betweenness_centrality(G, weight=weight))
    except Exception:
        out["betweenness"] = float("nan")
    try:
        UG = G.to_undirected() if G.is_directed() else G
        out["eigenvector"] = pd.Series(
            nx.eigenvector_centrality(UG, weight=weight, max_iter=1000))
    except Exception:
        out["eigenvector"] = float("nan")
    try:
        out["pagerank"] = pd.Series(nx.pagerank(G, weight=weight))
    except Exception:
        out["pagerank"] = float("nan")
    if G.is_directed():
        try:
            hubs, auths = nx.hits(G, max_iter=1000)
            out["hits_hub"] = pd.Series(hubs)
            out["hits_authority"] = pd.Series(auths)
        except Exception:
            pass
    return out.sort_values("pagerank", ascending=False)
