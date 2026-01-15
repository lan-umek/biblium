# -*- coding: utf-8 -*-
"""
Data Quality Module - Author disambiguation, duplicate detection, and data cleaning.

This module provides tools for:
- Author name disambiguation (matching name variants)
- Duplicate document detection
- Missing data analysis and reporting
- Data cleaning suggestions
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


# =============================================================================
# AUTHOR NAME DISAMBIGUATION
# =============================================================================

@dataclass
class AuthorCluster:
    """Represents a cluster of author name variants."""
    canonical_name: str
    variants: Set[str] = field(default_factory=set)
    document_indices: Set[int] = field(default_factory=set)
    affiliations: Set[str] = field(default_factory=set)
    orcids: Set[str] = field(default_factory=set)
    
    @property
    def total_documents(self) -> int:
        return len(self.document_indices)
    
    @property
    def all_names(self) -> Set[str]:
        return self.variants | {self.canonical_name}


def normalize_author_name(name: str) -> str:
    """
    Normalize an author name for comparison.
    
    - Lowercase
    - Remove accents/diacritics
    - Normalize whitespace
    - Remove punctuation except hyphens
    - Handle "Last, First" vs "First Last" formats
    
    Parameters
    ----------
    name : str
        Raw author name.
        
    Returns
    -------
    str
        Normalized name.
    """
    if not name or not isinstance(name, str):
        return ""
    
    # Lowercase
    name = name.lower().strip()
    
    # Remove accents
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )
    
    # Normalize unicode
    name = unicodedata.normalize("NFKC", name)
    
    # Remove punctuation except hyphens and spaces
    name = re.sub(r"[^\w\s-]", "", name)
    
    # Normalize whitespace
    name = " ".join(name.split())
    
    return name


def parse_author_name(name: str) -> Tuple[str, str, str]:
    """
    Parse author name into components.
    
    Parameters
    ----------
    name : str
        Author name in various formats.
        
    Returns
    -------
    tuple
        (last_name, first_name, initials)
    """
    if not name:
        return ("", "", "")
    
    name = name.strip()
    
    # Handle "Last, First" format
    if "," in name:
        parts = name.split(",", 1)
        last = parts[0].strip()
        first_part = parts[1].strip() if len(parts) > 1 else ""
    else:
        # Handle "First Last" format
        parts = name.split()
        if len(parts) == 1:
            return (parts[0], "", "")
        elif len(parts) == 2:
            # Could be "First Last" or "Last First-Initial"
            if len(parts[1]) <= 2 or parts[1].endswith("."):
                # Likely "Last Initial"
                last = parts[0]
                first_part = parts[1]
            else:
                # Likely "First Last"
                last = parts[-1]
                first_part = " ".join(parts[:-1])
        else:
            # Multiple parts - assume last word is surname
            last = parts[-1]
            first_part = " ".join(parts[:-1])
    
    # Extract initials
    initials = "".join(
        w[0].upper() for w in first_part.split() 
        if w and w[0].isalpha()
    )
    
    # Get first name (first word of first_part)
    first_words = first_part.split()
    first = first_words[0] if first_words else ""
    
    return (last.lower(), first.lower(), initials.upper())


def compute_author_similarity(
    name1: str,
    name2: str,
    method: str = "hybrid",
) -> float:
    """
    Compute similarity between two author names.
    
    Parameters
    ----------
    name1, name2 : str
        Author names to compare.
    method : str
        Similarity method: "exact", "initials", "fuzzy", "hybrid"
        
    Returns
    -------
    float
        Similarity score between 0 and 1.
    """
    if not name1 or not name2:
        return 0.0
    
    # Parse names
    last1, first1, init1 = parse_author_name(name1)
    last2, first2, init2 = parse_author_name(name2)
    
    # Normalize for comparison
    norm1 = normalize_author_name(name1)
    norm2 = normalize_author_name(name2)
    
    if method == "exact":
        return 1.0 if norm1 == norm2 else 0.0
    
    elif method == "initials":
        # Match if last names match and initials are compatible
        if last1 != last2:
            return 0.0
        
        # Check initial compatibility
        if init1 and init2:
            # Check if one is prefix of other
            shorter = min(init1, init2, key=len)
            longer = max(init1, init2, key=len)
            if longer.startswith(shorter):
                return 0.9
            elif init1[0] == init2[0]:
                return 0.7
            else:
                return 0.0
        elif init1 or init2:
            return 0.8  # One has initials, one doesn't
        else:
            return 0.9  # Both have no initials
    
    elif method == "fuzzy":
        if not FUZZY_AVAILABLE:
            # Fallback to simple ratio
            return _simple_similarity(norm1, norm2)
        
        # Use fuzzy matching
        ratio = fuzz.ratio(norm1, norm2) / 100.0
        token_ratio = fuzz.token_sort_ratio(norm1, norm2) / 100.0
        return max(ratio, token_ratio)
    
    elif method == "hybrid":
        # Combine multiple methods
        scores = []
        
        # Last name must match (with fuzzy tolerance)
        if last1 == last2:
            scores.append(1.0)
        elif FUZZY_AVAILABLE:
            last_sim = fuzz.ratio(last1, last2) / 100.0
            if last_sim < 0.8:
                return 0.0  # Last names too different
            scores.append(last_sim)
        else:
            if _simple_similarity(last1, last2) < 0.8:
                return 0.0
            scores.append(_simple_similarity(last1, last2))
        
        # Check first name / initials
        if first1 and first2:
            if first1 == first2:
                scores.append(1.0)
            elif first1[0] == first2[0]:
                # Same initial
                if FUZZY_AVAILABLE:
                    scores.append(0.5 + 0.5 * fuzz.ratio(first1, first2) / 100.0)
                else:
                    scores.append(0.7)
            else:
                scores.append(0.0)
        elif init1 and init2:
            if init1[0] == init2[0]:
                scores.append(0.8)
            else:
                scores.append(0.0)
        else:
            scores.append(0.5)  # Can't compare first names
        
        return sum(scores) / len(scores)
    
    return 0.0


def _simple_similarity(s1: str, s2: str) -> float:
    """Simple character-based similarity (fallback when fuzzywuzzy unavailable)."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    
    # Jaccard similarity on character bigrams
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s)-1))
    
    b1, b2 = bigrams(s1), bigrams(s2)
    if not b1 or not b2:
        return 0.0
    
    intersection = len(b1 & b2)
    union = len(b1 | b2)
    return intersection / union if union > 0 else 0.0


def disambiguate_authors(
    df: pd.DataFrame,
    author_column: str = "Authors",
    separator: str = "; ",
    affiliation_column: Optional[str] = None,
    orcid_column: Optional[str] = None,
    similarity_threshold: float = 0.85,
    method: str = "hybrid",
    use_affiliations: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, AuthorCluster], pd.DataFrame]:
    """
    Disambiguate author names across a bibliographic dataset.
    
    Groups author name variants that likely refer to the same person
    based on name similarity, affiliations, and ORCID identifiers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataset.
    author_column : str
        Column containing author names.
    separator : str
        Separator between authors in the column.
    affiliation_column : str, optional
        Column containing affiliations (helps disambiguation).
    orcid_column : str, optional
        Column containing ORCID identifiers.
    similarity_threshold : float
        Minimum similarity score to consider names as variants (0-1).
    method : str
        Similarity method: "exact", "initials", "fuzzy", "hybrid"
    use_affiliations : bool
        If True, use affiliations to help disambiguation.
    verbose : bool
        If True, print progress information.
        
    Returns
    -------
    tuple
        (clusters_dict, mapping_df)
        - clusters_dict: Dict mapping canonical name to AuthorCluster
        - mapping_df: DataFrame with columns [original_name, canonical_name, cluster_id]
    """
    if author_column not in df.columns:
        raise ValueError(f"Column '{author_column}' not found in DataFrame")
    
    # Extract all unique author names with their document indices
    author_docs = defaultdict(set)  # author -> set of doc indices
    author_affiliations = defaultdict(set)  # author -> set of affiliations
    author_orcids = defaultdict(set)  # author -> set of orcids
    
    for idx, row in df.iterrows():
        if pd.isna(row[author_column]):
            continue
        
        authors = str(row[author_column]).split(separator)
        
        # Get affiliations if available
        affiliations = []
        if affiliation_column and affiliation_column in df.columns and pd.notna(row.get(affiliation_column)):
            affiliations = str(row[affiliation_column]).split(separator)
        
        # Get ORCIDs if available
        orcids = []
        if orcid_column and orcid_column in df.columns and pd.notna(row.get(orcid_column)):
            orcids = str(row[orcid_column]).split(separator)
        
        for i, author in enumerate(authors):
            author = author.strip()
            if not author:
                continue
            
            author_docs[author].add(idx)
            
            if i < len(affiliations):
                author_affiliations[author].add(affiliations[i].strip())
            
            if i < len(orcids):
                orcid = orcids[i].strip()
                if orcid and orcid.lower() not in ("", "nan", "none"):
                    author_orcids[author].add(orcid)
    
    unique_authors = list(author_docs.keys())
    n_authors = len(unique_authors)
    
    if verbose:
        print(f"Found {n_authors} unique author names")
    
    # Build clusters using Union-Find approach
    parent = {a: a for a in unique_authors}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            # Use the one with more documents as parent
            if len(author_docs[px]) >= len(author_docs[py]):
                parent[py] = px
            else:
                parent[px] = py
    
    # First pass: group by ORCID (definitive match)
    orcid_to_authors = defaultdict(list)
    for author, orcids in author_orcids.items():
        for orcid in orcids:
            orcid_to_authors[orcid].append(author)
    
    for orcid, authors in orcid_to_authors.items():
        for i in range(1, len(authors)):
            union(authors[0], authors[i])
    
    if verbose:
        print(f"Grouped {sum(len(a) for a in orcid_to_authors.values() if len(a) > 1)} authors by ORCID")
    
    # Second pass: group by name similarity
    # Sort by normalized last name for efficiency
    sorted_authors = sorted(unique_authors, key=lambda a: parse_author_name(a)[0])
    
    comparisons = 0
    merges = 0
    
    for i, author1 in enumerate(sorted_authors):
        if verbose and i % 500 == 0 and i > 0:
            print(f"  Processing {i}/{n_authors} authors...")
        
        last1, _, _ = parse_author_name(author1)
        
        # Only compare with authors having similar last names
        for j in range(i + 1, len(sorted_authors)):
            author2 = sorted_authors[j]
            last2, _, _ = parse_author_name(author2)
            
            # Early termination if last names are too different
            if last1 and last2 and last1[0] != last2[0]:
                # Last names start with different letters - skip ahead
                if last2 > last1:
                    break
                continue
            
            # Skip if already in same cluster
            if find(author1) == find(author2):
                continue
            
            comparisons += 1
            
            # Compute similarity
            sim = compute_author_similarity(author1, author2, method=method)
            
            # Boost similarity if affiliations match
            if use_affiliations and sim >= similarity_threshold - 0.1:
                aff1 = author_affiliations.get(author1, set())
                aff2 = author_affiliations.get(author2, set())
                if aff1 and aff2:
                    # Check for affiliation overlap
                    aff_overlap = any(
                        _simple_similarity(a1.lower(), a2.lower()) > 0.8
                        for a1 in aff1 for a2 in aff2
                    )
                    if aff_overlap:
                        sim = min(1.0, sim + 0.1)
            
            if sim >= similarity_threshold:
                union(author1, author2)
                merges += 1
    
    if verbose:
        print(f"Made {comparisons} comparisons, {merges} merges")
    
    # Build final clusters
    clusters = defaultdict(lambda: AuthorCluster(""))
    
    for author in unique_authors:
        root = find(author)
        cluster = clusters[root]
        
        if not cluster.canonical_name:
            cluster.canonical_name = root
        
        cluster.variants.add(author)
        cluster.document_indices.update(author_docs[author])
        cluster.affiliations.update(author_affiliations.get(author, set()))
        cluster.orcids.update(author_orcids.get(author, set()))
    
    # Choose best canonical name for each cluster (most frequent)
    for root, cluster in clusters.items():
        if len(cluster.variants) > 1:
            # Choose the variant that appears in most documents
            best_name = max(
                cluster.variants,
                key=lambda v: len(author_docs[v])
            )
            cluster.canonical_name = best_name
    
    # Create mapping DataFrame
    mapping_records = []
    cluster_id = 0
    
    clusters_dict = {}
    for root, cluster in clusters.items():
        clusters_dict[cluster.canonical_name] = cluster
        
        for variant in cluster.variants:
            mapping_records.append({
                "original_name": variant,
                "canonical_name": cluster.canonical_name,
                "cluster_id": cluster_id,
                "cluster_size": len(cluster.variants),
                "total_documents": cluster.total_documents,
            })
        
        cluster_id += 1
    
    mapping_df = pd.DataFrame(mapping_records)
    
    # Statistics
    n_clusters = len(clusters_dict)
    n_with_variants = sum(1 for c in clusters_dict.values() if len(c.variants) > 1)
    
    if verbose:
        print(f"\nDisambiguation Results:")
        print(f"  Total unique names: {n_authors}")
        print(f"  Distinct authors (clusters): {n_clusters}")
        print(f"  Authors with variants: {n_with_variants}")
        print(f"  Reduction: {n_authors - n_clusters} names merged ({100*(n_authors-n_clusters)/n_authors:.1f}%)")
    
    return clusters_dict, mapping_df


def apply_author_disambiguation(
    df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    author_column: str = "Authors",
    separator: str = "; ",
    output_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply author disambiguation mapping to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    mapping_df : pd.DataFrame
        Mapping from disambiguate_authors().
    author_column : str
        Column containing author names.
    separator : str
        Separator between authors.
    output_column : str, optional
        Name for output column. If None, creates "Disambiguated Authors".
        
    Returns
    -------
    pd.DataFrame
        DataFrame with disambiguated author column.
    """
    if output_column is None:
        output_column = f"Disambiguated {author_column}"
    
    # Build mapping dict
    name_map = dict(zip(mapping_df["original_name"], mapping_df["canonical_name"]))
    
    def disambiguate_row(authors_str):
        if pd.isna(authors_str):
            return authors_str
        
        authors = str(authors_str).split(separator)
        disambiguated = []
        seen = set()
        
        for author in authors:
            author = author.strip()
            canonical = name_map.get(author, author)
            
            # Avoid duplicates after disambiguation
            if canonical not in seen:
                disambiguated.append(canonical)
                seen.add(canonical)
        
        return separator.join(disambiguated)
    
    df = df.copy()
    df[output_column] = df[author_column].apply(disambiguate_row)
    
    return df


def get_author_variants_report(
    clusters_dict: Dict[str, AuthorCluster],
    min_variants: int = 2,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a report of author name variants.
    
    Parameters
    ----------
    clusters_dict : dict
        Output from disambiguate_authors().
    min_variants : int
        Minimum number of variants to include.
    top_n : int, optional
        Limit to top N authors by document count.
        
    Returns
    -------
    pd.DataFrame
        Report with columns: canonical_name, variants, n_variants, n_documents
    """
    records = []
    
    for canonical, cluster in clusters_dict.items():
        if len(cluster.variants) >= min_variants:
            records.append({
                "canonical_name": canonical,
                "variants": "; ".join(sorted(cluster.variants - {canonical})),
                "n_variants": len(cluster.variants),
                "n_documents": cluster.total_documents,
                "affiliations": "; ".join(list(cluster.affiliations)[:3]),
                "orcids": "; ".join(cluster.orcids),
            })
    
    report_df = pd.DataFrame(records)
    
    if len(report_df) > 0:
        report_df = report_df.sort_values("n_documents", ascending=False)
        
        if top_n:
            report_df = report_df.head(top_n)
    
    return report_df.reset_index(drop=True)


# =============================================================================
# DUPLICATE DETECTION
# =============================================================================

def detect_duplicate_documents(
    df: pd.DataFrame,
    title_column: str = "Title",
    doi_column: Optional[str] = "DOI",
    year_column: Optional[str] = "Year",
    author_column: Optional[str] = "Authors",
    similarity_threshold: float = 0.90,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Set[int]]]:
    """
    Detect potential duplicate documents in a bibliographic dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataset.
    title_column : str
        Column containing document titles.
    doi_column : str, optional
        Column containing DOIs (exact match).
    year_column : str, optional
        Column containing publication years (must match for duplicates).
    author_column : str, optional
        Column containing authors (used for verification).
    similarity_threshold : float
        Minimum title similarity to consider as duplicate.
    verbose : bool
        If True, print progress.
        
    Returns
    -------
    tuple
        (duplicates_df, duplicate_groups)
        - duplicates_df: DataFrame with duplicate pairs and similarity scores
        - duplicate_groups: List of sets, each set contains indices of duplicate documents
    """
    n = len(df)
    
    if verbose:
        print(f"Checking {n} documents for duplicates...")
    
    # First pass: exact DOI matches
    doi_duplicates = []
    if doi_column and doi_column in df.columns:
        doi_groups = df.groupby(df[doi_column].str.lower().str.strip()).groups
        for doi, indices in doi_groups.items():
            if pd.notna(doi) and doi and len(indices) > 1:
                doi_duplicates.append(set(indices))
    
    if verbose:
        print(f"  Found {len(doi_duplicates)} DOI-based duplicate groups")
    
    # Second pass: title similarity
    title_duplicates = []
    
    # Normalize titles for comparison
    def normalize_title(t):
        if pd.isna(t):
            return ""
        t = str(t).lower()
        t = re.sub(r"[^\w\s]", "", t)
        return " ".join(t.split())
    
    titles = df[title_column].apply(normalize_title).tolist()
    years = df[year_column].tolist() if year_column and year_column in df.columns else [None] * n
    
    # Group by year for efficiency
    year_indices = defaultdict(list)
    for i, year in enumerate(years):
        year_indices[year].append(i)
    
    checked_pairs = set()
    
    for year, indices in year_indices.items():
        if len(indices) < 2:
            continue
        
        for i, idx1 in enumerate(indices):
            title1 = titles[idx1]
            if not title1:
                continue
            
            for idx2 in indices[i+1:]:
                if (idx1, idx2) in checked_pairs or (idx2, idx1) in checked_pairs:
                    continue
                
                checked_pairs.add((idx1, idx2))
                
                title2 = titles[idx2]
                if not title2:
                    continue
                
                # Quick length check
                len_ratio = min(len(title1), len(title2)) / max(len(title1), len(title2))
                if len_ratio < 0.7:
                    continue
                
                # Compute similarity
                if FUZZY_AVAILABLE:
                    sim = fuzz.ratio(title1, title2) / 100.0
                else:
                    sim = _simple_similarity(title1, title2)
                
                if sim >= similarity_threshold:
                    title_duplicates.append({
                        "index1": idx1,
                        "index2": idx2,
                        "title1": df.iloc[idx1][title_column],
                        "title2": df.iloc[idx2][title_column],
                        "similarity": sim,
                        "year": year,
                    })
    
    if verbose:
        print(f"  Found {len(title_duplicates)} title-based duplicate pairs")
    
    # Merge duplicate groups
    # Use Union-Find
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px
    
    # Apply DOI duplicates
    for group in doi_duplicates:
        indices = list(group)
        for i in range(1, len(indices)):
            union(indices[0], indices[i])
    
    # Apply title duplicates
    for dup in title_duplicates:
        union(dup["index1"], dup["index2"])
    
    # Build final groups
    groups = defaultdict(set)
    for i in range(n):
        root = find(i)
        groups[root].add(i)
    
    duplicate_groups = [g for g in groups.values() if len(g) > 1]
    
    # Create duplicates DataFrame
    duplicates_df = pd.DataFrame(title_duplicates)
    
    if verbose:
        n_duplicates = sum(len(g) - 1 for g in duplicate_groups)
        print(f"\nDuplicate Detection Results:")
        print(f"  Total documents: {n}")
        print(f"  Duplicate groups: {len(duplicate_groups)}")
        print(f"  Documents to remove: {n_duplicates}")
    
    return duplicates_df, duplicate_groups


def remove_duplicates(
    df: pd.DataFrame,
    duplicate_groups: List[Set[int]],
    keep: str = "first",
    citation_column: Optional[str] = "Cited by",
) -> pd.DataFrame:
    """
    Remove duplicate documents from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    duplicate_groups : list
        Output from detect_duplicate_documents().
    keep : str
        Which duplicate to keep: "first", "last", "most_cited"
    citation_column : str, optional
        Column for citations (used when keep="most_cited").
        
    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    indices_to_remove = set()
    
    for group in duplicate_groups:
        group_list = sorted(group)
        
        if keep == "first":
            # Keep first, remove rest
            indices_to_remove.update(group_list[1:])
        elif keep == "last":
            # Keep last, remove rest
            indices_to_remove.update(group_list[:-1])
        elif keep == "most_cited" and citation_column and citation_column in df.columns:
            # Keep most cited
            citations = [(i, df.iloc[i].get(citation_column, 0)) for i in group_list]
            citations.sort(key=lambda x: x[1] if pd.notna(x[1]) else 0, reverse=True)
            indices_to_remove.update(i for i, _ in citations[1:])
        else:
            # Default to first
            indices_to_remove.update(group_list[1:])
    
    return df.drop(index=list(indices_to_remove)).reset_index(drop=True)


# =============================================================================
# MISSING DATA ANALYSIS
# =============================================================================

def analyze_missing_data(
    df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Analyze missing data in a bibliographic dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataset.
    key_columns : list, optional
        Specific columns to analyze. If None, analyzes all.
        
    Returns
    -------
    pd.DataFrame
        Report with missing data statistics per column.
    """
    if key_columns is None:
        key_columns = df.columns.tolist()
    
    records = []
    n = len(df)
    
    for col in key_columns:
        if col not in df.columns:
            continue
        
        missing = df[col].isna().sum()
        empty = (df[col].astype(str).str.strip() == "").sum()
        
        records.append({
            "Column": col,
            "Missing (NaN)": missing,
            "Empty String": empty,
            "Total Missing": missing + empty,
            "% Missing": round(100 * (missing + empty) / n, 2) if n > 0 else 0,
            "% Complete": round(100 * (n - missing - empty) / n, 2) if n > 0 else 0,
            "Unique Values": df[col].nunique(),
            "Sample Value": df[col].dropna().iloc[0] if df[col].notna().any() else None,
        })
    
    report = pd.DataFrame(records)
    report = report.sort_values("% Missing", ascending=False)
    
    return report.reset_index(drop=True)


def get_data_quality_score(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate overall data quality score.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataset.
    weights : dict, optional
        Column weights for scoring. Default weights key bibliometric columns higher.
        
    Returns
    -------
    tuple
        (overall_score, details_df)
        - overall_score: Float between 0-100
        - details_df: Breakdown by column
    """
    if weights is None:
        weights = {
            "Title": 3.0,
            "Authors": 3.0,
            "Year": 2.0,
            "Source title": 2.0,
            "DOI": 1.5,
            "Abstract": 1.0,
            "Author Keywords": 1.0,
            "Cited by": 1.0,
        }
    
    n = len(df)
    records = []
    total_weight = 0
    weighted_score = 0
    
    for col in df.columns:
        weight = weights.get(col, 0.5)
        
        # Calculate completeness
        missing = df[col].isna().sum()
        empty = (df[col].astype(str).str.strip() == "").sum()
        completeness = 100 * (n - missing - empty) / n if n > 0 else 0
        
        col_score = completeness * weight
        weighted_score += col_score
        total_weight += weight
        
        if weight >= 1.0:  # Only report important columns
            records.append({
                "Column": col,
                "Weight": weight,
                "Completeness %": round(completeness, 1),
                "Weighted Score": round(col_score, 2),
            })
    
    overall_score = weighted_score / total_weight if total_weight > 0 else 0
    
    details_df = pd.DataFrame(records).sort_values("Weight", ascending=False)
    
    return round(overall_score, 1), details_df.reset_index(drop=True)
