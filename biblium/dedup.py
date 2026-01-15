# -*- coding: utf-8 -*-
"""
Deduplication Module for Biblium

Detects and merges duplicate papers across multiple databases.
Primary matching by DOI, with fallback to title similarity.

Merge Strategy:
- Citations: max(all sources), but keep Cited_by_<source> columns
- Abstract: longest non-truncated
- Authors: from highest priority source (best formatting)
- Keywords: union of all sources
- References: union of all sources
- Other fields: from highest priority source

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Union

import pandas as pd
import numpy as np

# Logging
try:
    from biblium.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default source priority (higher = more trusted for formatting)
DEFAULT_SOURCE_PRIORITY = {
    "scopus": 100,
    "wos": 90,
    "web of science": 90,
    "dimensions": 80,
    "openalex": 70,
    "open alex": 70,
    "oa": 70,
    "lens": 60,
    "pubmed": 50,
    "crossref": 40,
    "semantic scholar": 30,
    "unknown": 0,
}

# Fields to keep source-specific versions
KEEP_SOURCE_COLUMNS = [
    "Cited by",
    "References",
]

# Fields to merge by union (multi-value)
UNION_FIELDS = [
    "Author Keywords",
    "Index Keywords",
    "Keywords",
    "References",
]

# Fields to keep longest non-empty
LONGEST_FIELDS = [
    "Abstract",
    "Affiliations",
]

# Fields where max value is best
MAX_FIELDS = [
    "Cited by",
]


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class DeduplicationResult:
    """Result of deduplication process."""
    
    # Main output
    df: pd.DataFrame
    
    # Statistics
    total_input: int = 0
    total_output: int = 0
    duplicates_found: int = 0
    duplicates_by_doi: int = 0
    duplicates_by_title: int = 0
    
    # Tracking
    sources: List[str] = field(default_factory=list)
    merge_log: List[Dict] = field(default_factory=list)
    
    def summary(self) -> str:
        """Return summary string."""
        reduction = (1 - self.total_output / self.total_input) * 100 if self.total_input > 0 else 0
        return (
            f"Deduplication Result:\n"
            f"  Input records: {self.total_input}\n"
            f"  Output records: {self.total_output}\n"
            f"  Duplicates removed: {self.duplicates_found} ({reduction:.1f}%)\n"
            f"    - By DOI: {self.duplicates_by_doi}\n"
            f"    - By title: {self.duplicates_by_title}\n"
            f"  Sources: {', '.join(self.sources)}"
        )
    
    def __repr__(self) -> str:
        return self.summary()
    
    def get_merge_log_df(self) -> pd.DataFrame:
        """Get merge log as DataFrame."""
        return pd.DataFrame(self.merge_log)


# =============================================================================
# DOI NORMALIZATION
# =============================================================================

def normalize_doi(doi: Any) -> Optional[str]:
    """
    Normalize DOI to standard format.
    
    Handles various DOI formats:
    - 10.1234/abc
    - https://doi.org/10.1234/abc
    - doi:10.1234/abc
    - DOI: 10.1234/abc
    
    Returns
    -------
    str or None
        Normalized DOI (lowercase, no prefix) or None if invalid.
    """
    if pd.isna(doi) or not doi:
        return None
    
    doi = str(doi).strip().lower()
    
    # Remove common prefixes
    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "doi:",
        "doi ",
    ]
    
    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    
    # Validate DOI format (starts with 10.)
    if doi.startswith("10."):
        return doi.strip()
    
    return None


def normalize_title(title: Any) -> Optional[str]:
    """
    Normalize title for comparison.
    
    Removes punctuation, extra spaces, converts to lowercase.
    """
    if pd.isna(title) or not title:
        return None
    
    title = str(title).lower().strip()
    
    # Remove punctuation
    title = re.sub(r'[^\w\s]', '', title)
    
    # Normalize whitespace
    title = ' '.join(title.split())
    
    # Must have minimum length
    if len(title) < 10:
        return None
    
    return title


# =============================================================================
# CORE DEDUPLICATION
# =============================================================================

def find_duplicates_by_doi(
    df: pd.DataFrame,
    doi_col: str = "DOI",
) -> Dict[str, List[int]]:
    """
    Find duplicates by DOI.
    
    Returns
    -------
    dict
        Mapping of normalized DOI to list of row indices.
    """
    doi_groups = defaultdict(list)
    
    for idx, row in df.iterrows():
        doi = normalize_doi(row.get(doi_col))
        if doi:
            doi_groups[doi].append(idx)
    
    # Keep only groups with duplicates
    return {doi: indices for doi, indices in doi_groups.items() if len(indices) > 1}


def find_duplicates_by_title(
    df: pd.DataFrame,
    title_col: str = "Title",
    exclude_indices: Set[int] = None,
    similarity_threshold: float = 0.95,
) -> Dict[str, List[int]]:
    """
    Find duplicates by title similarity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    title_col : str
        Title column name.
    exclude_indices : set
        Indices already matched by DOI (skip these).
    similarity_threshold : float
        Minimum similarity ratio (0-1) to consider duplicate.
    
    Returns
    -------
    dict
        Mapping of normalized title to list of row indices.
    """
    if exclude_indices is None:
        exclude_indices = set()
    
    title_groups = defaultdict(list)
    
    for idx, row in df.iterrows():
        if idx in exclude_indices:
            continue
        
        title = normalize_title(row.get(title_col))
        if title:
            title_groups[title].append(idx)
    
    # Keep only groups with duplicates
    return {title: indices for title, indices in title_groups.items() if len(indices) > 1}


def get_source_priority(source: Any) -> int:
    """Get priority score for a source."""
    if pd.isna(source):
        return 0
    
    source = str(source).lower().strip()
    
    for key, priority in DEFAULT_SOURCE_PRIORITY.items():
        if key in source:
            return priority
    
    return 0


def merge_records(
    records: List[Dict],
    source_col: str = "Source_DB",
    separator: str = "; ",
) -> Dict[str, Any]:
    """
    Merge multiple records into one.
    
    Strategy:
    - Citations: max, but keep per-source columns
    - Abstract: longest
    - Keywords: union
    - Other fields: from highest priority source
    
    Parameters
    ----------
    records : list
        List of record dictionaries.
    source_col : str
        Column indicating source database.
    separator : str
        Separator for multi-value fields.
    
    Returns
    -------
    dict
        Merged record.
    """
    if len(records) == 1:
        merged = records[0].copy()
        merged["_sources"] = merged.get(source_col, "unknown")
        merged["_duplicate_count"] = 1
        return merged
    
    # Sort by source priority (highest first)
    sorted_records = sorted(
        records,
        key=lambda r: get_source_priority(r.get(source_col)),
        reverse=True
    )
    
    # Start with highest priority record
    merged = sorted_records[0].copy()
    sources = []
    
    for rec in records:
        src = str(rec.get(source_col, "unknown")).strip()
        if src and src not in sources:
            sources.append(src)
    
    # Process special fields
    for rec in sorted_records:
        src = str(rec.get(source_col, "unknown")).lower().replace(" ", "_")
        
        # Keep source-specific citation columns
        for col in KEEP_SOURCE_COLUMNS:
            if col in rec and pd.notna(rec[col]):
                source_col_name = f"{col}_{src}"
                if source_col_name not in merged or pd.isna(merged.get(source_col_name)):
                    merged[source_col_name] = rec[col]
    
    # MAX fields (citations)
    for col in MAX_FIELDS:
        values = []
        for rec in records:
            val = rec.get(col)
            if pd.notna(val):
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass
        if values:
            merged[col] = int(max(values))
    
    # LONGEST fields (abstract)
    for col in LONGEST_FIELDS:
        longest = ""
        for rec in records:
            val = rec.get(col)
            if pd.notna(val) and len(str(val)) > len(longest):
                longest = str(val)
        if longest:
            merged[col] = longest
    
    # UNION fields (keywords)
    for col in UNION_FIELDS:
        all_items = set()
        for rec in records:
            val = rec.get(col)
            if pd.notna(val):
                items = [i.strip() for i in str(val).split(separator) if i.strip()]
                all_items.update(items)
        if all_items:
            merged[col] = separator.join(sorted(all_items))
    
    # Add metadata
    merged["_sources"] = "; ".join(sources)
    merged["_duplicate_count"] = len(records)
    
    return merged


# =============================================================================
# MAIN DEDUPLICATION FUNCTION
# =============================================================================

def deduplicate(
    *dataframes: pd.DataFrame,
    source_names: List[str] = None,
    doi_col: str = "DOI",
    title_col: str = "Title",
    use_title_matching: bool = True,
    separator: str = "; ",
    verbose: bool = True,
) -> DeduplicationResult:
    """
    Deduplicate and merge records from multiple DataFrames.
    
    Parameters
    ----------
    *dataframes : pd.DataFrame
        One or more DataFrames to merge and deduplicate.
    source_names : list
        Names for each source (e.g., ["scopus", "wos", "openalex"]).
        If None, uses "source_1", "source_2", etc.
    doi_col : str
        Column containing DOI.
    title_col : str
        Column containing title.
    use_title_matching : bool
        Whether to match by title when DOI is missing.
    separator : str
        Separator for multi-value fields.
    verbose : bool
        Print progress.
    
    Returns
    -------
    DeduplicationResult
        Result containing merged DataFrame and statistics.
    
    Example
    -------
    >>> from biblium.dedup import deduplicate
    >>> 
    >>> # Load data from different sources
    >>> df_scopus = pd.read_csv("scopus_export.csv")
    >>> df_wos = pd.read_csv("wos_export.csv")
    >>> df_oa = pd.read_csv("openalex_export.csv")
    >>> 
    >>> # Deduplicate
    >>> result = deduplicate(
    ...     df_scopus, df_wos, df_oa,
    ...     source_names=["scopus", "wos", "openalex"]
    ... )
    >>> 
    >>> print(result)
    >>> merged_df = result.df
    """
    if len(dataframes) == 0:
        raise ValueError("At least one DataFrame required")
    
    # Set default source names
    if source_names is None:
        source_names = [f"source_{i+1}" for i in range(len(dataframes))]
    
    if len(source_names) != len(dataframes):
        raise ValueError("source_names must match number of dataframes")
    
    if verbose:
        logger.info(f"Deduplicating {len(dataframes)} sources: {source_names}")
    
    # Combine all DataFrames with source indicator
    combined_records = []
    
    for df, source_name in zip(dataframes, source_names):
        for _, row in df.iterrows():
            record = row.to_dict()
            record["Source_DB"] = source_name
            combined_records.append(record)
    
    total_input = len(combined_records)
    if verbose:
        logger.info(f"Total input records: {total_input}")
    
    # Create combined DataFrame for duplicate detection
    combined_df = pd.DataFrame(combined_records)
    
    # Find duplicates by DOI
    doi_duplicates = find_duplicates_by_doi(combined_df, doi_col)
    doi_matched_indices = set()
    for indices in doi_duplicates.values():
        doi_matched_indices.update(indices)
    
    duplicates_by_doi = sum(len(indices) - 1 for indices in doi_duplicates.values())
    
    if verbose:
        logger.info(f"Found {len(doi_duplicates)} DOI groups with duplicates ({duplicates_by_doi} duplicates)")
    
    # Find duplicates by title (excluding DOI-matched)
    title_duplicates = {}
    duplicates_by_title = 0
    
    if use_title_matching:
        title_duplicates = find_duplicates_by_title(
            combined_df, title_col,
            exclude_indices=doi_matched_indices
        )
        duplicates_by_title = sum(len(indices) - 1 for indices in title_duplicates.values())
        
        if verbose:
            logger.info(f"Found {len(title_duplicates)} title groups with duplicates ({duplicates_by_title} duplicates)")
    
    # Build merge groups
    # Each record belongs to at most one group
    record_to_group = {}
    groups = []
    
    # DOI groups
    for doi, indices in doi_duplicates.items():
        group_id = len(groups)
        groups.append({
            "type": "doi",
            "key": doi,
            "indices": indices,
        })
        for idx in indices:
            record_to_group[idx] = group_id
    
    # Title groups (only for records not in DOI groups)
    for title, indices in title_duplicates.items():
        # Filter out indices already in DOI groups
        remaining = [i for i in indices if i not in record_to_group]
        if len(remaining) > 1:
            group_id = len(groups)
            groups.append({
                "type": "title",
                "key": title[:50] + "..." if len(title) > 50 else title,
                "indices": remaining,
            })
            for idx in remaining:
                record_to_group[idx] = group_id
    
    # Merge records
    merged_records = []
    merge_log = []
    
    # Process groups
    for group in groups:
        group_records = [combined_records[i] for i in group["indices"]]
        merged = merge_records(group_records, separator=separator)
        merged_records.append(merged)
        
        merge_log.append({
            "type": group["type"],
            "key": group["key"],
            "sources": merged.get("_sources"),
            "count": len(group["indices"]),
        })
    
    # Add non-duplicate records
    for idx, record in enumerate(combined_records):
        if idx not in record_to_group:
            record = record.copy()
            record["_sources"] = record.get("Source_DB", "unknown")
            record["_duplicate_count"] = 1
            merged_records.append(record)
    
    # Create result DataFrame
    result_df = pd.DataFrame(merged_records)
    
    # Reorder columns: important columns first
    priority_cols = [
        "DOI", "Title", "Authors", "Year", "Source title",
        "Cited by", "Abstract", "_sources", "_duplicate_count"
    ]
    
    existing_priority = [c for c in priority_cols if c in result_df.columns]
    other_cols = [c for c in result_df.columns if c not in priority_cols]
    result_df = result_df[existing_priority + other_cols]
    
    total_output = len(result_df)
    duplicates_found = total_input - total_output
    
    if verbose:
        logger.info(f"Output records: {total_output} (removed {duplicates_found} duplicates)")
    
    return DeduplicationResult(
        df=result_df,
        total_input=total_input,
        total_output=total_output,
        duplicates_found=duplicates_found,
        duplicates_by_doi=duplicates_by_doi,
        duplicates_by_title=duplicates_by_title,
        sources=source_names,
        merge_log=merge_log,
    )


# =============================================================================
# SINGLE DATAFRAME DEDUPLICATION
# =============================================================================

def deduplicate_single(
    df: pd.DataFrame,
    doi_col: str = "DOI",
    title_col: str = "Title",
    use_title_matching: bool = True,
    keep: str = "first",
    verbose: bool = True,
) -> DeduplicationResult:
    """
    Deduplicate a single DataFrame (remove exact duplicates).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    doi_col : str
        DOI column.
    title_col : str
        Title column.
    use_title_matching : bool
        Match by title when DOI missing.
    keep : str
        Which duplicate to keep: "first", "last".
    verbose : bool
        Print progress.
    
    Returns
    -------
    DeduplicationResult
        Deduplicated result.
    """
    return deduplicate(
        df,
        source_names=["single"],
        doi_col=doi_col,
        title_col=title_col,
        use_title_matching=use_title_matching,
        verbose=verbose,
    )


# =============================================================================
# DUPLICATE DETECTION (WITHOUT MERGING)
# =============================================================================

def detect_duplicates(
    df: pd.DataFrame,
    doi_col: str = "DOI",
    title_col: str = "Title",
    use_title_matching: bool = True,
) -> pd.DataFrame:
    """
    Detect duplicates without merging - just flag them.
    
    Adds columns:
    - _is_duplicate: bool
    - _duplicate_group: group ID for duplicates
    - _duplicate_match_type: "doi" or "title"
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    doi_col : str
        DOI column.
    title_col : str
        Title column.
    use_title_matching : bool
        Match by title when DOI missing.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with duplicate flags added.
    """
    result = df.copy()
    result["_is_duplicate"] = False
    result["_duplicate_group"] = -1
    result["_duplicate_match_type"] = ""
    
    group_id = 0
    
    # DOI duplicates
    doi_groups = find_duplicates_by_doi(df, doi_col)
    for doi, indices in doi_groups.items():
        for idx in indices:
            result.loc[idx, "_is_duplicate"] = True
            result.loc[idx, "_duplicate_group"] = group_id
            result.loc[idx, "_duplicate_match_type"] = "doi"
        group_id += 1
    
    # Title duplicates
    if use_title_matching:
        doi_matched = set()
        for indices in doi_groups.values():
            doi_matched.update(indices)
        
        title_groups = find_duplicates_by_title(df, title_col, exclude_indices=doi_matched)
        for title, indices in title_groups.items():
            for idx in indices:
                result.loc[idx, "_is_duplicate"] = True
                result.loc[idx, "_duplicate_group"] = group_id
                result.loc[idx, "_duplicate_match_type"] = "title"
            group_id += 1
    
    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def merge_scopus_wos(
    scopus_df: pd.DataFrame,
    wos_df: pd.DataFrame,
    verbose: bool = True,
) -> DeduplicationResult:
    """
    Merge Scopus and Web of Science exports.
    
    Convenience function with optimized defaults for these two sources.
    """
    return deduplicate(
        scopus_df, wos_df,
        source_names=["scopus", "wos"],
        verbose=verbose,
    )


def merge_three_sources(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    names: List[str] = None,
    verbose: bool = True,
) -> DeduplicationResult:
    """
    Merge three database exports.
    
    Common use case: Scopus + WoS + OpenAlex
    """
    if names is None:
        names = ["source_1", "source_2", "source_3"]
    
    return deduplicate(df1, df2, df3, source_names=names, verbose=verbose)


# =============================================================================
# STATISTICS AND REPORTING
# =============================================================================

def get_source_overlap_matrix(
    result: DeduplicationResult,
) -> pd.DataFrame:
    """
    Get matrix showing overlap between sources.
    
    Parameters
    ----------
    result : DeduplicationResult
        Result from deduplicate().
    
    Returns
    -------
    pd.DataFrame
        Matrix with counts of shared records.
    """
    sources = result.sources
    matrix = pd.DataFrame(0, index=sources, columns=sources)
    
    for _, row in result.df.iterrows():
        record_sources = [s.strip() for s in str(row.get("_sources", "")).split(";")]
        for s1 in record_sources:
            if s1 in sources:
                for s2 in record_sources:
                    if s2 in sources:
                        matrix.loc[s1, s2] += 1
    
    return matrix


def get_unique_per_source(
    result: DeduplicationResult,
) -> Dict[str, int]:
    """
    Get count of records unique to each source.
    
    Returns
    -------
    dict
        Source name -> count of unique records.
    """
    unique_counts = {s: 0 for s in result.sources}
    
    for _, row in result.df.iterrows():
        sources = row.get("_sources", "")
        if ";" not in str(sources):
            # Only one source
            source = str(sources).strip()
            if source in unique_counts:
                unique_counts[source] += 1
    
    return unique_counts
