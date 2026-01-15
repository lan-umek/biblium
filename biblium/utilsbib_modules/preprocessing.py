# -*- coding: utf-8 -*-
"""
Database-specific preprocessing for bibliometric data.

This module provides preprocessing pipelines for different bibliographic databases:
- PubMed/MEDLINE
- Lens.org
- Dimensions
- Scopus (reference)
- OpenAlex (reference)
- Web of Science (reference)

Each preprocessor standardizes column names, handles database-specific formats,
and prepares data for unified analysis.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# =============================================================================
# STANDARD COLUMN NAMES
# =============================================================================

STANDARD_COLUMNS = {
    # Core identifiers
    "doc_id": ["DOI", "PMID", "EID", "Lens ID", "Publication ID", "id"],
    "doi": ["DOI", "doi"],
    "pmid": ["PMID", "PubMed ID", "pmid"],
    
    # Bibliographic info
    "title": ["Title", "title", "ArticleTitle", "TI"],
    "abstract": ["Abstract", "abstract", "AbstractText", "AB"],
    "year": ["Year", "publication_year", "PubYear", "Year Published", "PY", "DP"],
    "authors": ["Authors", "authors", "Author/s", "AU", "AuthorList"],
    "source": ["Source title", "Journal", "source_title", "Source Title", "JT", "SO"],
    
    # Keywords
    "author_keywords": ["Author Keywords", "Keywords", "keywords", "OT", "DE"],
    "index_keywords": ["Index Keywords", "MeSH Terms", "topics.display_name", "MH", "ID"],
    
    # Citations
    "cited_by": ["Cited by", "Times cited", "cited_by_count", "Scholarly Citations", "TC"],
    "references": ["References", "Cited References", "referenced_works", "Reference IDs", "CR"],
    
    # Affiliations
    "affiliations": ["Affiliations", "Authors Affiliations", "affiliations", "AD", "C1"],
    
    # Other
    "doc_type": ["Document Type", "Publication Type", "type", "PT", "DT"],
    "language": ["Language", "language", "LA"],
    "issn": ["ISSN", "ISSNs", "issn", "SN"],
    "volume": ["Volume", "volume", "VL", "VI"],
    "issue": ["Issue", "issue", "IS", "IP"],
    "pages": ["Pages", "Start Page", "Page start", "PG", "BP"],
    "publisher": ["Publisher", "publisher", "PB"],
    "open_access": ["Open Access", "is_open_access", "OA"],
    "fwci": ["fwci", "Field-Weighted Citation Impact", "FWCI"],
}


def find_column(df: pd.DataFrame, standard_name: str) -> Optional[str]:
    """
    Find the actual column name in DataFrame matching a standard name.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    standard_name : str
        Standard column name to look for.
        
    Returns
    -------
    str or None
        Actual column name if found, None otherwise.
    """
    if standard_name not in STANDARD_COLUMNS:
        # Try direct match
        if standard_name in df.columns:
            return standard_name
        return None
    
    candidates = STANDARD_COLUMNS[standard_name]
    for col in candidates:
        if col in df.columns:
            return col
    
    # Try case-insensitive match
    df_cols_lower = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in df_cols_lower:
            return df_cols_lower[col.lower()]
    
    return None


def standardize_columns(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Standardize column names to common format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column_mapping : dict, optional
        Custom mapping from current to standard names.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.
    """
    df = df.copy()
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Auto-map known columns
    rename_map = {}
    for standard, candidates in STANDARD_COLUMNS.items():
        for candidate in candidates:
            if candidate in df.columns and candidate != standard:
                # Only rename if standard name not already present
                if standard not in df.columns:
                    rename_map[candidate] = standard
                break
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


# =============================================================================
# PUBMED PREPROCESSING
# =============================================================================

def preprocess_pubmed(
    df: pd.DataFrame,
    separator: str = "; ",
    parse_date: bool = True,
) -> pd.DataFrame:
    """
    Preprocess PubMed/MEDLINE data for bibliometric analysis.
    
    Handles:
    - Author name standardization (Last, First format)
    - MeSH term extraction
    - Date parsing (DP field)
    - Affiliation parsing
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw PubMed data.
    separator : str
        Separator for multi-value fields.
    parse_date : bool
        If True, parse publication date to year.
        
    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    df = df.copy()
    
    # Standardize column names first
    df = standardize_columns(df)
    
    # Parse year from date field if needed
    if parse_date:
        # Find the year/date column (may be 'year', 'Year', 'DP', etc.)
        year_col = None
        for candidate in ["year", "Year", "DP", "PubDate", "Publication Date"]:
            if candidate in df.columns:
                year_col = candidate
                break
        
        if year_col and year_col in df.columns:
            # Check if it's already a clean year
            sample = df[year_col].dropna().iloc[0] if len(df[year_col].dropna()) > 0 else None
            if sample:
                sample_str = str(sample)
                if not sample_str.isdigit() or len(sample_str) != 4:
                    # Needs parsing
                    df["Year"] = df[year_col].apply(_extract_year_from_pubmed_date)
                else:
                    df["Year"] = pd.to_numeric(df[year_col], errors="coerce")
    
    # Standardize authors
    auth_col = None
    for candidate in ["authors", "Authors", "AU", "AuthorList"]:
        if candidate in df.columns:
            auth_col = candidate
            break
    
    if auth_col:
        df["Authors"] = df[auth_col].apply(
            lambda x: _standardize_pubmed_authors(x, separator)
        )
    
    # Parse MeSH terms (remove qualifiers, keep descriptors)
    mesh_col = find_column(df, "index_keywords")
    if mesh_col and mesh_col in df.columns:
        df["Index Keywords"] = df[mesh_col].apply(
            lambda x: _parse_mesh_terms(x, separator)
        )
    
    # Ensure numeric citations
    cite_col = find_column(df, "cited_by")
    if cite_col and cite_col in df.columns:
        df["Cited by"] = pd.to_numeric(df[cite_col], errors="coerce").fillna(0).astype(int)
    
    return df


def _extract_year_from_pubmed_date(date_str: str) -> Optional[int]:
    """Extract year from PubMed date format (e.g., '2023 Jan 15')."""
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # Try to find 4-digit year
    match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if match:
        return int(match.group())
    
    return None


def _standardize_pubmed_authors(authors: str, separator: str = "; ") -> str:
    """Standardize PubMed author format to 'Last, First' format."""
    if pd.isna(authors):
        return ""
    
    authors_str = str(authors).strip()
    if not authors_str:
        return ""
    
    # PubMed uses various separators
    parts = re.split(r'[;\n]', authors_str)
    
    standardized = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Already in "Last, First" format
        if "," in part:
            standardized.append(part)
        else:
            # "First Last" format - convert
            words = part.split()
            if len(words) >= 2:
                last = words[-1]
                first = " ".join(words[:-1])
                standardized.append(f"{last}, {first}")
            else:
                standardized.append(part)
    
    return separator.join(standardized)


def _parse_mesh_terms(mesh_str: str, separator: str = "; ") -> str:
    """Parse MeSH terms, removing qualifiers and keeping main descriptors."""
    if pd.isna(mesh_str):
        return ""
    
    mesh_str = str(mesh_str).strip()
    if not mesh_str:
        return ""
    
    terms = []
    for term in re.split(r'[;\n]', mesh_str):
        term = term.strip()
        if not term:
            continue
        
        # Remove asterisks (major topic markers)
        term = term.replace("*", "")
        
        # Remove qualifier (after /)
        if "/" in term:
            term = term.split("/")[0].strip()
        
        if term and term not in terms:
            terms.append(term)
    
    return separator.join(terms)


# =============================================================================
# LENS.ORG PREPROCESSING
# =============================================================================

def preprocess_lens(
    df: pd.DataFrame,
    separator: str = "; ",
) -> pd.DataFrame:
    """
    Preprocess Lens.org data for bibliometric analysis.
    
    Handles:
    - Author name parsing from JSON-like format
    - Affiliation extraction
    - Fields of study normalization
    - Citation count standardization
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw Lens.org data.
    separator : str
        Separator for multi-value fields.
        
    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    df = df.copy()
    
    # Standardize column names
    df = standardize_columns(df)
    
    # Parse authors
    auth_col = find_column(df, "authors")
    if auth_col and auth_col in df.columns:
        df["Authors"] = df[auth_col].apply(
            lambda x: _parse_lens_authors(x, separator)
        )
    
    # Parse affiliations
    aff_col = find_column(df, "affiliations")
    if aff_col and aff_col in df.columns:
        df["Affiliations"] = df[aff_col].apply(
            lambda x: _parse_lens_affiliations(x, separator)
        )
    
    # Standardize year
    year_col = find_column(df, "year")
    if year_col and year_col in df.columns:
        df["Year"] = pd.to_numeric(df[year_col], errors="coerce")
    
    # Standardize citations
    cite_col = find_column(df, "cited_by")
    if cite_col and cite_col in df.columns:
        df["Cited by"] = pd.to_numeric(df[cite_col], errors="coerce").fillna(0).astype(int)
    
    # Parse fields of study
    fos_col = None
    for candidate in ["Fields of Study", "fields_of_study", "Subject"]:
        if candidate in df.columns:
            fos_col = candidate
            break
    
    if fos_col:
        df["Index Keywords"] = df[fos_col].apply(
            lambda x: _parse_lens_fields(x, separator)
        )
    
    return df


def _parse_lens_authors(authors: str, separator: str = "; ") -> str:
    """Parse Lens.org author format."""
    if pd.isna(authors):
        return ""
    
    authors_str = str(authors).strip()
    if not authors_str:
        return ""
    
    # Lens may use semicolons or JSON-like format
    # Try to parse as list
    if authors_str.startswith("["):
        try:
            import json
            authors_list = json.loads(authors_str.replace("'", '"'))
            if isinstance(authors_list, list):
                return separator.join(str(a) for a in authors_list if a)
        except:
            pass
    
    # Simple split
    parts = re.split(r'[;\n|]', authors_str)
    return separator.join(p.strip() for p in parts if p.strip())


def _parse_lens_affiliations(affiliations: str, separator: str = "; ") -> str:
    """Parse Lens.org affiliation format."""
    if pd.isna(affiliations):
        return ""
    
    aff_str = str(affiliations).strip()
    if not aff_str:
        return ""
    
    # Similar to authors
    if aff_str.startswith("["):
        try:
            import json
            aff_list = json.loads(aff_str.replace("'", '"'))
            if isinstance(aff_list, list):
                return separator.join(str(a) for a in aff_list if a)
        except:
            pass
    
    parts = re.split(r'[;\n|]', aff_str)
    return separator.join(p.strip() for p in parts if p.strip())


def _parse_lens_fields(fields: str, separator: str = "; ") -> str:
    """Parse Lens.org fields of study."""
    if pd.isna(fields):
        return ""
    
    fields_str = str(fields).strip()
    if not fields_str:
        return ""
    
    # Handle JSON-like format
    if fields_str.startswith("["):
        try:
            import json
            fields_list = json.loads(fields_str.replace("'", '"'))
            if isinstance(fields_list, list):
                return separator.join(str(f) for f in fields_list if f)
        except:
            pass
    
    parts = re.split(r'[;\n|,]', fields_str)
    return separator.join(p.strip() for p in parts if p.strip())


# =============================================================================
# DIMENSIONS PREPROCESSING
# =============================================================================

def preprocess_dimensions(
    df: pd.DataFrame,
    separator: str = "; ",
) -> pd.DataFrame:
    """
    Preprocess Dimensions data for bibliometric analysis.
    
    Handles:
    - Author name standardization
    - FOR (Fields of Research) code parsing
    - Citation normalization
    - Affiliation extraction
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw Dimensions data.
    separator : str
        Separator for multi-value fields.
        
    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    df = df.copy()
    
    # Standardize column names
    df = standardize_columns(df)
    
    # Parse authors
    auth_col = find_column(df, "authors")
    if auth_col and auth_col in df.columns:
        df["Authors"] = df[auth_col].apply(
            lambda x: _parse_dimensions_authors(x, separator)
        )
    
    # Parse FOR categories
    for_col = None
    for candidate in ["FOR (ANZSRC 2020)", "FOR (ANZSRC) Categories", "Fields of Research", "Research Areas"]:
        if candidate in df.columns:
            for_col = candidate
            break
    
    if for_col:
        df["Index Keywords"] = df[for_col].apply(
            lambda x: _parse_dimensions_for(x, separator)
        )
    
    # Standardize year
    year_col = find_column(df, "year")
    if year_col and year_col in df.columns:
        df["Year"] = pd.to_numeric(df[year_col], errors="coerce")
    
    # Standardize citations
    cite_col = find_column(df, "cited_by")
    if cite_col and cite_col in df.columns:
        df["Cited by"] = pd.to_numeric(df[cite_col], errors="coerce").fillna(0).astype(int)
    
    # Extract FWCI if available
    fwci_col = find_column(df, "fwci")
    if fwci_col and fwci_col in df.columns:
        df["FWCI"] = pd.to_numeric(df[fwci_col], errors="coerce")
    
    return df


def _parse_dimensions_authors(authors: str, separator: str = "; ") -> str:
    """Parse Dimensions author format."""
    if pd.isna(authors):
        return ""
    
    authors_str = str(authors).strip()
    if not authors_str:
        return ""
    
    # Dimensions uses semicolons
    parts = authors_str.split(";")
    cleaned = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Remove affiliation in parentheses
        part = re.sub(r'\s*\([^)]*\)\s*', '', part)
        
        if part:
            cleaned.append(part)
    
    return separator.join(cleaned)


def _parse_dimensions_for(for_str: str, separator: str = "; ") -> str:
    """Parse Dimensions FOR (Fields of Research) categories."""
    if pd.isna(for_str):
        return ""
    
    for_str = str(for_str).strip()
    if not for_str:
        return ""
    
    # FOR codes may include numeric codes - extract text labels
    parts = re.split(r'[;\n]', for_str)
    terms = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Remove numeric code prefix (e.g., "0801 Artificial Intelligence")
        part = re.sub(r'^\d+\s+', '', part)
        
        if part and part not in terms:
            terms.append(part)
    
    return separator.join(terms)


# =============================================================================
# UNIFIED PREPROCESSOR
# =============================================================================

def preprocess_bibliographic_data(
    df: pd.DataFrame,
    db: str,
    separator: str = "; ",
    **kwargs,
) -> pd.DataFrame:
    """
    Preprocess bibliographic data from any supported database.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw bibliographic data.
    db : str
        Database name: 'pubmed', 'lens', 'dimensions', 'scopus', 'openalex', 'wos'
    separator : str
        Separator for multi-value fields.
    **kwargs
        Additional arguments passed to database-specific preprocessor.
        
    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with standardized columns.
    """
    db_lower = db.lower().strip()
    
    preprocessors = {
        "pubmed": preprocess_pubmed,
        "medline": preprocess_pubmed,
        "lens": preprocess_lens,
        "lens.org": preprocess_lens,
        "dimensions": preprocess_dimensions,
    }
    
    if db_lower in preprocessors:
        df = preprocessors[db_lower](df, separator=separator, **kwargs)
    else:
        # For scopus, openalex, wos - just standardize columns
        df = standardize_columns(df)
    
    # Common post-processing
    df = _common_postprocessing(df, separator)
    
    return df


def _common_postprocessing(df: pd.DataFrame, separator: str = "; ") -> pd.DataFrame:
    """Apply common postprocessing to all databases."""
    df = df.copy()
    
    # Ensure Year is integer
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        # Remove rows with invalid years
        df = df[df["Year"].notna() & (df["Year"] >= 1900) & (df["Year"] <= 2100)]
        df["Year"] = df["Year"].astype(int)
    
    # Ensure citations is integer
    if "Cited by" in df.columns:
        df["Cited by"] = pd.to_numeric(df["Cited by"], errors="coerce").fillna(0).astype(int)
    
    # Clean empty strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace({"": np.nan, " ": np.nan})
    
    return df


# =============================================================================
# DATABASE FORMAT DETECTION
# =============================================================================

def detect_database(df: pd.DataFrame) -> str:
    """
    Attempt to detect the source database from column names.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
        
    Returns
    -------
    str
        Detected database name or 'unknown'.
    """
    cols = set(df.columns)
    cols_lower = {c.lower() for c in cols}
    
    # Scopus indicators
    scopus_indicators = {"EID", "Cited by", "Source title", "Author Keywords"}
    if len(scopus_indicators & cols) >= 3:
        return "scopus"
    
    # OpenAlex indicators
    openalex_indicators = {"id", "doi", "display_name", "publication_year", "cited_by_count"}
    if len(openalex_indicators & cols_lower) >= 3:
        return "openalex"
    
    # PubMed indicators
    pubmed_indicators = {"PMID", "MeSH Terms", "Journal", "ArticleTitle"}
    if len(pubmed_indicators & cols) >= 2:
        return "pubmed"
    
    # Lens indicators
    lens_indicators = {"Lens ID", "Scholarly Citations", "Fields of Study"}
    if len(lens_indicators & cols) >= 2:
        return "lens"
    
    # Dimensions indicators
    dimensions_indicators = {"Publication ID", "Times cited", "FOR (ANZSRC 2020)"}
    if len(dimensions_indicators & cols) >= 2:
        return "dimensions"
    
    # WoS indicators
    wos_indicators = {"UT", "WC", "SC", "TC"}
    if len(wos_indicators & cols) >= 2:
        return "wos"
    
    return "unknown"
