# -*- coding: utf-8 -*-
"""
Text processing utilities - keywords, stopwords, NLP preprocessing.

This module contains:
- preprocess_keywords: Clean and harmonize keyword columns
- process_text_column: Process text with stopword removal
- merge_keywords_columns: Combine keyword columns
- build_combined_text: Build combined text fields
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

# NLTK imports (with fallbacks)
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.data import find
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    WordNetLemmatizer = None


# =============================================================================
# MODULE DATA
# =============================================================================

_fd = os.path.dirname(os.path.dirname(__file__))
stopwords_file = os.path.join(_fd, "additional files", "stopwords.xlsx")


# =============================================================================
# NLTK SETUP
# =============================================================================

def ensure_wordnet() -> None:
    """Ensure that the NLTK WordNet corpus is available."""
    if not NLTK_AVAILABLE:
        return
    try:
        find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def _ensure_nltk() -> bool:
    """Ensure NLTK resources are available."""
    if not NLTK_AVAILABLE:
        return False
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        except Exception:
            return False
    return True


# =============================================================================
# KEYWORD PROCESSING
# =============================================================================

def preprocess_keywords(
    df: pd.DataFrame,
    column: str,
    exclude_list: Optional[Union[List[str], pd.DataFrame]] = None,
    synonyms: Optional[Union[pd.DataFrame, Dict]] = None,
    lemmatize: bool = False,
    sep: str = "; ",
    *,
    normalize_compounds: Optional[str] = "space",
    fold_accents: bool = False,
    alt_separators: Optional[List[str]] = None,
    sort_keywords: bool = True,
    strip_punctuation: bool = True,
    normalize_underscores: bool = True,
) -> pd.DataFrame:
    """
    Clean and harmonize a keywords column.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column : str
        Name of the column containing keywords.
    exclude_list : list or DataFrame, optional
        Keywords to remove.
    synonyms : DataFrame or dict, optional
        Synonym mappings.
    lemmatize : bool
        If True, lemmatize each word.
    sep : str
        Separator used in the keyword column.
    normalize_compounds : {"space", "hyphen", None}
        How to handle hyphenated terms.
    fold_accents : bool
        If True, strip diacritics.
    alt_separators : list, optional
        Additional separators to accept.
    sort_keywords : bool
        If True, sort keywords alphabetically.
    strip_punctuation : bool
        If True, strip leading/trailing punctuation.
    normalize_underscores : bool
        If True, convert underscores to spaces.

    Returns
    -------
    DataFrame
        DataFrame with added "Processed {column}" column.
    """
    if column not in df.columns:
        print(f'Column "{column}" not found in the DataFrame.')
        return df

    df = df.copy()
    
    # Build canonicalization function
    def canonicalize(text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        
        # Fold accents if requested
        if fold_accents:
            text = "".join(
                c for c in unicodedata.normalize("NFD", text)
                if unicodedata.category(c) != "Mn"
            )
        
        # Normalize underscores
        if normalize_underscores:
            text = re.sub(r"_+", " ", text)
        
        # Normalize hyphens based on setting
        hyphen_pattern = r"[\u2010\u2011\u2012\u2013\u2014\u2212-]+"
        if normalize_compounds == "space":
            text = re.sub(hyphen_pattern, " ", text)
        elif normalize_compounds == "hyphen":
            text = re.sub(hyphen_pattern, "-", text)
            text = re.sub(r"\s+", "-", text)
        
        # Strip punctuation at ends
        if strip_punctuation:
            text = text.strip(".,;:!?\"'()[]{}")
        
        # Collapse whitespace
        text = " ".join(text.split())
        
        return text
    
    # Process exclude list
    exclude_set = set()
    if exclude_list is not None:
        if isinstance(exclude_list, pd.DataFrame):
            exclude_set = set(exclude_list.iloc[:, 0].dropna().astype(str))
        else:
            exclude_set = set(exclude_list)
        exclude_set = {canonicalize(e) for e in exclude_set}
    
    # Process synonyms
    synonym_map = {}
    if synonyms is not None:
        if isinstance(synonyms, pd.DataFrame):
            for _, row in synonyms.iterrows():
                old_val = canonicalize(str(row.iloc[0]))
                new_val = canonicalize(str(row.iloc[1]))
                synonym_map[old_val] = new_val
        elif isinstance(synonyms, dict):
            for new_kw, old_list in synonyms.items():
                new_kw_canon = canonicalize(new_kw)
                for old_kw in old_list:
                    synonym_map[canonicalize(old_kw)] = new_kw_canon
    
    # Setup lemmatizer
    lemmatizer = None
    if lemmatize and NLTK_AVAILABLE:
        ensure_wordnet()
        lemmatizer = WordNetLemmatizer()
    
    def process_cell(cell):
        if pd.isna(cell) or str(cell).strip() == "":
            return ""
        
        # Split on separators
        text = str(cell)
        if alt_separators:
            for alt_sep in alt_separators:
                text = text.replace(alt_sep, sep)
        
        keywords = text.split(sep)
        
        processed = []
        seen = set()
        
        for kw in keywords:
            kw = canonicalize(kw)
            if not kw:
                continue
            
            # Apply synonym mapping
            kw = synonym_map.get(kw, kw)
            
            # Skip excluded
            if kw in exclude_set:
                continue
            
            # Lemmatize if requested
            if lemmatizer:
                words = kw.split()
                words = [lemmatizer.lemmatize(w) for w in words]
                kw = " ".join(words)
            
            # Deduplicate
            if kw not in seen:
                seen.add(kw)
                processed.append(kw)
        
        if sort_keywords:
            processed.sort()
        
        return sep.join(processed)
    
    new_col = f"Processed {column}"
    df[new_col] = df[column].apply(process_cell)
    
    return df


def merge_keywords_columns(
    df: pd.DataFrame,
    author_col: str = "Author Keywords",
    index_col: str = "Index Keywords",
    sep: str = "; ",
) -> pd.Series:
    """
    Merge author and index keywords into a single column.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    author_col : str
        Author keywords column name.
    index_col : str
        Index keywords column name.
    sep : str
        Separator for keywords.

    Returns
    -------
    Series
        Combined keywords.
    """
    def merge_row(row):
        author = str(row.get(author_col, "")) if pd.notna(row.get(author_col)) else ""
        index = str(row.get(index_col, "")) if pd.notna(row.get(index_col)) else ""
        
        combined = []
        if author:
            combined.extend(author.split(sep))
        if index:
            combined.extend(index.split(sep))
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for kw in combined:
            kw = kw.strip()
            if kw and kw.lower() not in seen:
                seen.add(kw.lower())
                unique.append(kw)
        
        return sep.join(unique)
    
    return df.apply(merge_row, axis=1)


def merge_text_columns(
    df: pd.DataFrame,
    title_col: str = "Processed Title",
    abstract_col: str = "Processed Abstract",
    author_col: str = "Processed Author Keywords",
    index_col: str = "Processed Index Keywords",
) -> pd.DataFrame:
    """
    Merge multiple text columns into combined text fields.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    title_col, abstract_col, author_col, index_col : str
        Column names to merge.

    Returns
    -------
    DataFrame
        DataFrame with added combined text columns.
    """
    df = df.copy()
    
    def safe_concat(*texts):
        parts = []
        for t in texts:
            if pd.notna(t) and str(t).strip():
                parts.append(str(t).strip())
        return " ".join(parts)
    
    # Title + Abstract
    if title_col in df.columns and abstract_col in df.columns:
        df["Title and Abstract"] = df.apply(
            lambda r: safe_concat(r.get(title_col), r.get(abstract_col)),
            axis=1
        )
    
    # All text
    cols = [c for c in [title_col, abstract_col, author_col, index_col] if c in df.columns]
    if cols:
        df["All Text"] = df.apply(
            lambda r: safe_concat(*[r.get(c) for c in cols]),
            axis=1
        )
    
    return df


def build_combined_text(
    df: pd.DataFrame,
    include_index_keywords: bool = False,
) -> pd.DataFrame:
    """
    Build combined text fields from title, abstract, and keywords.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    include_index_keywords : bool
        Whether to include index keywords.

    Returns
    -------
    DataFrame
        DataFrame with combined text columns.
    """
    df = df.copy()
    
    # Determine which columns to use
    title_col = "Processed Title" if "Processed Title" in df.columns else "Title"
    abstract_col = "Processed Abstract" if "Processed Abstract" in df.columns else "Abstract"
    author_kw_col = "Processed Author Keywords" if "Processed Author Keywords" in df.columns else "Author Keywords"
    
    def combine(row):
        parts = []
        
        if title_col in df.columns and pd.notna(row.get(title_col)):
            parts.append(str(row[title_col]))
        
        if abstract_col in df.columns and pd.notna(row.get(abstract_col)):
            parts.append(str(row[abstract_col]))
        
        if author_kw_col in df.columns and pd.notna(row.get(author_kw_col)):
            parts.append(str(row[author_kw_col]))
        
        if include_index_keywords:
            index_kw_col = "Processed Index Keywords" if "Processed Index Keywords" in df.columns else "Index Keywords"
            if index_kw_col in df.columns and pd.notna(row.get(index_kw_col)):
                parts.append(str(row[index_kw_col]))
        
        return " ".join(parts)
    
    df["Combined Text"] = df.apply(combine, axis=1)
    
    return df


def process_text_column(
    df: pd.DataFrame,
    column: str,
    stopwords_file: Optional[str] = None,
    lang: str = "en",
    extra_stopwords: Optional[List[str]] = None,
    exclude_specific_stopwords: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Process a text column with stopword removal and cleaning.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column : str
        Column to process.
    stopwords_file : str, optional
        Path to stopwords Excel file.
    lang : str
        Language for NLTK stopwords.
    extra_stopwords : list, optional
        Additional stopwords to remove.
    exclude_specific_stopwords : list, optional
        Stopword categories to exclude.

    Returns
    -------
    DataFrame
        DataFrame with processed text column.
    """
    if column not in df.columns:
        return df
    
    df = df.copy()
    
    # Build stopwords set
    stop_set = set()
    
    # NLTK stopwords
    if _ensure_nltk():
        try:
            stop_set.update(stopwords.words(lang))
        except Exception:
            pass
    
    # Custom stopwords from file
    if stopwords_file and os.path.exists(stopwords_file):
        try:
            sw_df = pd.read_excel(stopwords_file, sheet_name="general")
            stop_set.update(sw_df.iloc[:, 0].dropna().astype(str).str.lower())
        except Exception:
            pass
    
    # Extra stopwords
    if extra_stopwords:
        stop_set.update(w.lower() for w in extra_stopwords)
    
    def process(text):
        if pd.isna(text) or not str(text).strip():
            return ""
        
        text = str(text).lower()
        
        # Remove punctuation except hyphens in words
        text = re.sub(r"[^\w\s-]", " ", text)
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords and short words
        words = [w for w in words if w not in stop_set and len(w) > 2]
        
        return " ".join(words)
    
    new_col = f"Processed {column}"
    df[new_col] = df[column].apply(process)
    
    return df
