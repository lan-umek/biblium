# -*- coding: utf-8 -*-
"""
Core utilities - foundational functions used throughout Biblium.

This module contains:
- File/folder operations
- Progress bars
- Basic data manipulation helpers
- Mapping and configuration utilities
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np

# Progress bar support
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        """Fallback when tqdm is not installed."""
        return iterable


# =============================================================================
# MODULE PATHS
# =============================================================================

fd = os.path.dirname(os.path.dirname(__file__))  # biblium package directory


# =============================================================================
# FOLDER OPERATIONS
# =============================================================================

def make_folder(folder_path: str) -> None:
    """Create a folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def make_folders(folder_paths: List[str]) -> None:
    """Create multiple folders if they don't exist."""
    for folder_path in folder_paths:
        make_folder(folder_path)


# =============================================================================
# PROGRESS BARS
# =============================================================================

def progress_bar(
    iterable,
    desc: str = "",
    total: Optional[int] = None,
    disable: bool = False,
    **kwargs
):
    """
    Wrap an iterable with a progress bar.
    
    Parameters
    ----------
    iterable : iterable
        The iterable to wrap.
    desc : str
        Description to show.
    total : int, optional
        Total number of items (auto-detected if not provided).
    disable : bool
        If True, disable the progress bar.
    **kwargs
        Additional arguments passed to tqdm.
        
    Returns
    -------
    iterable
        Wrapped iterable with progress tracking.
    """
    if not TQDM_AVAILABLE or disable:
        return iterable
    return tqdm(iterable, desc=desc, total=total, **kwargs)


def progress_apply(
    df: pd.DataFrame,
    func: Callable,
    axis: int = 1,
    desc: str = "Processing",
    disable: bool = False,
    **kwargs
) -> pd.Series:
    """
    Apply a function to a DataFrame with progress bar.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to process.
    func : callable
        Function to apply.
    axis : int
        Axis to apply along (0=columns, 1=rows).
    desc : str
        Progress bar description.
    disable : bool
        If True, disable progress bar.
    **kwargs
        Additional arguments passed to apply.
        
    Returns
    -------
    Series
        Result of apply operation.
    """
    if TQDM_AVAILABLE and not disable:
        tqdm.pandas(desc=desc)
        return df.progress_apply(func, axis=axis, **kwargs)
    return df.apply(func, axis=axis, **kwargs)


# =============================================================================
# ATTRIBUTE AND MAPPING UTILITIES
# =============================================================================

def rename_attributes(obj: Any, rename_map: Dict[str, str]) -> None:
    """
    Rename attributes of an object using a mapping dictionary.
    
    Parameters
    ----------
    obj : object
        Object whose attributes to rename.
    rename_map : dict
        Mapping from old names to new names.
    """
    for old_name, new_name in rename_map.items():
        if hasattr(obj, old_name):
            setattr(obj, new_name, getattr(obj, old_name))


def first_existing(df: pd.DataFrame, columns: List[str]) -> Optional[str]:
    """
    Return the first column name that exists in the DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to check.
    columns : list
        List of column names to check in priority order.
        
    Returns
    -------
    str or None
        First existing column name, or None if none exist.
    """
    for col in columns:
        if col in df.columns:
            return col
    return None


# =============================================================================
# REGEX BUILDERS
# =============================================================================

def build_exclusion_regex(
    exclude_list: List[str],
    case_sensitive: bool = False
) -> Optional[re.Pattern]:
    """
    Build a compiled regex pattern for excluding items.
    
    Parameters
    ----------
    exclude_list : list
        List of strings/patterns to exclude.
    case_sensitive : bool
        If False, pattern is case-insensitive.
        
    Returns
    -------
    Pattern or None
        Compiled regex pattern, or None if exclude_list is empty.
    """
    if not exclude_list:
        return None
    
    # Escape special regex characters and join with |
    escaped = [re.escape(item) for item in exclude_list]
    pattern = "|".join(escaped)
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern, flags)


def build_inclusion_regex(
    include_list: List[str],
    case_sensitive: bool = False,
    word_boundary: bool = True
) -> Optional[re.Pattern]:
    """
    Build a compiled regex pattern for including/matching items.
    
    Parameters
    ----------
    include_list : list
        List of strings/patterns to include.
    case_sensitive : bool
        If False, pattern is case-insensitive.
    word_boundary : bool
        If True, match whole words only.
        
    Returns
    -------
    Pattern or None
        Compiled regex pattern, or None if include_list is empty.
    """
    if not include_list:
        return None
    
    escaped = [re.escape(item) for item in include_list]
    if word_boundary:
        escaped = [rf"\b{item}\b" for item in escaped]
    pattern = "|".join(escaped)
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern, flags)


# =============================================================================
# MAPPING RECONSTRUCTION
# =============================================================================

def reconstruct_mapping(
    mapping_df: pd.DataFrame,
    alias_df: pd.DataFrame,
    key_col: str = "key",
    value_col: str = "value",
    alias_key_col: str = "alias",
    alias_value_col: str = "canonical"
) -> Dict[str, str]:
    """
    Reconstruct a mapping dictionary from DataFrames.
    
    Combines a primary mapping with aliases to create a unified
    lookup dictionary.
    
    Parameters
    ----------
    mapping_df : DataFrame
        Primary key-value mappings.
    alias_df : DataFrame
        Alias-to-canonical mappings.
    key_col, value_col : str
        Column names in mapping_df.
    alias_key_col, alias_value_col : str
        Column names in alias_df.
        
    Returns
    -------
    dict
        Combined mapping dictionary.
    """
    # Build primary mapping
    mapping = {}
    if mapping_df is not None and len(mapping_df) > 0:
        mapping = dict(zip(mapping_df[key_col], mapping_df[value_col]))
    
    # Add aliases
    if alias_df is not None and len(alias_df) > 0:
        for _, row in alias_df.iterrows():
            alias = row[alias_key_col]
            canonical = row[alias_value_col]
            if canonical in mapping:
                mapping[alias] = mapping[canonical]
            else:
                mapping[alias] = canonical
    
    return mapping


def reconstruct_mapping_from_excel(
    file_path: str,
    mapping_sheet: str = "mapping",
    alias_sheet: str = "alias",
    **kwargs
) -> Dict[str, str]:
    """
    Load and reconstruct a mapping from an Excel file.
    
    Parameters
    ----------
    file_path : str
        Path to Excel file.
    mapping_sheet : str
        Name of the mapping sheet.
    alias_sheet : str
        Name of the alias sheet.
    **kwargs
        Additional arguments passed to reconstruct_mapping.
        
    Returns
    -------
    dict
        Combined mapping dictionary.
    """
    mapping_df = pd.read_excel(file_path, sheet_name=mapping_sheet)
    alias_df = pd.read_excel(file_path, sheet_name=alias_sheet)
    return reconstruct_mapping(mapping_df, alias_df, **kwargs)
