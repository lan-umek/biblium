# -*- coding: utf-8 -*-
"""
Batch Processing Utilities for Large Datasets

Provides memory-efficient processing for datasets with 100k+ records.
Wraps existing functions to process data in chunks.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import gc
import warnings
from typing import Any, Callable, Dict, Generator, List, Optional, Union, Iterable
from functools import wraps

import pandas as pd
import numpy as np

# Progress bar support
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable


# =============================================================================
# CORE BATCH PROCESSING
# =============================================================================

def chunk_dataframe(
    df: pd.DataFrame,
    chunk_size: int = 10000,
) -> Generator[pd.DataFrame, None, None]:
    """
    Split DataFrame into chunks.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split.
    chunk_size : int
        Number of rows per chunk.
    
    Yields
    ------
    pd.DataFrame
        Chunk of the original DataFrame.
    
    Example
    -------
    >>> for chunk in chunk_dataframe(df, chunk_size=5000):
    ...     process(chunk)
    """
    n_rows = len(df)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        yield df.iloc[start:end].copy()


def batch_apply(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    chunk_size: int = 10000,
    desc: str = "Processing",
    verbose: bool = True,
    concat_results: bool = True,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Apply a function to DataFrame in batches.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    func : callable
        Function that takes DataFrame and returns DataFrame.
    chunk_size : int
        Rows per batch.
    desc : str
        Progress bar description.
    verbose : bool
        Show progress bar.
    concat_results : bool
        If True, concatenate results. If False, return list.
    
    Returns
    -------
    pd.DataFrame or list
        Combined results or list of chunk results.
    
    Example
    -------
    >>> def process_text(df):
    ...     df['processed'] = df['Abstract'].str.lower()
    ...     return df
    >>> result = batch_apply(df, process_text, chunk_size=5000)
    """
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    chunks = chunk_dataframe(df, chunk_size)
    
    if verbose and TQDM_AVAILABLE:
        chunks = tqdm(chunks, total=n_chunks, desc=desc)
    
    results = []
    for chunk in chunks:
        result = func(chunk)
        results.append(result)
        gc.collect()  # Free memory
    
    if concat_results:
        return pd.concat(results, ignore_index=True)
    return results


def batch_aggregate(
    df: pd.DataFrame,
    agg_func: Callable[[pd.DataFrame], Dict[str, Any]],
    chunk_size: int = 10000,
    merge_func: Callable[[List[Dict]], Dict] = None,
    desc: str = "Aggregating",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate DataFrame in batches (for counting, summing, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    agg_func : callable
        Function that aggregates a chunk and returns dict.
    chunk_size : int
        Rows per batch.
    merge_func : callable
        Function to merge chunk results. Default sums numeric values.
    desc : str
        Progress bar description.
    verbose : bool
        Show progress bar.
    
    Returns
    -------
    dict
        Merged aggregation results.
    
    Example
    -------
    >>> def count_authors(chunk):
    ...     from collections import Counter
    ...     authors = chunk['Authors'].str.split('; ').explode()
    ...     return dict(Counter(authors))
    >>> 
    >>> def merge_counts(results):
    ...     from collections import Counter
    ...     merged = Counter()
    ...     for r in results:
    ...         merged.update(r)
    ...     return dict(merged)
    >>> 
    >>> counts = batch_aggregate(df, count_authors, merge_func=merge_counts)
    """
    from collections import Counter
    
    # Default merge: sum counters
    if merge_func is None:
        def merge_func(results):
            merged = Counter()
            for r in results:
                if isinstance(r, dict):
                    merged.update(r)
            return dict(merged)
    
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    chunks = chunk_dataframe(df, chunk_size)
    
    if verbose and TQDM_AVAILABLE:
        chunks = tqdm(chunks, total=n_chunks, desc=desc)
    
    results = []
    for chunk in chunks:
        result = agg_func(chunk)
        results.append(result)
        gc.collect()
    
    return merge_func(results)


def batch_count_column(
    df: pd.DataFrame,
    column: str,
    sep: str = "; ",
    chunk_size: int = 10000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Count values in a multi-value column using batch processing.
    
    Efficient for columns like Authors, Keywords where values
    are separated by a delimiter.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to count.
    sep : str
        Value separator.
    chunk_size : int
        Rows per batch.
    verbose : bool
        Show progress.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns [item, count], sorted by count.
    
    Example
    -------
    >>> author_counts = batch_count_column(df, "Authors", sep="; ")
    """
    from collections import Counter
    
    def count_chunk(chunk):
        counter = Counter()
        for val in chunk[column].dropna():
            items = [i.strip() for i in str(val).split(sep) if i.strip()]
            counter.update(items)
        return dict(counter)
    
    def merge_counts(results):
        merged = Counter()
        for r in results:
            merged.update(r)
        return dict(merged)
    
    counts = batch_aggregate(
        df, count_chunk, 
        chunk_size=chunk_size,
        merge_func=merge_counts,
        desc=f"Counting {column}",
        verbose=verbose
    )
    
    result_df = pd.DataFrame([
        {"Item": k, "Count": v} for k, v in counts.items()
    ])
    
    if len(result_df) > 0:
        result_df = result_df.sort_values("Count", ascending=False).reset_index(drop=True)
    
    return result_df


def batch_text_process(
    df: pd.DataFrame,
    text_column: str,
    process_func: Callable[[str], Any],
    output_column: str = None,
    chunk_size: int = 5000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply text processing function in batches.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    text_column : str
        Column containing text.
    process_func : callable
        Function to apply to each text value.
    output_column : str
        Name for output column. Default: text_column + '_processed'.
    chunk_size : int
        Rows per batch (smaller for heavy NLP).
    verbose : bool
        Show progress.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with new processed column.
    
    Example
    -------
    >>> def extract_keywords(text):
    ...     # some NLP processing
    ...     return keywords
    >>> 
    >>> df = batch_text_process(df, "Abstract", extract_keywords, "Keywords_NLP")
    """
    if output_column is None:
        output_column = f"{text_column}_processed"
    
    def process_chunk(chunk):
        chunk = chunk.copy()
        chunk[output_column] = chunk[text_column].apply(
            lambda x: process_func(str(x)) if pd.notna(x) else None
        )
        return chunk
    
    return batch_apply(
        df, process_chunk,
        chunk_size=chunk_size,
        desc=f"Processing {text_column}",
        verbose=verbose
    )


# =============================================================================
# MEMORY-EFFICIENT FILE READING
# =============================================================================

def read_large_csv(
    filepath: str,
    chunk_size: int = 50000,
    process_func: Callable[[pd.DataFrame], pd.DataFrame] = None,
    verbose: bool = True,
    **read_kwargs,
) -> pd.DataFrame:
    """
    Read large CSV file in chunks.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file.
    chunk_size : int
        Rows per chunk.
    process_func : callable
        Optional function to process each chunk before combining.
    verbose : bool
        Show progress.
    **read_kwargs
        Additional arguments for pd.read_csv.
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame.
    
    Example
    -------
    >>> df = read_large_csv("big_file.csv", chunk_size=100000)
    """
    chunks = []
    
    reader = pd.read_csv(filepath, chunksize=chunk_size, **read_kwargs)
    
    if verbose and TQDM_AVAILABLE:
        reader = tqdm(reader, desc="Reading CSV")
    
    for chunk in reader:
        if process_func:
            chunk = process_func(chunk)
        chunks.append(chunk)
        gc.collect()
    
    return pd.concat(chunks, ignore_index=True)


def read_large_excel(
    filepath: str,
    chunk_size: int = 50000,
    sheet_name: Union[str, int] = 0,
    verbose: bool = True,
    **read_kwargs,
) -> pd.DataFrame:
    """
    Read large Excel file in chunks.
    
    Note: Excel doesn't support native chunking, so this reads
    the whole file but processes in chunks for memory efficiency
    during any transformations.
    
    Parameters
    ----------
    filepath : str
        Path to Excel file.
    chunk_size : int
        Rows per processing chunk.
    sheet_name : str or int
        Sheet to read.
    verbose : bool
        Show progress.
    **read_kwargs
        Additional arguments for pd.read_excel.
    
    Returns
    -------
    pd.DataFrame
        DataFrame.
    """
    if verbose:
        print(f"Reading Excel file: {filepath}")
    
    # Excel must be read entirely, but we can process in chunks after
    df = pd.read_excel(filepath, sheet_name=sheet_name, **read_kwargs)
    
    if verbose:
        print(f"Loaded {len(df)} rows")
    
    return df


# =============================================================================
# BATCH PROCESSING DECORATOR
# =============================================================================

def batchable(chunk_size: int = 10000, desc: str = None):
    """
    Decorator to make a function work in batches.
    
    The decorated function should accept a DataFrame as first argument
    and return a DataFrame.
    
    Parameters
    ----------
    chunk_size : int
        Default chunk size.
    desc : str
        Progress description.
    
    Example
    -------
    >>> @batchable(chunk_size=5000)
    ... def process_abstracts(df):
    ...     df['words'] = df['Abstract'].str.split().str.len()
    ...     return df
    >>> 
    >>> # Now can be called with batch_size parameter:
    >>> result = process_abstracts(big_df, batch_size=10000)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, batch_size: int = None, **kwargs):
            size = batch_size or chunk_size
            
            # If small enough, just run normally
            if len(df) <= size:
                return func(df, *args, **kwargs)
            
            # Otherwise batch process
            description = desc or f"Batch {func.__name__}"
            
            def apply_func(chunk):
                return func(chunk, *args, **kwargs)
            
            return batch_apply(df, apply_func, chunk_size=size, desc=description)
        
        return wrapper
    return decorator


# =============================================================================
# MEMORY MONITORING
# =============================================================================

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns
    -------
    dict
        Memory usage in MB.
    """
    import sys
    
    result = {
        "python_mb": sys.getsizeof(globals()) / (1024 * 1024),
    }
    
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        result["rss_mb"] = mem_info.rss / (1024 * 1024)
        result["vms_mb"] = mem_info.vms / (1024 * 1024)
    except ImportError:
        pass
    
    return result


def estimate_dataframe_memory(df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate memory usage of a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.
    
    Returns
    -------
    dict
        Memory estimates in MB.
    """
    deep_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    shallow_memory = df.memory_usage(deep=False).sum() / (1024 * 1024)
    
    return {
        "total_mb": deep_memory,
        "shallow_mb": shallow_memory,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "mb_per_1000_rows": deep_memory / (len(df) / 1000) if len(df) > 0 else 0,
    }


def suggest_chunk_size(
    df: pd.DataFrame,
    target_memory_mb: float = 500,
) -> int:
    """
    Suggest optimal chunk size based on target memory usage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Sample DataFrame (or full DataFrame).
    target_memory_mb : float
        Target memory per chunk in MB.
    
    Returns
    -------
    int
        Suggested chunk size.
    """
    mem_info = estimate_dataframe_memory(df)
    mb_per_1000 = mem_info["mb_per_1000_rows"]
    
    if mb_per_1000 <= 0:
        return 10000
    
    suggested = int((target_memory_mb / mb_per_1000) * 1000)
    
    # Clamp to reasonable range
    suggested = max(1000, min(100000, suggested))
    
    return suggested


# =============================================================================
# BATCH PROCESSING CONTEXT MANAGER
# =============================================================================

class BatchProcessor:
    """
    Context manager for batch processing with automatic memory management.
    
    Example
    -------
    >>> with BatchProcessor(df, chunk_size=10000) as processor:
    ...     for chunk in processor:
    ...         # process chunk
    ...         results.append(process(chunk))
    ...     
    ...     # Check stats
    ...     print(processor.stats)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        chunk_size: int = 10000,
        desc: str = "Processing",
        verbose: bool = True,
    ):
        self.df = df
        self.chunk_size = chunk_size
        self.desc = desc
        self.verbose = verbose
        self.stats = {
            "total_rows": len(df),
            "chunk_size": chunk_size,
            "n_chunks": (len(df) + chunk_size - 1) // chunk_size,
            "processed_rows": 0,
            "processed_chunks": 0,
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        return False
    
    def __iter__(self):
        chunks = chunk_dataframe(self.df, self.chunk_size)
        
        if self.verbose and TQDM_AVAILABLE:
            chunks = tqdm(chunks, total=self.stats["n_chunks"], desc=self.desc)
        
        for chunk in chunks:
            yield chunk
            self.stats["processed_rows"] += len(chunk)
            self.stats["processed_chunks"] += 1
            gc.collect()


# =============================================================================
# CONVENIENCE FUNCTIONS FOR BIBLIUM
# =============================================================================

def batch_count_authors(
    df: pd.DataFrame,
    authors_col: str = "Authors",
    sep: str = "; ",
    chunk_size: int = 10000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Count authors with batch processing."""
    return batch_count_column(df, authors_col, sep=sep, chunk_size=chunk_size, verbose=verbose)


def batch_count_keywords(
    df: pd.DataFrame,
    keywords_col: str = "Author Keywords",
    sep: str = "; ",
    chunk_size: int = 10000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Count keywords with batch processing."""
    return batch_count_column(df, keywords_col, sep=sep, chunk_size=chunk_size, verbose=verbose)


def batch_count_sources(
    df: pd.DataFrame,
    source_col: str = "Source title",
    chunk_size: int = 10000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Count sources with batch processing."""
    from collections import Counter
    
    def count_chunk(chunk):
        return dict(Counter(chunk[source_col].dropna()))
    
    def merge_counts(results):
        merged = Counter()
        for r in results:
            merged.update(r)
        return dict(merged)
    
    counts = batch_aggregate(
        df, count_chunk,
        chunk_size=chunk_size,
        merge_func=merge_counts,
        desc="Counting sources",
        verbose=verbose
    )
    
    result_df = pd.DataFrame([
        {"Source": k, "Count": v} for k, v in counts.items()
    ])
    
    if len(result_df) > 0:
        result_df = result_df.sort_values("Count", ascending=False).reset_index(drop=True)
    
    return result_df
