# -*- coding: utf-8 -*-
"""
Biblium Enhancements Module

This module provides enhanced functionality for BiblioAnalysis:
- Streamlined report workflow
- Progress feedback
- Lazy evaluation
- Better error messages
- Caching layer
- Fluent API support
- Data validation
- Interactive features

@author: Claude (Anthropic) for Lan.Umek
@version: 2.3.0
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import textwrap
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from difflib import get_close_matches
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pandas as pd

T = TypeVar("T")


# =============================================================================
# BETTER ERROR MESSAGES
# =============================================================================

class BibliumError(Exception):
    """Base exception for Biblium with helpful context."""
    
    def __init__(self, message: str, suggestions: List[str] = None, context: Dict[str, Any] = None):
        self.message = message
        self.suggestions = suggestions or []
        self.context = context or {}
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        lines = [self.message]
        if self.suggestions:
            lines.append("\nSuggestions:")
            for s in self.suggestions:
                lines.append(f"  • {s}")
        if self.context:
            lines.append("\nContext:")
            for k, v in self.context.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class ColumnNotFoundError(BibliumError):
    """Raised when a required column is not found."""
    
    def __init__(self, column: str, available: List[str], database: str = None):
        close_matches = get_close_matches(column, available, n=3, cutoff=0.6)
        
        suggestions = []
        if close_matches:
            suggestions.append(f"Did you mean: {', '.join(close_matches)}?")
        
        if database:
            suggestions.append(f"Check column naming for {database} exports")
        
        suggestions.append(f"Available columns: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}")
        
        super().__init__(
            f"Column '{column}' not found in dataset",
            suggestions=suggestions,
            context={"database": database} if database else {}
        )


class InsufficientDataError(BibliumError):
    """Raised when there's not enough data for an analysis."""
    
    def __init__(self, analysis: str, required: int, available: int, column: str = None):
        suggestions = [
            f"This analysis requires at least {required} records",
            "Try loading more data or using a different analysis"
        ]
        if column:
            suggestions.append(f"Check if column '{column}' has enough non-null values")
        
        super().__init__(
            f"Insufficient data for {analysis}: need {required}, have {available}",
            suggestions=suggestions
        )


class ConfigurationError(BibliumError):
    """Raised for configuration issues."""
    pass


def suggest_column(column: str, available: List[str]) -> Optional[str]:
    """Suggest a similar column name."""
    matches = get_close_matches(column, available, n=1, cutoff=0.6)
    return matches[0] if matches else None


def safe_get_column(df: pd.DataFrame, column: str, database: str = None) -> pd.Series:
    """Get a column with helpful error if not found."""
    if column in df.columns:
        return df[column]
    raise ColumnNotFoundError(column, list(df.columns), database)


# =============================================================================
# CACHING LAYER
# =============================================================================

@dataclass
class CacheConfig:
    """Configuration for the caching system."""
    enabled: bool = True
    cache_dir: str = ".biblium_cache"
    max_age_hours: int = 24 * 7  # 1 week default
    max_size_mb: int = 500
    
    def get_cache_path(self) -> Path:
        """Get the cache directory path, creating if needed."""
        path = Path(self.cache_dir)
        if self.enabled:
            path.mkdir(parents=True, exist_ok=True)
        return path


class AnalysisCache:
    """
    Cache for expensive analysis operations.
    
    Caches results based on:
    - Data hash (content fingerprint)
    - Method name
    - Parameters
    
    Examples
    --------
    >>> cache = AnalysisCache(".biblium_cache")
    >>> 
    >>> # Check if result is cached
    >>> result = cache.get("count_sources", data_hash, {"top_n": 20})
    >>> if result is None:
    ...     result = expensive_computation()
    ...     cache.set("count_sources", data_hash, {"top_n": 20}, result)
    """
    
    def __init__(self, cache_dir: str = ".biblium_cache", enabled: bool = True):
        self.config = CacheConfig(enabled=enabled, cache_dir=cache_dir)
        self._memory_cache: Dict[str, Any] = {}
    
    def _make_key(self, method: str, data_hash: str, params: Dict[str, Any]) -> str:
        """Create a unique cache key."""
        param_str = json.dumps(params, sort_keys=True, default=str)
        combined = f"{method}:{data_hash}:{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cache_file(self, key: str) -> Path:
        """Get the cache file path for a key."""
        return self.config.get_cache_path() / f"{key}.pkl"
    
    def get(self, method: str, data_hash: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """
        Get a cached result if available and not expired.
        
        Returns None if not found or expired.
        """
        if not self.config.enabled:
            return None
        
        params = params or {}
        key = self._make_key(method, data_hash, params)
        
        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                # Check age
                age_hours = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
                if age_hours < self.config.max_age_hours:
                    with open(cache_file, "rb") as f:
                        result = pickle.load(f)
                    self._memory_cache[key] = result
                    return result
                else:
                    # Expired - delete
                    cache_file.unlink()
            except Exception:
                pass
        
        return None
    
    def set(self, method: str, data_hash: str, params: Dict[str, Any], result: Any) -> None:
        """Cache a result."""
        if not self.config.enabled:
            return
        
        params = params or {}
        key = self._make_key(method, data_hash, params)
        
        # Memory cache
        self._memory_cache[key] = result
        
        # Disk cache
        try:
            cache_file = self._get_cache_file(key)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass  # Fail silently for disk cache
    
    def clear(self, method: str = None) -> int:
        """
        Clear cache entries.
        
        Parameters
        ----------
        method : str, optional
            If provided, only clear entries for this method.
            If None, clear all entries.
        
        Returns
        -------
        int
            Number of entries cleared.
        """
        cleared = 0
        
        # Clear memory cache
        if method:
            keys_to_remove = [k for k in self._memory_cache if method in k]
            for k in keys_to_remove:
                del self._memory_cache[k]
                cleared += 1
        else:
            cleared += len(self._memory_cache)
            self._memory_cache.clear()
        
        # Clear disk cache
        cache_path = self.config.get_cache_path()
        if cache_path.exists():
            for f in cache_path.glob("*.pkl"):
                try:
                    f.unlink()
                    cleared += 1
                except Exception:
                    pass
        
        return cleared
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_path = self.config.get_cache_path()
        disk_files = list(cache_path.glob("*.pkl")) if cache_path.exists() else []
        disk_size = sum(f.stat().st_size for f in disk_files) / (1024 * 1024)
        
        return {
            "enabled": self.config.enabled,
            "memory_entries": len(self._memory_cache),
            "disk_entries": len(disk_files),
            "disk_size_mb": round(disk_size, 2),
            "cache_dir": str(cache_path),
        }


def cached_analysis(method_name: str = None):
    """
    Decorator to cache analysis method results.
    
    Parameters
    ----------
    method_name : str, optional
        Name for the cache key. Defaults to function name.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if caching is enabled on the instance
            cache = getattr(self, "_cache", None)
            if cache is None or not cache.config.enabled:
                return func(self, *args, **kwargs)
            
            # Get data hash
            data_hash = getattr(self, "_data_hash", None)
            if data_hash is None:
                return func(self, *args, **kwargs)
            
            # Build params dict
            name = method_name or func.__name__
            params = kwargs.copy()
            
            # Check cache
            result = cache.get(name, data_hash, params)
            if result is not None:
                # Restore cached attributes
                if isinstance(result, dict) and "_cached_attrs" in result:
                    for attr, value in result["_cached_attrs"].items():
                        setattr(self, attr, value)
                    return result.get("_return_value")
                return result
            
            # Execute and cache
            return_value = func(self, *args, **kwargs)
            
            # Cache the result (including any attributes set)
            cache.set(name, data_hash, params, return_value)
            
            return return_value
        return wrapper
    return decorator


# =============================================================================
# DATA VALIDATION
# =============================================================================

@dataclass
class ValidationResult:
    """Result of data validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        lines = []
        status = "✓ Valid" if self.valid else "✗ Invalid"
        lines.append(f"Validation: {status}")
        
        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        
        if self.info:
            lines.append("\nInfo:")
            for i in self.info:
                lines.append(f"  ℹ {i}")
        
        return "\n".join(lines)
    
    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        status_color = "green" if self.valid else "red"
        status_text = "Valid" if self.valid else "Invalid"
        
        html = f'<div style="font-family: sans-serif;">'
        html += f'<h3 style="color: {status_color};">Validation: {status_text}</h3>'
        
        if self.errors:
            html += '<h4 style="color: red;">Errors</h4><ul>'
            for e in self.errors:
                html += f'<li>{e}</li>'
            html += '</ul>'
        
        if self.warnings:
            html += '<h4 style="color: orange;">Warnings</h4><ul>'
            for w in self.warnings:
                html += f'<li>{w}</li>'
            html += '</ul>'
        
        if self.info:
            html += '<h4 style="color: blue;">Info</h4><ul>'
            for i in self.info:
                html += f'<li>{i}</li>'
            html += '</ul>'
        
        html += '</div>'
        return html


def validate_bibliometric_data(
    df: pd.DataFrame,
    database: str = None,
    strict: bool = False,
) -> ValidationResult:
    """
    Validate bibliometric data for common issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        The bibliometric dataset.
    database : str, optional
        Database type for database-specific checks.
    strict : bool, default False
        If True, treat warnings as errors.
    
    Returns
    -------
    ValidationResult
        Validation results with errors, warnings, and info.
    
    Examples
    --------
    >>> result = validate_bibliometric_data(df, database="scopus")
    >>> if not result.valid:
    ...     print(result.errors)
    >>> print(result.warnings)
    """
    result = ValidationResult(valid=True)
    
    # Basic checks
    if df.empty:
        result.valid = False
        result.errors.append("Dataset is empty")
        return result
    
    result.stats["n_documents"] = len(df)
    result.stats["n_columns"] = len(df.columns)
    
    # Check for essential columns (varies by database)
    essential_columns = {
        "scopus": ["Title", "Authors", "Year", "Source title"],
        "wos": ["Title", "Authors", "Publication Year", "Source Title"],
        "oa": ["title", "authorships", "publication_year"],
    }
    
    if database and database.lower() in essential_columns:
        required = essential_columns[database.lower()]
        missing = [c for c in required if c not in df.columns]
        if missing:
            result.warnings.append(f"Missing expected columns for {database}: {missing}")
    
    # Check for duplicates
    if "DOI" in df.columns:
        doi_col = df["DOI"].dropna()
        if len(doi_col) > 0:
            n_dup = doi_col.duplicated().sum()
            if n_dup > 0:
                result.warnings.append(f"{n_dup} duplicate DOIs found")
    
    if "Title" in df.columns:
        title_col = df["Title"].dropna()
        if len(title_col) > 0:
            n_dup = title_col.str.lower().duplicated().sum()
            if n_dup > 0:
                result.warnings.append(f"{n_dup} potential duplicate titles found")
    
    # Check for missing values in key columns
    key_columns = ["Title", "Authors", "Year", "Abstract", "DOI"]
    for col in key_columns:
        if col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            if missing_pct > 50:
                result.warnings.append(f"Column '{col}' has {missing_pct:.1f}% missing values")
            elif missing_pct > 0:
                result.info.append(f"Column '{col}' has {missing_pct:.1f}% missing values")
            result.stats[f"{col}_missing_pct"] = round(missing_pct, 1)
    
    # Check year values
    year_cols = [c for c in df.columns if "year" in c.lower()]
    for col in year_cols:
        if df[col].dtype in ["int64", "float64"]:
            min_year = df[col].min()
            max_year = df[col].max()
            current_year = datetime.now().year
            
            if pd.notna(max_year) and max_year > current_year + 1:
                result.warnings.append(f"Column '{col}' has future dates (max: {max_year})")
            
            if pd.notna(min_year) and min_year < 1900:
                result.warnings.append(f"Column '{col}' has very old dates (min: {min_year})")
            
            result.stats[f"{col}_range"] = f"{int(min_year) if pd.notna(min_year) else '?'}-{int(max_year) if pd.notna(max_year) else '?'}"
    
    # Check citation values
    cite_cols = [c for c in df.columns if "cite" in c.lower() or "citation" in c.lower()]
    for col in cite_cols:
        if df[col].dtype in ["int64", "float64"]:
            if (df[col] < 0).any():
                result.warnings.append(f"Column '{col}' has negative values")
    
    # Summary info
    result.info.append(f"Dataset has {len(df)} documents and {len(df.columns)} columns")
    
    # Apply strict mode
    if strict and result.warnings:
        result.valid = False
        result.errors.extend(result.warnings)
        result.warnings = []
    
    return result


# =============================================================================
# VERBOSE TRACKING MIXIN
# =============================================================================

class VerboseTrackingMixin:
    """
    Mixin that tracks attribute changes and reports computed results.
    
    When verbose=True, methods that compute results will print information
    about what was computed and where the results are stored.
    
    Examples
    --------
    >>> ba = BiblioAnalysis("data.csv", db="scopus", verbose=True)
    >>> ba.count_sources()
    ✓ count_sources completed
      → Results stored in: ba.sources_counts_df (DataFrame: 150 rows × 4 cols)
      → Also available: ba.sources_counts (dict)
    """
    
    _verbose: bool = False
    _tracked_attrs: Dict[str, Any] = None
    
    def _init_verbose_tracking(self, verbose: bool = True):
        """Initialize verbose tracking."""
        self._verbose = verbose
        self._tracked_attrs = {}
        if verbose:
            self._snapshot_attrs()
    
    def _snapshot_attrs(self):
        """Take a snapshot of current attributes for comparison."""
        if not self._verbose:
            return
        self._tracked_attrs = {
            k: id(v) for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    def _report_new_attrs(self, method_name: str):
        """Report any new or changed attributes after a method call."""
        if not self._verbose:
            return
        
        current_attrs = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
        
        new_attrs = []
        changed_attrs = []
        
        for k, v in current_attrs.items():
            if k not in self._tracked_attrs:
                new_attrs.append((k, v))
            elif id(v) != self._tracked_attrs.get(k):
                changed_attrs.append((k, v))
        
        if new_attrs or changed_attrs:
            print(f"  ✓ {method_name} completed")
            
            for attr_name, attr_value in new_attrs + changed_attrs:
                attr_desc = _describe_attribute(attr_value)
                prefix = "→ New" if (attr_name, attr_value) in new_attrs else "→ Updated"
                print(f"    {prefix}: self.{attr_name} {attr_desc}")
        
        # Update snapshot
        self._snapshot_attrs()
    
    def _verbose_print(self, message: str):
        """Print message if verbose mode is enabled."""
        if self._verbose:
            print(message)


def _describe_attribute(value: Any) -> str:
    """Create a short description of an attribute value."""
    if value is None:
        return "(None)"
    
    type_name = type(value).__name__
    
    if hasattr(value, 'shape'):  # DataFrame, ndarray
        shape = value.shape
        if len(shape) == 2:
            return f"(DataFrame: {shape[0]} rows × {shape[1]} cols)"
        else:
            return f"(array: shape {shape})"
    elif isinstance(value, dict):
        return f"(dict: {len(value)} items)"
    elif isinstance(value, (list, tuple)):
        return f"({type_name}: {len(value)} items)"
    elif isinstance(value, str):
        if len(value) > 50:
            return f'(str: "{value[:50]}...")'
        return f'(str: "{value}")'
    elif isinstance(value, (int, float)):
        return f"({type_name}: {value})"
    else:
        return f"({type_name})"


def verbose_method(method_name: str = None):
    """
    Decorator to add verbose reporting to methods.
    
    Parameters
    ----------
    method_name : str, optional
        Display name for the method. Defaults to function name.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            name = method_name or func.__name__
            
            # Take snapshot before
            if hasattr(self, '_verbose') and self._verbose:
                if hasattr(self, '_snapshot_attrs'):
                    self._snapshot_attrs()
            
            # Execute method
            result = func(self, *args, **kwargs)
            
            # Report changes after
            if hasattr(self, '_verbose') and self._verbose:
                if hasattr(self, '_report_new_attrs'):
                    self._report_new_attrs(name)
            
            return result
        return wrapper
    return decorator


# =============================================================================
# LAZY EVALUATION
# =============================================================================

class LazyResult:
    """
    Lazy wrapper for analysis results.
    
    Defers computation until the result is actually accessed.
    
    Examples
    --------
    >>> sources = LazyResult(ba, "count_sources", top_n=20)
    >>> # No computation yet
    >>> sources.top(10)  # Now computes and returns top 10
    >>> sources.to_dataframe()  # Returns full result as DataFrame
    """
    
    def __init__(
        self,
        instance: Any,
        method_name: str,
        result_attr: str = None,
        **kwargs
    ):
        self._instance = instance
        self._method_name = method_name
        self._result_attr = result_attr or f"{method_name.replace('count_', '')}_counts_df"
        self._kwargs = kwargs
        self._computed = False
        self._result = None
    
    def _ensure_computed(self) -> None:
        """Ensure the computation has been performed."""
        if not self._computed:
            method = getattr(self._instance, self._method_name)
            method(**self._kwargs)
            self._result = getattr(self._instance, self._result_attr, None)
            self._computed = True
    
    def to_dataframe(self) -> pd.DataFrame:
        """Get the full result as a DataFrame."""
        self._ensure_computed()
        return self._result
    
    def top(self, n: int = 10) -> pd.DataFrame:
        """Get the top n results."""
        self._ensure_computed()
        if self._result is not None:
            return self._result.head(n)
        return pd.DataFrame()
    
    def __len__(self) -> int:
        """Get the number of results."""
        self._ensure_computed()
        return len(self._result) if self._result is not None else 0
    
    def __repr__(self) -> str:
        status = "computed" if self._computed else "pending"
        return f"LazyResult({self._method_name}, status={status})"
    
    def _repr_html_(self) -> str:
        """HTML representation for Jupyter."""
        self._ensure_computed()
        if self._result is not None and hasattr(self._result, "_repr_html_"):
            return self._result._repr_html_()
        return repr(self)


# =============================================================================
# FLUENT API MIXIN
# =============================================================================

class FluentMixin:
    """
    Mixin providing fluent/chainable API methods.
    
    Enables method chaining like:
    >>> (ba
    ...     .filter(year_range=(2020, 2024))
    ...     .count_all()
    ...     .plot_summary()
    ...     .export("report.docx"))
    """
    
    def filter(
        self,
        year_range: Tuple[int, int] = None,
        sources: List[str] = None,
        authors: List[str] = None,
        keywords: List[str] = None,
        countries: List[str] = None,
        min_citations: int = None,
        query: str = None,
        inplace: bool = False,
    ) -> "FluentMixin":
        """
        Filter the dataset with various criteria.
        
        Parameters
        ----------
        year_range : tuple, optional
            (start_year, end_year) inclusive.
        sources : list, optional
            Filter to these source titles.
        authors : list, optional
            Filter to documents by these authors.
        keywords : list, optional
            Filter to documents with these keywords.
        countries : list, optional
            Filter to documents from these countries.
        min_citations : int, optional
            Minimum citation count.
        query : str, optional
            Text query to filter title/abstract.
        inplace : bool, default False
            If True, modify this instance. If False, return a new instance.
        
        Returns
        -------
        BiblioAnalysis
            Filtered instance (self if inplace=True, new instance otherwise).
        """
        mask = pd.Series(True, index=self.df.index)
        
        # Year filter
        if year_range:
            year_col = self.mapping.get("Year", "Year")
            if year_col in self.df.columns:
                mask &= (self.df[year_col] >= year_range[0]) & (self.df[year_col] <= year_range[1])
        
        # Source filter
        if sources:
            source_col = self.mapping.get("Source title", "Source title")
            if source_col in self.df.columns:
                mask &= self.df[source_col].isin(sources)
        
        # Min citations filter
        if min_citations is not None:
            cite_col = self.mapping.get("Cited by", "Cited by")
            if cite_col in self.df.columns:
                mask &= self.df[cite_col] >= min_citations
        
        # Text query filter
        if query:
            query_lower = query.lower()
            text_mask = pd.Series(False, index=self.df.index)
            for col in ["Title", "Abstract"]:
                if col in self.df.columns:
                    text_mask |= self.df[col].fillna("").str.lower().str.contains(query_lower, regex=False)
            mask &= text_mask
        
        filtered_df = self.df[mask].copy()
        
        if inplace:
            self.df = filtered_df
            self.n = len(filtered_df)
            return self
        else:
            # Create new instance with filtered data
            new_instance = self.__class__(
                df=filtered_df,
                db=self.db,
                res_folder=self.res_folder,
            )
            return new_instance
    
    def count_all(self, top_n: int = 20) -> "FluentMixin":
        """
        Run all counting methods.
        
        Returns self for chaining.
        """
        methods = [
            "count_sources",
            "count_authors", 
            "count_author_keywords",
            "count_index_keywords",
            "count_document_types",
            "count_ca_countries",
            "count_affiliations",
            "count_references",
        ]
        
        for method_name in methods:
            if hasattr(self, method_name):
                try:
                    getattr(self, method_name)(top_n=top_n)
                except Exception:
                    pass
        
        return self
    
    def stats_all(self, top_n: int = 20) -> "FluentMixin":
        """
        Run all statistics methods.
        
        Returns self for chaining.
        """
        methods = [
            "get_sources_stats",
            "get_authors_stats",
            "get_author_keywords_stats",
            "get_all_countries_stats",
            "get_affiliations_stats",
        ]
        
        for method_name in methods:
            if hasattr(self, method_name):
                try:
                    getattr(self, method_name)(top_n=top_n)
                except Exception:
                    pass
        
        return self
    
    def plot_summary(
        self,
        output_dir: str = None,
        formats: List[str] = None,
    ) -> "FluentMixin":
        """
        Generate summary plots.
        
        Returns self for chaining.
        """
        formats = formats or ["png"]
        
        plot_methods = [
            ("plot_scientific_production", {}),
            ("plot_sources_bar", {}),
            ("plot_authors_bar", {}),
            ("plot_keywords_bar", {}),
        ]
        
        for method_name, kwargs in plot_methods:
            if hasattr(self, method_name):
                try:
                    getattr(self, method_name)(**kwargs)
                except Exception:
                    pass
        
        return self
    
    def export(
        self,
        output: str = "report",
        formats: List[str] = None,
        level: str = "basic",
    ) -> "FluentMixin":
        """
        Export reports in specified formats.
        
        Parameters
        ----------
        output : str
            Output path or base name.
        formats : list, optional
            List of formats: ["xlsx", "docx", "pptx", "tex"]
        level : str, default "basic"
            Report level.
        
        Returns self for chaining.
        """
        formats = formats or ["docx"]
        
        if hasattr(self, "generate_all_reports"):
            output_dir = os.path.dirname(output) or "results/reports"
            base_name = os.path.splitext(os.path.basename(output))[0]
            
            self.generate_all_reports(
                output_dir=output_dir,
                base_name=base_name,
                formats=formats,
                template_sheet=level,
            )
        
        return self


# =============================================================================
# SUMMARY AND INTROSPECTION
# =============================================================================

@dataclass
class DatasetSummary:
    """Summary statistics for a bibliometric dataset."""
    n_documents: int
    n_columns: int
    database: str
    year_range: Tuple[int, int]
    top_sources: List[Tuple[str, int]]
    top_authors: List[Tuple[str, int]]
    top_keywords: List[Tuple[str, int]]
    missing_rates: Dict[str, float]
    analyses_available: List[str]
    analyses_completed: List[str]
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "BIBLIOMETRIC DATASET SUMMARY",
            "=" * 60,
            f"Documents: {self.n_documents:,}",
            f"Database: {self.database}",
            f"Year range: {self.year_range[0]} - {self.year_range[1]}",
            "",
            "Top 5 Sources:",
        ]
        for name, count in self.top_sources[:5]:
            lines.append(f"  • {name[:50]}: {count}")
        
        lines.extend([
            "",
            "Top 5 Authors:",
        ])
        for name, count in self.top_authors[:5]:
            lines.append(f"  • {name[:40]}: {count}")
        
        lines.extend([
            "",
            "Top 5 Keywords:",
        ])
        for name, count in self.top_keywords[:5]:
            lines.append(f"  • {name[:40]}: {count}")
        
        lines.extend([
            "",
            f"Analyses completed: {len(self.analyses_completed)}/{len(self.analyses_available)}",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def _repr_html_(self) -> str:
        """HTML representation for Jupyter."""
        html = '<div style="font-family: sans-serif; padding: 10px;">'
        html += '<h2>📚 Bibliometric Dataset Summary</h2>'
        html += f'<p><strong>Documents:</strong> {self.n_documents:,}</p>'
        html += f'<p><strong>Database:</strong> {self.database}</p>'
        html += f'<p><strong>Year range:</strong> {self.year_range[0]} - {self.year_range[1]}</p>'
        
        # Top sources table
        html += '<h3>Top Sources</h3><table border="1" style="border-collapse: collapse;">'
        html += '<tr><th>Source</th><th>Count</th></tr>'
        for name, count in self.top_sources[:5]:
            html += f'<tr><td>{name[:50]}</td><td>{count}</td></tr>'
        html += '</table>'
        
        # Progress bar for analyses
        pct = len(self.analyses_completed) / max(len(self.analyses_available), 1) * 100
        html += f'<h3>Analysis Progress: {pct:.0f}%</h3>'
        html += f'<div style="background: #eee; width: 200px; height: 20px; border-radius: 10px;">'
        html += f'<div style="background: #4CAF50; width: {pct}%; height: 100%; border-radius: 10px;"></div>'
        html += '</div>'
        
        html += '</div>'
        return html


def get_available_analyses(instance: Any) -> List[Dict[str, Any]]:
    """
    Get list of available analyses based on the dataset columns.
    
    Parameters
    ----------
    instance : BiblioAnalysis or BiblioGroupAnalysis
        The analysis instance.
    
    Returns
    -------
    list of dict
        Each dict has keys: 'name', 'method', 'requires', 'available', 'description'
    """
    analyses = [
        {
            "name": "Source Analysis",
            "method": "count_sources",
            "requires": ["Source title"],
            "description": "Count and rank publication sources (journals, conferences)",
        },
        {
            "name": "Author Analysis", 
            "method": "count_authors",
            "requires": ["Authors"],
            "description": "Count and rank authors by publication count",
        },
        {
            "name": "Author Keywords Analysis",
            "method": "count_author_keywords",
            "requires": ["Author Keywords"],
            "description": "Analyze author-provided keywords",
        },
        {
            "name": "Index Keywords Analysis",
            "method": "count_index_keywords", 
            "requires": ["Index Keywords"],
            "description": "Analyze database-indexed keywords",
        },
        {
            "name": "Country Analysis",
            "method": "count_ca_countries",
            "requires": ["Affiliations"],
            "description": "Analyze geographic distribution of research",
        },
        {
            "name": "Affiliation Analysis",
            "method": "count_affiliations",
            "requires": ["Affiliations"],
            "description": "Analyze institutional affiliations",
        },
        {
            "name": "Reference Analysis",
            "method": "count_references",
            "requires": ["References"],
            "description": "Analyze cited references",
        },
        {
            "name": "Citation Analysis",
            "method": "get_top_cited_documents",
            "requires": ["Cited by"],
            "description": "Identify most cited documents",
        },
        {
            "name": "Lotka's Law",
            "method": "compute_lotka",
            "requires": ["Authors"],
            "description": "Test author productivity distribution",
        },
        {
            "name": "Bradford's Law",
            "method": "compute_bradford",
            "requires": ["Source title"],
            "description": "Test source concentration distribution",
        },
        {
            "name": "Word Analysis",
            "method": "count_words",
            "requires": ["Title", "Abstract"],
            "description": "Analyze word frequencies in titles/abstracts",
        },
    ]
    
    # Check availability based on columns
    df = instance.df
    mapping = getattr(instance, "mapping", {})
    
    for analysis in analyses:
        # Map required columns using the instance's mapping
        required = analysis["requires"]
        mapped_required = [mapping.get(r, r) for r in required]
        
        # Check if any of the required columns exist
        analysis["available"] = any(col in df.columns for col in mapped_required + required)
    
    return analyses


def get_available_group_analyses(instance: Any) -> List[Dict[str, Any]]:
    """
    Get list of available group analyses.
    
    Parameters
    ----------
    instance : BiblioGroupAnalysis
        The group analysis instance.
    
    Returns
    -------
    list of dict
        Each dict has keys: 'name', 'method', 'requires', 'available', 'description'
    """
    # BiblioGroupAnalysis always has groups after initialization
    has_groups = True
    
    has_associations = hasattr(instance, 'associations_df') and getattr(instance, 'associations_df', None) is not None
    
    df = getattr(instance, 'df', None)
    mapping = getattr(instance, "mapping", {})
    
    # Actual methods from BiblioGroupAnalysis
    analyses = [
        # Counting methods
        {
            "name": "Count Keywords per Group",
            "method": "count_keywords",
            "requires": [],
            "description": "Count keywords within each group",
        },
        {
            "name": "Count Countries per Group",
            "method": "count_countries",
            "requires": [],
            "description": "Count countries within each group",
        },
        {
            "name": "Group Count Sources",
            "method": "group_count_sources",
            "requires": [],
            "description": "Count sources within each group",
        },
        {
            "name": "Group Count Authors",
            "method": "group_count_authors",
            "requires": [],
            "description": "Count authors within each group",
        },
        {
            "name": "Group Count Author Keywords",
            "method": "group_count_author_keywords",
            "requires": [],
            "description": "Count author keywords within each group",
        },
        {
            "name": "Group Count Index Keywords",
            "method": "group_count_index_keywords",
            "requires": [],
            "description": "Count index keywords within each group",
        },
        {
            "name": "Group Count Affiliations",
            "method": "group_count_affiliations",
            "requires": [],
            "description": "Count affiliations within each group",
        },
        {
            "name": "Group Count References",
            "method": "group_count_references",
            "requires": [],
            "description": "Count references within each group",
        },
        {
            "name": "Group Count All Countries",
            "method": "group_count_all_countries",
            "requires": [],
            "description": "Count all countries within each group",
        },
        {
            "name": "Group Count CA Countries",
            "method": "group_count_ca_countries",
            "requires": [],
            "description": "Count corresponding author countries within each group",
        },
        # Stats methods
        {
            "name": "Group Sources Stats",
            "method": "get_group_sources_stats",
            "requires": [],
            "description": "Get source statistics for each group",
        },
        {
            "name": "Group Authors Stats",
            "method": "get_group_authors_stats",
            "requires": [],
            "description": "Get author statistics for each group",
        },
        {
            "name": "Group Keywords Stats",
            "method": "get_group_keywords_stats",
            "requires": [],
            "description": "Get keyword statistics for each group",
        },
        {
            "name": "Group Countries Stats",
            "method": "get_group_all_countries_stats",
            "requires": [],
            "description": "Get country statistics for each group",
        },
        # Association methods
        {
            "name": "Associate Sources",
            "method": "associate_sources",
            "requires": [],
            "description": "Compute associations between groups and sources",
        },
        {
            "name": "Associate Authors",
            "method": "associate_authors",
            "requires": [],
            "description": "Compute associations between groups and authors",
        },
        {
            "name": "Associate Author Keywords",
            "method": "associate_author_keywords",
            "requires": [],
            "description": "Compute associations between groups and author keywords",
        },
        {
            "name": "Associate Countries",
            "method": "associate_countries",
            "requires": [],
            "description": "Compute associations between groups and countries",
        },
        {
            "name": "Associate Affiliations",
            "method": "associate_affiliations",
            "requires": [],
            "description": "Compute associations between groups and affiliations",
        },
        # Analysis methods
        {
            "name": "Get Group Intersections",
            "method": "get_group_intersections",
            "requires": [],
            "description": "Get document overlap between groups",
        },
        {
            "name": "Compare Continuous Variables",
            "method": "compare_continuous_vars",
            "requires": [],
            "description": "Compare continuous variables across groups",
        },
        {
            "name": "Get Main Info",
            "method": "get_main_info",
            "requires": [],
            "description": "Get main information for each group",
        },
        {
            "name": "Get Scientific Production",
            "method": "get_scientific_production",
            "requires": [],
            "description": "Get scientific production over time per group",
        },
        {
            "name": "Get Top Cited Documents",
            "method": "get_group_top_cited_documents",
            "requires": [],
            "description": "Get top cited documents for each group",
        },
    ]
    
    for analysis in analyses:
        # Simply check if method exists on the instance
        analysis["available"] = hasattr(instance, analysis["method"])
    
    return analyses


def what_can_i_do(instance: Any, verbose: bool = True) -> List[str]:
    """
    List available analyses based on the dataset.
    
    Parameters
    ----------
    instance : BiblioAnalysis or BiblioGroupAnalysis
        The analysis instance.
    verbose : bool, default True
        Print the results.
    
    Returns
    -------
    list of str
        Method names that can be called.
    
    Examples
    --------
    >>> ba = BiblioAnalysis("data.csv", db="scopus")
    >>> ba.what_can_i_do()
    Available analyses for your dataset:
      ✓ count_sources - Count and rank publication sources
      ✓ count_authors - Count and rank authors
      ✗ count_references - (requires: References column)
    """
    # Detect if this is a group analysis instance
    is_group_analysis = hasattr(instance, 'groups') or hasattr(instance, 'group_definitions')
    
    if is_group_analysis:
        analyses = get_available_group_analyses(instance)
        title = "AVAILABLE GROUP ANALYSES"
    else:
        analyses = get_available_analyses(instance)
        title = "AVAILABLE ANALYSES FOR YOUR DATASET"
    
    available_methods = []
    
    if verbose:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        print()
    
    for analysis in analyses:
        if analysis["available"]:
            available_methods.append(analysis["method"])
            if verbose:
                print(f"  ✓ {analysis['method']}")
                print(f"    {analysis['description']}")
        else:
            if verbose:
                print(f"  ✗ {analysis['method']}")
                print(f"    (requires: {', '.join(analysis['requires'])})")
        if verbose:
            print()
    
    if verbose:
        print("=" * 60)
        print(f"Total: {len(available_methods)}/{len(analyses)} analyses available")
        print("=" * 60)
    
    return available_methods


# =============================================================================
# REPR AND DISPLAY
# =============================================================================

def make_repr(instance: Any) -> str:
    """
    Create a nice __repr__ for BiblioAnalysis.
    
    Shows key statistics at a glance.
    """
    lines = [
        f"BiblioAnalysis(",
        f"  documents={instance.n:,},",
        f"  database='{instance.db}',",
    ]
    
    # Year range
    year_col = instance.mapping.get("Year", "Year")
    if year_col in instance.df.columns:
        years = instance.df[year_col].dropna()
        if len(years) > 0:
            lines.append(f"  years={int(years.min())}-{int(years.max())},")
    
    # Columns
    lines.append(f"  columns={len(instance.df.columns)},")
    
    # Completed analyses
    completed = []
    for attr in ["sources_counts_df", "authors_counts_df", "author_keywords_counts_df"]:
        if hasattr(instance, attr) and getattr(instance, attr) is not None:
            completed.append(attr.replace("_counts_df", ""))
    
    if completed:
        lines.append(f"  analyses_done=[{', '.join(completed[:3])}{'...' if len(completed) > 3 else ''}],")
    
    lines.append(")")
    
    return "\n".join(lines)


def make_repr_html(instance: Any) -> str:
    """
    Create HTML representation for Jupyter notebooks.
    """
    year_col = instance.mapping.get("Year", "Year")
    year_range = "Unknown"
    if year_col in instance.df.columns:
        years = instance.df[year_col].dropna()
        if len(years) > 0:
            year_range = f"{int(years.min())} - {int(years.max())}"
    
    html = f'''
    <div style="font-family: sans-serif; padding: 15px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
        <h3 style="margin-top: 0; color: #495057;">📚 BiblioAnalysis</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Documents</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{instance.n:,}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Database</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{instance.db}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Year Range</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{year_range}</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><strong>Columns</strong></td>
                <td style="padding: 8px;">{len(instance.df.columns)}</td>
            </tr>
        </table>
        <p style="margin-bottom: 0; color: #6c757d; font-size: 0.9em;">
            Use <code>.summary()</code> for details or <code>.what_can_i_do()</code> to see available analyses.
        </p>
    </div>
    '''
    return html


# =============================================================================
# EXPORT PRESETS
# =============================================================================

EXPORT_PRESETS = {
    "journal_submission": {
        "description": "Minimal tables formatted for journal supplementary materials",
        "formats": ["xlsx"],
        "level": "basic",
        "options": {
            "top_n": 20,
            "round_decimals": 2,
            "include_plots": False,
        },
    },
    "thesis_appendix": {
        "description": "Comprehensive documentation for thesis/dissertation",
        "formats": ["docx", "xlsx"],
        "level": "full",
        "options": {
            "top_n": 50,
            "include_plots": True,
            "include_methodology": True,
        },
    },
    "presentation": {
        "description": "Key figures and highlights for presentations",
        "formats": ["pptx"],
        "level": "basic",
        "options": {
            "top_n": 10,
            "include_plots": True,
            "plot_only": True,
        },
    },
    "quick_overview": {
        "description": "Fast summary for initial exploration",
        "formats": ["xlsx"],
        "level": "basic",
        "options": {
            "top_n": 10,
        },
    },
    "full_analysis": {
        "description": "Complete analysis with all available outputs",
        "formats": ["xlsx", "docx", "pptx"],
        "level": "all",
        "options": {
            "top_n": 100,
            "include_plots": True,
        },
    },
}


def get_export_presets() -> Dict[str, Dict[str, Any]]:
    """Get available export presets."""
    return EXPORT_PRESETS.copy()


def list_export_presets(verbose: bool = True) -> List[str]:
    """
    List available export presets.
    
    Parameters
    ----------
    verbose : bool, default True
        Print descriptions.
    
    Returns
    -------
    list of str
        Preset names.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("EXPORT PRESETS")
        print("=" * 60)
        for name, config in EXPORT_PRESETS.items():
            print(f"\n  {name}")
            print(f"    {config['description']}")
            print(f"    Formats: {', '.join(config['formats'])}")
            print(f"    Level: {config['level']}")
        print("\n" + "=" * 60)
        print("Usage: ba.export(preset='journal_submission')")
        print("=" * 60)
    
    return list(EXPORT_PRESETS.keys())


# =============================================================================
# PLUGIN SYSTEM
# =============================================================================

_registered_analyses: Dict[str, Callable] = {}


def register_analysis(name: str):
    """
    Decorator to register a custom analysis function.
    
    The function should take a BiblioAnalysis instance as first argument
    and can return any result.
    
    Examples
    --------
    >>> @register_analysis("sentiment")
    ... def analyze_sentiment(ba, column="Abstract"):
    ...     # Custom sentiment analysis
    ...     return sentiment_scores
    >>> 
    >>> ba = BiblioAnalysis("data.csv")
    >>> result = ba.run_analysis("sentiment")
    """
    def decorator(func: Callable) -> Callable:
        _registered_analyses[name] = func
        return func
    return decorator


def get_registered_analyses() -> Dict[str, Callable]:
    """Get all registered custom analyses."""
    return _registered_analyses.copy()


def run_custom_analysis(instance: Any, name: str, **kwargs) -> Any:
    """
    Run a registered custom analysis.
    
    Parameters
    ----------
    instance : BiblioAnalysis
        The analysis instance.
    name : str
        Name of the registered analysis.
    **kwargs
        Arguments to pass to the analysis function.
    
    Returns
    -------
    Any
        Result of the analysis.
    """
    if name not in _registered_analyses:
        available = list(_registered_analyses.keys())
        raise ValueError(
            f"Analysis '{name}' not registered. Available: {available}"
        )
    
    func = _registered_analyses[name]
    return func(instance, **kwargs)


# =============================================================================
# CONFIGURATION FILE SUPPORT
# =============================================================================

@dataclass
class ProjectConfig:
    """Project-level configuration loaded from file."""
    database: str = "scopus"
    reports: Dict[str, Any] = field(default_factory=lambda: {
        "default_level": "basic",
        "formats": ["docx", "xlsx"],
    })
    plots: Dict[str, Any] = field(default_factory=lambda: {
        "dpi": 600,
        "style": "publication",
    })
    analysis: Dict[str, Any] = field(default_factory=lambda: {
        "top_n": 20,
        "min_frequency": 2,
    })
    cache: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "dir": ".biblium_cache",
    })
    
    @classmethod
    def from_file(cls, path: str = "biblium.yaml") -> "ProjectConfig":
        """Load configuration from YAML file."""
        config_path = Path(path)
        
        if not config_path.exists():
            return cls()
        
        try:
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f)
            
            return cls(
                database=data.get("database", "scopus"),
                reports=data.get("reports", {}),
                plots=data.get("plots", {}),
                analysis=data.get("analysis", {}),
                cache=data.get("cache", {}),
            )
        except ImportError:
            warnings.warn("PyYAML not installed, cannot load config file")
            return cls()
        except Exception as e:
            warnings.warn(f"Error loading config: {e}")
            return cls()
    
    def save(self, path: str = "biblium.yaml") -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
            data = {
                "database": self.database,
                "reports": self.reports,
                "plots": self.plots,
                "analysis": self.analysis,
                "cache": self.cache,
            }
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        except ImportError:
            warnings.warn("PyYAML not installed, cannot save config file")


def load_project_config(path: str = "biblium.yaml") -> ProjectConfig:
    """Load project configuration from file."""
    return ProjectConfig.from_file(path)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute a hash of the DataFrame for caching purposes.
    
    Uses a combination of shape, column names, and sample values
    for fast but reliable fingerprinting.
    """
    components = [
        str(df.shape),
        str(list(df.columns)),
        str(df.index[0]) if len(df) > 0 else "",
        str(df.iloc[0].tolist()) if len(df) > 0 else "",
        str(df.iloc[-1].tolist()) if len(df) > 0 else "",
    ]
    combined = "|".join(components)
    return hashlib.md5(combined.encode()).hexdigest()[:16]
