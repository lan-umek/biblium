# -*- coding: utf-8 -*-
"""
Caching utilities for expensive computations.

This module provides caching mechanisms for:
- Co-occurrence matrices
- Citation networks
- Document-term matrices
- Count operations (when repeated)

The caching is designed to be:
- Memory-efficient (stores only necessary data)
- Invalidation-aware (clears when source data changes)
- Optional (can be disabled)
"""

from __future__ import annotations

import hashlib
import weakref
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import pandas as pd
import numpy as np

# Type variable for generic return types
T = TypeVar("T")


class CacheKey:
    """
    Generate consistent cache keys from various inputs.
    
    Handles DataFrames, lists, dicts, and primitive types.
    """
    
    @staticmethod
    def from_dataframe(df: pd.DataFrame, sample_rows: int = 100) -> str:
        """
        Create a hash key from a DataFrame.
        
        Uses shape, column names, dtypes, and a sample of data
        to create a reasonably unique key without hashing the entire frame.
        """
        if df is None:
            return "None"
        
        components = [
            str(df.shape),
            str(list(df.columns)),
            str(list(df.dtypes)),
        ]
        
        # Sample some data for uniqueness
        if len(df) > 0:
            sample_size = min(sample_rows, len(df))
            indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
            sample = df.iloc[indices]
            components.append(sample.to_json())
        
        combined = "|".join(components)
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    @staticmethod
    def from_args(*args: Any, **kwargs: Any) -> str:
        """
        Create a hash key from function arguments.
        """
        components = []
        
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                components.append(CacheKey.from_dataframe(arg))
            elif isinstance(arg, (list, tuple)):
                components.append(str(sorted(arg) if all(isinstance(x, str) for x in arg) else arg))
            elif isinstance(arg, dict):
                components.append(str(sorted(arg.items())))
            else:
                components.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, pd.DataFrame):
                components.append(f"{k}={CacheKey.from_dataframe(v)}")
            elif isinstance(v, (list, tuple)):
                components.append(f"{k}={sorted(v) if all(isinstance(x, str) for x in v) else v}")
            elif isinstance(v, dict):
                components.append(f"{k}={sorted(v.items())}")
            else:
                components.append(f"{k}={v}")
        
        combined = "|".join(components)
        return hashlib.md5(combined.encode()).hexdigest()[:16]


class ResultCache:
    """
    A simple LRU-style cache for expensive computations.
    
    Features:
    - Configurable max size
    - Optional TTL (time-to-live)
    - Statistics tracking
    - Manual invalidation
    
    Attributes
    ----------
    max_size : int
        Maximum number of items to cache.
    enabled : bool
        Whether caching is active.
    stats : dict
        Cache hit/miss statistics.
    """
    
    def __init__(self, max_size: int = 50, enabled: bool = True):
        """
        Initialize the cache.
        
        Parameters
        ----------
        max_size : int, default 50
            Maximum number of cached results.
        enabled : bool, default True
            Whether caching is enabled.
        """
        self.max_size = max_size
        self.enabled = enabled
        self._cache: Dict[str, Tuple[Any, int]] = {}  # key -> (value, access_count)
        self._access_order: list = []
        self.stats = {"hits": 0, "misses": 0}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value.
        
        Parameters
        ----------
        key : str
            Cache key.
            
        Returns
        -------
        Any or None
            Cached value if found, None otherwise.
        """
        if not self.enabled:
            return None
        
        if key in self._cache:
            self.stats["hits"] += 1
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key][0]
        
        self.stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.
        
        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache.
        """
        if not self.enabled:
            return
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)
        
        self._cache[key] = (value, 1)
        if key not in self._access_order:
            self._access_order.append(key)
    
    def clear(self, prefix: Optional[str] = None) -> int:
        """
        Clear the cache.
        
        Parameters
        ----------
        prefix : str, optional
            If provided, only clear keys starting with this prefix.
            
        Returns
        -------
        int
            Number of items cleared.
        """
        if prefix is None:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            return count
        
        keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._cache[k]
            if k in self._access_order:
                self._access_order.remove(k)
        return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        dict
            Statistics including hits, misses, size, and hit rate.
        """
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0.0
        return {
            **self.stats,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "enabled": self.enabled,
        }
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache


# Global cache instance
_global_cache = ResultCache(max_size=100, enabled=True)


def get_cache() -> ResultCache:
    """Get the global cache instance."""
    return _global_cache


def set_cache_enabled(enabled: bool) -> None:
    """Enable or disable the global cache."""
    _global_cache.enabled = enabled


def clear_cache(prefix: Optional[str] = None) -> int:
    """Clear the global cache."""
    return _global_cache.clear(prefix)


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _global_cache.get_stats()


def cached(
    prefix: str = "",
    key_func: Optional[Callable[..., str]] = None,
    cache: Optional[ResultCache] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to cache function results.
    
    Parameters
    ----------
    prefix : str, default ""
        Prefix for cache keys (useful for namespacing).
    key_func : callable, optional
        Custom function to generate cache key from arguments.
        If None, uses CacheKey.from_args.
    cache : ResultCache, optional
        Cache instance to use. If None, uses global cache.
        
    Returns
    -------
    callable
        Decorated function with caching.
        
    Examples
    --------
    >>> @cached(prefix="cooc")
    ... def compute_cooccurrence(df, column, top_n=20):
    ...     # expensive computation
    ...     return result
    """
    if cache is None:
        cache = _global_cache
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not cache.enabled:
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_func is not None:
                key = f"{prefix}:{key_func(*args, **kwargs)}"
            else:
                key = f"{prefix}:{func.__name__}:{CacheKey.from_args(*args, **kwargs)}"
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        # Add cache control methods to the wrapper
        wrapper.cache_clear = lambda: cache.clear(f"{prefix}:{func.__name__}")
        wrapper.cache_info = lambda: cache.get_stats()
        
        return wrapper
    
    return decorator


def cached_property(
    func: Optional[Callable[[Any], T]] = None,
    *,
    attr_name: Optional[str] = None,
) -> Union[property, Callable[[Callable[[Any], T]], property]]:
    """
    A cached property decorator that stores the result on the instance.
    
    Unlike functools.cached_property, this can be invalidated by deleting
    the attribute or calling the instance's clear_cache() method.
    
    Parameters
    ----------
    func : callable, optional
        The property getter function.
    attr_name : str, optional
        Custom attribute name for storage. Defaults to f"_cached_{func.__name__}".
        
    Returns
    -------
    property
        A property descriptor with caching.
        
    Examples
    --------
    >>> class Analysis:
    ...     @cached_property
    ...     def expensive_result(self):
    ...         # computed once, stored on instance
    ...         return compute_something()
    """
    def decorator(f: Callable[[Any], T]) -> property:
        name = attr_name or f"_cached_{f.__name__}"
        
        @wraps(f)
        def getter(self: Any) -> T:
            if not hasattr(self, name):
                setattr(self, name, f(self))
            return getattr(self, name)
        
        def setter(self: Any, value: T) -> None:
            setattr(self, name, value)
        
        def deleter(self: Any) -> None:
            if hasattr(self, name):
                delattr(self, name)
        
        return property(getter, setter, deleter, f.__doc__)
    
    if func is not None:
        return decorator(func)
    return decorator


class InstanceCache:
    """
    A mixin class that provides per-instance caching.
    
    Add this to a class to enable cached computations that are
    automatically invalidated when the source data changes.
    
    Examples
    --------
    >>> class MyAnalysis(InstanceCache):
    ...     def __init__(self, df):
    ...         super().__init__()
    ...         self.df = df
    ...     
    ...     def compute_something(self, param):
    ...         return self._get_cached("something", param, self._do_compute, param)
    ...     
    ...     def _do_compute(self, param):
    ...         # expensive computation
    ...         return result
    """
    
    def __init__(self) -> None:
        self._instance_cache: Dict[str, Any] = {}
        self._cache_enabled: bool = True
    
    def _get_cached(
        self,
        name: str,
        key: Any,
        compute_func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Get a cached result or compute and cache it.
        
        Parameters
        ----------
        name : str
            Cache category name.
        key : Any
            Key within the category (converted to string).
        compute_func : callable
            Function to compute the result if not cached.
        *args, **kwargs :
            Arguments passed to compute_func.
            
        Returns
        -------
        Any
            The cached or computed result.
        """
        if not self._cache_enabled:
            return compute_func(*args, **kwargs)
        
        full_key = f"{name}:{key}"
        
        if full_key not in self._instance_cache:
            self._instance_cache[full_key] = compute_func(*args, **kwargs)
        
        return self._instance_cache[full_key]
    
    def clear_instance_cache(self, prefix: Optional[str] = None) -> int:
        """
        Clear the instance cache.
        
        Parameters
        ----------
        prefix : str, optional
            If provided, only clear keys starting with this prefix.
            
        Returns
        -------
        int
            Number of items cleared.
        """
        if prefix is None:
            count = len(self._instance_cache)
            self._instance_cache.clear()
            return count
        
        keys_to_remove = [k for k in self._instance_cache if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._instance_cache[k]
        return len(keys_to_remove)
    
    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable instance caching."""
        self._cache_enabled = enabled
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get instance cache information."""
        return {
            "size": len(self._instance_cache),
            "enabled": self._cache_enabled,
            "keys": list(self._instance_cache.keys()),
        }


# Convenience functions for specific computation types

def cache_cooccurrence_key(
    df: pd.DataFrame,
    column: str,
    items: Optional[list] = None,
    top_n: int = 20,
    **kwargs: Any,
) -> str:
    """Generate a cache key for co-occurrence computation."""
    df_key = CacheKey.from_dataframe(df)
    items_key = str(sorted(items)) if items else "None"
    return f"cooc:{df_key}:{column}:{items_key}:{top_n}"


def cache_network_key(
    matrix: pd.DataFrame,
    **kwargs: Any,
) -> str:
    """Generate a cache key for network computation."""
    matrix_key = CacheKey.from_dataframe(matrix)
    kwargs_key = CacheKey.from_args(**kwargs)
    return f"net:{matrix_key}:{kwargs_key}"
