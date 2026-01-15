# -*- coding: utf-8 -*-
"""
Lazy loading utilities for optional dependencies.

This module provides lazy import functionality to speed up the initial
`import biblium` by deferring heavy dependencies until they're actually used.

Usage
-----
Instead of:
    import sklearn
    from sklearn.cluster import KMeans

Use:
    from biblium.lazy import sklearn
    # sklearn is not imported yet
    
    kmeans = sklearn.cluster.KMeans()  # Now sklearn.cluster is imported

Or for specific submodules:
    from biblium.lazy import get_sklearn_cluster
    KMeans = get_sklearn_cluster().KMeans
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Optional


class LazyModule:
    """
    A lazy module loader that defers import until first attribute access.
    
    Parameters
    ----------
    module_name : str
        Full module name to import (e.g., "sklearn.cluster").
    package : str, optional
        Package for relative imports.
        
    Examples
    --------
    >>> sklearn = LazyModule("sklearn")
    >>> # sklearn not imported yet
    >>> kmeans = sklearn.cluster.KMeans()  # Now imported
    """
    
    def __init__(self, module_name: str, package: Optional[str] = None):
        self._module_name = module_name
        self._package = package
        self._module = None
    
    def _load(self) -> Any:
        """Load the module if not already loaded."""
        if self._module is None:
            self._module = importlib.import_module(self._module_name, self._package)
        return self._module
    
    def __getattr__(self, name: str) -> Any:
        """Load module on first attribute access."""
        module = self._load()
        return getattr(module, name)
    
    def __dir__(self) -> list:
        """Return module's directory after loading."""
        return dir(self._load())
    
    def __repr__(self) -> str:
        if self._module is None:
            return f"<LazyModule '{self._module_name}' (not loaded)>"
        return f"<LazyModule '{self._module_name}' (loaded)>"


class LazySubmoduleLoader:
    """
    Loader for specific submodules with caching.
    
    This is used for cases where we need specific submodules
    like sklearn.cluster or scipy.stats.
    """
    
    _cache: Dict[str, Any] = {}
    
    @classmethod
    def get(cls, module_name: str) -> Any:
        """
        Get a module, loading it if necessary.
        
        Parameters
        ----------
        module_name : str
            Full module path (e.g., "sklearn.cluster").
            
        Returns
        -------
        module
            The imported module.
        """
        if module_name not in cls._cache:
            cls._cache[module_name] = importlib.import_module(module_name)
        return cls._cache[module_name]
    
    @classmethod
    def is_loaded(cls, module_name: str) -> bool:
        """Check if a module is already loaded."""
        return module_name in cls._cache
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the module cache."""
        cls._cache.clear()


# =============================================================================
# LAZY MODULE INSTANCES
# =============================================================================

# These are created but not imported until first use
sklearn = LazyModule("sklearn")
scipy = LazyModule("scipy")
nltk = LazyModule("nltk")
networkx = LazyModule("networkx")
igraph = LazyModule("igraph")
prince = LazyModule("prince")
umap = LazyModule("umap")
wordcloud = LazyModule("wordcloud")
matplotlib = LazyModule("matplotlib")
seaborn = LazyModule("seaborn")


# =============================================================================
# SUBMODULE GETTERS
# =============================================================================

def get_sklearn_cluster():
    """Get sklearn.cluster module."""
    return LazySubmoduleLoader.get("sklearn.cluster")


def get_sklearn_decomposition():
    """Get sklearn.decomposition module."""
    return LazySubmoduleLoader.get("sklearn.decomposition")


def get_sklearn_feature_extraction():
    """Get sklearn.feature_extraction.text module."""
    return LazySubmoduleLoader.get("sklearn.feature_extraction.text")


def get_sklearn_metrics():
    """Get sklearn.metrics module."""
    return LazySubmoduleLoader.get("sklearn.metrics")


def get_sklearn_preprocessing():
    """Get sklearn.preprocessing module."""
    return LazySubmoduleLoader.get("sklearn.preprocessing")


def get_scipy_stats():
    """Get scipy.stats module."""
    return LazySubmoduleLoader.get("scipy.stats")


def get_scipy_sparse():
    """Get scipy.sparse module."""
    return LazySubmoduleLoader.get("scipy.sparse")


def get_scipy_cluster():
    """Get scipy.cluster.hierarchy module."""
    return LazySubmoduleLoader.get("scipy.cluster.hierarchy")


def get_scipy_spatial():
    """Get scipy.spatial.distance module."""
    return LazySubmoduleLoader.get("scipy.spatial.distance")


def get_nltk_sentiment():
    """Get nltk.sentiment module."""
    return LazySubmoduleLoader.get("nltk.sentiment")


def get_nltk_stem():
    """Get nltk.stem module."""
    return LazySubmoduleLoader.get("nltk.stem")


def get_nltk_tokenize():
    """Get nltk.tokenize module."""
    return LazySubmoduleLoader.get("nltk.tokenize")


def get_networkx():
    """Get networkx module."""
    return LazySubmoduleLoader.get("networkx")


def get_matplotlib_pyplot():
    """Get matplotlib.pyplot module."""
    return LazySubmoduleLoader.get("matplotlib.pyplot")


def get_seaborn():
    """Get seaborn module."""
    return LazySubmoduleLoader.get("seaborn")


# =============================================================================
# OPTIONAL DEPENDENCY CHECKERS
# =============================================================================

_OPTIONAL_AVAILABLE: Dict[str, Optional[bool]] = {
    "igraph": None,
    "prince": None,
    "umap": None,
    "wordcloud": None,
    "community": None,  # python-louvain
    "bertopic": None,
}


def is_available(package_name: str) -> bool:
    """
    Check if an optional package is available.
    
    Parameters
    ----------
    package_name : str
        Name of the package to check.
        
    Returns
    -------
    bool
        True if the package can be imported.
    """
    if package_name not in _OPTIONAL_AVAILABLE:
        _OPTIONAL_AVAILABLE[package_name] = None
    
    if _OPTIONAL_AVAILABLE[package_name] is None:
        try:
            importlib.import_module(package_name)
            _OPTIONAL_AVAILABLE[package_name] = True
        except ImportError:
            _OPTIONAL_AVAILABLE[package_name] = False
    
    return _OPTIONAL_AVAILABLE[package_name]


def require(package_name: str, feature: str = "this feature") -> Any:
    """
    Import a package or raise a helpful error.
    
    Parameters
    ----------
    package_name : str
        Name of the package to import.
    feature : str
        Description of the feature requiring this package.
        
    Returns
    -------
    module
        The imported module.
        
    Raises
    ------
    ImportError
        If the package is not available.
    """
    if not is_available(package_name):
        raise ImportError(
            f"The '{package_name}' package is required for {feature}. "
            f"Install it with: pip install {package_name}"
        )
    return importlib.import_module(package_name)


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON PATTERNS
# =============================================================================

def import_sklearn_if_needed(*submodules: str) -> Dict[str, Any]:
    """
    Import sklearn submodules only when needed.
    
    Parameters
    ----------
    *submodules : str
        Submodule names like "cluster", "decomposition", "metrics".
        
    Returns
    -------
    dict
        Mapping of submodule names to module objects.
        
    Examples
    --------
    >>> mods = import_sklearn_if_needed("cluster", "metrics")
    >>> KMeans = mods["cluster"].KMeans
    """
    result = {}
    for name in submodules:
        result[name] = LazySubmoduleLoader.get(f"sklearn.{name}")
    return result


def import_scipy_if_needed(*submodules: str) -> Dict[str, Any]:
    """
    Import scipy submodules only when needed.
    
    Parameters
    ----------
    *submodules : str
        Submodule names like "stats", "sparse", "cluster.hierarchy".
        
    Returns
    -------
    dict
        Mapping of submodule names to module objects.
    """
    result = {}
    for name in submodules:
        full_name = f"scipy.{name}"
        result[name] = LazySubmoduleLoader.get(full_name)
    return result
