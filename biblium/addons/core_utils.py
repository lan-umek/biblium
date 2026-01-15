# -*- coding: utf-8 -*-
"""
Core Utilities Bridge for Biblium Addons

This module provides a bridge to core biblium functionality,
allowing addons to use existing implementations where available.

Usage in addons:
    from biblium.addons.core_utils import get_core_function, CORE_AVAILABLE
    
    if CORE_AVAILABLE:
        h_idx = get_core_function('h_index')(citations)
    else:
        # Fallback implementation
        h_idx = local_h_index(citations)
"""

import warnings

# Try to import core biblium
CORE_AVAILABLE = False
CORE_UTILSBIB = None
CORE_PLOTBIB = None
CORE_BIBSTATS = None

try:
    from biblium import utilsbib as CORE_UTILSBIB
    CORE_AVAILABLE = True
except ImportError:
    pass

try:
    from biblium import plotbib as CORE_PLOTBIB
except ImportError:
    pass

try:
    from biblium import bibstats as CORE_BIBSTATS
except ImportError:
    pass


# =============================================================================
# FUNCTION REGISTRY - Maps addon function names to core equivalents
# =============================================================================

CORE_FUNCTION_MAP = {
    # Impact metrics (utilsbib)
    'h_index': ('utilsbib', 'h_index'),
    'g_index': ('utilsbib', 'g_index'),
    'hg_index': ('utilsbib', 'hg_index'),
    'tapered_h_index': ('utilsbib', 'tapered_h_index'),
    'chi_index': ('utilsbib', 'chi_index'),
    'a_index': ('utilsbib', 'a_index'),
    'r_index': ('utilsbib', 'r_index'),
    'w_index': ('utilsbib', 'w_index'),
    't_index': ('utilsbib', 't_index'),
    'pi_index': ('utilsbib', 'pi_index'),
    'gini_index': ('utilsbib', 'gini_index'),
    
    # Sleeping beauty (utilsbib)
    'calculate_beauty_coefficient': ('utilsbib', 'calculate_beauty_coefficient'),
    'calculate_awakening_intensity': ('utilsbib', 'calculate_awakening_intensity'),
    'extract_sleeping_beauties': ('utilsbib', 'extract_sleeping_beauties'),
    
    # Text processing (utilsbib)
    'analyze_sentiment': ('utilsbib', 'analyze_sentiment'),
    'topic_modeling': ('utilsbib', 'topic_modeling'),
    'preprocess_keywords': ('utilsbib', 'preprocess_keywords'),
    'process_text_column': ('utilsbib', 'process_text_column'),
    
    # Clustering (utilsbib)
    'cluster_kmeans': ('utilsbib', 'cluster_kmeans'),
    'cluster_hierarchical': ('utilsbib', 'cluster_hierarchical'),
    'conceptual_structure_analysis': ('utilsbib', 'conceptual_structure_analysis'),
    
    # Country/geographic (utilsbib)
    'extract_countries_from_affiliations': ('utilsbib', 'extract_countries_from_affiliations'),
    'correct_country_name': ('utilsbib', 'correct_country_name'),
    'get_ca_country': ('utilsbib', 'get_ca_country'),
    
    # Network building (utilsbib)
    'build_citation_network': ('utilsbib', 'build_citation_network'),
    'build_historiograph': ('utilsbib', 'build_historiograph'),
    'louvain_partition': ('utilsbib', 'louvain_partition'),
    
    # Plotting (plotbib)
    'plot_lotka_distribution': ('plotbib', 'plot_lotka_distribution'),
    'plot_bradford_distribution': ('plotbib', 'plot_bradford_distribution'),
    'plot_bradford_zones': ('plotbib', 'plot_bradford_zones'),
    'plot_zipf_distribution': ('plotbib', 'plot_zipf_distribution'),
    'plot_heatmap': ('plotbib', 'plot_heatmap'),
    'plot_wordcloud': ('plotbib', 'plot_wordcloud'),
}


def get_core_function(name: str):
    """
    Get a function from core biblium by name.
    
    Parameters
    ----------
    name : str
        Function name (e.g., 'h_index', 'analyze_sentiment')
    
    Returns
    -------
    callable or None
        The function if available, None otherwise.
    
    Example
    -------
    >>> h_index_func = get_core_function('h_index')
    >>> if h_index_func:
    ...     result = h_index_func(citations_list)
    """
    if not CORE_AVAILABLE:
        return None
    
    if name not in CORE_FUNCTION_MAP:
        return None
    
    module_name, func_name = CORE_FUNCTION_MAP[name]
    
    if module_name == 'utilsbib' and CORE_UTILSBIB:
        return getattr(CORE_UTILSBIB, func_name, None)
    elif module_name == 'plotbib' and CORE_PLOTBIB:
        return getattr(CORE_PLOTBIB, func_name, None)
    elif module_name == 'bibstats' and CORE_BIBSTATS:
        return getattr(CORE_BIBSTATS, func_name, None)
    
    return None


def use_core_or_fallback(core_name: str, fallback_func, *args, **kwargs):
    """
    Try to use core function, fall back to provided function if unavailable.
    
    Parameters
    ----------
    core_name : str
        Name of core function to try.
    fallback_func : callable
        Fallback function if core unavailable.
    *args, **kwargs
        Arguments to pass to the function.
    
    Returns
    -------
    Any
        Result from core or fallback function.
    """
    core_func = get_core_function(core_name)
    if core_func is not None:
        try:
            return core_func(*args, **kwargs)
        except Exception as e:
            warnings.warn(f"Core function {core_name} failed: {e}. Using fallback.")
    
    return fallback_func(*args, **kwargs)


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================

def get_h_index(citations):
    """Calculate h-index using core or fallback."""
    core_func = get_core_function('h_index')
    if core_func:
        return core_func(citations)
    
    # Fallback implementation
    citations_sorted = sorted(citations, reverse=True)
    h = 0
    for i, c in enumerate(citations_sorted, 1):
        if c >= i:
            h = i
        else:
            break
    return h


def get_sentiment(text, **kwargs):
    """Analyze sentiment using core or fallback."""
    core_func = get_core_function('analyze_sentiment')
    if core_func:
        import pandas as pd
        df = pd.DataFrame({'text': [text]})
        result = core_func(df, column='text', **kwargs)
        return result
    return None


def get_countries_from_affiliations(df, aff_col='Affiliations', **kwargs):
    """Extract countries using core or fallback."""
    core_func = get_core_function('extract_countries_from_affiliations')
    if core_func:
        return core_func(df, column=aff_col, **kwargs)
    return None


# =============================================================================
# AVAILABILITY CHECKS
# =============================================================================

def check_core_available():
    """Check if core biblium is available and print status."""
    print(f"Core biblium available: {CORE_AVAILABLE}")
    if CORE_AVAILABLE:
        print(f"  - utilsbib: {'Yes' if CORE_UTILSBIB else 'No'}")
        print(f"  - plotbib: {'Yes' if CORE_PLOTBIB else 'No'}")
        print(f"  - bibstats: {'Yes' if CORE_BIBSTATS else 'No'}")
    return CORE_AVAILABLE


def list_available_core_functions():
    """List all core functions that can be used from addons."""
    available = []
    for name in CORE_FUNCTION_MAP:
        if get_core_function(name) is not None:
            available.append(name)
    return available
