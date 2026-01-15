# -*- coding: utf-8 -*-
"""
Statistical utilities - bibliometric indices, laws, and metrics.

This module contains:
- h_index, g_index, and other citation indices
- Bradford, Lotka, Zipf law computations
- Performance indicators
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import curve_fit


# =============================================================================
# CITATION INDICES
# =============================================================================

def h_index(citations: List[int], alpha: float = 1.0) -> int:
    """
    Compute the h-index from a list of citation counts.

    Parameters
    ----------
    citations : list
        List of citation counts.
    alpha : float
        Scaling factor (default 1.0 for standard h-index).

    Returns
    -------
    int
        The h-index value.
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    h = 0
    for i, c in enumerate(citations, 1):
        if c >= i * alpha:
            h = i
        else:
            break
    return h


def g_index(citations: List[int]) -> int:
    """
    Compute the g-index from a list of citation counts.

    The g-index is the largest number g such that the top g papers
    have together at least g² citations.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    int
        The g-index value.
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    cumsum = 0
    g = 0
    for i, c in enumerate(citations, 1):
        cumsum += c
        if cumsum >= i * i:
            g = i
        else:
            break
    return g


def hg_index(citations: List[int]) -> float:
    """
    Compute the hg-index (geometric mean of h and g).

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The hg-index value.
    """
    h = h_index(citations)
    g = g_index(citations)
    return np.sqrt(h * g)


def c_index(
    citations: List[int],
    thresholds: List[int] = None
) -> Dict[str, int]:
    """
    Compute c-indices for various citation thresholds.

    Parameters
    ----------
    citations : list
        List of citation counts.
    thresholds : list
        Thresholds for counting (default: [1, 5, 10, 20, 50, 100, 1000]).

    Returns
    -------
    dict
        Dictionary with threshold keys and counts.
    """
    if thresholds is None:
        thresholds = [1, 5, 10, 20, 50, 100, 1000]
    
    citations = [c for c in citations if c is not None]
    return {f"C{t}": sum(1 for c in citations if c >= t) for t in thresholds}


def tapered_h_index(citations: List[int]) -> float:
    """
    Compute the tapered h-index.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The tapered h-index value.
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    h = h_index(citations)
    if h == 0:
        return 0.0
    
    total = 0.0
    for i, c in enumerate(citations[:h], 1):
        total += min(c, i) / i
    return total


def chi_index(citations: List[int]) -> float:
    """
    Compute the chi-index.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The chi-index value.
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    if not citations:
        return 0.0
    
    h = h_index(citations)
    if h == 0:
        return 0.0
    
    return sum(citations[:h]) / h


def a_index(citations: List[int]) -> float:
    """
    Compute the A-index (average citations of h-core papers).

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The A-index value.
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    h = h_index(citations)
    if h == 0:
        return 0.0
    return sum(citations[:h]) / h


def r_index(citations: List[int]) -> float:
    """
    Compute the R-index (square root of total citations in h-core).

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The R-index value.
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    h = h_index(citations)
    if h == 0:
        return 0.0
    return np.sqrt(sum(citations[:h]))


def h2_index(citations: List[int]) -> float:
    """
    Compute the h2-index (h-index squared divided by total citations).

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The h2-index value.
    """
    h = h_index(citations)
    total = sum(c for c in citations if c is not None)
    if total == 0:
        return 0.0
    return (h * h) / total


def w_index(citations: List[int]) -> int:
    """
    Compute the w-index.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    int
        The w-index value.
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    w = 0
    for i, c in enumerate(citations, 1):
        if c >= 10 * i:
            w = i
        else:
            break
    return w


def gini_index(citations: List[int]) -> float:
    """
    Compute the Gini index for citation distribution.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The Gini index (0 = perfect equality, 1 = perfect inequality).
    """
    citations = [c for c in citations if c is not None]
    if not citations or sum(citations) == 0:
        return 0.0
    
    citations = sorted(citations)
    n = len(citations)
    cumsum = np.cumsum(citations)
    return (2 * sum((i + 1) * c for i, c in enumerate(citations)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def m_index(citations: List[int]) -> float:
    """
    Compute the m-index (median of h-core citations).
    
    The m-index is the median number of citations received by papers 
    in the Hirsch core. Uses median instead of mean for robustness
    against skewed distributions.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The m-index value.
    
    References
    ----------
    Bornmann et al. (2008) "Are there better indices for evaluation 
    purposes than the h index?"
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    h = h_index(citations)
    if h == 0:
        return 0.0
    return float(np.median(citations[:h]))


def ar_index(citations: List[int], ages: List[int] = None) -> float:
    """
    Compute the AR-index (age-weighted R-index).
    
    The AR-index accounts for the age of publications, giving more weight
    to recent citations. Unlike h-index, it can decrease over time if
    recent performance drops.

    Parameters
    ----------
    citations : list
        List of citation counts.
    ages : list, optional
        List of paper ages (years since publication). If None,
        assumes all papers are 1 year old.

    Returns
    -------
    float
        The AR-index value.
    
    References
    ----------
    Jin et al. (2007) "The R- and AR-indices: Complementing the h-index"
    """
    citations = [c for c in citations if c is not None]
    if ages is None:
        ages = [1] * len(citations)
    
    # Sort by citations descending
    pairs = sorted(zip(citations, ages), key=lambda x: x[0], reverse=True)
    citations = [p[0] for p in pairs]
    ages = [max(1, p[1]) for p in pairs]  # Ensure no division by zero
    
    h = h_index(citations)
    if h == 0:
        return 0.0
    
    return np.sqrt(sum(citations[i] / ages[i] for i in range(h)))


def m_quotient(citations: List[int], years_active: int) -> float:
    """
    Compute the m-quotient (h-index divided by career length).
    
    Proposed by Hirsch to compare scientists at different career stages.

    Parameters
    ----------
    citations : list
        List of citation counts.
    years_active : int
        Number of years since first publication.

    Returns
    -------
    float
        The m-quotient value.
    
    References
    ----------
    Hirsch (2005) "An index to quantify an individual's scientific 
    research output"
    """
    if years_active <= 0:
        return 0.0
    h = h_index(citations)
    return h / years_active


def h2_upper_index(citations: List[int]) -> int:
    """
    Compute the h(2)-index (Kosmulski's h-squared index).
    
    A scientist's h(2)-index is the highest number such that h(2) of 
    their papers have at least h(2)² citations each.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    int
        The h(2)-index value.
    
    References
    ----------
    Kosmulski (2006) "A new Hirsch-type index saves time and works 
    equally well as the original h-index"
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    h2 = 0
    for i, c in enumerate(citations, 1):
        if c >= i * i:
            h2 = i
        else:
            break
    return h2


def e_index(citations: List[int]) -> float:
    """
    Compute the e-index (excess citations in h-core).
    
    The e-index captures the excess citations ignored by h-index.
    It is independent of h-index, unlike most other variants.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The e-index value.
    
    References
    ----------
    Zhang (2009) "The e-Index, Complementing the h-Index for Excess Citations"
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    h = h_index(citations)
    if h == 0:
        return 0.0
    
    excess = sum(citations[i] - h for i in range(h))
    return np.sqrt(excess)


def q2_index(citations: List[int]) -> float:
    """
    Compute the q²-index (geometric mean of h and m indices).
    
    Combines quantitative (h-index) and qualitative (m-index) measures.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The q²-index value.
    
    References
    ----------
    Cabrerizo et al. (2010) "q2-Index: Quantitative and Qualitative 
    Evaluation Based on the Number and Impact of Papers in the Hirsch Core"
    """
    h = h_index(citations)
    m = m_index(citations)
    return np.sqrt(h * m)


def hw_index(citations: List[int]) -> float:
    """
    Compute the hw-index (weighted h-index).
    
    An h-index weighted by citation impact, sensitive to performance changes.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The hw-index value.
    
    References
    ----------
    Egghe & Rousseau (2008) "An h-index weighted by citation impact"
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    if not citations:
        return 0.0
    
    # Find r0: largest j where cumulative citations >= j²
    cumsum = 0
    r0 = 0
    for j, c in enumerate(citations, 1):
        cumsum += c
        if cumsum >= j:
            r0 = j
    
    if r0 == 0:
        return 0.0
    
    return np.sqrt(sum(citations[:r0]))


def pi_index(citations: List[int], n_papers: int = None) -> float:
    """
    Compute the π-index.
    
    Equal to 1/100 of citations received by the "elite set" of papers.
    Elite set size = (10 * log10(N)) - 10, where N is total papers.

    Parameters
    ----------
    citations : list
        List of citation counts.
    n_papers : int, optional
        Total number of papers. If None, uses len(citations).

    Returns
    -------
    float
        The π-index value.
    
    References
    ----------
    Vinkler (2009) "The π-index: a new indicator for assessing scientific impact"
    """
    citations = sorted([c for c in citations if c is not None], reverse=True)
    if n_papers is None:
        n_papers = len(citations)
    
    if n_papers <= 1:
        return 0.0
    
    # Elite set size
    elite_size = max(1, int(10 * np.log10(n_papers) - 10))
    elite_size = min(elite_size, len(citations))
    
    return sum(citations[:elite_size]) / 100


def i10_index(citations: List[int]) -> int:
    """
    Compute the i10-index (number of papers with at least 10 citations).
    
    Simple metric popularized by Google Scholar.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    int
        The i10-index value.
    """
    return sum(1 for c in citations if c is not None and c >= 10)


def i100_index(citations: List[int]) -> int:
    """
    Compute the i100-index (number of papers with at least 100 citations).

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    int
        The i100-index value.
    """
    return sum(1 for c in citations if c is not None and c >= 100)


def v_index(citations: List[int]) -> float:
    """
    Compute the v-index (percentage of papers forming h-index).

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The v-index value (0-100).
    
    References
    ----------
    Riikonen & Vihinen (2008) "National research contributions"
    """
    citations = [c for c in citations if c is not None]
    n = len(citations)
    if n == 0:
        return 0.0
    
    h = h_index(citations)
    return (h / n) * 100


def h_norm_index(citations: List[int]) -> float:
    """
    Compute the normalized h-index (h divided by number of papers).

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The normalized h-index value.
    
    References
    ----------
    Sidiropoulos et al. (2007) "Generalized Hirsch h-index"
    """
    citations = [c for c in citations if c is not None]
    n = len(citations)
    if n == 0:
        return 0.0
    
    h = h_index(citations)
    return h / n


def h_frac_index(citations: List[int], n_authors_per_paper: List[int] = None) -> float:
    """
    Compute the fractional h-index (accounts for co-authorship).
    
    Papers are counted fractionally by 1/number_of_authors.

    Parameters
    ----------
    citations : list
        List of citation counts.
    n_authors_per_paper : list, optional
        Number of authors for each paper. If None, assumes single author.

    Returns
    -------
    float
        The fractional h-index value.
    
    References
    ----------
    Schreiber (2008) "To share the fame in a fair way"
    """
    citations = [c for c in citations if c is not None]
    if not citations:
        return 0.0
    
    if n_authors_per_paper is None:
        n_authors_per_paper = [1] * len(citations)
    
    # Sort by citations descending
    pairs = sorted(zip(citations, n_authors_per_paper), key=lambda x: x[0], reverse=True)
    
    # Compute effective rank
    eff_rank = 0.0
    hm = 0.0
    for cit, n_auth in pairs:
        eff_rank += 1.0 / max(1, n_auth)
        if cit >= eff_rank:
            hm = eff_rank
        else:
            break
    
    return hm


def profit_index(citations: List[int]) -> float:
    """
    Compute the p-index (performance index).
    
    Balances activity (total citations) and excellence (mean citation rate).
    Defined as (C² / N)^(1/3) where C = total citations, N = number of papers.

    Parameters
    ----------
    citations : list
        List of citation counts.

    Returns
    -------
    float
        The p-index value.
    
    References
    ----------
    Prathap (2010) "The 100 most prolific economists using the p-index"
    """
    citations = [c for c in citations if c is not None]
    n = len(citations)
    if n == 0:
        return 0.0
    
    total_citations = sum(citations)
    return (total_citations ** 2 / n) ** (1/3)


def compute_all_h_indices(
    citations: List[int],
    ages: List[int] = None,
    n_authors: List[int] = None,
    years_active: int = None,
) -> Dict[str, float]:
    """
    Compute all available h-index variants.

    Parameters
    ----------
    citations : list
        List of citation counts per paper.
    ages : list, optional
        Paper ages in years (for AR-index).
    n_authors : list, optional
        Number of authors per paper (for fractional h-index).
    years_active : int, optional
        Career length in years (for m-quotient).

    Returns
    -------
    dict
        Dictionary with all computed indices.
    """
    citations = [c for c in citations if c is not None]
    if not citations:
        return {}
    
    results = {
        # Core indices
        "h_index": h_index(citations),
        "g_index": g_index(citations),
        "hg_index": hg_index(citations),
        
        # h-core based
        "a_index": a_index(citations),
        "m_index": m_index(citations),
        "r_index": r_index(citations),
        "e_index": e_index(citations),
        "q2_index": q2_index(citations),
        
        # Weighted/modified
        "h2_index": h2_upper_index(citations),
        "hw_index": hw_index(citations),
        "tapered_h": tapered_h_index(citations),
        
        # Other variants
        "w_index": w_index(citations),
        "pi_index": pi_index(citations),
        "profit_index": profit_index(citations),
        "h_norm": h_norm_index(citations),
        "v_index": v_index(citations),
        
        # Citation thresholds
        "i10_index": i10_index(citations),
        "i100_index": i100_index(citations),
        
        # Distribution
        "gini_index": gini_index(citations),
        
        # Summary stats
        "total_citations": sum(citations),
        "n_papers": len(citations),
        "mean_citations": np.mean(citations),
        "median_citations": np.median(citations),
        "max_citations": max(citations),
    }
    
    # Optional indices requiring additional data
    if ages is not None and len(ages) == len(citations):
        results["ar_index"] = ar_index(citations, ages)
    
    if n_authors is not None and len(n_authors) == len(citations):
        results["h_frac"] = h_frac_index(citations, n_authors)
    
    if years_active is not None and years_active > 0:
        results["m_quotient"] = m_quotient(citations, years_active)
    
    return results


# =============================================================================
# BIBLIOMETRIC LAWS
# =============================================================================

def compute_lotka_distribution(
    author_counts: pd.DataFrame,
    count_col: str = "Number of documents"
) -> pd.DataFrame:
    """
    Compute Lotka's law distribution.

    Parameters
    ----------
    author_counts : DataFrame
        DataFrame with author publication counts.
    count_col : str
        Column containing document counts.

    Returns
    -------
    DataFrame
        Distribution of authors by productivity.
    """
    counts = author_counts[count_col].value_counts().sort_index()
    total_authors = counts.sum()
    
    result = pd.DataFrame({
        "Documents": counts.index,
        "Authors": counts.values,
        "Proportion": counts.values / total_authors,
        "Cumulative Authors": counts.cumsum().values,
        "Cumulative Proportion": counts.cumsum().values / total_authors,
    })
    
    return result


def evaluate_lotka_fit(
    lotka_df: pd.DataFrame,
    doc_col: str = "Documents",
    author_col: str = "Authors"
) -> Dict[str, float]:
    """
    Evaluate fit to Lotka's law.

    Parameters
    ----------
    lotka_df : DataFrame
        Output from compute_lotka_distribution.
    doc_col : str
        Column with document counts.
    author_col : str
        Column with author counts.

    Returns
    -------
    dict
        Fit statistics including exponent and R².
    """
    x = np.log(lotka_df[doc_col].values)
    y = np.log(lotka_df[author_col].values)
    
    # Fit linear model in log-log space
    coeffs = np.polyfit(x, y, 1)
    exponent = -coeffs[0]
    
    # R² calculation
    y_pred = coeffs[0] * x + coeffs[1]
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "exponent": exponent,
        "r_squared": r_squared,
        "expected_exponent": 2.0,  # Lotka's original finding
    }


def compute_bradford_distribution(
    source_counts: pd.DataFrame,
    count_col: str = "Number of documents"
) -> pd.DataFrame:
    """
    Compute Bradford's law zones.

    Parameters
    ----------
    source_counts : DataFrame
        DataFrame with source publication counts.
    count_col : str
        Column containing document counts.

    Returns
    -------
    DataFrame
        Sources with Bradford zone assignments.
    """
    df = source_counts.sort_values(count_col, ascending=False).copy()
    df = df.reset_index(drop=True)
    
    total_docs = df[count_col].sum()
    df["Cumulative Docs"] = df[count_col].cumsum()
    df["Cumulative Proportion"] = df["Cumulative Docs"] / total_docs
    
    # Assign zones (roughly thirds by cumulative documents)
    df["Zone"] = pd.cut(
        df["Cumulative Proportion"],
        bins=[0, 0.333, 0.667, 1.0],
        labels=["Core", "Zone 2", "Zone 3"],
        include_lowest=True
    )
    
    return df


def compute_zipf_distribution_from_counts(
    counts_df: pd.DataFrame,
    count_col: str = "Number of documents"
) -> pd.DataFrame:
    """
    Prepare data for Zipf's law analysis.

    Parameters
    ----------
    counts_df : DataFrame
        DataFrame with item counts.
    count_col : str
        Column containing counts.

    Returns
    -------
    DataFrame
        Data with rank and frequency for Zipf analysis.
    """
    df = counts_df.sort_values(count_col, ascending=False).copy()
    df = df.reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    df["Log Rank"] = np.log(df["Rank"])
    df["Log Frequency"] = np.log(df[count_col])
    
    return df


def evaluate_zipf_fit(
    zipf_df: pd.DataFrame,
    rank_col: str = "Log Rank",
    freq_col: str = "Log Frequency"
) -> Dict[str, float]:
    """
    Evaluate fit to Zipf's law.

    Parameters
    ----------
    zipf_df : DataFrame
        Output from compute_zipf_distribution_from_counts.
    rank_col : str
        Column with log ranks.
    freq_col : str
        Column with log frequencies.

    Returns
    -------
    dict
        Fit statistics including exponent and R².
    """
    x = zipf_df[rank_col].values
    y = zipf_df[freq_col].values
    
    # Remove any infinite values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    
    if len(x) < 2:
        return {"exponent": np.nan, "r_squared": np.nan}
    
    coeffs = np.polyfit(x, y, 1)
    exponent = -coeffs[0]
    
    y_pred = coeffs[0] * x + coeffs[1]
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "exponent": exponent,
        "r_squared": r_squared,
        "expected_exponent": 1.0,  # Zipf's law
    }


def evaluate_prices_law(
    author_counts: pd.DataFrame,
    count_col: str = "Number of documents"
) -> Dict[str, Union[int, float]]:
    """
    Evaluate Price's law (square root of authors produce half of papers).

    Parameters
    ----------
    author_counts : DataFrame
        DataFrame with author publication counts.
    count_col : str
        Column containing document counts.

    Returns
    -------
    dict
        Price's law statistics.
    """
    n_authors = len(author_counts)
    total_docs = author_counts[count_col].sum()
    
    # Sort by productivity
    sorted_df = author_counts.sort_values(count_col, ascending=False)
    
    # Number predicted by Price's law
    n_elite = int(np.sqrt(n_authors))
    
    # Actual contribution of top sqrt(n) authors
    elite_docs = sorted_df[count_col].head(n_elite).sum()
    elite_proportion = elite_docs / total_docs if total_docs > 0 else 0
    
    return {
        "n_authors": n_authors,
        "n_elite_predicted": n_elite,
        "elite_proportion": elite_proportion,
        "expected_proportion": 0.5,
        "price_satisfied": elite_proportion >= 0.5,
    }


def evaluate_pareto_principle(
    counts_df: pd.DataFrame,
    count_col: str = "Number of documents"
) -> Dict[str, Union[int, float]]:
    """
    Evaluate the 80/20 Pareto principle.

    Parameters
    ----------
    counts_df : DataFrame
        DataFrame with item counts.
    count_col : str
        Column containing counts.

    Returns
    -------
    dict
        Pareto principle statistics.
    """
    n_items = len(counts_df)
    total = counts_df[count_col].sum()
    
    sorted_df = counts_df.sort_values(count_col, ascending=False)
    sorted_df["Cumulative"] = sorted_df[count_col].cumsum()
    sorted_df["Cumulative Prop"] = sorted_df["Cumulative"] / total
    
    # Find how many items produce 80% of output
    n_for_80 = (sorted_df["Cumulative Prop"] <= 0.8).sum() + 1
    prop_items_for_80 = n_for_80 / n_items if n_items > 0 else 0
    
    # Find what top 20% produces
    n_top_20 = max(1, int(n_items * 0.2))
    top_20_output = sorted_df[count_col].head(n_top_20).sum()
    top_20_proportion = top_20_output / total if total > 0 else 0
    
    return {
        "n_items": n_items,
        "n_items_for_80pct": n_for_80,
        "proportion_items_for_80pct": prop_items_for_80,
        "top_20pct_produces": top_20_proportion,
        "pareto_ratio": top_20_proportion / 0.8 if top_20_proportion > 0 else 0,
    }


# =============================================================================
# CITATION DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_citation_distribution(
    df: pd.DataFrame,
    citations_col: str = "Cited by",
) -> Dict:
    """
    Analyze citation distribution characteristics.
    
    Computes comprehensive statistics about the citation distribution including
    basic stats, percentiles, inequality measures (Gini, h-index), and 
    distribution fitting (log-normal).
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    citations_col : str
        Column containing citation counts.
    
    Returns
    -------
    dict
        Dictionary containing:
        - n_papers: int - Total number of papers
        - citations: array - Raw citation values
        - basic_stats: dict - Mean, median, std, max, min, sum
        - percentiles: dict - 25th, 50th, 75th, 90th, 95th, 99th percentiles
        - uncited: dict - Count and proportion of uncited papers
        - highly_cited: dict - Threshold and count for top 10%
        - h_index: int - Hirsch index
        - g_index: int - Egghe's g-index
        - gini_coefficient: float - Citation inequality (0=equal, 1=unequal)
        - skewness: float - Distribution skewness
        - kurtosis: float - Distribution kurtosis  
        - distribution_fit: dict - Log-normal fit parameters
        - citation_classes: DataFrame - Papers grouped by citation class
        - class_distribution: dict - Count per citation class
    """
    from scipy import stats as scipy_stats
    
    # Get citation counts
    if citations_col not in df.columns:
        # Try alternative names
        for alt in ["Cited by", "Citations", "cited_by", "TC", "Times Cited"]:
            if alt in df.columns:
                citations_col = alt
                break
        else:
            raise ValueError(f"Citation column '{citations_col}' not found")
    
    citations = df[citations_col].fillna(0).astype(int).values
    citations = citations[citations >= 0]  # Remove any negative values
    
    if len(citations) == 0:
        raise ValueError("No citation data found")
    
    n = len(citations)
    
    # Basic statistics
    mean_cit = np.mean(citations)
    median_cit = np.median(citations)
    std_cit = np.std(citations)
    max_cit = np.max(citations)
    min_cit = np.min(citations)
    sum_cit = np.sum(citations)
    
    basic_stats = {
        "mean": mean_cit,
        "median": median_cit,
        "std": std_cit,
        "max": max_cit,
        "min": min_cit,
        "sum": sum_cit,
    }
    
    # Percentiles
    percentiles = {
        25: np.percentile(citations, 25),
        50: np.percentile(citations, 50),
        75: np.percentile(citations, 75),
        90: np.percentile(citations, 90),
        95: np.percentile(citations, 95),
        99: np.percentile(citations, 99),
    }
    
    # Uncited papers
    n_uncited = np.sum(citations == 0)
    prop_uncited = n_uncited / n
    
    uncited = {
        "count": int(n_uncited),
        "proportion": prop_uncited,
        "percentage": prop_uncited * 100,
    }
    
    # Highly cited (top 10%)
    highly_cited_threshold = np.percentile(citations, 90)
    n_highly_cited = np.sum(citations >= highly_cited_threshold)
    
    highly_cited = {
        "threshold": highly_cited_threshold,
        "count": int(n_highly_cited),
        "proportion": n_highly_cited / n,
    }
    
    # H-index
    sorted_cit = np.sort(citations)[::-1]
    h_idx = sum(1 for i, c in enumerate(sorted_cit) if c >= i + 1)
    
    # G-index
    cumsum = np.cumsum(sorted_cit)
    g_idx = 0
    for i in range(1, len(sorted_cit) + 1):
        if cumsum[i-1] >= i * i:
            g_idx = i
    
    # Gini coefficient
    sorted_cit_asc = np.sort(citations)
    cumsum_gini = np.cumsum(sorted_cit_asc)
    gini = 1 - 2 * np.sum(cumsum_gini) / (n * sum_cit) if sum_cit > 0 else 0
    
    # Skewness and kurtosis
    skewness = scipy_stats.skew(citations)
    kurtosis = scipy_stats.kurtosis(citations)
    
    # Log-normal fit (for positive citations)
    positive_citations = citations[citations > 0]
    distribution_fit = {}
    
    if len(positive_citations) > 10:
        log_cit = np.log(positive_citations)
        mu = np.mean(log_cit)
        sigma = np.std(log_cit)
        
        # KS test for goodness of fit
        try:
            ks_stat, ks_p = scipy_stats.kstest(
                positive_citations, "lognorm", args=(sigma, 0, np.exp(mu))
            )
            distribution_fit = {
                "type": "lognormal",
                "mu": mu,
                "sigma": sigma,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
            }
        except:
            distribution_fit = {
                "type": "lognormal",
                "mu": mu,
                "sigma": sigma,
            }
    
    # Citation classes
    def classify_citation(c, p90, p75, p50):
        if c == 0:
            return "Uncited"
        elif c >= p90:
            return "Highly Cited (Top 10%)"
        elif c >= p75:
            return "Well Cited (Top 25%)"
        elif c >= p50:
            return "Average"
        else:
            return "Low Cited"
    
    p90 = percentiles[90]
    p75 = percentiles[75]
    p50 = percentiles[50]
    
    classes = [classify_citation(c, p90, p75, p50) for c in citations]
    
    class_order = ["Uncited", "Low Cited", "Average", "Well Cited (Top 25%)", "Highly Cited (Top 10%)"]
    class_counts = pd.Series(classes).value_counts()
    
    class_distribution = {}
    for cls in class_order:
        count = class_counts.get(cls, 0)
        class_distribution[cls] = {
            "count": int(count),
            "proportion": count / n,
            "percentage": count / n * 100,
        }
    
    # Create citation classes DataFrame
    citation_classes_df = pd.DataFrame({
        "Class": class_order,
        "Count": [class_distribution[c]["count"] for c in class_order],
        "Percentage": [class_distribution[c]["percentage"] for c in class_order],
    })
    
    # Histogram data for plotting
    hist_counts, hist_bins = np.histogram(citations, bins=50)
    histogram_data = pd.DataFrame({
        "bin_start": hist_bins[:-1],
        "bin_end": hist_bins[1:],
        "count": hist_counts,
    })
    
    # Log-scale histogram (excluding zeros)
    if len(positive_citations) > 0:
        log_bins = np.logspace(0, np.log10(max(positive_citations) + 1), 30)
        log_counts, log_bins_out = np.histogram(positive_citations, bins=log_bins)
        log_histogram_data = pd.DataFrame({
            "bin_start": log_bins_out[:-1],
            "bin_end": log_bins_out[1:],
            "count": log_counts,
        })
    else:
        log_histogram_data = pd.DataFrame()
    
    return {
        "n_papers": n,
        "citations": citations,
        "basic_stats": basic_stats,
        "percentiles": percentiles,
        "uncited": uncited,
        "highly_cited": highly_cited,
        "h_index": h_idx,
        "g_index": g_idx,
        "gini_coefficient": gini,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "distribution_fit": distribution_fit,
        "citation_classes": citation_classes_df,
        "class_distribution": class_distribution,
        "histogram_data": histogram_data,
        "log_histogram_data": log_histogram_data,
    }


# =============================================================================
# COLLABORATION ANALYSIS
# =============================================================================

def analyze_collaboration(
    df: pd.DataFrame,
    authors_col: str = "Authors",
    year_col: str = "Year",
    sep: str = "; ",
) -> Dict:
    """
    Analyze collaboration patterns in bibliographic data.
    
    Computes collaboration metrics including Collaboration Index,
    Degree of Collaboration, and Collaboration Coefficient, along
    with author count distributions and temporal trends.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    authors_col : str
        Column containing author names (separated by sep).
    year_col : str
        Column containing publication year.
    sep : str
        Separator between author names.
    
    Returns
    -------
    dict
        Dictionary containing collaboration metrics and distributions.
    
    Notes
    -----
    Collaboration Index (CI): Average number of authors per paper.
    Values > 1 indicate collaborative research.
    
    Degree of Collaboration (DC): Proportion of multi-authored papers.
    Range 0-1, higher values indicate more collaboration.
    
    Collaboration Coefficient (CC): Modified index accounting for
    author count distribution. Range 0-1.
    CC = 1 - (Σ(1/j × fj)) / N
    where fj is papers with j authors, N is total papers.
    """
    from scipy import stats as scipy_stats
    from collections import Counter
    
    # Find authors column
    if authors_col not in df.columns:
        for alt in ["Authors", "Author", "authors", "AU", "Authors or Inventors"]:
            if alt in df.columns:
                authors_col = alt
                break
        else:
            raise ValueError(f"Authors column '{authors_col}' not found")
    
    # Find year column
    year_col_found = None
    if year_col in df.columns:
        year_col_found = year_col
    else:
        for alt in ["Year", "Publication Year", "PY", "year"]:
            if alt in df.columns:
                year_col_found = alt
                break
    
    author_counts = []
    year_data = {}
    
    for idx, row in df.iterrows():
        authors = row.get(authors_col, "")
        
        if pd.isna(authors) or not authors:
            continue
        
        if isinstance(authors, str):
            # Handle multiple possible separators
            if sep in authors:
                author_list = [a.strip() for a in authors.split(sep) if a.strip()]
            elif ";" in authors:
                author_list = [a.strip() for a in authors.split(";") if a.strip()]
            elif "," in authors and authors.count(",") > 1:
                # Multiple commas might indicate comma-separated authors
                author_list = [a.strip() for a in authors.split(",") if a.strip()]
            else:
                # Single author or unknown format
                author_list = [authors.strip()] if authors.strip() else []
        else:
            continue
        
        n_authors = len(author_list)
        if n_authors > 0:
            author_counts.append(n_authors)
            
            if year_col_found is not None:
                year = row.get(year_col_found)
                if year is not None and pd.notna(year):
                    try:
                        year_int = int(float(year))
                        if 1900 <= year_int <= 2100:
                            if year_int not in year_data:
                                year_data[year_int] = []
                            year_data[year_int].append(n_authors)
                    except:
                        pass
    
    if not author_counts:
        raise ValueError("No author data found. Check the authors column and separator.")
    
    author_counts = np.array(author_counts)
    n = len(author_counts)
    
    # Basic counts
    single_author = int(np.sum(author_counts == 1))
    multi_author = int(np.sum(author_counts > 1))
    
    # Collaboration Index (CI) - mean authors per paper
    collaboration_index = float(np.mean(author_counts))
    
    # Degree of Collaboration (DC) - proportion of multi-authored papers
    degree_of_collaboration = multi_author / n if n > 0 else 0
    
    # Collaboration Coefficient (CC)
    # CC = 1 - (Σ(1/j * fj)) / N
    f_counts = Counter(author_counts)
    cc_sum = sum((1.0 / j) * fj for j, fj in f_counts.items() if j > 0)
    collaboration_coefficient = 1 - (cc_sum / n) if n > 0 else 0
    
    # Basic statistics
    try:
        mode_result = scipy_stats.mode(author_counts, keepdims=False)
        mode_val = int(mode_result.mode) if hasattr(mode_result, 'mode') else int(author_counts[0])
    except:
        mode_val = int(author_counts[0]) if len(author_counts) > 0 else 1
    
    basic_stats = {
        "mean": float(np.mean(author_counts)),
        "median": float(np.median(author_counts)),
        "std": float(np.std(author_counts)),
        "max": int(np.max(author_counts)),
        "min": int(np.min(author_counts)),
        "mode": mode_val,
    }
    
    # Author count distribution
    distribution = Counter(author_counts)
    dist_data = []
    for n_auth in sorted(distribution.keys()):
        count = distribution[n_auth]
        dist_data.append({
            "n_authors": int(n_auth),
            "n_papers": count,
            "proportion": count / n,
            "percentage": count / n * 100,
        })
    
    author_distribution = pd.DataFrame(dist_data)
    
    # Temporal trend - aggregate by year
    temporal_data = []
    for year in sorted(year_data.keys()):
        counts = year_data[year]
        if len(counts) > 0:
            n_single = sum(1 for c in counts if c == 1)
            n_multi = len(counts) - n_single
            temporal_data.append({
                "year": year,
                "n_papers": len(counts),
                "mean_authors": float(np.mean(counts)),
                "median_authors": float(np.median(counts)),
                "max_authors": int(np.max(counts)),
                "min_authors": int(np.min(counts)),
                "single_author_count": n_single,
                "multi_author_count": n_multi,
                "single_author_pct": n_single / len(counts) * 100 if len(counts) > 0 else 0,
                "multi_author_pct": n_multi / len(counts) * 100 if len(counts) > 0 else 0,
            })
    
    temporal_trend = pd.DataFrame(temporal_data)
    
    # Collaboration categories
    def categorize_collaboration(n):
        if n == 1:
            return "Single Author"
        elif n == 2:
            return "Pair (2 authors)"
        elif n <= 4:
            return "Small Team (3-4)"
        elif n <= 10:
            return "Medium Team (5-10)"
        else:
            return "Large Team (>10)"
    
    collab_categories = [categorize_collaboration(c) for c in author_counts]
    category_order = ["Single Author", "Pair (2 authors)", "Small Team (3-4)", 
                      "Medium Team (5-10)", "Large Team (>10)"]
    category_counts = Counter(collab_categories)
    
    collaboration_types = pd.DataFrame({
        "Type": category_order,
        "Count": [category_counts.get(t, 0) for t in category_order],
        "Percentage": [category_counts.get(t, 0) / n * 100 if n > 0 else 0 for t in category_order],
    })
    
    return {
        "n_papers": n,
        "collaboration_index": collaboration_index,
        "degree_of_collaboration": degree_of_collaboration,
        "collaboration_coefficient": collaboration_coefficient,
        "single_author_papers": single_author,
        "multi_author_papers": multi_author,
        "max_authors": int(np.max(author_counts)),
        "basic_stats": basic_stats,
        "author_distribution": author_distribution,
        "temporal_trend": temporal_trend,
        "collaboration_types": collaboration_types,
        "author_counts": author_counts,  # Raw data for plotting
    }


# =============================================================================
# ALTMETRICS ANALYSIS
# =============================================================================

def analyze_altmetrics(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    doi_col: str = "DOI",
    title_col: str = "Title",
    citations_col: str = "Cited by",
    year_col: str = "Year",
    simulate: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Analyze altmetrics (alternative metrics) for papers in dataset.
    
    Altmetrics measure non-traditional impact: social media attention,
    news coverage, policy mentions, Wikipedia citations, etc.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input bibliographic dataframe.
    id_col : str
        Document ID column.
    doi_col : str
        DOI column.
    title_col : str
        Title column.
    citations_col : str
        Citations column (for correlation analysis).
    year_col : str
        Year column.
    simulate : bool
        If True, simulate realistic altmetric data for demonstration.
        If False, extract from existing columns if available.
    random_state : int
        Random seed for simulation.
    verbose : bool
        Print progress.
    
    Returns
    -------
    dict
        Dictionary containing:
        - summary_df: DataFrame with altmetric data per paper
        - source_coverage: Dict of source coverage percentages
        - correlation_matrix: Correlation between metrics
        - temporal_trends: Trends by year
        - top_papers: Top papers by altmetric score
        - statistics: Summary statistics
    """
    if verbose:
        print("Analyzing altmetrics...")
    
    n = len(df)
    
    # Get basic info
    ids = df[id_col].values if id_col in df.columns else np.arange(n)
    dois = df[doi_col].values if doi_col in df.columns else [None] * n
    titles = df[title_col].values if title_col in df.columns else [None] * n
    citations = df[citations_col].fillna(0).astype(int).values if citations_col in df.columns else np.zeros(n)
    years = df[year_col].values if year_col in df.columns else [None] * n
    
    if simulate:
        # Generate realistic simulated altmetric data
        altmetrics_data = _simulate_altmetrics(
            n, citations, years, random_state, verbose
        )
    else:
        # Try to extract from existing columns
        altmetrics_data = _extract_altmetrics_from_df(df, verbose)
    
    # Build summary dataframe
    summary_df = pd.DataFrame({
        "Doc ID": ids,
        "DOI": dois,
        "Title": titles,
        "Year": years,
        "Citations": citations,
        **altmetrics_data,
    })
    
    # Calculate composite scores
    summary_df["Social Score"] = (
        summary_df["Twitter"] * 1.0 +
        summary_df["Facebook"] * 0.5 +
        summary_df["Reddit"] * 1.5
    )
    
    summary_df["Scholarly Score"] = (
        summary_df["Mendeley"] * 1.0 +
        summary_df["Wikipedia"] * 3.0
    )
    
    summary_df["Public Score"] = (
        summary_df["News"] * 2.0 +
        summary_df["Blogs"] * 1.0
    )
    
    summary_df["Practice Score"] = (
        summary_df["Policy"] * 5.0 +
        summary_df["Patents"] * 4.0
    )
    
    # Calculate Altmetric Score (composite)
    summary_df["Altmetric Score"] = (
        summary_df["Twitter"] * 0.5 +
        summary_df["Mendeley"] * 0.5 +
        summary_df["News"] * 3.0 +
        summary_df["Blogs"] * 1.0 +
        summary_df["Reddit"] * 0.5 +
        summary_df["Wikipedia"] * 5.0 +
        summary_df["Policy"] * 5.0 +
        summary_df["Patents"] * 3.0
    )
    
    # Calculate percentiles
    from scipy.stats import percentileofscore
    scores = summary_df["Altmetric Score"].values
    summary_df["Altmetric Percentile"] = [
        percentileofscore(scores, s) for s in scores
    ]
    
    # Source coverage
    source_cols = ["Twitter", "Mendeley", "News", "Blogs", "Reddit", 
                   "Wikipedia", "Policy", "Patents", "GitHub"]
    source_coverage = {}
    for col in source_cols:
        if col in summary_df.columns:
            coverage = (summary_df[col] > 0).sum() / n * 100
            source_coverage[col] = round(coverage, 1)
    
    # Correlation matrix
    metric_cols = ["Altmetric Score", "Twitter", "Mendeley", "News", 
                   "Wikipedia", "Social Score", "Scholarly Score", 
                   "Public Score", "Practice Score", "Citations"]
    available_cols = [c for c in metric_cols if c in summary_df.columns]
    correlation_matrix = summary_df[available_cols].corr()
    
    # Temporal trends
    temporal_trends = pd.DataFrame()
    if year_col in df.columns:
        valid_years = summary_df[summary_df["Year"].notna()].copy()
        valid_years["Year"] = valid_years["Year"].astype(int)
        valid_years = valid_years[(valid_years["Year"] >= 1990) & (valid_years["Year"] <= 2030)]
        
        if len(valid_years) > 0:
            year_stats = valid_years.groupby("Year").agg({
                "Altmetric Score": ["mean", "median", "sum", "count"],
                "Twitter": "sum",
                "Mendeley": "sum",
            }).reset_index()
            year_stats.columns = ["Year", "Mean Score", "Median Score", 
                                 "Total Score", "Papers", "Total Twitter", "Total Mendeley"]
            temporal_trends = year_stats
    
    # Top papers
    top_papers = summary_df.nlargest(20, "Altmetric Score")[[
        "Doc ID", "Title", "Year", "Altmetric Score", "Citations",
        "Twitter", "Mendeley", "News", "Policy"
    ]]
    
    # Statistics
    stats = {
        "total_papers": n,
        "with_attention": int((summary_df["Altmetric Score"] > 0).sum()),
        "attention_rate": round((summary_df["Altmetric Score"] > 0).sum() / n * 100, 1),
        "mean_score": round(summary_df["Altmetric Score"].mean(), 2),
        "median_score": round(summary_df["Altmetric Score"].median(), 2),
        "max_score": round(summary_df["Altmetric Score"].max(), 2),
        "with_twitter": int((summary_df["Twitter"] > 0).sum()),
        "with_mendeley": int((summary_df["Mendeley"] > 0).sum()),
        "with_news": int((summary_df["News"] > 0).sum()),
        "with_policy": int((summary_df["Policy"] > 0).sum()),
        "with_patents": int((summary_df["Patents"] > 0).sum()),
    }
    
    # Citation-altmetric correlation
    if "Citations" in summary_df.columns:
        nonzero = summary_df[(summary_df["Citations"] > 0) & (summary_df["Altmetric Score"] > 0)]
        if len(nonzero) > 10:
            from scipy.stats import spearmanr
            corr, p = spearmanr(nonzero["Citations"], nonzero["Altmetric Score"])
            stats["citation_altmetric_correlation"] = round(corr, 3)
    
    if verbose:
        print(f"  Papers analyzed: {n}")
        print(f"  With altmetric attention: {stats['with_attention']} ({stats['attention_rate']:.1f}%)")
        print(f"  Mean altmetric score: {stats['mean_score']:.2f}")
    
    return {
        "summary_df": summary_df,
        "source_coverage": source_coverage,
        "correlation_matrix": correlation_matrix,
        "temporal_trends": temporal_trends,
        "top_papers": top_papers,
        "statistics": stats,
    }


def _simulate_altmetrics(
    n: int,
    citations: np.ndarray,
    years: np.ndarray,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Simulate realistic altmetric data based on citations and recency.
    
    Creates power-law distributed metrics correlated with citations.
    """
    if verbose:
        print("  Simulating altmetric data for demonstration...")
    
    np.random.seed(random_state)
    current_year = pd.Timestamp.now().year
    
    # Initialize arrays
    twitter = np.zeros(n, dtype=int)
    facebook = np.zeros(n, dtype=int)
    reddit = np.zeros(n, dtype=int)
    mendeley = np.zeros(n, dtype=int)
    news = np.zeros(n, dtype=int)
    blogs = np.zeros(n, dtype=int)
    wikipedia = np.zeros(n, dtype=int)
    policy = np.zeros(n, dtype=int)
    patents = np.zeros(n, dtype=int)
    github = np.zeros(n, dtype=int)
    
    for i in range(n):
        cit = int(citations[i]) if pd.notna(citations[i]) else 0
        year = int(years[i]) if pd.notna(years[i]) else current_year
        
        # Base attention proportional to citations
        base_attention = np.log1p(cit) * np.random.uniform(0.5, 1.5)
        
        # Recency boost
        age = current_year - year
        recency_factor = max(0.2, 1.0 - age * 0.1)
        
        # Twitter (power law, ~30% have mentions)
        if np.random.random() < 0.3 + 0.1 * min(base_attention, 2):
            twitter[i] = int(np.random.pareto(1.5) * base_attention * recency_factor * 5)
        
        # Mendeley (more common, ~60%)
        if np.random.random() < 0.6:
            mendeley[i] = int(np.random.pareto(1.2) * cit * 0.5)
        
        # News (rare, ~5-10%)
        if np.random.random() < 0.05 + 0.05 * min(base_attention, 2):
            news[i] = int(np.random.pareto(2) * base_attention * 2)
        
        # Blogs (~15%)
        if np.random.random() < 0.15:
            blogs[i] = int(np.random.pareto(2) * base_attention)
        
        # Reddit (rare, ~5%)
        if np.random.random() < 0.05:
            reddit[i] = int(np.random.pareto(2) * base_attention * 2)
        
        # Facebook (~10%)
        if np.random.random() < 0.10:
            facebook[i] = int(np.random.pareto(1.8) * base_attention)
        
        # Wikipedia (very rare, ~2%)
        if np.random.random() < 0.02 * min(base_attention, 3):
            wikipedia[i] = int(np.random.uniform(1, 5))
        
        # Policy (rare, high-citation papers)
        if cit > 50 and np.random.random() < 0.05:
            policy[i] = int(np.random.uniform(1, 3))
        
        # Patents (rare, high-citation papers)
        if cit > 30 and np.random.random() < 0.03:
            patents[i] = int(np.random.uniform(1, 5))
        
        # GitHub (~5%)
        if np.random.random() < 0.05:
            github[i] = int(np.random.uniform(1, 10))
    
    return {
        "Twitter": twitter,
        "Facebook": facebook,
        "Reddit": reddit,
        "Mendeley": mendeley,
        "News": news,
        "Blogs": blogs,
        "Wikipedia": wikipedia,
        "Policy": policy,
        "Patents": patents,
        "GitHub": github,
    }


def _extract_altmetrics_from_df(
    df: pd.DataFrame,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract altmetrics from existing DataFrame columns.
    
    Looks for common column names for various metrics.
    """
    if verbose:
        print("  Extracting altmetrics from DataFrame columns...")
    
    n = len(df)
    
    # Column name mappings
    col_mappings = {
        "Twitter": ["twitter", "tweets", "twitter_mentions", "x_mentions"],
        "Facebook": ["facebook", "fb_mentions"],
        "Reddit": ["reddit", "reddit_mentions"],
        "Mendeley": ["mendeley", "mendeley_readers"],
        "News": ["news", "news_mentions", "news_count"],
        "Blogs": ["blogs", "blog_mentions"],
        "Wikipedia": ["wikipedia", "wiki_mentions"],
        "Policy": ["policy", "policy_mentions", "policy_documents"],
        "Patents": ["patents", "patent_citations", "patent_count"],
        "GitHub": ["github", "github_repos", "code_repos"],
    }
    
    result = {}
    
    for metric, possible_cols in col_mappings.items():
        found = False
        for col in possible_cols:
            # Case-insensitive match
            matching_cols = [c for c in df.columns if c.lower() == col.lower()]
            if matching_cols:
                result[metric] = df[matching_cols[0]].fillna(0).astype(int).values
                found = True
                break
        
        if not found:
            result[metric] = np.zeros(n, dtype=int)
    
    return result


def get_altmetric_attention_categories(
    summary_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Categorize papers by type of altmetric attention.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary dataframe from analyze_altmetrics.
    
    Returns
    -------
    dict
        DataFrames for different attention categories.
    """
    categories = {}
    
    # High social media attention
    social_threshold = summary_df["Social Score"].quantile(0.9)
    categories["high_social"] = summary_df[
        summary_df["Social Score"] >= social_threshold
    ].copy()
    
    # Policy/practice impact
    categories["policy_impact"] = summary_df[
        (summary_df["Policy"] > 0) | (summary_df["Patents"] > 0)
    ].copy()
    
    # News coverage
    categories["news_coverage"] = summary_df[
        summary_df["News"] > 0
    ].copy()
    
    # Wikipedia mentioned
    categories["wikipedia"] = summary_df[
        summary_df["Wikipedia"] > 0
    ].copy()
    
    # High citation but low altmetric (sleeping giants)
    if "Citations" in summary_df.columns:
        cit_90 = summary_df["Citations"].quantile(0.9)
        alt_25 = summary_df["Altmetric Score"].quantile(0.25)
        categories["high_citation_low_altmetric"] = summary_df[
            (summary_df["Citations"] >= cit_90) & 
            (summary_df["Altmetric Score"] <= alt_25)
        ].copy()
        
        # High altmetric but low citation (viral papers)
        alt_90 = summary_df["Altmetric Score"].quantile(0.9)
        cit_25 = summary_df["Citations"].quantile(0.25)
        categories["high_altmetric_low_citation"] = summary_df[
            (summary_df["Altmetric Score"] >= alt_90) & 
            (summary_df["Citations"] <= cit_25)
        ].copy()
    
    return categories


# =============================================================================
# NOVELTY / ATYPICALITY ANALYSIS
# =============================================================================

def analyze_novelty(
    df: pd.DataFrame,
    keywords_col: str = None,
    subject_col: str = None,
    references_col: str = None,
    year_col: str = "Year",
    title_col: str = "Title",
    id_col: str = None,
    min_keyword_freq: int = 2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyze novelty and atypicality of papers based on unusual combinations.
    
    Computes within-dataset novelty measures identifying papers that combine
    elements (keywords, subjects, references) in unusual ways relative to
    the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic DataFrame.
    keywords_col : str, optional
        Column with keywords (semicolon or comma separated).
        Auto-detected from: Author Keywords, Keywords, DE, ID.
    subject_col : str, optional
        Column with subject categories.
        Auto-detected from: Subject Area, WC, SC, Categories.
    references_col : str, optional
        Column with references (for reference diversity).
        Auto-detected from: References, Cited References, CR.
    year_col : str
        Column with publication year.
    title_col : str
        Column with paper titles.
    id_col : str, optional
        Column with paper IDs. Auto-detected or uses index.
    min_keyword_freq : int
        Minimum frequency for a keyword to be included in analysis.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    dict
        Dictionary containing:
        - novelty_df: DataFrame with novelty scores per paper
        - keyword_novelty: Keyword combination novelty analysis
        - subject_bridging: Subject category bridging analysis
        - reference_diversity: Reference diversity analysis (if available)
        - top_novel_papers: Top 20 most novel papers
        - statistics: Summary statistics
        - temporal_trends: Novelty trends over time
    """
    from collections import Counter
    from itertools import combinations
    
    if verbose:
        print("Analyzing novelty and atypicality...")
    
    n_papers = len(df)
    
    # Auto-detect columns
    if keywords_col is None:
        for col in ["Author Keywords", "Keywords", "DE", "ID", "Index Keywords"]:
            if col in df.columns:
                keywords_col = col
                break
    
    if subject_col is None:
        for col in ["Subject Area", "WC", "SC", "Categories", "Subject Categories", 
                    "Research Areas", "Web of Science Categories"]:
            if col in df.columns:
                subject_col = col
                break
    
    if references_col is None:
        for col in ["References", "Cited References", "CR", "cited_by_ids"]:
            if col in df.columns:
                references_col = col
                break
    
    if id_col is None:
        for col in ["unique-id", "EID", "UT", "DOI", "id"]:
            if col in df.columns:
                id_col = col
                break
    
    # Initialize result DataFrame
    novelty_df = pd.DataFrame(index=df.index)
    if id_col and id_col in df.columns:
        novelty_df["ID"] = df[id_col]
    else:
        novelty_df["ID"] = df.index
    
    if title_col in df.columns:
        novelty_df["Title"] = df[title_col]
    
    if year_col in df.columns:
        novelty_df["Year"] = pd.to_numeric(df[year_col], errors="coerce")
    
    # ==========================================================================
    # 1. KEYWORD COMBINATION NOVELTY
    # ==========================================================================
    keyword_novelty = None
    keyword_scores = np.zeros(n_papers)
    
    if keywords_col and keywords_col in df.columns:
        if verbose:
            print(f"  Computing keyword novelty from '{keywords_col}'...")
        
        # Parse keywords for each paper
        paper_keywords = []
        for idx, row in df.iterrows():
            kw = row[keywords_col]
            if pd.isna(kw):
                paper_keywords.append([])
            else:
                # Split by semicolon or comma
                kw_str = str(kw)
                if ";" in kw_str:
                    kws = [k.strip().lower() for k in kw_str.split(";") if k.strip()]
                else:
                    kws = [k.strip().lower() for k in kw_str.split(",") if k.strip()]
                paper_keywords.append(kws)
        
        # Count keyword frequencies
        all_keywords = [kw for kws in paper_keywords for kw in kws]
        keyword_freq = Counter(all_keywords)
        
        # Filter to keywords appearing at least min_keyword_freq times
        valid_keywords = {k for k, v in keyword_freq.items() if v >= min_keyword_freq}
        
        # Count keyword pair frequencies
        pair_freq = Counter()
        for kws in paper_keywords:
            valid_kws = [k for k in kws if k in valid_keywords]
            for pair in combinations(sorted(valid_kws), 2):
                pair_freq[pair] += 1
        
        # Compute novelty score for each paper
        # Novelty = sum of 1/freq(pair) for all pairs, normalized
        total_papers_with_pairs = sum(1 for kws in paper_keywords 
                                      if len([k for k in kws if k in valid_keywords]) >= 2)
        
        for i, kws in enumerate(paper_keywords):
            valid_kws = [k for k in kws if k in valid_keywords]
            if len(valid_kws) >= 2:
                pairs = list(combinations(sorted(valid_kws), 2))
                # Novelty: inverse frequency of pairs (rarer = more novel)
                pair_scores = []
                for pair in pairs:
                    freq = pair_freq.get(pair, 1)
                    # Normalize by max possible frequency
                    rarity = 1.0 / freq
                    pair_scores.append(rarity)
                
                # Average rarity of pairs
                keyword_scores[i] = np.mean(pair_scores) if pair_scores else 0
        
        # Normalize to 0-1 scale
        if keyword_scores.max() > 0:
            keyword_scores = keyword_scores / keyword_scores.max()
        
        # Find most novel keyword combinations
        novel_pairs = [(pair, freq) for pair, freq in pair_freq.items() if freq == 1]
        common_pairs = pair_freq.most_common(20)
        
        keyword_novelty = {
            "n_unique_keywords": len(valid_keywords),
            "n_keyword_pairs": len(pair_freq),
            "n_unique_pairs": len(novel_pairs),
            "unique_pair_ratio": len(novel_pairs) / max(len(pair_freq), 1),
            "most_common_pairs": common_pairs,
            "example_novel_pairs": novel_pairs[:20],
            "mean_novelty": float(np.mean(keyword_scores[keyword_scores > 0])) if any(keyword_scores > 0) else 0,
        }
    
    novelty_df["Keyword Novelty"] = keyword_scores
    
    # ==========================================================================
    # 2. SUBJECT CATEGORY BRIDGING
    # ==========================================================================
    subject_bridging = None
    bridging_scores = np.zeros(n_papers)
    
    if subject_col and subject_col in df.columns:
        if verbose:
            print(f"  Computing subject bridging from '{subject_col}'...")
        
        # Parse subjects for each paper
        paper_subjects = []
        for idx, row in df.iterrows():
            subj = row[subject_col]
            if pd.isna(subj):
                paper_subjects.append([])
            else:
                subj_str = str(subj)
                if ";" in subj_str:
                    subjs = [s.strip() for s in subj_str.split(";") if s.strip()]
                elif "|" in subj_str:
                    subjs = [s.strip() for s in subj_str.split("|") if s.strip()]
                else:
                    subjs = [s.strip() for s in subj_str.split(",") if s.strip()]
                paper_subjects.append(subjs)
        
        # Count subject frequencies
        all_subjects = [s for ss in paper_subjects for s in ss]
        subject_freq = Counter(all_subjects)
        
        # Count subject pair co-occurrences
        subject_pair_freq = Counter()
        for subjs in paper_subjects:
            if len(subjs) >= 2:
                for pair in combinations(sorted(subjs), 2):
                    subject_pair_freq[pair] += 1
        
        # Bridging score: papers spanning rarely combined subjects
        for i, subjs in enumerate(paper_subjects):
            if len(subjs) >= 2:
                pairs = list(combinations(sorted(subjs), 2))
                pair_rarities = []
                for pair in pairs:
                    freq = subject_pair_freq.get(pair, 1)
                    pair_rarities.append(1.0 / freq)
                bridging_scores[i] = np.mean(pair_rarities) if pair_rarities else 0
            elif len(subjs) == 1:
                # Single subject - no bridging
                bridging_scores[i] = 0
        
        # Normalize to 0-1
        if bridging_scores.max() > 0:
            bridging_scores = bridging_scores / bridging_scores.max()
        
        # Find papers bridging rare subject combinations
        rare_bridges = [(pair, freq) for pair, freq in subject_pair_freq.items() if freq <= 2]
        
        subject_bridging = {
            "n_subjects": len(subject_freq),
            "n_subject_pairs": len(subject_pair_freq),
            "n_rare_bridges": len(rare_bridges),
            "most_common_combinations": subject_pair_freq.most_common(20),
            "rare_combinations": rare_bridges[:20],
            "mean_bridging": float(np.mean(bridging_scores[bridging_scores > 0])) if any(bridging_scores > 0) else 0,
            "subject_distribution": dict(subject_freq.most_common(30)),
        }
    
    novelty_df["Subject Bridging"] = bridging_scores
    
    # ==========================================================================
    # 3. REFERENCE DIVERSITY
    # ==========================================================================
    reference_diversity = None
    diversity_scores = np.zeros(n_papers)
    
    if references_col and references_col in df.columns:
        if verbose:
            print(f"  Computing reference diversity from '{references_col}'...")
        
        # Count references per paper and extract source journals
        ref_counts = []
        ref_sources = []
        
        for idx, row in df.iterrows():
            refs = row[references_col]
            if pd.isna(refs):
                ref_counts.append(0)
                ref_sources.append([])
            else:
                ref_str = str(refs)
                # Split references
                if "; " in ref_str:
                    ref_list = ref_str.split("; ")
                elif "\n" in ref_str:
                    ref_list = ref_str.split("\n")
                else:
                    ref_list = [ref_str]
                
                ref_counts.append(len(ref_list))
                
                # Try to extract source journals from references
                sources = []
                for ref in ref_list:
                    # Common patterns: journal names are often in italics or after year
                    # This is a simplified heuristic
                    parts = ref.split(",")
                    if len(parts) >= 2:
                        # Try to find journal-like part (often 2nd or 3rd element)
                        for part in parts[1:4]:
                            part = part.strip()
                            if len(part) > 3 and not part.isdigit():
                                sources.append(part[:50])  # Truncate long strings
                                break
                ref_sources.append(sources)
        
        # Diversity = number of unique sources / total references (Shannon-like)
        for i, sources in enumerate(ref_sources):
            if len(sources) > 0:
                unique_sources = len(set(sources))
                total_sources = len(sources)
                # Simpson diversity index
                if total_sources > 1:
                    source_counts = Counter(sources)
                    simpson = 1 - sum(c * (c - 1) for c in source_counts.values()) / (total_sources * (total_sources - 1))
                    diversity_scores[i] = simpson
                else:
                    diversity_scores[i] = 0
        
        reference_diversity = {
            "mean_references": float(np.mean(ref_counts)),
            "median_references": float(np.median(ref_counts)),
            "papers_with_refs": sum(1 for c in ref_counts if c > 0),
            "mean_diversity": float(np.mean(diversity_scores[diversity_scores > 0])) if any(diversity_scores > 0) else 0,
        }
    
    novelty_df["Reference Diversity"] = diversity_scores
    
    # ==========================================================================
    # 4. COMPOSITE NOVELTY SCORE
    # ==========================================================================
    # Combine available measures
    available_scores = []
    weights = []
    
    if keyword_novelty:
        available_scores.append(keyword_scores)
        weights.append(0.4)
    
    if subject_bridging:
        available_scores.append(bridging_scores)
        weights.append(0.35)
    
    if reference_diversity:
        available_scores.append(diversity_scores)
        weights.append(0.25)
    
    if available_scores:
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        composite = np.zeros(n_papers)
        for score, weight in zip(available_scores, weights):
            composite += score * weight
        novelty_df["Composite Novelty"] = composite
    else:
        novelty_df["Composite Novelty"] = 0
    
    # ==========================================================================
    # 5. IDENTIFY TOP NOVEL PAPERS
    # ==========================================================================
    top_novel = novelty_df.nlargest(20, "Composite Novelty").copy()
    
    # ==========================================================================
    # 6. TEMPORAL TRENDS
    # ==========================================================================
    temporal_trends = None
    if year_col in df.columns:
        novelty_df["Year"] = pd.to_numeric(df[year_col], errors="coerce")
        yearly = novelty_df.dropna(subset=["Year"]).groupby("Year").agg({
            "Composite Novelty": ["mean", "median", "std", "count"],
            "Keyword Novelty": "mean",
            "Subject Bridging": "mean",
            "Reference Diversity": "mean",
        }).reset_index()
        yearly.columns = ["Year", "Mean Novelty", "Median Novelty", "Std Novelty", 
                         "Papers", "Mean Keyword Novelty", "Mean Subject Bridging",
                         "Mean Reference Diversity"]
        yearly = yearly[(yearly["Year"] >= 1900) & (yearly["Year"] <= 2100)]
        temporal_trends = yearly
    
    # ==========================================================================
    # 7. STATISTICS
    # ==========================================================================
    composite = novelty_df["Composite Novelty"]
    statistics = {
        "n_papers": n_papers,
        "mean_novelty": float(composite.mean()),
        "median_novelty": float(composite.median()),
        "std_novelty": float(composite.std()),
        "max_novelty": float(composite.max()),
        "highly_novel_count": int((composite > composite.quantile(0.9)).sum()),
        "highly_novel_threshold": float(composite.quantile(0.9)),
        "has_keywords": keywords_col is not None and keywords_col in df.columns,
        "has_subjects": subject_col is not None and subject_col in df.columns,
        "has_references": references_col is not None and references_col in df.columns,
    }
    
    if verbose:
        print(f"\n📊 Novelty Analysis Results")
        print(f"=" * 50)
        print(f"Papers analyzed: {n_papers}")
        print(f"Mean composite novelty: {statistics['mean_novelty']:.3f}")
        print(f"Highly novel papers (top 10%): {statistics['highly_novel_count']}")
        if keyword_novelty:
            print(f"Unique keywords: {keyword_novelty['n_unique_keywords']}")
            print(f"Novel keyword pairs (appear once): {keyword_novelty['n_unique_pairs']}")
        if subject_bridging:
            print(f"Subject categories: {subject_bridging['n_subjects']}")
            print(f"Rare subject bridges: {subject_bridging['n_rare_bridges']}")
    
    return {
        "novelty_df": novelty_df,
        "keyword_novelty": keyword_novelty,
        "subject_bridging": subject_bridging,
        "reference_diversity": reference_diversity,
        "top_novel_papers": top_novel,
        "statistics": statistics,
        "temporal_trends": temporal_trends,
    }


def fetch_openalex_novelty(
    dois: List[str],
    email: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fetch novelty-related data from OpenAlex API for given DOIs.
    
    Uses OpenAlex concepts and topics to compute broader novelty measures
    by comparing to global concept co-occurrence patterns.
    
    Parameters
    ----------
    dois : list
        List of DOIs to fetch.
    email : str, optional
        Email for polite API access (higher rate limits).
    verbose : bool
        Print progress.
    
    Returns
    -------
    dict
        Dictionary with OpenAlex-enhanced novelty data.
    """
    import requests
    import time
    
    if verbose:
        print(f"Fetching OpenAlex data for {len(dois)} papers...")
    
    base_url = "https://api.openalex.org/works"
    headers = {"User-Agent": f"Biblium/2.7 (mailto:{email})" if email else "Biblium/2.7"}
    
    results = []
    concepts_data = []
    
    for i, doi in enumerate(dois):
        if pd.isna(doi) or not doi:
            continue
        
        # Clean DOI
        doi_clean = str(doi).strip()
        if doi_clean.startswith("http"):
            doi_clean = doi_clean.split("doi.org/")[-1]
        
        try:
            url = f"{base_url}/https://doi.org/{doi_clean}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract concepts with scores
                concepts = data.get("concepts", [])
                topics = data.get("topics", [])
                
                paper_result = {
                    "doi": doi_clean,
                    "id": data.get("id"),
                    "title": data.get("title"),
                    "concepts": [(c.get("display_name"), c.get("score", 0)) 
                                for c in concepts],
                    "topics": [(t.get("display_name"), t.get("score", 0)) 
                              for t in topics] if topics else [],
                    "cited_by_count": data.get("cited_by_count", 0),
                    "publication_year": data.get("publication_year"),
                }
                results.append(paper_result)
                
                # Track all concepts
                for c in concepts:
                    concepts_data.append({
                        "doi": doi_clean,
                        "concept": c.get("display_name"),
                        "level": c.get("level"),
                        "score": c.get("score", 0),
                    })
            
            # Rate limiting
            if i > 0 and i % 10 == 0:
                time.sleep(0.1)
                if verbose:
                    print(f"  Processed {i}/{len(dois)} papers...")
                    
        except Exception as e:
            if verbose:
                print(f"  Error fetching {doi_clean}: {e}")
    
    if verbose:
        print(f"  Successfully fetched {len(results)} papers")
    
    # Compute concept co-occurrence novelty
    if results:
        # Build concept co-occurrence matrix
        from collections import Counter
        from itertools import combinations
        concept_pairs = Counter()
        
        for paper in results:
            concepts = [c[0] for c in paper["concepts"] if c[1] > 0.3]  # High-confidence concepts
            for pair in combinations(sorted(concepts), 2):
                concept_pairs[pair] += 1
        
        # Compute novelty for each paper
        for paper in results:
            concepts = [c[0] for c in paper["concepts"] if c[1] > 0.3]
            if len(concepts) >= 2:
                pairs = list(combinations(sorted(concepts), 2))
                rarities = [1.0 / concept_pairs.get(p, 1) for p in pairs]
                paper["concept_novelty"] = np.mean(rarities)
            else:
                paper["concept_novelty"] = 0
        
        # Normalize
        max_novelty = max(p["concept_novelty"] for p in results) or 1
        for paper in results:
            paper["concept_novelty_normalized"] = paper["concept_novelty"] / max_novelty
    
    return {
        "papers": results,
        "concepts_data": pd.DataFrame(concepts_data) if concepts_data else pd.DataFrame(),
        "n_fetched": len(results),
        "n_requested": len(dois),
    }


def get_novelty_categories(
    novelty_df: pd.DataFrame,
    novelty_col: str = "Composite Novelty",
) -> Dict[str, pd.DataFrame]:
    """
    Categorize papers by novelty level.
    
    Parameters
    ----------
    novelty_df : pd.DataFrame
        DataFrame with novelty scores.
    novelty_col : str
        Column containing novelty scores.
    
    Returns
    -------
    dict
        Dictionary with DataFrames for each category.
    """
    categories = {}
    
    scores = novelty_df[novelty_col]
    
    # Percentile-based categories
    p25 = scores.quantile(0.25)
    p50 = scores.quantile(0.50)
    p75 = scores.quantile(0.75)
    p90 = scores.quantile(0.90)
    
    categories["highly_novel"] = novelty_df[scores >= p90].copy()
    categories["novel"] = novelty_df[(scores >= p75) & (scores < p90)].copy()
    categories["moderate"] = novelty_df[(scores >= p50) & (scores < p75)].copy()
    categories["conventional"] = novelty_df[(scores >= p25) & (scores < p50)].copy()
    categories["traditional"] = novelty_df[scores < p25].copy()
    
    return categories


# =============================================================================
# SENTIMENT ANALYSIS (Enhanced)
# =============================================================================

# Scientific certainty markers for specialized sentiment analysis
SCIENTIFIC_CERTAINTY_MARKERS = {
    "high_certainty": [
        r"\bdemonstrate[sd]?\b", r"\bprove[sd]?\b", r"\bconfirm[sed]?\b",
        r"\bestablish(?:es|ed)?\b", r"\bclearly\b", r"\bdefinitely\b",
        r"\bundoubtedly\b", r"\bcertainly\b", r"\bevident\b",
        r"\bverif(?:y|ied|ies)\b", r"\bvalidat(?:e|ed|es)\b",
    ],
    "moderate_certainty": [
        r"\bindicate[sd]?\b", r"\bshow[sed]?\b", r"\bfind[s]?\b", r"\bfound\b",
        r"\bsuggest[sed]?\b", r"\bimply\b", r"\bimplies\b", r"\breveal[sed]?\b",
        r"\bsupport[sed]?\b",
    ],
    "low_certainty": [
        r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bpossibly\b",
        r"\bperhaps\b", r"\bpotentially\b", r"\bappear[s]?\b",
        r"\bseem[s]?\b", r"\btend[s]?\b", r"\blikely\b", r"\bprobably\b",
    ],
    "hedging": [
        r"\bsomewhat\b", r"\brelatively\b", r"\bpartially\b",
        r"\bto\s+some\s+extent\b", r"\bin\s+part\b", r"\bgenerally\b",
        r"\btypically\b", r"\busually\b", r"\boften\b", r"\bmost(?:ly)?\b",
    ],
}

# Positive/negative word lists for scientific context
SCIENTIFIC_POSITIVE_WORDS = {
    "significant", "novel", "innovative", "effective", "successful", "improved",
    "excellent", "superior", "robust", "promising", "breakthrough", "advanced",
    "efficient", "optimal", "remarkable", "outstanding", "comprehensive",
    "substantial", "compelling", "conclusive", "rigorous", "strong", "clear",
}

SCIENTIFIC_NEGATIVE_WORDS = {
    "limited", "failed", "poor", "weak", "insufficient", "inadequate",
    "problematic", "controversial", "challenging", "difficult", "unclear",
    "inconclusive", "inconsistent", "biased", "flawed", "questionable",
    "marginal", "negligible", "unreliable", "contradictory", "ambiguous",
}


def analyze_sentiment_advanced(
    df: pd.DataFrame,
    text_column: str = None,
    year_column: str = "Year",
    title_column: str = "Title",
    id_column: str = None,
    sentiment_threshold: float = 0.05,
    use_vader: bool = True,
    use_textblob: bool = True,
    use_scientific: bool = True,
    top_words: int = 20,
    analyze_temporal: bool = True,
    analyze_by_source: bool = False,
    source_column: str = "Source",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Perform comprehensive sentiment analysis on bibliographic text data.
    
    This enhanced function provides multiple sentiment analysis approaches:
    - VADER sentiment (optimized for social media but works well for general text)
    - TextBlob sentiment (pattern-based)
    - Scientific certainty analysis (domain-specific markers)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with text data.
    text_column : str, optional
        Column containing text to analyze. Auto-detected from:
        Abstract, Processed Abstract, Title, etc.
    year_column : str
        Column with publication year for temporal analysis.
    title_column : str
        Column with document titles.
    id_column : str, optional
        Column with document IDs. Uses index if not specified.
    sentiment_threshold : float
        Threshold for classifying as positive/negative (default 0.05).
    use_vader : bool
        Use NLTK's VADER sentiment analyzer.
    use_textblob : bool
        Use TextBlob sentiment analysis.
    use_scientific : bool
        Apply scientific text-specific analysis (certainty markers).
    top_words : int
        Number of top words to extract per sentiment category.
    analyze_temporal : bool
        Perform temporal trend analysis.
    analyze_by_source : bool
        Analyze sentiment by publication source.
    source_column : str
        Column with source/journal names.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    dict
        Comprehensive results containing:
        - sentiment_df: DataFrame with all sentiment scores
        - statistics: Summary statistics
        - temporal_trends: Sentiment over time (if analyze_temporal)
        - source_analysis: Sentiment by source (if analyze_by_source)
        - word_analysis: Top words per sentiment category
        - certainty_analysis: Scientific certainty metrics
        - correlations: Correlations with other variables
        - top_positive: Top positive documents
        - top_negative: Top negative documents
    """
    import re
    from collections import Counter
    
    if verbose:
        print("Performing advanced sentiment analysis...")
    
    n_docs = len(df)
    
    # Auto-detect text column
    if text_column is None:
        for col in ["Abstract", "Processed Abstract", "abstract", "AB", 
                    "Title", "TI", "Combined Text"]:
            if col in df.columns:
                text_column = col
                break
    
    if text_column is None or text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found. Available columns: {list(df.columns)}")
    
    if verbose:
        print(f"  Analyzing text from column: '{text_column}'")
    
    # Auto-detect ID column
    if id_column is None:
        for col in ["unique-id", "EID", "UT", "DOI", "id"]:
            if col in df.columns:
                id_column = col
                break
    
    # Initialize result DataFrame
    sentiment_df = pd.DataFrame(index=df.index)
    
    if id_column and id_column in df.columns:
        sentiment_df["ID"] = df[id_column]
    else:
        sentiment_df["ID"] = df.index
    
    if title_column in df.columns:
        sentiment_df["Title"] = df[title_column]
    
    if year_column in df.columns:
        sentiment_df["Year"] = pd.to_numeric(df[year_column], errors="coerce")
    
    # ==========================================================================
    # VADER SENTIMENT
    # ==========================================================================
    vader_scores = np.zeros(n_docs)
    vader_positive = np.zeros(n_docs)
    vader_negative = np.zeros(n_docs)
    vader_neutral = np.zeros(n_docs)
    
    if use_vader:
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            # Download lexicon if needed
            try:
                nltk.download("vader_lexicon", quiet=True)
            except:
                pass
            
            sia = SentimentIntensityAnalyzer()
            
            if verbose:
                print("  Computing VADER sentiment scores...")
            
            for i, row in enumerate(df.itertuples()):
                text = getattr(row, text_column.replace(" ", "_").replace("-", "_"), "")
                if pd.isna(text) or not text:
                    text = str(df.iloc[i][text_column]) if not pd.isna(df.iloc[i][text_column]) else ""
                
                if text:
                    scores = sia.polarity_scores(str(text))
                    vader_scores[i] = scores["compound"]
                    vader_positive[i] = scores["pos"]
                    vader_negative[i] = scores["neg"]
                    vader_neutral[i] = scores["neu"]
        except ImportError:
            if verbose:
                print("  VADER not available (nltk required)")
            use_vader = False
    
    sentiment_df["VADER Compound"] = vader_scores
    sentiment_df["VADER Positive"] = vader_positive
    sentiment_df["VADER Negative"] = vader_negative
    sentiment_df["VADER Neutral"] = vader_neutral
    
    # ==========================================================================
    # TEXTBLOB SENTIMENT
    # ==========================================================================
    textblob_polarity = np.zeros(n_docs)
    textblob_subjectivity = np.zeros(n_docs)
    
    if use_textblob:
        try:
            from textblob import TextBlob
            
            if verbose:
                print("  Computing TextBlob sentiment scores...")
            
            for i, row in enumerate(df.itertuples()):
                text = getattr(row, text_column.replace(" ", "_").replace("-", "_"), "")
                if pd.isna(text) or not text:
                    text = str(df.iloc[i][text_column]) if not pd.isna(df.iloc[i][text_column]) else ""
                
                if text:
                    blob = TextBlob(str(text))
                    textblob_polarity[i] = blob.sentiment.polarity
                    textblob_subjectivity[i] = blob.sentiment.subjectivity
        except ImportError:
            if verbose:
                print("  TextBlob not available")
            use_textblob = False
    
    sentiment_df["TextBlob Polarity"] = textblob_polarity
    sentiment_df["TextBlob Subjectivity"] = textblob_subjectivity
    
    # ==========================================================================
    # SCIENTIFIC CERTAINTY ANALYSIS
    # ==========================================================================
    certainty_scores = np.zeros(n_docs)
    hedging_scores = np.zeros(n_docs)
    certainty_markers_high = np.zeros(n_docs)
    certainty_markers_moderate = np.zeros(n_docs)
    certainty_markers_low = np.zeros(n_docs)
    certainty_markers_hedging = np.zeros(n_docs)
    scientific_positive = np.zeros(n_docs)
    scientific_negative = np.zeros(n_docs)
    
    if use_scientific:
        if verbose:
            print("  Computing scientific certainty markers...")
        
        for i, row in enumerate(df.itertuples()):
            text = getattr(row, text_column.replace(" ", "_").replace("-", "_"), "")
            if pd.isna(text) or not text:
                text = str(df.iloc[i][text_column]) if not pd.isna(df.iloc[i][text_column]) else ""
            
            if text:
                text_lower = str(text).lower()
                
                # Count certainty markers
                for pattern in SCIENTIFIC_CERTAINTY_MARKERS["high_certainty"]:
                    certainty_markers_high[i] += len(re.findall(pattern, text_lower))
                for pattern in SCIENTIFIC_CERTAINTY_MARKERS["moderate_certainty"]:
                    certainty_markers_moderate[i] += len(re.findall(pattern, text_lower))
                for pattern in SCIENTIFIC_CERTAINTY_MARKERS["low_certainty"]:
                    certainty_markers_low[i] += len(re.findall(pattern, text_lower))
                for pattern in SCIENTIFIC_CERTAINTY_MARKERS["hedging"]:
                    certainty_markers_hedging[i] += len(re.findall(pattern, text_lower))
                
                # Calculate certainty score
                total_markers = (certainty_markers_high[i] + certainty_markers_moderate[i] + 
                               certainty_markers_low[i] + certainty_markers_hedging[i])
                
                if total_markers > 0:
                    certainty_scores[i] = (
                        certainty_markers_high[i] * 1.0 +
                        certainty_markers_moderate[i] * 0.7 +
                        certainty_markers_low[i] * 0.3 +
                        certainty_markers_hedging[i] * 0.2
                    ) / total_markers
                    hedging_scores[i] = certainty_markers_hedging[i] / total_markers
                else:
                    certainty_scores[i] = 0.5
                    hedging_scores[i] = 0
                
                # Count scientific positive/negative words
                words = set(text_lower.split())
                scientific_positive[i] = len(words & SCIENTIFIC_POSITIVE_WORDS)
                scientific_negative[i] = len(words & SCIENTIFIC_NEGATIVE_WORDS)
    
    sentiment_df["Certainty Score"] = certainty_scores
    sentiment_df["Hedging Score"] = hedging_scores
    sentiment_df["High Certainty Markers"] = certainty_markers_high
    sentiment_df["Moderate Certainty Markers"] = certainty_markers_moderate
    sentiment_df["Low Certainty Markers"] = certainty_markers_low
    sentiment_df["Hedging Markers"] = certainty_markers_hedging
    sentiment_df["Scientific Positive Words"] = scientific_positive
    sentiment_df["Scientific Negative Words"] = scientific_negative
    
    # ==========================================================================
    # COMPOSITE SENTIMENT SCORE
    # ==========================================================================
    # Combine available scores into a composite
    available_scores = []
    weights = []
    
    if use_vader and np.any(vader_scores != 0):
        available_scores.append(vader_scores)
        weights.append(0.4)
    
    if use_textblob and np.any(textblob_polarity != 0):
        available_scores.append(textblob_polarity)
        weights.append(0.4)
    
    if use_scientific:
        # Normalize scientific sentiment to -1 to 1 range
        sci_sentiment = np.zeros(n_docs)
        for i in range(n_docs):
            total = scientific_positive[i] + scientific_negative[i]
            if total > 0:
                sci_sentiment[i] = (scientific_positive[i] - scientific_negative[i]) / total
        available_scores.append(sci_sentiment)
        weights.append(0.2)
    
    if available_scores:
        weights = np.array(weights) / sum(weights)
        composite = np.zeros(n_docs)
        for score, weight in zip(available_scores, weights):
            composite += score * weight
        sentiment_df["Composite Sentiment"] = composite
    else:
        sentiment_df["Composite Sentiment"] = vader_scores
    
    # ==========================================================================
    # SENTIMENT CATEGORIES
    # ==========================================================================
    sentiment_df["Sentiment Category"] = sentiment_df["Composite Sentiment"].apply(
        lambda x: "Positive" if x > sentiment_threshold 
        else "Negative" if x < -sentiment_threshold 
        else "Neutral"
    )
    
    # Fine-grained categories
    def get_fine_category(score):
        if score >= 0.5:
            return "Very Positive"
        elif score >= 0.2:
            return "Moderately Positive"
        elif score > sentiment_threshold:
            return "Slightly Positive"
        elif score <= -0.5:
            return "Very Negative"
        elif score <= -0.2:
            return "Moderately Negative"
        elif score < -sentiment_threshold:
            return "Slightly Negative"
        else:
            return "Neutral"
    
    sentiment_df["Sentiment Category Fine"] = sentiment_df["Composite Sentiment"].apply(get_fine_category)
    
    # ==========================================================================
    # STATISTICS
    # ==========================================================================
    composite = sentiment_df["Composite Sentiment"]
    
    statistics = {
        "n_documents": n_docs,
        "mean_sentiment": float(composite.mean()),
        "median_sentiment": float(composite.median()),
        "std_sentiment": float(composite.std()),
        "min_sentiment": float(composite.min()),
        "max_sentiment": float(composite.max()),
        "skewness": float(composite.skew()) if hasattr(composite, "skew") else 0,
        "positive_count": int((composite > sentiment_threshold).sum()),
        "negative_count": int((composite < -sentiment_threshold).sum()),
        "neutral_count": int(((composite >= -sentiment_threshold) & (composite <= sentiment_threshold)).sum()),
        "positive_pct": float((composite > sentiment_threshold).sum() / n_docs * 100),
        "negative_pct": float((composite < -sentiment_threshold).sum() / n_docs * 100),
        "neutral_pct": float(((composite >= -sentiment_threshold) & (composite <= sentiment_threshold)).sum() / n_docs * 100),
        "sentiment_threshold": sentiment_threshold,
    }
    
    if use_vader:
        statistics["vader_mean"] = float(vader_scores.mean())
        statistics["vader_std"] = float(vader_scores.std())
    
    if use_textblob:
        statistics["textblob_polarity_mean"] = float(textblob_polarity.mean())
        statistics["textblob_subjectivity_mean"] = float(textblob_subjectivity.mean())
    
    if use_scientific:
        statistics["certainty_mean"] = float(certainty_scores.mean())
        statistics["hedging_mean"] = float(hedging_scores.mean())
    
    # ==========================================================================
    # TEMPORAL TRENDS
    # ==========================================================================
    temporal_trends = None
    
    if analyze_temporal and year_column in df.columns:
        if verbose:
            print("  Analyzing temporal trends...")
        
        sentiment_df["Year"] = pd.to_numeric(df[year_column], errors="coerce")
        
        yearly = sentiment_df.dropna(subset=["Year"]).groupby("Year").agg({
            "Composite Sentiment": ["mean", "median", "std", "count"],
            "Certainty Score": "mean",
            "Hedging Score": "mean",
        }).reset_index()
        
        yearly.columns = ["Year", "Mean Sentiment", "Median Sentiment", "Std Sentiment",
                         "Document Count", "Mean Certainty", "Mean Hedging"]
        
        # Filter valid years
        yearly = yearly[(yearly["Year"] >= 1900) & (yearly["Year"] <= 2100)]
        
        # Add category percentages per year
        yearly_categories = sentiment_df.dropna(subset=["Year"]).groupby("Year")["Sentiment Category"].value_counts(normalize=True).unstack(fill_value=0) * 100
        yearly_categories = yearly_categories.reset_index()
        
        if "Positive" not in yearly_categories.columns:
            yearly_categories["Positive"] = 0
        if "Negative" not in yearly_categories.columns:
            yearly_categories["Negative"] = 0
        if "Neutral" not in yearly_categories.columns:
            yearly_categories["Neutral"] = 0
        
        temporal_trends = yearly.merge(yearly_categories[["Year", "Positive", "Negative", "Neutral"]], 
                                       on="Year", how="left")
        temporal_trends.rename(columns={"Positive": "Positive %", "Negative": "Negative %", "Neutral": "Neutral %"}, inplace=True)
    
    # ==========================================================================
    # SOURCE ANALYSIS
    # ==========================================================================
    source_analysis = None
    
    if analyze_by_source and source_column in df.columns:
        if verbose:
            print("  Analyzing sentiment by source...")
        
        sentiment_df["Source"] = df[source_column]
        
        source_stats = sentiment_df.groupby("Source").agg({
            "Composite Sentiment": ["mean", "median", "std", "count"],
            "Certainty Score": "mean",
        }).reset_index()
        
        source_stats.columns = ["Source", "Mean Sentiment", "Median Sentiment", 
                               "Std Sentiment", "Document Count", "Mean Certainty"]
        
        # Filter to sources with at least 5 documents
        source_analysis = source_stats[source_stats["Document Count"] >= 5].sort_values(
            "Mean Sentiment", ascending=False
        )
    
    # ==========================================================================
    # WORD ANALYSIS
    # ==========================================================================
    word_analysis = {}
    
    if verbose:
        print("  Analyzing words by sentiment category...")
    
    for category in ["Positive", "Negative", "Neutral"]:
        category_texts = df.loc[
            sentiment_df["Sentiment Category"] == category, text_column
        ].dropna().astype(str)
        
        if len(category_texts) > 0:
            all_words = " ".join(category_texts).lower().split()
            # Filter short words and stopwords
            filtered_words = [w for w in all_words if len(w) > 3 and w.isalpha()]
            word_counts = Counter(filtered_words)
            word_analysis[category] = {
                "top_words": word_counts.most_common(top_words),
                "total_words": len(all_words),
                "unique_words": len(set(all_words)),
                "documents": len(category_texts),
            }
    
    # ==========================================================================
    # CERTAINTY ANALYSIS
    # ==========================================================================
    certainty_analysis = None
    
    if use_scientific:
        certainty_analysis = {
            "mean_certainty": float(certainty_scores.mean()),
            "median_certainty": float(np.median(certainty_scores)),
            "mean_hedging": float(hedging_scores.mean()),
            "total_high_certainty_markers": int(certainty_markers_high.sum()),
            "total_moderate_certainty_markers": int(certainty_markers_moderate.sum()),
            "total_low_certainty_markers": int(certainty_markers_low.sum()),
            "total_hedging_markers": int(certainty_markers_hedging.sum()),
            "avg_scientific_positive_per_doc": float(scientific_positive.mean()),
            "avg_scientific_negative_per_doc": float(scientific_negative.mean()),
            "certainty_distribution": {
                "high": int((certainty_scores >= 0.7).sum()),
                "moderate": int(((certainty_scores >= 0.4) & (certainty_scores < 0.7)).sum()),
                "low": int((certainty_scores < 0.4).sum()),
            }
        }
    
    # ==========================================================================
    # TOP DOCUMENTS
    # ==========================================================================
    top_positive = sentiment_df.nlargest(20, "Composite Sentiment").copy()
    top_negative = sentiment_df.nsmallest(20, "Composite Sentiment").copy()
    
    # ==========================================================================
    # CORRELATIONS
    # ==========================================================================
    correlations = None
    
    numeric_cols = ["Composite Sentiment", "VADER Compound", "TextBlob Polarity",
                   "Certainty Score", "Hedging Score", "TextBlob Subjectivity"]
    
    if year_column in df.columns:
        sentiment_df["Year"] = pd.to_numeric(df[year_column], errors="coerce")
        numeric_cols.append("Year")
    
    # Add citations if available
    for cite_col in ["Cited by", "Times Cited", "TC", "Citation Count"]:
        if cite_col in df.columns:
            sentiment_df[cite_col] = pd.to_numeric(df[cite_col], errors="coerce")
            numeric_cols.append(cite_col)
            break
    
    available_cols = [c for c in numeric_cols if c in sentiment_df.columns]
    if len(available_cols) > 1:
        correlations = sentiment_df[available_cols].corr()
    
    # ==========================================================================
    # SUMMARY OUTPUT
    # ==========================================================================
    if verbose:
        print(f"\n📊 Sentiment Analysis Results")
        print("=" * 50)
        print(f"Documents analyzed: {n_docs}")
        print(f"Mean composite sentiment: {statistics['mean_sentiment']:.3f}")
        print(f"Positive: {statistics['positive_count']} ({statistics['positive_pct']:.1f}%)")
        print(f"Negative: {statistics['negative_count']} ({statistics['negative_pct']:.1f}%)")
        print(f"Neutral: {statistics['neutral_count']} ({statistics['neutral_pct']:.1f}%)")
        if use_scientific:
            print(f"Mean certainty: {certainty_analysis['mean_certainty']:.3f}")
            print(f"Mean hedging: {certainty_analysis['mean_hedging']:.3f}")
    
    return {
        "sentiment_df": sentiment_df,
        "statistics": statistics,
        "temporal_trends": temporal_trends,
        "source_analysis": source_analysis,
        "word_analysis": word_analysis,
        "certainty_analysis": certainty_analysis,
        "correlations": correlations,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "text_column": text_column,
        "methods_used": {
            "vader": use_vader,
            "textblob": use_textblob,
            "scientific": use_scientific,
        },
    }


def get_sentiment_by_entity(
    df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    entity_column: str,
    sentiment_column: str = "Composite Sentiment",
    min_count: int = 5,
    separator: str = ";",
) -> pd.DataFrame:
    """
    Analyze sentiment by entity (e.g., by author, keyword, country).
    
    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame with entity column.
    sentiment_df : pd.DataFrame
        DataFrame with sentiment scores (from analyze_sentiment_advanced).
    entity_column : str
        Column containing entities (can be multi-valued with separator).
    sentiment_column : str
        Column with sentiment scores to aggregate.
    min_count : int
        Minimum number of documents per entity.
    separator : str
        Separator for multi-valued fields.
    
    Returns
    -------
    pd.DataFrame
        Entity-level sentiment statistics.
    """
    if entity_column not in df.columns:
        raise ValueError(f"Entity column '{entity_column}' not found")
    
    if sentiment_column not in sentiment_df.columns:
        raise ValueError(f"Sentiment column '{sentiment_column}' not found")
    
    # Explode multi-valued entities
    entity_sentiment = []
    
    for idx in df.index:
        entities = df.loc[idx, entity_column]
        sentiment = sentiment_df.loc[idx, sentiment_column] if idx in sentiment_df.index else np.nan
        
        if pd.isna(entities) or pd.isna(sentiment):
            continue
        
        if isinstance(entities, str):
            entity_list = [e.strip() for e in entities.split(separator) if e.strip()]
        else:
            entity_list = [str(entities)]
        
        for entity in entity_list:
            entity_sentiment.append({
                "Entity": entity,
                "Sentiment": sentiment,
            })
    
    if not entity_sentiment:
        return pd.DataFrame()
    
    entity_df = pd.DataFrame(entity_sentiment)
    
    # Aggregate by entity
    result = entity_df.groupby("Entity").agg({
        "Sentiment": ["mean", "median", "std", "count"]
    }).reset_index()
    
    result.columns = ["Entity", "Mean Sentiment", "Median Sentiment", "Std Sentiment", "Document Count"]
    
    # Filter by minimum count
    result = result[result["Document Count"] >= min_count].sort_values("Mean Sentiment", ascending=False)
    
    return result


def compare_sentiment_groups(
    sentiment_df: pd.DataFrame,
    group_column: str,
    sentiment_column: str = "Composite Sentiment",
) -> Dict[str, Any]:
    """
    Compare sentiment across groups using statistical tests.
    
    Parameters
    ----------
    sentiment_df : pd.DataFrame
        DataFrame with sentiment scores and group labels.
    group_column : str
        Column containing group labels.
    sentiment_column : str
        Column with sentiment scores.
    
    Returns
    -------
    dict
        Statistical comparison results.
    """
    from scipy import stats as scipy_stats
    
    if group_column not in sentiment_df.columns:
        raise ValueError(f"Group column '{group_column}' not found")
    
    groups = sentiment_df.groupby(group_column)[sentiment_column].apply(list).to_dict()
    group_names = list(groups.keys())
    
    result = {
        "n_groups": len(group_names),
        "groups": {},
    }
    
    # Per-group statistics
    for name, values in groups.items():
        values = [v for v in values if not pd.isna(v)]
        result["groups"][name] = {
            "n": len(values),
            "mean": np.mean(values) if values else 0,
            "median": np.median(values) if values else 0,
            "std": np.std(values) if values else 0,
        }
    
    # Statistical tests
    if len(group_names) == 2:
        # Two groups: t-test and Mann-Whitney U
        g1, g2 = [np.array([v for v in groups[n] if not pd.isna(v)]) for n in group_names[:2]]
        
        if len(g1) > 1 and len(g2) > 1:
            t_stat, t_pval = scipy_stats.ttest_ind(g1, g2)
            u_stat, u_pval = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
            
            result["t_test"] = {"statistic": t_stat, "p_value": t_pval}
            result["mann_whitney_u"] = {"statistic": u_stat, "p_value": u_pval}
    
    elif len(group_names) > 2:
        # Multiple groups: ANOVA and Kruskal-Wallis
        group_arrays = [np.array([v for v in groups[n] if not pd.isna(v)]) for n in group_names]
        group_arrays = [g for g in group_arrays if len(g) > 0]
        
        if len(group_arrays) >= 2:
            f_stat, f_pval = scipy_stats.f_oneway(*group_arrays)
            h_stat, h_pval = scipy_stats.kruskal(*group_arrays)
            
            result["anova"] = {"f_statistic": f_stat, "p_value": f_pval}
            result["kruskal_wallis"] = {"h_statistic": h_stat, "p_value": h_pval}
    
    return result
