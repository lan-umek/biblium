# -*- coding: utf-8 -*-
"""
Crosstabs / Contingency Table Analysis
======================================
Analyze associations between categorical variables.

Features:
- Cross-tabulation (frequency and percentage tables)
- Chi-squared test of independence
- Fisher's exact test (for 2x2 tables)
- Cramér's V effect size
- Phi coefficient (for 2x2 tables)
- Expected frequencies
- Standardized residuals
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class ChiSquaredResult:
    """Result of Chi-squared test."""
    statistic: float = 0.0
    p_value: float = 0.0
    df: int = 0
    is_significant: bool = False
    min_expected: float = 0.0  # Minimum expected frequency
    cells_below_5: int = 0  # Number of cells with expected < 5
    valid: bool = True  # Whether assumptions are met
    warning: str = ""


@dataclass
class FisherResult:
    """Result of Fisher's exact test."""
    odds_ratio: float = 0.0
    p_value: float = 0.0
    is_significant: bool = False
    ci_lower: float = 0.0
    ci_upper: float = 0.0


@dataclass
class EffectSize:
    """Effect size measures for contingency tables."""
    cramers_v: float = 0.0
    cramers_v_interpretation: str = ""
    phi: float = 0.0  # Only for 2x2 tables
    phi_interpretation: str = ""
    contingency_coef: float = 0.0  # Pearson's contingency coefficient


@dataclass
class CrosstabResult:
    """Complete result of crosstab analysis."""
    # Variables
    row_var: str = ""
    col_var: str = ""
    
    # Dimensions
    n_rows: int = 0
    n_cols: int = 0
    n_total: int = 0
    is_2x2: bool = False
    
    # Tables
    observed: pd.DataFrame = None  # Observed frequencies
    expected: pd.DataFrame = None  # Expected frequencies
    row_pct: pd.DataFrame = None  # Row percentages
    col_pct: pd.DataFrame = None  # Column percentages
    total_pct: pd.DataFrame = None  # Total percentages
    residuals: pd.DataFrame = None  # Standardized residuals
    
    # Tests
    chi_squared: ChiSquaredResult = None
    fisher: FisherResult = None  # Only for 2x2
    
    # Effect sizes
    effect_size: EffectSize = None
    
    # Interpretation
    interpretation: str = ""


def compute_crosstab(
    df: pd.DataFrame,
    row_var: str,
    col_var: str,
    alpha: float = 0.05,
    verbose: bool = True,
) -> CrosstabResult:
    """
    Compute cross-tabulation and statistical tests.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    row_var : str
        Row variable name (categorical)
    col_var : str
        Column variable name (categorical)
    alpha : float
        Significance level
    verbose : bool
        Print progress
        
    Returns
    -------
    CrosstabResult
    """
    from scipy import stats
    
    result = CrosstabResult(row_var=row_var, col_var=col_var)
    
    # Validate columns
    if row_var not in df.columns:
        raise ValueError(f"Row variable '{row_var}' not found")
    if col_var not in df.columns:
        raise ValueError(f"Column variable '{col_var}' not found")
    
    # Clean data
    clean_df = df[[row_var, col_var]].dropna()
    result.n_total = len(clean_df)
    
    if result.n_total < 2:
        raise ValueError("Insufficient data for analysis")
    
    # Create observed frequency table
    observed = pd.crosstab(clean_df[row_var], clean_df[col_var], margins=True, margins_name="Total")
    
    # Store without margins for calculations
    obs_no_margins = pd.crosstab(clean_df[row_var], clean_df[col_var])
    
    result.observed = observed
    result.n_rows = len(obs_no_margins.index)
    result.n_cols = len(obs_no_margins.columns)
    result.is_2x2 = (result.n_rows == 2 and result.n_cols == 2)
    
    if verbose:
        print(f"Crosstab Analysis: {row_var} × {col_var}")
        print(f"  Dimensions: {result.n_rows} × {result.n_cols}")
        print(f"  Total N: {result.n_total}")
    
    # Calculate percentages
    result.row_pct = pd.crosstab(clean_df[row_var], clean_df[col_var], normalize='index', margins=True, margins_name="Total") * 100
    result.col_pct = pd.crosstab(clean_df[row_var], clean_df[col_var], normalize='columns', margins=True, margins_name="Total") * 100
    result.total_pct = pd.crosstab(clean_df[row_var], clean_df[col_var], normalize='all', margins=True, margins_name="Total") * 100
    
    # Chi-squared test
    chi2_result = ChiSquaredResult()
    try:
        chi2, p, dof, expected = stats.chi2_contingency(obs_no_margins)
        
        chi2_result.statistic = chi2
        chi2_result.p_value = p
        chi2_result.df = dof
        chi2_result.is_significant = p < alpha
        
        # Create expected frequencies DataFrame
        result.expected = pd.DataFrame(
            expected,
            index=obs_no_margins.index,
            columns=obs_no_margins.columns
        )
        
        # Check assumptions
        chi2_result.min_expected = expected.min()
        chi2_result.cells_below_5 = np.sum(expected < 5)
        total_cells = expected.size
        
        if chi2_result.min_expected < 1:
            chi2_result.valid = False
            chi2_result.warning = "Some expected frequencies < 1. Chi-squared may be invalid."
        elif chi2_result.cells_below_5 > 0.2 * total_cells:
            chi2_result.valid = False
            chi2_result.warning = f"More than 20% of cells have expected frequency < 5 ({chi2_result.cells_below_5}/{total_cells} cells)."
        
        # Calculate standardized residuals
        obs_array = obs_no_margins.values
        residuals = (obs_array - expected) / np.sqrt(expected)
        result.residuals = pd.DataFrame(
            residuals,
            index=obs_no_margins.index,
            columns=obs_no_margins.columns
        )
        
    except Exception as e:
        chi2_result.valid = False
        chi2_result.warning = str(e)
    
    result.chi_squared = chi2_result
    
    # Fisher's exact test (for 2x2 tables)
    if result.is_2x2:
        fisher_result = FisherResult()
        try:
            odds_ratio, p_fisher = stats.fisher_exact(obs_no_margins.values)
            fisher_result.odds_ratio = odds_ratio
            fisher_result.p_value = p_fisher
            fisher_result.is_significant = p_fisher < alpha
            
            # Confidence interval for odds ratio (approximate)
            if odds_ratio > 0 and not np.isinf(odds_ratio):
                a, b, c, d = obs_no_margins.values.flatten()
                se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if min(a,b,c,d) > 0 else np.nan
                if not np.isnan(se_log_or):
                    log_or = np.log(odds_ratio)
                    fisher_result.ci_lower = np.exp(log_or - 1.96 * se_log_or)
                    fisher_result.ci_upper = np.exp(log_or + 1.96 * se_log_or)
                    
        except Exception as e:
            if verbose:
                print(f"  Fisher's test error: {e}")
        
        result.fisher = fisher_result
    
    # Effect sizes
    effect = EffectSize()
    
    # Cramér's V
    if chi2_result.statistic > 0:
        min_dim = min(result.n_rows - 1, result.n_cols - 1)
        if min_dim > 0:
            effect.cramers_v = np.sqrt(chi2_result.statistic / (result.n_total * min_dim))
            
            # Interpretation (Cohen's guidelines adapted for Cramér's V)
            if min_dim == 1:  # 2xk or kx2 table
                if effect.cramers_v < 0.1:
                    effect.cramers_v_interpretation = "Negligible"
                elif effect.cramers_v < 0.3:
                    effect.cramers_v_interpretation = "Small"
                elif effect.cramers_v < 0.5:
                    effect.cramers_v_interpretation = "Medium"
                else:
                    effect.cramers_v_interpretation = "Large"
            else:  # Larger tables
                if effect.cramers_v < 0.07:
                    effect.cramers_v_interpretation = "Negligible"
                elif effect.cramers_v < 0.21:
                    effect.cramers_v_interpretation = "Small"
                elif effect.cramers_v < 0.35:
                    effect.cramers_v_interpretation = "Medium"
                else:
                    effect.cramers_v_interpretation = "Large"
    
    # Phi coefficient (only meaningful for 2x2)
    if result.is_2x2:
        effect.phi = np.sqrt(chi2_result.statistic / result.n_total)
        if effect.phi < 0.1:
            effect.phi_interpretation = "Negligible"
        elif effect.phi < 0.3:
            effect.phi_interpretation = "Small"
        elif effect.phi < 0.5:
            effect.phi_interpretation = "Medium"
        else:
            effect.phi_interpretation = "Large"
    
    # Contingency coefficient
    if chi2_result.statistic > 0:
        effect.contingency_coef = np.sqrt(chi2_result.statistic / (chi2_result.statistic + result.n_total))
    
    result.effect_size = effect
    
    # Generate interpretation
    if chi2_result.is_significant:
        result.interpretation = (
            f"There is a statistically significant association between {row_var} and {col_var} "
            f"(χ²({chi2_result.df}) = {chi2_result.statistic:.3f}, p = {chi2_result.p_value:.4f}). "
        )
        if result.is_2x2:
            result.interpretation += f"Effect size: φ = {effect.phi:.3f} ({effect.phi_interpretation}). "
            if result.fisher:
                result.interpretation += f"Odds ratio = {result.fisher.odds_ratio:.2f}."
        else:
            result.interpretation += f"Effect size: Cramér's V = {effect.cramers_v:.3f} ({effect.cramers_v_interpretation})."
    else:
        result.interpretation = (
            f"No statistically significant association was found between {row_var} and {col_var} "
            f"(χ²({chi2_result.df}) = {chi2_result.statistic:.3f}, p = {chi2_result.p_value:.4f})."
        )
    
    if chi2_result.warning:
        result.interpretation += f"\n\nWarning: {chi2_result.warning}"
        if result.is_2x2 and result.fisher:
            result.interpretation += f" Consider using Fisher's exact test (p = {result.fisher.p_value:.4f})."
    
    if verbose:
        print(f"  Chi-squared: χ²({chi2_result.df}) = {chi2_result.statistic:.3f}, p = {chi2_result.p_value:.4f}")
        if result.is_2x2 and result.fisher:
            print(f"  Fisher's exact: p = {result.fisher.p_value:.4f}, OR = {result.fisher.odds_ratio:.3f}")
        print(f"  Cramér's V: {effect.cramers_v:.3f} ({effect.cramers_v_interpretation})")
    
    return result


def get_categorical_columns(df: pd.DataFrame, max_categories: int = 50) -> List[str]:
    """Get list of columns suitable for crosstab analysis."""
    cat_cols = []
    for col in df.columns:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= max_categories:
            cat_cols.append(col)
    return cat_cols


def format_crosstab_table(
    observed: pd.DataFrame,
    row_pct: pd.DataFrame = None,
    col_pct: pd.DataFrame = None,
    show_pct: str = "row",  # "row", "col", "both", "none"
) -> pd.DataFrame:
    """
    Format crosstab with counts and percentages.
    
    Parameters
    ----------
    observed : pd.DataFrame
        Observed frequencies
    row_pct : pd.DataFrame
        Row percentages
    col_pct : pd.DataFrame
        Column percentages
    show_pct : str
        Which percentages to show
        
    Returns
    -------
    pd.DataFrame
        Formatted table with counts and percentages
    """
    if show_pct == "none" or (row_pct is None and col_pct is None):
        return observed.copy()
    
    result = observed.copy().astype(str)
    
    for row in observed.index:
        for col in observed.columns:
            count = observed.loc[row, col]
            
            if show_pct == "row" and row_pct is not None:
                pct = row_pct.loc[row, col]
                result.loc[row, col] = f"{count} ({pct:.1f}%)"
            elif show_pct == "col" and col_pct is not None:
                pct = col_pct.loc[row, col]
                result.loc[row, col] = f"{count} ({pct:.1f}%)"
            elif show_pct == "both":
                r_pct = row_pct.loc[row, col] if row_pct is not None else 0
                c_pct = col_pct.loc[row, col] if col_pct is not None else 0
                result.loc[row, col] = f"{count}\n({r_pct:.1f}% | {c_pct:.1f}%)"
    
    return result
