# -*- coding: utf-8 -*-
"""
Correlation Analysis
====================
Analyze correlations between numeric variables.

Features:
- Multiple correlation coefficients (Pearson, Spearman, Kendall)
- Correlation matrix with p-values
- Correlation matrix visualization (heatmap)
- Scatter plot matrix with histograms on diagonal
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class CorrelationPair:
    """Correlation result for a pair of variables."""
    var1: str = ""
    var2: str = ""
    correlation: float = 0.0
    p_value: float = 0.0
    n: int = 0
    is_significant: bool = False
    

@dataclass
class CorrelationResult:
    """Complete correlation analysis result."""
    variables: List[str] = field(default_factory=list)
    n_vars: int = 0
    method: str = "pearson"
    method_name: str = "Pearson"
    alpha: float = 0.05
    
    # Matrices
    corr_matrix: pd.DataFrame = None
    p_matrix: pd.DataFrame = None
    n_matrix: pd.DataFrame = None  # Sample sizes (may vary due to missing data)
    
    # Pairwise results
    pairs: List[CorrelationPair] = field(default_factory=list)
    
    # Statistics
    n_significant: int = 0
    n_total_pairs: int = 0


def compute_correlation(
    df: pd.DataFrame,
    variables: List[str] = None,
    method: str = "pearson",
    alpha: float = 0.05,
    verbose: bool = True,
) -> CorrelationResult:
    """
    Compute correlation matrix with p-values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    variables : list
        List of numeric variables to correlate (default: all numeric)
    method : str
        Correlation method: 'pearson', 'spearman', 'kendall'
    alpha : float
        Significance level
    verbose : bool
        Print progress
        
    Returns
    -------
    CorrelationResult
    """
    from scipy import stats
    
    result = CorrelationResult(method=method, alpha=alpha)
    
    # Method names
    method_names = {
        'pearson': 'Pearson',
        'spearman': 'Spearman',
        'kendall': 'Kendall'
    }
    result.method_name = method_names.get(method, method.capitalize())
    
    # Get numeric variables
    if variables is None:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Filter to only numeric
        variables = [v for v in variables if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]
    
    if len(variables) < 2:
        raise ValueError("At least 2 numeric variables required")
    
    result.variables = variables
    result.n_vars = len(variables)
    
    if verbose:
        print(f"Correlation Analysis: {result.method_name}")
        print(f"  Variables: {result.n_vars}")
    
    # Subset dataframe
    data = df[variables].copy()
    
    # Initialize matrices
    n = len(variables)
    corr_mat = np.zeros((n, n))
    p_mat = np.zeros((n, n))
    n_mat = np.zeros((n, n), dtype=int)
    
    # Compute pairwise correlations
    pairs = []
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                corr_mat[i, j] = 1.0
                p_mat[i, j] = 0.0
                n_mat[i, j] = data[var1].notna().sum()
            else:
                # Get complete cases for this pair
                valid = data[[var1, var2]].dropna()
                n_valid = len(valid)
                n_mat[i, j] = n_valid
                
                if n_valid < 3:
                    corr_mat[i, j] = np.nan
                    p_mat[i, j] = np.nan
                else:
                    x, y = valid[var1].values, valid[var2].values
                    
                    if method == 'pearson':
                        r, p = stats.pearsonr(x, y)
                    elif method == 'spearman':
                        r, p = stats.spearmanr(x, y)
                    elif method == 'kendall':
                        r, p = stats.kendalltau(x, y)
                    else:
                        r, p = stats.pearsonr(x, y)
                    
                    corr_mat[i, j] = r
                    p_mat[i, j] = p
                    
                    # Store pair result (only upper triangle to avoid duplicates)
                    if i < j:
                        pair = CorrelationPair(
                            var1=var1,
                            var2=var2,
                            correlation=r,
                            p_value=p,
                            n=n_valid,
                            is_significant=p < alpha
                        )
                        pairs.append(pair)
    
    # Create DataFrames
    result.corr_matrix = pd.DataFrame(corr_mat, index=variables, columns=variables)
    result.p_matrix = pd.DataFrame(p_mat, index=variables, columns=variables)
    result.n_matrix = pd.DataFrame(n_mat, index=variables, columns=variables)
    
    # Sort pairs by absolute correlation
    pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
    result.pairs = pairs
    
    # Count significant
    result.n_significant = sum(1 for p in pairs if p.is_significant)
    result.n_total_pairs = len(pairs)
    
    if verbose:
        print(f"  Significant pairs: {result.n_significant}/{result.n_total_pairs}")
        if pairs:
            top = pairs[0]
            print(f"  Strongest: {top.var1} × {top.var2} = {top.correlation:.3f}")
    
    return result


def get_correlation_interpretation(r: float) -> str:
    """Get interpretation of correlation coefficient magnitude."""
    r_abs = abs(r)
    if r_abs < 0.1:
        return "Negligible"
    elif r_abs < 0.3:
        return "Weak"
    elif r_abs < 0.5:
        return "Moderate"
    elif r_abs < 0.7:
        return "Strong"
    else:
        return "Very strong"


def plot_correlation_matrix(
    result: CorrelationResult,
    show_values: bool = True,
    show_significance: bool = True,
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = None,
    title: str = None,
) -> "matplotlib.figure.Figure":
    """
    Plot correlation matrix heatmap.
    
    Parameters
    ----------
    result : CorrelationResult
        Correlation analysis result
    show_values : bool
        Show correlation values in cells
    show_significance : bool
        Mark significant correlations with asterisks
    cmap : str
        Colormap name
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    n = result.n_vars
    if figsize is None:
        size = max(6, min(12, n * 0.8))
        figsize = (size, size)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    corr = result.corr_matrix.values
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'{result.method_name} Correlation', fontsize=10)
    
    # Set ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(result.variables, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(result.variables, fontsize=9)
    
    # Add values
    if show_values:
        for i in range(n):
            for j in range(n):
                r = corr[i, j]
                p = result.p_matrix.values[i, j]
                
                if np.isnan(r):
                    text = "-"
                else:
                    text = f"{r:.2f}"
                    if show_significance and i != j and p < result.alpha:
                        text += "*"
                
                # Choose text color based on background
                color = "white" if abs(r) > 0.5 else "black"
                ax.text(j, i, text, ha='center', va='center', 
                       fontsize=8, color=color)
    
    # Remove gridlines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Title
    if title is None:
        title = f"{result.method_name} Correlation Matrix"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_scatter_matrix(
    df: pd.DataFrame,
    variables: List[str],
    figsize: Tuple[int, int] = None,
    alpha: float = 0.5,
    color: str = "#3498db",
    hist_color: str = "#2c3e50",
    title: str = None,
) -> "matplotlib.figure.Figure":
    """
    Plot scatter matrix with histograms on diagonal.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    variables : list
        Variables to include
    figsize : tuple
        Figure size
    alpha : float
        Point transparency
    color : str
        Scatter plot color
    hist_color : str
        Histogram color
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    n = len(variables)
    if figsize is None:
        size = max(8, min(14, n * 2))
        figsize = (size, size)
    
    fig, axes = plt.subplots(n, n, figsize=figsize)
    
    # Handle single variable case
    if n == 1:
        axes = np.array([[axes]])
    
    data = df[variables].dropna()
    
    for i, var_i in enumerate(variables):
        for j, var_j in enumerate(variables):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(data[var_i], bins=20, color=hist_color, alpha=0.7, edgecolor='white')
                ax.set_ylabel('')
            else:
                # Off-diagonal: scatter plot
                ax.scatter(data[var_j], data[var_i], alpha=alpha, s=15, c=color, edgecolors='none')
            
            # Remove gridlines
            ax.grid(False)
            
            # Labels only on edges
            if i == n - 1:
                ax.set_xlabel(var_j, fontsize=9)
            else:
                ax.set_xticklabels([])
            
            if j == 0:
                ax.set_ylabel(var_i, fontsize=9)
            else:
                ax.set_yticklabels([])
            
            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Smaller tick labels
            ax.tick_params(labelsize=7)
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()
