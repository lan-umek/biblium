# -*- coding: utf-8 -*-
"""
Compare Means Analysis
======================
SPSS-like statistical analysis for comparing means across groups.

Features:
- Descriptive statistics by group
- Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)
- Homogeneity of variance (Levene's test)
- T-test (independent samples, 2 groups)
- ANOVA (3+ groups) with post-hoc tests
- Non-parametric alternatives (Mann-Whitney U, Kruskal-Wallis)
- Effect size measures (Cohen's d, eta-squared)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd


@dataclass
class GroupDescriptives:
    """Descriptive statistics for a single group."""
    group_name: str
    n: int = 0
    mean: float = 0.0
    std: float = 0.0
    se: float = 0.0  # Standard error
    median: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    q1: float = 0.0  # 25th percentile
    q3: float = 0.0  # 75th percentile
    skewness: float = 0.0
    kurtosis: float = 0.0
    ci_lower: float = 0.0  # 95% CI lower
    ci_upper: float = 0.0  # 95% CI upper


@dataclass
class NormalityTest:
    """Result of normality test."""
    test_name: str
    statistic: float = 0.0
    p_value: float = 0.0
    is_normal: bool = True  # p > 0.05


@dataclass
class HomogeneityTest:
    """Result of homogeneity of variance test."""
    test_name: str = "Levene's Test"
    statistic: float = 0.0
    p_value: float = 0.0
    is_homogeneous: bool = True  # p > 0.05


@dataclass
class StatisticalTest:
    """Result of a statistical test."""
    test_name: str
    statistic: float = 0.0
    df: float = 0.0  # Degrees of freedom
    df2: float = 0.0  # Second df for F-tests
    p_value: float = 0.0
    effect_size: float = 0.0
    effect_size_name: str = ""
    effect_size_interpretation: str = ""
    is_significant: bool = False  # p < 0.05
    notes: str = ""


@dataclass
class PostHocResult:
    """Result of post-hoc pairwise comparison."""
    group1: str
    group2: str
    mean_diff: float = 0.0
    statistic: float = 0.0
    p_value: float = 0.0
    p_adjusted: float = 0.0  # Bonferroni/other correction
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    is_significant: bool = False


@dataclass
class CompareMeansResult:
    """Complete result of compare means analysis."""
    # Variable info
    dependent_var: str = ""
    grouping_var: str = ""
    n_groups: int = 0
    n_total: int = 0
    
    # Descriptives
    overall_descriptives: GroupDescriptives = None
    group_descriptives: List[GroupDescriptives] = field(default_factory=list)
    
    # Assumption tests
    normality_tests: Dict[str, NormalityTest] = field(default_factory=dict)
    homogeneity_test: HomogeneityTest = None
    assumptions_met: bool = True
    
    # Main tests
    parametric_test: StatisticalTest = None
    nonparametric_test: StatisticalTest = None
    
    # Post-hoc (for 3+ groups)
    post_hoc_results: List[PostHocResult] = field(default_factory=list)
    post_hoc_method: str = ""
    
    # Recommendations
    recommended_test: str = ""
    interpretation: str = ""


def compute_descriptives(data: pd.Series, group_name: str = "Overall") -> GroupDescriptives:
    """
    Compute descriptive statistics for a data series.
    
    Parameters
    ----------
    data : pd.Series
        Numeric data
    group_name : str
        Name of the group
        
    Returns
    -------
    GroupDescriptives
    """
    from scipy import stats
    
    data = data.dropna()
    n = len(data)
    
    if n == 0:
        return GroupDescriptives(group_name=group_name)
    
    mean = data.mean()
    std = data.std(ddof=1) if n > 1 else 0
    se = std / np.sqrt(n) if n > 0 else 0
    
    # 95% CI
    if n > 1:
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
    else:
        ci_lower = ci_upper = mean
    
    # Skewness and kurtosis
    try:
        skewness = stats.skew(data) if n > 2 else 0
        kurtosis = stats.kurtosis(data) if n > 3 else 0
    except:
        skewness = kurtosis = 0
    
    return GroupDescriptives(
        group_name=group_name,
        n=n,
        mean=mean,
        std=std,
        se=se,
        median=data.median(),
        min_val=data.min(),
        max_val=data.max(),
        q1=data.quantile(0.25),
        q3=data.quantile(0.75),
        skewness=skewness,
        kurtosis=kurtosis,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def test_normality(data: pd.Series, group_name: str = "") -> NormalityTest:
    """
    Test for normality using Shapiro-Wilk test.
    
    Parameters
    ----------
    data : pd.Series
        Numeric data
    group_name : str
        Name of the group for labeling
        
    Returns
    -------
    NormalityTest
    """
    from scipy import stats
    
    data = data.dropna()
    n = len(data)
    
    test_name = f"Shapiro-Wilk ({group_name})" if group_name else "Shapiro-Wilk"
    
    if n < 3:
        return NormalityTest(test_name=test_name, statistic=np.nan, p_value=np.nan, is_normal=True)
    
    if n > 5000:
        # Use Kolmogorov-Smirnov for large samples
        test_name = f"Kolmogorov-Smirnov ({group_name})" if group_name else "Kolmogorov-Smirnov"
        # Standardize data for K-S test against normal
        data_std = (data - data.mean()) / data.std()
        stat, p = stats.kstest(data_std, 'norm')
    else:
        try:
            stat, p = stats.shapiro(data)
        except:
            return NormalityTest(test_name=test_name, statistic=np.nan, p_value=np.nan, is_normal=True)
    
    return NormalityTest(
        test_name=test_name,
        statistic=stat,
        p_value=p,
        is_normal=p > 0.05
    )


def test_homogeneity(groups: List[pd.Series]) -> HomogeneityTest:
    """
    Test homogeneity of variance using Levene's test.
    
    Parameters
    ----------
    groups : list of pd.Series
        Data for each group
        
    Returns
    -------
    HomogeneityTest
    """
    from scipy import stats
    
    # Remove NaN and filter empty groups
    clean_groups = [g.dropna() for g in groups]
    clean_groups = [g for g in clean_groups if len(g) > 0]
    
    if len(clean_groups) < 2:
        return HomogeneityTest(statistic=np.nan, p_value=np.nan, is_homogeneous=True)
    
    try:
        stat, p = stats.levene(*clean_groups)
    except:
        return HomogeneityTest(statistic=np.nan, p_value=np.nan, is_homogeneous=True)
    
    return HomogeneityTest(
        test_name="Levene's Test",
        statistic=stat,
        p_value=p,
        is_homogeneous=p > 0.05
    )


def cohens_d(group1: pd.Series, group2: pd.Series) -> Tuple[float, str]:
    """
    Calculate Cohen's d effect size for two groups.
    
    Returns
    -------
    (d, interpretation)
    """
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0, "N/A"
    
    mean1, mean2 = g1.mean(), g2.mean()
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0, "N/A"
    
    d = abs(mean1 - mean2) / pooled_std
    
    # Interpretation
    if d < 0.2:
        interp = "Negligible"
    elif d < 0.5:
        interp = "Small"
    elif d < 0.8:
        interp = "Medium"
    else:
        interp = "Large"
    
    return d, interp


def eta_squared(f_stat: float, df_between: int, df_within: int) -> Tuple[float, str]:
    """
    Calculate eta-squared effect size for ANOVA.
    
    Returns
    -------
    (eta_sq, interpretation)
    """
    if np.isnan(f_stat) or df_between <= 0:
        return 0.0, "N/A"
    
    ss_between = f_stat * df_between
    ss_total = ss_between + df_within
    
    if ss_total == 0:
        return 0.0, "N/A"
    
    eta_sq = ss_between / (ss_between + df_within)
    
    # Interpretation
    if eta_sq < 0.01:
        interp = "Negligible"
    elif eta_sq < 0.06:
        interp = "Small"
    elif eta_sq < 0.14:
        interp = "Medium"
    else:
        interp = "Large"
    
    return eta_sq, interp


def independent_t_test(group1: pd.Series, group2: pd.Series, equal_var: bool = True) -> StatisticalTest:
    """
    Perform independent samples t-test.
    
    Parameters
    ----------
    group1, group2 : pd.Series
        Data for each group
    equal_var : bool
        If True, use Student's t-test; if False, use Welch's t-test
        
    Returns
    -------
    StatisticalTest
    """
    from scipy import stats
    
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    test_name = "Independent t-test" if equal_var else "Welch's t-test"
    
    if len(g1) < 2 or len(g2) < 2:
        return StatisticalTest(test_name=test_name, notes="Insufficient data")
    
    try:
        stat, p = stats.ttest_ind(g1, g2, equal_var=equal_var)
        
        # Degrees of freedom
        if equal_var:
            df = len(g1) + len(g2) - 2
        else:
            # Welch-Satterthwaite approximation
            v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
            n1, n2 = len(g1), len(g2)
            df = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        
        # Effect size
        d, d_interp = cohens_d(g1, g2)
        
        return StatisticalTest(
            test_name=test_name,
            statistic=stat,
            df=df,
            p_value=p,
            effect_size=d,
            effect_size_name="Cohen's d",
            effect_size_interpretation=d_interp,
            is_significant=p < 0.05,
        )
    except Exception as e:
        return StatisticalTest(test_name=test_name, notes=str(e))


def mann_whitney_u(group1: pd.Series, group2: pd.Series) -> StatisticalTest:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).
    
    Parameters
    ----------
    group1, group2 : pd.Series
        Data for each group
        
    Returns
    -------
    StatisticalTest
    """
    from scipy import stats
    
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    if len(g1) < 1 or len(g2) < 1:
        return StatisticalTest(test_name="Mann-Whitney U", notes="Insufficient data")
    
    try:
        stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        
        # Effect size: rank-biserial correlation
        n1, n2 = len(g1), len(g2)
        r = 1 - (2 * stat) / (n1 * n2)
        
        if abs(r) < 0.1:
            r_interp = "Negligible"
        elif abs(r) < 0.3:
            r_interp = "Small"
        elif abs(r) < 0.5:
            r_interp = "Medium"
        else:
            r_interp = "Large"
        
        return StatisticalTest(
            test_name="Mann-Whitney U",
            statistic=stat,
            p_value=p,
            effect_size=abs(r),
            effect_size_name="Rank-biserial r",
            effect_size_interpretation=r_interp,
            is_significant=p < 0.05,
        )
    except Exception as e:
        return StatisticalTest(test_name="Mann-Whitney U", notes=str(e))


def one_way_anova(groups: Dict[str, pd.Series]) -> StatisticalTest:
    """
    Perform one-way ANOVA.
    
    Parameters
    ----------
    groups : dict
        {group_name: data_series}
        
    Returns
    -------
    StatisticalTest
    """
    from scipy import stats
    
    clean_groups = {k: v.dropna() for k, v in groups.items()}
    clean_groups = {k: v for k, v in clean_groups.items() if len(v) > 0}
    
    if len(clean_groups) < 2:
        return StatisticalTest(test_name="One-way ANOVA", notes="Need at least 2 groups")
    
    try:
        group_data = list(clean_groups.values())
        stat, p = stats.f_oneway(*group_data)
        
        # Degrees of freedom
        k = len(clean_groups)
        n_total = sum(len(g) for g in group_data)
        df_between = k - 1
        df_within = n_total - k
        
        # Effect size
        eta_sq, eta_interp = eta_squared(stat, df_between, df_within)
        
        return StatisticalTest(
            test_name="One-way ANOVA",
            statistic=stat,
            df=df_between,
            df2=df_within,
            p_value=p,
            effect_size=eta_sq,
            effect_size_name="η² (Eta-squared)",
            effect_size_interpretation=eta_interp,
            is_significant=p < 0.05,
        )
    except Exception as e:
        return StatisticalTest(test_name="One-way ANOVA", notes=str(e))


def welch_anova(groups: Dict[str, pd.Series]) -> StatisticalTest:
    """
    Perform Welch's ANOVA (robust to unequal variances).
    
    Parameters
    ----------
    groups : dict
        {group_name: data_series}
        
    Returns
    -------
    StatisticalTest
    """
    try:
        from scipy import stats
        
        clean_groups = {k: v.dropna() for k, v in groups.items()}
        clean_groups = {k: v for k, v in clean_groups.items() if len(v) > 0}
        
        if len(clean_groups) < 2:
            return StatisticalTest(test_name="Welch's ANOVA", notes="Need at least 2 groups")
        
        # Try using scipy.stats.alexandergovern (Welch's ANOVA equivalent)
        group_data = list(clean_groups.values())
        
        try:
            result = stats.alexandergovern(*group_data)
            stat, p = result.statistic, result.pvalue
        except:
            # Fallback to regular ANOVA
            stat, p = stats.f_oneway(*group_data)
        
        k = len(clean_groups)
        n_total = sum(len(g) for g in group_data)
        df_between = k - 1
        df_within = n_total - k
        
        eta_sq, eta_interp = eta_squared(stat, df_between, df_within)
        
        return StatisticalTest(
            test_name="Welch's ANOVA",
            statistic=stat,
            df=df_between,
            df2=df_within,
            p_value=p,
            effect_size=eta_sq,
            effect_size_name="η² (Eta-squared)",
            effect_size_interpretation=eta_interp,
            is_significant=p < 0.05,
        )
    except Exception as e:
        return StatisticalTest(test_name="Welch's ANOVA", notes=str(e))


def kruskal_wallis(groups: Dict[str, pd.Series]) -> StatisticalTest:
    """
    Perform Kruskal-Wallis H test (non-parametric alternative to ANOVA).
    
    Parameters
    ----------
    groups : dict
        {group_name: data_series}
        
    Returns
    -------
    StatisticalTest
    """
    from scipy import stats
    
    clean_groups = {k: v.dropna() for k, v in groups.items()}
    clean_groups = {k: v for k, v in clean_groups.items() if len(v) > 0}
    
    if len(clean_groups) < 2:
        return StatisticalTest(test_name="Kruskal-Wallis H", notes="Need at least 2 groups")
    
    try:
        group_data = list(clean_groups.values())
        stat, p = stats.kruskal(*group_data)
        
        # Effect size: epsilon-squared
        n_total = sum(len(g) for g in group_data)
        k = len(clean_groups)
        epsilon_sq = (stat - k + 1) / (n_total - k) if n_total > k else 0
        epsilon_sq = max(0, epsilon_sq)
        
        if epsilon_sq < 0.01:
            eps_interp = "Negligible"
        elif epsilon_sq < 0.06:
            eps_interp = "Small"
        elif epsilon_sq < 0.14:
            eps_interp = "Medium"
        else:
            eps_interp = "Large"
        
        return StatisticalTest(
            test_name="Kruskal-Wallis H",
            statistic=stat,
            df=k - 1,
            p_value=p,
            effect_size=epsilon_sq,
            effect_size_name="ε² (Epsilon-squared)",
            effect_size_interpretation=eps_interp,
            is_significant=p < 0.05,
        )
    except Exception as e:
        return StatisticalTest(test_name="Kruskal-Wallis H", notes=str(e))


def tukey_hsd(groups: Dict[str, pd.Series]) -> List[PostHocResult]:
    """
    Perform Tukey's HSD post-hoc test.
    
    Parameters
    ----------
    groups : dict
        {group_name: data_series}
        
    Returns
    -------
    list of PostHocResult
    """
    try:
        from scipy import stats
        
        clean_groups = {k: v.dropna() for k, v in groups.items()}
        clean_groups = {k: v for k, v in clean_groups.items() if len(v) > 0}
        
        if len(clean_groups) < 2:
            return []
        
        group_names = list(clean_groups.keys())
        group_data = list(clean_groups.values())
        
        # Try statsmodels for Tukey HSD
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            
            # Prepare data for statsmodels
            all_data = []
            all_groups = []
            for name, data in clean_groups.items():
                all_data.extend(data.tolist())
                all_groups.extend([name] * len(data))
            
            result = pairwise_tukeyhsd(all_data, all_groups, alpha=0.05)
            
            post_hoc = []
            for i in range(len(result.summary().data) - 1):
                row = result.summary().data[i + 1]
                post_hoc.append(PostHocResult(
                    group1=str(row[0]),
                    group2=str(row[1]),
                    mean_diff=float(row[2]),
                    p_adjusted=float(row[3]),
                    ci_lower=float(row[4]),
                    ci_upper=float(row[5]),
                    is_significant=bool(row[6]),
                ))
            
            return post_hoc
            
        except ImportError:
            # Fallback: pairwise t-tests with Bonferroni correction
            return bonferroni_pairwise(groups)
            
    except Exception as e:
        print(f"Post-hoc error: {e}")
        return []


def bonferroni_pairwise(groups: Dict[str, pd.Series]) -> List[PostHocResult]:
    """
    Perform pairwise t-tests with Bonferroni correction.
    
    Parameters
    ----------
    groups : dict
        {group_name: data_series}
        
    Returns
    -------
    list of PostHocResult
    """
    from scipy import stats
    from itertools import combinations
    
    clean_groups = {k: v.dropna() for k, v in groups.items()}
    clean_groups = {k: v for k, v in clean_groups.items() if len(v) > 0}
    
    group_names = list(clean_groups.keys())
    n_comparisons = len(list(combinations(group_names, 2)))
    
    if n_comparisons == 0:
        return []
    
    results = []
    for g1, g2 in combinations(group_names, 2):
        data1 = clean_groups[g1]
        data2 = clean_groups[g2]
        
        mean_diff = data1.mean() - data2.mean()
        
        try:
            stat, p = stats.ttest_ind(data1, data2)
            p_adj = min(p * n_comparisons, 1.0)  # Bonferroni
            
            # CI for mean difference
            se_diff = np.sqrt(data1.var()/len(data1) + data2.var()/len(data2))
            df = len(data1) + len(data2) - 2
            t_crit = stats.t.ppf(0.975, df)
            ci_lower = mean_diff - t_crit * se_diff
            ci_upper = mean_diff + t_crit * se_diff
            
            results.append(PostHocResult(
                group1=g1,
                group2=g2,
                mean_diff=mean_diff,
                statistic=stat,
                p_value=p,
                p_adjusted=p_adj,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                is_significant=p_adj < 0.05,
            ))
        except:
            pass
    
    return results


def dunn_test(groups: Dict[str, pd.Series]) -> List[PostHocResult]:
    """
    Perform Dunn's test (non-parametric post-hoc).
    
    Parameters
    ----------
    groups : dict
        {group_name: data_series}
        
    Returns
    -------
    list of PostHocResult
    """
    from scipy import stats
    from itertools import combinations
    
    clean_groups = {k: v.dropna() for k, v in groups.items()}
    clean_groups = {k: v for k, v in clean_groups.items() if len(v) > 0}
    
    group_names = list(clean_groups.keys())
    n_comparisons = len(list(combinations(group_names, 2)))
    
    if n_comparisons == 0:
        return []
    
    results = []
    for g1, g2 in combinations(group_names, 2):
        data1 = clean_groups[g1]
        data2 = clean_groups[g2]
        
        try:
            stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            p_adj = min(p * n_comparisons, 1.0)  # Bonferroni
            
            results.append(PostHocResult(
                group1=g1,
                group2=g2,
                mean_diff=data1.median() - data2.median(),  # Use median diff
                statistic=stat,
                p_value=p,
                p_adjusted=p_adj,
                is_significant=p_adj < 0.05,
            ))
        except:
            pass
    
    return results


def compare_means(
    df: pd.DataFrame,
    dependent_var: str,
    grouping_var: str,
    alpha: float = 0.05,
    verbose: bool = True,
) -> CompareMeansResult:
    """
    Comprehensive comparison of means across groups.
    
    Automatically selects appropriate tests based on:
    - Number of groups (2: t-test, 3+: ANOVA)
    - Normality of data
    - Homogeneity of variance
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    dependent_var : str
        Name of the dependent variable (numeric)
    grouping_var : str
        Name of the grouping variable (categorical)
    alpha : float
        Significance level (default 0.05)
    verbose : bool
        Print progress
        
    Returns
    -------
    CompareMeansResult
    """
    result = CompareMeansResult(
        dependent_var=dependent_var,
        grouping_var=grouping_var,
    )
    
    # Get data
    if dependent_var not in df.columns:
        raise ValueError(f"Dependent variable '{dependent_var}' not found in dataset")
    if grouping_var not in df.columns:
        raise ValueError(f"Grouping variable '{grouping_var}' not found in dataset")
    
    # Clean data
    clean_df = df[[dependent_var, grouping_var]].dropna()
    
    # Convert dependent to numeric
    try:
        clean_df[dependent_var] = pd.to_numeric(clean_df[dependent_var], errors='coerce')
        clean_df = clean_df.dropna()
    except:
        raise ValueError(f"Cannot convert '{dependent_var}' to numeric")
    
    result.n_total = len(clean_df)
    
    if result.n_total < 3:
        raise ValueError("Insufficient data for analysis")
    
    # Get groups
    groups = {}
    for group_name, group_data in clean_df.groupby(grouping_var):
        groups[str(group_name)] = group_data[dependent_var]
    
    result.n_groups = len(groups)
    
    if verbose:
        print(f"Compare Means Analysis")
        print(f"  Dependent variable: {dependent_var}")
        print(f"  Grouping variable: {grouping_var}")
        print(f"  Total N: {result.n_total}")
        print(f"  Number of groups: {result.n_groups}")
    
    if result.n_groups < 2:
        raise ValueError("Need at least 2 groups for comparison")
    
    # Overall descriptives
    result.overall_descriptives = compute_descriptives(
        clean_df[dependent_var], "Overall"
    )
    
    # Group descriptives
    for group_name, group_data in groups.items():
        desc = compute_descriptives(group_data, group_name)
        result.group_descriptives.append(desc)
        
        if verbose:
            print(f"  {group_name}: n={desc.n}, M={desc.mean:.3f}, SD={desc.std:.3f}")
    
    # Normality tests
    all_normal = True
    for group_name, group_data in groups.items():
        norm_test = test_normality(group_data, group_name)
        result.normality_tests[group_name] = norm_test
        if not norm_test.is_normal:
            all_normal = False
    
    # Homogeneity test
    result.homogeneity_test = test_homogeneity(list(groups.values()))
    
    result.assumptions_met = all_normal and result.homogeneity_test.is_homogeneous
    
    if verbose:
        print(f"  Normality assumption: {'Met' if all_normal else 'Violated'}")
        print(f"  Homogeneity of variance: {'Met' if result.homogeneity_test.is_homogeneous else 'Violated'}")
    
    # Perform appropriate tests
    if result.n_groups == 2:
        # Two groups: t-test
        group_names = list(groups.keys())
        g1, g2 = groups[group_names[0]], groups[group_names[1]]
        
        # Parametric
        if result.homogeneity_test.is_homogeneous:
            result.parametric_test = independent_t_test(g1, g2, equal_var=True)
        else:
            result.parametric_test = independent_t_test(g1, g2, equal_var=False)
        
        # Non-parametric
        result.nonparametric_test = mann_whitney_u(g1, g2)
        
    else:
        # Three or more groups: ANOVA
        # Parametric
        if result.homogeneity_test.is_homogeneous:
            result.parametric_test = one_way_anova(groups)
        else:
            result.parametric_test = welch_anova(groups)
        
        # Non-parametric
        result.nonparametric_test = kruskal_wallis(groups)
        
        # Post-hoc tests if significant
        if result.parametric_test.is_significant:
            result.post_hoc_results = tukey_hsd(groups)
            result.post_hoc_method = "Tukey HSD"
        
        if not result.post_hoc_results and result.nonparametric_test.is_significant:
            result.post_hoc_results = dunn_test(groups)
            result.post_hoc_method = "Dunn's test (Bonferroni)"
    
    # Recommendations
    if result.assumptions_met:
        result.recommended_test = result.parametric_test.test_name
    else:
        result.recommended_test = result.nonparametric_test.test_name
    
    # Interpretation
    if result.assumptions_met:
        test = result.parametric_test
    else:
        test = result.nonparametric_test
    
    if test.is_significant:
        result.interpretation = (
            f"There is a statistically significant difference between groups "
            f"({test.test_name}: p = {test.p_value:.4f}). "
            f"Effect size: {test.effect_size_name} = {test.effect_size:.3f} ({test.effect_size_interpretation})."
        )
    else:
        result.interpretation = (
            f"No statistically significant difference was found between groups "
            f"({test.test_name}: p = {test.p_value:.4f})."
        )
    
    if verbose:
        print(f"\nResults:")
        print(f"  Parametric: {result.parametric_test.test_name}, p = {result.parametric_test.p_value:.4f}")
        print(f"  Non-parametric: {result.nonparametric_test.test_name}, p = {result.nonparametric_test.p_value:.4f}")
        print(f"  Recommended: {result.recommended_test}")
    
    return result


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns in dataframe."""
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def get_categorical_columns(df: pd.DataFrame, max_categories: int = 20) -> List[str]:
    """Get list of categorical columns suitable for grouping."""
    cat_cols = []
    for col in df.columns:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= max_categories:
            cat_cols.append(col)
    return cat_cols
