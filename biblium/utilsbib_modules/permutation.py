# -*- coding: utf-8 -*-
"""
biblium.utilsbib_modules.permutation
====================================

Permutation-based inference for bibliometric group analyses with overlapping
groups and/or multi-valued entities.

Motivation
----------
When groups are non-disjoint (a document can belong to several groups,
e.g. thematic clusters defined by keywords) or the associated entity is
multi-valued (authors, keywords, countries), the classical Pearson
chi-squared test of independence is biased. The independence-of-
observations assumption is violated, marginals are inflated, and asymptotic
p-values are anti-conservative (too small).

This module provides a permutation framework that:

1. Holds the document x entity indicator matrix fixed.
2. Permutes the document x group indicator matrix row-wise.
3. Recomputes the chosen test statistic on each permuted contingency table.
4. Reports p = (1 + #{T_b >= T_0}) / (1 + B).

Row-wise permutation of the group matrix preserves
(i) the marginal sizes of each group,
(ii) the marginal entity frequencies,
(iii) the overlap structure of groups across documents, and
(iv) the multi-valued structure of the entity --
breaking only the document-level association between group membership and
entity occurrence, which is exactly the null hypothesis of interest.

Public components
-----------------
PermutationResult
    Dataclass holding the observed statistic, the null distribution, the
    p-value(s), a Clopper-Pearson 95% CI for the p-value, and bookkeeping.

permutation_test
    Core engine. Takes group and entity indicator matrices and a
    statistic callable.

Test statistics:
    chi2_statistic, cramers_v_statistic, total_inertia_statistic,
    dimension_inertias_statistic, standardized_residuals_statistic.

adjust_p_values
    Multiple-testing correction (Benjamini-Hochberg, Holm, Bonferroni).

adaptive_n_permutations
    Pick B from a target time budget via a brief micro-benchmark.

is_partition
    Detect a disjoint group structure (and warn the caller that asymptotic
    chi-squared is sufficient in that case).

Author: Lan Umek
Version: 2.16.0
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np

try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


# =============================================================================
# RESULT CONTAINER
# =============================================================================


@dataclass
class PermutationResult:
    """
    Container for the result of a permutation test.

    Attributes
    ----------
    observed : float or np.ndarray
        The observed test statistic. Scalar for global tests
        (``chi2``, ``cramers_v``, ``total_inertia``), vector for
        ``dimension_inertias_statistic`` (one entry per CA dimension),
        and matrix for ``standardized_residuals_statistic``
        (groups x entities).
    null_distribution : np.ndarray or None
        Stack of statistics under the null. Shape is
        ``(B,) + observed.shape``. ``None`` if ``return_null=False``.
    p_value : float or np.ndarray
        Permutation p-value(s), same shape as ``observed``.
    p_value_ci : tuple or np.ndarray or None
        Clopper-Pearson 95% CI for the p-value, useful as a sanity check
        when B is small. ``None`` for vector/matrix statistics by default.
    n_permutations : int
        Number of permutations actually performed.
    statistic_name : str
        Human-readable label for the statistic (used in plots, reports).
    alternative : str
        ``"greater"`` (default) or ``"two-sided"``.
    seed : int or None
        Seed used for ``np.random.default_rng``.
    elapsed_seconds : float
        Wall-clock time spent in the permutation loop.
    """

    observed: Union[float, np.ndarray]
    null_distribution: Optional[np.ndarray]
    p_value: Union[float, np.ndarray]
    p_value_ci: Optional[Union[Tuple[float, float], np.ndarray]] = None
    n_permutations: int = 0
    statistic_name: str = ""
    alternative: str = "greater"
    seed: Optional[int] = None
    elapsed_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a short human-readable summary string."""
        if np.isscalar(self.observed):
            obs = float(self.observed)
            p = float(self.p_value)  # type: ignore[arg-type]
            line = (
                f"PermutationResult({self.statistic_name}): "
                f"observed = {obs:.4g}, p = {p:.4g} "
                f"(B = {self.n_permutations}, alt = {self.alternative})"
            )
            if isinstance(self.p_value_ci, tuple):
                lo, hi = self.p_value_ci
                line += f", 95% CI for p: [{lo:.4g}, {hi:.4g}]"
            return line
        else:
            obs_arr = np.asarray(self.observed)
            return (
                f"PermutationResult({self.statistic_name}): "
                f"observed shape {obs_arr.shape}, "
                f"B = {self.n_permutations}, alt = {self.alternative}"
            )


# =============================================================================
# DISJOINT-GROUP DETECTION
# =============================================================================


def is_partition(group_matrix: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Test whether a group indicator matrix encodes a disjoint partition.

    A partition has 0/1 entries and rows that sum to exactly 1, i.e.
    each document belongs to exactly one group.

    Parameters
    ----------
    group_matrix : np.ndarray, shape (n_docs, n_groups)
        Binary indicator matrix.
    atol : float, default 1e-8
        Absolute tolerance for floating-point row sums.

    Returns
    -------
    bool
        True if the matrix is a disjoint partition.
    """
    G = np.asarray(group_matrix)
    if G.ndim != 2:
        return False
    # 0/1 only
    is_binary = np.all((np.isclose(G, 0, atol=atol)) | (np.isclose(G, 1, atol=atol)))
    if not is_binary:
        return False
    row_sums = G.sum(axis=1)
    return bool(np.all(np.isclose(row_sums, 1, atol=atol)))


def warn_if_partition(group_matrix: np.ndarray) -> bool:
    """
    Warn the caller if groups are disjoint, and return that flag.

    When groups are disjoint, the asymptotic chi-squared test of
    independence is unbiased and permutation inference is computationally
    wasteful (though not wrong).

    Returns
    -------
    bool
        True if groups are disjoint.
    """
    if is_partition(group_matrix):
        warnings.warn(
            "Groups appear to be disjoint (each document belongs to exactly "
            "one group). The asymptotic chi-squared test is unbiased in this "
            "case; a permutation test is unnecessary unless you want exact "
            "small-sample inference.",
            UserWarning,
            stacklevel=2,
        )
        return True
    return False


# =============================================================================
# ADAPTIVE NUMBER OF PERMUTATIONS
# =============================================================================


def adaptive_n_permutations(
    group_matrix: np.ndarray,
    entity_matrix: np.ndarray,
    statistic: Callable[[np.ndarray], Any],
    target_seconds: float = 5.0,
    n_min: int = 999,
    n_max: int = 99_999,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Pick a number of permutations matching a target wall-clock budget.

    Runs a brief micro-benchmark (``n_probe`` permutations) to estimate the
    per-permutation cost, then chooses B so that B * cost ~= target_seconds,
    clipped to ``[n_min, n_max]``.

    Parameters
    ----------
    group_matrix, entity_matrix : np.ndarray
        Same matrices that will be used in the actual test.
    statistic : callable
        Same statistic that will be used in the actual test.
    target_seconds : float, default 5.0
        Wall-clock budget for the permutation loop.
    n_min : int, default 999
        Minimum B (gives p-value resolution down to ~1e-3).
    n_max : int, default 99_999
        Hard upper bound (gives p-value resolution down to ~1e-5).
    rng : np.random.Generator, optional
        Source of randomness; ``np.random.default_rng()`` if None.

    Returns
    -------
    int
        Recommended number of permutations.
    """
    if rng is None:
        rng = np.random.default_rng()

    G = np.asarray(group_matrix)
    E = np.asarray(entity_matrix)
    n_docs = G.shape[0]
    n_probe = 5

    t0 = time.perf_counter()
    for _ in range(n_probe):
        perm = rng.permutation(n_docs)
        T_b = G[perm].T @ E
        _ = statistic(T_b)
    elapsed = time.perf_counter() - t0
    if elapsed <= 0:
        return n_max

    per_perm = elapsed / n_probe
    B = int(target_seconds / per_perm)
    B = max(n_min, min(n_max, B))
    # Use 2 * 10**k - 1 conventions are nice but not essential; just round to a
    # value of the form 9, 99, 999, 9999, 99999 below the budget.
    nice_levels = [999, 1999, 4999, 9999, 19_999, 49_999, 99_999]
    chosen = nice_levels[0]
    for level in nice_levels:
        if level <= B:
            chosen = level
        else:
            break
    return max(n_min, min(n_max, chosen))


# =============================================================================
# TEST STATISTICS
# =============================================================================


def _expected_under_independence(table: np.ndarray) -> np.ndarray:
    """Compute the expected-counts table under row-column independence."""
    table = np.asarray(table, dtype=float)
    total = table.sum()
    if total <= 0:
        return np.zeros_like(table)
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    return (row @ col) / total


def chi2_statistic(table: np.ndarray) -> float:
    """
    Pearson chi-squared statistic for an n_groups x n_entities table.

    Returns 0 for empty / degenerate tables. Cells with expected count 0
    are skipped (their contribution is undefined and would be NaN).
    """
    O = np.asarray(table, dtype=float)
    E = _expected_under_independence(O)
    mask = E > 0
    if not np.any(mask):
        return 0.0
    diff = O - E
    return float(np.sum(diff[mask] ** 2 / E[mask]))


def cramers_v_statistic(table: np.ndarray) -> float:
    """
    Cramer's V effect-size statistic.

    V = sqrt(chi^2 / (n * (min(r, c) - 1))). Bounded in [0, 1] and
    asymptotically invariant to sample size, which makes it a more
    interpretable test statistic across studies than raw chi^2.
    """
    O = np.asarray(table, dtype=float)
    n = O.sum()
    r, c = O.shape
    k = min(r, c) - 1
    if n <= 0 or k <= 0:
        return 0.0
    chi2 = chi2_statistic(O)
    return float(np.sqrt(chi2 / (n * k)))


def total_inertia_statistic(table: np.ndarray) -> float:
    """
    Total inertia of a contingency table for correspondence analysis.

    Equal to chi^2 / n. Total inertia is the sum of the squared singular
    values of the standardised residual matrix, hence a natural global
    statistic for CA.
    """
    O = np.asarray(table, dtype=float)
    n = O.sum()
    if n <= 0:
        return 0.0
    return chi2_statistic(O) / n


def _standardized_residual_matrix(table: np.ndarray) -> np.ndarray:
    """
    Return the matrix of standardised Pearson residuals.

    Z_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - p_i.) * (1 - p_.j))
    where p_i. = row_i / n and p_.j = col_j / n. Standardised residuals
    are approximately N(0, 1) under independence in the disjoint case;
    here they are used purely as a centred and scaled cell-level
    statistic for permutation inference.
    """
    O = np.asarray(table, dtype=float)
    n = O.sum()
    if n <= 0:
        return np.zeros_like(O)
    row = O.sum(axis=1, keepdims=True)
    col = O.sum(axis=0, keepdims=True)
    E = (row @ col) / n
    p_row = row / n
    p_col = col / n
    var = E * (1 - p_row) * (1 - p_col)
    out = np.zeros_like(O)
    mask = var > 0
    out[mask] = (O[mask] - E[mask]) / np.sqrt(var[mask])
    return out


def standardized_residuals_statistic(table: np.ndarray) -> np.ndarray:
    """Return cell-level standardised residuals as a matrix statistic."""
    return _standardized_residual_matrix(table)


def dimension_inertias_statistic(
    table: np.ndarray,
    n_dimensions: Optional[int] = None,
) -> np.ndarray:
    """
    Per-dimension inertias from correspondence analysis (squared SVs).

    Lets the caller test how many CA dimensions are individually
    significant: dimension k is "real" if its inertia exceeds what is
    typically seen under permutation.

    Parameters
    ----------
    table : np.ndarray
        Contingency table (groups x entities).
    n_dimensions : int, optional
        Number of leading dimensions to keep. If None, returns
        min(r-1, c-1) values (the maximal CA rank).

    Returns
    -------
    np.ndarray
        Vector of length ``n_dimensions`` with eigenvalues (squared
        singular values of the standardised residual matrix divided by n).
    """
    O = np.asarray(table, dtype=float)
    n = O.sum()
    r, c = O.shape
    max_dim = max(min(r, c) - 1, 0)
    if n_dimensions is None:
        n_dimensions = max_dim
    n_dimensions = int(min(n_dimensions, max_dim))
    if n <= 0 or n_dimensions <= 0:
        return np.zeros(max(n_dimensions, 0))

    row = O.sum(axis=1, keepdims=True)
    col = O.sum(axis=0, keepdims=True)
    E = (row @ col) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(E > 0, (O - E) / np.sqrt(E * n), 0.0)
    # Singular values of S; eigenvalues of CA = sigma^2.
    try:
        sv = np.linalg.svd(S, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.zeros(n_dimensions)
    eig = sv ** 2
    out = np.zeros(n_dimensions)
    out[: min(n_dimensions, len(eig))] = eig[:n_dimensions]
    return out


# =============================================================================
# MULTIPLE-TESTING CORRECTION
# =============================================================================


def adjust_p_values(
    p_values: np.ndarray,
    method: Literal["bh", "holm", "bonferroni", "none"] = "bh",
) -> np.ndarray:
    """
    Adjust a vector or matrix of p-values for multiple testing.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values.
    method : {"bh", "holm", "bonferroni", "none"}, default "bh"
        - "bh" : Benjamini-Hochberg FDR control.
        - "holm" : Holm step-down FWER control.
        - "bonferroni" : Bonferroni FWER control (most conservative).
        - "none" : pass through.

    Returns
    -------
    np.ndarray
        Adjusted p-values, same shape as ``p_values``, clipped to [0, 1].
    """
    p = np.asarray(p_values, dtype=float)
    shape = p.shape
    flat = p.flatten()
    m = flat.size
    if method == "none" or m == 0:
        return p.copy()

    if method == "bonferroni":
        return np.clip(flat * m, 0.0, 1.0).reshape(shape)

    order = np.argsort(flat, kind="mergesort")
    ordered = flat[order]

    if method == "holm":
        adjusted = np.empty(m)
        running_max = 0.0
        for i in range(m):
            val = (m - i) * ordered[i]
            running_max = max(running_max, val)
            adjusted[i] = min(running_max, 1.0)
        out = np.empty(m)
        out[order] = adjusted
        return out.reshape(shape)

    if method == "bh":
        adjusted = np.empty(m)
        running_min = 1.0
        for i in reversed(range(m)):
            val = ordered[i] * m / (i + 1)
            running_min = min(running_min, val)
            adjusted[i] = running_min
        adjusted = np.clip(adjusted, 0.0, 1.0)
        out = np.empty(m)
        out[order] = adjusted
        return out.reshape(shape)

    raise ValueError(f"Unknown adjustment method: {method!r}")


# =============================================================================
# CORE ENGINE
# =============================================================================


def _clopper_pearson_ci(
    successes: int, n: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Exact 95% CI for a binomial proportion (Clopper-Pearson)."""
    try:
        from scipy.stats import beta as _beta  # type: ignore
    except ImportError:
        # Wilson-style fallback
        if n == 0:
            return (0.0, 1.0)
        p_hat = successes / n
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        return (max(0.0, p_hat - 1.96 * se), min(1.0, p_hat + 1.96 * se))
    if n == 0:
        return (0.0, 1.0)
    if successes == 0:
        lo = 0.0
    else:
        lo = float(_beta.ppf(alpha / 2, successes, n - successes + 1))
    if successes == n:
        hi = 1.0
    else:
        hi = float(_beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return (lo, hi)


def permutation_test(
    *,
    group_matrix: np.ndarray,
    entity_matrix: np.ndarray,
    statistic: Callable[[np.ndarray], Any] = chi2_statistic,
    n_permutations: Optional[int] = None,
    target_seconds: float = 5.0,
    alternative: Literal["greater", "two-sided"] = "greater",
    random_state: Optional[Union[int, np.random.Generator]] = None,
    statistic_name: str = "",
    return_null: bool = True,
    show_progress: bool = False,
    warn_disjoint: bool = True,
) -> PermutationResult:
    """
    Permutation test of association between groups and entities.

    The test holds the document x entity indicator matrix fixed and
    permutes the rows of the document x group indicator matrix B times.
    For each permutation it forms the contingency table
    ``T_b = G_perm.T @ E`` and applies ``statistic``.

    Parameters
    ----------
    group_matrix : np.ndarray, shape (n_docs, n_groups)
        Binary or 0/1 group membership indicators (overlap allowed).
    entity_matrix : np.ndarray, shape (n_docs, n_entities)
        Document-level indicators (binary or counts) for the entity
        being associated (e.g. authors, keywords, countries).
    statistic : callable, default chi2_statistic
        A function that maps a contingency table (n_groups x n_entities)
        to a scalar, vector, or matrix statistic.
    n_permutations : int, optional
        Number of permutations. If None, ``adaptive_n_permutations``
        chooses a value matching ``target_seconds``.
    target_seconds : float, default 5.0
        Time budget when ``n_permutations`` is None.
    alternative : {"greater", "two-sided"}, default "greater"
        - "greater" : right-tail (typical for chi^2, V, inertia).
        - "two-sided" : symmetric around the null mean (used for
          standardised residuals where sign matters).
    random_state : int or np.random.Generator, optional
        Seed or generator for reproducibility.
    statistic_name : str
        Label stored in the result for downstream reporting.
    return_null : bool, default True
        Keep the full null distribution in the result. Set to False to
        reduce memory when the statistic is matrix-valued and B is large.
    show_progress : bool, default False
        Show a tqdm progress bar over permutations.
    warn_disjoint : bool, default True
        Warn (once) if ``group_matrix`` is a disjoint partition.

    Returns
    -------
    PermutationResult

    Notes
    -----
    The reported p-value uses the conservative formula
    ``p = (1 + #{T_b >= T_0}) / (1 + B)`` (Phipson & Smyth 2010), which
    avoids the spurious p = 0 that the naive formula
    ``#{T_b >= T_0} / B`` can produce.
    """
    G = np.asarray(group_matrix, dtype=float)
    E = np.asarray(entity_matrix, dtype=float)
    if G.ndim != 2 or E.ndim != 2:
        raise ValueError("group_matrix and entity_matrix must both be 2-D")
    if G.shape[0] != E.shape[0]:
        raise ValueError(
            f"group_matrix and entity_matrix must have the same number of "
            f"rows; got {G.shape[0]} and {E.shape[0]}."
        )
    n_docs = G.shape[0]
    if n_docs == 0:
        raise ValueError("Empty input matrices.")

    if warn_disjoint:
        warn_if_partition(G)

    if isinstance(random_state, np.random.Generator):
        rng = random_state
        seed_repr = None
    else:
        rng = np.random.default_rng(random_state)
        seed_repr = random_state

    # Observed statistic
    T0 = G.T @ E
    observed = statistic(T0)
    observed_arr = np.asarray(observed)
    is_scalar = observed_arr.ndim == 0

    # Decide B
    if n_permutations is None:
        B = adaptive_n_permutations(
            G, E, statistic, target_seconds=target_seconds, rng=rng
        )
    else:
        B = int(n_permutations)
    if B < 1:
        raise ValueError("n_permutations must be >= 1.")

    # Allocate counters / null storage
    if is_scalar:
        if return_null:
            null = np.empty(B, dtype=float)
        else:
            null = None
        if alternative == "greater":
            count_extreme = 0
        else:
            count_extreme = 0  # populated after the loop using stored null or running stats
    else:
        if return_null:
            null = np.empty((B,) + observed_arr.shape, dtype=float)
        else:
            null = None
        count_extreme = np.zeros(observed_arr.shape, dtype=np.int64)

    # Main loop
    iterator = range(B)
    if show_progress and _TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="permutations", total=B)

    t_start = time.perf_counter()
    for b in iterator:
        perm = rng.permutation(n_docs)
        T_b = G[perm].T @ E
        s_b = statistic(T_b)

        if is_scalar:
            s_b_val = float(s_b)
            if return_null:
                null[b] = s_b_val  # type: ignore[index]
            if alternative == "greater":
                if s_b_val >= float(observed):  # type: ignore[arg-type]
                    count_extreme += 1
            # for two-sided we'll recompute after the loop
        else:
            s_b_arr = np.asarray(s_b)
            if return_null:
                null[b] = s_b_arr  # type: ignore[index]
            if alternative == "greater":
                count_extreme += (s_b_arr >= observed_arr).astype(np.int64)
            else:  # two-sided handled below using |x|
                count_extreme += (np.abs(s_b_arr) >= np.abs(observed_arr)).astype(
                    np.int64
                )

    elapsed = time.perf_counter() - t_start

    # Finalize p-values
    if is_scalar:
        if alternative == "two-sided":
            if return_null and null is not None:
                # symmetrize around null mean for robustness
                center = float(np.mean(null))
                obs_centered = abs(float(observed) - center)
                count_extreme = int(np.sum(np.abs(null - center) >= obs_centered))
            else:
                # Without null storage, fall back to twice the one-sided
                p_one = (1 + count_extreme) / (1 + B)
                p_value = min(1.0, 2 * p_one)
                ci = _clopper_pearson_ci(count_extreme, B)
                return PermutationResult(
                    observed=float(observed),
                    null_distribution=None,
                    p_value=p_value,
                    p_value_ci=ci,
                    n_permutations=B,
                    statistic_name=statistic_name,
                    alternative=alternative,
                    seed=seed_repr,
                    elapsed_seconds=elapsed,
                )
        p_value = (1 + count_extreme) / (1 + B)
        ci = _clopper_pearson_ci(count_extreme, B)
        return PermutationResult(
            observed=float(observed),
            null_distribution=null,
            p_value=float(p_value),
            p_value_ci=ci,
            n_permutations=B,
            statistic_name=statistic_name,
            alternative=alternative,
            seed=seed_repr,
            elapsed_seconds=elapsed,
        )
    else:
        p_value = (1 + count_extreme) / (1 + B)
        return PermutationResult(
            observed=observed_arr,
            null_distribution=null,
            p_value=p_value,
            p_value_ci=None,  # cell-wise CI is overkill; users do BH on p_value
            n_permutations=B,
            statistic_name=statistic_name,
            alternative=alternative,
            seed=seed_repr,
            elapsed_seconds=elapsed,
        )


# =============================================================================
# CONVENIENCE WRAPPER (DataFrame-friendly)
# =============================================================================


def assoc_permutation_test(
    group_matrix,
    entity_matrix,
    *,
    test: Literal[
        "chi2", "cramers_v", "total_inertia", "dimension_inertias", "residuals"
    ] = "chi2",
    n_permutations: Optional[int] = None,
    target_seconds: float = 5.0,
    n_dimensions: Optional[int] = None,
    multiple_testing: Literal["bh", "holm", "bonferroni", "none"] = "bh",
    random_state: Optional[Union[int, np.random.Generator]] = None,
    show_progress: bool = False,
    warn_disjoint: bool = True,
) -> PermutationResult:
    """
    DataFrame-friendly wrapper around :func:`permutation_test`.

    Accepts pandas DataFrames or NumPy arrays for both matrices and
    dispatches to one of the built-in test statistics.

    Parameters
    ----------
    group_matrix : pd.DataFrame or np.ndarray, shape (n_docs, n_groups)
    entity_matrix : pd.DataFrame or np.ndarray, shape (n_docs, n_entities)
    test : str
        Which built-in test to run. ``"residuals"`` returns a matrix of
        cell-level p-values with ``multiple_testing`` applied; the other
        options return scalar or vector statistics.
    multiple_testing : {"bh", "holm", "bonferroni", "none"}, default "bh"
        Used only for ``test="residuals"``.

    Returns
    -------
    PermutationResult
        For ``test="residuals"``, the ``extra`` dict holds an additional
        key ``"p_value_adjusted"`` with the BH/Holm/Bonferroni-adjusted
        cell-level p-values.
    """
    G = np.asarray(group_matrix.values if hasattr(group_matrix, "values") else group_matrix)
    E = np.asarray(
        entity_matrix.values if hasattr(entity_matrix, "values") else entity_matrix
    )

    if test == "chi2":
        result = permutation_test(
            group_matrix=G,
            entity_matrix=E,
            statistic=chi2_statistic,
            n_permutations=n_permutations,
            target_seconds=target_seconds,
            alternative="greater",
            random_state=random_state,
            statistic_name="chi2",
            show_progress=show_progress,
            warn_disjoint=warn_disjoint,
        )
        return result

    if test == "cramers_v":
        return permutation_test(
            group_matrix=G,
            entity_matrix=E,
            statistic=cramers_v_statistic,
            n_permutations=n_permutations,
            target_seconds=target_seconds,
            alternative="greater",
            random_state=random_state,
            statistic_name="cramers_v",
            show_progress=show_progress,
            warn_disjoint=warn_disjoint,
        )

    if test == "total_inertia":
        return permutation_test(
            group_matrix=G,
            entity_matrix=E,
            statistic=total_inertia_statistic,
            n_permutations=n_permutations,
            target_seconds=target_seconds,
            alternative="greater",
            random_state=random_state,
            statistic_name="total_inertia",
            show_progress=show_progress,
            warn_disjoint=warn_disjoint,
        )

    if test == "dimension_inertias":
        def _stat(table: np.ndarray) -> np.ndarray:
            return dimension_inertias_statistic(table, n_dimensions=n_dimensions)

        return permutation_test(
            group_matrix=G,
            entity_matrix=E,
            statistic=_stat,
            n_permutations=n_permutations,
            target_seconds=target_seconds,
            alternative="greater",
            random_state=random_state,
            statistic_name="dimension_inertias",
            show_progress=show_progress,
            warn_disjoint=warn_disjoint,
        )

    if test == "residuals":
        result = permutation_test(
            group_matrix=G,
            entity_matrix=E,
            statistic=standardized_residuals_statistic,
            n_permutations=n_permutations,
            target_seconds=target_seconds,
            alternative="two-sided",
            random_state=random_state,
            statistic_name="standardized_residuals",
            show_progress=show_progress,
            warn_disjoint=warn_disjoint,
        )
        # Apply multiple-testing correction across the cell-level p-values
        adj = adjust_p_values(np.asarray(result.p_value), method=multiple_testing)
        result.extra["p_value_adjusted"] = adj
        result.extra["multiple_testing"] = multiple_testing
        return result

    raise ValueError(f"Unknown test {test!r}")


__all__ = [
    "PermutationResult",
    "is_partition",
    "warn_if_partition",
    "adaptive_n_permutations",
    "chi2_statistic",
    "cramers_v_statistic",
    "total_inertia_statistic",
    "standardized_residuals_statistic",
    "dimension_inertias_statistic",
    "adjust_p_values",
    "permutation_test",
    "assoc_permutation_test",
]
