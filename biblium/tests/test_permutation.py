# -*- coding: utf-8 -*-
"""
Tests for ``biblium.utilsbib_modules.permutation``.

These cover the public API:
- core ``permutation_test`` (with explicit ``n_permutations``)
- assoc-style wrapper ``assoc_permutation_test`` over each ``test=`` option
- ``adjust_p_values`` matches scipy's BH on test inputs
- ``is_partition`` / ``warn_if_partition`` boundary cases

All tests use small ``n_permutations`` (199 / 499) to keep them under a second.
The "greater"-tail Phipson–Smyth formula  p = (1 + #{T_b ≥ T_0}) / (1 + B)
is checked numerically against the public ``p_value``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from biblium.utilsbib_modules.permutation import (
    PermutationResult,
    adjust_p_values,
    assoc_permutation_test,
    chi2_statistic,
    cramers_v_statistic,
    is_partition,
    permutation_test,
    standardized_residuals_statistic,
    total_inertia_statistic,
    warn_if_partition,
)


# ---------------------------------------------------------------------------
# Test statistics on tiny known-input matrices
# ---------------------------------------------------------------------------


class TestStatisticsAreCallable:
    """All bundled statistics return finite scalars (or arrays) on a toy matrix."""

    @pytest.fixture()
    def contingency(self):
        # 3 groups x 4 entities, dense enough that chi-square is well-defined
        return np.array([
            [10,  5,  2,  3],
            [ 4, 12,  6,  1],
            [ 1,  3, 15,  8],
        ], dtype=float)

    def test_chi2_returns_scalar(self, contingency):
        v = chi2_statistic(contingency)
        assert np.isscalar(v) or (hasattr(v, "shape") and v.shape == ())
        assert np.isfinite(v)
        assert v > 0

    def test_cramers_v_in_unit_interval(self, contingency):
        v = cramers_v_statistic(contingency)
        assert 0.0 <= float(v) <= 1.0

    def test_total_inertia_nonnegative(self, contingency):
        v = total_inertia_statistic(contingency)
        assert float(v) >= 0.0

    def test_residuals_returns_matrix(self, contingency):
        r = standardized_residuals_statistic(contingency)
        assert hasattr(r, "shape")
        assert r.shape == contingency.shape


class TestChi2AgainstScipy:
    """``chi2_statistic`` must equal scipy.stats.chi2_contingency on the same input."""

    def test_matches_scipy_chi2_contingency(self):
        rng = np.random.default_rng(42)
        for _ in range(5):
            M = rng.integers(0, 25, size=(3, 4))
            ours = float(chi2_statistic(M.astype(float)))
            theirs, _, _, _ = sps.chi2_contingency(M, correction=False)
            assert ours == pytest.approx(theirs, rel=1e-10)


# ---------------------------------------------------------------------------
# Core permutation engine
# ---------------------------------------------------------------------------


class TestPermutationTestCore:
    """Direct calls to ``permutation_test`` with explicit budget."""

    def test_returns_PermutationResult(self, tiny_doc_term_matrix, disjoint_groups):
        res = permutation_test(
            group_matrix=disjoint_groups,
            entity_matrix=tiny_doc_term_matrix,
            statistic=chi2_statistic,
            n_permutations=199,
            random_state=0,
            warn_disjoint=False,
        )
        assert isinstance(res, PermutationResult)

    def test_p_value_in_unit_interval(self, tiny_doc_term_matrix, disjoint_groups):
        res = permutation_test(
            group_matrix=disjoint_groups,
            entity_matrix=tiny_doc_term_matrix,
            statistic=chi2_statistic,
            n_permutations=199,
            random_state=0,
            warn_disjoint=False,
        )
        # For scalar-statistic tests the p_value is a scalar in [0, 1].
        p = float(np.asarray(res.p_value).ravel()[0])
        assert 0.0 < p <= 1.0  # Phipson–Smyth never produces 0

    def test_phipson_smyth_formula(self, tiny_doc_term_matrix, disjoint_groups):
        """``p = (1 + #{T_b >= T_0}) / (1 + B)``."""
        B = 199
        res = permutation_test(
            group_matrix=disjoint_groups,
            entity_matrix=tiny_doc_term_matrix,
            statistic=chi2_statistic,
            n_permutations=B,
            random_state=0,
            return_null=True,
            warn_disjoint=False,
        )
        T0 = float(np.asarray(res.observed).ravel()[0])
        null = np.asarray(res.null_distribution).ravel()
        n_ge = int(np.sum(null >= T0))
        expected = (1 + n_ge) / (1 + B)
        got = float(np.asarray(res.p_value).ravel()[0])
        assert got == pytest.approx(expected, rel=1e-12)

    def test_reproducibility_with_seed(
        self, tiny_doc_term_matrix, disjoint_groups
    ):
        kw = dict(
            group_matrix=disjoint_groups,
            entity_matrix=tiny_doc_term_matrix,
            statistic=chi2_statistic,
            n_permutations=199,
            warn_disjoint=False,
        )
        a = permutation_test(random_state=2026, **kw)
        b = permutation_test(random_state=2026, **kw)
        assert float(np.asarray(a.p_value).ravel()[0]) == pytest.approx(
            float(np.asarray(b.p_value).ravel()[0])
        )

    def test_strong_signal_yields_small_p(
        self, tiny_doc_term_matrix, disjoint_groups
    ):
        # Term 0 is concentrated in the first 10 docs; with 4 groups of 5 docs,
        # the true association is very strong. p should be small.
        res = permutation_test(
            group_matrix=disjoint_groups,
            entity_matrix=tiny_doc_term_matrix,
            statistic=chi2_statistic,
            n_permutations=499,
            random_state=0,
            warn_disjoint=False,
        )
        p = float(np.asarray(res.p_value).ravel()[0])
        assert p < 0.05


# ---------------------------------------------------------------------------
# assoc_permutation_test wrapper
# ---------------------------------------------------------------------------


class TestAssocPermutationTest:
    """High-level ``assoc_permutation_test(test=...)`` for each statistic kind."""

    @pytest.mark.parametrize("test", [
        "chi2", "cramers_v", "total_inertia", "residuals",
    ])
    def test_each_test_kind_runs(
        self, test, tiny_doc_term_matrix, overlapping_groups
    ):
        res = assoc_permutation_test(
            overlapping_groups,
            tiny_doc_term_matrix,
            test=test,
            n_permutations=199,
            random_state=0,
            warn_disjoint=False,
        )
        assert isinstance(res, PermutationResult)
        # p-values are always in [0, 1]
        p = np.asarray(res.p_value)
        assert np.all((p >= 0) & (p <= 1))

    def test_residuals_returns_matrix_of_pvalues(
        self, tiny_doc_term_matrix, overlapping_groups
    ):
        res = assoc_permutation_test(
            overlapping_groups,
            tiny_doc_term_matrix,
            test="residuals",
            n_permutations=199,
            random_state=0,
            warn_disjoint=False,
        )
        p = np.asarray(res.p_value)
        # Residuals: one p-value per (group, entity) cell
        assert p.shape == (overlapping_groups.shape[1],
                           tiny_doc_term_matrix.shape[1])

    def test_disjoint_warning_is_emitted(
        self, tiny_doc_term_matrix, disjoint_groups
    ):
        # warn_disjoint=True (the default) must emit a UserWarning when groups
        # form a partition; the test still returns a result.
        with pytest.warns(UserWarning, match="disjoint|partition"):
            res = assoc_permutation_test(
                disjoint_groups,
                tiny_doc_term_matrix,
                test="chi2",
                n_permutations=99,
                random_state=0,
                warn_disjoint=True,
            )
        assert isinstance(res, PermutationResult)


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------


class TestAdjustPValues:
    """``adjust_p_values`` against scipy's reference implementation."""

    def test_bh_matches_scipy(self):
        rng = np.random.default_rng(0)
        p = rng.random(40)
        ours = adjust_p_values(p, method="bh")
        theirs = sps.false_discovery_control(p, method="bh")
        np.testing.assert_allclose(ours, theirs, atol=1e-15)

    def test_bonferroni(self):
        p = np.array([0.01, 0.04, 0.5])
        adj = adjust_p_values(p, method="bonferroni")
        np.testing.assert_allclose(adj, np.minimum(1.0, p * 3))

    def test_holm_is_monotone_and_at_least_p(self):
        rng = np.random.default_rng(1)
        p = np.sort(rng.random(20))
        adj = adjust_p_values(p, method="holm")
        # Adjusted p-values are always >= raw, and capped at 1
        assert np.all(adj >= p - 1e-15)
        assert np.all(adj <= 1.0 + 1e-15)

    def test_none_passes_through(self):
        p = np.array([0.1, 0.2, 0.05])
        np.testing.assert_array_equal(adjust_p_values(p, method="none"), p)

    def test_invalid_method_raises(self):
        with pytest.raises((ValueError, KeyError)):
            adjust_p_values(np.array([0.1]), method="not-a-real-method")


# ---------------------------------------------------------------------------
# Partition detection
# ---------------------------------------------------------------------------


class TestIsPartition:
    """Detect whether group_matrix encodes a disjoint partition."""

    def test_disjoint_groups_is_partition(self, disjoint_groups):
        assert is_partition(disjoint_groups) is True

    def test_overlapping_is_not_partition(self):
        # Construct groups where some rows belong to multiple groups
        g = np.array([
            [1, 1, 0],   # in groups 0 and 1
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 1],
        ])
        assert is_partition(g) is False

    def test_zero_row_is_not_partition(self):
        # Row sums must equal exactly 1; a row of all zeros breaks partition.
        g = np.array([
            [1, 0],
            [0, 0],
            [0, 1],
        ])
        assert is_partition(g) is False

    def test_warn_if_partition_disjoint(self, disjoint_groups):
        with pytest.warns(UserWarning):
            warn_if_partition(disjoint_groups)

    def test_warn_if_partition_overlapping_silent(self):
        g = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ])
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_partition(g)
        # No partition warnings expected on overlapping groups
        partition_msgs = [
            w for w in caught
            if "disjoint" in str(w.message).lower()
            or "partition" in str(w.message).lower()
        ]
        assert partition_msgs == []
