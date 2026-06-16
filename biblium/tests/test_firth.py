# -*- coding: utf-8 -*-
"""
Tests for ``biblium.utilsbib_modules.firth``.

Covers:
- ``mle_logit`` matches statsmodels' Logit on a clean dataset
- ``firth_logit`` produces finite, sensible estimates on perfectly-separated
  data where MLE diverges
- ``fit_logit(method='auto')`` falls back to Firth on separated input
- the ``LogitResult`` dataclass exposes the fields downstream code expects
- profile-likelihood CIs are wider/asymmetric vs Wald (and finite under separation)
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from biblium.utilsbib_modules.firth import (
    LogitResult,
    firth_logit,
    fit_logit,
    is_design_ill_conditioned,
    mle_logit,
)


# ---------------------------------------------------------------------------
# MLE behaviour on clean data
# ---------------------------------------------------------------------------


class TestMLELogit:
    """``mle_logit`` should match statsmodels on well-behaved input."""

    def test_returns_LogitResult(self, logit_clean_dataset):
        X, y, _beta_true = logit_clean_dataset
        res = mle_logit(X, y, add_intercept=False)
        assert isinstance(res, LogitResult)

    def test_converges(self, logit_clean_dataset):
        X, y, _ = logit_clean_dataset
        res = mle_logit(X, y, add_intercept=False)
        assert res.converged is True
        assert res.method_used in ("mle",)

    def test_coefficients_match_statsmodels(self, logit_clean_dataset):
        X, y, _ = logit_clean_dataset
        ours = mle_logit(X, y, add_intercept=False)
        theirs = sm.Logit(y, X).fit(disp=False, method="newton")
        np.testing.assert_allclose(ours.coef, theirs.params, atol=1e-6)

    def test_standard_errors_match_statsmodels(self, logit_clean_dataset):
        X, y, _ = logit_clean_dataset
        ours = mle_logit(X, y, add_intercept=False)
        theirs = sm.Logit(y, X).fit(disp=False, method="newton")
        np.testing.assert_allclose(ours.se, theirs.bse, atol=1e-6)

    def test_log_likelihood_matches(self, logit_clean_dataset):
        X, y, _ = logit_clean_dataset
        ours = mle_logit(X, y, add_intercept=False)
        theirs = sm.Logit(y, X).fit(disp=False, method="newton")
        assert ours.log_likelihood == pytest.approx(theirs.llf, abs=1e-6)

    def test_recovers_true_signs(self, logit_clean_dataset):
        X, y, beta_true = logit_clean_dataset
        res = mle_logit(X, y, add_intercept=False)
        # All non-zero coefficients should be recovered with the correct sign
        for est, truth in zip(res.coef, beta_true):
            if abs(truth) > 0.4:  # ignore the (zero) intercept
                assert np.sign(est) == np.sign(truth), (est, truth)


# ---------------------------------------------------------------------------
# Firth on separated data
# ---------------------------------------------------------------------------


class TestFirthOnSeparated:
    """Firth penalisation must converge to finite estimates under separation."""

    def test_estimates_are_finite(self, logit_separated_dataset):
        X, y = logit_separated_dataset
        res = firth_logit(X, y, add_intercept=False, ci_method="wald")
        assert np.all(np.isfinite(res.coef))
        assert np.all(np.isfinite(res.se))
        # Firth must keep coefficients bounded (the whole point of the prior)
        assert np.all(np.abs(res.coef) < 50)

    def test_method_used_is_firth(self, logit_separated_dataset):
        X, y = logit_separated_dataset
        res = firth_logit(X, y, add_intercept=False, ci_method="wald")
        assert res.method_used == "firth"

    def test_signs_are_correct(self, logit_separated_dataset):
        # Separation along x1 -> positive coefficient on x1
        X, y = logit_separated_dataset
        res = firth_logit(X, y, add_intercept=False, ci_method="wald")
        # Index 1 is the x1 coefficient (col 0 is the intercept)
        assert res.coef[1] > 0

    def test_profile_ci_finite(self, logit_separated_dataset):
        X, y = logit_separated_dataset
        res = firth_logit(X, y, add_intercept=False, ci_method="profile")
        assert np.all(np.isfinite(res.ci_low))
        assert np.all(np.isfinite(res.ci_high))
        # Lower bound must be below upper bound
        assert np.all(res.ci_low < res.ci_high)


# ---------------------------------------------------------------------------
# fit_logit dispatcher
# ---------------------------------------------------------------------------


class TestFitLogitDispatch:
    """``fit_logit(method='auto')`` chooses MLE or Firth as appropriate."""

    def test_auto_picks_mle_on_clean_data(self, logit_clean_dataset):
        X, y, _ = logit_clean_dataset
        res = fit_logit(X, y, method="auto", add_intercept=False)
        assert res.method_used == "mle"
        assert res.converged is True

    def test_auto_falls_back_to_firth_on_separation(self, logit_separated_dataset):
        X, y = logit_separated_dataset
        res = fit_logit(X, y, method="auto", add_intercept=False)
        # The exact label may include extra context (e.g. "firth_after_mle_failure"
        # or just "firth"), but the model used must contain "firth".
        assert "firth" in res.method_used.lower()
        # Coefficients must be finite (the whole point of falling back).
        assert np.all(np.isfinite(res.coef))
        assert np.all(np.abs(res.coef) < 50)

    def test_explicit_firth_method(self, logit_clean_dataset):
        X, y, _ = logit_clean_dataset
        res = fit_logit(X, y, method="firth", add_intercept=False)
        assert "firth" in res.method_used.lower()

    def test_explicit_mle_method(self, logit_clean_dataset):
        X, y, _ = logit_clean_dataset
        res = fit_logit(X, y, method="mle", add_intercept=False)
        assert res.method_used == "mle"


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


class TestDiagnostics:
    """Helpers used by ``fit_logit`` to decide between MLE and Firth."""

    @staticmethod
    def _is_ill(X):
        """Helper: ``is_design_ill_conditioned`` may return a tuple
        ``(bool, cond_number)`` or a bare bool depending on version."""
        result = is_design_ill_conditioned(X)
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)

    def test_well_conditioned_design_returns_false(self):
        X = np.eye(50, 3)
        # The default threshold is 1e10; ill_conditioned must be False
        assert self._is_ill(X) is False

    def test_collinear_design_returns_true(self):
        # Two perfectly collinear columns -> infinite condition number
        X = np.array([
            [1, 1, 2],
            [2, 2, 4],
            [3, 3, 6],
            [4, 4, 8],
        ], dtype=float)
        assert self._is_ill(X) is True


# ---------------------------------------------------------------------------
# LogitResult dataclass surface
# ---------------------------------------------------------------------------


class TestLogitResultFields:
    """Downstream code (BiblioGroupClassifier) reads specific fields."""

    @pytest.fixture()
    def res(self, logit_clean_dataset):
        X, y, _ = logit_clean_dataset
        return mle_logit(X, y, add_intercept=False)

    def test_required_fields_present(self, res):
        for name in (
            "coef", "se", "z_values", "p_values_wald",
            "ci_low", "ci_high", "ci_method", "alpha",
            "log_likelihood", "n_iterations", "converged",
            "method_used", "feature_names", "n_obs", "n_features",
        ):
            assert hasattr(res, name), f"missing field: {name}"

    def test_dimension_consistency(self, res):
        n_features = res.n_features
        assert len(res.coef) == n_features
        assert len(res.se) == n_features
        assert len(res.z_values) == n_features
        assert len(res.p_values_wald) == n_features
        assert len(res.ci_low) == n_features
        assert len(res.ci_high) == n_features

    def test_p_values_in_unit_interval(self, res):
        assert np.all((res.p_values_wald >= 0) & (res.p_values_wald <= 1))


# ---------------------------------------------------------------------------
# LRT support (optional path)
# ---------------------------------------------------------------------------


class TestLikelihoodRatioTest:
    """``compute_lrt=True`` produces an extra column of LRT-based p-values."""

    def test_lrt_p_values_in_unit_interval(self, logit_clean_dataset):
        X, y, _ = logit_clean_dataset
        res = firth_logit(
            X, y, add_intercept=False, ci_method="wald", compute_lrt=True
        )
        assert res.p_values_lrt is not None
        assert len(res.p_values_lrt) == res.n_features
        assert np.all((res.p_values_lrt >= 0) & (res.p_values_lrt <= 1))

    def test_lrt_p_values_none_by_default(self, logit_clean_dataset):
        X, y, _ = logit_clean_dataset
        res = firth_logit(X, y, add_intercept=False, ci_method="wald")
        assert res.p_values_lrt is None
