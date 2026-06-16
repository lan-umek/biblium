# -*- coding: utf-8 -*-
"""
Integration tests that exercise the public functions
``assoc_permutation_test`` and ``firth_logit`` together on synthetic
bibliometric-shaped data.

The full ``BiblioGroupClassifier.logistic_regression_analysis`` workflow
involves heavy machinery (BiblioStats, vocab extraction, preprocessing)
that is out of scope for the unit-level test suite. We test the
*statistical* layer directly: build a doc-term matrix and a group
matrix, run permutation chi2 and a per-group Firth logit, and check
the outputs are coherent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from biblium.utilsbib_modules.firth import fit_logit
from biblium.utilsbib_modules.permutation import (
    adjust_p_values,
    assoc_permutation_test,
)


# ---------------------------------------------------------------------------
# Tiny synthetic doc-term / group matrices
# ---------------------------------------------------------------------------


@pytest.fixture()
def doc_term_and_groups(rng):
    """
    A 60-doc x 8-term doc-term matrix with three disjoint year-bucket
    groups. The first three terms are concentrated in groups 0/1/2
    respectively, so chi2 should detect a strong association.
    """
    n_docs, n_terms = 60, 8
    M = (rng.random((n_docs, n_terms)) < 0.25).astype(int)
    # Inject signal: term 0 is enriched in docs 0..19 (group 0),
    # term 1 in docs 20..39 (group 1), term 2 in docs 40..59 (group 2).
    M[:20, 0] = 1
    M[20:40, 1] = 1
    M[40:, 2] = 1

    G = np.zeros((n_docs, 3), dtype=int)
    G[:20, 0] = 1
    G[20:40, 1] = 1
    G[40:, 2] = 1
    return M, G


# ---------------------------------------------------------------------------
# Permutation chi2 + multiple-testing on a residuals matrix
# ---------------------------------------------------------------------------


class TestPermutationOnDocTerm:
    """End-to-end permutation chi2 on a synthetic doc-term x group setup."""

    def test_chi2_detects_signal(self, doc_term_and_groups):
        M, G = doc_term_and_groups
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = assoc_permutation_test(
                G, M,
                test="chi2",
                n_permutations=499,
                random_state=2026,
                warn_disjoint=False,
            )
        p = float(np.asarray(res.p_value).ravel()[0])
        assert p < 0.05

    def test_residuals_pvalues_have_correct_shape(self, doc_term_and_groups):
        M, G = doc_term_and_groups
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = assoc_permutation_test(
                G, M,
                test="residuals",
                n_permutations=199,
                random_state=2026,
                warn_disjoint=False,
            )
        p = np.asarray(res.p_value)
        assert p.shape == (G.shape[1], M.shape[1])

    def test_bh_correction_dominates_raw(self, doc_term_and_groups):
        """Applying BH on flat p-values must produce values >= raw."""
        M, G = doc_term_and_groups
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = assoc_permutation_test(
                G, M,
                test="residuals",
                n_permutations=199,
                random_state=2026,
                multiple_testing="none",  # raw p-values
                warn_disjoint=False,
            )
        p = np.asarray(res.p_value).ravel()
        bh = adjust_p_values(p, method="bh")
        assert np.all(bh + 1e-12 >= p)


# ---------------------------------------------------------------------------
# Per-group Firth logit on the same setup
# ---------------------------------------------------------------------------


class TestFirthOnDocTerm:
    """Run Firth logit per group, terms-as-features."""

    def test_each_group_yields_finite_coefs(self, doc_term_and_groups):
        M, G = doc_term_and_groups
        for g in range(G.shape[1]):
            y = G[:, g]
            X = np.column_stack([np.ones(M.shape[0]), M])  # add intercept manually
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fit_logit(X, y, method="auto", add_intercept=False)
            assert np.all(np.isfinite(res.coef)), g
            # Method label should be one of the known strings
            assert res.method_used.lower() in (
                "mle", "firth", "firth_after_mle_failure",
            )

    def test_signal_term_has_positive_coef(self, doc_term_and_groups):
        M, G = doc_term_and_groups
        # Group 0 is "documents 0..19" and term 0 is enriched there.
        # The Firth coefficient for term 0 must be positive.
        y = G[:, 0]
        X = np.column_stack([np.ones(M.shape[0]), M])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_logit(X, y, method="auto", add_intercept=False)
        # Coef index 1 is term 0 (intercept is 0)
        assert res.coef[1] > 0
