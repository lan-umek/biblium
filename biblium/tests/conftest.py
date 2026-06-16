# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for the biblium test suite.

The fixtures here aim to keep tests
- *deterministic* (every RNG state is seeded explicitly),
- *hermetic*    (no network access; all data is generated or shipped
                 in ``tests/fixtures/``),
- *fast*        (default permutation budgets are tiny).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to ``tests/fixtures/``."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture()
def tmp_cache_dir(tmp_path) -> Path:
    """Per-test temporary directory for on-disk caches."""
    p = tmp_path / "cache"
    p.mkdir()
    return p


# ---------------------------------------------------------------------------
# Random number generators
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    """Fixed-seed RNG used by individual tests; never bleeds between tests."""
    return np.random.default_rng(20260430)


# ---------------------------------------------------------------------------
# Synthetic bibliometric data
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_doc_term_matrix(rng: np.random.Generator) -> np.ndarray:
    """
    A tiny binary document-by-term matrix (20 docs, 6 terms).

    Used for permutation tests over chi-square / Cramer's V style
    statistics. Designed to have one strong association
    (term 0 with the first half of documents) so that p-values come
    out small with very few permutations.
    """
    n_docs, n_terms = 20, 6
    M = (rng.random((n_docs, n_terms)) < 0.3).astype(int)
    # Inject a strong signal: term 0 is present in the first 10 docs only
    M[:10, 0] = 1
    M[10:, 0] = 0
    return M


@pytest.fixture()
def disjoint_groups() -> np.ndarray:
    """20 documents partitioned into 4 disjoint groups of 5."""
    g = np.zeros((20, 4), dtype=int)
    for i in range(4):
        g[i * 5:(i + 1) * 5, i] = 1
    return g


@pytest.fixture()
def overlapping_groups(rng: np.random.Generator) -> np.ndarray:
    """20 documents x 4 overlapping groups (rows can sum to 0, 1, or >1)."""
    g = (rng.random((20, 4)) < 0.45).astype(int)
    # Make sure each group has at least one member, otherwise tests of the
    # statistic are degenerate.
    for j in range(g.shape[1]):
        if g[:, j].sum() == 0:
            g[j, j] = 1
    return g


# ---------------------------------------------------------------------------
# Logistic-regression toy datasets
# ---------------------------------------------------------------------------


@pytest.fixture()
def logit_clean_dataset(rng: np.random.Generator):
    """
    A well-behaved logistic-regression dataset where MLE converges
    cleanly. Use to compare ``mle_logit`` against ``statsmodels.Logit``.
    """
    n, p = 200, 3
    X = rng.standard_normal((n, p))
    beta = np.array([0.5, -1.0, 1.5])
    logit = X @ beta
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    # Add an intercept column at position 0
    X = np.column_stack([np.ones(n), X])
    return X, y, np.r_[0.0, beta]  # true intercept is 0


@pytest.fixture()
def logit_separated_dataset():
    """
    A perfectly-separated dataset along x1.

    With this fixture:
    - ``mle_logit`` either fails to converge or produces unreasonably
      large coefficients (a sign of separation).
    - ``firth_logit`` produces finite, sensible estimates (penalisation
      regularises the score equation).
    - ``fit_logit(method='auto')`` should detect the separation and
      fall back to Firth.
    """
    n = 30
    x1 = np.linspace(-3, 3, n)
    y = (x1 > 0).astype(int)
    x2 = np.linspace(-1, 1, n)
    X = np.column_stack([np.ones(n), x1, x2])
    return X, y


# ---------------------------------------------------------------------------
# COBISS sample fixtures (parser tests)
# ---------------------------------------------------------------------------


@pytest.fixture()
def cobiss_sample_text(fixtures_dir: Path) -> str:
    """Markdown-rendered sample of a COBISS+ personal bibliography (7 records)."""
    return (fixtures_dir / "cobiss_sample.md").read_text(encoding="utf-8")


@pytest.fixture()
def cobiss_full_sample_text(fixtures_dir: Path) -> str:
    """Larger sample (14 records) covering edge cases:
    et al. in the middle, 12-author records, names with apostrophes,
    multi-word surnames, role tags."""
    return (fixtures_dir / "cobiss_full_sample.md").read_text(encoding="utf-8")
