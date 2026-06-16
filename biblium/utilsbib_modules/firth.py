# -*- coding: utf-8 -*-
"""
biblium.utilsbib_modules.firth_logit
====================================

Firth-penalized logistic regression with optional auto-detection
of (near-)singular design matrices and (quasi-)complete separation.

Motivation
----------
Standard maximum likelihood estimation (MLE) of a logistic regression
fails or yields unbounded coefficients when the design matrix is
(nearly) singular or when the binary outcome is perfectly separable by
some linear combination of predictors. Both situations occur frequently
in bibliometric group analyses with overlapping groups and rare
predictors (e.g. a keyword that appears in only one or two documents,
all of which belong to the same group).

Firth (1993) proposed adding a Jeffreys-prior penalty term
``(1/2) log|I(beta)|`` to the log-likelihood, which:

- guarantees a finite maximum penalised likelihood estimate (PMLE)
  even under complete separation;
- removes the leading-order bias of the MLE;
- is invariant under reparametrisation.

Heinze and Schemper (2002) recommend profile-likelihood (PL) confidence
intervals over Wald CIs for Firth estimates, because Wald CIs may still
mislead when the underlying separation is severe.

References
----------
Firth, D. (1993). Bias reduction of maximum likelihood estimates.
    Biometrika, 80(1), 27-38.
Heinze, G., & Schemper, M. (2002). A solution to the problem of
    separation in logistic regression. Statistics in Medicine, 21(16),
    2409-2419.

Public components
-----------------
LogitResult
    Dataclass holding coefficients, standard errors, z-values, Wald
    and (optionally) likelihood-ratio p-values, Wald or
    profile-likelihood CIs, fit diagnostics, and a label of which
    method was actually used.

firth_logit
    Fit Firth-penalised logistic regression directly.

mle_logit
    Fit ordinary MLE logistic regression (provided for completeness
    and for the auto-detect path).

fit_logit
    Auto-detect routine. Inspects the design matrix and the MLE result;
    falls back to Firth on (near-)singular X or evidence of separation.

is_design_ill_conditioned
    Diagnostic for the design matrix.

detect_separation_after_fit
    Diagnostic applied to a fitted MLE to detect (quasi-)complete
    separation post hoc.

Author: Lan Umek
Version: 2.16.0
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit, log1p

# =============================================================================
# CONSTANTS
# =============================================================================

# Clip linear predictor to avoid overflow in exp; |eta| > 30 already gives
# pi extremely close to 0 or 1.
_ETA_CLIP = 30.0

# Singularity diagnostic threshold on cond(X.T @ X). Conventional rule of
# thumb: cond > 1e10 is "near singular" for double precision.
_COND_THRESHOLD_DEFAULT = 1e10

# Coefficient threshold above which we suspect quasi-complete separation
# in a fitted MLE. Heinze (1999) suggests 10-20; we use 15 as a soft default.
_SEP_COEF_THRESHOLD = 15.0


# =============================================================================
# RESULT CONTAINER
# =============================================================================


@dataclass
class LogitResult:
    """
    Container for the result of a (Firth or MLE) logistic regression fit.

    Attributes
    ----------
    coef : np.ndarray, shape (p,)
        Estimated coefficients (intercept first if ``add_intercept=True``
        was used).
    se : np.ndarray, shape (p,)
        Standard errors from the inverse Fisher information at the fit.
    z_values : np.ndarray, shape (p,)
        coef / se.
    p_values_wald : np.ndarray, shape (p,)
        Two-sided Wald p-values from the standard normal distribution.
    p_values_lrt : np.ndarray or None
        Two-sided likelihood-ratio test p-values for ``H_0: beta_j = 0``.
        ``None`` unless ``compute_lrt=True`` was passed.
    ci_low, ci_high : np.ndarray, shape (p,)
        Lower / upper bounds of the requested confidence intervals.
    ci_method : str
        ``"wald"`` or ``"profile"``.
    alpha : float
        The CI level used (default 0.05 for 95% CIs).
    log_likelihood : float
        Unpenalised log-likelihood at the fitted coefficients.
    log_likelihood_penalised : float or None
        Firth penalised log-likelihood; ``None`` for plain MLE.
    n_iterations : int
        Newton-Raphson iterations used.
    converged : bool
        Whether the optimiser declared convergence.
    method_used : str
        ``"firth"``, ``"mle"``, or ``"firth_after_mle_failure"``.
    feature_names : list of str
        Predictor labels (length p).
    n_obs : int
    n_features : int
    extra : dict
        Free-form diagnostics (separation flags, condition number, ...).
    """

    coef: np.ndarray
    se: np.ndarray
    z_values: np.ndarray
    p_values_wald: np.ndarray
    p_values_lrt: Optional[np.ndarray]
    ci_low: np.ndarray
    ci_high: np.ndarray
    ci_method: str
    alpha: float
    log_likelihood: float
    log_likelihood_penalised: Optional[float]
    null_log_likelihood: float = 0.0  # llf of intercept-only model, for pseudo-R^2
    n_iterations: int = 0
    converged: bool = False
    method_used: str = ""
    feature_names: List[str] = field(default_factory=list)
    n_obs: int = 0
    n_features: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a tidy summary DataFrame, one row per coefficient."""
        cols = {
            "coef": self.coef,
            "se": self.se,
            "z": self.z_values,
            "p_wald": self.p_values_wald,
            f"ci_low_{int((1 - self.alpha) * 100)}": self.ci_low,
            f"ci_high_{int((1 - self.alpha) * 100)}": self.ci_high,
        }
        if self.p_values_lrt is not None:
            cols["p_lrt"] = self.p_values_lrt
        df = pd.DataFrame(cols, index=self.feature_names)
        df.index.name = "feature"
        return df

    def to_statsmodels_compat_summary(self) -> pd.DataFrame:
        """
        Return a coefficient table in the column-name style produced by
        ``statsmodels.LogitResults.summary2().tables[1]``.

        This is provided so that downstream code (e.g. ``save_logistic_results``)
        that was written against the statsmodels output format keeps working
        when its source is replaced by a Firth/MLE fit from this module.

        Columns: ``Coef.``, ``Std.Err.``, ``z``, ``P>|z|``, ``[0.025``, ``0.975]``.
        """
        lo_pct = (self.alpha / 2)
        hi_pct = 1 - lo_pct
        return pd.DataFrame(
            {
                "Coef.": self.coef,
                "Std.Err.": self.se,
                "z": self.z_values,
                "P>|z|": self.p_values_wald,
                f"[{lo_pct:.3f}": self.ci_low,
                f"{hi_pct:.3f}]": self.ci_high,
            },
            index=self.feature_names,
        )

    # ------------------------------------------------------------------
    # statsmodels-compatible aliases (so a LogitResult is a drop-in
    # replacement for ``statsmodels.LogitResults`` without needing a
    # separate adapter object).
    # ------------------------------------------------------------------
    @property
    def params(self) -> pd.Series:
        return pd.Series(self.coef, index=self.feature_names, name="coef")

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(self.p_values_wald, index=self.feature_names, name="p_wald")

    @property
    def bse(self) -> pd.Series:
        return pd.Series(self.se, index=self.feature_names, name="se")

    @property
    def tvalues(self) -> pd.Series:
        return pd.Series(self.z_values, index=self.feature_names, name="z")

    @property
    def llf(self) -> float:
        return self.log_likelihood

    @property
    def llnull(self) -> float:
        return self.null_log_likelihood

    @property
    def nobs(self) -> int:
        return self.n_obs

    @property
    def aic(self) -> float:
        return -2.0 * self.log_likelihood + 2.0 * self.n_features

    @property
    def bic(self) -> float:
        return -2.0 * self.log_likelihood + self.n_features * np.log(max(self.n_obs, 1))

    @property
    def prsquared(self) -> float:
        if self.null_log_likelihood == 0:
            return 0.0
        return 1.0 - self.log_likelihood / self.null_log_likelihood

    def conf_int(self, alpha: Optional[float] = None) -> pd.DataFrame:
        if alpha is not None and abs(alpha - self.alpha) > 1e-9:
            warnings.warn(
                f"conf_int(alpha={alpha}) ignored; CIs were computed at "
                f"alpha={self.alpha} during fitting.",
                UserWarning, stacklevel=2,
            )
        return pd.DataFrame(
            {0: self.ci_low, 1: self.ci_high}, index=self.feature_names,
        )

    def __repr__(self) -> str:
        return (
            f"LogitResult(method={self.method_used!r}, "
            f"n_obs={self.n_obs}, n_features={self.n_features}, "
            f"converged={self.converged}, "
            f"log_lik={self.log_likelihood:.4g})"
        )


# =============================================================================
# Statsmodels-compatible adapter
# =============================================================================


class StatsmodelsCompatAdapter:
    """
    Lightweight adapter that exposes the attributes commonly used from
    ``statsmodels.LogitResults`` (``params``, ``bse``, ``nobs``, ``aic``,
    ``bic``, ``prsquared``) on top of a ``LogitResult``.

    This is intended for legacy code that consumes statsmodels logit
    results and that we do not want to rewrite immediately.
    """

    def __init__(self, logit_result: LogitResult, y: np.ndarray):
        self._result = logit_result
        y_arr = np.asarray(y, dtype=float).ravel()
        n = int(y_arr.size)
        k = int(logit_result.n_features)

        # Coefficient series indexed by feature name
        self.params = pd.Series(logit_result.coef, index=logit_result.feature_names)
        self.bse = pd.Series(logit_result.se, index=logit_result.feature_names)
        self.tvalues = pd.Series(logit_result.z_values, index=logit_result.feature_names)
        self.pvalues = pd.Series(
            logit_result.p_values_wald, index=logit_result.feature_names
        )

        # Information criteria from the unpenalised log-likelihood
        ll = float(logit_result.log_likelihood)
        self.llf = ll
        self.nobs = n
        self.aic = -2.0 * ll + 2.0 * k
        self.bic = -2.0 * ll + k * np.log(max(n, 1))

        # McFadden pseudo-R^2 against the intercept-only Bernoulli null
        p = float(np.clip(y_arr.mean(), 1e-15, 1 - 1e-15))
        ll_null = float(n * (p * np.log(p) + (1 - p) * np.log(1 - p)))
        self.llnull = ll_null
        self.prsquared = (1.0 - ll / ll_null) if ll_null != 0.0 else 0.0

        # Pass-throughs commonly used by exporters
        self.df_model = k - 1  # excluding intercept
        self.df_resid = n - k
        self.converged = logit_result.converged
        self.method_used = logit_result.method_used

    def __getattr__(self, item):
        # Last-resort fallback to underlying LogitResult attributes
        return getattr(self._result, item)


# =============================================================================
# CORE NUMERICAL HELPERS
# =============================================================================


def _safe_predict(X: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (eta, pi) with eta clipped to ``[-_ETA_CLIP, _ETA_CLIP]``."""
    eta = np.clip(X @ beta, -_ETA_CLIP, _ETA_CLIP)
    pi = expit(eta)
    return eta, pi


def _log_likelihood(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """
    Numerically stable Bernoulli log-likelihood:
    sum_i [ y_i * eta_i - log(1 + exp(eta_i)) ].
    """
    eta = np.clip(X @ beta, -_ETA_CLIP, _ETA_CLIP)
    # log(1 + exp(eta)) computed stably:
    # = max(0, eta) + log1p(exp(-|eta|))
    pos = np.maximum(eta, 0.0)
    log_one_plus_exp = pos + log1p(np.exp(-np.abs(eta)))
    return float(np.sum(y * eta - log_one_plus_exp))


def _fisher_information(X: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Return X' W X with W = diag(pi * (1 - pi))."""
    W = pi * (1.0 - pi)
    Xw = X * W[:, None]
    return X.T @ Xw  # symmetric, p x p


def _safe_inverse(M: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Invert a symmetric matrix; fall back to pseudoinverse on singularity.
    Returns ``(M_inv, was_pinv)``.
    """
    try:
        return np.linalg.inv(M), False
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M), True


def _penalised_log_likelihood(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """
    Firth-penalised log-likelihood:
    l_pen(beta) = l(beta) + (1/2) log|I(beta)|.

    Uses ``slogdet`` for numerical robustness; returns ``-inf`` if the
    information matrix is non-positive-definite (so step halving will
    reject the step).
    """
    _, pi = _safe_predict(X, beta)
    I = _fisher_information(X, pi)
    sign, logdet = np.linalg.slogdet(I)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf
    return _log_likelihood(X, y, beta) + 0.5 * logdet


# =============================================================================
# DESIGN-MATRIX DIAGNOSTICS
# =============================================================================


def is_design_ill_conditioned(
    X: np.ndarray,
    threshold: float = _COND_THRESHOLD_DEFAULT,
) -> Tuple[bool, float]:
    """
    Test whether ``X.T @ X`` is (near-)singular.

    Returns
    -------
    flag : bool
        True if cond(X.T @ X) > threshold or X is rank-deficient.
    cond : float
        The 2-norm condition number of X.T @ X (np.inf if rank-deficient).
    """
    X = np.asarray(X, dtype=float)
    XtX = X.T @ X
    try:
        cond = float(np.linalg.cond(XtX))
    except np.linalg.LinAlgError:
        cond = np.inf
    flag = (not np.isfinite(cond)) or cond > threshold
    return flag, cond


def detect_separation_after_fit(
    coef: np.ndarray,
    converged: bool,
    coef_threshold: float = _SEP_COEF_THRESHOLD,
) -> Tuple[bool, str]:
    """
    Flag (quasi-)complete separation based on a fitted MLE.

    Heuristics:

    - Non-convergence is a strong signal.
    - Any |coef| above ``coef_threshold`` (default 15, corresponding to an
      odds ratio of e^15 ~ 3.3 million) is implausible in practice and
      typically arises only under separation.
    """
    if not converged:
        return True, "MLE did not converge"
    max_abs = float(np.max(np.abs(coef))) if coef.size else 0.0
    if max_abs > coef_threshold:
        return True, f"Implausibly large coefficient (max |coef| = {max_abs:.2f})"
    return False, ""


# =============================================================================
# CORE FITTERS
# =============================================================================


def _newton_logit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    firth: bool,
    max_iter: int,
    tol: float,
    init_beta: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, bool, np.ndarray]:
    """
    Newton-Raphson logistic regression with optional Firth correction
    and step halving.

    Returns
    -------
    beta : np.ndarray
        Fitted coefficients.
    n_iter : int
        Number of iterations taken.
    converged : bool
        Whether the convergence criterion was met.
    I_inv : np.ndarray
        Inverse Fisher information at the fit (used for SEs).
    """
    n, p = X.shape
    beta = np.zeros(p) if init_beta is None else np.asarray(init_beta, dtype=float).copy()

    # Objective for step halving
    if firth:
        objective = lambda b: _penalised_log_likelihood(X, y, b)
    else:
        objective = lambda b: _log_likelihood(X, y, b)

    obj_old = objective(beta)
    converged = False

    for iteration in range(1, max_iter + 1):
        _, pi = _safe_predict(X, beta)
        I = _fisher_information(X, pi)
        I_inv, used_pinv = _safe_inverse(I)

        # Score (with Firth correction term if requested)
        if firth:
            # Hat-matrix diagonals h_i = (X I^-1 X')_ii * W_i
            XIinv = X @ I_inv
            diag_hat = np.einsum("ij,ij->i", XIinv, X)  # (X I^-1 X')_ii
            W = pi * (1.0 - pi)
            h = diag_hat * W
            U = X.T @ (y - pi + h * (0.5 - pi))
        else:
            U = X.T @ (y - pi)

        delta = I_inv @ U

        # Step halving
        step = 1.0
        beta_new = beta + delta
        obj_new = objective(beta_new)
        n_halving = 0
        while (not np.isfinite(obj_new) or obj_new < obj_old - 1e-12) and n_halving < 30:
            step *= 0.5
            beta_new = beta + step * delta
            obj_new = objective(beta_new)
            n_halving += 1

        if not np.isfinite(obj_new):
            # Cannot make any progress: bail out flagged as non-converged.
            break

        beta = beta_new
        obj_old = obj_new

        # Convergence check on the (possibly reduced) step
        if np.max(np.abs(step * delta)) < tol:
            converged = True
            break

    # Final information matrix at the fit
    _, pi = _safe_predict(X, beta)
    I_final = _fisher_information(X, pi)
    I_inv_final, _ = _safe_inverse(I_final)
    return beta, iteration, converged, I_inv_final


# -----------------------------------------------------------------------------
# Public fitters
# -----------------------------------------------------------------------------


def _prepare_design(
    X: Union[np.ndarray, pd.DataFrame],
    add_intercept: bool,
    feature_names: Optional[Sequence[str]],
) -> Tuple[np.ndarray, List[str]]:
    """Coerce X to ndarray and add intercept; build feature names."""
    if isinstance(X, pd.DataFrame):
        names = list(X.columns.astype(str))
        Xa = X.to_numpy(dtype=float, copy=True)
    else:
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim != 2:
            raise ValueError("X must be 2-D.")
        if feature_names is None:
            names = [f"x{i}" for i in range(Xa.shape[1])]
        else:
            names = list(feature_names)
            if len(names) != Xa.shape[1]:
                raise ValueError(
                    f"feature_names has length {len(names)} but X has "
                    f"{Xa.shape[1]} columns."
                )

    if add_intercept:
        # Avoid duplicating an existing constant column
        if not np.allclose(Xa[:, 0], 1.0) if Xa.shape[1] > 0 else True:
            Xa = np.column_stack([np.ones(Xa.shape[0]), Xa])
            names = ["const"] + names

    return Xa, names


def firth_logit(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    *,
    add_intercept: bool = True,
    max_iter: int = 200,
    tol: float = 1e-6,
    feature_names: Optional[Sequence[str]] = None,
    ci_method: Literal["wald", "profile"] = "profile",
    alpha: float = 0.05,
    compute_lrt: bool = False,
) -> LogitResult:
    """
    Fit Firth-penalised logistic regression.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame, shape (n, p)
        Design matrix. Column names of a DataFrame are used as
        ``feature_names``.
    y : np.ndarray or pd.Series, shape (n,)
        Binary outcome (0/1).
    add_intercept : bool, default True
        Prepend a constant column unless one is already present.
    max_iter : int, default 100
        Maximum Newton-Raphson iterations.
    tol : float, default 1e-8
        Convergence tolerance on max absolute step.
    feature_names : sequence of str, optional
        Used only when X is a plain ndarray.
    ci_method : {"wald", "profile"}, default "profile"
        ``"profile"`` is recommended by Heinze & Schemper (2002) for Firth
        and is the default; ``"wald"`` is faster but less reliable when
        the underlying problem involved separation.
    alpha : float, default 0.05
        CI significance level (95% CIs by default).
    compute_lrt : bool, default False
        If True, also compute likelihood-ratio p-values for each
        coefficient (one penalised refit per coefficient -- expensive).

    Returns
    -------
    LogitResult

    Notes
    -----
    Standard errors come from the inverse Fisher information at the
    fitted coefficients (i.e. the standard sandwich denominator before
    the Firth penalty), in line with the ``logistf`` R package.
    """
    Xa, names = _prepare_design(X, add_intercept, feature_names)
    ya = np.asarray(y, dtype=float).ravel()
    if ya.shape[0] != Xa.shape[0]:
        raise ValueError(
            f"X has {Xa.shape[0]} rows but y has {ya.shape[0]} values."
        )
    if not np.all((ya == 0) | (ya == 1)):
        raise ValueError("y must be binary (0/1).")
    if np.unique(ya).size < 2:
        raise ValueError("y has only one class; cannot fit logistic regression.")

    beta, n_iter, converged, I_inv = _newton_logit(
        Xa, ya, firth=True, max_iter=max_iter, tol=tol
    )

    se = np.sqrt(np.clip(np.diag(I_inv), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, beta / se, 0.0)
    p_wald = 2 * (1 - stats.norm.cdf(np.abs(z)))

    ll = _log_likelihood(Xa, ya, beta)
    ll_pen = _penalised_log_likelihood(Xa, ya, beta)

    # Confidence intervals
    if ci_method == "wald":
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_low = beta - z_crit * se
        ci_high = beta + z_crit * se
    else:  # profile
        ci_low, ci_high = _profile_likelihood_ci(
            Xa, ya, beta, ll_pen, alpha=alpha,
            firth=True, max_iter=max_iter, tol=tol,
            wald_se=se,
        )

    p_lrt = None
    if compute_lrt:
        p_lrt = _likelihood_ratio_pvalues(
            Xa, ya, beta, ll_pen, firth=True, max_iter=max_iter, tol=tol
        )

    # Null log-likelihood (intercept-only model) for McFadden pseudo-R^2
    p_bar = float(np.mean(ya))
    if 0.0 < p_bar < 1.0:
        ll_null = float(ya.size) * (
            p_bar * np.log(p_bar) + (1.0 - p_bar) * np.log(1.0 - p_bar)
        )
    else:
        ll_null = 0.0

    return LogitResult(
        coef=beta,
        se=se,
        z_values=z,
        p_values_wald=p_wald,
        p_values_lrt=p_lrt,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method=ci_method,
        alpha=alpha,
        log_likelihood=ll,
        log_likelihood_penalised=ll_pen,
        null_log_likelihood=ll_null,
        n_iterations=n_iter,
        converged=converged,
        method_used="firth",
        feature_names=names,
        n_obs=int(Xa.shape[0]),
        n_features=int(Xa.shape[1]),
        extra={},
    )


def mle_logit(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    *,
    add_intercept: bool = True,
    max_iter: int = 200,
    tol: float = 1e-6,
    feature_names: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> LogitResult:
    """
    Fit ordinary MLE logistic regression (no Firth correction).

    Provided primarily so that ``fit_logit(method='auto')`` has a clean
    in-house fallback path; for stand-alone MLE, ``statsmodels.Logit``
    remains a reasonable choice.
    """
    Xa, names = _prepare_design(X, add_intercept, feature_names)
    ya = np.asarray(y, dtype=float).ravel()

    beta, n_iter, converged, I_inv = _newton_logit(
        Xa, ya, firth=False, max_iter=max_iter, tol=tol
    )

    se = np.sqrt(np.clip(np.diag(I_inv), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, beta / se, 0.0)
    p_wald = 2 * (1 - stats.norm.cdf(np.abs(z)))

    ll = _log_likelihood(Xa, ya, beta)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = beta - z_crit * se
    ci_high = beta + z_crit * se

    sep, msg = detect_separation_after_fit(beta, converged)

    # Null log-likelihood (intercept-only model) for McFadden pseudo-R^2
    p_bar = float(np.mean(ya))
    if 0.0 < p_bar < 1.0:
        ll_null = float(ya.size) * (
            p_bar * np.log(p_bar) + (1.0 - p_bar) * np.log(1.0 - p_bar)
        )
    else:
        ll_null = 0.0

    return LogitResult(
        coef=beta,
        se=se,
        z_values=z,
        p_values_wald=p_wald,
        p_values_lrt=None,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method="wald",
        alpha=alpha,
        log_likelihood=ll,
        log_likelihood_penalised=None,
        null_log_likelihood=ll_null,
        n_iterations=n_iter,
        converged=converged,
        method_used="mle",
        feature_names=names,
        n_obs=int(Xa.shape[0]),
        n_features=int(Xa.shape[1]),
        extra={"separation_detected": sep, "separation_msg": msg},
    )


def fit_logit(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    *,
    method: Literal["auto", "firth", "mle"] = "auto",
    add_intercept: bool = True,
    max_iter: int = 200,
    tol: float = 1e-6,
    feature_names: Optional[Sequence[str]] = None,
    ci_method: Optional[Literal["wald", "profile"]] = None,
    alpha: float = 0.05,
    compute_lrt: bool = False,
    cond_threshold: float = _COND_THRESHOLD_DEFAULT,
) -> LogitResult:
    """
    Fit logistic regression with auto-detection of pathological designs.

    Parameters
    ----------
    method : {"auto", "firth", "mle"}, default "auto"
        - "firth" : always use Firth.
        - "mle"   : always use MLE (no separation handling).
        - "auto"  : inspect the design matrix; if cond(X.T X) is above
          ``cond_threshold`` use Firth directly. Otherwise try MLE; if
          MLE diverges or shows signs of separation, fall back to Firth.
    cond_threshold : float, default 1e10
        Threshold on cond(X.T @ X) for the auto path.
    ci_method : {"wald", "profile"}, optional
        Defaults to "profile" for Firth and "wald" for MLE.
    Other parameters
        See ``firth_logit`` and ``mle_logit``.

    Returns
    -------
    LogitResult
        ``method_used`` records which fitter actually produced the
        result. ``extra`` carries diagnostic flags.
    """
    Xa, names = _prepare_design(X, add_intercept, feature_names)
    ya = np.asarray(y, dtype=float).ravel()

    if method == "firth":
        ci = ci_method if ci_method is not None else "profile"
        return firth_logit(
            Xa, ya, add_intercept=False, max_iter=max_iter, tol=tol,
            feature_names=names, ci_method=ci, alpha=alpha,
            compute_lrt=compute_lrt,
        )

    if method == "mle":
        ci = ci_method if ci_method is not None else "wald"
        if ci != "wald":
            warnings.warn(
                "ci_method='profile' is only implemented for Firth fits; "
                "using Wald CIs for MLE.",
                UserWarning, stacklevel=2,
            )
        return mle_logit(
            Xa, ya, add_intercept=False, max_iter=max_iter, tol=tol,
            feature_names=names, alpha=alpha,
        )

    # Auto path
    ill, cond = is_design_ill_conditioned(Xa, threshold=cond_threshold)
    if ill:
        ci = ci_method if ci_method is not None else "profile"
        result = firth_logit(
            Xa, ya, add_intercept=False, max_iter=max_iter, tol=tol,
            feature_names=names, ci_method=ci, alpha=alpha,
            compute_lrt=compute_lrt,
        )
        result.extra["auto_reason"] = (
            f"cond(X.T@X) = {cond:.2e} > {cond_threshold:.0e}"
        )
        result.extra["cond_xtx"] = cond
        return result

    # Try MLE first
    mle = mle_logit(
        Xa, ya, add_intercept=False, max_iter=max_iter, tol=tol,
        feature_names=names, alpha=alpha,
    )
    sep_flag = mle.extra.get("separation_detected", False)
    if sep_flag:
        ci = ci_method if ci_method is not None else "profile"
        result = firth_logit(
            Xa, ya, add_intercept=False, max_iter=max_iter, tol=tol,
            feature_names=names, ci_method=ci, alpha=alpha,
            compute_lrt=compute_lrt,
        )
        result.method_used = "firth_after_mle_failure"
        result.extra["auto_reason"] = mle.extra.get("separation_msg", "")
        result.extra["mle_attempt"] = {
            "converged": mle.converged,
            "max_abs_coef": float(np.max(np.abs(mle.coef))) if mle.coef.size else 0.0,
        }
        result.extra["cond_xtx"] = cond
        return result

    mle.extra["cond_xtx"] = cond
    mle.extra["auto_reason"] = "MLE accepted (no ill-conditioning, no separation)"
    return mle


# =============================================================================
# PROFILE-LIKELIHOOD CI
# =============================================================================


def _refit_with_constraint(
    X: np.ndarray,
    y: np.ndarray,
    j: int,
    fixed_value: float,
    *,
    firth: bool,
    init_beta: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[float, bool]:
    """
    Refit logistic regression with ``beta[j]`` held at ``fixed_value``.

    Returns the (penalised) log-likelihood at the constrained MLE and a
    convergence flag.

    Implementation note
    -------------------
    For Firth, the penalty term is ``(1/2) log|I_full(beta)|``, where
    ``I_full`` is the Fisher information of the *unrestricted* model.
    A naive offset+reduced-design implementation would compute the
    penalty on ``I_red`` (the reduced design), which is mathematically
    a different penalty and gives incorrect profile-likelihood values.

    We therefore evaluate the objective on the *full* coefficient vector
    (with ``beta[j] = fixed_value``) and only compute the *gradient and
    Hessian step* with respect to the free coordinates. The score uses
    the full-design hat values; the Hessian step uses the reduced
    Fisher information (a standard Firth approximation).
    """
    n, p = X.shape

    if p == 1:
        # Only one parameter, and it is fixed: no optimisation needed.
        beta_full = np.array([fixed_value], dtype=float)
        if firth:
            return _penalised_log_likelihood(X, y, beta_full), True
        return _log_likelihood(X, y, beta_full), True

    keep = [k for k in range(p) if k != j]
    X_red = X[:, keep]

    def make_full(beta_red: np.ndarray) -> np.ndarray:
        beta_full = np.empty(p)
        beta_full[j] = fixed_value
        beta_full[keep] = beta_red
        return beta_full

    if firth:
        objective = lambda b_red: _penalised_log_likelihood(X, y, make_full(b_red))
    else:
        objective = lambda b_red: _log_likelihood(X, y, make_full(b_red))

    beta_red = init_beta[keep].copy()
    obj_old = objective(beta_red)
    converged = False

    for iteration in range(1, max_iter + 1):
        beta_full = make_full(beta_red)
        _, pi = _safe_predict(X, beta_full)
        W = pi * (1.0 - pi)

        # Reduced information for the Newton step
        Xw_red = X_red * W[:, None]
        I_red = X_red.T @ Xw_red
        I_red_inv, _ = _safe_inverse(I_red)

        if firth:
            # Full-design hat values for the Firth-corrected score
            I_full = _fisher_information(X, pi)
            I_full_inv, _ = _safe_inverse(I_full)
            XIinv_full = X @ I_full_inv
            diag_hat_full = np.einsum("ij,ij->i", XIinv_full, X)
            h_full = diag_hat_full * W
            U_red = X_red.T @ (y - pi + h_full * (0.5 - pi))
        else:
            U_red = X_red.T @ (y - pi)

        delta = I_red_inv @ U_red

        step = 1.0
        beta_red_new = beta_red + delta
        obj_new = objective(beta_red_new)
        n_halving = 0
        while (not np.isfinite(obj_new) or obj_new < obj_old - 1e-12) and n_halving < 30:
            step *= 0.5
            beta_red_new = beta_red + step * delta
            obj_new = objective(beta_red_new)
            n_halving += 1

        if not np.isfinite(obj_new):
            break
        beta_red = beta_red_new
        obj_old = obj_new
        if np.max(np.abs(step * delta)) < tol:
            converged = True
            break

    return obj_old, converged


def _profile_likelihood_ci(
    X: np.ndarray,
    y: np.ndarray,
    beta_hat: np.ndarray,
    ll_full: float,
    *,
    alpha: float,
    firth: bool,
    max_iter: int,
    tol: float,
    wald_se: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute profile-likelihood CIs for every coefficient.

    For each j we find b such that
    ``2 * (ll_full - ll_constrained(beta_j = b)) == chi2_{1, 1-alpha}``,
    using Brent's method. Initial brackets come from the Wald CI,
    expanded outwards if needed.
    """
    p = beta_hat.size
    half_target = 0.5 * stats.chi2.ppf(1 - alpha, df=1)  # 1.92 for 95%

    ci_low = np.empty(p)
    ci_high = np.empty(p)

    z_crit = stats.norm.ppf(1 - alpha / 2)

    for j in range(p):
        b_hat = beta_hat[j]
        se_j = wald_se[j] if wald_se[j] > 0 else 1.0

        def gap(b: float) -> float:
            ll_c, _conv = _refit_with_constraint(
                X, y, j, b, firth=firth, init_beta=beta_hat,
                max_iter=max_iter, tol=tol,
            )
            return (ll_full - ll_c) - half_target  # zero crossing = CI bound

        # ----- Lower bound: search to the left of b_hat -----
        ci_low[j] = _bracket_and_solve(gap, anchor=b_hat, step=z_crit * se_j,
                                       direction=-1, max_expansions=8)
        # ----- Upper bound: search to the right of b_hat -----
        ci_high[j] = _bracket_and_solve(gap, anchor=b_hat, step=z_crit * se_j,
                                        direction=+1, max_expansions=8)

    return ci_low, ci_high


def _bracket_and_solve(
    gap: callable,
    anchor: float,
    step: float,
    direction: int,
    max_expansions: int = 8,
) -> float:
    """
    Find a zero crossing of ``gap`` starting from ``anchor`` and moving
    in ``direction`` (+/-1). ``step`` is the initial Wald-based step.

    Returns the bound; falls back to anchor +/- ``max_expansions * step``
    when no crossing is found (effectively an unbounded CI).
    """
    if step <= 0:
        step = 1.0
    a = anchor
    fa = gap(a)
    # gap(anchor) should be approximately -half_target (i.e. negative)
    b = anchor + direction * step
    fb = gap(b)
    expansions = 0
    while np.sign(fa) == np.sign(fb) and expansions < max_expansions:
        step *= 2.0
        b = anchor + direction * step
        fb = gap(b)
        expansions += 1
    if np.sign(fa) == np.sign(fb):
        # Could not bracket: report unbounded CI on this side
        return -np.inf if direction < 0 else np.inf
    try:
        if direction < 0:
            sol = optimize.brentq(gap, b, a, xtol=1e-4, rtol=1e-4, maxiter=50)
        else:
            sol = optimize.brentq(gap, a, b, xtol=1e-4, rtol=1e-4, maxiter=50)
        return float(sol)
    except (ValueError, RuntimeError):
        return -np.inf if direction < 0 else np.inf


# =============================================================================
# LIKELIHOOD-RATIO P-VALUES
# =============================================================================


def _likelihood_ratio_pvalues(
    X: np.ndarray,
    y: np.ndarray,
    beta_hat: np.ndarray,
    ll_full: float,
    *,
    firth: bool,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    Per-coefficient likelihood-ratio p-values for ``H_0: beta_j = 0``.

    For each j, refit the model with beta_j fixed at 0, get
    ``ll_restricted``, and report
    ``P(chi2_1 >= 2 * (ll_full - ll_restricted))``.
    """
    p = beta_hat.size
    pvals = np.empty(p)
    for j in range(p):
        ll_r, _conv = _refit_with_constraint(
            X, y, j, 0.0, firth=firth, init_beta=beta_hat,
            max_iter=max_iter, tol=tol,
        )
        lr = max(0.0, 2.0 * (ll_full - ll_r))
        pvals[j] = float(stats.chi2.sf(lr, df=1))
    return pvals


__all__ = [
    "LogitResult",
    "firth_logit",
    "mle_logit",
    "fit_logit",
    "is_design_ill_conditioned",
    "detect_separation_after_fit",
]
