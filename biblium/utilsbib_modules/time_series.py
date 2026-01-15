# -*- coding: utf-8 -*-
"""
Time Series Analysis Utilities.

This module provides functions for temporal analysis of bibliometric data:
- Publication trends
- Citation evolution
- Growth rate analysis
- Trend prediction
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def compute_publication_trend(
    df: pd.DataFrame,
    year_col: str = "Year",
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute publication counts by year.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic data.
    year_col : str
        Column containing publication years.
    min_year : int, optional
        Start year (inclusive).
    max_year : int, optional
        End year (inclusive).
        
    Returns
    -------
    pd.DataFrame
        Year, Count, Cumulative, Growth Rate columns.
    """
    years = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    
    if min_year is None:
        min_year = years.min()
    if max_year is None:
        max_year = years.max()
    
    # Count by year
    counts = years.value_counts().reindex(
        range(min_year, max_year + 1), fill_value=0
    ).sort_index()
    
    result = pd.DataFrame({
        "Year": counts.index,
        "Count": counts.values,
    })
    
    # Add cumulative
    result["Cumulative"] = result["Count"].cumsum()
    
    # Add growth rate
    result["Growth Rate (%)"] = result["Count"].pct_change() * 100
    result["Growth Rate (%)"] = result["Growth Rate (%)"].round(2)
    
    return result


def compute_citation_evolution(
    df: pd.DataFrame,
    year_col: str = "Year",
    cite_col: str = "Cited by",
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute citation metrics by publication year.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic data.
    year_col : str
        Column containing publication years.
    cite_col : str
        Column containing citation counts.
    min_year : int, optional
        Start year (inclusive).
    max_year : int, optional
        End year (inclusive).
        
    Returns
    -------
    pd.DataFrame
        Year, Total Citations, Mean Citations, Median Citations, etc.
    """
    work = df.copy()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work[cite_col] = pd.to_numeric(work[cite_col], errors="coerce").fillna(0)
    
    work = work.dropna(subset=[year_col])
    work[year_col] = work[year_col].astype(int)
    
    if min_year:
        work = work[work[year_col] >= min_year]
    if max_year:
        work = work[work[year_col] <= max_year]
    
    # Group by year
    grouped = work.groupby(year_col)[cite_col].agg([
        ("Documents", "count"),
        ("Total Citations", "sum"),
        ("Mean Citations", "mean"),
        ("Median Citations", "median"),
        ("Max Citations", "max"),
    ]).reset_index()
    
    grouped = grouped.rename(columns={year_col: "Year"})
    grouped["Mean Citations"] = grouped["Mean Citations"].round(2)
    
    return grouped


def compute_growth_rates(
    trend_df: pd.DataFrame,
    count_col: str = "Count",
    year_col: str = "Year",
) -> pd.DataFrame:
    """
    Compute various growth rate metrics.
    
    Parameters
    ----------
    trend_df : pd.DataFrame
        Publication trend DataFrame.
    count_col : str
        Column with counts.
    year_col : str
        Column with years.
        
    Returns
    -------
    pd.DataFrame
        Growth metrics including CAGR, average growth, etc.
    """
    if len(trend_df) < 2:
        return pd.DataFrame()
    
    counts = trend_df[count_col].values
    years = trend_df[year_col].values
    
    # Year-over-year growth
    yoy_growth = np.diff(counts) / np.where(counts[:-1] == 0, 1, counts[:-1]) * 100
    
    # CAGR (Compound Annual Growth Rate)
    n_years = len(years) - 1
    if counts[0] > 0 and counts[-1] > 0 and n_years > 0:
        cagr = (np.power(counts[-1] / counts[0], 1 / n_years) - 1) * 100
    else:
        cagr = np.nan
    
    # Moving average growth
    window = min(3, len(counts))
    ma = pd.Series(counts).rolling(window).mean()
    ma_growth = ma.pct_change() * 100
    
    result = pd.DataFrame({
        "Year": years[1:],
        "YoY Growth (%)": yoy_growth.round(2),
        "3-Year MA": ma.values[1:],
        "MA Growth (%)": ma_growth.values[1:],
    })
    
    # Add summary row
    summary = pd.DataFrame([{
        "Year": "Summary",
        "YoY Growth (%)": np.nanmean(yoy_growth),
        "3-Year MA": np.nan,
        "MA Growth (%)": np.nanmean(ma_growth.dropna()),
        "CAGR (%)": cagr,
    }])
    
    result = pd.concat([result, summary], ignore_index=True)
    
    return result


def predict_future_publications(
    trend_df: pd.DataFrame,
    n_years: int = 5,
    method: str = "linear",
    count_col: str = "Count",
    year_col: str = "Year",
) -> pd.DataFrame:
    """
    Predict future publication counts.
    
    Parameters
    ----------
    trend_df : pd.DataFrame
        Historical publication trend.
    n_years : int
        Number of years to predict.
    method : str
        Prediction method: 'linear', 'exponential', 'polynomial'.
    count_col : str
        Column with counts.
    year_col : str
        Column with years.
        
    Returns
    -------
    pd.DataFrame
        Predictions with confidence intervals.
    """
    years = trend_df[year_col].values
    counts = trend_df[count_col].values
    
    # Fit model
    if method == "linear":
        coeffs = np.polyfit(years, counts, 1)
        predict_func = np.poly1d(coeffs)
    elif method == "exponential":
        # Log-linear regression
        log_counts = np.log(counts + 1)
        coeffs = np.polyfit(years, log_counts, 1)
        predict_func = lambda x: np.exp(np.poly1d(coeffs)(x)) - 1
    elif method == "polynomial":
        coeffs = np.polyfit(years, counts, 2)
        predict_func = np.poly1d(coeffs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Generate predictions
    last_year = int(years[-1])
    future_years = np.arange(last_year + 1, last_year + n_years + 1)
    predictions = predict_func(future_years)
    
    # Ensure non-negative
    predictions = np.maximum(predictions, 0)
    
    # Simple confidence interval (based on historical variance)
    residuals = counts - predict_func(years)
    std_err = np.std(residuals)
    
    result = pd.DataFrame({
        "Year": future_years,
        "Predicted": predictions.round(0).astype(int),
        "Lower 95%": np.maximum(predictions - 1.96 * std_err, 0).round(0).astype(int),
        "Upper 95%": (predictions + 1.96 * std_err).round(0).astype(int),
    })
    
    return result


def compute_doubling_time(
    trend_df: pd.DataFrame,
    count_col: str = "Count",
    year_col: str = "Year",
) -> float:
    """
    Compute the doubling time of publications.
    
    Parameters
    ----------
    trend_df : pd.DataFrame
        Publication trend DataFrame.
    count_col : str
        Column with counts.
    year_col : str
        Column with years.
        
    Returns
    -------
    float
        Estimated doubling time in years.
    """
    cumulative = trend_df[count_col].cumsum().values
    years = trend_df[year_col].values
    
    if cumulative[-1] <= cumulative[0] * 2:
        return np.inf  # Hasn't doubled yet
    
    # Find when it doubled
    target = cumulative[0] * 2
    for i, cum in enumerate(cumulative):
        if cum >= target:
            if i == 0:
                return 0
            # Interpolate
            frac = (target - cumulative[i-1]) / (cum - cumulative[i-1])
            return (years[i] - years[i-1]) * frac + (years[i-1] - years[0])
    
    return np.inf


def compute_half_life(
    df: pd.DataFrame,
    year_col: str = "Year",
    cite_col: str = "Cited by",
    current_year: Optional[int] = None,
) -> float:
    """
    Compute the citation half-life (median age of cited papers).
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic data.
    year_col : str
        Column containing publication years.
    cite_col : str
        Column containing citation counts.
    current_year : int, optional
        Reference year. Defaults to max year in data.
        
    Returns
    -------
    float
        Citation half-life in years.
    """
    work = df.copy()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work[cite_col] = pd.to_numeric(work[cite_col], errors="coerce").fillna(0)
    work = work.dropna(subset=[year_col])
    
    if current_year is None:
        current_year = int(work[year_col].max())
    
    work["Age"] = current_year - work[year_col].astype(int)
    
    # Weight ages by citations
    total_cites = work[cite_col].sum()
    if total_cites == 0:
        return np.nan
    
    # Find median age (weighted)
    work = work.sort_values("Age")
    work["Cumulative Citations"] = work[cite_col].cumsum()
    
    half_total = total_cites / 2
    
    for _, row in work.iterrows():
        if row["Cumulative Citations"] >= half_total:
            return row["Age"]
    
    return work["Age"].max()


def detect_trend_changes(
    trend_df: pd.DataFrame,
    count_col: str = "Count",
    year_col: str = "Year",
    sensitivity: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Detect significant changes in publication trends.
    
    Parameters
    ----------
    trend_df : pd.DataFrame
        Publication trend DataFrame.
    count_col : str
        Column with counts.
    year_col : str
        Column with years.
    sensitivity : float
        Standard deviations for significance threshold.
        
    Returns
    -------
    list
        List of change points with year and type (increase/decrease).
    """
    years = trend_df[year_col].values
    counts = trend_df[count_col].values
    
    if len(counts) < 3:
        return []
    
    # Compute year-over-year changes
    changes = np.diff(counts)
    
    # Compute thresholds
    mean_change = np.mean(changes)
    std_change = np.std(changes)
    
    if std_change == 0:
        return []
    
    # Detect significant changes
    change_points = []
    for i, change in enumerate(changes):
        z_score = (change - mean_change) / std_change
        
        if abs(z_score) > sensitivity:
            change_points.append({
                "Year": int(years[i + 1]),
                "Type": "increase" if change > 0 else "decrease",
                "Magnitude": abs(change),
                "Z-Score": round(z_score, 2),
            })
    
    return change_points


# =============================================================================
# GROWTH MODEL FITTING
# =============================================================================

def fit_growth_model(
    df: pd.DataFrame,
    year_col: str = "Year",
    model_type: str = "auto",
    forecast_years: int = 5,
    min_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fit bibliometric growth models to publication data.
    
    Fits exponential, logistic, power law, or linear models to annual publication
    counts and provides forecasts. Models are selected based on AIC if 'auto' is chosen.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe with year column.
    year_col : str
        Column containing publication years.
    model_type : str
        Model type: "exponential", "logistic", "power", "linear", "auto".
        If "auto", selects best model based on AIC.
    forecast_years : int
        Number of years to forecast beyond the data.
    min_year : int, optional
        Minimum year to include. If None, uses earliest year >= 1900.
    
    Returns
    -------
    dict
        Dictionary containing:
        - model_type: str - The fitted model type
        - parameters: dict - Model parameters
        - r_squared: float - Coefficient of determination
        - aic: float - Akaike Information Criterion
        - bic: float - Bayesian Information Criterion
        - growth_rate: float - Annual growth rate
        - doubling_time: float or None - Time to double (for exponential/logistic)
        - prediction_df: pd.DataFrame - Historical + forecast data
        - comparison_df: pd.DataFrame - Comparison of all models (if auto)
    """
    from scipy.optimize import curve_fit
    
    # Count papers per year
    years_series = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    year_counts = years_series.value_counts().sort_index()
    years = np.array(year_counts.index)
    counts = np.array(year_counts.values)
    
    # Filter to reasonable years
    if min_year is None:
        min_year = 1900
    mask = years >= min_year
    years = years[mask]
    counts = counts[mask]
    
    if len(years) < 4:
        raise ValueError("Not enough data points for growth model fitting (need at least 4 years)")
    
    # Normalize years for fitting (helps numerical stability)
    year_offset = years.min()
    t = years - year_offset
    
    # Define growth models
    def exponential(t, a, b):
        """Exponential: y = a * exp(b * t)"""
        return a * np.exp(b * t)
    
    def logistic(t, L, k, t0):
        """Logistic: y = L / (1 + exp(-k * (t - t0)))"""
        return L / (1 + np.exp(-k * (t - t0)))
    
    def power_law(t, a, b):
        """Power law: y = a * (t + 1)^b"""
        return a * (t + 1) ** b
    
    def linear(t, a, b):
        """Linear: y = a * t + b"""
        return a * t + b
    
    # Model specifications: (function, initial_params, bounds)
    models = {
        "exponential": (
            exponential, 
            [max(1, counts[0]), 0.05], 
            [(0.001, None), (0.001, 0.5)]
        ),
        "logistic": (
            logistic, 
            [max(counts) * 2, 0.1, len(t) / 2], 
            [(max(counts), max(counts) * 10), (0.01, 1), (0, len(t) * 2)]
        ),
        "power": (
            power_law, 
            [max(1, counts[0]), 1.5], 
            [(0.001, None), (0.1, 5)]
        ),
        "linear": (
            linear, 
            [np.mean(np.diff(counts)) if len(counts) > 1 else 1, counts[0]], 
            [(-np.inf, np.inf), (-np.inf, np.inf)]
        ),
    }
    
    def _fit_model(name: str) -> Optional[Dict]:
        """Fit a single model and return results."""
        func, p0, bounds_list = models[name]
        bounds_lower = [b[0] if b[0] is not None else -np.inf for b in bounds_list]
        bounds_upper = [b[1] if b[1] is not None else np.inf for b in bounds_list]
        
        try:
            popt, pcov = curve_fit(
                func, t, counts, 
                p0=p0, 
                bounds=(bounds_lower, bounds_upper), 
                maxfev=10000
            )
            fitted = func(t, *popt)
            residuals = counts - fitted
            
            # Metrics
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((counts - np.mean(counts)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            n = len(counts)
            k = len(popt)
            aic = n * np.log(ss_res / n + 1e-10) + 2 * k
            bic = n * np.log(ss_res / n + 1e-10) + k * np.log(n)
            rmse = np.sqrt(ss_res / n)
            
            return {
                "name": name,
                "popt": popt,
                "fitted": fitted,
                "residuals": residuals,
                "r_squared": r_squared,
                "aic": aic,
                "bic": bic,
                "rmse": rmse,
            }
        except Exception:
            return None
    
    # Fit all models for comparison
    all_fits = {}
    for name in models.keys():
        result = _fit_model(name)
        if result is not None:
            all_fits[name] = result
    
    if not all_fits:
        raise ValueError("Could not fit any growth model to the data")
    
    # Select model
    if model_type == "auto":
        # Select by lowest AIC
        selected = min(all_fits.values(), key=lambda x: x["aic"])
        model_type = selected["name"]
    else:
        if model_type not in all_fits:
            # Fallback to linear if requested model failed
            model_type = "linear" if "linear" in all_fits else list(all_fits.keys())[0]
        selected = all_fits[model_type]
    
    # Extract parameters with meaningful names
    func, _, _ = models[model_type]
    popt = selected["popt"]
    
    param_names = {
        "exponential": ["initial_value", "growth_rate"],
        "logistic": ["carrying_capacity", "growth_rate", "midpoint_year"],
        "power": ["coefficient", "exponent"],
        "linear": ["slope", "intercept"],
    }
    parameters = dict(zip(param_names[model_type], popt))
    
    # Adjust midpoint year for logistic to actual year
    if model_type == "logistic":
        parameters["midpoint_year"] = parameters["midpoint_year"] + year_offset
    
    # Calculate growth rate and doubling time
    if model_type == "exponential":
        growth_rate = popt[1]  # b parameter
        doubling_time = np.log(2) / growth_rate if growth_rate > 0 else None
    elif model_type == "logistic":
        growth_rate = popt[1]  # k parameter
        doubling_time = np.log(2) / growth_rate if growth_rate > 0 else None
    elif model_type == "linear":
        # Average growth rate relative to mean
        growth_rate = popt[0] / np.mean(counts) if np.mean(counts) > 0 else 0
        doubling_time = None
    elif model_type == "power":
        growth_rate = popt[1]  # exponent
        doubling_time = None
    else:
        growth_rate = 0
        doubling_time = None
    
    # Generate predictions (historical + forecast)
    future_t = np.arange(t[-1] + 1, t[-1] + 1 + forecast_years)
    future_years = future_t + year_offset
    future_fitted = func(future_t, *popt)
    
    # Ensure non-negative predictions
    future_fitted = np.maximum(future_fitted, 0)
    
    prediction_df = pd.DataFrame({
        "Year": np.concatenate([years, future_years]),
        "Observed": np.concatenate([counts, [np.nan] * forecast_years]),
        "Fitted": np.concatenate([selected["fitted"], future_fitted]),
        "Is Forecast": [False] * len(years) + [True] * forecast_years,
    })
    prediction_df["Fitted"] = prediction_df["Fitted"].round(1)
    
    # Model comparison dataframe
    comparison_df = pd.DataFrame([
        {
            "Model": name,
            "R²": round(fit["r_squared"], 4),
            "AIC": round(fit["aic"], 2),
            "BIC": round(fit["bic"], 2),
            "RMSE": round(fit["rmse"], 2),
        }
        for name, fit in all_fits.items()
    ]).sort_values("AIC")
    
    return {
        "model_type": model_type,
        "parameters": parameters,
        "r_squared": selected["r_squared"],
        "aic": selected["aic"],
        "bic": selected["bic"],
        "rmse": selected["rmse"],
        "growth_rate": growth_rate,
        "doubling_time": doubling_time,
        "fitted_values": selected["fitted"],
        "residuals": selected["residuals"],
        "prediction_df": prediction_df,
        "comparison_df": comparison_df,
        "years": years,
        "counts": counts,
    }


def fit_life_cycle_model(
    df: pd.DataFrame,
    year_col: str = "Year",
    forecast_years: int = 10,
) -> Dict[str, Any]:
    """
    Fit a life cycle (logistic S-curve) model to cumulative publication data.
    
    This is useful for analyzing the maturity of a research field. The logistic
    model estimates when a field will reach saturation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    year_col : str
        Column containing publication years.
    forecast_years : int
        Years to forecast.
    
    Returns
    -------
    dict
        Dictionary containing:
        - saturation_k: float - Carrying capacity (max cumulative pubs)
        - peak_year: float - Inflection point year (50% of saturation)
        - growth_rate: float - Intrinsic growth rate
        - growth_duration: float - Years from 10% to 90% saturation
        - current_phase: str - "emerging", "growth", "maturity", or "saturation"
        - progress: float - Current progress to saturation (0-1)
        - r_squared: float - Model fit quality
        - prediction_df: pd.DataFrame - Historical + forecast
    """
    from scipy.optimize import curve_fit
    
    # Count papers per year
    years_series = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    year_counts = years_series.value_counts().sort_index()
    years = np.array(year_counts.index)
    counts = np.array(year_counts.values)
    
    # Filter to reasonable years
    mask = years >= 1900
    years = years[mask]
    counts = counts[mask]
    
    # Compute cumulative counts
    cumulative = np.cumsum(counts)
    
    if len(years) < 4:
        raise ValueError("Not enough data points for life cycle analysis")
    
    # Normalize years
    year_offset = years.min()
    t = years - year_offset
    
    # Logistic model for cumulative data
    def logistic_cumulative(t, K, r, Tm):
        """Cumulative logistic: y = K / (1 + exp(-r * (t - Tm)))"""
        return K / (1 + np.exp(-r * (t - Tm)))
    
    # Initial estimates
    K0 = cumulative[-1] * 2  # Carrying capacity guess
    r0 = 0.2  # Growth rate guess
    Tm0 = t[len(t) // 2]  # Midpoint guess
    
    try:
        popt, pcov = curve_fit(
            logistic_cumulative, t, cumulative,
            p0=[K0, r0, Tm0],
            bounds=([cumulative[-1], 0.01, 0], [cumulative[-1] * 20, 2, len(t) * 3]),
            maxfev=10000
        )
    except Exception as e:
        raise ValueError(f"Could not fit life cycle model: {e}")
    
    K, r, Tm = popt
    
    # Fitted values
    fitted = logistic_cumulative(t, K, r, Tm)
    residuals = cumulative - fitted
    
    # Metrics
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((cumulative - np.mean(cumulative)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Growth duration (time from 10% to 90% of K)
    # At 10%: t_10 = Tm - ln(9)/r
    # At 90%: t_90 = Tm + ln(9)/r
    growth_duration = 2 * np.log(9) / r
    
    # Current progress
    progress = cumulative[-1] / K
    
    # Determine phase
    if progress < 0.1:
        current_phase = "emerging"
    elif progress < 0.5:
        current_phase = "growth"
    elif progress < 0.9:
        current_phase = "maturity"
    else:
        current_phase = "saturation"
    
    # Peak annual production (derivative at inflection point)
    peak_annual = K * r / 4
    
    # Forecast
    future_t = np.arange(t[-1] + 1, t[-1] + 1 + forecast_years)
    future_years = future_t + year_offset
    future_cumulative = logistic_cumulative(future_t, K, r, Tm)
    
    # Convert cumulative forecast to annual
    all_cumulative = np.concatenate([fitted, future_cumulative])
    all_annual = np.diff(np.concatenate([[0], all_cumulative]))
    
    prediction_df = pd.DataFrame({
        "Year": np.concatenate([years, future_years]),
        "Observed": np.concatenate([counts, [np.nan] * forecast_years]),
        "Observed Cumulative": np.concatenate([cumulative, [np.nan] * forecast_years]),
        "Fitted Annual": all_annual,
        "Fitted Cumulative": np.concatenate([fitted, future_cumulative]),
        "Is Forecast": [False] * len(years) + [True] * forecast_years,
    })
    
    return {
        "saturation_k": K,
        "peak_year": Tm + year_offset,
        "growth_rate": r,
        "growth_duration": growth_duration,
        "peak_annual": peak_annual,
        "current_phase": current_phase,
        "progress": progress,
        "r_squared": r_squared,
        "prediction_df": prediction_df,
        "years": years,
        "cumulative": cumulative,
        "fitted_cumulative": fitted,
    }
