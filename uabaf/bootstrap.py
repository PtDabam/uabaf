"""
uabaf.bootstrap
===============
BCa (Bias-Corrected and Accelerated) bootstrap confidence interval engine.

This module is the statistical core of UABAF. It wraps any fairness metric
function with a confidence interval, communicating how much uncertainty
exists in a point estimate — especially important on small datasets where
the same model can produce very different metric values across random splits.

Why BCa over a standard percentile bootstrap?
---------------------------------------------
A standard percentile bootstrap assumes the bootstrap distribution is
symmetric and unbiased. On small datasets with imbalanced groups this
assumption often fails — the distribution is skewed and the centre is
shifted away from the true value. BCa applies two corrections:

  z0  — bias correction  : shifts the percentile levels to account for
                            the fraction of bootstrap values below the
                            observed estimate.
  a   — acceleration     : adjusts for the rate at which the standard
                            error changes, estimated via jackknife
                            leave-one-out resampling.

Together these give intervals with coverage closer to the nominal 95%
on small and imbalanced samples.

Reference
---------
Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
Chapman & Hall. Chapter 14.
"""

import numpy as np
from scipy import stats

from .metrics import fairness_metrics

__all__ = [
    "bca_ci",
    "bca_ci_all_metrics",
]

# Default settings used throughout the package
DEFAULT_B      = 2000   # number of bootstrap resamples
DEFAULT_ALPHA  = 0.05   # significance level → 95% CI
METRICS        = ('dpd', 'eod', 'eop', 'di')


# ── Internal helpers ──────────────────────────────────────────────────────────

def _bootstrap_distribution(y_true, y_pred, sensitive,
                             metric_key, B, random_state):
    """
    Draw B bootstrap resamples and return the metric value for each.

    Parameters
    ----------
    y_true      : np.ndarray
    y_pred      : np.ndarray
    sensitive   : np.ndarray
    metric_key  : str — one of 'dpd', 'eod', 'eop', 'di'
    B           : int — number of resamples
    random_state: int

    Returns
    -------
    np.ndarray of length ≤ B (nan values are dropped)
    """
    rng = np.random.default_rng(random_state)
    n   = len(y_true)

    values = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        val = fairness_metrics(
            y_true[idx], y_pred[idx], sensitive[idx]
        )[metric_key]
        values.append(val)

    boot = np.array(values, dtype=float)
    return boot[~np.isnan(boot)]


def _jackknife_values(y_true, y_pred, sensitive, metric_key):
    """
    Leave-one-out jackknife estimates used to compute the acceleration
    factor (a) in the BCa formula.

    For each observation i, recomputes the metric with that observation
    removed. The spread of these n values captures how sensitive the
    metric is to individual data points.

    Returns
    -------
    np.ndarray of length n (may contain nan for degenerate leave-outs)
    """
    n = len(y_true)
    jack_vals = np.empty(n, dtype=float)

    for i in range(n):
        idx = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
        jack_vals[i] = fairness_metrics(
            y_true[idx], y_pred[idx], sensitive[idx]
        )[metric_key]

    return jack_vals


def _acceleration_factor(jack_vals):
    """
    Compute the BCa acceleration factor (a) from jackknife values.

    Formula
    -------
        a = Σ(θ̄ - θᵢ)³  /  [ 6 · (Σ(θ̄ - θᵢ)²)^(3/2) ]

    where θ̄ is the mean of the jackknife values and θᵢ is the
    leave-one-out estimate with observation i removed.

    Returns 0.0 if the denominator is zero (flat jackknife distribution).
    """
    valid = jack_vals[~np.isnan(jack_vals)]
    if len(valid) == 0:
        return 0.0

    mean   = np.mean(valid)
    diffs  = mean - valid
    num    = np.sum(diffs ** 3)
    denom  = 6.0 * (np.sum(diffs ** 2) ** 1.5)

    return float(num / denom) if denom != 0.0 else 0.0


def _adjusted_percentiles(z0, a, alpha):
    """
    Compute the BCa-adjusted percentile levels for the lower and upper
    bounds of the confidence interval.

    Parameters
    ----------
    z0    : float — bias-correction factor
    a     : float — acceleration factor
    alpha : float — significance level (e.g. 0.05 for 95% CI)

    Returns
    -------
    (p_lo, p_hi) : percentile levels in [0, 1]
    """
    z_lo = stats.norm.ppf(alpha / 2)        # e.g. -1.96 for alpha=0.05
    z_hi = stats.norm.ppf(1.0 - alpha / 2)  # e.g. +1.96

    def _adj(z_a):
        inner = z0 + (z0 + z_a) / (1.0 - a * (z0 + z_a))
        return float(stats.norm.cdf(inner))

    p_lo = np.clip(_adj(z_lo), 0.001, 0.999)
    p_hi = np.clip(_adj(z_hi), 0.001, 0.999)
    return p_lo, p_hi


# ── Public API ────────────────────────────────────────────────────────────────

def bca_ci(y_true, y_pred, sensitive, metric_key,
           B=DEFAULT_B, alpha=DEFAULT_ALPHA, random_state=42):
    """
    Compute a BCa bootstrap confidence interval for a single fairness metric.

    Parameters
    ----------
    y_true      : array-like, shape (n,) — true binary labels
    y_pred      : array-like, shape (n,) — predicted binary labels
    sensitive   : array-like, shape (n,) — binary sensitive attribute
    metric_key  : str — one of 'dpd', 'eod', 'eop', 'di'
    B           : int — number of bootstrap resamples (default 2000)
    alpha       : float — significance level (default 0.05 → 95% CI)
    random_state: int — random seed for reproducibility (default 42)

    Returns
    -------
    dict with keys:
        'point'  — observed metric value on the original data
        'ci_lo'  — lower bound of the BCa confidence interval
        'ci_hi'  — upper bound of the BCa confidence interval
        'width'  — ci_hi − ci_lo  (measure of uncertainty)
        'n'      — number of valid bootstrap resamples used
        'metric' — the metric_key passed in

    Notes
    -----
    - Returns np.nan for ci_lo, ci_hi, and width if fewer than 10
      valid bootstrap values could be computed (degenerate data).
    - The jackknife loop runs n leave-one-out passes which can be slow
      for large n. For exploratory work use B=500; use B=2000 for final
      results as per the UABAF methodology.

    Examples
    --------
    >>> import numpy as np
    >>> from uabaf.bootstrap import bca_ci
    >>> y_true = np.array([1,0,1,0,1,0,1,0,1,0,1,0])
    >>> y_pred = np.array([1,0,1,1,0,0,1,0,1,1,0,0])
    >>> s      = np.array([1,1,1,1,1,1,0,0,0,0,0,0])
    >>> result = bca_ci(y_true, y_pred, s, 'dpd', B=500)
    >>> print(result['point'], result['ci_lo'], result['ci_hi'])
    """
    y_true    = np.asarray(y_true,    dtype=float)
    y_pred    = np.asarray(y_pred,    dtype=float)
    sensitive = np.asarray(sensitive, dtype=float)

    # ── Point estimate ────────────────────────────────────────────────
    point = fairness_metrics(y_true, y_pred, sensitive)[metric_key]

    # ── Bootstrap distribution ────────────────────────────────────────
    boot_vals = _bootstrap_distribution(
        y_true, y_pred, sensitive, metric_key, B, random_state)

    if len(boot_vals) < 10:
        return {
            'metric': metric_key,
            'point' : point,
            'ci_lo' : np.nan,
            'ci_hi' : np.nan,
            'width' : np.nan,
            'n'     : len(boot_vals),
        }

    # ── Bias-correction factor z0 ─────────────────────────────────────
    # Proportion of bootstrap values strictly below the point estimate.
    # If z0 = 0 the bootstrap distribution is centred on the point estimate
    # (no bias). Values away from 0 shift the CI percentiles accordingly.
    prop_below = float(np.mean(boot_vals < point))
    # Clamp to avoid ppf(0) or ppf(1) = ±inf
    prop_below = np.clip(prop_below, 1e-6, 1 - 1e-6)
    z0 = float(stats.norm.ppf(prop_below))

    # ── Acceleration factor a (jackknife) ─────────────────────────────
    jack_vals = _jackknife_values(y_true, y_pred, sensitive, metric_key)
    a = _acceleration_factor(jack_vals)

    # ── BCa-adjusted percentile levels ────────────────────────────────
    p_lo, p_hi = _adjusted_percentiles(z0, a, alpha)

    ci_lo = float(np.percentile(boot_vals, 100.0 * p_lo))
    ci_hi = float(np.percentile(boot_vals, 100.0 * p_hi))
    width = ci_hi - ci_lo

    return {
        'metric': metric_key,
        'point' : float(point),
        'ci_lo' : round(ci_lo, 6),
        'ci_hi' : round(ci_hi, 6),
        'width' : round(width, 6),
        'n'     : len(boot_vals),
    }


def bca_ci_all_metrics(y_true, y_pred, sensitive,
                        B=DEFAULT_B, alpha=DEFAULT_ALPHA, random_state=42):
    """
    Compute BCa confidence intervals for all four UABAF fairness metrics.

    This is more efficient than calling bca_ci() four times separately
    because the bootstrap resampling loop computes all four metrics in
    a single pass over the B resamples, and the jackknife loop runs once
    and is reused across all four metrics.

    Parameters
    ----------
    y_true      : array-like, shape (n,) — true binary labels
    y_pred      : array-like, shape (n,) — predicted binary labels
    sensitive   : array-like, shape (n,) — binary sensitive attribute
    B           : int   — number of bootstrap resamples (default 2000)
    alpha       : float — significance level (default 0.05 → 95% CI)
    random_state: int   — random seed (default 42)

    Returns
    -------
    dict keyed by metric name, each value is a dict with keys:
        'point', 'ci_lo', 'ci_hi', 'width', 'n', 'metric'

    Examples
    --------
    >>> from uabaf.bootstrap import bca_ci_all_metrics
    >>> results = bca_ci_all_metrics(y_true, y_pred, s_test)
    >>> for metric, res in results.items():
    ...     print(f"{metric}: {res['point']:.4f}  CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]")
    """
    y_true    = np.asarray(y_true,    dtype=float)
    y_pred    = np.asarray(y_pred,    dtype=float)
    sensitive = np.asarray(sensitive, dtype=float)

    n   = len(y_true)
    rng = np.random.default_rng(random_state)

    # ── Single bootstrap pass — all four metrics at once ─────────────
    boot_all = {m: [] for m in METRICS}
    for _ in range(B):
        idx     = rng.integers(0, n, size=n)
        metrics = fairness_metrics(y_true[idx], y_pred[idx], sensitive[idx])
        for m in METRICS:
            boot_all[m].append(metrics[m])

    # Drop nans
    boot_arrays = {
        m: np.array(v, dtype=float) for m, v in boot_all.items()
    }
    boot_arrays = {
        m: arr[~np.isnan(arr)] for m, arr in boot_arrays.items()
    }

    # ── Single jackknife pass — reused for all four metrics ───────────
    jack_all = {m: [] for m in METRICS}
    for i in range(n):
        idx = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
        m_vals = fairness_metrics(
            y_true[idx], y_pred[idx], sensitive[idx])
        for m in METRICS:
            jack_all[m].append(m_vals[m])

    jack_arrays = {
        m: np.array(v, dtype=float) for m, v in jack_all.items()
    }

    # ── Assemble results ──────────────────────────────────────────────
    results = {}
    point_estimates = fairness_metrics(y_true, y_pred, sensitive)

    for m in METRICS:
        point     = point_estimates[m]
        boot_vals = boot_arrays[m]

        if len(boot_vals) < 10:
            results[m] = {
                'metric': m, 'point': float(point),
                'ci_lo': np.nan, 'ci_hi': np.nan,
                'width': np.nan, 'n': len(boot_vals),
            }
            continue

        prop_below = np.clip(float(np.mean(boot_vals < point)), 1e-6, 1 - 1e-6)
        z0 = float(stats.norm.ppf(prop_below))
        a  = _acceleration_factor(jack_arrays[m])
        p_lo, p_hi = _adjusted_percentiles(z0, a, alpha)

        ci_lo = float(np.percentile(boot_vals, 100.0 * p_lo))
        ci_hi = float(np.percentile(boot_vals, 100.0 * p_hi))

        results[m] = {
            'metric': m,
            'point' : round(float(point), 6),
            'ci_lo' : round(ci_lo, 6),
            'ci_hi' : round(ci_hi, 6),
            'width' : round(ci_hi - ci_lo, 6),
            'n'     : len(boot_vals),
        }

    return results
