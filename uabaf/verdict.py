"""
uabaf.verdict
=============
Fairness thresholds and uncertainty-aware verdict assignment.

This module translates numeric metric results from the bootstrap engine
into one of four verdict categories that communicate both whether a model
passes a fairness threshold AND how much confidence we have in that
conclusion given the width of the confidence interval.

Verdict categories
------------------
✅ PASS — High Confidence
    The metric is within the fair range AND the CI is narrow enough
    to trust the conclusion. The model is likely fair on this metric.

⚠️  PASS — Low Confidence
    The metric is within the fair range BUT the CI is wide. The model
    appears fair but the small dataset means we cannot be certain.
    Recommendation: collect more data before acting on this result.

❌ FAIL — High Confidence
    The metric breaches the fair range AND the CI is narrow. The bias
    finding is reliable. Action is warranted.

🔍 FAIL — Low Confidence
    The metric breaches the fair range BUT the CI is wide. The result
    is inconclusive — the true value could fall either side of the
    threshold. Do not act on this finding without more data.

Thresholds
----------
The thresholds below follow established fairness literature and legal
precedent (the 80% rule for Disparate Impact). They can be overridden
by passing a custom thresholds dict to assign_verdict() or
assign_all_verdicts() if the use case requires stricter or looser bounds.
"""

import numpy as np

__all__ = [
    "THRESHOLDS",
    "CI_WIDTH_THRESHOLD",
    "METRIC_LABELS",
    "assign_verdict",
    "assign_all_verdicts",
]


# ── Default thresholds ────────────────────────────────────────────────────────

THRESHOLDS = {
    # Difference metrics: fair if |value| ≤ threshold
    'dpd': ('diff',  0.10),
    'eod': ('diff',  0.10),
    'eop': ('diff',  0.10),
    # Ratio metric: fair if lo ≤ value ≤ hi  (the "80% rule")
    'di':  ('ratio', (0.80, 1.20)),
}

# A CI wider than this is considered too uncertain to act on confidently.
# 0.15 means the interval spans more than 15 percentage points.
CI_WIDTH_THRESHOLD = 0.15

# Human-readable labels used in reports and plots
METRIC_LABELS = {
    'dpd': 'Demographic Parity Difference',
    'eod': 'Equalized Odds Difference',
    'eop': 'Equal Opportunity Difference',
    'di' : 'Disparate Impact Ratio',
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _breaches_threshold(metric_key, point, thresholds):
    """
    Return True if the point estimate falls outside the fair range.

    Parameters
    ----------
    metric_key : str   — one of 'dpd', 'eod', 'eop', 'di'
    point      : float — observed metric value
    thresholds : dict  — threshold definitions (defaults to THRESHOLDS)

    Returns
    -------
    bool — True means the metric fails the fairness check
    """
    if np.isnan(point):
        return False  # cannot make a judgment on missing data

    kind, threshold = thresholds[metric_key]

    if kind == 'diff':
        return abs(point) > threshold
    elif kind == 'ratio':
        lo, hi = threshold
        return point < lo or point > hi
    return False


def _confidence_level(width, ci_width_threshold):
    """
    Classify the CI width as HIGH or LOW confidence.

    Parameters
    ----------
    width              : float — ci_hi − ci_lo
    ci_width_threshold : float — boundary between HIGH and LOW confidence

    Returns
    -------
    str — 'HIGH' or 'LOW'
    """
    if np.isnan(width):
        return 'LOW'
    return 'HIGH' if width <= ci_width_threshold else 'LOW'


# ── Public API ────────────────────────────────────────────────────────────────

def assign_verdict(metric_key, point, ci_lo, ci_hi,
                   thresholds=None, ci_width_threshold=None):
    """
    Assign an uncertainty-aware verdict for a single metric result.

    Parameters
    ----------
    metric_key         : str   — one of 'dpd', 'eod', 'eop', 'di'
    point              : float — observed metric value
    ci_lo              : float — lower bound of the BCa CI
    ci_hi              : float — upper bound of the BCa CI
    thresholds         : dict  — override default THRESHOLDS if needed
    ci_width_threshold : float — override default CI_WIDTH_THRESHOLD

    Returns
    -------
    dict with keys:
        'verdict'     — full verdict string with emoji
        'pass_fail'   — 'PASS' or 'FAIL'
        'confidence'  — 'HIGH' or 'LOW'
        'breaches'    — bool, True if point estimate fails threshold
        'width'       — CI width (ci_hi − ci_lo)
        'metric'      — metric_key passed in
        'label'       — human-readable metric name

    Examples
    --------
    >>> from uabaf.verdict import assign_verdict
    >>> assign_verdict('dpd', point=0.05, ci_lo=0.01, ci_hi=0.09)
    {'verdict': '✅ PASS — High Confidence', 'pass_fail': 'PASS', ...}

    >>> assign_verdict('dpd', point=0.14, ci_lo=0.02, ci_hi=0.26)
    {'verdict': '🔍 FAIL — Low Confidence', 'pass_fail': 'FAIL', ...}
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    if ci_width_threshold is None:
        ci_width_threshold = CI_WIDTH_THRESHOLD

    width    = ci_hi - ci_lo if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else np.nan
    breaches = _breaches_threshold(metric_key, point, thresholds)
    conf     = _confidence_level(width, ci_width_threshold)

    if not breaches and conf == 'HIGH':
        verdict = '✅ PASS — High Confidence'
    elif not breaches and conf == 'LOW':
        verdict = '⚠️  PASS — Low Confidence'
    elif breaches and conf == 'HIGH':
        verdict = '❌ FAIL — High Confidence'
    else:
        verdict = '🔍 FAIL — Low Confidence'

    return {
        'metric'    : metric_key,
        'label'     : METRIC_LABELS.get(metric_key, metric_key),
        'verdict'   : verdict,
        'pass_fail' : 'PASS' if not breaches else 'FAIL',
        'confidence': conf,
        'breaches'  : breaches,
        'width'     : round(float(width), 6) if not np.isnan(width) else np.nan,
    }


def assign_all_verdicts(bootstrap_results,
                        thresholds=None, ci_width_threshold=None):
    """
    Assign verdicts for all metrics from a bca_ci_all_metrics() result.

    This is the main function called by the auditor — it takes the full
    output of bca_ci_all_metrics() and returns a verdict for each metric
    in a single call.

    Parameters
    ----------
    bootstrap_results  : dict — output of bca_ci_all_metrics(), keyed
                                by metric name
    thresholds         : dict  — override default THRESHOLDS if needed
    ci_width_threshold : float — override default CI_WIDTH_THRESHOLD

    Returns
    -------
    dict keyed by metric name, each value is the dict returned by
    assign_verdict() plus the point estimate and CI bounds:
        'point', 'ci_lo', 'ci_hi', 'width',
        'verdict', 'pass_fail', 'confidence', 'breaches', 'label'

    Examples
    --------
    >>> from uabaf.bootstrap import bca_ci_all_metrics
    >>> from uabaf.verdict import assign_all_verdicts
    >>> ci_results = bca_ci_all_metrics(y_true, y_pred, sensitive)
    >>> verdicts = assign_all_verdicts(ci_results)
    >>> for m, v in verdicts.items():
    ...     print(f"{v['label']}: {v['verdict']}")
    """
    verdicts = {}
    for metric_key, res in bootstrap_results.items():
        v = assign_verdict(
            metric_key  = metric_key,
            point       = res['point'],
            ci_lo       = res['ci_lo'],
            ci_hi       = res['ci_hi'],
            thresholds  = thresholds,
            ci_width_threshold = ci_width_threshold,
        )
        # Merge in the numeric values for convenient access downstream
        v.update({
            'point': res['point'],
            'ci_lo': res['ci_lo'],
            'ci_hi': res['ci_hi'],
        })
        verdicts[metric_key] = v

    return verdicts
