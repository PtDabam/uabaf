"""
uabaf.metrics
=============
Fairness metric functions for binary classification models.

All functions operate on numpy arrays and are stateless — they take
predictions and group labels as inputs and return scalar values.
This makes them safe to call inside bootstrap resampling loops.

Metrics implemented
-------------------
- Demographic Parity Difference  (DPD)
- Equalized Odds Difference       (EOD)
- Equal Opportunity Difference    (EOP)
- Disparate Impact Ratio          (DI)

Conventions used throughout
----------------------------
- sensitive : binary array, 1 = privileged group, 0 = unprivileged group
- y_true    : binary array, 1 = positive outcome, 0 = negative outcome
- y_pred    : binary array, 1 = predicted positive, 0 = predicted negative
- All differences are computed as  privileged − unprivileged  so a
  positive value means the privileged group is favoured.
"""

import numpy as np

__all__ = [
    "demographic_parity_diff",
    "equalized_odds_diff",
    "equal_opportunity_diff",
    "disparate_impact_ratio",
    "fairness_metrics",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _positive_rate(y_pred, mask):
    """
    Fraction of predictions that are positive within a subgroup.

    Parameters
    ----------
    y_pred : np.ndarray  — predicted labels (0/1)
    mask   : np.ndarray  — boolean mask selecting the subgroup

    Returns
    -------
    float or np.nan if the subgroup is empty
    """
    sub = y_pred[mask]
    if len(sub) == 0:
        return np.nan
    return float(np.mean(sub == 1))


def _true_positive_rate(y_true, y_pred, mask):
    """
    TPR (recall) within a subgroup: TP / (TP + FN).
    Returns np.nan if the subgroup has no actual positives.
    """
    sub_true = y_true[mask]
    sub_pred = y_pred[mask]
    pos_mask = sub_true == 1
    if pos_mask.sum() == 0:
        return np.nan
    return float(np.mean(sub_pred[pos_mask] == 1))


def _false_positive_rate(y_true, y_pred, mask):
    """
    FPR within a subgroup: FP / (FP + TN).
    Returns np.nan if the subgroup has no actual negatives.
    """
    sub_true = y_true[mask]
    sub_pred = y_pred[mask]
    neg_mask = sub_true == 0
    if neg_mask.sum() == 0:
        return np.nan
    return float(np.mean(sub_pred[neg_mask] == 1))


def _group_masks(sensitive):
    """
    Return boolean masks for the privileged (1) and unprivileged (0) groups.
    """
    privileged   = sensitive == 1
    unprivileged = sensitive == 0
    return privileged, unprivileged


# ── Public metric functions ───────────────────────────────────────────────────

def demographic_parity_diff(y_true, y_pred, sensitive):
    """
    Demographic Parity Difference (DPD).

    Measures whether both groups receive positive predictions at equal rates,
    regardless of the true label.

    Formula
    -------
        DPD = P(ŷ=1 | S=1) − P(ŷ=1 | S=0)

    Ideal value : 0.0
    Fair range  : |DPD| ≤ 0.10
    Sign        : positive → privileged group favoured

    Parameters
    ----------
    y_true    : array-like, shape (n,)  — true binary labels (unused here
                but kept for a consistent signature across all metrics)
    y_pred    : array-like, shape (n,)  — predicted binary labels
    sensitive : array-like, shape (n,)  — binary sensitive attribute

    Returns
    -------
    float — DPD value, or np.nan if either group is empty
    """
    y_true    = np.asarray(y_true)
    y_pred    = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    priv, unpriv = _group_masks(sensitive)
    pr_priv   = _positive_rate(y_pred, priv)
    pr_unpriv = _positive_rate(y_pred, unpriv)

    if np.isnan(pr_priv) or np.isnan(pr_unpriv):
        return np.nan
    return pr_priv - pr_unpriv


def equalized_odds_diff(y_true, y_pred, sensitive):
    """
    Equalized Odds Difference (EOD).

    Measures whether both groups have equal TPR *and* equal FPR.
    Takes the larger of the two differences so that a single number
    captures the worst violation across both rates.

    Formula
    -------
        EOD = max( |TPR_priv − TPR_unpriv|,  |FPR_priv − FPR_unpriv| )

    Ideal value : 0.0
    Fair range  : EOD ≤ 0.10

    Parameters
    ----------
    y_true    : array-like, shape (n,)  — true binary labels
    y_pred    : array-like, shape (n,)  — predicted binary labels
    sensitive : array-like, shape (n,)  — binary sensitive attribute

    Returns
    -------
    float — EOD value (always ≥ 0), or np.nan if data is insufficient
    """
    y_true    = np.asarray(y_true)
    y_pred    = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    priv, unpriv = _group_masks(sensitive)

    tpr_priv   = _true_positive_rate(y_true, y_pred, priv)
    tpr_unpriv = _true_positive_rate(y_true, y_pred, unpriv)
    fpr_priv   = _false_positive_rate(y_true, y_pred, priv)
    fpr_unpriv = _false_positive_rate(y_true, y_pred, unpriv)

    tpr_diff = abs(tpr_priv - tpr_unpriv)
    fpr_diff = abs(fpr_priv - fpr_unpriv)

    if np.isnan(tpr_diff) and np.isnan(fpr_diff):
        return np.nan
    return float(np.nanmax([tpr_diff, fpr_diff]))


def equal_opportunity_diff(y_true, y_pred, sensitive):
    """
    Equal Opportunity Difference (EOP).

    A relaxed version of Equalized Odds that focuses only on the
    True Positive Rate — ensuring both groups are equally likely to
    receive a positive prediction when the true label is positive.

    Formula
    -------
        EOP = |TPR_priv − TPR_unpriv|

    Ideal value : 0.0
    Fair range  : EOP ≤ 0.10

    Parameters
    ----------
    y_true    : array-like, shape (n,)  — true binary labels
    y_pred    : array-like, shape (n,)  — predicted binary labels
    sensitive : array-like, shape (n,)  — binary sensitive attribute

    Returns
    -------
    float — EOP value (always ≥ 0), or np.nan if data is insufficient
    """
    y_true    = np.asarray(y_true)
    y_pred    = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    priv, unpriv = _group_masks(sensitive)

    tpr_priv   = _true_positive_rate(y_true, y_pred, priv)
    tpr_unpriv = _true_positive_rate(y_true, y_pred, unpriv)

    if np.isnan(tpr_priv) or np.isnan(tpr_unpriv):
        return np.nan
    return abs(tpr_priv - tpr_unpriv)


def disparate_impact_ratio(y_true, y_pred, sensitive):
    """
    Disparate Impact Ratio (DI).

    Ratio of the positive prediction rate of the unprivileged group
    to that of the privileged group. The 80% rule (DI ≥ 0.80) is the
    most widely cited legal threshold in employment discrimination law.

    Formula
    -------
        DI = P(ŷ=1 | S=0) / P(ŷ=1 | S=1)

    Ideal value : 1.0  (both groups have equal positive rates)
    Fair range  : 0.80 ≤ DI ≤ 1.20
    Sign        : < 1.0 → unprivileged group receives fewer positive predictions

    Parameters
    ----------
    y_true    : array-like, shape (n,)  — true binary labels (unused here
                but kept for a consistent signature across all metrics)
    y_pred    : array-like, shape (n,)  — predicted binary labels
    sensitive : array-like, shape (n,)  — binary sensitive attribute

    Returns
    -------
    float — DI value, or np.nan if either group is empty or PR_priv is 0
    """
    y_true    = np.asarray(y_true)
    y_pred    = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    priv, unpriv = _group_masks(sensitive)
    pr_priv   = _positive_rate(y_pred, priv)
    pr_unpriv = _positive_rate(y_pred, unpriv)

    if np.isnan(pr_priv) or np.isnan(pr_unpriv):
        return np.nan
    if pr_priv == 0.0:
        # Privileged group receives no positive predictions at all —
        # DI is undefined; return nan rather than divide by zero.
        return np.nan
    return pr_unpriv / pr_priv


def fairness_metrics(y_true, y_pred, sensitive):
    """
    Compute all four UABAF fairness metrics in a single call.

    This is the main function called by the bootstrap engine and the
    auditor — it returns a dict so that a single resample pass produces
    all four values at once, avoiding four separate resampling loops.

    Parameters
    ----------
    y_true    : array-like, shape (n,)  — true binary labels
    y_pred    : array-like, shape (n,)  — predicted binary labels
    sensitive : array-like, shape (n,)  — binary sensitive attribute
                (1 = privileged, 0 = unprivileged)

    Returns
    -------
    dict with keys:
        'dpd' — Demographic Parity Difference
        'eod' — Equalized Odds Difference
        'eop' — Equal Opportunity Difference
        'di'  — Disparate Impact Ratio

    Examples
    --------
    >>> import numpy as np
    >>> from uabaf.metrics import fairness_metrics
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_pred = np.array([1, 0, 1, 1, 0, 0])
    >>> sensitive = np.array([1, 1, 1, 0, 0, 0])
    >>> fairness_metrics(y_true, y_pred, sensitive)
    {'dpd': 0.333..., 'eod': 0.5, 'eop': 0.5, 'di': 0.333...}
    """
    y_true    = np.asarray(y_true)
    y_pred    = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    return {
        'dpd': demographic_parity_diff(y_true, y_pred, sensitive),
        'eod': equalized_odds_diff(y_true, y_pred, sensitive),
        'eop': equal_opportunity_diff(y_true, y_pred, sensitive),
        'di' : disparate_impact_ratio(y_true, y_pred, sensitive),
    }
