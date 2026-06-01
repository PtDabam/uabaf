"""
uabaf.auditor
=============
Stage 2 + Stage 3 orchestration and the AuditReport class.

This is the main entry point for UABAF. It coordinates all three stages
of the pipeline and exposes results through a single AuditReport object.

Pipeline
--------
Stage 1  profile_dataset()       — dataset readiness checks
Stage 2  bca_ci_all_metrics()    — fairness metrics + BCa CIs
Stage 3  assign_all_verdicts()   — uncertainty-aware verdict assignment

Typical usage
-------------
>>> from uabaf import AuditReport
>>>
>>> report = AuditReport(
...     model     = clf,
...     X_test    = X_test,
...     y_test    = y_test,
...     sensitive = s_test,
... )
>>> report.summary()
>>> report.plot()
>>> df = report.to_dataframe()

With Stage 1 dataset profiling
-------------------------------
>>> report = AuditReport(
...     model        = clf,
...     X_test       = X_test,
...     y_test       = y_test,
...     sensitive    = s_test,
...     df           = df,
...     features     = feature_cols,
...     target       = 'label',
...     dataset_name = 'German Credit',
... )
>>> report.summary()   # prints Stage 1 findings then Stage 3 verdicts
"""

import numpy as np
import pandas as pd

from .bootstrap import bca_ci_all_metrics, DEFAULT_B, DEFAULT_ALPHA
from .verdict   import assign_all_verdicts, METRIC_LABELS
from .profiler  import profile_dataset
from .visualise import plot_ci_intervals, plot_verdict_summary

__all__ = ["AuditReport"]


class AuditReport:
    """
    Full UABAF audit for a fitted binary classifier.

    Runs all three pipeline stages and stores results as attributes
    so the user can inspect, print, plot, or export them.

    Parameters
    ----------
    model        : fitted sklearn-compatible classifier
                   Must implement predict(X).
    X_test       : array-like, shape (n, p)
                   Test feature matrix.
    y_test       : array-like, shape (n,)
                   True binary labels for the test set.
    sensitive    : array-like, shape (n,)
                   Binary sensitive attribute for the test set.
                   1 = privileged group, 0 = unprivileged group.
    B            : int — bootstrap resamples (default 2000)
    alpha        : float — significance level (default 0.05 → 95% CI)
    random_state : int — random seed (default 42)
    model_name   : str — optional label shown in reports and plots
    dataset_name : str — optional label shown in reports and plots

    Stage 1 parameters (all optional — Stage 1 is skipped if df is None)
    ----------------------------------------------------------------------
    df           : pd.DataFrame — the dataset used to train/test the model
    features     : list of str  — feature column names in df
    target       : str          — target column name in df

    Attributes
    ----------
    profile      : ProfileReport or None  — Stage 1 findings
    ci_results   : dict — Stage 2 BCa CI results keyed by metric
    verdicts     : dict — Stage 3 verdict dicts keyed by metric
    y_pred       : np.ndarray — model predictions on X_test
    passed       : bool — True if no metric has FAIL — High Confidence

    Examples
    --------
    >>> report = AuditReport(clf, X_test, y_test, sensitive=s_test,
    ...                      model_name='Random Forest',
    ...                      dataset_name='COMPAS')
    >>> report.summary()
    >>> fig = report.plot()
    >>> fig.savefig('audit.png', dpi=150, bbox_inches='tight')
    >>> df = report.to_dataframe()
    """

    def __init__(self, model, X_test, y_test, sensitive,
                 B=DEFAULT_B, alpha=DEFAULT_ALPHA, random_state=42,
                 model_name="Model", dataset_name="Dataset",
                 df=None, features=None, target=None):

        self.model_name   = model_name
        self.dataset_name = dataset_name
        self.B            = B
        self.alpha        = alpha

        # ── Prepare arrays ────────────────────────────────────────────
        self.X_test    = np.asarray(X_test)
        self.y_test    = np.asarray(y_test)
        self.sensitive = np.asarray(sensitive)

        # ── Get predictions ───────────────────────────────────────────
        self.y_pred = model.predict(self.X_test)

        # ── Stage 1: Dataset profiling (optional) ─────────────────────
        if df is not None and features is not None and target is not None:
            self.profile = profile_dataset(
                df           = df,
                features     = features,
                sensitive     = sensitive if isinstance(sensitive, str)
                               else _find_sensitive_col(df, self.sensitive),
                target       = target,
                dataset_name = dataset_name,
            )
        else:
            self.profile = None

        # ── Stage 2: BCa bootstrap CIs ────────────────────────────────
        self.ci_results = bca_ci_all_metrics(
            y_true       = self.y_test,
            y_pred       = self.y_pred,
            sensitive    = self.sensitive,
            B            = B,
            alpha        = alpha,
            random_state = random_state,
        )

        # ── Stage 3: Verdict assignment ───────────────────────────────
        self.verdicts = assign_all_verdicts(self.ci_results)

        # ── Overall pass/fail ─────────────────────────────────────────
        # The audit passes overall only if no metric is a high-confidence
        # fail — low-confidence fails are inconclusive, not definitive.
        self.passed = not any(
            v['pass_fail'] == 'FAIL' and v['confidence'] == 'HIGH'
            for v in self.verdicts.values()
        )

    # ── Reporting ─────────────────────────────────────────────────────

    def summary(self, show_stage1=True):
        """
        Print a formatted audit summary to stdout.

        Prints Stage 1 findings (if available) followed by the Stage 3
        verdict table for all four metrics.

        Parameters
        ----------
        show_stage1 : bool — if True and Stage 1 was run, print the
                             dataset readiness report first
        """
        width = 68

        # ── Stage 1 ───────────────────────────────────────────────────
        if self.profile is not None and show_stage1:
            self.profile.print_report()
            print()

        # ── Header ────────────────────────────────────────────────────
        print("=" * width)
        print("UABAF  STAGE 3 — AUDIT REPORT")
        print(f"Dataset : {self.dataset_name}")
        print(f"Model   : {self.model_name}")
        print(f"Bootstrap resamples : {self.B}   "
              f"Confidence level : {int((1 - self.alpha) * 100)}%")
        print("=" * width)

        # ── Column headers ────────────────────────────────────────────
        print(f"\n{'Metric':<30} {'Point':>7}  {'CI Lo':>7}  "
              f"{'CI Hi':>7}  {'Width':>6}  Verdict")
        print("─" * width)

        # ── One row per metric ────────────────────────────────────────
        for metric_key, v in self.verdicts.items():
            point = v['point']
            lo    = v['ci_lo']
            hi    = v['ci_hi']
            width_val = v['width']

            lo_str    = f"{lo:.4f}"    if not np.isnan(lo)         else "  nan "
            hi_str    = f"{hi:.4f}"    if not np.isnan(hi)         else "  nan "
            width_str = f"{width_val:.4f}" if not np.isnan(width_val) else "  nan "

            print(f"{v['label']:<30} {point:>7.4f}  {lo_str:>7}  "
                  f"{hi_str:>7}  {width_str:>6}  {v['verdict']}")

        # ── Overall verdict ───────────────────────────────────────────
        print("─" * width)
        overall = ("✅ OVERALL: PASS" if self.passed
                   else "❌ OVERALL: FAIL")
        note = ("  (no high-confidence failures)" if self.passed
                else "  (one or more high-confidence failures)")
        print(f"{overall}{note}")
        print("=" * width)

    def plot(self, figsize=None, annotate=True):
        """
        Plot BCa confidence intervals for all four metrics.

        Returns the matplotlib Figure so the caller can save or display it.

        Parameters
        ----------
        figsize  : tuple or None — override default figure size
        annotate : bool — show point estimate and CI as text on each bar

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = report.plot()
        >>> fig.savefig('audit.png', dpi=150, bbox_inches='tight')
        """
        title = f"{self.dataset_name} — {self.model_name}"
        return plot_ci_intervals(
            self.verdicts,
            title    = title,
            figsize  = figsize,
            annotate = annotate,
        )

    def to_dataframe(self):
        """
        Return audit results as a pandas DataFrame.

        One row per metric with columns:
        metric, label, point, ci_lo, ci_hi, width,
        pass_fail, confidence, verdict, breaches

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> df = report.to_dataframe()
        >>> df.to_csv('audit_results.csv', index=False)
        """
        rows = []
        for metric_key, v in self.verdicts.items():
            rows.append({
                'dataset'   : self.dataset_name,
                'model'     : self.model_name,
                'metric'    : metric_key,
                'label'     : v['label'],
                'point'     : v['point'],
                'ci_lo'     : v['ci_lo'],
                'ci_hi'     : v['ci_hi'],
                'width'     : v['width'],
                'pass_fail' : v['pass_fail'],
                'confidence': v['confidence'],
                'verdict'   : v['verdict'],
                'breaches'  : v['breaches'],
            })
        return pd.DataFrame(rows)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return (f"AuditReport(dataset='{self.dataset_name}', "
                f"model='{self.model_name}', "
                f"n={len(self.y_test)}, "
                f"overall={status})")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_sensitive_col(df, sensitive_array):
    """
    Try to identify the sensitive attribute column name in df by matching
    the array values. Used when sensitive is passed as an array rather
    than a column name string.
    Falls back to '__sensitive__' if no match is found.
    """
    for col in df.columns:
        if np.array_equal(df[col].values, sensitive_array):
            return col
    return '__sensitive__'
