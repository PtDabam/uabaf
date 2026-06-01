"""
uabaf.visualise
===============
Visualisation functions for UABAF audit results.

All functions accept the output of assign_all_verdicts() and return
matplotlib Figure objects so the caller can choose to display, save,
or embed them in a report.

Plots provided
--------------
plot_ci_intervals   — horizontal CI bars for all four metrics, one
                      panel per metric, coloured by verdict
plot_verdict_summary — compact colour-coded summary table of verdicts
                       across multiple models or datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from .verdict import METRIC_LABELS, THRESHOLDS, CI_WIDTH_THRESHOLD

__all__ = [
    "plot_ci_intervals",
    "plot_verdict_summary",
]

# ── Colour palette ────────────────────────────────────────────────────────────
# Mapped to the four verdict categories for consistency across all plots.

VERDICT_COLORS = {
    'PASS_HIGH': '#16A34A',   # green  — ✅ PASS High Confidence
    'PASS_LOW' : '#86EFAC',   # light green — ⚠️  PASS Low Confidence
    'FAIL_HIGH': '#DC2626',   # red    — ❌ FAIL High Confidence
    'FAIL_LOW' : '#FCA5A5',   # light red   — 🔍 FAIL Low Confidence
    'THRESHOLD': '#6B7280',   # grey   — threshold lines
    'IDEAL'    : '#D1D5DB',   # light grey — ideal value line
}

VERDICT_LABELS = {
    'PASS_HIGH': '✅ PASS — High Confidence',
    'PASS_LOW' : '⚠️  PASS — Low Confidence',
    'FAIL_HIGH': '❌ FAIL — High Confidence',
    'FAIL_LOW' : '🔍 FAIL — Low Confidence',
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _verdict_color_key(pass_fail, confidence):
    """Map pass_fail + confidence to a VERDICT_COLORS key."""
    if pass_fail == 'PASS' and confidence == 'HIGH':
        return 'PASS_HIGH'
    elif pass_fail == 'PASS' and confidence == 'LOW':
        return 'PASS_LOW'
    elif pass_fail == 'FAIL' and confidence == 'HIGH':
        return 'FAIL_HIGH'
    else:
        return 'FAIL_LOW'


def _draw_threshold_lines(ax, metric_key):
    """
    Draw the fairness threshold line(s) on an axis.
    Difference metrics get two symmetric lines; DI gets two asymmetric ones
    plus a dotted ideal line at 1.0.
    """
    kind, threshold = THRESHOLDS[metric_key]

    if kind == 'diff':
        for val in [-threshold, threshold]:
            ax.axvline(val, color=VERDICT_COLORS['THRESHOLD'],
                       linewidth=1.2, linestyle='--', alpha=0.8,
                       label=f'threshold ±{threshold}' if val > 0 else None)
        ax.axvline(0, color=VERDICT_COLORS['IDEAL'],
                   linewidth=0.8, linestyle=':', alpha=0.6)
    elif kind == 'ratio':
        lo, hi = threshold
        ax.axvline(lo, color=VERDICT_COLORS['THRESHOLD'],
                   linewidth=1.2, linestyle='--', alpha=0.8,
                   label=f'threshold [{lo}, {hi}]')
        ax.axvline(hi, color=VERDICT_COLORS['THRESHOLD'],
                   linewidth=1.2, linestyle='--', alpha=0.8)
        ax.axvline(1.0, color=VERDICT_COLORS['IDEAL'],
                   linewidth=0.8, linestyle=':', alpha=0.6,
                   label='ideal (1.0)')


def _format_ci_label(point, ci_lo, ci_hi):
    """Build a compact annotation string for a CI bar."""
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return f"{point:.3f}\n(CI unavailable)"
    return f"{point:.3f}\n[{ci_lo:.3f}, {ci_hi:.3f}]"


# ── Public API ────────────────────────────────────────────────────────────────

def plot_ci_intervals(verdict_results, title="UABAF Audit — BCa 95% CIs",
                      figsize=None, annotate=True):
    """
    Plot BCa confidence intervals for all four fairness metrics.

    Each metric gets its own panel. The CI is drawn as a horizontal bar,
    the point estimate as a dot, and fairness thresholds as dashed vertical
    lines. Bars are coloured by verdict category.

    Parameters
    ----------
    verdict_results : dict — output of assign_all_verdicts(), keyed by
                             metric name. Each value must have keys:
                             'point', 'ci_lo', 'ci_hi', 'pass_fail',
                             'confidence', 'label', 'verdict'
    title           : str  — figure title
    figsize         : tuple or None — override default figure size
    annotate        : bool — if True, show point estimate and CI as text
                             on each bar

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from uabaf.visualise import plot_ci_intervals
    >>> fig = plot_ci_intervals(verdicts, title="German Credit — LR")
    >>> fig.savefig("audit_plot.png", dpi=150, bbox_inches='tight')
    >>> plt.show()
    """
    metrics = ['dpd', 'eod', 'eop', 'di']
    n_metrics = len(metrics)

    if figsize is None:
        figsize = (5 * n_metrics, 3.5)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    for ax, metric_key in zip(axes, metrics):
        if metric_key not in verdict_results:
            ax.set_visible(False)
            continue

        res        = verdict_results[metric_key]
        point      = res['point']
        ci_lo      = res['ci_lo']
        ci_hi      = res['ci_hi']
        pass_fail  = res['pass_fail']
        confidence = res['confidence']
        color_key  = _verdict_color_key(pass_fail, confidence)
        bar_color  = VERDICT_COLORS[color_key]

        # ── CI bar ────────────────────────────────────────────────
        if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
            ax.barh(0, ci_hi - ci_lo, left=ci_lo, height=0.4,
                    color=bar_color, alpha=0.85, zorder=2)
            # Whisker caps
            for x in [ci_lo, ci_hi]:
                ax.plot(x, 0, '|', color=bar_color,
                        markersize=12, markeredgewidth=2, zorder=3)

        # ── Point estimate ────────────────────────────────────────
        ax.plot(point, 0, 'o', color=bar_color,
                markersize=9, markeredgecolor='white',
                markeredgewidth=1.5, zorder=4)

        # ── Threshold lines ───────────────────────────────────────
        _draw_threshold_lines(ax, metric_key)

        # ── Annotation ────────────────────────────────────────────
        if annotate:
            label = _format_ci_label(point, ci_lo, ci_hi)
            ax.text(point, 0.28, label,
                    ha='center', va='bottom', fontsize=7.5,
                    color='#1F2937', zorder=5)

        # ── Verdict badge ─────────────────────────────────────────
        ax.text(0.5, -0.32, res['verdict'],
                transform=ax.transAxes,
                ha='center', va='top', fontsize=7.5,
                color=bar_color, fontweight='bold')

        # ── Formatting ────────────────────────────────────────────
        ax.set_title(res['label'], fontsize=9, fontweight='bold', pad=8)
        ax.set_yticks([])
        ax.set_xlabel('Metric Value', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='x', alpha=0.3, zorder=0)
        ax.spines[['top', 'right', 'left']].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=VERDICT_COLORS[k], label=VERDICT_LABELS[k])
        for k in ['PASS_HIGH', 'PASS_LOW', 'FAIL_HIGH', 'FAIL_LOW']
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color=VERDICT_COLORS['THRESHOLD'],
                   linestyle='--', label='Fairness threshold')
    )
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    return fig


def plot_verdict_summary(results_dict, title="UABAF — Verdict Summary"):
    """
    Plot a colour-coded summary grid of verdicts across multiple
    models or datasets.

    Useful for comparing audit results side-by-side — for example
    Logistic Regression vs Random Forest across three datasets.

    Parameters
    ----------
    results_dict : dict — nested dict structured as:
                     { row_label: { metric_key: verdict_dict } }
                   where verdict_dict has keys 'pass_fail' and 'confidence'.
                   Row labels become the y-axis (e.g. "German Credit / LR").
                   Each row must cover all four metrics.
    title        : str — figure title

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> summary = {
    ...     'German Credit / LR': verdicts_german_lr,
    ...     'German Credit / RF': verdicts_german_rf,
    ...     'COMPAS / LR'       : verdicts_compas_lr,
    ... }
    >>> fig = plot_verdict_summary(summary)
    >>> plt.show()
    """
    metrics    = ['dpd', 'eod', 'eop', 'di']
    row_labels = list(results_dict.keys())
    n_rows     = len(row_labels)
    n_cols     = len(metrics)

    fig, ax = plt.subplots(figsize=(n_cols * 2.2, n_rows * 0.8 + 1.5))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    col_labels = [METRIC_LABELS[m] for m in metrics]

    for r, row_label in enumerate(row_labels):
        row_data = results_dict[row_label]
        for c, metric_key in enumerate(metrics):
            if metric_key not in row_data:
                continue
            res       = row_data[metric_key]
            color_key = _verdict_color_key(res['pass_fail'], res['confidence'])
            color     = VERDICT_COLORS[color_key]

            rect = mpatches.FancyBboxPatch(
                (c + 0.05, n_rows - r - 0.95), 0.9, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='white', linewidth=1.5,
                transform=ax.transData, zorder=2
            )
            ax.add_patch(rect)

            # Emoji icon in the cell
            icons = {'PASS_HIGH': '✅', 'PASS_LOW': '⚠️',
                     'FAIL_HIGH': '❌', 'FAIL_LOW': '🔍'}
            ax.text(c + 0.5, n_rows - r - 0.55,
                    icons.get(color_key, ''),
                    ha='center', va='center', fontsize=13, zorder=3)

    # Column headers
    for c, label in enumerate(col_labels):
        ax.text(c + 0.5, n_rows + 0.1, label,
                ha='center', va='bottom', fontsize=8.5,
                fontweight='bold', color='#374151')

    # Row labels
    for r, label in enumerate(row_labels):
        ax.text(-0.1, n_rows - r - 0.55, label,
                ha='right', va='center', fontsize=9, color='#374151')

    ax.set_xlim(-0.1, n_cols + 0.1)
    ax.set_ylim(-0.2, n_rows + 0.5)
    ax.axis('off')

    # Legend
    legend_handles = [
        mpatches.Patch(color=VERDICT_COLORS[k], label=VERDICT_LABELS[k])
        for k in ['PASS_HIGH', 'PASS_LOW', 'FAIL_HIGH', 'FAIL_LOW']
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    return fig
