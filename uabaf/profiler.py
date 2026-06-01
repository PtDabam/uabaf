"""
uabaf.profiler
==============
Stage 1 of the UABAF pipeline — Dataset Readiness Profiling.

Runs before model training or metric computation to flag dataset
characteristics that could make fairness audit results unreliable.
A dataset that fails Stage 1 checks does not necessarily mean the
audit cannot proceed, but the findings should be reported alongside
results and interpreted with appropriate caution.

Checks performed
----------------
1. Subgroup size         — minimum group size for stable bootstrap CIs
2. Class imbalance       — target label skew within each sensitive group
3. Proxy correlation     — feature correlation with the sensitive attribute
4. Missing data          — columns with high proportions of missing values

Severity levels
---------------
CRITICAL  — finding severe enough that audit results are likely unreliable.
            Strongly recommended to collect more data before proceeding.
WARNING   — finding that warrants caution in interpreting results.
INFO      — informational finding, no action required.
"""

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "profile_dataset",
    "ProfileReport",
]

# ── Default thresholds ────────────────────────────────────────────────────────

# Subgroup size
MIN_CRITICAL  = 30   # below this → CRITICAL
MIN_WARNING   = 50   # below this → WARNING

# Class imbalance: majority / minority ratio within a group
IMBALANCE_THRESHOLD = 3.0

# Proxy correlation with sensitive attribute
PROXY_THRESHOLD = 0.40

# Missing data fraction per column
MISSING_THRESHOLD = 0.05


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cramer_v(x, y):
    """
    Cramér's V — association between two categorical / binary arrays.
    Returns a value in [0, 1] where 0 = no association, 1 = perfect.
    Used for low-cardinality features (≤ 10 unique values).
    """
    ct   = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n    = ct.sum().sum()
    k    = min(ct.shape) - 1
    if k == 0 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def _point_biserial(x, y):
    """
    Absolute point-biserial correlation between a continuous feature x
    and a binary sensitive attribute y.
    Returns a value in [0, 1].
    """
    corr, _ = stats.pointbiserialr(x, y)
    return abs(float(corr))


def _proxy_correlation(feature_values, sensitive_values, cardinality_cutoff=10):
    """
    Choose the appropriate correlation measure based on feature cardinality
    and return the association strength as a value in [0, 1].
    """
    if feature_values.nunique() <= cardinality_cutoff:
        return _cramer_v(feature_values, sensitive_values)
    return _point_biserial(feature_values.values, sensitive_values.values)


def _imbalance_ratio(series):
    """
    Majority-to-minority class ratio for a binary series.
    Returns np.inf if only one class is present.
    """
    vc = series.value_counts()
    if len(vc) < 2:
        return np.inf
    return float(vc.iloc[0] / vc.iloc[1])


# ── ProfileReport class ───────────────────────────────────────────────────────

class ProfileReport:
    """
    Container for Stage 1 dataset readiness findings.

    Attributes
    ----------
    dataset_name : str
    findings     : list of dicts, each with keys:
                     check, severity, group (optional), feature (optional),
                     message, value
    passed       : bool — True if no CRITICAL findings
    summary      : str  — one-line readiness verdict

    Methods
    -------
    print_report()   — print a formatted report to stdout
    to_dataframe()   — return findings as a pandas DataFrame
    """

    def __init__(self, dataset_name, findings):
        self.dataset_name = dataset_name
        self.findings     = findings
        self._critical    = [f for f in findings if f['severity'] == 'CRITICAL']
        self._warnings    = [f for f in findings if f['severity'] == 'WARNING']
        self.passed       = len(self._critical) == 0

        if self._critical:
            self.summary = (
                f"NOT READY — {len(self._critical)} critical issue(s). "
                "Collect more data before auditing."
            )
        elif self._warnings:
            self.summary = (
                f"PROCEED WITH CAUTION — {len(self._warnings)} warning(s). "
                "Interpret results carefully."
            )
        else:
            self.summary = "READY — No issues found. Proceed with audit."

    def print_report(self):
        """Print a formatted Stage 1 report to stdout."""
        width = 65
        print("=" * width)
        print("UABAF  STAGE 1 — DATASET READINESS REPORT")
        if self.dataset_name:
            print(f"Dataset: {self.dataset_name}")
        print("=" * width)

        icons = {'CRITICAL': '🔴', 'WARNING': '🟡', 'INFO': '🟢'}

        # Group findings by check
        checks = {}
        for f in self.findings:
            checks.setdefault(f['check'], []).append(f)

        check_titles = {
            'subgroup_size'   : '[1] Subgroup Size Check',
            'class_imbalance' : '[2] Class Imbalance Check',
            'proxy_correlation': '[3] Proxy Correlation Check',
            'missing_data'    : '[4] Missing Data Check',
        }

        for check_key, title in check_titles.items():
            print(f"\n{title}")
            group = checks.get(check_key, [])
            if not group:
                print("  🟢 No issues found.")
                continue
            for f in group:
                icon = icons.get(f['severity'], '  ')
                print(f"  {icon} {f['message']}")

        print(f"\n{'─' * width}")
        print(f"SUMMARY: {self.summary}")
        print("=" * width)

    def to_dataframe(self):
        """Return all findings as a pandas DataFrame."""
        return pd.DataFrame(self.findings)


# ── Public API ────────────────────────────────────────────────────────────────

def profile_dataset(df, features, sensitive, target,
                    dataset_name="",
                    min_critical=MIN_CRITICAL,
                    min_warning=MIN_WARNING,
                    imbalance_threshold=IMBALANCE_THRESHOLD,
                    proxy_threshold=PROXY_THRESHOLD,
                    missing_threshold=MISSING_THRESHOLD):
    """
    Run Stage 1 dataset readiness checks and return a ProfileReport.

    Parameters
    ----------
    df                  : pd.DataFrame — the dataset
    features            : list of str  — feature column names
    sensitive           : str          — sensitive attribute column name
    target              : str          — target label column name
    dataset_name        : str          — optional label for the report
    min_critical        : int   — subgroup size below which → CRITICAL
    min_warning         : int   — subgroup size below which → WARNING
    imbalance_threshold : float — majority/minority ratio above which
                                  → WARNING for class imbalance
    proxy_threshold     : float — correlation above which → WARNING
                                  for proxy features
    missing_threshold   : float — missing fraction above which → WARNING

    Returns
    -------
    ProfileReport

    Examples
    --------
    >>> from uabaf.profiler import profile_dataset
    >>> report = profile_dataset(df, features, sensitive='sex',
    ...                          target='credit_risk',
    ...                          dataset_name='German Credit')
    >>> report.print_report()
    >>> df_findings = report.to_dataframe()
    """
    findings = []

    # ── Check 1: Subgroup size ────────────────────────────────────────
    group_counts = df[sensitive].value_counts().to_dict()

    for group_val, count in group_counts.items():
        if count < min_critical:
            severity = 'CRITICAL'
            msg = (f"Group '{sensitive}={group_val}': n={count} — "
                   f"below critical minimum ({min_critical}). "
                   "Bootstrap CIs unreliable.")
        elif count < min_warning:
            severity = 'WARNING'
            msg = (f"Group '{sensitive}={group_val}': n={count} — "
                   f"below recommended minimum ({min_warning}). "
                   "CIs may be unstable.")
        else:
            severity = 'INFO'
            msg = f"Group '{sensitive}={group_val}': n={count} — OK."

        findings.append({
            'check'   : 'subgroup_size',
            'severity': severity,
            'group'   : group_val,
            'feature' : sensitive,
            'value'   : count,
            'message' : msg,
        })

    # ── Check 2: Class imbalance within each group ────────────────────
    for group_val in sorted(df[sensitive].unique()):
        mask  = df[sensitive] == group_val
        ratio = _imbalance_ratio(df.loc[mask, target])

        if ratio == np.inf:
            severity = 'CRITICAL'
            msg = (f"Group '{sensitive}={group_val}': only one class "
                   "present in target — metric computation impossible.")
        elif ratio > imbalance_threshold:
            severity = 'WARNING'
            msg = (f"Group '{sensitive}={group_val}': class ratio "
                   f"{ratio:.2f}:1 — may distort Equal Opportunity metric.")
        else:
            severity = 'INFO'
            msg = (f"Group '{sensitive}={group_val}': class ratio "
                   f"{ratio:.2f}:1 — OK.")

        findings.append({
            'check'   : 'class_imbalance',
            'severity': severity,
            'group'   : group_val,
            'feature' : target,
            'value'   : round(ratio, 4),
            'message' : msg,
        })

    # ── Check 3: Proxy correlation ────────────────────────────────────
    proxy_findings = []
    for feat in features:
        try:
            corr = _proxy_correlation(df[feat], df[sensitive])
        except Exception:
            continue

        if corr >= proxy_threshold:
            proxy_findings.append((feat, round(corr, 4)))
            findings.append({
                'check'   : 'proxy_correlation',
                'severity': 'WARNING',
                'group'   : None,
                'feature' : feat,
                'value'   : round(corr, 4),
                'message' : (f"Feature '{feat}' has correlation "
                             f"{corr:.3f} with '{sensitive}' — "
                             "potential proxy for sensitive attribute."),
            })

    if not proxy_findings:
        findings.append({
            'check'   : 'proxy_correlation',
            'severity': 'INFO',
            'group'   : None,
            'feature' : None,
            'value'   : 0.0,
            'message' : (f"No features exceed proxy threshold "
                         f"({proxy_threshold}) for '{sensitive}'."),
        })

    # ── Check 4: Missing data ─────────────────────────────────────────
    all_cols    = features + [sensitive, target]
    missing_pct = df[all_cols].isnull().mean()
    flagged     = missing_pct[missing_pct > missing_threshold]

    if flagged.empty:
        findings.append({
            'check'   : 'missing_data',
            'severity': 'INFO',
            'group'   : None,
            'feature' : None,
            'value'   : 0.0,
            'message' : (f"No columns exceed {missing_threshold:.0%} "
                         "missing values — OK."),
        })
    else:
        for col, pct in flagged.items():
            findings.append({
                'check'   : 'missing_data',
                'severity': 'WARNING',
                'group'   : None,
                'feature' : col,
                'value'   : round(pct, 4),
                'message' : (f"Column '{col}': {pct:.1%} missing — "
                             "consider imputation before auditing."),
            })

    return ProfileReport(dataset_name=dataset_name, findings=findings)
