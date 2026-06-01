"""
Tests for uabaf.profiler
"""

import numpy as np
import pandas as pd
import pytest
from uabaf.profiler import profile_dataset, ProfileReport


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_df():
    """A clean dataset that should pass all Stage 1 checks."""
    np.random.seed(42)
    n = 200
    gender   = np.tile([1, 0], n // 2)
    approved = np.where(gender == 1,
                        np.random.choice([0, 1], n, p=[0.4, 0.6]),
                        np.random.choice([0, 1], n, p=[0.45, 0.55]))
    return pd.DataFrame({
        'age'     : np.random.randint(20, 70, n),
        'income'  : np.random.normal(50000, 10000, n),
        'gender'  : gender,
        'approved': approved,
    })


@pytest.fixture
def small_df():
    """Dataset with a critically small subgroup."""
    np.random.seed(0)
    n = 50
    return pd.DataFrame({
        'age'      : np.random.randint(20, 70, n),
        'income'   : np.random.normal(50000, 10000, n),
        'gender'   : np.array([1] * 45 + [0] * 5),   # only 5 in group 0
        'approved' : np.random.randint(0, 2, n),
    })


@pytest.fixture
def proxy_df():
    """Dataset where a feature is a strong proxy for the sensitive attribute."""
    np.random.seed(1)
    n = 200
    gender = np.tile([1, 0], n // 2)
    income = np.where(gender == 1,
                      np.random.normal(70000, 5000, n),
                      np.random.normal(30000, 5000, n))  # strong correlation
    return pd.DataFrame({
        'income'  : income,
        'gender'  : gender,
        'approved': np.random.randint(0, 2, n),
    })


@pytest.fixture
def missing_df():
    """Dataset with missing values above the threshold."""
    np.random.seed(2)
    n = 200
    df = pd.DataFrame({
        'age'      : np.random.randint(20, 70, n).astype(float),
        'income'   : np.random.normal(50000, 10000, n),
        'gender'   : np.tile([1, 0], n // 2),
        'approved' : np.random.randint(0, 2, n),
    })
    # Introduce 20% missing values in 'age'
    df.loc[df.sample(frac=0.2, random_state=42).index, 'age'] = np.nan
    return df


# ── profile_dataset ───────────────────────────────────────────────────────────

class TestProfileDataset:

    def test_returns_profile_report(self, clean_df):
        report = profile_dataset(clean_df, ['age', 'income'],
                                  'gender', 'approved')
        assert isinstance(report, ProfileReport)

    def test_clean_dataset_passes(self, clean_df):
        report = profile_dataset(clean_df, ['age', 'income'],
                                  'gender', 'approved',
                                  imbalance_threshold=4.0)
        assert report.passed is True

    def test_critical_subgroup_size_fails(self, small_df):
        report = profile_dataset(small_df, ['age', 'income'],
                                  'gender', 'approved',
                                  min_critical=30)
        assert report.passed is False
        critical = [f for f in report.findings if f['severity'] == 'CRITICAL']
        assert len(critical) > 0

    def test_proxy_feature_detected(self, proxy_df):
        report = profile_dataset(proxy_df, ['income'],
                                  'gender', 'approved',
                                  proxy_threshold=0.40)
        proxy_warnings = [
            f for f in report.findings
            if f['check'] == 'proxy_correlation' and f['severity'] == 'WARNING'
        ]
        assert len(proxy_warnings) > 0

    def test_missing_data_detected(self, missing_df):
        report = profile_dataset(missing_df, ['age', 'income'],
                                  'gender', 'approved',
                                  missing_threshold=0.05)
        missing_warnings = [
            f for f in report.findings
            if f['check'] == 'missing_data' and f['severity'] == 'WARNING'
        ]
        assert len(missing_warnings) > 0

    def test_dataset_name_stored(self, clean_df):
        report = profile_dataset(clean_df, ['age', 'income'],
                                  'gender', 'approved',
                                  dataset_name='Test Dataset')
        assert report.dataset_name == 'Test Dataset'

    def test_findings_cover_all_four_checks(self, clean_df):
        report = profile_dataset(clean_df, ['age', 'income'],
                                  'gender', 'approved')
        checks_present = {f['check'] for f in report.findings}
        assert {'subgroup_size', 'class_imbalance',
                'proxy_correlation', 'missing_data'}.issubset(checks_present)


# ── ProfileReport ─────────────────────────────────────────────────────────────

class TestProfileReport:

    def test_to_dataframe_returns_dataframe(self, clean_df):
        report = profile_dataset(clean_df, ['age', 'income'],
                                  'gender', 'approved')
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert 'check' in df.columns
        assert 'severity' in df.columns

    def test_summary_string_set(self, clean_df):
        report = profile_dataset(clean_df, ['age', 'income'],
                                  'gender', 'approved')
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_print_report_runs_without_error(self, clean_df, capsys):
        report = profile_dataset(clean_df, ['age', 'income'],
                                  'gender', 'approved')
        report.print_report()
        captured = capsys.readouterr()
        assert 'STAGE 1' in captured.out
