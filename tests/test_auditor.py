"""
Tests for uabaf.auditor
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from uabaf import AuditReport


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fitted_model_and_data():
    """Logistic Regression fitted on synthetic biased data."""
    np.random.seed(42)
    n         = 300
    X         = np.random.randn(n, 4)
    sensitive = np.tile([1, 0], n // 2)
    y         = (X[:, 0] + X[:, 1] > 0).astype(int)
    y[sensitive == 0] = np.where(
        np.random.rand((sensitive == 0).sum()) > 0.45,
        0, y[sensitive == 0]
    )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_idx, test_idx in sss.split(X, sensitive):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        s_train, s_test = sensitive[train_idx], sensitive[test_idx]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test, s_test


# ── AuditReport construction ──────────────────────────────────────────────────

class TestAuditReportConstruction:

    def test_basic_construction(self, fitted_model_and_data):
        clf, X_test, y_test, s_test = fitted_model_and_data
        report = AuditReport(clf, X_test, y_test, sensitive=s_test,
                             B=200, random_state=42)
        assert report is not None

    def test_y_pred_generated(self, fitted_model_and_data):
        clf, X_test, y_test, s_test = fitted_model_and_data
        report = AuditReport(clf, X_test, y_test, sensitive=s_test,
                             B=200, random_state=42)
        assert len(report.y_pred) == len(y_test)

    def test_ci_results_has_all_metrics(self, fitted_model_and_data):
        clf, X_test, y_test, s_test = fitted_model_and_data
        report = AuditReport(clf, X_test, y_test, sensitive=s_test,
                             B=200, random_state=42)
        assert set(report.ci_results.keys()) == {'dpd', 'eod', 'eop', 'di'}

    def test_verdicts_has_all_metrics(self, fitted_model_and_data):
        clf, X_test, y_test, s_test = fitted_model_and_data
        report = AuditReport(clf, X_test, y_test, sensitive=s_test,
                             B=200, random_state=42)
        assert set(report.verdicts.keys()) == {'dpd', 'eod', 'eop', 'di'}

    def test_passed_is_bool(self, fitted_model_and_data):
        clf, X_test, y_test, s_test = fitted_model_and_data
        report = AuditReport(clf, X_test, y_test, sensitive=s_test,
                             B=200, random_state=42)
        assert isinstance(report.passed, bool)

    def test_no_stage1_without_df(self, fitted_model_and_data):
        clf, X_test, y_test, s_test = fitted_model_and_data
        report = AuditReport(clf, X_test, y_test, sensitive=s_test,
                             B=200, random_state=42)
        assert report.profile is None

    def test_model_name_stored(self, fitted_model_and_data):
        clf, X_test, y_test, s_test = fitted_model_and_data
        report = AuditReport(clf, X_test, y_test, sensitive=s_test,
                             B=200, model_name='Test Model')
        assert report.model_name == 'Test Model'

    def test_dataset_name_stored(self, fitted_model_and_data):
        clf, X_test, y_test, s_test = fitted_model_and_data
        report = AuditReport(clf, X_test, y_test, sensitive=s_test,
                             B=200, dataset_name='Test Dataset')
        assert report.dataset_name == 'Test Dataset'


# ── AuditReport methods ───────────────────────────────────────────────────────

class TestAuditReportMethods:

    @pytest.fixture
    def report(self, fitted_model_and_data):
        clf, X_test, y_test, s_test = fitted_model_and_data
        return AuditReport(clf, X_test, y_test, sensitive=s_test,
                           B=200, random_state=42,
                           model_name='LR', dataset_name='Synthetic')

    def test_summary_prints_without_error(self, report, capsys):
        report.summary()
        captured = capsys.readouterr()
        assert 'STAGE 3' in captured.out
        assert 'OVERALL' in captured.out

    def test_plot_returns_figure(self, report):
        fig = report.plot()
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_to_dataframe_returns_dataframe(self, report):
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_to_dataframe_has_four_rows(self, report):
        df = report.to_dataframe()
        assert len(df) == 4

    def test_to_dataframe_columns(self, report):
        df = report.to_dataframe()
        expected = {'dataset', 'model', 'metric', 'label',
                    'point', 'ci_lo', 'ci_hi', 'width',
                    'pass_fail', 'confidence', 'verdict', 'breaches'}
        assert expected.issubset(set(df.columns))

    def test_repr_contains_model_name(self, report):
        assert 'LR' in repr(report)

    def test_repr_contains_dataset_name(self, report):
        assert 'Synthetic' in repr(report)

    def test_repr_contains_overall_status(self, report):
        r = repr(report)
        assert 'PASS' in r or 'FAIL' in r
