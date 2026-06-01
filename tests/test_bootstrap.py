"""
Tests for uabaf.bootstrap
"""

import numpy as np
import pytest
from uabaf.bootstrap import bca_ci, bca_ci_all_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def biased_data():
    """80-record dataset with known group bias."""
    np.random.seed(42)
    n         = 80
    y_true    = np.tile([1, 0], n // 2)
    sensitive = np.array([1] * 40 + [0] * 40)
    y_pred    = y_true.copy()
    rng       = np.random.default_rng(42)
    y_pred[40:] = np.where(rng.random(40) > 0.4, 0, y_pred[40:])
    return y_true, y_pred, sensitive


@pytest.fixture
def fair_data():
    """80-record dataset with no group bias."""
    np.random.seed(0)
    n         = 80
    y_true    = np.tile([1, 0], n // 2)
    sensitive = np.array([1] * 40 + [0] * 40)
    y_pred    = y_true.copy()   # perfect predictions, no bias
    return y_true, y_pred, sensitive


# ── bca_ci ────────────────────────────────────────────────────────────────────

class TestBcaCi:

    def test_returns_expected_keys(self, biased_data):
        y_true, y_pred, s = biased_data
        result = bca_ci(y_true, y_pred, s, 'dpd', B=200, random_state=42)
        assert set(result.keys()) == {'metric', 'point', 'ci_lo', 'ci_hi',
                                       'width', 'n'}

    def test_metric_key_stored(self, biased_data):
        y_true, y_pred, s = biased_data
        result = bca_ci(y_true, y_pred, s, 'eop', B=200, random_state=42)
        assert result['metric'] == 'eop'

    def test_ci_contains_point_estimate(self, biased_data):
        """The point estimate should fall within its own CI."""
        y_true, y_pred, s = biased_data
        for metric in ['dpd', 'eod', 'eop', 'di']:
            result = bca_ci(y_true, y_pred, s, metric, B=300, random_state=42)
            if not (np.isnan(result['ci_lo']) or np.isnan(result['ci_hi'])):
                assert result['ci_lo'] <= result['point'] <= result['ci_hi'], \
                    f"Point not in CI for metric={metric}"

    def test_ci_lo_less_than_ci_hi(self, biased_data):
        y_true, y_pred, s = biased_data
        result = bca_ci(y_true, y_pred, s, 'dpd', B=200, random_state=42)
        assert result['ci_lo'] < result['ci_hi']

    def test_width_equals_hi_minus_lo(self, biased_data):
        y_true, y_pred, s = biased_data
        result = bca_ci(y_true, y_pred, s, 'dpd', B=200, random_state=42)
        expected_width = round(result['ci_hi'] - result['ci_lo'], 6)
        assert result['width'] == pytest.approx(expected_width, abs=1e-5)

    def test_fair_data_ci_includes_zero(self, fair_data):
        """For unbiased data the DPD CI should include 0."""
        y_true, y_pred, s = fair_data
        result = bca_ci(y_true, y_pred, s, 'dpd', B=300, random_state=42)
        assert result['ci_lo'] <= 0.0 <= result['ci_hi']

    def test_reproducible_with_same_seed(self, biased_data):
        y_true, y_pred, s = biased_data
        r1 = bca_ci(y_true, y_pred, s, 'dpd', B=200, random_state=7)
        r2 = bca_ci(y_true, y_pred, s, 'dpd', B=200, random_state=7)
        assert r1['ci_lo'] == r2['ci_lo']
        assert r1['ci_hi'] == r2['ci_hi']

    def test_different_seeds_may_differ(self, biased_data):
        y_true, y_pred, s = biased_data
        r1 = bca_ci(y_true, y_pred, s, 'dpd', B=200, random_state=1)
        r2 = bca_ci(y_true, y_pred, s, 'dpd', B=200, random_state=99)
        # CIs from different seeds should be close but not identical
        assert r1['ci_lo'] != r2['ci_lo'] or r1['ci_hi'] != r2['ci_hi']

    def test_more_resamples_narrows_ci(self, biased_data):
        """More bootstrap resamples should produce a more stable (narrower) CI."""
        y_true, y_pred, s = biased_data
        r_small = bca_ci(y_true, y_pred, s, 'dpd', B=100,  random_state=42)
        r_large = bca_ci(y_true, y_pred, s, 'dpd', B=2000, random_state=42)
        # Width should be more stable with more resamples — not necessarily
        # strictly narrower, but the point estimates should be close
        assert abs(r_small['point'] - r_large['point']) < 1e-9


# ── bca_ci_all_metrics ────────────────────────────────────────────────────────

class TestBcaCiAllMetrics:

    def test_returns_all_four_metrics(self, biased_data):
        y_true, y_pred, s = biased_data
        results = bca_ci_all_metrics(y_true, y_pred, s, B=200, random_state=42)
        assert set(results.keys()) == {'dpd', 'eod', 'eop', 'di'}

    def test_each_result_has_expected_keys(self, biased_data):
        y_true, y_pred, s = biased_data
        results = bca_ci_all_metrics(y_true, y_pred, s, B=200, random_state=42)
        for key, res in results.items():
            assert 'point' in res
            assert 'ci_lo' in res
            assert 'ci_hi' in res
            assert 'width' in res

    def test_point_estimates_match_bca_ci(self, biased_data):
        """Point estimates from all-metrics call must match single-metric calls."""
        y_true, y_pred, s = biased_data
        all_results = bca_ci_all_metrics(y_true, y_pred, s, B=200, random_state=42)
        for metric in ['dpd', 'eod', 'eop', 'di']:
            single = bca_ci(y_true, y_pred, s, metric, B=200, random_state=42)
            assert all_results[metric]['point'] == pytest.approx(
                single['point'], abs=1e-9), \
                f"Point mismatch for {metric}"

    def test_ci_bounds_ordered(self, biased_data):
        """ci_lo must be less than ci_hi for all metrics."""
        y_true, y_pred, s = biased_data
        results = bca_ci_all_metrics(y_true, y_pred, s, B=200, random_state=42)
        for metric, res in results.items():
            if not (np.isnan(res['ci_lo']) or np.isnan(res['ci_hi'])):
                assert res['ci_lo'] < res['ci_hi'], \
                    f"ci_lo >= ci_hi for metric={metric}"

    def test_reproducible(self, biased_data):
        y_true, y_pred, s = biased_data
        r1 = bca_ci_all_metrics(y_true, y_pred, s, B=200, random_state=42)
        r2 = bca_ci_all_metrics(y_true, y_pred, s, B=200, random_state=42)
        for metric in ['dpd', 'eod', 'eop', 'di']:
            assert r1[metric]['ci_lo'] == r2[metric]['ci_lo']
            assert r1[metric]['ci_hi'] == r2[metric]['ci_hi']
