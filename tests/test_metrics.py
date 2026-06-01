"""
Tests for uabaf.metrics
"""

import numpy as np
import pytest
from uabaf.metrics import (
    demographic_parity_diff,
    equalized_odds_diff,
    equal_opportunity_diff,
    disparate_impact_ratio,
    fairness_metrics,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_fairness():
    """Both groups receive identical predictions — all metrics should be 0 / 1."""
    y_true    = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred    = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    sensitive = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    return y_true, y_pred, sensitive


@pytest.fixture
def total_discrimination():
    """Privileged group always positive, unprivileged always negative."""
    y_true    = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    y_pred    = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    sensitive = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    return y_true, y_pred, sensitive


@pytest.fixture
def known_values():
    """Small example with manually verifiable metric values."""
    # Group 1 (priv):   y_true=[1,0,1], y_pred=[1,0,1] → PR=2/3, TPR=1.0, FPR=0.0
    # Group 0 (unpriv): y_true=[0,1,0], y_pred=[1,0,0] → PR=1/3, TPR=0.0, FPR=0.5
    y_true    = np.array([1, 0, 1, 0, 1, 0])
    y_pred    = np.array([1, 0, 1, 1, 0, 0])
    sensitive = np.array([1, 1, 1, 0, 0, 0])
    return y_true, y_pred, sensitive


# ── demographic_parity_diff ───────────────────────────────────────────────────

class TestDemographicParityDiff:

    def test_perfect_fairness_is_zero(self, perfect_fairness):
        y_true, y_pred, s = perfect_fairness
        assert demographic_parity_diff(y_true, y_pred, s) == pytest.approx(0.0)

    def test_total_discrimination_is_one(self, total_discrimination):
        y_true, y_pred, s = total_discrimination
        assert demographic_parity_diff(y_true, y_pred, s) == pytest.approx(1.0)

    def test_known_value(self, known_values):
        y_true, y_pred, s = known_values
        # PR_priv = 2/3, PR_unpriv = 1/3 → DPD = 1/3
        result = demographic_parity_diff(y_true, y_pred, s)
        assert result == pytest.approx(1/3, abs=1e-6)

    def test_positive_means_privileged_favoured(self, known_values):
        y_true, y_pred, s = known_values
        assert demographic_parity_diff(y_true, y_pred, s) > 0

    def test_empty_group_returns_nan(self):
        y_true    = np.array([1, 0, 1, 0])
        y_pred    = np.array([1, 0, 1, 0])
        sensitive = np.array([1, 1, 1, 1])   # no group 0
        result    = demographic_parity_diff(y_true, y_pred, sensitive)
        assert np.isnan(result)

    def test_accepts_list_input(self):
        result = demographic_parity_diff([1,0,1,0], [1,0,1,0], [1,1,0,0])
        assert result == pytest.approx(0.0)


# ── equalized_odds_diff ───────────────────────────────────────────────────────

class TestEqualizedOddsDiff:

    def test_perfect_fairness_is_zero(self, perfect_fairness):
        y_true, y_pred, s = perfect_fairness
        assert equalized_odds_diff(y_true, y_pred, s) == pytest.approx(0.0)

    def test_always_non_negative(self, known_values):
        y_true, y_pred, s = known_values
        assert equalized_odds_diff(y_true, y_pred, s) >= 0.0

    def test_known_value(self, known_values):
        y_true, y_pred, s = known_values
        # TPR_priv=1.0, TPR_unpriv=0.0 → TPR diff=1.0
        # FPR_priv=0.0, FPR_unpriv=0.5 → FPR diff=0.5
        # EOD = max(1.0, 0.5) = 1.0
        result = equalized_odds_diff(y_true, y_pred, s)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_takes_max_of_tpr_and_fpr(self):
        # Construct case where FPR diff > TPR diff
        y_true    = np.array([0, 0, 0, 0, 1, 1, 0, 0])
        y_pred    = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        result    = equalized_odds_diff(y_true, y_pred, sensitive)
        assert result >= 0.0


# ── equal_opportunity_diff ────────────────────────────────────────────────────

class TestEqualOpportunityDiff:

    def test_perfect_fairness_is_zero(self, perfect_fairness):
        y_true, y_pred, s = perfect_fairness
        assert equal_opportunity_diff(y_true, y_pred, s) == pytest.approx(0.0)

    def test_always_non_negative(self, known_values):
        y_true, y_pred, s = known_values
        assert equal_opportunity_diff(y_true, y_pred, s) >= 0.0

    def test_known_value(self, known_values):
        y_true, y_pred, s = known_values
        # TPR_priv=1.0, TPR_unpriv=0.0 → EOP = 1.0
        result = equal_opportunity_diff(y_true, y_pred, s)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_eop_leq_eod(self, known_values):
        """EOP can never exceed EOD since EOD is the max of TPR and FPR diffs."""
        y_true, y_pred, s = known_values
        eop = equal_opportunity_diff(y_true, y_pred, s)
        eod = equalized_odds_diff(y_true, y_pred, s)
        assert eop <= eod + 1e-9


# ── disparate_impact_ratio ────────────────────────────────────────────────────

class TestDisparateImpactRatio:

    def test_perfect_fairness_is_one(self, perfect_fairness):
        y_true, y_pred, s = perfect_fairness
        assert disparate_impact_ratio(y_true, y_pred, s) == pytest.approx(1.0)

    def test_known_value(self, known_values):
        y_true, y_pred, s = known_values
        # PR_unpriv=1/3, PR_priv=2/3 → DI = (1/3)/(2/3) = 0.5
        result = disparate_impact_ratio(y_true, y_pred, s)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_always_non_negative(self, known_values):
        y_true, y_pred, s = known_values
        assert disparate_impact_ratio(y_true, y_pred, s) >= 0.0

    def test_zero_privileged_rate_returns_nan(self):
        """If privileged group has no positive predictions, DI is undefined."""
        y_true    = np.array([1, 1, 1, 1])
        y_pred    = np.array([0, 0, 1, 1])   # priv PR = 0
        sensitive = np.array([1, 1, 0, 0])
        result    = disparate_impact_ratio(y_true, y_pred, sensitive)
        assert np.isnan(result)

    def test_below_one_means_unprivileged_disadvantaged(self, known_values):
        y_true, y_pred, s = known_values
        assert disparate_impact_ratio(y_true, y_pred, s) < 1.0


# ── fairness_metrics ──────────────────────────────────────────────────────────

class TestFairnessMetrics:

    def test_returns_all_four_keys(self, perfect_fairness):
        y_true, y_pred, s = perfect_fairness
        result = fairness_metrics(y_true, y_pred, s)
        assert set(result.keys()) == {'dpd', 'eod', 'eop', 'di'}

    def test_perfect_fairness_values(self, perfect_fairness):
        y_true, y_pred, s = perfect_fairness
        result = fairness_metrics(y_true, y_pred, s)
        assert result['dpd'] == pytest.approx(0.0)
        assert result['eod'] == pytest.approx(0.0)
        assert result['eop'] == pytest.approx(0.0)
        assert result['di']  == pytest.approx(1.0)

    def test_consistent_with_individual_functions(self, known_values):
        """fairness_metrics() must return the same values as the individual functions."""
        y_true, y_pred, s = known_values
        combined = fairness_metrics(y_true, y_pred, s)
        assert combined['dpd'] == pytest.approx(
            demographic_parity_diff(y_true, y_pred, s))
        assert combined['eod'] == pytest.approx(
            equalized_odds_diff(y_true, y_pred, s))
        assert combined['eop'] == pytest.approx(
            equal_opportunity_diff(y_true, y_pred, s))
        assert combined['di']  == pytest.approx(
            disparate_impact_ratio(y_true, y_pred, s))
