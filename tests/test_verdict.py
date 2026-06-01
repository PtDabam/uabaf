"""
Tests for uabaf.verdict
"""

import numpy as np
import pytest
from uabaf.verdict import assign_verdict, assign_all_verdicts, THRESHOLDS


# ── assign_verdict ────────────────────────────────────────────────────────────

class TestAssignVerdict:

    def test_pass_high_confidence(self):
        result = assign_verdict('dpd', point=0.05, ci_lo=0.01, ci_hi=0.09)
        assert result['verdict']    == '✅ PASS — High Confidence'
        assert result['pass_fail']  == 'PASS'
        assert result['confidence'] == 'HIGH'
        assert result['breaches']   == False

    def test_pass_low_confidence(self):
        result = assign_verdict('dpd', point=0.05, ci_lo=-0.10, ci_hi=0.20)
        assert result['verdict']    == '⚠️  PASS — Low Confidence'
        assert result['pass_fail']  == 'PASS'
        assert result['confidence'] == 'LOW'

    def test_fail_high_confidence(self):
        result = assign_verdict('dpd', point=0.15, ci_lo=0.11, ci_hi=0.19)
        assert result['verdict']    == '❌ FAIL — High Confidence'
        assert result['pass_fail']  == 'FAIL'
        assert result['confidence'] == 'HIGH'
        assert result['breaches']   == True

    def test_fail_low_confidence(self):
        result = assign_verdict('dpd', point=0.15, ci_lo=0.02, ci_hi=0.28)
        assert result['verdict']    == '🔍 FAIL — Low Confidence'
        assert result['pass_fail']  == 'FAIL'
        assert result['confidence'] == 'LOW'

    def test_all_four_metrics_accepted(self):
        for metric in ['dpd', 'eod', 'eop', 'di']:
            # Use safe values that won't breach thresholds
            point = 0.85 if metric == 'di' else 0.05
            result = assign_verdict(metric, point=point,
                                    ci_lo=point - 0.03,
                                    ci_hi=point + 0.03)
            assert 'verdict' in result

    def test_di_within_range_passes(self):
        result = assign_verdict('di', point=0.90, ci_lo=0.85, ci_hi=0.95)
        assert result['pass_fail'] == 'PASS'

    def test_di_below_range_fails(self):
        result = assign_verdict('di', point=0.70, ci_lo=0.65, ci_hi=0.75)
        assert result['pass_fail'] == 'FAIL'

    def test_di_above_range_fails(self):
        result = assign_verdict('di', point=1.30, ci_lo=1.25, ci_hi=1.35)
        assert result['pass_fail'] == 'FAIL'

    def test_width_computed_correctly(self):
        result = assign_verdict('dpd', point=0.05, ci_lo=0.01, ci_hi=0.09)
        assert result['width'] == pytest.approx(0.08, abs=1e-6)

    def test_nan_ci_gives_low_confidence(self):
        result = assign_verdict('dpd', point=0.05,
                                ci_lo=np.nan, ci_hi=np.nan)
        assert result['confidence'] == 'LOW'

    def test_label_is_human_readable(self):
        result = assign_verdict('dpd', point=0.05, ci_lo=0.01, ci_hi=0.09)
        assert result['label'] == 'Demographic Parity Difference'

    def test_custom_threshold_override(self):
        """Users should be able to pass stricter thresholds."""
        strict = {'dpd': ('diff', 0.03), 'eod': ('diff', 0.03),
                  'eop': ('diff', 0.03), 'di': ('ratio', (0.95, 1.05))}
        # 0.05 passes default threshold but fails strict threshold
        default_result = assign_verdict('dpd', 0.05, 0.03, 0.07)
        strict_result  = assign_verdict('dpd', 0.05, 0.03, 0.07,
                                        thresholds=strict)
        assert default_result['pass_fail'] == 'PASS'
        assert strict_result['pass_fail']  == 'FAIL'


# ── assign_all_verdicts ───────────────────────────────────────────────────────

class TestAssignAllVerdicts:

    @pytest.fixture
    def mock_ci_results(self):
        return {
            'dpd': {'point': 0.05, 'ci_lo': 0.01, 'ci_hi': 0.09},
            'eod': {'point': 0.15, 'ci_lo': 0.11, 'ci_hi': 0.19},
            'eop': {'point': 0.08, 'ci_lo': 0.04, 'ci_hi': 0.12},
            'di' : {'point': 0.85, 'ci_lo': 0.81, 'ci_hi': 0.89},
        }

    def test_returns_all_four_metrics(self, mock_ci_results):
        verdicts = assign_all_verdicts(mock_ci_results)
        assert set(verdicts.keys()) == {'dpd', 'eod', 'eop', 'di'}

    def test_point_values_preserved(self, mock_ci_results):
        verdicts = assign_all_verdicts(mock_ci_results)
        for metric, res in mock_ci_results.items():
            assert verdicts[metric]['point'] == res['point']

    def test_ci_bounds_preserved(self, mock_ci_results):
        verdicts = assign_all_verdicts(mock_ci_results)
        for metric, res in mock_ci_results.items():
            assert verdicts[metric]['ci_lo'] == res['ci_lo']
            assert verdicts[metric]['ci_hi'] == res['ci_hi']

    def test_each_verdict_has_required_keys(self, mock_ci_results):
        verdicts = assign_all_verdicts(mock_ci_results)
        required = {'verdict', 'pass_fail', 'confidence',
                    'breaches', 'width', 'point', 'ci_lo', 'ci_hi'}
        for metric, v in verdicts.items():
            assert required.issubset(set(v.keys())), \
                f"Missing keys in verdict for {metric}"
