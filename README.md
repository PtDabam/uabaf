# UABAF — Uncertainty-Aware Bias Auditing Framework

A Python package for auditing fairness in machine learning models trained on
**small datasets**, using BCa bootstrap confidence intervals to communicate
uncertainty alongside fairness metric point estimates.

---

## The problem it solves

Standard fairness toolkits (AIF360, Fairlearn) report point estimates only:

```
Demographic Parity Difference: 0.12  ← FAIL
```

On small datasets, that number could easily shift from 0.08 to 0.16 with a
different random seed. UABAF adds a confidence interval and an
uncertainty-aware verdict:

```
Demographic Parity Difference: 0.12  CI: [0.04, 0.21]  ⚠️  FAIL — Low Confidence
```

---

## Installation

```bash
# Core package
pip install uabaf

# With AIF360 + Fairlearn comparison support
pip install uabaf[compare]

# For development
pip install uabaf[dev]
```

> **Note:** UABAF pins `numpy<2` because AIF360 requires it.

---

## Quick start

```python
from uabaf import AuditReport

# model  — any sklearn-compatible fitted classifier
# X_test — feature matrix (numpy array or DataFrame)
# y_test — true labels
# s_test — sensitive attribute vector (binary: 0=unprivileged, 1=privileged)

report = AuditReport(model, X_test, y_test, sensitive=s_test)
report.summary()   # prints Stage 1 + Stage 3 verdict table
report.plot()      # BCa CI interval plots for all metrics
```

---

## Fairness metrics

| Key   | Metric                         | Threshold      |
|-------|--------------------------------|----------------|
| `dpd` | Demographic Parity Difference  | \|dpd\| ≤ 0.10 |
| `eod` | Equalized Odds Difference      | \|eod\| ≤ 0.10 |
| `eop` | Equal Opportunity Difference   | \|eop\| ≤ 0.10 |
| `di`  | Disparate Impact Ratio         | 0.80 – 1.20    |

---

## Verdict categories

| Verdict                      | Meaning                                              |
|------------------------------|------------------------------------------------------|
| ✅ PASS — High Confidence    | Metric within threshold, narrow CI                   |
| ⚠️  PASS — Low Confidence    | Metric within threshold but wide CI — collect more data |
| ❌ FAIL — High Confidence    | Metric breaches threshold, narrow CI                 |
| 🔍 FAIL — Low Confidence     | Metric breaches threshold but wide CI — inconclusive |

---

## Research context

UABAF was developed as part of an M.Tech thesis at the University of Buea,
Department of Computer Engineering, under the supervision of Dr. Nyanga Bernard Y.

---

## License

MIT
