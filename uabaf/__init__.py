"""
UABAF — Uncertainty-Aware Bias Auditing Framework
==================================================
A framework for auditing fairness in machine learning models
trained on small datasets, using BCa bootstrap confidence intervals
to communicate uncertainty alongside fairness metric point estimates.

Basic usage
-----------
>>> from uabaf import AuditReport
>>> report = AuditReport(model, X_test, y_test, sensitive=s_test)
>>> report.summary()
>>> report.plot()

Modules
-------
uabaf.profiler   — Stage 1: dataset readiness checks
uabaf.metrics    — fairness metric functions (DPD, EOD, EOP, DI)
uabaf.bootstrap  — BCa bootstrap confidence interval engine
uabaf.verdict    — thresholds and verdict assignment logic
uabaf.auditor    — Stage 2+3 orchestration and AuditReport class
uabaf.visualise  — CI interval plots and audit visualisations
"""

__version__ = "0.1.0"
__author__  = "Balgah Sounders Junior"

# Populated progressively as modules are added.
# Each import below is uncommented when its module file is created.

from .auditor import AuditReport                # noqa: F401
from .profiler import profile_dataset       # noqa: F401
from .metrics import fairness_metrics       # noqa: F401
from .bootstrap import bca_ci, bca_ci_all_metrics  # noqa: F401
from .verdict import assign_verdict, assign_all_verdicts  # noqa: F401
from .visualise import plot_ci_intervals, plot_verdict_summary  # noqa: F401

__all__ = [
    "__version__",
]
