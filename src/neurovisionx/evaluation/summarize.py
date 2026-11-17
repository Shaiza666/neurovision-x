"""Mean, standard deviation, and confidence intervals from recorded values.

Uses the sample standard deviation (ddof=1) and a Student-t interval when
n >= 2. A single observation has no CI. Empty input is not filled with zeros.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


def summarize_values(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    ddof: int = 1,
) -> dict[str, Any]:
    finite = [float(v) for v in values if v == v]
    n = len(finite)
    if n == 0:
        return {
            "mean": None,
            "std": None,
            "n": 0,
            "ci_low": None,
            "ci_high": None,
            "ci_level": float(confidence),
            "ci_method": None,
        }
    mean = sum(finite) / n
    if n == 1:
        return {
            "mean": mean,
            "std": None,
            "n": 1,
            "ci_low": None,
            "ci_high": None,
            "ci_level": float(confidence),
            "ci_method": "undefined_n_lt_2",
        }
    var = sum((x - mean) ** 2 for x in finite) / (n - ddof) if n > ddof else 0.0
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    try:
        from scipy.stats import t

        tcrit = float(t.ppf((1.0 + confidence) / 2.0, n - 1))
        method = "student_t"
    except ImportError:
        tcrit = 1.959963984540054
        method = "normal_approx_scipy_missing"
    return {
        "mean": mean,
        "std": std,
        "n": n,
        "ci_low": mean - tcrit * se,
        "ci_high": mean + tcrit * se,
        "ci_level": float(confidence),
        "ci_method": method,
    }


def format_summary(row: dict[str, Any] | None, *, digits: int = 3) -> str:
    """Human-readable cell. Missing measurements stay 'not measured'."""
    if not row or not row.get("n"):
        return "not measured"
    mean = row.get("mean")
    if mean is None:
        return "not measured"
    n = int(row["n"])
    std = row.get("std")
    body = f"{mean:.{digits}f}"
    if std is not None:
        body = f"{body} ± {std:.{digits}f}"
    if row.get("ci_low") is not None and row.get("ci_high") is not None:
        body = f"{body} [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"
    return f"{body} (n={n})"
