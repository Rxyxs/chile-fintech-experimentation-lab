"""Multiple-testing correction across guardrail metrics.

A real experiment is never read off one metric: a primary metric plus several
guardrails (latency, unsubscribe rate, complaint rate, revenue) are tested
together, and checking each at a raw alpha=0.05 inflates the family-wise false
positive rate directly with the number of guardrails. Benjamini-Hochberg
controls the *expected proportion* of false discoveries among the rejected
hypotheses (FDR), which is less conservative than Bonferroni and standard
practice for experimentation guardrail panels.
"""
from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> dict:
    """Returns which hypotheses are rejected under BH-FDR control at `alpha`,
    plus the BH-adjusted ("q") value for each, in the original input order.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return {"reject": np.array([], dtype=bool), "q_values": np.array([])}

    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha

    below = ranked <= thresholds
    if below.any():
        k_max = np.max(np.where(below)[0])
        reject_ranked = np.zeros(m, dtype=bool)
        reject_ranked[: k_max + 1] = True
    else:
        reject_ranked = np.zeros(m, dtype=bool)

    q_ranked = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    reject = np.empty(m, dtype=bool)
    q_values = np.empty(m, dtype=float)
    reject[order] = reject_ranked
    q_values[order] = q_ranked

    return {"reject": reject, "q_values": q_values}
