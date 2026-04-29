"""Sanity check: Hill estimator recovers alpha on synthetic Pareto draws."""

from __future__ import annotations

import numpy as np
from lib.metrics import hill_curve, hill_estimate


def test_hill_recovers_pareto_alpha() -> None:
    rng = np.random.default_rng(0)
    alpha_true = 2.5
    n = 20_000
    # Pareto(alpha) with scale 1: F(x) = 1 - x^{-alpha} for x >= 1.
    radii = (1.0 - rng.uniform(size=n)) ** (-1.0 / alpha_true)

    result = hill_estimate(radii)
    # default k = 10% of n
    assert 2.3 <= result["alpha"] <= 2.7, result

    k, alpha_hat = hill_curve(radii)
    # pick the estimate at k = 200 (deep enough to be stable but not too deep)
    idx = int(np.argmin(np.abs(k - 200)))
    assert 2.3 <= float(alpha_hat[idx]) <= 2.7, (
        f"alpha_hat[k=200] = {alpha_hat[idx]!r}"
    )
