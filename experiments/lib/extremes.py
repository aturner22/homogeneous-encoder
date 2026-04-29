"""Extreme-value fits for return-level plots.

Default method: Peaks-Over-Threshold (POT) with a Generalized Pareto
distribution (GPD). For sample sizes typical of ERA5 reanalysis
(~35k obs per variable) this gives much tighter CIs on return levels
than annual block-maxima GEV fits (24 blocks → shape SE 0.2–0.3).
A GEV block-maxima alternative is provided for appendix robustness.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def fit_gpd_pot(
    x: np.ndarray,
    threshold_quantile: float = 0.975,
) -> dict[str, float]:
    """Fit a GPD to exceedances of a high threshold.

    Returns ``shape`` (xi), ``scale`` (sigma), ``threshold`` (u),
    ``n_exceedances``, and ``rate`` (exceedances / total observations).
    """
    x = np.asarray(x, dtype=np.float64)
    u = float(np.quantile(x, threshold_quantile))
    exc = x[x > u] - u
    if exc.size < 10:
        raise ValueError(
            f"too few exceedances ({exc.size}) for GPD fit at q={threshold_quantile}"
        )
    shape, _, scale = stats.genpareto.fit(exc, floc=0.0)
    return {
        "shape": float(shape),
        "scale": float(scale),
        "threshold": u,
        "n_exceedances": int(exc.size),
        "rate": float(exc.size) / float(x.size),
    }


def return_level_gpd(
    params: dict[str, float],
    return_periods: np.ndarray,
    n_per_year: float,
) -> np.ndarray:
    """Return levels r(T) for return periods T (years) under POT/GPD.

    Formula:  r(T) = u + (sigma / xi) * ((T * n_per_year * rate)^xi - 1)
    with the xi→0 limit r(T) = u + sigma * log(T * n_per_year * rate).
    """
    u = params["threshold"]
    sigma = params["scale"]
    xi = params["shape"]
    rate = params["rate"]
    T = np.asarray(return_periods, dtype=np.float64)
    m = T * n_per_year * rate
    if abs(xi) < 1e-6:
        return u + sigma * np.log(m)
    return u + (sigma / xi) * (m ** xi - 1.0)


def fit_gev_bm(x: np.ndarray, block_size: int) -> dict[str, float]:
    """Fit a GEV to block maxima. ``block_size`` is samples per block."""
    x = np.asarray(x, dtype=np.float64)
    n_blocks = x.size // block_size
    if n_blocks < 5:
        raise ValueError(f"too few blocks ({n_blocks}) for GEV fit")
    trimmed = x[: n_blocks * block_size].reshape(n_blocks, block_size)
    maxima = trimmed.max(axis=1)
    shape, loc, scale = stats.genextreme.fit(maxima)
    # scipy's genextreme uses the opposite shape sign convention to EVT.
    return {
        "shape": float(-shape),
        "loc": float(loc),
        "scale": float(scale),
        "n_blocks": int(n_blocks),
    }


def return_level_gev(
    params: dict[str, float],
    return_periods: np.ndarray,
) -> np.ndarray:
    """Return levels r(T) (years) under annual-block-maxima GEV."""
    mu = params["loc"]
    sigma = params["scale"]
    xi = params["shape"]
    T = np.asarray(return_periods, dtype=np.float64)
    y = -np.log(1.0 - 1.0 / T)
    if abs(xi) < 1e-6:
        return mu - sigma * np.log(y)
    return mu + (sigma / xi) * (y ** (-xi) - 1.0)


def return_level_ci(
    x: np.ndarray,
    return_periods: np.ndarray,
    *,
    n_per_year: float,
    threshold_quantile: float = 0.975,
    n_boot: int = 500,
    ci: float = 0.90,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap CI bands for POT/GPD return levels.

    Returns (lower, upper) arrays aligned with ``return_periods``.
    """
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    T = np.asarray(return_periods, dtype=np.float64)
    boots = np.empty((n_boot, T.size), dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(x, size=x.size, replace=True)
        try:
            params = fit_gpd_pot(sample, threshold_quantile=threshold_quantile)
            boots[b] = return_level_gpd(params, T, n_per_year=n_per_year)
        except ValueError:
            boots[b] = np.nan
    alpha = 1.0 - ci
    lo = np.nanquantile(boots, alpha / 2, axis=0)
    hi = np.nanquantile(boots, 1.0 - alpha / 2, axis=0)
    return lo, hi
