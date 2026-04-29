"""Per-marginal pre-standardisation to a standard Pareto distribution.

The transform composes two stages applied **per ambient coordinate**:

1. **Probability integral transform (PIT)** mapping the marginal to
   ``Uniform(0,1)``. We use a semi-parametric Coles (2001 §4) construction:

   - **Bulk** (samples below the threshold ``u_j`` chosen at quantile
     ``threshold_quantile``): rank-based empirical CDF with the Weibull
     plotting position ``F̂(x) = rank(x) / (n+1)``.

   - **Tail** (samples above ``u_j``): analytic Generalised Pareto CDF
     fitted by ``lib.extremes.fit_gpd_pot``. The combined CDF is

         F̂(x) = (1 - rate)                                if x ≤ u_j
         F̂(x) = (1 - rate) + rate * G(x - u_j; xi, sigma)  if x > u_j

     where ``rate`` is the empirical exceedance rate and ``G`` is the
     GPD CDF with shape ``xi`` and scale ``sigma``.

   This is the standard extreme-value PIT: rank-based in the bulk for
   robustness, GPD in the tail so out-of-sample tail extrapolation is
   principled rather than capped at the empirical maximum.

2. **Pareto inverse CDF**, with two target families selected by
   ``pareto_kind``:

   - ``"one_sided"`` (default): standard Pareto
     ``F^{-1}(u) = (1 - u)^{-1/alpha}``. Output support ``[1, ∞)``.
   - ``"two_sided"``: continuous symmetric Lomax,
     ``F^{-1}(u) = sign(2u-1) * ((1 - |2u-1|)^{-1/alpha} - 1)``. Output
     support ``ℝ``, both tails regularly varying with index ``alpha``.
     ``F^{-1}(0.5) = 0`` and the inverse is smooth at zero. Use this
     when the input has both signs and the geometric symmetry around
     the origin should be preserved.

   The joint copula is preserved by either kind.

The transform fits on a train batch and is then applied verbatim to val
/ test so that no information from the held-out splits leaks into the
empirical CDF or the GPD parameters.

The two-stage rationale: the rank-based bulk captures the empirical
copula faithfully, while the GPD tail handles deep-tail bias and
out-of-sample extremes — both of which a plain rank PIT mishandles.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import torch

from .extremes import fit_gpd_pot

_EPS = 1e-6
_VALID_KINDS = ("one_sided", "two_sided")


def _pareto_inverse_cdf(u: np.ndarray, *, alpha: float, kind: str) -> np.ndarray:
    """Map Uniform(0,1) -> Pareto target. ``kind`` selects one_sided / two_sided."""
    if kind == "one_sided":
        return (1.0 - u) ** (-1.0 / alpha)
    if kind == "two_sided":
        sign = np.where(u > 0.5, 1.0, -1.0)
        v = np.clip(1.0 - np.abs(2.0 * u - 1.0), _EPS, 1.0)
        return sign * (v ** (-1.0 / alpha) - 1.0)
    raise ValueError(
        f"unknown pareto_kind={kind!r}, expected one of {_VALID_KINDS}"
    )


def _as_numpy_2d(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected (n,) or (n,d) array, got shape {arr.shape}")
    return arr


def fit_pareto_marginal_transform(
    fit_x: np.ndarray | torch.Tensor,
    *,
    pareto_alpha: float = 1.0,
    threshold_quantile: float = 0.975,
    pareto_kind: str = "one_sided",
) -> dict:
    """Fit the per-marginal GPD-tail PIT on ``fit_x`` (train batch).

    Returns a dict carrying everything needed to evaluate the transform
    on new data: per-column sorted bulk values (for rank lookup), the
    fitted GPD parameters, the target Pareto ``alpha``, and the
    ``pareto_kind`` selecting one-sided vs two-sided Pareto inverse.
    Columns where the GPD fit fails (e.g. degenerate variance) fall
    back to a pure rank-based PIT.
    """
    if pareto_kind not in _VALID_KINDS:
        raise ValueError(
            f"unknown pareto_kind={pareto_kind!r}, expected one of {_VALID_KINDS}"
        )
    arr = _as_numpy_2d(fit_x)
    n, d = arr.shape
    columns: list[dict] = []
    for j in range(d):
        sorted_col = np.sort(arr[:, j])
        threshold = float(np.quantile(sorted_col, threshold_quantile))
        rate = float(np.mean(arr[:, j] > threshold))
        try:
            gpd = fit_gpd_pot(arr[:, j], threshold_quantile=threshold_quantile)
            fitted = True
            shape = gpd["shape"]
            scale = gpd["scale"]
        except (ValueError, RuntimeError) as exc:
            warnings.warn(
                f"GPD fit failed on column {j} ({exc}); falling back to "
                f"rank-based PIT for this margin.", stacklevel=2,
            )
            fitted = False
            shape = 0.0
            scale = 1.0
        columns.append({
            "sorted": sorted_col,
            "n": n,
            "threshold": threshold,
            "rate": rate,
            "shape": shape,
            "scale": scale,
            "gpd_fitted": fitted,
        })
    return {
        "columns": columns,
        "alpha": float(pareto_alpha),
        "threshold_quantile": float(threshold_quantile),
        "pareto_kind": pareto_kind,
        "n_dims": d,
    }


def _empirical_cdf(value: np.ndarray, sorted_fit: np.ndarray, n: int) -> np.ndarray:
    """Weibull-position rank-based F̂(x) = rank(x) / (n+1) on a sorted fit batch."""
    ranks = np.searchsorted(sorted_fit, value, side="right")
    return ranks.astype(np.float64) / float(n + 1)


def _gpd_cdf(excess: np.ndarray, *, shape: float, scale: float) -> np.ndarray:
    """G(y; xi, sigma) for y >= 0; with the xi -> 0 limit."""
    if abs(shape) < 1e-10:
        return 1.0 - np.exp(-excess / scale)
    base = 1.0 + shape * excess / scale
    return 1.0 - np.power(np.maximum(base, 0.0), -1.0 / shape)


def apply_pareto_marginal_transform(
    x: np.ndarray | torch.Tensor,
    fitted: dict,
    *,
    eps: float = _EPS,
) -> torch.Tensor:
    """Apply the fitted GPD-tail PIT, then the standard Pareto inverse CDF.

    Inputs above the per-column threshold are evaluated through the GPD
    CDF; inputs at-or-below threshold use the rank-based bulk CDF. The
    returned tensor has the same shape as ``x`` and dtype ``float32``.
    """
    arr = _as_numpy_2d(x)
    if arr.shape[1] != fitted["n_dims"]:
        raise ValueError(
            f"input has {arr.shape[1]} dims, fit was on {fitted['n_dims']} dims"
        )
    alpha = fitted["alpha"]
    kind = fitted.get("pareto_kind", "one_sided")
    out = np.empty_like(arr, dtype=np.float64)

    for j, col_fit in enumerate(fitted["columns"]):
        col = arr[:, j]
        u_col = np.empty_like(col, dtype=np.float64)
        threshold = col_fit["threshold"]
        rate = col_fit["rate"]

        bulk_mask = col <= threshold
        if bulk_mask.any():
            u_col[bulk_mask] = _empirical_cdf(
                col[bulk_mask], col_fit["sorted"], col_fit["n"],
            )
        tail_mask = ~bulk_mask
        if tail_mask.any():
            if col_fit["gpd_fitted"]:
                gpd = _gpd_cdf(
                    col[tail_mask] - threshold,
                    shape=col_fit["shape"], scale=col_fit["scale"],
                )
                u_col[tail_mask] = (1.0 - rate) + rate * gpd
            else:
                u_col[tail_mask] = _empirical_cdf(
                    col[tail_mask], col_fit["sorted"], col_fit["n"],
                )

        u_col = np.clip(u_col, eps, 1.0 - eps)
        out[:, j] = _pareto_inverse_cdf(u_col, alpha=alpha, kind=kind)

    return torch.as_tensor(out, dtype=torch.float32)


def fit_apply_pareto_margins(
    train_x: np.ndarray | torch.Tensor,
    others: Sequence[np.ndarray | torch.Tensor] = (),
    *,
    pareto_alpha: float = 1.0,
    threshold_quantile: float = 0.975,
    pareto_kind: str = "one_sided",
) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
    """Convenience: fit on ``train_x`` then apply to it and to ``others``.

    Returns ``(train_transformed, [other_transformed, ...], fitted)``.
    The ``fitted`` dict can be persisted alongside artifacts for later
    inversion or replication.
    """
    fitted = fit_pareto_marginal_transform(
        train_x,
        pareto_alpha=pareto_alpha,
        threshold_quantile=threshold_quantile,
        pareto_kind=pareto_kind,
    )
    train_t = apply_pareto_marginal_transform(train_x, fitted)
    others_t = [apply_pareto_marginal_transform(o, fitted) for o in others]
    return train_t, others_t, fitted
