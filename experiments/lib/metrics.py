"""Tail-preservation evaluation metrics.

The first block replicates the four metrics from the paper's original
plan (Hill estimator, extreme quantile errors, angular tail distance,
encoder homogeneity check). The second block adds the tail-conditional
metrics that actually separate HAE from the StandardAE baseline:

- ``binned_reconstruction_error`` - per-radial-bin reconstruction MSE;
- ``tail_conditional_mse`` - MSE restricted to the top radial quantile;
- ``extrapolation_mse`` - reconstruction MSE at out-of-training scales
  synthesised through the true embedding module;
- ``tail_angular_coordinates`` - 2-D principal-axis projection of the
  true tail cone, reused for every model so panels are comparable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Division-safety floors. Two regimes intentionally:
#   - NORM_EPS guards first-order norms / unit-vector normalisations where
#     1e-12 is well above float64 noise for the magnitudes we touch.
#   - SQNORM_EPS guards squared-norm denominators (``sum(x**2)``) which can
#     legitimately underflow 1e-24 on tiny inputs; 1e-30 stays below the
#     smallest meaningful squared magnitude without hitting subnormals.
NORM_EPS: float = 1e-12
SQNORM_EPS: float = 1e-30


# -----------------------------------------------------------------------------
# Hill estimator
# -----------------------------------------------------------------------------


def hill_curve(radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (k_values, alpha_hat_k) over k = 2 ... n-1.

    Uses positive radii only. Sorted in decreasing order so that
    X_(1) >= X_(2) >= ... >= X_(n).

    alpha_hat_k = 1 / (1/k * sum_{i=1}^k log(X_(i) / X_(k+1)))
    """
    radii = np.asarray(radii, dtype=np.float64)
    radii = radii[radii > 0.0]
    if radii.size < 10:
        raise ValueError(f"hill_curve needs at least 10 positive radii, got {radii.size}")

    sorted_desc = np.sort(radii)[::-1]
    n = sorted_desc.size

    k_values = np.arange(2, n, dtype=np.int64)
    log_order_stats = np.log(sorted_desc)

    # cumulative sum of log X_(i) for i = 1..k
    cumulative_log = np.cumsum(log_order_stats)
    # H_k = (1/k) * sum_{i=1}^k log(X_(i) / X_(k+1))
    #     = (cumulative_log[k-1] / k) - log_order_stats[k]
    numerators = cumulative_log[k_values - 1] / k_values - log_order_stats[k_values]
    alpha_hat_k = 1.0 / numerators
    return k_values.astype(np.float64), alpha_hat_k


def hill_drift(alpha_latent: float, alpha_ambient: float, p: float) -> float:
    """Proposition 1 consistency: |alpha_latent * p - alpha_ambient|.

    If the encoder is exactly p-homogeneous then by Proposition 1 the
    latent tail index is alpha_ambient / p, so ``alpha_latent * p``
    should equal ``alpha_ambient``. This metric measures the drift from
    that identity without depending on knowing the true ambient alpha
    (Hill estimates are biased in finite samples).
    """
    return float(abs(float(alpha_latent) * float(p) - float(alpha_ambient)))


def hill_estimate(
    radii: np.ndarray, k_fraction: float = 0.1
) -> dict[str, float]:
    """Point estimate of tail index alpha via the Hill estimator.

    Uses the top ``k = max(10, floor(k_fraction * n))`` order statistics.
    Also returns a 95% normal confidence interval using the asymptotic
    variance ``alpha^2 / k``.
    """
    radii = np.asarray(radii, dtype=np.float64)
    radii = radii[radii > 0.0]
    n = int(radii.size)
    if n < 20:
        raise ValueError(f"hill_estimate needs at least 20 positive radii, got {n}")

    k = max(10, int(np.floor(k_fraction * n)))
    k = min(k, n - 1)

    sorted_desc = np.sort(radii)[::-1]
    log_top_k = np.log(sorted_desc[:k])
    log_threshold = np.log(sorted_desc[k])
    hill_denominator = float(np.mean(log_top_k) - log_threshold)
    if abs(hill_denominator) < 1e-15:
        alpha_hat = float("inf")
    else:
        alpha_hat = 1.0 / hill_denominator

    std_err = alpha_hat / float(np.sqrt(k))
    return {
        "alpha": float(alpha_hat),
        "k": int(k),
        "n": int(n),
        "std_err": float(std_err),
        "ci_low": float(alpha_hat - 1.96 * std_err),
        "ci_high": float(alpha_hat + 1.96 * std_err),
    }


# -----------------------------------------------------------------------------
# Extreme quantile errors
# -----------------------------------------------------------------------------


def extreme_quantile_errors(
    reference: np.ndarray,
    candidate: np.ndarray,
    levels: Sequence[float] = (0.99, 0.999, 0.9999),
) -> dict[str, dict[str, float]]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    for level in levels:
        q_ref = float(np.quantile(reference, level))
        q_cand = float(np.quantile(candidate, level))
        abs_error = abs(q_cand - q_ref)
        denom = max(abs(q_ref), NORM_EPS)
        rel_error = abs_error / denom
        out[f"q{level}"] = {
            "reference": q_ref,
            "candidate": q_cand,
            "abs_error": float(abs_error),
            "rel_error": float(rel_error),
        }
    return out


# -----------------------------------------------------------------------------
# Angular tail distance (sliced Wasserstein on the unit sphere)
# -----------------------------------------------------------------------------


def _select_tail_directions(
    x: np.ndarray, radial_quantile: float
) -> np.ndarray:
    radii = np.linalg.norm(x, axis=1)
    threshold = float(np.quantile(radii, radial_quantile))
    mask = radii >= threshold
    if mask.sum() < 10:
        raise ValueError(
            f"Not enough samples above radial quantile {radial_quantile}: {int(mask.sum())}"
        )
    tail_points = x[mask]
    tail_norms = np.linalg.norm(tail_points, axis=1, keepdims=True)
    return tail_points / np.maximum(tail_norms, NORM_EPS)


def angular_tail_distance(
    x: np.ndarray,
    x_other: np.ndarray,
    *,
    radial_quantile: float = 0.95,
    num_slices: int = 100,
    seed: int = 0,
    centre: np.ndarray | None = None,
) -> float:
    """Sliced Wasserstein distance between empirical angular tail measures.

    Both ``x`` and ``x_other`` are (N, D) arrays in the *same* ambient
    space. The top ``1 - radial_quantile`` fraction is selected from each
    by recentred radius (``||x - centre||``; ``centre = 0`` if not given),
    projected onto the unit sphere of the recentred frame, and a 1-D
    Wasserstein distance is computed along ``num_slices`` random unit
    directions. The result is the mean over slices.
    """
    x = np.asarray(x, dtype=np.float64)
    x_other = np.asarray(x_other, dtype=np.float64)
    if x.shape[1] != x_other.shape[1]:
        raise ValueError(
            f"Ambient dims differ: {x.shape[1]} vs {x_other.shape[1]}"
        )
    if centre is None:
        centre = np.zeros(x.shape[1], dtype=np.float64)
    else:
        centre = np.asarray(centre, dtype=np.float64).reshape(x.shape[1])

    directions_a = _select_tail_directions(x - centre, radial_quantile)
    directions_b = _select_tail_directions(x_other - centre, radial_quantile)

    rng = np.random.default_rng(int(seed))
    slice_vectors = rng.standard_normal(size=(num_slices, x.shape[1]))
    slice_vectors /= np.maximum(
        np.linalg.norm(slice_vectors, axis=1, keepdims=True), NORM_EPS
    )

    projected_a = directions_a @ slice_vectors.T
    projected_b = directions_b @ slice_vectors.T

    # resample both sets to the same length by interpolating sorted values
    sorted_a = np.sort(projected_a, axis=0)
    sorted_b = np.sort(projected_b, axis=0)
    n_common = max(sorted_a.shape[0], sorted_b.shape[0])
    grid = np.linspace(0.0, 1.0, n_common, dtype=np.float64)
    grid_a = np.linspace(0.0, 1.0, sorted_a.shape[0], dtype=np.float64)
    grid_b = np.linspace(0.0, 1.0, sorted_b.shape[0], dtype=np.float64)

    distances = []
    for slice_index in range(num_slices):
        interp_a = np.interp(grid, grid_a, sorted_a[:, slice_index])
        interp_b = np.interp(grid, grid_b, sorted_b[:, slice_index])
        distances.append(float(np.mean(np.abs(interp_a - interp_b))))

    return float(np.mean(distances))


# -----------------------------------------------------------------------------
# Encoder homogeneity diagnostic
# -----------------------------------------------------------------------------


@torch.no_grad()
def encoder_homogeneity_error(
    model: nn.Module,
    x_probe: torch.Tensor,
    p: float,
    scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
) -> dict[str, float]:
    """Measure ``|| f(lambda*(x-c)+c) - lambda^p f(x) ||^2 / || lambda^p f(x) ||^2``.

    The encoder ``f(x) = g(x - c)`` is ``p``-homogeneous in the recentred
    coordinate, so we scale ``(x - c)`` by ``lambda`` and add ``c`` back
    before encoding. Models without a ``centre`` attribute use ``c = 0``,
    which reduces to the original raw-frame test. Returns a dict keyed by
    the scale with the mean relative squared error on the probe batch,
    plus the worst case across scales.
    """
    model.eval()
    centre = getattr(model, "centre", None)
    if centre is None:
        centre = torch.zeros(x_probe.shape[1], device=x_probe.device, dtype=x_probe.dtype)
    else:
        centre = centre.to(device=x_probe.device, dtype=x_probe.dtype)

    encoded = model.encode(x_probe)
    base_z = encoded["z"]

    errors: dict[str, float] = {}
    worst = 0.0
    for scale in scales:
        scaled_input = scale * (x_probe - centre) + centre
        scaled_z_reference = (scale ** p) * base_z
        scaled_encoded = model.encode(scaled_input)["z"]
        numerator = torch.sum((scaled_encoded - scaled_z_reference) ** 2, dim=1)
        denominator = torch.sum(scaled_z_reference ** 2, dim=1) + NORM_EPS
        relative_squared = torch.mean(numerator / denominator).item()
        errors[f"scale_{scale}"] = float(relative_squared)
        worst = max(worst, float(relative_squared))

    errors["worst"] = worst
    return errors


# -----------------------------------------------------------------------------
# Tail-conditional reconstruction metrics
# -----------------------------------------------------------------------------


def binned_reconstruction_error(
    x: np.ndarray,
    x_hat: np.ndarray,
    *,
    n_bins: int = 12,
    log_bins: bool = True,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.9999,
    centre: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-sample ``||x - x_hat||^2`` aggregated into radial bins.

    Bins use the recentred radius ``||x - centre||`` (``centre = 0`` if
    not given). x-axis of the returned curve is the geometric mean of
    each bin edge (log_bins=True) or the midpoint (log_bins=False).
    y-axis returns median and IQR per bin, plus the sample count so the
    caller can down-weight or drop under-populated bins.
    """
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    if x.shape != x_hat.shape:
        raise ValueError(
            f"shape mismatch: {x.shape} vs {x_hat.shape}"
        )
    if centre is None:
        centre = np.zeros(x.shape[1], dtype=np.float64)
    else:
        centre = np.asarray(centre, dtype=np.float64).reshape(x.shape[1])

    radii = np.linalg.norm(x - centre, axis=1)
    per_sample_sq_err = np.sum((x - x_hat) ** 2, axis=1)

    low_edge = float(np.quantile(radii, lower_quantile))
    high_edge = float(np.quantile(radii, upper_quantile))
    if not np.isfinite(low_edge) or not np.isfinite(high_edge) or high_edge <= low_edge:
        raise ValueError(
            f"degenerate bin range: [{low_edge}, {high_edge}]"
        )

    if log_bins:
        edges = np.geomspace(low_edge, high_edge, n_bins + 1)
    else:
        edges = np.linspace(low_edge, high_edge, n_bins + 1)

    bin_indices = np.digitize(radii, edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    centers = np.sqrt(edges[:-1] * edges[1:]) if log_bins else 0.5 * (edges[:-1] + edges[1:])
    median = np.full(n_bins, np.nan, dtype=np.float64)
    q25 = np.full(n_bins, np.nan, dtype=np.float64)
    q75 = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        counts[bin_index] = count
        if count == 0:
            continue
        errors_in_bin = per_sample_sq_err[mask]
        median[bin_index] = float(np.median(errors_in_bin))
        q25[bin_index] = float(np.quantile(errors_in_bin, 0.25))
        q75[bin_index] = float(np.quantile(errors_in_bin, 0.75))

    return {
        "centers": centers,
        "edges": edges,
        "median": median,
        "q25": q25,
        "q75": q75,
        "counts": counts,
    }


def tail_conditional_mse(
    x: np.ndarray,
    x_hat: np.ndarray,
    *,
    radial_quantile: float = 0.95,
    centre: np.ndarray | None = None,
) -> float:
    """Scalar reconstruction MSE on samples with ``||x - centre|| >= q``-quantile."""
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    if centre is None:
        centre = np.zeros(x.shape[1], dtype=np.float64)
    else:
        centre = np.asarray(centre, dtype=np.float64).reshape(x.shape[1])
    radii = np.linalg.norm(x - centre, axis=1)
    threshold = float(np.quantile(radii, radial_quantile))
    mask = radii >= threshold
    if mask.sum() < 10:
        raise ValueError(
            f"tail slice too small: {int(mask.sum())} samples above q={radial_quantile}"
        )
    diff = x[mask] - x_hat[mask]
    return float(np.mean(diff ** 2))


@torch.no_grad()
def extrapolation_mse(
    model: nn.Module,
    embedding: nn.Module,
    *,
    D: int,
    m: int,
    scale_multipliers: Sequence[float],
    n_samples: int = 2048,
    base_radius: float = 1.0,
    sample_seed: int = 999,
    device: str = "cpu",
) -> dict[str, Any]:
    """Reconstruction MSE on samples drawn at out-of-training radii.

    We draw ``n_samples`` directions ``u ~ Uniform(S^{m-1})`` once, then
    for every ``lambda`` in ``scale_multipliers`` build
    ``y = lambda * base_radius * u`` and feed ``x = embedding(y)`` to
    the model. MSE is reported per scale. For HAE with ``p=1`` this is
    bounded by the angular correction's residual; for StandardAE it
    grows monotonically once lambda leaves the training support.
    """
    model.eval()
    embedding = embedding.to(device)
    rng = np.random.default_rng(int(sample_seed))
    directions = rng.standard_normal(size=(n_samples, m)).astype(np.float32)
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), NORM_EPS
    )
    directions_tensor = torch.from_numpy(directions).to(device)

    scales_array = np.asarray(list(scale_multipliers), dtype=np.float64)
    mse_values = np.full(scales_array.shape, np.nan, dtype=np.float64)

    for index, lam in enumerate(scales_array):
        y = float(lam) * float(base_radius) * directions_tensor
        x = embedding(y)
        forward_pass = model(x)
        x_hat = forward_pass["x_hat"]
        mse_values[index] = float(torch.mean((x - x_hat) ** 2).item())

    return {
        "scales": scales_array,
        "mse": mse_values,
        "base_radius": float(base_radius),
    }


def tail_angular_coordinates(
    x_true: np.ndarray,
    x_by_model: Mapping[str, np.ndarray],
    *,
    radial_quantile: float = 0.95,
    rank: int = 2,
) -> dict[str, np.ndarray]:
    """Project the top-quantile tail cones of several models onto a common basis.

    The projection basis is computed from the *true* tail directions so
    every model is scattered in the same plane. Points from a model
    whose angular distribution matches the truth cover the same cloud;
    collapsing or distorted clouds are immediately visible.
    """
    x_true = np.asarray(x_true, dtype=np.float64)
    true_radii = np.linalg.norm(x_true, axis=1)
    threshold = float(np.quantile(true_radii, radial_quantile))
    true_mask = true_radii >= threshold
    if true_mask.sum() < 10:
        raise ValueError(
            f"tail slice too small: {int(true_mask.sum())} samples above q={radial_quantile}"
        )

    true_tail = x_true[true_mask]
    true_directions = true_tail / np.maximum(
        np.linalg.norm(true_tail, axis=1, keepdims=True), NORM_EPS
    )
    centered = true_directions - true_directions.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection_basis = vh[:rank].T

    output: dict[str, np.ndarray] = {
        "basis": projection_basis,
        "truth": true_directions @ projection_basis,
    }

    for model_name, x_model in x_by_model.items():
        x_model = np.asarray(x_model, dtype=np.float64)
        if x_model.shape != x_true.shape:
            raise ValueError(
                f"shape mismatch for {model_name}: {x_model.shape} vs {x_true.shape}"
            )
        model_tail = x_model[true_mask]
        model_directions = model_tail / np.maximum(
            np.linalg.norm(model_tail, axis=1, keepdims=True), NORM_EPS
        )
        output[model_name] = model_directions @ projection_basis
    return output
