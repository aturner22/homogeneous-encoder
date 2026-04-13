"""Shared evaluation routines for experiment scripts.

``evaluate_model`` runs a trained model on a held-out tensor and returns
a flat dict of metrics covering reconstruction quality, Hill estimates
(ambient / latent / reconstructed), extreme quantile errors, angular
tail distance and - for neural models - the numerical homogeneity check.

``train_and_evaluate`` is the full per-model pipeline used by every
experiment script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import TrainConfig
from .metrics import (
    angular_tail_distance,
    binned_reconstruction_error,
    encoder_homogeneity_error,
    extrapolation_mse,
    extreme_quantile_errors,
    hill_curve,
    hill_drift,
    hill_estimate,
    homogeneity_scan,
    tail_conditional_mse,
)
from .models import HomogeneousAutoencoder, PCABaseline, StandardAutoencoder
from .train import fit_pca_baseline, train


DEFAULT_HOMOGENEITY_SCAN_SCALES = tuple(
    float(value) for value in np.geomspace(0.1, 100.0, num=13)
)
DEFAULT_EXTRAPOLATION_SCALES = (1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    x_test: torch.Tensor,
    *,
    alpha_true: float,
    p_encoder: float,
    hill_k_fraction: float = 0.1,
    angular_quantile: float = 0.95,
    homogeneity_scales: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    homogeneity_scan_scales: Tuple[float, ...] = DEFAULT_HOMOGENEITY_SCAN_SCALES,
    homogeneity_probe_size: int = 512,
    binned_error_n_bins: int = 12,
    tail_conditional_quantile: float = 0.95,
    extrapolation_scales: Tuple[float, ...] = DEFAULT_EXTRAPOLATION_SCALES,
    extrapolation_sample_seed: int = 999,
    embedding: Optional[nn.Module] = None,
    embedding_dims: Optional[Tuple[int, int]] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run all metrics on a held-out tensor.

    ``embedding`` + ``embedding_dims`` are optional: if both are set we
    also compute the out-of-training-range extrapolation MSE by
    re-synthesising samples at large radii through the same embedding
    that produced the training data. For ``exp01`` (curved surface)
    there is no single ``FlexibleToyEmbedding`` module available, so
    extrapolation is skipped.
    """
    model.eval()
    x_device = x_test.to(device)
    forward_pass = model(x_device)
    x_hat_tensor = forward_pass["x_hat"]
    latent_z_tensor = forward_pass["z"]

    x_hat = x_hat_tensor.cpu().numpy()
    x_numpy = x_test.cpu().numpy()
    latent_z = latent_z_tensor.cpu().numpy()

    delta_norm: Optional[np.ndarray] = None
    if isinstance(model, HomogeneousAutoencoder):
        delta_tensor = forward_pass.get("delta")
        if delta_tensor is not None:
            delta_norm = np.linalg.norm(delta_tensor.cpu().numpy(), axis=1)

    ambient_radii = np.linalg.norm(x_numpy, axis=1)
    reconstructed_radii = np.linalg.norm(x_hat, axis=1)
    latent_radii = np.linalg.norm(latent_z, axis=1)

    reconstruction_mse = float(np.mean((x_numpy - x_hat) ** 2))

    hill_ambient = hill_estimate(ambient_radii, k_fraction=hill_k_fraction)
    hill_latent = hill_estimate(latent_radii, k_fraction=hill_k_fraction)
    hill_reconstructed = hill_estimate(reconstructed_radii, k_fraction=hill_k_fraction)

    # Expected latent tail index if encoder is exactly p-homogeneous:
    #   alpha / p    (Proposition 1 of the paper)
    expected_latent_alpha = alpha_true / p_encoder

    # Deprecated: compares to the unobservable true alpha rather than the
    # (biased) ambient Hill. Kept for backward-compat with older sweep JSONs.
    alpha_error_reconstructed = abs(hill_reconstructed["alpha"] - alpha_true)
    alpha_error_latent = abs(hill_latent["alpha"] - expected_latent_alpha)

    # Preferred: Proposition 1 consistency check against the ambient Hill.
    hill_drift_latent = hill_drift(
        alpha_latent=hill_latent["alpha"],
        alpha_ambient=hill_ambient["alpha"],
        p=p_encoder,
    )
    hill_drift_reconstructed = hill_drift(
        alpha_latent=hill_reconstructed["alpha"],
        alpha_ambient=hill_ambient["alpha"],
        p=1.0,
    )

    quantile_errors = extreme_quantile_errors(
        reference=ambient_radii, candidate=reconstructed_radii
    )

    angular_distance = angular_tail_distance(
        x=x_numpy,
        x_other=x_hat,
        radial_quantile=angular_quantile,
    )

    binned_error = binned_reconstruction_error(
        x=x_numpy, x_hat=x_hat, n_bins=binned_error_n_bins
    )
    tail_cond_mse = tail_conditional_mse(
        x=x_numpy, x_hat=x_hat, radial_quantile=tail_conditional_quantile
    )

    metrics: Dict[str, Any] = {
        "reconstruction_mse": reconstruction_mse,
        "alpha_true": float(alpha_true),
        "alpha_expected_latent": float(expected_latent_alpha),
        "hill_ambient": hill_ambient,
        "hill_latent": hill_latent,
        "hill_reconstructed": hill_reconstructed,
        "alpha_error_reconstructed": float(alpha_error_reconstructed),
        "alpha_error_latent": float(alpha_error_latent),
        "hill_drift_latent": float(hill_drift_latent),
        "hill_drift_reconstructed": float(hill_drift_reconstructed),
        "quantile_errors": quantile_errors,
        "angular_tail_distance": float(angular_distance),
        "tail_conditional_mse": float(tail_cond_mse),
    }

    if not isinstance(model, PCABaseline):
        probe = x_device[: min(homogeneity_probe_size, len(x_device))]
        metrics["homogeneity_error"] = encoder_homogeneity_error(
            model=model,
            x_probe=probe,
            p=p_encoder,
            scales=homogeneity_scales,
        )
        scan = homogeneity_scan(
            model=model, x_probe=probe, p=p_encoder, scales=homogeneity_scan_scales
        )
        metrics["_homogeneity_scan"] = scan

    if embedding is not None and embedding_dims is not None:
        D, m = embedding_dims
        extrap = extrapolation_mse(
            model=model,
            embedding=embedding,
            D=D,
            m=m,
            scale_multipliers=extrapolation_scales,
            sample_seed=extrapolation_sample_seed,
            device=device,
        )
        metrics["_extrapolation"] = extrap
        scales_arr = extrap["scales"]
        mse_arr = extrap["mse"]
        # record MSE at a "large" scale as a scalar summary for sweep plots
        large_scale_index = int(np.argmin(np.abs(scales_arr - 10.0)))
        metrics["extrapolation_mse_at_10"] = float(mse_arr[large_scale_index])

    # keep raw arrays for downstream plotting, not serialized by default
    metrics["_ambient_radii"] = ambient_radii
    metrics["_reconstructed_radii"] = reconstructed_radii
    metrics["_latent_radii"] = latent_radii
    metrics["_x_test"] = x_numpy
    metrics["_x_hat"] = x_hat
    metrics["_binned_error"] = binned_error
    if delta_norm is not None:
        metrics["_delta_norm"] = delta_norm
    return metrics


def train_and_evaluate(
    model_name: str,
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    test_data: torch.Tensor,
    config: TrainConfig,
    *,
    alpha_true: float,
    p_for_hill: float,
    embedding: Optional[nn.Module] = None,
    embedding_dims: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if verbose:
        print(f"[{model_name}] training...")
    if isinstance(model, PCABaseline):
        history = fit_pca_baseline(model, train_data, val_data, config)
    elif isinstance(model, (HomogeneousAutoencoder, StandardAutoencoder)):
        history = train(model, train_data, val_data, config, verbose=verbose)
    else:
        raise TypeError(f"Unknown model type for train_and_evaluate: {type(model).__name__}")

    if verbose:
        print(f"[{model_name}] evaluating...")
    metrics = evaluate_model(
        model=model,
        x_test=test_data,
        alpha_true=alpha_true,
        p_encoder=p_for_hill,
        embedding=embedding,
        embedding_dims=embedding_dims,
        device=config.device,
    )
    metrics["_history"] = history
    return metrics


def serializable(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip arrays prefixed with underscore so metrics are JSON-safe."""
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def write_metrics_json(metrics_by_model: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    payload = {name: serializable(metrics) for name, metrics in metrics_by_model.items()}
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
