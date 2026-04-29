"""Shared evaluation routines for experiment scripts.

``_evaluate_model`` runs a trained model on a held-out tensor and returns
a flat dict of metrics covering reconstruction quality, Hill estimates
(ambient / latent / reconstructed), extreme quantile errors, angular
tail distance and - for neural models - the numerical homogeneity check.
It is a private helper; drivers call the full ``train_and_evaluate``
pipeline instead.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .artifacts import artifact_path, load_run_artifact, save_run_artifact
from .config import TrainConfig
from .metrics import (
    angular_tail_distance,
    binned_reconstruction_error,
    encoder_homogeneity_error,
    extrapolation_mse,
    extreme_quantile_errors,
    hill_drift,
    hill_estimate,
    tail_conditional_mse,
)
from .models import HomogeneousAutoencoder, PCABaseline, StandardAutoencoder
from .train import fit_pca_baseline, train

DEFAULT_EXTRAPOLATION_SCALES = (1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0)


HOMOGENEITY_PROBE_SIZE = 512
BINNED_ERROR_N_BINS = 12
EXTRAPOLATION_SAMPLE_SEED = 999


@torch.no_grad()
def _evaluate_model(
    model: nn.Module,
    x_test: torch.Tensor,
    *,
    alpha_true: float,
    p_encoder: float,
    hill_k_fraction: float = 0.1,
    angular_quantile: float = 0.95,
    homogeneity_scales: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    tail_conditional_quantile: float = 0.95,
    extrapolation_scales: tuple[float, ...] = DEFAULT_EXTRAPOLATION_SCALES,
    embedding: nn.Module | None = None,
    embedding_dims: tuple[int, int] | None = None,
    train_radii: np.ndarray | None = None,
    eval_centre: torch.Tensor | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run all metrics on a held-out tensor.

    ``embedding`` + ``embedding_dims`` are optional: if both are set we
    also compute the out-of-training-range extrapolation MSE by
    re-synthesising samples at large radii through the same embedding
    that produced the training data. For ``exp01`` (curved surface)
    there is no single ``FlexibleToyEmbedding`` module available, so
    extrapolation is skipped.

    ``train_radii`` (optional) is the array of ambient radii on the
    training set. When given, it is used to anchor the extrapolation
    scan to the training envelope: ``base_radius`` for
    ``extrapolation_mse`` becomes ``quantile(train_radii, 0.99)`` so
    that ``lambda=1`` sits at the edge of training support and
    ``lambda=10, 100`` are genuinely beyond it. Without this anchor,
    "extrapolation at lambda=10" is misleading when training samples
    routinely reach radii of 10+ (as happens for heavy-tailed Pareto
    draws with alpha < 2).
    """
    model.eval()
    x_device = x_test.to(device)
    forward_pass = model(x_device)
    x_hat_tensor = forward_pass["x_hat"]
    latent_z_tensor = forward_pass["z"]

    x_hat = x_hat_tensor.cpu().numpy()
    x_numpy = x_test.cpu().numpy()
    latent_z = latent_z_tensor.cpu().numpy()

    # Resolve the canonical evaluation centre. All ambient-space radii,
    # tail thresholds, and angular projections in this function use
    # ||x - eval_centre|| so HAE/StdAE/PCA are scored on the same
    # recentred view of the data. A None eval_centre means "no
    # recentring" (raw radii) for backward compatibility.
    if eval_centre is None:
        centre_np = np.zeros(x_numpy.shape[1], dtype=np.float64)
    else:
        centre_np = np.asarray(eval_centre.cpu(), dtype=np.float64).reshape(x_numpy.shape[1])
    centred_x = x_numpy - centre_np
    centred_x_hat = x_hat - centre_np

    delta_norm: np.ndarray | None = None
    if isinstance(model, HomogeneousAutoencoder):
        delta_tensor = forward_pass.get("delta")
        if delta_tensor is not None:
            delta_norm = np.linalg.norm(delta_tensor.cpu().numpy(), axis=1)

    ambient_radii = np.linalg.norm(centred_x, axis=1)
    reconstructed_radii = np.linalg.norm(centred_x_hat, axis=1)
    latent_radii = np.linalg.norm(latent_z, axis=1)

    train_radius_p99: float | None = None
    if train_radii is not None:
        train_radii_np = np.asarray(train_radii, dtype=np.float64)
        if train_radii_np.size:
            train_radius_p99 = float(np.quantile(train_radii_np, 0.99))

    reconstruction_mse = float(np.mean((x_numpy - x_hat) ** 2))

    hill_ambient = hill_estimate(ambient_radii, k_fraction=hill_k_fraction)
    hill_latent = hill_estimate(latent_radii, k_fraction=hill_k_fraction)
    hill_reconstructed = hill_estimate(reconstructed_radii, k_fraction=hill_k_fraction)

    # Expected latent tail index if encoder is exactly p-homogeneous:
    #   alpha / p    (Proposition 1 of the paper)
    expected_latent_alpha = alpha_true / p_encoder

    # Proposition 1 consistency check against the ambient Hill.
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
        centre=centre_np,
    )

    binned_error = binned_reconstruction_error(
        x=x_numpy, x_hat=x_hat, n_bins=BINNED_ERROR_N_BINS, centre=centre_np
    )
    tail_cond_mse = tail_conditional_mse(
        x=x_numpy, x_hat=x_hat, radial_quantile=tail_conditional_quantile,
        centre=centre_np,
    )

    metrics: dict[str, Any] = {
        "reconstruction_mse": reconstruction_mse,
        "alpha_true": float(alpha_true),
        "alpha_expected_latent": float(expected_latent_alpha),
        "hill_ambient": hill_ambient,
        "hill_latent": hill_latent,
        "hill_reconstructed": hill_reconstructed,
        "hill_drift_latent": float(hill_drift_latent),
        "hill_drift_reconstructed": float(hill_drift_reconstructed),
        "quantile_errors": quantile_errors,
        "angular_tail_distance": float(angular_distance),
        "tail_conditional_mse": float(tail_cond_mse),
    }

    if not isinstance(model, PCABaseline):
        probe = x_device[: min(HOMOGENEITY_PROBE_SIZE, len(x_device))]
        metrics["homogeneity_error"] = encoder_homogeneity_error(
            model=model,
            x_probe=probe,
            p=p_encoder,
            scales=homogeneity_scales,
        )

    if embedding is not None and embedding_dims is not None:
        D, m = embedding_dims
        # Anchor the extrapolation probe to the training envelope when
        # train_radii were supplied; otherwise fall back to unit radius.
        extrap_base_radius = (
            train_radius_p99 if train_radius_p99 is not None else 1.0
        )
        extrap = extrapolation_mse(
            model=model,
            embedding=embedding,
            D=D,
            m=m,
            scale_multipliers=extrapolation_scales,
            sample_seed=EXTRAPOLATION_SAMPLE_SEED,
            base_radius=extrap_base_radius,
            device=device,
        )
        metrics["_extrapolation"] = extrap
        scales_arr = extrap["scales"]
        mse_arr = extrap["mse"]
        # record MSE at a "large" scale as a scalar summary for sweep plots
        large_scale_index = int(np.argmin(np.abs(scales_arr - 10.0)))
        metrics["extrapolation_mse_at_10"] = float(mse_arr[large_scale_index])

    # keep raw arrays for downstream plotting, not serialized by default
    metrics["_ambient_radii"] = ambient_radii  # ||x - eval_centre||
    metrics["_reconstructed_radii"] = reconstructed_radii  # ||x_hat - eval_centre||
    metrics["_latent_radii"] = latent_radii
    metrics["_latent_codes"] = latent_z
    metrics["_x_test"] = x_numpy
    metrics["_x_hat"] = x_hat
    metrics["_eval_centre"] = centre_np
    metrics["_binned_error"] = binned_error
    if train_radius_p99 is not None:
        metrics["train_radius_p99"] = float(train_radius_p99)
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
    embedding: nn.Module | None = None,
    embedding_dims: tuple[int, int] | None = None,
    artifact_path: Path | None = None,
    eval_centre: torch.Tensor | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train ``model`` then evaluate on ``test_data``.

    ``eval_centre`` is the canonical ambient centre used to define
    "recentred radius" for every radius-based metric (Hill, binned
    reconstruction, tail-conditional MSE, angular tail SW, scale-scan).
    Pass HAE's learnt ``model.centre`` for HAE itself; for the other
    models in the same experiment, pass the *same* centre so all three
    are scored on the same view of the data. ``None`` (default) means:
    use the model's own centre if it has one (HAE), otherwise zero.
    """
    if verbose:
        print(f"[{model_name}] training...")
    if isinstance(model, PCABaseline):
        history = fit_pca_baseline(model, train_data, val_data, config)
    elif isinstance(model, (HomogeneousAutoencoder, StandardAutoencoder)):
        history = train(model, train_data, val_data, config, verbose=verbose)
    else:
        raise TypeError(f"Unknown model type for train_and_evaluate: {type(model).__name__}")

    if eval_centre is None and isinstance(model, HomogeneousAutoencoder):
        eval_centre = model.centre.detach().clone()

    if verbose:
        print(f"[{model_name}] evaluating...")
    centre_np_for_train = (
        np.zeros(train_data.shape[1], dtype=np.float64)
        if eval_centre is None
        else np.asarray(eval_centre.cpu(), dtype=np.float64).reshape(train_data.shape[1])
    )
    train_radii = np.linalg.norm(
        train_data.cpu().numpy() - centre_np_for_train, axis=1
    )
    metrics = _evaluate_model(
        model=model,
        x_test=test_data,
        alpha_true=alpha_true,
        p_encoder=p_for_hill,
        embedding=embedding,
        embedding_dims=embedding_dims,
        train_radii=train_radii,
        eval_centre=eval_centre,
        device=config.device,
    )
    metrics["_history"] = history
    if artifact_path is not None:
        save_run_artifact(
            Path(artifact_path), metrics=metrics, model=model, config=config
        )
    return metrics


def train_zoo(
    models_by_name: Mapping[str, nn.Module | None],
    *,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    test_data: torch.Tensor,
    config: TrainConfig,
    alpha_true: float,
    p_for_hill: float,
    seed_dir: Path | None = None,
    force_retrain: bool = False,
    require_cache: bool = False,
    embedding: nn.Module | None = None,
    embedding_dims: tuple[int, int] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Train and evaluate a dict of models with a shared canonical centre.

    Iterates ``models_by_name`` in insertion order. The first
    ``HomogeneousAutoencoder`` to train (or whose cached metrics already
    record an ``_eval_centre``) fixes the canonical evaluation centre,
    which is then threaded as ``eval_centre`` into every subsequent
    model's ``train_and_evaluate`` call. This guarantees Hill drift,
    binned reconstruction error, tail-conditional MSE, angular tail
    distance, and the reconstruction scale-scan all use the same
    ``||x - c||`` definition of "ambient radius" across the zoo.

    Caching: if ``seed_dir`` is given, each model's pickle at
    ``<seed_dir>/<name>.pkl`` is loaded when present and ``force_retrain``
    is False. ``require_cache=True`` makes a missing pickle a hard error
    (plot-only mode). When ``models_by_name[name] is None``, the cache
    must hit; this is the multi-seed sweep convention where the model
    instance is built only when training is actually needed.

    Returns a dict ``{name: metrics}`` in the same order as
    ``models_by_name``. The returned ``metrics["_eval_centre"]`` will be
    identical for all models (the canonical centre).
    """
    metrics_by_model: dict[str, dict[str, Any]] = {}
    canonical_centre: torch.Tensor | None = None

    for name, model in models_by_name.items():
        pkl = artifact_path(seed_dir, name) if seed_dir is not None else None

        if pkl is not None and pkl.exists() and not force_retrain:
            metrics = load_run_artifact(pkl)["metrics"]
            if verbose:
                print(f"[{name}] loaded cached artifact from {pkl}")
        else:
            if pkl is not None and require_cache:
                raise FileNotFoundError(
                    f"--plot-only requested but cached artifact missing at {pkl}. "
                    f"Remove --plot-only to train, or regenerate with --force-retrain."
                )
            if model is None:
                raise ValueError(
                    f"Model instance for {name!r} is None but no cached "
                    f"artifact at {pkl!s}; build the model before calling "
                    f"train_zoo when training is required."
                )
            metrics = train_and_evaluate(
                model_name=name,
                model=model,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                config=config,
                alpha_true=alpha_true,
                p_for_hill=p_for_hill,
                embedding=embedding,
                embedding_dims=embedding_dims,
                artifact_path=pkl,
                eval_centre=canonical_centre,
                verbose=verbose,
            )

        metrics_by_model[name] = metrics

        # Capture the canonical centre once, from the first HAE we see.
        if canonical_centre is None:
            if isinstance(model, HomogeneousAutoencoder):
                canonical_centre = model.centre.detach().clone()
            elif metrics.get("_eval_centre") is not None:
                centre_arr = np.asarray(metrics["_eval_centre"], dtype=np.float32)
                canonical_centre = torch.tensor(centre_arr)

    return metrics_by_model


def serializable(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Strip arrays prefixed with underscore so metrics are JSON-safe."""
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def write_metrics_json(metrics_by_model: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    payload = {name: serializable(metrics) for name, metrics in metrics_by_model.items()}
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
