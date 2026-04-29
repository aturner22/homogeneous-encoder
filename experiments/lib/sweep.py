"""Grid-sweep helper for the flexible toy manifold experiments.

Each sweep varies exactly one dataset/model parameter, fixes all others,
and at each point trains the three-model zoo (HomogeneousAE, StandardAE,
PCA) over multiple seeds so the viz layer can draw shaded bands. The
result is a tidy nested dict keyed as
``result["metrics"][model_name][metric_key]["mean" | "std"]``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import artifact_path
from .config import FlexibleToyConfig
from .data import FlexibleToyEmbedding, generate_flexible_toy
from .evaluation import train_zoo
from .models import (
    HomogeneousAutoencoder,
    PCABaseline,
    StandardAutoencoder,
    compute_matched_hidden_dim,
    count_parameters,
)
from .viz import save_diagnostic_panel_set, save_sweep_metric

SCALAR_METRIC_KEYS: tuple[str, ...] = (
    "reconstruction_mse",
    "tail_conditional_mse",
    "hill_drift_latent",
    "hill_drift_reconstructed",
    "angular_tail_distance",
    "extrapolation_mse_at_10",
)


MODEL_NAMES: tuple[str, ...] = ("HomogeneousAE", "StandardAE", "PCA")


def _make_datasets(config: FlexibleToyConfig, seed_offset: int = 0):
    """Generate train/val/test splits; tail-holdout when ``config.tail_holdout_quantile`` is set, else random."""
    kwargs = dict(
        D=config.D,
        m=config.m,
        alpha=config.alpha,
        kappa=config.kappa,
        curvature_rank=config.curvature_rank,
        embedding_seed=config.embedding_seed,
    )

    if config.tail_holdout_quantile is not None:
        # Draw one pool, then split by radius so max(train_radius) < min(test_radius).
        total = config.n_train + config.n_val + config.n_test
        pool = generate_flexible_toy(
            total, sample_seed=config.seed + seed_offset + 1, **kwargs
        )
        radii = torch.linalg.norm(pool, dim=1)
        threshold = torch.quantile(radii, float(config.tail_holdout_quantile))
        tail_mask = radii >= threshold
        tail_indices = torch.nonzero(tail_mask, as_tuple=False).squeeze(-1)
        bulk_indices = torch.nonzero(~tail_mask, as_tuple=False).squeeze(-1)

        generator = torch.Generator().manual_seed(
            config.seed + seed_offset + 101
        )
        bulk_perm = bulk_indices[
            torch.randperm(len(bulk_indices), generator=generator)
        ]
        n_train_actual = min(config.n_train, len(bulk_perm) - config.n_val)
        if n_train_actual <= 0:
            raise ValueError(
                f"tail-holdout bulk too small: {len(bulk_perm)} bulk samples "
                f"but config wants {config.n_train} train + {config.n_val} val"
            )
        train_indices = bulk_perm[:n_train_actual]
        val_indices = bulk_perm[n_train_actual : n_train_actual + config.n_val]
        return pool[train_indices], pool[val_indices], pool[tail_indices]

    train_data = generate_flexible_toy(
        config.n_train, sample_seed=config.seed + seed_offset + 1, **kwargs
    )
    val_data = generate_flexible_toy(
        config.n_val, sample_seed=config.seed + seed_offset + 2, **kwargs
    )
    test_data = generate_flexible_toy(
        config.n_test, sample_seed=config.seed + seed_offset + 3, **kwargs
    )
    return train_data, val_data, test_data


def build_embedding(config: FlexibleToyConfig) -> FlexibleToyEmbedding:
    """Return the same embedding module that ``generate_flexible_toy`` uses."""
    return FlexibleToyEmbedding(
        D=config.D,
        m=config.m,
        kappa=config.kappa,
        curvature_rank=config.curvature_rank,
        embedding_seed=config.embedding_seed,
    )


def build_model_zoo(config: FlexibleToyConfig) -> dict[str, Any]:
    """Build the three-model zoo with parameter-matched hidden dims.

    HAE uses ``config.hidden_dim``. StdAE gets a larger hidden_dim
    computed so that its total parameter count matches HAE's.
    """
    hae = HomogeneousAutoencoder(
        D=config.D,
        m=config.m,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        p_homogeneity=config.p_homogeneity,
        learnable_centre=config.learnable_centre,
    )
    hae_params = count_parameters(hae)
    stdae_hidden = compute_matched_hidden_dim(
        hae_params, config.D, config.m, config.hidden_layers,
    )
    stdae = StandardAutoencoder(
        D=config.D,
        m=config.m,
        hidden_dim=stdae_hidden,
        hidden_layers=config.hidden_layers,
    )
    print(
        f"  HAE  hidden_dim={config.hidden_dim:4d}  params={hae_params:>10,}\n"
        f"  AE   hidden_dim={stdae_hidden:4d}  params={count_parameters(stdae):>10,}"
    )
    return {
        "HomogeneousAE": hae,
        "StandardAE": stdae,
        "PCA": PCABaseline(D=config.D, m=config.m),
    }


def _scalar(metrics: Mapping[str, Any], key: str) -> float:
    if key not in metrics:
        raise KeyError(
            f"Metric {key!r} missing from metrics dict. "
            f"Available keys: {sorted(metrics)!r}"
        )
    value = metrics[key]
    return float("nan") if value is None else float(value)


def train_zoo_multiseed(
    config: FlexibleToyConfig,
    *,
    seed_artifact_dir: Path | None = None,
    force_retrain: bool = False,
    require_cache: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train and evaluate the three-model zoo ``config.num_seeds`` times.

    Caching
    -------
    When ``seed_artifact_dir`` is given, each ``(seed_index, model_name)``
    pair is persisted as ``<seed_artifact_dir>/seed<i>/<model>.pkl``.
    On subsequent calls that pickle is loaded in place of retraining.

    Pass ``force_retrain=True`` to ignore the cache and overwrite the
    pickles. Pass ``require_cache=True`` (plot-only mode) to error
    loudly if any cache entry is missing, rather than silently training.

    Returns:
        {
            "per_seed": [
                {model_name: full_metrics_dict}, ...     # length num_seeds
            ],
            "aggregate": {
                model_name: {
                    metric_key: {"mean": float, "std": float, "values": list}
                }
            }
        }
    """
    per_seed: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, dict[str, Any]]] = {
        name: {key: {"values": []} for key in SCALAR_METRIC_KEYS} for name in MODEL_NAMES
    }

    embedding = build_embedding(config)

    for seed_index in range(int(config.num_seeds)):
        seed_offset = seed_index * 97
        torch.manual_seed(config.seed + seed_offset)
        np.random.seed(config.seed + seed_offset)

        train_data, val_data, test_data = _make_datasets(config, seed_offset=seed_offset)

        seed_dir: Path | None = (
            seed_artifact_dir / f"seed{seed_index}" if seed_artifact_dir else None
        )
        needs_build = force_retrain or seed_dir is None or not all(
            artifact_path(seed_dir, name).exists() for name in MODEL_NAMES
        )
        models = build_model_zoo(config) if needs_build else {name: None for name in MODEL_NAMES}

        point = train_zoo(
            models,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            config=config,
            alpha_true=config.alpha,
            p_for_hill=config.p_homogeneity,
            embedding=embedding,
            embedding_dims=(config.D, config.m),
            seed_dir=seed_dir,
            force_retrain=force_retrain,
            require_cache=require_cache,
            verbose=verbose,
        )
        for model_name in MODEL_NAMES:
            metrics = point[model_name]
            for metric_key in SCALAR_METRIC_KEYS:
                aggregate[model_name][metric_key]["values"].append(
                    _scalar(metrics, metric_key)
                )
        per_seed.append(point)

    for metric_table in aggregate.values():
        for cell in metric_table.values():
            values = np.asarray(cell["values"], dtype=np.float64)
            cell["mean"] = float(np.nanmean(values))
            cell["std"] = float(np.nanstd(values))

    return {
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def run_flexible_toy_sweep(
    base_config: FlexibleToyConfig,
    parameter_name: str,
    parameter_values: Sequence[Any],
    *,
    artifact_root: Path | None = None,
    force_retrain: bool = False,
    require_cache: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Sweep a single parameter on the flexible toy with multi-seed aggregation.

    Returns a dict with:
        parameter_name   : str
        parameter_values : list
        metrics          : {model_name: {metric_key: {"mean": np.ndarray,
                                                       "std":  np.ndarray}}}
        raw              : list of per-point multi-seed results (serialised)
        per_seed_points  : list of per-point per-seed metric dicts
    """
    series: dict[str, dict[str, dict[str, list[float]]]] = {
        name: {
            key: {"mean": [], "std": []} for key in SCALAR_METRIC_KEYS
        }
        for name in MODEL_NAMES
    }
    raw_per_point: list[dict[str, Any]] = []
    per_seed_points: list[dict[str, Any]] = []

    for value in parameter_values:
        if verbose:
            print(f"\n--- {parameter_name} = {value} ---")

        config = dataclass_replace(base_config, **{parameter_name: value})
        seed_artifact_dir: Path | None = None
        if artifact_root is not None:
            safe_value = str(value).replace("/", "_")
            seed_artifact_dir = (
                Path(artifact_root) / f"point_{parameter_name}={safe_value}"
            )
        multiseed = train_zoo_multiseed(
            config,
            seed_artifact_dir=seed_artifact_dir,
            force_retrain=force_retrain,
            require_cache=require_cache,
            verbose=False,
        )
        aggregate = multiseed["aggregate"]

        for name in MODEL_NAMES:
            for key in SCALAR_METRIC_KEYS:
                cell = aggregate[name][key]
                series[name][key]["mean"].append(float(cell["mean"]))
                series[name][key]["std"].append(float(cell["std"]))
            if verbose:
                agg = aggregate[name]
                print(
                    f"    {name:<15} "
                    f"mse={agg['reconstruction_mse']['mean']:.5f} "
                    f"tail_mse={agg['tail_conditional_mse']['mean']:.5f} "
                    f"hill_drift={agg['hill_drift_latent']['mean']:.3f} "
                    f"extrap10={agg['extrapolation_mse_at_10']['mean']:.3f}"
                )

        raw_per_point.append(
            {
                name: {
                    key: {
                        "mean": float(aggregate[name][key]["mean"]),
                        "std": float(aggregate[name][key]["std"]),
                        "values": list(map(float, aggregate[name][key]["values"])),
                    }
                    for key in SCALAR_METRIC_KEYS
                }
                for name in MODEL_NAMES
            }
        )
        per_seed_points.append(
            {
                name: [
                    multiseed["per_seed"][seed_index][name]
                    for seed_index in range(len(multiseed["per_seed"]))
                ]
                for name in MODEL_NAMES
            }
        )

    # convert the per-parameter lists to numpy arrays for the plotting layer
    metric_series: dict[str, dict[str, dict[str, np.ndarray]]] = {
        name: {
            key: {
                "mean": np.asarray(series[name][key]["mean"], dtype=np.float64),
                "std": np.asarray(series[name][key]["std"], dtype=np.float64),
            }
            for key in SCALAR_METRIC_KEYS
        }
        for name in MODEL_NAMES
    }

    return {
        "parameter_name": parameter_name,
        "parameter_values": [
            v if isinstance(v, (int, float)) else str(v) for v in parameter_values
        ],
        "metrics": metric_series,
        "raw": raw_per_point,
        "per_seed_points": per_seed_points,
    }


def write_sweep_json(sweep_result: dict[str, Any], path: Path) -> None:
    payload = {
        "parameter_name": sweep_result["parameter_name"],
        "parameter_values": sweep_result["parameter_values"],
        "raw": sweep_result["raw"],
    }
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)


# -----------------------------------------------------------------------------
# exp03 / exp04 / exp05-style canonical one-parameter sweeps.
# -----------------------------------------------------------------------------

_CANONICAL_SWEEP_METRICS: tuple[tuple[str, str, str | None], ...] = (
    ("hill_drift_latent", "Hill Drift", None),
    ("extrapolation_mse_at_10", "Extrapolation MSE at Scale 10", "log"),
    ("tail_conditional_mse", "Tail-conditional MSE (Top 5%)", "log"),
)


def run_and_plot_param_sweep(
    base_config: FlexibleToyConfig,
    *,
    parameter_name: str,
    parameter_values: Sequence[Any],
    xlabel: str,
    fig_prefix: str,
    force_retrain: bool = False,
    require_cache: bool = False,
) -> dict[str, Any]:
    """Canonical one-parameter sweep used by exp03/04/05.

    Runs the multi-seed sweep, writes ``sweep.json``, saves the three
    headline figures (Hill drift / extrapolation / tail MSE) plus the
    per-grid-point diagnostic panel set, and returns the raw sweep
    result for any additional processing the caller needs.
    """
    output = Path(base_config.output_dir)
    result = run_flexible_toy_sweep(
        base_config=base_config,
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        artifact_root=output,
        force_retrain=force_retrain,
        require_cache=require_cache,
    )
    write_sweep_json(result, output / "sweep.json")

    seed0_series = single_seed_series(result["raw"])

    for metric_key, ylabel, yscale in _CANONICAL_SWEEP_METRICS:
        save_sweep_metric(
            parameter_values=parameter_values,
            series_by_model=seed0_series,
            output_path=output / f"fig_{fig_prefix}_{_METRIC_SLUG[metric_key]}.png",
            metric_key=metric_key,
            xlabel=xlabel,
            ylabel=ylabel,
            yscale=yscale,
            show_bands=False,
        )

    for value, point in zip(parameter_values, result["per_seed_points"], strict=True):
        subdir = output / f"point_{parameter_name}={value}"
        subdir.mkdir(exist_ok=True)
        save_diagnostic_panel_set(
            {name: point[name][0] for name in MODEL_NAMES},
            subdir,
            p=base_config.p_homogeneity,
        )

    return result


def single_seed_series(
    raw_per_point: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Rebuild a ``series_by_model`` where ``mean`` = seed 0's value per point."""
    return {
        name: {
            key: {
                "mean": np.asarray(
                    [pt[name][key]["values"][0] for pt in raw_per_point],
                    dtype=np.float64,
                ),
                "std": np.zeros(len(raw_per_point), dtype=np.float64),
            }
            for key in SCALAR_METRIC_KEYS
        }
        for name in MODEL_NAMES
    }


_METRIC_SLUG: dict[str, str] = {
    "hill_drift_latent": "hill_drift",
    "extrapolation_mse_at_10": "extrapolation",
    "tail_conditional_mse": "tail_mse",
}
