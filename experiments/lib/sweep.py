"""Grid-sweep helper for the flexible toy manifold experiments.

Each sweep varies exactly one dataset/model parameter, fixes all others,
and at each point trains the three-model zoo (HomogeneousAE, StandardAE,
PCA) over multiple seeds so the viz layer can draw shaded bands. The
result is a tidy nested dict keyed as
``result["metrics"][model_name][metric_key]["mean" | "std"]``.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .config import FlexibleToyConfig
from .data import FlexibleToyEmbedding, generate_flexible_toy
from .evaluation import serializable, train_and_evaluate
from .models import HomogeneousAutoencoder, PCABaseline, StandardAutoencoder


SCALAR_METRIC_KEYS: Tuple[str, ...] = (
    "reconstruction_mse",
    "tail_conditional_mse",
    "hill_drift_latent",
    "hill_drift_reconstructed",
    "alpha_error_reconstructed",
    "alpha_error_latent",
    "angular_tail_distance",
    "extrapolation_mse_at_10",
)


METRIC_NAMES = SCALAR_METRIC_KEYS  # backwards-compat alias


MODEL_NAMES: Tuple[str, ...] = ("HomogeneousAE", "StandardAE", "PCA")


def _make_datasets(config: FlexibleToyConfig, seed_offset: int = 0):
    kwargs = dict(
        D=config.D,
        m=config.m,
        alpha=config.alpha,
        kappa=config.kappa,
        curvature_rank=config.curvature_rank,
        embedding_seed=config.embedding_seed,
    )
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


def build_model_zoo(config: FlexibleToyConfig) -> Dict[str, Any]:
    return {
        "HomogeneousAE": HomogeneousAutoencoder(
            D=config.D,
            m=config.m,
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
            p_homogeneity=config.p_homogeneity,
        ),
        "StandardAE": StandardAutoencoder(
            D=config.D,
            m=config.m,
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
        ),
        "PCA": PCABaseline(D=config.D, m=config.m),
    }


def _scalar(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return float("nan")
    return float(value)


def train_zoo_multiseed(
    config: FlexibleToyConfig,
    *,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate the three-model zoo ``config.num_seeds`` times.

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
    per_seed: List[Dict[str, Any]] = []
    aggregate: Dict[str, Dict[str, Dict[str, Any]]] = {
        name: {key: {"values": []} for key in SCALAR_METRIC_KEYS} for name in MODEL_NAMES
    }

    embedding = build_embedding(config)

    for seed_index in range(int(config.num_seeds)):
        seed_offset = seed_index * 97
        torch.manual_seed(config.seed + seed_offset)
        np.random.seed(config.seed + seed_offset)

        train_data, val_data, test_data = _make_datasets(config, seed_offset=seed_offset)
        models = build_model_zoo(config)

        point: Dict[str, Any] = {}
        for model_name, model in models.items():
            metrics = train_and_evaluate(
                model_name=model_name,
                model=model,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                config=config,
                alpha_true=config.alpha,
                p_for_hill=config.p_homogeneity,
                embedding=embedding,
                embedding_dims=(config.D, config.m),
                verbose=verbose,
            )
            point[model_name] = metrics
            for metric_key in SCALAR_METRIC_KEYS:
                aggregate[model_name][metric_key]["values"].append(
                    _scalar(metrics, metric_key)
                )
        per_seed.append(point)

    for model_name, metric_table in aggregate.items():
        for metric_key, cell in metric_table.items():
            values = np.asarray(cell["values"], dtype=np.float64)
            cell["mean"] = float(np.nanmean(values))
            cell["std"] = float(np.nanstd(values))

    return {"per_seed": per_seed, "aggregate": aggregate}


def run_flexible_toy_sweep(
    base_config: FlexibleToyConfig,
    parameter_name: str,
    parameter_values: Sequence[Any],
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Sweep a single parameter on the flexible toy with multi-seed aggregation.

    Returns a dict with:
        parameter_name   : str
        parameter_values : list
        metrics          : {model_name: {metric_key: {"mean": np.ndarray,
                                                       "std":  np.ndarray}}}
        raw              : list of per-point multi-seed results (serialised)
        per_seed_points  : list of per-point per-seed metric dicts
    """
    series: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        name: {
            key: {"mean": [], "std": []} for key in SCALAR_METRIC_KEYS
        }
        for name in MODEL_NAMES
    }
    raw_per_point: List[Dict[str, Any]] = []
    per_seed_points: List[Dict[str, Any]] = []

    for value in parameter_values:
        if verbose:
            print(f"\n--- {parameter_name} = {value} ---")

        config = dataclass_replace(base_config, **{parameter_name: value})
        multiseed = train_zoo_multiseed(config, verbose=False)
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
    metric_series: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
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


def write_sweep_json(sweep_result: Dict[str, Any], path: Path) -> None:
    import json

    payload = {
        "parameter_name": sweep_result["parameter_name"],
        "parameter_values": sweep_result["parameter_values"],
        "raw": sweep_result["raw"],
    }
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
