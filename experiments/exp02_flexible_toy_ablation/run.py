"""exp02: main ablation on the flexible toy manifold.

Trains the three-model zoo (HomogeneousAE, StandardAE, PCA) on the
flexible toy and saves the standard exp02 panel set via
``save_diagnostic_panel_set``: latent-vs-ambient radius, Hill curves,
extrapolation, binned recon error, and HAE correction diagnostics.

Also writes ``summary.json``, ``summary_table.csv``, and
``per_seed_metrics.json`` for the appendix.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from lib.cli import init_experiment, parse_standard_args
from lib.config import FlexibleToyConfig
from lib.evaluation import serializable
from lib.sweep import MODEL_NAMES, SCALAR_METRIC_KEYS, train_zoo_multiseed
from lib.viz import (
    plot_training_history,
    save_diagnostic_panel_set,
)

SUMMARY_METRIC_ORDER = (
    ("reconstruction_mse", "Recon MSE (bulk)", "sci"),
    ("tail_conditional_mse", "Recon MSE (tail q0.95)", "sci"),
    ("hill_drift_latent", "|alpha_lat*p - alpha_amb|", "num"),
    ("hill_drift_reconstructed", "|alpha_rec - alpha_amb|", "num"),
    ("angular_tail_distance", "Angular tail SW", "num"),
    ("extrapolation_mse_at_10", "Extrapolation MSE (lambda=10)", "sci"),
)


def _write_summary_csv(
    summary_table: Mapping[str, Mapping[str, Mapping[str, float]]],
    metric_order,
    output_path: Path,
) -> None:
    lines = ["metric," + ",".join(summary_table.keys())]
    for metric_key, display_name, _unit in metric_order:
        row = [display_name]
        for model_name in summary_table:
            cell = summary_table[model_name].get(metric_key)
            if cell is None:
                row.append("")
                continue
            row.append(f"{cell['mean']:.6e}+/-{cell['std']:.6e}")
        lines.append(",".join(row))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extra_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tail-holdout-quantile", type=float, default=None,
        help="Hold out samples with radius above this quantile as the test "
             "set; train/val drawn from bulk. Strict extrapolation regime.",
    )
    parser.add_argument(
        "--output-subdir", type=str, default=None,
        help="Subdirectory under results/ for this run's outputs.",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Pareto tail index for the latent radius. Default 1.8 (heavy, "
             "matches Hill-drift headline). Use 3.5 for a bounded training "
             "envelope so extrapolation at lambda=10 is genuinely OOD.",
    )
    parser.add_argument(
        "--kappa", type=float, default=None,
        help="Curvature strength of the flexible embedding.",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=None,
        help="Number of seeds for multi-seed aggregation. Overrides "
             "FlexibleToyConfig.num_seeds.",
    )


def main() -> None:
    args = parse_standard_args(description=__doc__, extra=_extra_args)

    config_kwargs: dict[str, object] = {
        "tail_holdout_quantile": args.tail_holdout_quantile,
    }
    if args.alpha is not None:
        config_kwargs["alpha"] = float(args.alpha)
    if args.kappa is not None:
        config_kwargs["kappa"] = float(args.kappa)
    if args.num_seeds is not None:
        config_kwargs["num_seeds"] = int(args.num_seeds)

    config = init_experiment(
        Path(__file__),
        FlexibleToyConfig,
        subdir=args.output_subdir,
        **config_kwargs,
    )

    result = train_zoo_multiseed(
        config,
        seed_artifact_dir=Path(config.output_dir),
        force_retrain=args.force_retrain,
        require_cache=args.plot_only,
        verbose=True,
    )
    per_seed = result["per_seed"]
    aggregate = result["aggregate"]

    first_point = per_seed[0]
    alpha_ambient = float(first_point["HomogeneousAE"]["hill_ambient"]["alpha"])
    print(f"Ambient Hill estimate (from data): {alpha_ambient:.3f}")

    output = Path(config.output_dir)

    save_diagnostic_panel_set(first_point, output, p=config.p_homogeneity)
    print(f"Wrote single-panel PNGs to {output}")

    # Diagnostic training curves (not a paper figure)
    histories_by_model = {name: first_point[name]["_history"] for name in MODEL_NAMES}
    diagnostic_dir = output / "diagnostic"
    diagnostic_dir.mkdir(exist_ok=True)
    plot_training_history(histories_by_model, diagnostic_dir / "training_history.png")

    summary_table: dict[str, dict[str, dict[str, float]]] = {
        name: {
            key: {
                "mean": float(aggregate[name][key]["mean"]),
                "std": float(aggregate[name][key]["std"]),
            }
            for key in SCALAR_METRIC_KEYS
        }
        for name in MODEL_NAMES
    }

    _write_summary_csv(summary_table, SUMMARY_METRIC_ORDER, output / "summary_table.csv")
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_table, handle, indent=2, sort_keys=True)

    per_seed_serialisable = [
        {name: serializable(per_seed[seed_index][name]) for name in MODEL_NAMES}
        for seed_index in range(len(per_seed))
    ]
    with open(output / "per_seed_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(per_seed_serialisable, handle, indent=2, sort_keys=True)

    print("\n=== exp02 summary ===")
    header = f"{'metric':<30}" + "".join(f"{name:>18}" for name in MODEL_NAMES)
    print(header)
    for metric_key, display_name, _unit in SUMMARY_METRIC_ORDER:
        row = f"{display_name:<30}"
        for name in MODEL_NAMES:
            cell = summary_table[name][metric_key]
            row += f"  {cell['mean']:>9.3e}±{cell['std']:>6.1e}"
        print(row)


if __name__ == "__main__":
    main()
