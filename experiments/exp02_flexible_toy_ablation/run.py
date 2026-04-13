"""exp02: main ablation on the flexible toy manifold.

Trains the three-model zoo (HomogeneousAE, StandardAE, PCA) on the
flexible toy and saves the standard exp02 panel set via
``save_exp02_panel_set``: latent-vs-ambient radius, Hill curves,
extrapolation, binned recon error, and HAE correction diagnostics.

Also writes ``summary.json``, ``summary_table.csv``, and
``per_seed_metrics.json`` for the appendix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import FlexibleToyConfig, ensure_output_dir  # noqa: E402
from lib.evaluation import serializable  # noqa: E402
from lib.sweep import MODEL_NAMES, SCALAR_METRIC_KEYS, train_zoo_multiseed  # noqa: E402
from lib.viz import plot_training_history, save_exp02_panel_set  # noqa: E402


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
        for model_name in summary_table.keys():
            cell = summary_table[model_name].get(metric_key)
            if cell is None:
                row.append("")
                continue
            row.append(f"{cell['mean']:.6e}+/-{cell['std']:.6e}")
        lines.append(",".join(row))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "results"
    config = FlexibleToyConfig(output_dir=str(output_dir))
    ensure_output_dir(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"Config: {config}")
    print(f"Device: {config.device}")

    result = train_zoo_multiseed(config, verbose=True)
    per_seed = result["per_seed"]
    aggregate = result["aggregate"]

    first_point = per_seed[0]
    alpha_ambient = float(first_point["HomogeneousAE"]["hill_ambient"]["alpha"])
    print(f"Ambient Hill estimate (from data): {alpha_ambient:.3f}")

    output = Path(config.output_dir)

    save_exp02_panel_set(first_point, output, p=config.p_homogeneity)
    print(f"Wrote single-panel PNGs to {output}")

    # Diagnostic training curves (not a paper figure)
    histories_by_model = {name: first_point[name]["_history"] for name in MODEL_NAMES}
    diagnostic_dir = output / "diagnostic"
    diagnostic_dir.mkdir(exist_ok=True)
    plot_training_history(histories_by_model, diagnostic_dir / "training_history.png")

    summary_table: Dict[str, Dict[str, Dict[str, float]]] = {
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
