"""exp01: curved surface in R^3 on the paper's updated architecture.

Continuity anchor with the original minimal example. Trains the
HomogeneousAE and a StandardAE baseline on the same curved surface
(D=3, m=2) and emits separate single-panel PNGs: 3-D scatters of
truth + reconstructions, the encoder homogeneity residual scan, and
the full exp02 diagnostic panel set (latent-vs-ambient radius, Hill
curves, binned recon error, HAE correction diagnostics).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import CurvedSurfaceConfig, ensure_output_dir  # noqa: E402
from lib.data import generate_curved_surface  # noqa: E402
from lib.evaluation import serializable, train_and_evaluate  # noqa: E402
from lib.models import (  # noqa: E402
    HomogeneousAutoencoder,
    StandardAutoencoder,
    compute_matched_hidden_dim,
    count_parameters,
)
from lib.viz import (  # noqa: E402
    MODEL_LABELS,
    plot_training_history,
    save_curved_surface_scatter,
    save_exp02_panel_set,
    save_homogeneity_scan,
)


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "results"
    config = CurvedSurfaceConfig(output_dir=str(output_dir))
    ensure_output_dir(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"Config: {config}")
    print(f"Device: {config.device}")

    train_data = generate_curved_surface(config.n_train, seed=config.seed + 1)
    val_data = generate_curved_surface(config.n_val, seed=config.seed + 2)
    test_data = generate_curved_surface(config.n_test, seed=config.seed + 3)

    hae = HomogeneousAutoencoder(
        D=3,
        m=2,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        p_homogeneity=config.p_homogeneity,
    )
    hae_params = count_parameters(hae)
    stdae_hidden = compute_matched_hidden_dim(
        hae_params, 3, 2, config.hidden_layers,
    )
    stdae = StandardAutoencoder(
        D=3,
        m=2,
        hidden_dim=stdae_hidden,
        hidden_layers=config.hidden_layers,
    )
    models = {"HomogeneousAE": hae, "StandardAE": stdae}
    for name, model in models.items():
        print(f"  {name}: {count_parameters(model):,} parameters")

    # The curved-surface construction uses a smooth radial profile whose
    # heavy tail comes from a Student-t with df = student_df_t. That value
    # is the ground-truth alpha for the Hill plot.
    alpha_true = float(config.student_df_t)

    metrics_by_model = {}
    for name, model in models.items():
        metrics_by_model[name] = train_and_evaluate(
            model_name=name,
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            config=config,
            alpha_true=alpha_true,
            p_for_hill=config.p_homogeneity,
        )

    output = Path(config.output_dir)
    x_test = metrics_by_model["HomogeneousAE"]["_x_test"]
    truth_radii = np.linalg.norm(x_test, axis=1)

    save_curved_surface_scatter(
        x_test, output / "fig_curved_truth.png",
        title="truth",
    )
    for name in metrics_by_model:
        save_curved_surface_scatter(
            metrics_by_model[name]["_x_hat"],
            output / f"fig_curved_{name}.png",
            color_by=truth_radii,
            title=_label(name),
        )

    scan_by_model = {
        name: metrics_by_model[name]["_homogeneity_scan"]
        for name in metrics_by_model
        if "_homogeneity_scan" in metrics_by_model[name]
    }
    save_homogeneity_scan(scan_by_model, output / "fig_homogeneity_scan.png")

    save_exp02_panel_set(metrics_by_model, output, p=config.p_homogeneity)
    print(f"Wrote single-panel PNGs to {output}")

    # Diagnostic training curves (not a paper figure)
    diagnostic_dir = output / "diagnostic"
    diagnostic_dir.mkdir(exist_ok=True)
    plot_training_history(
        {name: metrics_by_model[name]["_history"] for name in metrics_by_model},
        diagnostic_dir / "training_history.png",
    )

    # Persist scalar summary
    summary = {name: serializable(metrics_by_model[name]) for name in metrics_by_model}
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    alpha_ambient = summary["HomogeneousAE"]["hill_ambient"]["alpha"]
    print("\n=== exp01 summary ===")
    print(f"  alpha_true (Student-t df)  = {alpha_true:.3f}")
    print(f"  hill_ambient (from data)   = {alpha_ambient:.3f}")
    print(f"  Prop 1 target (amb/p)      = {alpha_ambient / config.p_homogeneity:.3f}")
    for name in metrics_by_model:
        row = summary[name]
        print(
            f"  {name}:\n"
            f"    reconstruction_mse  = {row['reconstruction_mse']:.5f}\n"
            f"    hill_latent         = {row['hill_latent']['alpha']:.3f}\n"
            f"    hill_drift_latent   = {row['hill_drift_latent']:.4f}\n"
            f"    angular_tail_dist   = {row['angular_tail_distance']:.5f}\n"
            f"    homogeneity_worst   = "
            f"{row.get('homogeneity_error', {}).get('worst', float('nan')):.2e}"
        )


if __name__ == "__main__":
    main()
