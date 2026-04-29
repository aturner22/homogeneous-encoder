"""exp08: curved surface with Pareto-margin pre-standardisation.

Mirrors exp01's data + training + evaluation pipeline. Before any
training, each ambient coordinate of the train / val / test data is
mapped through a per-marginal GPD-tail PIT (rank bulk + GPD tail) and
then the standard Pareto inverse CDF, so the joint copula is preserved
but every marginal becomes a standard Pareto with the configured tail
index. Two extra diagnostic figures verify the transform itself
(per-dim histograms and Hill curves before vs after); the rest of the
plot suite is identical to exp01.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lib.artifacts import artifact_path
from lib.cli import init_experiment, parse_standard_args
from lib.config import CurvedSurfaceParetoMarginsConfig
from lib.data import generate_curved_surface
from lib.evaluation import serializable, train_zoo
from lib.models import (
    HomogeneousAutoencoder,
    StandardAutoencoder,
    compute_matched_hidden_dim,
    count_parameters,
)
from lib.preprocessing import fit_apply_pareto_margins
from lib.viz import (
    MODEL_LABELS,
    plot_training_history,
    save_curved_surface_scatter,
    save_diagnostic_panel_set,
    save_hero_curved_surface,
    save_marginal_pareto_hill,
    save_marginal_pareto_histograms,
    save_overlay_reconstruction,
    save_overlay_reconstruction_panels,
)


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def main() -> None:
    args = parse_standard_args(description=__doc__)
    config = init_experiment(Path(__file__), CurvedSurfaceParetoMarginsConfig)

    train_raw = generate_curved_surface(config.n_train, seed=config.seed + 1)
    val_raw = generate_curved_surface(config.n_val, seed=config.seed + 2)
    test_raw = generate_curved_surface(config.n_test, seed=config.seed + 3)

    if not config.pre_standardize_to_pareto_margins:
        raise RuntimeError(
            "exp08 requires pre_standardize_to_pareto_margins=True; check the "
            "CurvedSurfaceParetoMarginsConfig defaults."
        )

    print(f"Pre-standardising marginals to Pareto(alpha="
          f"{config.pareto_target_alpha:g}) via GPD-tail PIT "
          f"(threshold q={config.pit_threshold_quantile})")
    train_data, (val_data, test_data), _ = fit_apply_pareto_margins(
        train_raw, [val_raw, test_raw],
        pareto_alpha=config.pareto_target_alpha,
        threshold_quantile=config.pit_threshold_quantile,
    )

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Diagnostic figures for the transform itself (use train data so the
    # before/after comparison is on the same samples).
    save_marginal_pareto_histograms(
        train_raw.numpy(),
        train_data.numpy(),
        output / "fig_pareto_marginal_histograms.png",
    )
    save_marginal_pareto_hill(
        train_raw.numpy(),
        train_data.numpy(),
        output / "fig_pareto_marginal_hill.png",
        target_alpha=config.pareto_target_alpha,
    )

    model_names = ("HomogeneousAE", "StandardAE")
    seed_dir = output / "seed0"
    cached = {
        name: artifact_path(seed_dir, name).exists() for name in model_names
    }
    all_cached = all(cached.values())
    if args.plot_only and not all_cached:
        missing = [n for n, ok in cached.items() if not ok]
        raise FileNotFoundError(
            f"--plot-only requested but missing cached artifacts for: {missing}. "
            f"Run without --plot-only first."
        )

    hae = HomogeneousAutoencoder(
        D=3,
        m=2,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        p_homogeneity=config.p_homogeneity,
        learnable_centre=config.learnable_centre,
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

    # Ambient tail index of the *transformed* data is the Pareto target.
    alpha_true = float(config.pareto_target_alpha)

    metrics_by_model = train_zoo(
        models,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
        alpha_true=alpha_true,
        p_for_hill=config.p_homogeneity,
        seed_dir=seed_dir,
        force_retrain=args.force_retrain,
    )

    x_test = metrics_by_model["HomogeneousAE"]["_x_test"]
    truth_radii = np.linalg.norm(x_test, axis=1)
    positive_truth = truth_radii[truth_radii > 0]
    color_vmin = float(positive_truth.min()) if positive_truth.size else 1.0
    color_vmax = float(np.quantile(truth_radii, 0.97))

    save_curved_surface_scatter(
        x_test, output / "fig_curved_truth.png",
        title="Truth",
        vmin=color_vmin, vmax=color_vmax,
    )
    for name in metrics_by_model:
        save_curved_surface_scatter(
            metrics_by_model[name]["_x_hat"],
            output / f"fig_curved_{name}.png",
            color_by=truth_radii,
            title=_label(name),
            vmin=color_vmin, vmax=color_vmax,
        )
        save_overlay_reconstruction(
            x_test,
            metrics_by_model[name]["_x_hat"],
            output / f"fig_curved_overlay_{name}.png",
            title=f"{_label(name)}: Original vs Reconstruction",
        )

    save_overlay_reconstruction_panels(
        x_test,
        {name: metrics_by_model[name]["_x_hat"] for name in metrics_by_model},
        output / "fig_curved_overlay_panels.png",
    )

    save_diagnostic_panel_set(metrics_by_model, output, p=config.p_homogeneity)

    save_hero_curved_surface(
        x_test,
        metrics_by_model,
        output / "fig_hero.png",
        vmin=color_vmin, vmax=color_vmax,
    )
    print(f"Wrote single-panel PNGs to {output}")

    diagnostic_dir = output / "diagnostic"
    diagnostic_dir.mkdir(exist_ok=True)
    plot_training_history(
        {name: metrics_by_model[name]["_history"] for name in metrics_by_model},
        diagnostic_dir / "training_history.png",
    )

    summary = {name: serializable(metrics_by_model[name]) for name in metrics_by_model}
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    alpha_ambient = summary["HomogeneousAE"]["hill_ambient"]["alpha"]
    print("\n=== exp08 summary ===")
    print(f"  alpha_true (Pareto target) = {alpha_true:.3f}")
    print(f"  hill_ambient (post-PIT)    = {alpha_ambient:.3f}")
    print(f"  Prop 1 target (amb/p)      = "
          f"{alpha_ambient / config.p_homogeneity:.3f}")
    for name in metrics_by_model:
        row = summary[name]
        print(
            f"  {name}:\n"
            f"    reconstruction_mse  = {row['reconstruction_mse']:.5f}\n"
            f"    hill_latent         = {row['hill_latent']['alpha']:.3f}\n"
            f"    hill_drift_latent   = {row['hill_drift_latent']:.4f}\n"
            f"    angular_tail_dist   = {row['angular_tail_distance']:.5f}"
        )


if __name__ == "__main__":
    main()
