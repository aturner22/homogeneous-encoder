"""Train and evaluate the three-model zoo on a preprocessed ERA5 tensor.

Loads a torch tensor produced by ``preprocess.py``, splits it into
train/val/test, and runs the same three-way comparison as ``exp02`` with
the same metrics. Auto-estimates the ambient tail index α from the
training data via the Hill estimator unless ``--alpha-assumed`` is set.

Hyperparameters (epochs, hidden_dim, latent_dim, lambda_*, ...) are read
from the ``per_variable:`` block of ``config.yaml`` keyed by
``--var`` (e.g. ``u10``, ``tp``, ``t2m``, ``tensor``). Missing keys
fall through to ``per_variable.default``.

Usage:
    python experiments/climate/run.py --config config.yaml --var tensor
    python experiments/climate/run.py --config config.yaml --var u10
    python experiments/climate/run.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from lib.artifacts import artifact_path
from lib.cli import parse_standard_args
from lib.config import FlexibleToyConfig, ensure_output_dir, save_config
from lib.data import generate_flexible_toy
from lib.determinism import enable_deterministic
from lib.evaluation import train_zoo, write_metrics_json
from lib.metrics import hill_estimate
from lib.preprocessing import fit_apply_pareto_margins
from lib.models import (
    HomogeneousAutoencoder,
    PCABaseline,
    StandardAutoencoder,
    compute_matched_hidden_dim,
    count_parameters,
)
from lib.viz import (
    plot_training_history,
    save_diagnostic_panel_set,
    save_marginal_pareto_hill,
    save_marginal_pareto_histograms,
)

MODEL_NAMES = ("HomogeneousAE", "StandardAE", "PCA")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is not installed.") from exc
    with open(path, encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def _resolve_variable_config(yaml_config: dict[str, Any], var: str) -> dict[str, Any]:
    """Return per-variable hyperparameters, with `default` as fallback."""
    per_variable = yaml_config.get("per_variable", {}) or {}
    default = dict(per_variable.get("default", {}))
    overrides = per_variable.get(var, {}) or {}
    default.update(overrides)
    return default


def _split(tensor: torch.Tensor, fractions=(0.7, 0.15, 0.15)):
    total = len(tensor)
    generator = torch.Generator().manual_seed(0)
    permutation = torch.randperm(total, generator=generator)
    n_train = int(fractions[0] * total)
    n_val = int(fractions[1] * total)
    train_indices = permutation[:n_train]
    val_indices = permutation[n_train : n_train + n_val]
    test_indices = permutation[n_train + n_val :]
    return tensor[train_indices], tensor[val_indices], tensor[test_indices]


def _tail_holdout_split(
    tensor: torch.Tensor, tail_quantile: float, val_fraction: float = 0.15,
):
    """Hold out the top-radius samples as an extrapolation test set.

    Samples with ``‖x‖ >= quantile(tail_quantile)`` go to test. The
    remainder (bulk) is randomly split into train/val. This guarantees
    ``max(train_radius) < min(test_radius)`` — a true out-of-distribution
    extrapolation regime for the supervisor's hypothesis: HAE's
    decoder is approximately homogeneous at large radius, so it should
    reconstruct the tail better than AE, which has no such inductive bias.
    """
    radii = torch.linalg.norm(tensor, dim=1)
    threshold = torch.quantile(radii, tail_quantile)
    tail_mask = radii >= threshold
    tail_indices = torch.nonzero(tail_mask, as_tuple=False).squeeze(-1)
    bulk_indices = torch.nonzero(~tail_mask, as_tuple=False).squeeze(-1)

    generator = torch.Generator().manual_seed(0)
    bulk_perm = bulk_indices[torch.randperm(len(bulk_indices), generator=generator)]
    n_val = int(val_fraction * len(bulk_perm))
    val_indices = bulk_perm[:n_val]
    train_indices = bulk_perm[n_val:]
    return (
        tensor[train_indices], tensor[val_indices], tensor[tail_indices],
        float(threshold),
    )


def _load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"No tensor at {path}. Run preprocess.py first.")
    tensor = torch.load(path)
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected torch.Tensor in {path}, got {type(tensor)}")
    return tensor.to(torch.float32)


def _synthetic_tensor() -> torch.Tensor:
    return generate_flexible_toy(
        sample_count=4000,
        D=8,
        m=3,
        alpha=2.5,
        kappa=0.3,
        curvature_rank=6,
        embedding_seed=17,
        sample_seed=17,
    )


def _plot_pca_scree(
    train_data: torch.Tensor, chosen_m: int, path: Path,
) -> None:
    """PCA scree curve: cumulative explained variance vs latent dimension.

    Diagnostic to sanity-check the chosen m against a linear lower bound on
    intrinsic dimension. HAE/StdAE can do no better than PCA in the linear
    sense, so this curve is an optimistic floor for where recon MSE can go.
    """
    X = train_data.numpy()
    singular_values = np.linalg.svd(X, compute_uv=False)
    eig = singular_values ** 2
    cum = np.cumsum(eig) / eig.sum()

    dims = np.arange(1, len(cum) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dims, cum, color="black", linewidth=1)
    for frac in (0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, frac) + 1)
        ax.axhline(frac, color="gray", linestyle=":", linewidth=0.6)
        ax.text(dims[-1], frac, f"  {int(frac * 100)}%: m={k}",
                va="center", ha="right", fontsize=8)
    ax.axvline(chosen_m, color="tab:red", linestyle="--", linewidth=1,
               label=f"chosen m={chosen_m}")
    chosen_frac = cum[min(chosen_m, len(cum)) - 1]
    ax.set_xlabel("latent dimension m")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"PCA scree (chosen m captures {chosen_frac:.1%} variance)")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PCA scree: m={chosen_m} captures {chosen_frac:.1%} variance "
          f"(90%@m={int(np.searchsorted(cum, 0.90) + 1)}, "
          f"95%@m={int(np.searchsorted(cum, 0.95) + 1)}, "
          f"99%@m={int(np.searchsorted(cum, 0.99) + 1)})")


def _estimate_alpha(train_data: torch.Tensor) -> float:
    """Estimate ambient tail index from training radii via Hill estimator."""
    radii = np.linalg.norm(train_data.numpy(), axis=1)
    result = hill_estimate(radii, k_fraction=0.1)
    alpha = result["alpha"]
    ci_lo, ci_hi = result["ci_low"], result["ci_high"]
    print(f"Auto-estimated ambient alpha: {alpha:.3f}  "
          f"(95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
    return alpha


def _climate_extra_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=str,
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to YAML config with per_variable hyperparameter block.",
    )
    parser.add_argument(
        "--var", type=str, default="tensor",
        help="Short variable name selecting per_variable[<var>] hyperparams "
             "(e.g. tensor, u10, tp, t2m). Unknown names fall back to default.",
    )
    parser.add_argument(
        "--tensor", type=str, default=None,
        help="Override tensor path. Defaults to per_variable[var].tensor.",
    )
    parser.add_argument(
        "--results-subdir", type=str, default=None,
        help="Subdirectory under results/ (defaults to --var).",
    )
    parser.add_argument(
        "--meta", type=str, default="data/era5_meta.npz",
        help="Path to grid metadata for spatial plots.",
    )
    parser.add_argument(
        "--alpha-assumed", type=float, default=None,
        help="Manual ambient tail index override (skips auto-estimation).",
    )
    parser.add_argument(
        "--tail-holdout-quantile", type=float, default=None,
        help="If set, hold out samples with radius above this quantile as "
             "the test set, training on the bulk only.",
    )


def main() -> None:
    args = parse_standard_args(description=__doc__, extra=_climate_extra_args)

    root = Path(__file__).resolve().parent

    # Load YAML hyperparameters for the requested variable.
    yaml_config = _load_yaml(Path(args.config))
    var_cfg = _resolve_variable_config(yaml_config, args.var)
    n_per_year = float(yaml_config.get("samples_per_year", 1460))

    # Resolve tensor and results paths.
    tensor_rel = args.tensor or var_cfg.get("tensor", "data/era5_tensor.pt")
    results_subdir = args.results_subdir or args.var
    output_dir = root / "results" / results_subdir

    # Load data.
    if args.dry_run:
        print("[dry-run] using synthetic tensor")
        tensor = _synthetic_tensor()
    else:
        tensor = _load_tensor(root / tensor_rel)

    print(f"Tensor shape: {tuple(tensor.shape)}")
    D = int(tensor.shape[1])

    # Intrinsic dimension: YAML override (skipped under --dry-run since the
    # synthetic tensor is small), else a conservative auto-pick.
    if "latent_dim" in var_cfg and not args.dry_run:
        m = int(var_cfg["latent_dim"])
    else:
        m = min(8, max(2, D // 4))
    print(f"D={D}, m={m}")

    # Estimate alpha from the FULL tensor so the drift metric still
    # references the true ambient tail even when training on the bulk.
    if args.alpha_assumed is not None:
        alpha = args.alpha_assumed
        print(f"Using provided alpha: {alpha:.3f}")
    else:
        alpha = _estimate_alpha(tensor)

    # Split: random 70/15/15 by default, or tail-holdout for the
    # extrapolation experiment.
    tail_threshold: float | None = None
    if args.tail_holdout_quantile is not None:
        train_data, val_data, test_data, tail_threshold = _tail_holdout_split(
            tensor, tail_quantile=args.tail_holdout_quantile,
        )
        print(f"Tail-holdout split (q={args.tail_holdout_quantile}): "
              f"train={len(train_data)}, val={len(val_data)}, "
              f"test={len(test_data)} (radii >= {tail_threshold:.4f})")
        train_radii = np.linalg.norm(train_data.numpy(), axis=1)
        test_radii = np.linalg.norm(test_data.numpy(), axis=1)
        print(f"  max(train radius) = {train_radii.max():.4f}, "
              f"min(test radius) = {test_radii.min():.4f}, "
              f"max(test radius) = {test_radii.max():.4f}")
    else:
        train_data, val_data, test_data = _split(tensor)
        print(f"Split: train={len(train_data)}, val={len(val_data)}, "
              f"test={len(test_data)}")

    # Optional Pareto-margin pre-standardisation. Fit on train only and
    # apply to val/test to avoid leakage. Applied *after* preprocess.py's
    # log1p + z-score (so e.g. tp is already on (-inf, +inf) when we hit
    # this line) and *after* split.
    output_dir.mkdir(parents=True, exist_ok=True)
    if var_cfg.get("pre_standardize_to_pareto_margins", False):
        pareto_alpha = float(var_cfg.get("pareto_target_alpha", 1.0))
        threshold_q = float(var_cfg.get("pit_threshold_quantile", 0.975))
        pareto_kind = str(var_cfg.get("pareto_target_kind", "one_sided"))
        print(f"Pre-standardising marginals to {pareto_kind} Pareto(alpha="
              f"{pareto_alpha:g}) via GPD-tail PIT (threshold q={threshold_q})")
        train_raw = train_data
        train_data, (val_data, test_data), _ = fit_apply_pareto_margins(
            train_raw, [val_data, test_data],
            pareto_alpha=pareto_alpha,
            threshold_quantile=threshold_q,
            pareto_kind=pareto_kind,
        )
        save_marginal_pareto_histograms(
            train_raw.numpy(),
            train_data.numpy(),
            output_dir / "fig_pareto_marginal_histograms.png",
            pareto_kind=pareto_kind,
        )
        save_marginal_pareto_hill(
            train_raw.numpy(),
            train_data.numpy(),
            output_dir / "fig_pareto_marginal_hill.png",
            target_alpha=pareto_alpha,
            pareto_kind=pareto_kind,
        )
        # The post-PIT data has known target tail index by construction;
        # override the auto-estimated alpha so downstream Hill drift is
        # computed against the right target.
        alpha = pareto_alpha
        print(f"Overriding ambient alpha to Pareto target: {alpha:.3f}")

    # PCA scree — sanity-check m against linear intrinsic dim.
    _plot_pca_scree(train_data, m, output_dir / "pca_scree.png")

    # Build config. Only copy whitelisted keys out of var_cfg so a typo
    # in YAML is visible rather than silently stashed on the dataclass.
    hyperparam_keys = (
        "epochs", "batch_size", "hidden_dim", "hidden_layers",
        "recon_patience", "penalty_patience", "warmup_max_epochs",
        "learning_rate",
        "lambda_max", "lambda_growth",
        "recon_tolerance", "ratchet_ema", "weight_decay",
        "pre_standardize_to_pareto_margins",
        "pareto_target_alpha", "pit_threshold_quantile",
        "pareto_target_kind",
    )
    config_overrides: dict[str, Any] = dict(
        output_dir=str(output_dir), D=D, m=m, alpha=alpha,
    )
    for key in hyperparam_keys:
        if key in var_cfg:
            config_overrides[key] = var_cfg[key]
    config = FlexibleToyConfig(**config_overrides)
    ensure_output_dir(config)
    save_config(config)

    # Build models with parameter matching.
    hae = HomogeneousAutoencoder(
        D=D, m=m,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        p_homogeneity=config.p_homogeneity,
        learnable_centre=config.learnable_centre,
    )
    hae_params = count_parameters(hae)
    stdae_hidden = compute_matched_hidden_dim(
        hae_params, D, m, config.hidden_layers,
    )
    stdae = StandardAutoencoder(
        D=D, m=m,
        hidden_dim=stdae_hidden,
        hidden_layers=config.hidden_layers,
    )
    pca = PCABaseline(D=D, m=m)

    print(f"  HAE  hidden_dim={config.hidden_dim:4d}  params={hae_params:>10,}")
    print(f"  AE   hidden_dim={stdae_hidden:4d}  params={count_parameters(stdae):>10,}")

    # Train and evaluate (cache-aware).
    enable_deterministic(config.seed)
    seed_dir = Path(config.output_dir) / "seed0"
    models_by_name = {"HomogeneousAE": hae, "StandardAE": stdae, "PCA": pca}
    metrics_by_model = train_zoo(
        models_by_name,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
        alpha_true=alpha,
        p_for_hill=config.p_homogeneity,
        seed_dir=seed_dir,
        force_retrain=args.force_retrain,
        require_cache=args.plot_only,
    )

    # Write metrics JSON.
    write_metrics_json(metrics_by_model, Path(config.output_dir) / "metrics.json")

    # Standard exp02-style diagnostic plots.
    out = Path(config.output_dir)
    save_diagnostic_panel_set(metrics_by_model, out, p=config.p_homogeneity)
    print(f"Wrote diagnostic PNGs to {out}")

    # Training history plots.
    histories = {name: metrics_by_model[name]["_history"] for name in MODEL_NAMES}
    diag_dir = out / "diagnostic"
    diag_dir.mkdir(exist_ok=True)
    plot_training_history(histories, diag_dir / "training_history.png")

    # Climate-specific plots (if metadata available).
    meta_path = root / args.meta
    if meta_path.exists() and not args.dry_run:
        try:
            from viz_climate import (
                plot_marginal_distributions,
                plot_return_level_curves,
                plot_sample_fields,
                plot_spatial_recon_error,
                plot_tail_qq,
            )
            meta = np.load(meta_path, allow_pickle=True)
            lats = meta["lats"]
            lons = meta["lons"]
            var_names = list(meta["var_names"])
            n_cells = int(meta["n_cells"])
            # If this is a univariate tensor, restrict var_names to the one
            # variable present (inferred from tensor filename: era5_<short>.pt).
            n_vars_in_tensor = D // n_cells
            if n_vars_in_tensor < len(var_names):
                tensor_stem = Path(tensor_rel).stem  # e.g. "era5_u10"
                short = tensor_stem.replace("era5_", "")
                if short in var_names:
                    var_names = [short]
                else:
                    var_names = var_names[:n_vars_in_tensor]

            # Use test set for diagnostics.
            x_test_np = test_data.numpy()
            for name in MODEL_NAMES:
                x_hat = metrics_by_model[name]["_x_hat"]
                model_dir = out / name
                model_dir.mkdir(exist_ok=True)

                plot_spatial_recon_error(
                    x_test_np, x_hat, lats, lons, var_names, n_cells,
                    model_dir / "spatial_recon_error.png",
                )
                plot_sample_fields(
                    x_test_np, x_hat, lats, lons, var_names, n_cells,
                    [0, len(x_test_np) // 2, -1],
                    model_dir / "sample_fields.png",
                )
                plot_marginal_distributions(
                    x_test_np, x_hat, var_names, n_cells,
                    model_dir / "marginal_distributions.png",
                )
                plot_tail_qq(
                    x_test_np, x_hat, var_names, n_cells,
                    model_dir / "tail_qq.png",
                )

            # One return-level plot per variable, overlaying empirical
            # vs HAE vs AE. Per-sample norm over grid cells is the scalar.
            series: dict[str, np.ndarray] = {
                "empirical": np.linalg.norm(x_test_np, axis=1),
            }
            for name in MODEL_NAMES:
                xh = metrics_by_model[name]["_x_hat"]
                series[name] = np.linalg.norm(xh, axis=1)
            plot_return_level_curves(
                series,
                out / "return_level.png",
                n_per_year=n_per_year,
                title=f"{var_names[0]} return levels (POT/GPD)" if var_names else None,
            )
            print(f"Wrote climate-specific plots to {out}")
        except ImportError:
            print("viz_climate.py not found, skipping climate-specific plots")
        except Exception as exc:
            print(f"Warning: climate plots failed: {exc}")

    # Print summary.
    print(f"\n=== Climate run summary (alpha_ambient={alpha:.3f}) ===")
    header = f"{'metric':<30}" + "".join(f"{name:>18}" for name in MODEL_NAMES)
    print(header)
    for key in ("reconstruction_mse", "tail_conditional_mse",
                "hill_drift_latent", "hill_drift_reconstructed"):
        row = f"{key:<30}"
        for name in MODEL_NAMES:
            val = metrics_by_model[name].get(key)
            if val is not None:
                row += f"  {float(val):>15.5f}"
            else:
                row += f"  {'N/A':>15}"
        print(row)

    print(f"\nDone. Metrics written to {Path(config.output_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()
