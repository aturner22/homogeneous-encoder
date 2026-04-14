"""exp07: sweep network hidden dimension for fixed D=10, m=3.

Trains the three-model zoo across ``hidden_dim in {32, 64, 128, 256}``
with matched StdAE parameter counts at each point. Tests the hypothesis
that HAE degrades earlier than StdAE as network capacity shrinks, since
HAE's capacity is split across 5 constrained sub-networks.

Saves four single-panel PNGs:

- ``fig_hidden_dim_hill_drift.png``  — Proposition 1 drift vs hidden_dim
- ``fig_hidden_dim_extrapolation.png`` — extrapolation MSE at lambda=10
- ``fig_hidden_dim_tail_mse.png``    — tail-conditional MSE
- ``fig_hidden_dim_reconstruction.png`` — overall reconstruction MSE

Also prints a table of (HAE hidden_dim, HAE params, StdAE matched
hidden_dim, StdAE params) for transparency.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import FlexibleToyConfig, ensure_output_dir  # noqa: E402
from lib.models import (  # noqa: E402
    HomogeneousAutoencoder,
    StandardAutoencoder,
    compute_matched_hidden_dim,
    count_parameters,
)
from lib.sweep import MODEL_NAMES, run_flexible_toy_sweep, write_sweep_json  # noqa: E402
from lib.viz import save_exp02_panel_set, save_sweep_metric  # noqa: E402


def _print_param_table(
    hidden_dims: list[int],
    base_config: FlexibleToyConfig,
) -> None:
    """Print a table of HAE and matched StdAE parameter counts."""
    print("\n  hidden_dim |   HAE params | StdAE h_dim | StdAE params |  ratio")
    print("  " + "-" * 66)
    for hd in hidden_dims:
        hae = HomogeneousAutoencoder(
            D=base_config.D, m=base_config.m,
            hidden_dim=hd, hidden_layers=base_config.hidden_layers,
            p_homogeneity=base_config.p_homogeneity,
        )
        hae_p = count_parameters(hae)
        stdae_hd = compute_matched_hidden_dim(
            hae_p, base_config.D, base_config.m, base_config.hidden_layers,
        )
        stdae = StandardAutoencoder(
            D=base_config.D, m=base_config.m,
            hidden_dim=stdae_hd, hidden_layers=base_config.hidden_layers,
        )
        stdae_p = count_parameters(stdae)
        ratio = stdae_p / hae_p if hae_p > 0 else float("nan")
        print(f"  {hd:>10d} | {hae_p:>12,} | {stdae_hd:>11d} | {stdae_p:>12,} | {ratio:.4f}")


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "results"
    base_config = FlexibleToyConfig(output_dir=str(output_dir))
    ensure_output_dir(base_config)

    hidden_dims = [32, 64, 128, 256]

    _print_param_table(hidden_dims, base_config)

    result = run_flexible_toy_sweep(
        base_config=base_config,
        parameter_name="hidden_dim",
        parameter_values=hidden_dims,
    )

    output = Path(base_config.output_dir)
    write_sweep_json(result, output / "sweep.json")

    series_by_model = result["metrics"]
    xlabel = "hidden dimension"

    save_sweep_metric(
        parameter_values=hidden_dims,
        series_by_model=series_by_model,
        output_path=output / "fig_hidden_dim_reconstruction.png",
        metric_key="reconstruction_mse",
        xlabel=xlabel,
        ylabel="reconstruction MSE",
        yscale="log",
    )
    save_sweep_metric(
        parameter_values=hidden_dims,
        series_by_model=series_by_model,
        output_path=output / "fig_hidden_dim_hill_drift.png",
        metric_key="hill_drift_latent",
        xlabel=xlabel,
        ylabel=r"$|\hat\alpha_{\mathrm{lat}}\, p - \hat\alpha_{\mathrm{amb}}|$",
    )
    save_sweep_metric(
        parameter_values=hidden_dims,
        series_by_model=series_by_model,
        output_path=output / "fig_hidden_dim_extrapolation.png",
        metric_key="extrapolation_mse_at_10",
        xlabel=xlabel,
        ylabel=r"extrapolation MSE at $\lambda=10$",
        yscale="log",
    )
    save_sweep_metric(
        parameter_values=hidden_dims,
        series_by_model=series_by_model,
        output_path=output / "fig_hidden_dim_tail_mse.png",
        metric_key="tail_conditional_mse",
        xlabel=xlabel,
        ylabel=r"tail-conditional MSE ($q_{0.95}$)",
        yscale="log",
    )

    for value, point in zip(hidden_dims, result["per_seed_points"]):
        subdir = output / f"point_hidden_dim={value}"
        subdir.mkdir(exist_ok=True)
        save_exp02_panel_set(
            {name: point[name][0] for name in MODEL_NAMES},
            subdir,
            p=base_config.p_homogeneity,
        )

    print(f"\nexp07 done. Sweep PNGs + per-point diagnostics written to {output}")


if __name__ == "__main__":
    main()
