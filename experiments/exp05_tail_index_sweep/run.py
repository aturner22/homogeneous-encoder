"""exp05: sweep tail index alpha for fixed (D=10, m=3, p=1).

Trains the three-model zoo across ``alpha in {1.5, 2.5, 4.0}`` with
multi-seed aggregation and saves three single-panel PNGs:

- ``fig_alpha_hill_drift.png`` — Proposition 1 drift
  ``|alpha_latent * p - alpha_ambient|`` vs alpha.
- ``fig_alpha_extrapolation.png`` — extrapolation MSE at lambda=10 vs alpha.
- ``fig_alpha_tail_mse.png`` — tail-conditional reconstruction MSE vs alpha.

The tail index is the key parameter in the paper's theory — at small
alpha the StandardAE baseline should fail far more dramatically than at
large alpha, because the training density in the tail gets sparser.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import FlexibleToyConfig, ensure_output_dir  # noqa: E402
from lib.sweep import MODEL_NAMES, run_flexible_toy_sweep, write_sweep_json  # noqa: E402
from lib.viz import save_exp02_panel_set, save_sweep_metric  # noqa: E402


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "results"
    base_config = FlexibleToyConfig(output_dir=str(output_dir))
    ensure_output_dir(base_config)

    alpha_values = [1.5, 2.5, 4.0]
    result = run_flexible_toy_sweep(
        base_config=base_config,
        parameter_name="alpha",
        parameter_values=alpha_values,
    )

    output = Path(base_config.output_dir)
    write_sweep_json(result, output / "sweep.json")

    series_by_model = result["metrics"]
    xlabel = r"tail index $\alpha$"

    save_sweep_metric(
        parameter_values=alpha_values,
        series_by_model=series_by_model,
        output_path=output / "fig_alpha_hill_drift.png",
        metric_key="hill_drift_latent",
        xlabel=xlabel,
        ylabel=r"$|\hat\alpha_{\mathrm{lat}}\, p - \hat\alpha_{\mathrm{amb}}|$",
    )
    save_sweep_metric(
        parameter_values=alpha_values,
        series_by_model=series_by_model,
        output_path=output / "fig_alpha_extrapolation.png",
        metric_key="extrapolation_mse_at_10",
        xlabel=xlabel,
        ylabel=r"extrapolation MSE at $\lambda=10$",
        yscale="log",
    )
    save_sweep_metric(
        parameter_values=alpha_values,
        series_by_model=series_by_model,
        output_path=output / "fig_alpha_tail_mse.png",
        metric_key="tail_conditional_mse",
        xlabel=xlabel,
        ylabel=r"tail-conditional MSE ($q_{0.95}$)",
        yscale="log",
    )
    for value, point in zip(alpha_values, result["per_seed_points"]):
        subdir = output / f"point_alpha={value}"
        subdir.mkdir(exist_ok=True)
        save_exp02_panel_set(
            {name: point[name][0] for name in MODEL_NAMES},
            subdir,
            p=base_config.p_homogeneity,
        )

    print(f"\nexp05 done. Sweep PNGs + per-point diagnostics written to {output}")


if __name__ == "__main__":
    main()
