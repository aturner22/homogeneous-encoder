"""exp03: sweep ambient dimension D for fixed intrinsic dim m=3.

Trains the three-model zoo across ``D in {5, 10, 20}`` with multi-seed
aggregation and saves three single-panel PNGs:

- ``fig_D_hill_drift.png`` — Proposition 1 drift
  ``|alpha_latent * p - alpha_ambient|`` vs D.
- ``fig_D_extrapolation.png`` — extrapolation MSE at lambda=10 vs D.
- ``fig_D_tail_mse.png`` — tail-conditional reconstruction MSE vs D.
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

    D_values = [5, 10, 20]
    result = run_flexible_toy_sweep(
        base_config=base_config,
        parameter_name="D",
        parameter_values=D_values,
    )

    output = Path(base_config.output_dir)
    write_sweep_json(result, output / "sweep.json")

    series_by_model = result["metrics"]
    xlabel = r"ambient dimension $D$"

    save_sweep_metric(
        parameter_values=D_values,
        series_by_model=series_by_model,
        output_path=output / "fig_D_hill_drift.png",
        metric_key="hill_drift_latent",
        xlabel=xlabel,
        ylabel=r"$|\hat\alpha_{\mathrm{lat}}\, p - \hat\alpha_{\mathrm{amb}}|$",
    )
    save_sweep_metric(
        parameter_values=D_values,
        series_by_model=series_by_model,
        output_path=output / "fig_D_extrapolation.png",
        metric_key="extrapolation_mse_at_10",
        xlabel=xlabel,
        ylabel=r"extrapolation MSE at $\lambda=10$",
        yscale="log",
    )
    save_sweep_metric(
        parameter_values=D_values,
        series_by_model=series_by_model,
        output_path=output / "fig_D_tail_mse.png",
        metric_key="tail_conditional_mse",
        xlabel=xlabel,
        ylabel=r"tail-conditional MSE ($q_{0.95}$)",
        yscale="log",
    )
    for value, point in zip(D_values, result["per_seed_points"]):
        subdir = output / f"point_D={value}"
        subdir.mkdir(exist_ok=True)
        save_exp02_panel_set(
            {name: point[name][0] for name in MODEL_NAMES},
            subdir,
            p=base_config.p_homogeneity,
        )

    print(f"\nexp03 done. Sweep PNGs + per-point diagnostics written to {output}")


if __name__ == "__main__":
    main()
