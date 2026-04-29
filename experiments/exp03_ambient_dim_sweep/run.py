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

from lib.cli import init_experiment, parse_standard_args
from lib.config import FlexibleToyConfig
from lib.sweep import run_and_plot_param_sweep


def main() -> None:
    args = parse_standard_args(description=__doc__)
    config = init_experiment(Path(__file__), FlexibleToyConfig)
    run_and_plot_param_sweep(
        config,
        parameter_name="D",
        parameter_values=[5, 10, 20],
        xlabel="Ambient Dimension",
        fig_prefix="D",
        force_retrain=args.force_retrain,
        require_cache=args.plot_only,
    )
    print(f"\nexp03 done. Outputs in {config.output_dir}")


if __name__ == "__main__":
    main()
