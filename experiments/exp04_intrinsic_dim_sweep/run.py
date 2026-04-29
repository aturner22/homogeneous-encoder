"""exp04: sweep intrinsic dimension m for fixed ambient dim D=10.

Trains the three-model zoo across ``m in {2, 3, 5}`` with multi-seed
aggregation and saves three single-panel PNGs:

- ``fig_m_hill_drift.png`` — Proposition 1 drift
  ``|alpha_latent * p - alpha_ambient|`` vs m.
- ``fig_m_extrapolation.png`` — extrapolation MSE at lambda=10 vs m.
- ``fig_m_tail_mse.png`` — tail-conditional reconstruction MSE vs m.

Note: ``m`` is passed to every model at construction time (we assume
the manifold dimension is known a priori).
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
        parameter_name="m",
        parameter_values=[2, 3, 5],
        xlabel="Intrinsic Dimension",
        fig_prefix="m",
        force_retrain=args.force_retrain,
        require_cache=args.plot_only,
    )
    print(f"\nexp04 done. Outputs in {config.output_dir}")


if __name__ == "__main__":
    main()
