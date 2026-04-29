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

from lib.cli import init_experiment, parse_standard_args
from lib.config import FlexibleToyConfig
from lib.sweep import run_and_plot_param_sweep


def main() -> None:
    args = parse_standard_args(description=__doc__)
    config = init_experiment(Path(__file__), FlexibleToyConfig)
    run_and_plot_param_sweep(
        config,
        parameter_name="alpha",
        parameter_values=[1.5, 2.5, 4.0],
        xlabel="Tail Index",
        fig_prefix="alpha",
        force_retrain=args.force_retrain,
        require_cache=args.plot_only,
    )
    print(f"\nexp05 done. Outputs in {config.output_dir}")


if __name__ == "__main__":
    main()
