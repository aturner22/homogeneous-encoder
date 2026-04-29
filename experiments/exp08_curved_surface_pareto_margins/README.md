# exp08 — curved surface with Pareto-margin pre-standardisation

## Hypothesis

Same setup as `exp01_curved_surface` (a 2-D manifold embedded in R^3
with heavy Student-t radii), but each ambient coordinate is
pre-standardised to a standard Pareto distribution before training.
The transform is a per-marginal GPD-tail PIT (rank-based bulk +
Generalised Pareto fit above the configurable threshold) followed by
the standard Pareto inverse CDF F^{-1}(u) = (1-u)^{-1/α}. The copula
(joint dependence) is preserved; only the marginals are reshaped.

This isolates whether the HAE's tail-preservation guarantee transfers
cleanly when the input has perfectly aligned, parametrically known
marginals — a setting where the regular-variation framework should
match the data exactly by construction.

## Config

`lib.config.CurvedSurfaceParetoMarginsConfig` (extends
`CurvedSurfaceConfig`). Sets `pre_standardize_to_pareto_margins=True`;
inherits all other curved-surface knobs. Configurable transform
parameters (defaults inherited from `TrainConfig`):

- `pareto_target_alpha` (default 1.0) — target Pareto tail index;
- `pit_threshold_quantile` (default 0.975) — GPD threshold quantile.

## Run

```bash
PYTHONPATH=experiments python experiments/exp08_curved_surface_pareto_margins/run.py
PYTHONPATH=experiments python experiments/exp08_curved_surface_pareto_margins/run.py --plot-only
```

## Outputs

In `results/`:

- `fig_pareto_marginal_histograms.png` and `fig_pareto_marginal_hill.png`
  — per-dimension diagnostics of the transform itself (before vs after).
- The same downstream suite as exp01: `fig_hero.png`, 3-D ambient
  scatters and reconstruction overlays (`fig_curved_*`,
  `fig_curved_overlay_*`), latent Hill curves, latent-vs-ambient
  radius, HAE correction diagnostics, training history.
- `metrics.json` carries scalar metrics per model.
- `seed0/` holds per-model pickled artifacts for cache-aware re-plots.
