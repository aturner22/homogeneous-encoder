# exp09 — curved surface with two-sided Pareto-margin pre-standardisation

## Hypothesis

Same toy setup as `exp08_curved_surface_pareto_margins`, but each
ambient coordinate is mapped to a **continuous symmetric Lomax** target
rather than a one-sided standard Pareto. The inverse CDF used in the
PIT step is

```
F⁻¹(u) = sign(2u − 1) · ((1 − |2u − 1|)⁻¹ᐟᵅ − 1)
```

Both tails are regularly varying with index `α = pareto_target_alpha`,
the support is the full real line, and `F⁻¹(0.5) = 0`. Geometric
symmetry of the original 3-D manifold around the origin survives the
transform — unlike the one-sided variant in exp08, where every
coordinate ends up positive.

## Config

`lib.config.CurvedSurfaceParetoMarginsTwoSidedConfig` extends
`CurvedSurfaceParetoMarginsConfig` and only overrides
`pareto_target_kind = "two_sided"`. All other knobs (PIT threshold,
target tail index, training schedule, learnable centring) inherit from
the one-sided exp08 config.

## Run

```bash
PYTHONPATH=experiments python experiments/exp09_curved_surface_pareto_margins_two_sided/run.py
PYTHONPATH=experiments python experiments/exp09_curved_surface_pareto_margins_two_sided/run.py --plot-only
```

## Outputs

In `results/`:

- `fig_pareto_marginal_histograms.png` — symlog x-axis on the lower
  row so both heavy tails of the post-transform marginals show.
- `fig_pareto_marginal_hill.png` — Hill estimates of `|x|` on the
  post-transform marginals (since both tails carry index `α`).
- The same downstream suite as exp01 / exp08: `fig_hero.png`, 3-D
  ambient scatters and reconstruction overlays, latent Hill curves,
  latent-vs-ambient radius, HAE correction diagnostics, training
  history.
- `metrics.json`; `seed0/` cached artifacts for `--plot-only`.
