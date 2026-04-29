# exp06 — homogeneity degree sweep

## Hypothesis

The tightest numerical check of Proposition 1: if the encoder is
exactly p-homogeneous, the latent tail index should track
`α_ambient / p` for every p. The ambient Hill estimate is the
observable target — the true α is not used as a reference, so this
isolates the Proposition's claim from estimator bias.

## Config

`lib.config.FlexibleToyConfig`. Swept axis: `p_homogeneity ∈
{0.5, 1.0, 2.0}`. All other knobs (D=10, m=3, alpha=2.5, 5 seeds)
live at the `FlexibleToyConfig` defaults.

## Run

From the repo root:

```bash
make exp06          # train + plot (cache-aware)
make exp06-plots    # replot from cached pickles
```

## Outputs

`results/fig_latent_hill_vs_p.png` plots absolute latent Hill vs p
with the `α_ambient / p` reference curve overlaid.
`exp06_summary.json` records the worst homogeneity residual per
point for the appendix. Per-point caches live under
`results/point_p_homogeneity=<value>/seed<i>/`.
