# exp03 — ambient dimension sweep

## Hypothesis

The Proposition 1 drift `|α_latent · p − α_ambient|` should stay
near-zero for HAE as the ambient dimension D grows, while StandardAE
drifts upward because it has no structural constraint coupling
latent and ambient tails.

## Config

`lib.config.FlexibleToyConfig`. Swept axis: `D ∈ {5, 10, 20}`. All
other knobs (m=3, p=1, alpha=2.5, 5 seeds) live at the
`FlexibleToyConfig` defaults.

## Run

From the repo root:

```bash
make exp03          # train + plot (cache-aware)
make exp03-plots    # replot from cached pickles
```

## Outputs

`results/` holds three sweep panels: `fig_D_hill_drift.png`,
`fig_D_extrapolation.png`, `fig_D_tail_mse.png`, plus `sweep.json`
with aggregated mean/std for every scalar metric. Per-point caches
live under `results/point_D=<value>/seed<i>/`.
