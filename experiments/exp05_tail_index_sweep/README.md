# exp05 — tail index sweep

## Hypothesis

The tail index α is the key parameter in the paper's theory. At
small α the tail is heavy and training density in the tail is
sparse; a StandardAE with no radial inductive bias should degrade
dramatically there, while HAE — structurally anchored to
Proposition 1 — should stay flat.

## Config

`lib.config.FlexibleToyConfig`. Swept axis: `alpha ∈ {1.5, 2.5, 4.0}`.
All other knobs (D=10, m=3, p=1, 5 seeds) live at the
`FlexibleToyConfig` defaults.

## Run

From the repo root:

```bash
make exp05          # train + plot (cache-aware)
make exp05-plots    # replot from cached pickles
```

## Outputs

`results/` holds three sweep panels: `fig_alpha_hill_drift.png`,
`fig_alpha_extrapolation.png`, `fig_alpha_tail_mse.png`, plus
`sweep.json`. Per-point caches live under
`results/point_alpha=<value>/seed<i>/`.
