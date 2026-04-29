# exp04 — intrinsic dimension sweep

## Hypothesis

Proposition 1 drift should be invariant to the intrinsic manifold
dimension m (the encoder is told m a priori). HAE should stay flat
while StandardAE drifts more as m grows — it has to invert a
higher-dimensional embedding without the homogeneity constraint.

## Config

`lib.config.FlexibleToyConfig`. Swept axis: `m ∈ {2, 3, 5}`. All
other knobs (D=10, p=1, alpha=2.5, 5 seeds) live at the
`FlexibleToyConfig` defaults.

## Run

From the repo root:

```bash
make exp04          # train + plot (cache-aware)
make exp04-plots    # replot from cached pickles
```

## Outputs

`results/` holds three sweep panels: `fig_m_hill_drift.png`,
`fig_m_extrapolation.png`, `fig_m_tail_mse.png`, plus `sweep.json`.
Per-point caches live under `results/point_m=<value>/seed<i>/`.
