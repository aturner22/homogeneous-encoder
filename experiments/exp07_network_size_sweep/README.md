# exp07 — network size sweep

## Hypothesis

HAE's capacity is split across five constrained sub-networks
(encoder, decoder, correction). StandardAE concentrates all
parameters in a single encode/decode pair. So as hidden width
shrinks, HAE should hit the capacity floor earlier. StandardAE hidden
dim is matched to HAE's total parameter count at each point.

## Config

`lib.config.FlexibleToyConfig`. Swept axis:
`hidden_dim ∈ {32, 64, 128, 256}`. All other knobs (D=10, m=3, p=1,
alpha=2.5, 5 seeds) live at the `FlexibleToyConfig` defaults.

## Run

From the repo root:

```bash
make exp07          # train + plot (cache-aware)
make exp07-plots    # replot from cached pickles
```

## Outputs

`results/` holds four sweep panels:
`fig_hidden_dim_hill_drift.png`, `fig_hidden_dim_extrapolation.png`,
`fig_hidden_dim_tail_mse.png`, `fig_hidden_dim_reconstruction.png`,
plus `sweep.json`. A parameter-count table is printed to stdout
during training for transparency. Per-point caches live under
`results/point_hidden_dim=<value>/seed<i>/`.
