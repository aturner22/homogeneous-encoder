# `experiments/` — one directory per experiment

Every experiment exposes a single `run.py` CLI entry point and, on
invocation, writes its artifacts under `results/` in the same
directory. All of them share the training / evaluation / viz library
under `experiments/lib/`. See `experiments/lib/README.md` for the
library reference and the pickle-artifact schema.

## Quick reference

| Dir | Purpose | Paper figures |
|---|---|---|
| `exp01_curved_surface/` | Curved surface in R³ with a Student-t radial profile | `fig:exp01-*` |
| `exp02_flexible_toy_ablation/` | Flexible toy manifold, 3-model zoo, multi-seed headline | `fig:headline`, `fig:exp02-*` |
| `exp03_ambient_dim_sweep/` | Sweep ambient dim `D ∈ {5, 10, 20}` | appendix |
| `exp04_intrinsic_dim_sweep/` | Sweep intrinsic dim `m ∈ {2, 3, 5}` | appendix |
| `exp05_tail_index_sweep/` | Sweep Pareto tail index `α ∈ {1.5, 2.5, 4.0}` | appendix |
| `exp06_homogeneity_degree_sweep/` | Sweep `p ∈ {0.5, 1.0, 2.0}`; latent Hill vs `p` | `fig_latent_hill_vs_p` |
| `exp07_network_size_sweep/` | Sweep `hidden_dim ∈ {32, 64, 128, 256}` | appendix |
| `climate/` | ERA5 reanalysis on 4 variables (u10, tp, t2m, joint) | `fig:climate-*` |

## Common CLI surface

Every `run.py` accepts:

| Flag | Behaviour |
|---|---|
| *(default)* | Cache-aware. Trains any model whose pickle is missing; loads the rest. |
| `--plot-only` | Load the cache only; error if any pickle is missing. Never retrains. |
| `--force-retrain` | Ignore the cache and retrain every model. |

`exp02_flexible_toy_ablation/run.py` has additional flags for
`--num-seeds`, `--alpha`, `--kappa`, `--tail-holdout-quantile`,
`--output-subdir` (see the Makefile for the paper's default
configuration).

`climate/run.py` has a much richer CLI (see `run_all.sh`).

## Recommended entry point

Invoke experiments through the repo-level `Makefile`:

```bash
make exp02                   # run one experiment
make exp02-plots             # replot that experiment from cache
make plots                   # replot every experiment
make reproduce               # full rebuild + paper compile
```

`make` forwards `ARGS=...` to the underlying `run.py`, so
`make exp04 ARGS=--force-retrain` is available.

## Determinism

Each `run.py` calls `lib.determinism.enable_deterministic(seed)`
before any data generation or model construction. Combined with seeded
data generators, synthetic experiments produce bit-identical scalar
metrics on CPU across repeated runs.

## Directory conventions

```
<exp>/
├── run.py          # CLI entry point
└── results/        # artifacts + figures (gitignored or cleaned per-run)
    ├── config.json
    ├── seed0/<Model>.pkl      # state_dict + full evaluation dict
    └── fig_*.{pdf,png}
```

Sweep experiments add one more nesting level:

```
<exp>/results/point_<param>=<value>/seed<i>/<Model>.pkl
```
