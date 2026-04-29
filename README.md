# p-homogeneous encoder — reproduction repository

Code and experiments for the paper on p-homogeneous autoencoders that
preserve regular variation in the tails. This repository is structured
so that every figure in `2025_report/neurips_2025.tex` can be
regenerated from scratch or from cached training artifacts with a
single `make` target.

## Quickstart

```bash
uv sync                           # install python deps + editable-install experiments/lib/ as `lib`
uv sync --extra climate --group dev  # add xarray/cdsapi/netcdf4 + ruff/pytest
make reproduce                    # train + evaluate + plot every experiment, then compile paper
```

Climate experiments additionally require a CDS account and
`~/.cdsapirc`; see `experiments/climate/README.md`.

The `lib` package is installed in editable mode by `uv sync`, so every
`run.py` can simply `from lib.config import …` regardless of the
directory it is invoked from. There is no `sys.path` manipulation.

Worked example — one synthetic experiment end-to-end:

```bash
make exp01           # trains 3 models × 1 seed; writes experiments/exp01_curved_surface/results/seed0/*.pkl and fig_*.png
make exp01-plots     # replot from those pickles without retraining
```

`make reproduce` runs:

1. The seven synthetic experiments (`exp01`–`exp07`) — training takes
   longest for `exp02` (5 seeds × 3 models) and the four sweep
   experiments (`exp03`–`exp07`, several parameter grid points each).
2. The climate experiment (`experiments/climate/run_all.sh`) — requires
   ERA5 data; see `experiments/climate/README.md` for acquisition.
3. `pdflatex` on `2025_report/neurips_2025.tex` twice to resolve refs.

## Fast iteration: replot without retraining

After any experiment has been run once, its per-(seed, model)
**training artifacts** are cached as pickles under
`experiments/<exp>/results/<run>/seed<i>/<Model>.pkl`. These hold the
torch state-dict **and** the full evaluation dict (radii, Hill curves,
binned errors, training histories — everything the plot layer
consumes). Regenerating a figure never retrains.

```bash
make clean-plots         # delete every fig_*.{pdf,png}; keep the pickles
make plots               # regenerate every figure from the cached pickles
make paper               # pdflatex twice
```

`make plots` typically runs in well under a minute end-to-end because
no models are re-fit.

## Per-experiment targets

| Target | Experiment | Figures (in paper) | Notes |
|---|---|---|---|
| `make exp01` | Curved surface (D=3, m=2) | `fig:exp01-surfaces`, `fig:exp01-diagnostics`, `fig:exp01-correction`, `fig:exp01-extra` | Single-seed, Student-t tail |
| `make exp02` | Flexible toy headline (D=10, m=3, α=1.8) | `fig:headline`, `fig:exp02-correction`, `fig:exp02-extra` | Runs with `--num-seeds 5 --output-subdir seeds5` so Figure 3 has ±1σ bands |
| `make exp03` | Ambient-dim sweep (D ∈ {5, 10, 20}) | appendix | 3 grid points × 1 seed |
| `make exp04` | Intrinsic-dim sweep (m ∈ {2, 3, 5}) | appendix | |
| `make exp05` | Tail-index sweep (α ∈ {1.5, 2.5, 4.0}) | appendix | |
| `make exp06` | Homogeneity-degree sweep (p ∈ {0.5, 1.0, 2.0}) | appendix | Produces `fig_latent_hill_vs_p` |
| `make exp07` | Network-size sweep (hidden_dim ∈ {32, 64, 128, 256}) | appendix | |
| `make climate` | ERA5 u10 / tp / t2m / joint tensor | `fig:climate-*` | Requires `~/.cdsapirc`; see below |

Every experiment target accepts `ARGS=` to forward CLI flags to the
underlying `run.py`. For instance:

```bash
make exp02 ARGS=--force-retrain       # ignore cache and retrain
make exp05 ARGS=--force-retrain       # only exp05's cache is rebuilt
```

Plot-only equivalents exist for each: `make exp01-plots`,
`make exp02-plots`, …, `make climate-plots`.

## Artifact layout

```
experiments/<exp>/results/<run_name>/
├── config.json                 # run's TrainConfig snapshot (human-readable)
├── summary.json                # scalar aggregates across seeds
├── per_seed_metrics.json       # JSON-safe per-seed metrics (no raw arrays)
├── seed0/
│   ├── HomogeneousAE.pkl       # state_dict + full evaluation dict + history
│   ├── StandardAE.pkl
│   └── PCA.pkl
├── seed1/ ...                  # only for multi-seed runs
└── fig_*.{pdf,png}             # regenerated from the pickles
```

For sweep experiments (exp03–07) there is an extra level:
`results/<run>/point_<param>=<value>/seed<i>/<Model>.pkl`.

The pickle schema is documented in `experiments/lib/README.md`
(section **Artifacts**). It is version-stamped; incompatible changes
bump `ARTIFACT_VERSION` in `experiments/lib/artifacts.py`.

## Environment

- Python ≥ 3.12 (`pyproject.toml`).
- CPU is supported; CUDA is used automatically if available
  (`default_device` in `experiments/lib/config.py`).
- `uv sync` installs the full dependency set. Climate-only extras
  (`xarray`, `netcdf4`, `cdsapi`) are in the optional
  `[project.optional-dependencies] climate` group:
  `uv sync --extra climate`.

## Determinism

Every `run.py` calls `lib.determinism.enable_deterministic(seed)`
before training. This sets:

- `random`, `numpy`, `torch`, `torch.cuda` RNG states.
- `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`.
- `torch.backends.cudnn.deterministic = True`,
  `torch.use_deterministic_algorithms(True, warn_only=True)`.

CPU runs are bit-reproducible across machines with the same library
versions. GPU runs are deterministic to within cuDNN warnings for ops
without a deterministic kernel.

## Data dependencies

- **Synthetic experiments** (exp01–07): no external data. All datasets
  are produced from seeded generators in `experiments/lib/data.py`.
- **Climate**: ERA5 reanalysis via the Copernicus CDS API. Requires a
  free CDS account and `~/.cdsapirc`. See
  `experiments/climate/README.md` for the acquisition pipeline
  (`download.py` → `preprocess.py` → `run.py`).

## Repository layout

```
homogeneous-encoder/
├── Makefile                         # orchestrator (this is the entry point)
├── README.md                        # you are here
├── pyproject.toml                   # uv / pip dependencies
├── experiments/
│   ├── README.md                    # per-experiment table
│   ├── lib/                         # shared training/eval/viz library
│   │   ├── README.md                # library internals, artifact schema
│   │   ├── artifacts.py             # pickle save/load
│   │   ├── determinism.py           # enable_deterministic
│   │   ├── config.py                # TrainConfig, FlexibleToyConfig, ...
│   │   ├── data.py                  # toy data generators
│   │   ├── models.py                # HAE, StdAE, PCA
│   │   ├── train.py                 # training loop
│   │   ├── evaluation.py            # train_and_evaluate, metrics dict
│   │   ├── metrics.py               # Hill, tail, homogeneity metrics
│   │   ├── sweep.py                 # train_zoo_multiseed, run_flexible_toy_sweep
│   │   ├── cli.py                   # parse_standard_args, init_experiment
│   │   └── viz/                     # plot primitives (manifold, tail, sweep, panel sets)
│   ├── exp01_curved_surface/
│   ├── exp02_flexible_toy_ablation/
│   ├── exp03_ambient_dim_sweep/
│   ├── exp04_intrinsic_dim_sweep/
│   ├── exp05_tail_index_sweep/
│   ├── exp06_homogeneity_degree_sweep/
│   ├── exp07_network_size_sweep/
│   └── climate/
│       ├── README.md                # ERA5 acquisition + run_all.sh
│       ├── download.py
│       ├── preprocess.py
│       ├── run.py
│       ├── run_all.sh               # canonical 4-variable driver
│       └── viz_climate.py
└── 2025_report/
    ├── README.md                    # pdflatex instructions
    └── neurips_2025.tex
```

Paper sources live in `2025_report/`. The sibling `2026_report/` and
`archive/` trees are historical scratch space and are not part of the
active build.

## Linting and tests

```bash
uv run ruff check experiments/     # lint (config in pyproject.toml)
uv run pytest tests/               # smoke tests (artifact round-trip, determinism, Hill estimator)
```

Ruff is configured in `[tool.ruff]` / `[tool.ruff.lint]` (rules `E, F, I,
UP, B, SIM, PLC`). The dev group in `pyproject.toml` pulls `ruff` and
`pytest`; `uv sync --group dev` installs both.

## Known gotchas

- `exp02` **writes its cache under `results/seeds5/`** because the
  headline figure needs 5 seeds. The root-level `results/` holds an
  older 1-seed run for reference and is not part of the paper workflow.
- `make plots` errors loudly if any artifact is missing. Run the
  corresponding `make exp0N` (without `--plot-only`) first.
- The climate fast-path (`make climate-plots`) walks the four
  per-variable subdirectories (`u10/`, `tp/`, `t2m/`, `tensor/`) and
  only replots subdirectories that already have a `seed0/` cache.
