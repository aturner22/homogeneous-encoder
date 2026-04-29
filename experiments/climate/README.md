# `experiments/climate/` — ERA5 reanalysis validation

Real-data companion to the synthetic experiments. Trains the same
three-model zoo (`HomogeneousAE`, `StandardAE`, `PCA`) on surface
meteorological fields from the ERA5 reanalysis and evaluates the same
tail-preservation metrics that drive the paper's headline figures.

## Pipeline

```
download.py   cdsapi -> data/*.nc                      (needs ~/.cdsapirc)
preprocess.py data/*.nc -> data/era5_*.pt + era5_meta.npz
run.py        data/era5_*.pt -> results/<var>/{seed0/,fig_*,...}
run_all.sh    loops run.py over {u10, tp, t2m, tensor}  (canonical driver)
```

All four scripts honour `--dry-run`, which short-circuits data loading
with a synthetic tensor so the pipeline can be smoke-tested before any
real data exists.

## Data acquisition

1. **CDS account and API key** — sign up at
   <https://cds.climate.copernicus.eu> and place your key in
   `~/.cdsapirc`. `cdsapi` is part of the optional dependency group:
   ```bash
   uv sync --extra climate
   ```

2. **Configure the request** — `config.yaml` declares the variable
   list, geographic area, time span, and temporal resolution. Inspect
   it before downloading — ERA5 tiles can be large. A `profiles:`
   block defines overlays (e.g. `hires`) that `download.py` /
   `preprocess.py` apply via `--profile NAME`.

3. **Download** and preprocess:
   ```bash
   uv run python download.py --config config.yaml
   uv run python preprocess.py --config config.yaml
   # or with a profile overlay:
   uv run python download.py --config config.yaml --profile hires
   ```
   `preprocess.py` writes one tensor per variable plus a joint
   `era5_tensor.pt` and a `era5_meta.npz` carrying the lat/lon grid
   metadata used by `viz_climate.plot_spatial_recon_error` etc.

## Running

The canonical driver is `run_all.sh`, which loops `run.py` over the
four tensors with per-variable hyperparameters:

```bash
bash run_all.sh
```

Or invoke `run.py` directly:

```bash
uv run python run.py --tensor data/era5_u10.pt --results-subdir u10
uv run python run.py --tensor data/era5_tp.pt  --results-subdir tp --tail-holdout-quantile 0.9
```

`run.py --help` lists every tuning knob (`--alpha-assumed`,
`--auto-alpha`, `--lambda-max`, `--ratchet-ema`, `--weight-decay`, …).

## Reproducibility flags

Consistent with the synthetic experiments, `run.py` supports:

| Flag | Behaviour |
|---|---|
| *(default)* | Cache-aware. Loads `seed0/<Model>.pkl` if present, trains otherwise. |
| `--plot-only` | Load cache only; error if missing. |
| `--force-retrain` | Ignore cache and retrain every model. |

From the repo root, the Makefile exposes:

```bash
make climate             # run_all.sh over all four tensors
make climate-plots       # replot every variable that already has a seed0/ cache
```

## Outputs

Per-variable subdirectory layout:

```
results/<var>/
├── config.json
├── metrics.json                    # scalar metrics, JSON-safe
├── seed0/
│   ├── HomogeneousAE.pkl
│   ├── StandardAE.pkl
│   └── PCA.pkl
├── fig_latent_hill.{pdf,png}
├── fig_extrapolation.{pdf,png}     # diagnostic panel set
├── fig_hae_correction_vs_radius.{pdf,png}
├── return_level.{pdf,png}          # POT/GPD return-level plot
├── pca_scree.{pdf,png}
├── HomogeneousAE/spatial_recon_error.{pdf,png}, sample_fields.{pdf,png}, ...
├── StandardAE/...
├── PCA/...
└── diagnostic/training_history.{pdf,png}
```

## Notes

- Data is never committed: `data/` and `results/` are gitignored.
- The joint `era5_tensor.pt` stacks all four surface variables into a
  single ambient vector. It exercises a much larger `D` than the
  synthetic experiments (hundreds of features) and is the most
  demanding run in the pipeline.
- `run.py` estimates the ambient tail index α from the training radii
  with a Hill estimator by default; pass `--alpha-assumed` to override.
- The return-level plot assumes 1460 samples/year (four 6-hourly ERA5
  samples/day); adjust via `n_per_year` in `viz_climate.py` if the
  upstream temporal aggregation differs.
