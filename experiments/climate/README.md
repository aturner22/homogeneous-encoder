# Climate (ERA5) experiment — scaffold only

This directory will hold the large-scale meteorological validation. The
scaffold is in place; the concrete data choices are **deferred** until a
later session at the user's request. Nothing in here runs ERA5
downloads or training until you explicitly configure it.

## What is decided

- Target: joint multivariate surface fields from ERA5.
- Access: `cdsapi` with `~/.cdsapirc` credentials on this machine.
- Model zoo: the same three models used in `exp02`
  (`HomogeneousAutoencoder`, `StandardAutoencoder`, `PCABaseline`).
- Evaluation: the same four metrics from `lib/metrics.py`.

## What is NOT yet decided (and should be decided before running)

- Variable list (e.g. 2m temperature, total precipitation, 10m wind
  components, mean sea level pressure).
- Geographic area (global / Europe / specific bounding box).
- Pressure levels (if any) — surface-only vs multi-level.
- Time span (years) and temporal resolution (hourly / daily / monthly
  maxima).
- Temporal aggregation / decorrelation (block maxima, thresholded
  exceedances, ...).
- How to stack variables and grid points into an ambient vector `x`:
  - per-grid-cell multivariate (each row = one time at one grid point),
  - regional vectorisation (each row = all grid points at one time), or
  - a fixed set of stations.

When those are decided, put them in a yaml file (see below).

## Pipeline

```
experiments/climate/
  download.py    # cdsapi fetch driven by a request yaml
  preprocess.py  # netCDF -> torch tensor with variable-stacking choice
  run.py         # loads the tensor, trains all 3 models, writes metrics
  config.yaml    # <-- you create this when the choices above are made
  data/          # gitignored; netCDF and .pt live here
  results/       # gitignored
```

All three scripts are importable without touching the network:
`preprocess.py` and `run.py` both support a `--dry-run` that verifies
imports and control flow on a synthetic tensor so the pipeline can be
smoke-tested before any real data exists.
