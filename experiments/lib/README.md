# `experiments/lib/` — shared library for synthetic-toy experiments

Shared training, evaluation, sweep, and visualisation code used by all six
experiment drivers (`exp01`–`exp06`). The library provides three model
implementations, a single training loop, a unified evaluation function, a
parameter-sweep harness, and publication-quality plot primitives.

## Architecture

```
config.py ───┐
data.py      ├──► models.py ──► train.py ──► evaluation.py ──► sweep.py
             │                                                    │
             └──► viz.py  ◄───────────────────────────────────────┘
```

Every experiment driver follows the same pipeline:

```
config → generate data → build model zoo → train_zoo_multiseed / run_flexible_toy_sweep
       → save_exp02_panel_set + save_sweep_metric → summary.json
```

---

## Module reference

### `config.py`

Dataclass-based experiment configuration.

| Class | Used by | Key fields |
|---|---|---|
| `TrainConfig` | base | `n_train=25000`, `epochs=500`, `patience=50`, `batch_size=512`, `hidden_dim=256`, `hidden_layers=4`, `p_homogeneity=1.0`, `lambda_cor=0.1`, `num_seeds=1`, `device`, `seed=42` |
| `CurvedSurfaceConfig(TrainConfig)` | exp01 | `student_df_t=2.8`, `D=3, m=2` fixed, `p_homogeneity=2.0`, smaller net (`hidden_dim=192`, `hidden_layers=3`) |
| `FlexibleToyConfig(TrainConfig)` | exp02–06 | `D=10`, `m=3`, `alpha=1.8`, `kappa=1.0`, `curvature_rank=8`, `embedding_seed=1234` |

- `default_device()` — returns `"cuda"` if available, else `"cpu"`.
- `ensure_output_dir(config)` — creates `config.output_dir` if it doesn't exist.

`alpha=1.8` (default) places the flexible toy in the infinite-variance
regime, where the StandardAE baseline's tail failure is most visible.
`p_homogeneity` is the paper's _p_ for the _p_-homogeneous encoder.

### `data.py`

Toy data generators with analytically known tail structure.

**Curved surface** (exp01 only):

- `generate_curved_surface(n, seed, ...)` — samples a smooth surface in
  R^3 with heavy-tailed radial profile from a correlated Student-_t_
  (`df = student_df_t`). The tail index equals `student_df_t`.

**Flexible toy manifold** (exp02–06):

- `FlexibleToyEmbedding(D, m, kappa, curvature_rank, embedding_seed)` —
  fixed smooth embedding `phi(y) = A y + kappa B tanh(C y)` from R^m to
  R^D. Asymptotically linear: `phi(lambda y) / lambda → A y` as
  `lambda → inf`, so `x = phi(y)` inherits the exact tail index of
  `||y||`.
- `generate_flexible_toy(n, *, D, m, alpha, kappa, ...)` — draws
  `r ~ Pareto(alpha)`, `u ~ Uniform(S^{m-1})`, builds `y = r * u`, and
  applies `FlexibleToyEmbedding`. The resulting `x` is regularly varying
  in R^D with tail index exactly `alpha`.

### `models.py`

Three models sharing a common `encode/decode/forward → dict` interface.

**`HomogeneousAutoencoder(D, m, hidden_dim, hidden_layers, p_homogeneity)`**

The paper's exact architecture:

- **Encoder** (exactly _p_-homogeneous by construction):
  ```
  theta = x / ||x||,  r = ||x||
  a(theta) = softplus(A_phi(theta))     # positive scalar
  e(theta) = normalize(E_phi(theta))    # unit vector in R^m
  f(x) = r^p * a(theta) * e(theta)
  ```

- **Decoder** (asymptotically _1/p_-homogeneous via bounded correction delta):
  ```
  rho = ||z||,  eta = z / rho
  a_tilde   = softplus(A_psi(eta))
  e_tilde   = normalize(E_psi(eta))
  s         = 1 / (1 + rho)
  delta     = Delta_psi(eta, s) - Delta_psi(eta, 0)
  c_tilde   = normalize(e_tilde + delta)
  h(z)      = rho^(1/p) * a_tilde * c_tilde
  ```
  `delta` vanishes as `rho → inf` (since `s → 0`), giving asymptotic
  homogeneity. This is verified empirically by the correction diagnostics.

- **Forward dict keys:** `z, r, theta, a, e, x_hat, rho, eta, a_tilde,
  e_tilde, delta, c_tilde`.

**`StandardAutoencoder(D, m, hidden_dim, hidden_layers)`** — MLP
autoencoder with MSE loss. No homogeneity, no correction.

**`PCABaseline(D, m)`** — uncentered top-_m_ SVD. Exactly 1-homogeneous
(the encoder is a linear map `x → x @ V`), so it transports regular
variation perfectly by Proposition 1. Used as the linear baseline that
preserves tails but loses on reconstruction quality.

**Loss functions:**

- `homogeneous_loss(x, forward_pass, lambda_cor)` — MSE +
  `lambda_cor * mean(log1p(rho) * ||c_tilde - e_tilde||)`. The
  `log1p(rho)` weighting pushes the correction penalty toward the tail.
- `standard_loss(x, forward_pass, lambda_cor)` — MSE only (signature
  parity with `homogeneous_loss`).
- `count_parameters(model)` — total trainable parameters.

### `train.py`

Training loop shared by all neural models.

- `train(model, train_data, val_data, config, *, verbose=True)` — Adam
  optimiser with best-validation-total checkpointing and early stopping
  (stops when val loss hasn't improved for `config.patience` epochs).
  Returns a `history` dict with keys `epoch`, `train_total`,
  `train_reconstruction`, `train_penalty`, `val_total`,
  `val_reconstruction`, `val_penalty` (each a list of floats).
- `fit_pca_baseline(model, train_data, val_data, config)` — one-shot SVD
  fitting. Returns a history dict with the same schema (single-element
  lists) for plotting compatibility.

`_select_loss_fn(model)` dispatches `HomogeneousAutoencoder` →
`homogeneous_loss`, `StandardAutoencoder` → `standard_loss`.

**PCA must be trained via `fit_pca_baseline`, not `train`** — the neural
loop assumes an `optimizer.step()` gradient path.

### `metrics.py`

Tail-preservation evaluation metrics in four groups.

**Hill estimator:**

- `hill_curve(radii)` → `(k_values, alpha_hat_k)` for the full stability
  plot used by the latent Hill figure.
- `hill_estimate(radii, k_fraction=0.1)` → point estimate dict with
  `alpha, k, n, std_err, ci_low, ci_high`.
- `hill_drift(alpha_latent, alpha_ambient, p)` →
  `|alpha_latent * p - alpha_ambient|`. **The preferred scalar metric.**
  Proposition 1 says `alpha_latent * p = alpha_ambient` for a
  _p_-homogeneous encoder, so this measures departure from the identity
  without depending on the unobservable true alpha. (Both Hill estimates
  share the same finite-sample bias, so the difference cancels it.)

**Extreme-quantile / angular:**

- `extreme_quantile_errors(reference, candidate, levels=(0.99, 0.999, 0.9999))`
  — absolute and relative errors of the top-quantile radii.
- `angular_tail_distance(x, x_other, *, radial_quantile=0.95, num_slices=100)`
  — sliced Wasserstein distance between the unit-sphere projections of
  each dataset's top radial quantile.

**Encoder-homogeneity diagnostics:**

- `encoder_homogeneity_error(model, x_probe, p, scales)` → per-scale
  relative residual `||f(lambda x) - lambda^p f(x)||^2 / ||lambda^p f(x)||^2`
  plus a `worst` scalar across all scales. HAE gives ~1e-10 (numerical
  precision); StdAE gives order-unity.
- `homogeneity_scan(model, x_probe, p, scales)` → full scale-vs-residual
  curve for the homogeneity residual figure.

**Reconstruction-quality:**

- `binned_reconstruction_error(x, x_hat, *, n_bins=12, log_bins=True)` —
  per-radial-bin `||x - x_hat||^2` with median, IQR, and counts.
- `tail_conditional_mse(x, x_hat, *, radial_quantile=0.95)` — MSE
  restricted to the top radial quantile. The key separator: StdAE beats
  HAE on bulk MSE but loses on the tail.
- `extrapolation_mse(model, embedding, *, D, m, scale_multipliers, ...)`
  — synthesises samples at radii outside the training support through the
  true embedding and measures per-scale reconstruction MSE.
- `tail_angular_coordinates(x_true, x_by_model, *, radial_quantile, rank=2)`
  — projects tail cones onto a common SVD-of-truth basis for cross-model
  scatter comparison.

### `evaluation.py`

Unified evaluation pipeline.

- `evaluate_model(model, x_test, *, alpha_true, p_encoder, ...)` — runs
  every metric from `metrics.py` on a held-out tensor and returns a flat
  dict. Key behaviour:
  - Stores both the deprecated `alpha_error_*` scalars (comparing to
    unobservable true alpha) and the preferred `hill_drift_*` scalars
    (comparing to ambient Hill), for backwards compatibility.
  - Extracts `_delta_norm = ||delta||` per sample only for
    `HomogeneousAutoencoder` (the decoder correction diagnostic).
  - Keeps raw arrays prefixed with underscore (`_ambient_radii`,
    `_latent_radii`, `_reconstructed_radii`, `_x_test`, `_x_hat`,
    `_binned_error`, `_extrapolation`, `_homogeneity_scan`,
    `_delta_norm`, `_history`) for downstream plotting.
  - Extrapolation is silently skipped when `embedding`/`embedding_dims`
    are not provided (exp01 case — the curved surface has no single
    embedding module).
- `train_and_evaluate(model_name, model, train_data, val_data, test_data,
  config, *, alpha_true, p_for_hill, embedding=None, ...)` — per-model
  pipeline: dispatches to `train` or `fit_pca_baseline`, then runs
  `evaluate_model`. Attaches `_history` to the metrics dict.
- `serializable(metrics)` — strips underscore-prefixed keys for JSON.
- `write_metrics_json(metrics_by_model, path)` — JSON-safe dump.

### `sweep.py`

Parameter-sweep harness for `FlexibleToyConfig` experiments.

**Constants:**

- `MODEL_NAMES = ("HomogeneousAE", "StandardAE", "PCA")` — canonical
  model order, load-bearing for plot colouring and legend ordering.
- `SCALAR_METRIC_KEYS` — the exact set of scalar keys aggregated into
  mean/std tables:
  ```
  reconstruction_mse, tail_conditional_mse, hill_drift_latent,
  hill_drift_reconstructed, alpha_error_reconstructed,
  alpha_error_latent, angular_tail_distance, extrapolation_mse_at_10
  ```

**Functions:**

- `build_embedding(config)` — rebuilds the `FlexibleToyEmbedding` used by
  `generate_flexible_toy` so `extrapolation_mse` can feed out-of-range
  samples through the same map.
- `build_model_zoo(config)` → `{"HomogeneousAE": ..., "StandardAE": ..., "PCA": ...}`.
- `train_zoo_multiseed(config, *, verbose=False)` → trains and evaluates
  the three-model zoo `config.num_seeds` times, returning:
  ```python
  {"per_seed": [{model_name: full_metrics_dict}, ...],
   "aggregate": {model_name: {metric_key: {"mean", "std", "values"}}}}
  ```
  Seed offsets are 97-spaced so the data splits differ each run. The same
  embedding is reused across seeds.
- `run_flexible_toy_sweep(base_config, parameter_name, parameter_values)`
  — one `train_zoo_multiseed` per grid point. Returns:
  ```python
  {"parameter_name", "parameter_values",
   "metrics": {model: {metric_key: {"mean": array, "std": array}}},
   "raw": [...],
   "per_seed_points": [{model: [seed0_metrics, seed1_metrics, ...]}, ...]}
  ```
  `per_seed_points` carries the raw underscore-prefixed arrays needed by
  `save_exp02_panel_set`. `write_sweep_json` only serialises the `raw`
  field, so JSON safety is preserved.
- `write_sweep_json(sweep_result, path)`.

### `viz.py`

Publication-quality matplotlib primitives. One PNG per call.

**Module-level setup:** `_apply_aesthetic()` runs at import, setting STIX
serif font, hidden top/right spines, soft grid, inward ticks, DPI 220.
All primitives inherit this automatically.

**Per-model palette:**

| Model | Colour | Label | Marker |
|---|---|---|---|
| HomogeneousAE | `#d62728` (red) | HAE | `o` |
| StandardAE | `#1f77b4` (blue) | AE | `s` |
| PCA | `#2ca02c` (green) | PCA | `^` |

**Sweep primitives:**

- `save_sweep_metric(parameter_values, series_by_model, output_path, *,
  metric_key, xlabel, ylabel, yscale=None, reference_curve=None)` —
  generic shaded-band curve per model, used by exp03–06.

**Tail-diagnostic primitives:**

- `save_latent_hill_curves(latent_curves_by_model, output_path, *,
  ambient_curve, alpha_ambient, p)` — Hill stability curves for each
  model's latent radii, with the ambient curve and the Proposition 1
  target `alpha_ambient / p` as dashed references.
- `save_latent_vs_ambient_radius(radii_by_model, output_path, *, p)` —
  log-log scatter of `||z||` vs `||x||` with a slope-_p_ reference line.
  HAE lands on the line; StdAE bends in the tail. Rasterised.
- `save_extrapolation_curve(extrap_by_model, output_path)` — MSE vs scale
  multiplier (log-log).
- `save_binned_recon_error(binned_by_model, output_path)` — median binned
  MSE vs ambient radius.

**HAE correction (delta) diagnostics:**

- `save_correction_magnitude_scatter(latent_radii, delta_norms, output_path)`
  — scatter of `||delta||` vs `||z||` with a quantile-binned mean line.
  Shows the correction decaying as `rho` grows.
- `save_correction_by_regime(latent_radii, delta_norms, output_path)` —
  bar chart of mean `||delta||` by latent-radius percentile bins
  `[<=50%, 50-75%, 75-90%, 90-95%, 95-99%, >99%]`. Bars decrease into
  the tail, visualising asymptotic homogeneity.

**Panel-set helper:**

- `save_exp02_panel_set(point, output_dir, *, p, prefix="")` — writes the
  full diagnostic panel set (up to 6 PNGs) for one training point.
  Composes the primitives above. Extrapolation and correction plots are
  skipped if the required data is absent.

**Exp01-specific:**

- `save_curved_surface_scatter(data, output_path, *, color_by, title)` —
  single 3-D scatter coloured by ambient radius.
- `save_homogeneity_scan(scan_by_model, output_path)` — encoder
  homogeneity residual vs scale (log-log).

**Diagnostic (not paper figures):**

- `plot_training_history(histories_by_model, output_path)` — two-panel
  training/validation loss curves.
- `plot_curved_surface_diagnostic(...)` — legacy 2x2 compound figure
  (retained for back-compat, no longer called by exp01).

---

## End-to-end driver flow

Every experiment driver follows the same pattern:

1. Build a config (`FlexibleToyConfig` or `CurvedSurfaceConfig`), call
   `ensure_output_dir`.
2. Seed `numpy` and `torch` from `config.seed`.
3. **Single-point** (exp01, exp02): call `train_zoo_multiseed(config)`.
   **Sweep** (exp03–06): call `run_flexible_toy_sweep(base_config,
   parameter_name, values)`.
4. Pass `result["per_seed"][0]` or each sweep point's first seed to
   `save_exp02_panel_set` for the diagnostic panels.
5. (Sweeps only) Call `save_sweep_metric` for each aggregate metric curve.
6. Persist `summary.json` / `sweep.json` / `per_seed_metrics.json`.

---

## Invariants and gotchas

- **`MODEL_NAMES` order is load-bearing** — viz primitives iterate in this
  order to keep colours and legend entries consistent across figures.
- **Only `HomogeneousAutoencoder.decode` returns `delta`**; StdAE and PCA
  do not. `evaluate_model` only attaches `_delta_norm` for HAE.
- **`_extrapolation` is absent when no `embedding` is supplied** — that's
  why exp01 (curved surface) produces 5 diagnostic PNGs instead of 6.
- **`hill_drift_latent`** (`|alpha_lat * p - alpha_amb|`) is the preferred
  scalar for the Proposition 1 consistency check. `alpha_error_latent`
  (comparing to unobservable true alpha) is retained only for
  backwards-compatible sweep JSONs.
- **Homogeneity is structural for HAE** — the scalar
  `homogeneity_error.worst` appears in every summary table, but the
  visual verification (the homogeneity-scan panel) lives in a single
  PNG per experiment.
- **`_apply_aesthetic()` runs at import time** — do not call it again in
  drivers.
- **PCA must be trained via `fit_pca_baseline`**, not `train` — the neural
  training loop assumes a gradient-based optimiser.
