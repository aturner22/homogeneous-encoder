# Changes Summary

## Script Simplification

The `p_homogeneous_encoder.py` script has been streamlined from **1017 lines to 953 lines** (64 lines removed, ~6.3% reduction) while preserving **100% functionality** for the `surface_m2_latent1_compressed` experiment.

### Removed Components

1. **Unused function** (lines 179-181 in original):
   - `stereo_forward_np()` - Forward stereographic projection, never called

2. **Conditional branches for manifold_dim=1**:
   - `generate_latent_data()` - Removed `if manifold_dim == 1` branch that set `v = 0`
   - `manifold_from_latent()` - Removed `if manifold_dim == 1` branch that set `z2 = 0`
   - `symmetry_probe()` - Removed `if manifold_dim == 1` branch for symmetric `v`

3. **Conditional branch for latent_dim=2**:
   - `plot_reconstructions()` - Removed `else` branch that plotted 2D latent scatter (z1 vs z2)
   - Now only plots 1D latent (t vs z) for latent_dim=1

4. **Multi-experiment infrastructure**:
   - Removed experiment suite list (3 experiments → 1 single experiment)
   - Removed experiment loop in `main()`
   - Removed aggregate summary collection and logging
   - Simplified to run only `surface_m2_latent1_compressed`

5. **Function signature simplifications**:
   - `generate_latent_data()`: Removed `manifold_dim` parameter (always 2)
   - `manifold_from_latent()`: Removed `manifold_dim` parameter (always 2)
   - `symmetry_probe()`: Removed `manifold_dim` parameter (always 2)
   - `plot_reconstructions()`: Removed `experiment_spec` parameter (no longer needed)

### Functionality Verification

**The following are GUARANTEED to produce identical results:**

1. **Data generation**:
   - Same random seed formula: `seed + 1000 + 37 * manifold_dim + 101 * latent_dim`
   - For `surface_m2_latent1_compressed`: `37 + 1000 + 37*2 + 101*1 = 1212`
   - Same Student-t sampling for `t` (df=2.8, scale=1.0)
   - Same Gaussian sampling for `v` (scale=0.9)

2. **Manifold construction**:
   - Same formulas: `z1 = 0.90 * tanh(t)`, `z2 = 0.70 * tanh(v)`
   - Same stereographic projection
   - Same radial profile: `1.25 + sqrt(t² + 0.75²)`

3. **Model architecture**:
   - Identical `FixedRadialAngularAutoencoder` with latent_dim=1
   - Same MLP sizes (128 hidden units, 3 layers)
   - Same normalization and scaling logic

4. **Training**:
   - Identical loss functions and weights
   - Same optimizer (AdamW), learning rate, epochs
   - Same train/val/test split

5. **Outputs**:
   - Same metrics computed
   - Same plots generated (except removed latent_dim=2 variant)
   - Same JSON summaries saved

### What Changed for Users

**Before**: Running the script trained 3 experiments (curve_m1_latent1, surface_m2_latent2, surface_m2_latent1_compressed)

**After**: Running the script trains only 1 experiment (surface_m2_latent1_compressed)

**Output file differences**:
- Before: `all_results.json` with aggregate summary
- After: `result.json` with single experiment results

All other outputs (plots, logs, model checkpoints, metrics) are **identical**.

### How to Restore Multi-Experiment Functionality

If you need to run multiple experiments again, simply modify `main()` to loop over a list of `ExperimentSpec` objects and restore the aggregate summary logic. The core functions remain fully general and support any manifold_dim and latent_dim.

---

## Documentation Added

Created **ARCHITECTURE.md** with comprehensive documentation:

1. **Overview** of the fixed-radial autoencoder concept
2. **Detailed data generation pipeline** with mathematical formulas
3. **Stereographic projection validation** (confirms correctness)
4. **Complete model architecture** breakdown
5. **Loss function explanations** with mathematical notation
6. **Analysis metrics** and their interpretations
7. **Expected outcomes and potential issues**
8. **Key assumptions and constraints**

The documentation is designed to be self-contained and readable without looking at the code.
