# Latent Space Structure Clarification

## TL;DR

**The full latent representation is 2D, not 1D!**

- **Angular latent**: `z ∈ ℝ¹` (learned, compresses directions on S² from 2D to 1D)
- **Radial component**: `radial_scalar ∈ ℝ⁺` (preserved exactly, NOT learned)
- **Total latent dimension**: 2D = 1D (angular) + 1D (radial)

The parameter `latent_dim=1` is **misleading**—it only refers to the angular latent dimension, not the full latent space.

---

## Detailed Breakdown

### Data Space
- **Input**: Points `x ∈ ℝ³` (ambient 3D space)
- **Manifold**: 2D surface embedded in ℝ³
- **Parameterization**: `x = center + ρ × u` where:
  - `ρ ∈ ℝ⁺` is the radial distance
  - `u ∈ S²` is the direction (unit sphere, intrinsically 2D)

### Encoder Output (Full Latent Representation)

The encoder produces **TWO** components:

1. **Angular latent** `z ∈ ℝ¹`:
   ```python
   z = angular_encoder(u)  # MLP compresses direction on S² to 1D
   ```
   - **Learned** via backpropagation
   - Compresses 2D angular information (direction on sphere) to 1D
   - This is the bottleneck being tested

2. **Radial scalar** `radial_scalar ∈ ℝ⁺`:
   ```python
   radial_scalar = ||x - center||₂
   ```
   - **NOT learned** (no gradients flow through this)
   - Simply extracted and preserved
   - Passed directly from encoder to decoder

**Total latent dimensionality**: 2D (one for each component)

### Decoder Input (Full Latent Representation)

The decoder takes **BOTH** components:

```python
def decode(self, radial_scalar, angular_latent):
    u_hat = normalize(angular_decoder(angular_latent))  # 1D -> S²
    x_hat = center + radial_scalar * u_hat              # Combine angular + radial
    return x_hat
```

### What Does `latent_dim=1` Mean?

The parameter `latent_dim` in `ExperimentSpec` refers **ONLY** to the angular latent dimension:

| Experiment | `latent_dim` | Angular latent | Radial | **Total latent** | Compression? |
|------------|--------------|----------------|--------|------------------|--------------|
| `curve_m1_latent1` | 1 | ℝ¹ | ℝ⁺ | 2D | No (1D → 1D on curve) |
| `surface_m2_latent2` | 2 | ℝ² | ℝ⁺ | 3D | No (2D → 2D on S²) |
| `surface_m2_latent1_compressed` | **1** | **ℝ¹** | ℝ⁺ | **2D** | **Yes (2D → 1D on S²)** |

So `surface_m2_latent1_compressed` has:
- Manifold dimension: 2 (2D surface)
- Angular latent dimension: 1 (compressed)
- **Full latent dimension: 2** (1 angular + 1 radial)

### Why This Design?

The motivation is to **isolate and study angular compression**:

1. **Radial information is "easy"**: It's just a scalar distance, no geometric complexity
2. **Angular information is "hard"**: Directions on S² have spherical geometry, compression is non-trivial
3. **By preserving radial exactly**, we can focus on testing whether the angular part can be compressed

The experiment asks: *"Can we compress the 2D angular structure (directions on S²) down to 1D while still reconstructing the manifold accurately?"*

### Common Misconceptions

❌ **Wrong**: "This is a 1D latent autoencoder"
- The latent space is 2D: `(z, radial_scalar)`

❌ **Wrong**: "We're compressing 3D points to 1D"
- We're compressing 2D directions (on S²) to 1D, while preserving radial (1D) separately

❌ **Wrong**: "The radial component is learned"
- The radial component is extracted in the encoder and passed through unchanged

✅ **Correct**: "We're testing whether 1D angular latent + 1D preserved radial (total 2D) can represent a 2D manifold"

---

## Implications for Analysis

### Collision Analysis
When analyzing "latent collisions", remember:
- Points far apart in ambient space might have similar `z` (angular latent)
- BUT they might have different `radial_scalar`
- So they won't actually collide in the **full 2D latent space** `(z, radial_scalar)`

**Current collision metric only checks `z` distances**, ignoring radial differences!

### Intrinsic Dimensionality
If you compute the intrinsic dimensionality of the latent space, you should get **≈2**, not 1, because `radial_scalar` varies independently of `z`.

### Information Content
The information content should be measured across **both** latent components:
- `I(x; z, radial_scalar)` (correct)
- NOT just `I(x; z)` (incomplete)

---

## Recommendations

### For Future Work

1. **Rename `latent_dim` → `angular_latent_dim`** to avoid confusion
2. **Report full latent dimensionality** (angular + radial) in results
3. **Modify collision metrics** to consider full 2D latent space
4. **Consider learning radial as well** (current design assumes radial is "free")

### For Understanding Results

When interpreting the experiment `surface_m2_latent1_compressed`:
- The **compression bottleneck** is in the angular component only (2D → 1D)
- The **total representation** is still 2D (1D angular + 1D radial)
- Reconstruction quality tests whether 1D is sufficient for the angular structure
- But the radial structure is trivially preserved (no compression)
