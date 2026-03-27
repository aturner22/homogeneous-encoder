# P-Homogeneous Autoencoder with Adaptive Asymptotic Homogeneity

## Problem Setup

### Data Properties

We have data **X** with the following properties:

1. **Manifold support**: X lies on a 2D manifold M ⊂ ℝ³
2. **Regular variation in ambient space**: The distribution of X exhibits regular variation in ℝ³:
   ```
   t P(X/t ∈ A) → μ(A)  as t → ∞
   ```
   for some limit measure μ (the tail measure)

3. **Goal**: Learn an encoding f: ℝ³ → ℝ² that:
   - Compresses the 2D manifold to 2D latent space (no dimensional reduction)
   - Preserves the tail measure via pushforward: μ_W = f_* μ_X
   - Allows faithful reconstruction and generation
   - Handles non-homogeneous manifolds (not necessarily radially symmetric)

### Why Homogeneity?

**Key result from extreme value theory**: If f is p-homogeneous (f(λx) = λ^p f(x)), then the pushforward of a regularly varying measure is also regularly varying with transformed tail index.

This allows us to:
- Perform EVT analysis in the lower-dimensional latent space
- Generate new samples from tail distributions
- Preserve tail index in the latent representation

**Challenge**: Real manifolds are not necessarily homogeneous everywhere, but may be asymptotically homogeneous in the tail (where regular variation applies).

## Mathematical Framework

### Decomposition

Any point x ∈ ℝ³ can be decomposed as:
```
x = center + r · u
```
where:
- r = ||x - center|| ∈ ℝ⁺ (radius)
- u = (x - center)/r ∈ S² (direction on 2-sphere)

### Product-Form Encoding

We use a product-form latent representation:
```
w = r^p · z
```
where:
- p > 0 is the homogeneity parameter
- z ∈ ℝ² is an angular encoding (depends only on direction u)

This construction ensures **encoder p-homogeneity**:
```
encode(λx) = (λr)^p · z = λ^p · encode(x)
```

### Latent Space Structure

The 2D latent w ∈ ℝ² has natural polar coordinates:
- Radius: r_w = ||w|| = r^p · ||z||
- Angle: θ = w/||w|| ∈ S¹ (1D circle, represented as 2D unit vector)

This gives us **2 degrees of freedom** to encode the 2D manifold:
- No dimensional reduction!
- Both degrees of freedom (r_w, θ) are used by the decoder

## Architecture: Three-Component Decoder with Adaptive Homogeneity

### Encoder (P-Homogeneous)

The encoder maps x ∈ ℝ³ to w ∈ ℝ²:

**Step 1**: Decompose input
```
r = ||x - center||
u = (x - center)/r  ∈ S²
r^p = r^p  (radial component raised to power p)
```

**Step 2**: Angular encoding (decomposed into two components)
```
e(u): S² → ℝ²    (MLP + normalization to S¹)
a(u): S² → ℝ⁺    (MLP + softplus)
z = a(u) · e(u)  (scalar × unit vector)
```

**Step 3**: Combine with radius
```
w = r^p · z
```

**Key properties**:
- **P-homogeneous**: encode(λx) = λ^p encode(x) ✓
- **Bounded scaling**: a(u) is implicitly bounded (continuous function on compact S²)
- **Tail preservation**: Tail measure of X is preserved in W via pushforward

### Decoder (Asymptotically Homogeneous with Three Components)

The decoder maps w ∈ ℝ² back to x̂ ∈ ℝ³ using **three separate components**:

**Step 1**: Extract latent components
```
r_w = ||w||
θ = w/||w||  ∈ S¹ (represented as 2D unit vector)
r_reconstructed = r_w^(1/p)  (invert the p-power)
```

**Step 2**: Three-component decoding

1. **Homogeneous baseline direction**:
   ```
   c_base(θ): S¹ → S²
   ```
   - Only depends on angle θ
   - Provides baseline angular reconstruction
   - This component is inherently homogeneous

2. **Sparse non-homogeneous correction**:
   ```
   c_correction(θ, r_w): S¹ × ℝ → ℝ³
   ```
   - Depends on both angle θ and log(r_w)
   - Subject to **L1 sparsity penalty**
   - Allows deviation from homogeneity when needed
   - Encouraged to → 0 in the tail

3. **Angle-dependent magnitude scaling**:
   ```
   b(θ): S¹ → ℝ⁺
   ```
   - Only depends on angle θ
   - Bounded (continuous on compact domain)
   - Learns to invert encoder's a(u)

**Step 3**: Combine components
```
c = normalize(c_base(θ) + c_correction(θ, r_w))  [project to S²]
h = b(θ) · c
x̂ = center + r_reconstructed · h
```

## Why This Design Works

### Encoder: Exactly P-Homogeneous

The encoder treats any point λx by:
1. Extracting direction u (same for λx as for x)
2. Scaling radius: r → λr
3. Applying homogeneous transformation: w = (λr)^p · z = λ^p · r^p · z

This works because **radial scaling in ambient space is well-defined**, even if λx is off-manifold. The projection π(λx) = u is the same direction.

### Decoder: Adaptive Asymptotic Homogeneity

The decoder has three sources of information:
1. **θ alone** → c_base(θ): Homogeneous baseline
2. **(θ, r_w)** → c_correction(θ, r_w): Non-homogeneous when needed
3. **θ alone** → b(θ): Bounded angle-dependent scaling

**Key insight**: Sparsity penalty on c_correction creates **adaptive homogeneity**:

- **Near origin** (complex manifold structure):
  - c_correction can be non-zero
  - Allows flexibility to handle non-homogeneous regions
  - Sparsity penalty is dominated by reconstruction loss

- **In the tail** (cone-like asymptotic structure):
  - Manifold becomes asymptotically homogeneous (due to regular variation)
  - c_correction provides diminishing benefit
  - Sparsity penalty drives c_correction → 0
  - Decoder becomes effectively homogeneous: c ≈ c_base(θ)

This matches the actual geometry: manifolds supported by regularly varying data are asymptotically (but not necessarily locally) homogeneous!

## Loss Function

```python
L_total = L_reconstruction + λ_angular · L_angular + λ_radial · L_radial + λ_sparse · L_sparse
```

### 1. Full Reconstruction Loss
```
L_reconstruction = ||x̂ - x||²
```

### 2. Angular Reconstruction Loss
Geodesic distance on S² between true and decoded directions:
```
L_angular = arccos²(u · c)
```

### 3. Radial Reconstruction Loss
Error in reconstructed radius (accounting for scaling):
```
r_total = r_reconstructed · b
L_radial = (r_total - r)²
```

### 4. Sparsity Loss (NEW)
L1 penalty on the correction term:
```
L_sparse = mean(|c_correction|)
```

**Hyperparameters**:
- λ_angular = 1.0
- λ_radial = 1.0
- λ_sparse = 0.01 (encourages c_correction to be small)

## Tail Preservation Analysis

### Encoder: Exact Tail Preservation

Since the encoder is p-homogeneous:
```
μ_W = encode_* μ_X
```
The tail measure is **exactly preserved** via the pushforward.

### Decoder: Asymptotic Tail Preservation

For generated samples W_new ~ tail distribution in latent space:

**Radial behavior**:
```
||X̂|| = r_w^(1/p) · b(θ) · ||c||
      = r_w^(1/p) · b(θ)    (since c is normalized to S²)
```

If b is bounded: b_min ≤ b(θ) ≤ b_max, then:
```
b_min · r_w^(1/p) ≤ ||X̂|| ≤ b_max · r_w^(1/p)
```

**Asymptotic behavior**: ||X̂|| ~ r_w^(1/p) ~ r  as r_w → ∞

**Tail index preservation**:
```
P(||X̂|| > t) ~ P(r_w^(1/p) > t/b) ~ P(r_w > (t/b)^p) ~ t^{-α}
```

The tail index α is preserved! ✓

**Angular behavior**:
```
X̂ / ||X̂|| ≈ c(θ, r_w)
```

As r_w → ∞:
- Sparsity drives c_correction → 0
- c → c_base(θ) (depends only on latent angle)
- Angular structure becomes homogeneous

## Implementation Details

### Encoder Components

```python
# e: S² → S¹ (normalized angular encoding)
self.encoder_angular = MultiLayerPerceptron(
    input_dim=3,      # u ∈ S²
    output_dim=2,     # ℝ² (will be normalized to S¹)
)

# a: S² → ℝ⁺ (magnitude scaling)
self.encoder_magnitude = MultiLayerPerceptron(
    input_dim=3,      # u ∈ S²
    output_dim=1,     # ℝ⁺
)
```

**Forward pass**:
```python
e_raw = self.encoder_angular(u)
e = unit_vector_torch(e_raw)  # Normalize to S¹
a = torch.nn.functional.softplus(self.encoder_magnitude(u))
z = a[:, None] * e
w = r_p[:, None] * z
```

### Decoder Components

```python
# c_base: S¹ → S² (homogeneous baseline)
self.decoder_angular_base = MultiLayerPerceptron(
    input_dim=2,      # θ (1D circle as 2D vector)
    output_dim=3,     # ℝ³ (will be normalized to S²)
)

# c_correction: S¹ × ℝ → ℝ³ (sparse non-homogeneous correction)
self.decoder_angular_correction = MultiLayerPerceptron(
    input_dim=3,      # [θ (2D), log(r_w) (1D)]
    output_dim=3,     # ℝ³
)

# b: S¹ → ℝ⁺ (angle-dependent radius scaling)
self.decoder_magnitude = MultiLayerPerceptron(
    input_dim=2,      # θ (1D circle as 2D vector)
    output_dim=1,     # ℝ⁺
)
```

**Forward pass**:
```python
r_w = torch.linalg.norm(w, dim=1)
r_reconstructed = r_w ** (1.0 / self.p_homogeneity)
theta = unit_vector_torch(w)

# Homogeneous baseline
c_base_raw = self.decoder_angular_base(theta)
c_base = unit_vector_torch(c_base_raw)

# Sparse correction
log_r_w = torch.log(r_w + 1e-8)
c_corr_input = torch.cat([theta, log_r_w[:, None]], dim=1)
c_correction_raw = self.decoder_angular_correction(c_corr_input)

# Combine
c = unit_vector_torch(c_base + c_correction_raw)
b = torch.nn.functional.softplus(self.decoder_magnitude(theta))

h = b[:, None] * c
x_hat = self.center + r_reconstructed[:, None] * h
```

## Key Theoretical Properties

### Property 1: Encoder Tail Preservation
If X is regularly varying with tail measure μ_X and the encoder is p-homogeneous, then W = encode(X) is regularly varying with tail measure μ_W = encode_* μ_X.

### Property 2: Decoder Boundedness
Since b(θ): S¹ → ℝ⁺ is continuous and S¹ is compact, b is bounded: ∃ b_min, b_max such that b_min ≤ b(θ) ≤ b_max.

### Property 3: Asymptotic Homogeneity via Sparsity
The sparsity penalty λ_sparse · ||c_correction||_1 encourages:
```
c_correction → 0  when reconstruction can be achieved with c_base alone
```

In the tail where the manifold is asymptotically homogeneous (due to regular variation), c_base(θ) alone should suffice, so c_correction → 0.

### Property 4: Tail Index Preservation in Generation
For W_new sampled from the tail distribution in latent space:
```
P(||decode(W_new)|| > t) ~ t^{-α}
```
where α is the tail index of the original data.

## Advantages of This Approach

1. **Guaranteed tail preservation in encoding**: Encoder is exactly p-homogeneous
2. **Full manifold coverage**: Decoder uses both latent degrees of freedom (r_w, θ)
3. **Adaptive homogeneity**: Automatically learns where non-homogeneity is needed
4. **Asymptotic correctness**: Becomes homogeneous in the tail where it matters for EVT
5. **Theoretical guarantees**: Bounded scaling ensures tail index preservation
6. **Generative modeling**: Can sample from latent tail distributions and decode faithfully
7. **Handles non-homogeneous manifolds**: Works for arbitrary manifolds, not just cones/spheres

## Comparison with Previous Approaches

### Previous: Two-Component Decoder with Radius Conditioning
```
c(θ, r_w): S¹ × ℝ → S²  (single angular decoder seeing radius)
b(θ): S¹ → ℝ⁺
```

**Problem**: No explicit incentive for asymptotic homogeneity. The network could learn arbitrarily complex radius dependence even in the tail.

### Current: Three-Component Decoder with Sparsity
```
c_base(θ): S¹ → S²           (homogeneous baseline)
c_correction(θ, r_w): ℝ³     (sparse correction)
b(θ): S¹ → ℝ⁺
```

**Advantage**:
- Explicit separation between homogeneous and non-homogeneous parts
- Sparsity penalty drives asymptotic homogeneity
- Interpretable: can monitor ||c_correction|| to see where non-homogeneity is used
- Better generalization in the tail (where less data exists)

## Monitoring and Diagnostics

### During Training

Track the sparsity metric:
```
sparse_loss = mean(|c_correction|)
```

**Expected behavior**:
- Initial: ~0.02-0.05 (correction is active)
- Training progresses: Should decrease as network learns efficient baseline
- Interpretation: Lower values → more homogeneous decoder

### Post-Training Analysis

1. **Visualize c_correction magnitude vs r_w**:
   - Plot ||c_correction|| as a function of r_w
   - Should decrease for large r_w (tail region)

2. **Compare reconstructions**:
   - Decode using full c = c_base + c_correction
   - Decode using only c_base (force homogeneity)
   - Difference shows where non-homogeneity is essential

3. **Tail sample generation**:
   - Sample large r_w from Pareto distribution
   - Check that reconstructions look reasonable
   - Verify tail index is preserved empirically

## Future Directions

1. **Adaptive sparsity weighting**:
   - Use λ_sparse · f(r_w) where f(r_w) increases with r_w
   - Stronger penalty in tail, weaker near origin

2. **Architectural constraints on c_correction**:
   - Use c_correction = g(θ) / (1 + log(r_w)) to enforce convergence

3. **Multi-resolution analysis**:
   - Separate manifold into regions by radius
   - Study how c_correction varies across regions

4. **Higher dimensions**:
   - Extend to higher-dimensional manifolds and latent spaces
   - Study how sparsity scales with dimension

5. **Alternative sparsity penalties**:
   - L0 approximations (learned masks)
   - Group sparsity (sparse in certain directions only)
