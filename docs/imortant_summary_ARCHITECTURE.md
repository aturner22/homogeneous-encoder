# P-Homogeneous Autoencoder for Manifold-Supported Regularly Varying Data

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
   - Compresses the 2D manifold to 2D latent space
   - Preserves the tail measure via pushforward: μ_W = f_* μ_X
   - Allows faithful reconstruction and generation

### Why Homogeneity?

**Key result from extreme value theory**: If f is p-homogeneous (f(λx) = λ^p f(x)), then the pushforward of a regularly varying measure is also regularly varying with transformed tail index.

This allows us to:
- Perform EVT analysis in the lower-dimensional latent space
- Generate new samples from tail distributions
- Preserve tail index and (asymptotically) angular measure

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
- Angle: θ = w/||w|| ∈ S¹

This gives us **2 degrees of freedom** to encode the 2D manifold.

## Architecture: Decomposed Encoder-Decoder

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

**Outputs**:
- w: 2D latent vector
- r: original radius
- z: angular encoding z = a(u) · e(u)
- e: normalized angular component on S¹
- a: magnitude scaling factor
- u: original direction on S²

### Decoder (Asymptotically Homogeneous)

The decoder maps w ∈ ℝ² back to x̂ ∈ ℝ³:

**Step 1**: Extract latent components
```
r_w = ||w||
θ = w/||w||  ∈ S¹
r_reconstructed = r_w^(1/p)  (invert the p-power)
```

**Step 2**: Angular decoding with radius conditioning (KEY CHANGE)
```
c(θ, r_w): S¹ × ℝ⁺ → ℝ³    (MLP + normalization to S²)
b(θ, r_w): S¹ × ℝ⁺ → ℝ⁺    (MLP + softplus)
h = b(θ, r_w) · c(θ, r_w)   (scalar × unit vector)
```

**Step 3**: Scale and shift
```
x̂ = center + r_reconstructed · h
  = center + r_w^(1/p) · b(θ, r_w) · c(θ, r_w)
```

**Key properties**:
- **NOT locally homogeneous**: decode(λ^p w) ≠ λ decode(w) in general
- **Asymptotically homogeneous**: As r_w → ∞, behavior becomes homogeneous
- **Full manifold coverage**: c(θ, r_w) can span the full 2D surface (not just a 1D curve)
- **Bounded scaling**: b(θ, r_w) should be bounded to preserve tail index

**Outputs**:
- x̂: reconstructed ambient point
- r_w: latent radius ||w||
- r_reconstructed: r_w^(1/p)
- θ: angular direction in latent space
- c: decoded direction on S²
- b: decoded magnitude scaling
- h: combined output b · c

## Why This Asymmetry Works

### Encoder Can Be Locally Homogeneous

The encoder treats any point λx by:
1. Extracting direction u (same for λx as for x)
2. Scaling radius: r → λr
3. Applying homogeneous transformation: w = (λr)^p · z = λ^p · r^p · z

This works because **radial scaling in ambient space is well-defined**, even if λx is off-manifold. The projection π(λx) = u is the same direction.

### Decoder Should Only Be Asymptotically Homogeneous

The decoder reconstructs points on the manifold, which:
- **Near origin**: May have complex, non-homogeneous structure
- **At infinity**: Becomes asymptotically homogeneous (due to regular variation of data)

By allowing c(θ, r_w) and b(θ, r_w) to depend on r_w:
- We can adapt to non-homogeneous manifold structure at finite scales
- We preserve asymptotic homogeneity for tail preservation
- We gain full 2D coverage of the manifold

## Tail Preservation Analysis

### Encoder: Exact Tail Preservation

Since the encoder is p-homogeneous:
```
μ_W = encode_* μ_X
```
The tail measure is **exactly preserved** via the pushforward.

### Decoder: Asymptotic Tail Preservation

For generated samples W_new ~ tail distribution in latent space:

**With bounded b**:
```
||X̂|| = r_w^(1/p) · b(θ, r_w) · ||c(θ, r_w)||
      = r_w^(1/p) · b(θ, r_w)    (since c is normalized)
```

If b is bounded: b_min ≤ b(θ, r_w) ≤ b_max, then:
```
b_min · r_w^(1/p) ≤ ||X̂|| ≤ b_max · r_w^(1/p)
```

**Asymptotic behavior**:
```
||X̂|| ~ r_w^(1/p) ~ r  as r_w → ∞
```

**Tail index preservation**:
```
P(||X̂|| > t) ~ P(r_w^(1/p) > t/b) ~ P(r_w > (t/b)^p) ~ t^{-α}
```

The tail index α is preserved! ✓

### Round-Trip Analysis

For x → w → x̂:
```
||x̂|| = r_w^(1/p) · b(θ, r_w)
      = (r^p · a(u))^(1/p) · b(e(u), r_w)
      = r · a(u)^(1/p) · b(e(u), r_w)
```

With both a and b bounded:
```
||x̂|| / ||x|| = a(u)^(1/p) · b(e(u), r_w) ∈ [C_min, C_max]
```

Asymptotically: ||x̂|| ~ ||x||, preserving regular variation ✓

## Implementation Details

### Encoder Components

```python
# e: S² → S¹ (normalized angular encoding)
self.encoder_angular = MultiLayerPerceptron(
    input_dim=3,      # u ∈ S²
    output_dim=2,     # ℝ² (will be normalized to S¹)
    hidden_width=hidden_width,
    hidden_depth=hidden_depth,
)

# a: S² → ℝ⁺ (magnitude scaling)
self.encoder_magnitude = MultiLayerPerceptron(
    input_dim=3,      # u ∈ S²
    output_dim=1,     # ℝ⁺
    hidden_width=hidden_width,
    hidden_depth=hidden_depth,
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

### Decoder Components (MODIFIED)

```python
# c: S¹ × ℝ⁺ → S² (radius-conditioned angular decoding)
self.decoder_angular = MultiLayerPerceptron(
    input_dim=3,      # [θ (2D), log(r_w) (1D)]
    output_dim=3,     # ℝ³ (will be normalized to S²)
    hidden_width=hidden_width,
    hidden_depth=hidden_depth,
)

# b: S¹ × ℝ⁺ → ℝ⁺ (radius-conditioned magnitude scaling)
self.decoder_magnitude = MultiLayerPerceptron(
    input_dim=3,      # [θ (2D), log(r_w) (1D)]
    output_dim=1,     # ℝ⁺
    hidden_width=hidden_width,
    hidden_depth=hidden_depth,
)
```

**Forward pass**:
```python
r_w = torch.linalg.norm(w, dim=1)
r_reconstructed = r_w ** (1.0 / self.p_homogeneity)
theta = unit_vector_torch(w)

# Condition on log(r_w) for scale-invariance properties
log_r_w = torch.log(r_w + 1e-8)  # Add epsilon for stability
decoder_input = torch.cat([theta, log_r_w[:, None]], dim=1)

c_raw = self.decoder_angular(decoder_input)
c = unit_vector_torch(c_raw)  # Normalize to S²
b = torch.nn.functional.softplus(self.decoder_magnitude(decoder_input))

h = b[:, None] * c
x_hat = self.center + r_reconstructed[:, None] * h
```

### Why log(r_w)?

Using log(r_w) as the radial feature:
1. **Scale-invariant growth**: log(λ^p r_w) = p·log(λ) + log(r_w) grows slowly with λ
2. **Asymptotic convergence**: As r_w → ∞, the influence of log(r_w) becomes relatively smaller
3. **Numerical stability**: Avoids extremely large values
4. **Slow variation**: Encourages the decoder to converge to limiting behavior

## Loss Function

We use three loss components:

### 1. Full Reconstruction Loss
```
L_full = ||x̂ - x||²
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

### Combined Loss
```
L_total = L_full + λ_angular · L_angular + λ_radial · L_radial
```

where λ_angular and λ_radial are hyperparameters (currently set to 1.0).

## Key Theoretical Results

### Theorem 1: Encoder Tail Preservation
If X is regularly varying with tail measure μ_X and the encoder is p-homogeneous, then W = encode(X) is regularly varying with tail measure μ_W = encode_* μ_X.

### Theorem 2: Asymptotic Decoder Homogeneity
If:
- b(θ, r_w) is bounded
- c(θ, r_w) → c_∞(θ) as r_w → ∞
- b(θ, r_w) → b_∞(θ) as r_w → ∞

Then:
```
lim_{λ→∞} ||decode(λ^p w)|| / (λ · ||decode(w)||) = 1
```

### Corollary: Generation Faithfulness
Generated samples W_new from the tail distribution in latent space decode to X̂_new with the same tail index as the original data.

## Advantages of This Approach

1. **Tail preservation**: EVT properties are maintained in latent space
2. **Full manifold coverage**: Decoder can span the entire 2D manifold
3. **Flexibility at finite scales**: Decoder adapts to non-homogeneous structure
4. **Asymptotic correctness**: Homogeneity emerges in the tail where it matters
5. **Theoretical guarantees**: Bounded scaling ensures tail index preservation
6. **Generative modeling**: Can sample from latent tail distributions and decode faithfully

## Open Questions and Future Work

1. **Optimal r_w features**: Is log(r_w) the best choice, or should we use other scale-invariant features?
2. **Explicit boundedness**: Should we explicitly constrain b to a bounded range, or rely on implicit boundedness?
3. **Angular measure preservation**: How well is the angular measure preserved under this asymptotically homogeneous decoding?
4. **Multiple manifolds**: Can this extend to data supported on unions of manifolds?
5. **Higher dimensions**: Does this scale to higher-dimensional manifolds and latent spaces?
