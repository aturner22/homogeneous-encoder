# NeurIPS 2025 Paper: Complete Structure and TODO List

## Paper Title
**Learning Tail-Preserving Representations for Manifold-Supported Data: Homogeneous Autoencoders with Adaptive Asymptotic Homogeneity**

---

## SECTION 1: INTRODUCTION (1.5 pages)

### TODO List:
- [ ] Write motivation paragraph (3-4 sentences)
  - Heavy-tailed data is ubiquitous (finance, climate, networks, cybersecurity)
  - EVT analysis requires understanding tail behavior
  - High-dimensional data needs dimensionality reduction
  - Standard methods don't preserve tail structure

- [ ] Add concrete motivating example
  - Financial portfolio risk: need to analyze extreme losses in lower dimensions
  - Or: Climate extreme events analysis with many variables

- [ ] Problem statement paragraph
  - Input: X ∈ ℝ^D on manifold M, regularly varying distribution
  - Goal: Learn encoder f: ℝ^D → ℝ^m (m < D) preserving tail measure
  - Requirements: reconstruction + generation + EVT in latent space

- [ ] Challenges paragraph
  - Standard autoencoders break tail structure
  - Homogeneous maps exist but manifolds aren't always homogeneous
  - Need adaptive mechanism: flexible locally, homogeneous in tail

- [ ] Our approach (high-level only)
  - Encoder: exactly p-homogeneous (w = r^p · g(u))
  - Decoder: three components (baseline + sparse correction + scaling)
  - Sparsity drives asymptotic homogeneity

- [ ] Contributions list with forward references
  1. **Architecture**: Three-component decoder (Sec 4)
  2. **Theory**: Tail preservation analysis (Sec 5)
  3. **Experiments**: Synthetic + real-world validation (Sec 6)
     - Quantitative claim: "X% better tail index preservation"
     - Generation quality claim
  4. **Insights**: When does it work? (Sec 7)
  5. **Code**: github.com/TODO

- [ ] Add figure reference (Figure 1: architecture overview)

---

## SECTION 2: BACKGROUND AND RELATED WORK (1 page)

### TODO List:
- [ ] **Extreme Value Theory in ML** (1 paragraph)
  - Existing EVT applications in ML
  - Tail modeling approaches
  - Gap: no dimensionality reduction preserving tails
  - Citations: [TODO: add 3-5 papers]

- [ ] **Manifold Learning and Autoencoders** (1 paragraph)
  - Autoencoders, VAEs, β-VAEs
  - Geometric deep learning, hyperbolic/spherical embeddings
  - None preserve heavy-tailed structure
  - Citations: [TODO: add 5-8 papers]

- [ ] **Homogeneous Functions and Scaling** (1 paragraph)
  - Role in EVT (regular variation)
  - Scale-invariant representations
  - Our contribution: learning homogeneous maps
  - Citations: [TODO: add 2-3 EVT theory papers]

- [ ] **Radial/Angular Decompositions** (1 paragraph)
  - Radial basis functions
  - Star-shaped parametrizations
  - Our contribution: adaptive asymptotic behavior
  - Citations: [TODO: add 2-3 papers]

---

## SECTION 3: PROBLEM SETUP AND PRELIMINARIES (0.5 pages)

### TODO List:
- [ ] **Data Model**
  - X ∈ M ⊂ ℝ^D (manifold M has dimension m)
  - Regular variation: tP(X/t ∈ A) → μ(A) as t → ∞
  - Tail measure μ encodes extreme behavior

- [ ] **Goal**
  - Learn encoder f: ℝ^D → ℝ^m
  - Decoder g: ℝ^m → ℝ^D
  - Requirements:
    1. Tail preservation: μ_W = f_* μ_X
    2. Reconstruction: ||X - g(f(X))|| small
    3. Generation: sample from tail in latent, decode faithfully

- [ ] **Key Constraint**
  - Encoder must be p-homogeneous for tail preservation
  - f(λx) = λ^p f(x) ⟹ pushforward preserves regular variation

- [ ] Add notation table if needed

---

## SECTION 4: METHOD - HOMOGENEOUS AUTOENCODERS (2 pages)

### TODO List:

### 4.1: Encoder Architecture (0.5 pages)
- [ ] **Decomposition**
  - x = center + r · u where r = ||x - center||, u ∈ S^(D-1)
  - Product form: w = r^p · a(u) · e(u)
  - e(u): S^(D-1) → S^(m-1) (angular encoding, normalized)
  - a(u): S^(D-1) → ℝ^+ (magnitude scaling)

- [ ] **Implementation**
  - Two MLPs: encoder_angular and encoder_magnitude
  - Apply softplus to ensure a > 0
  - Normalize e to unit sphere

- [ ] **Homogeneity**
  - Proof sketch: encode(λx) = (λr)^p · a(u) · e(u) = λ^p · encode(x)
  - Therefore tail measure is preserved: μ_W = encode_* μ_X

- [ ] Add pseudocode or architecture diagram component

### 4.2: Decoder Challenge (0.5 pages)
- [ ] **The Problem**
  - Latent w ∈ ℝ^m has 2 degrees of freedom: r_w = ||w||, θ = w/||w|| ∈ S^(m-1)
  - Manifold M has dimension m (same!)
  - Previous attempt: c(θ): S^(m-1) → S^(D-1) collapses to (m-1)-D curve

- [ ] **Why Homogeneity is Hard**
  - Want: decode(λ^p w) = λ · decode(w) for generation
  - But: manifolds aren't necessarily homogeneous everywhere
  - Solution needed: adaptive approach

- [ ] Add figure showing collapsed decoder output vs desired

### 4.3: Three-Component Decoder (1 page)
- [ ] **Architecture Overview**
  1. c_base(θ): S^(m-1) → S^(D-1) (homogeneous baseline)
  2. c_correction(θ, log r_w): S^(m-1) × ℝ → ℝ^D (sparse correction)
  3. b(θ): S^(m-1) → ℝ^+ (angle-dependent scaling)

- [ ] **Forward Pass**
  ```
  r_w = ||w||
  θ = w / ||w||
  r_reconstructed = r_w^(1/p)

  c_base = normalize(MLP_base(θ))
  c_correction = MLP_correction([θ, log(r_w)])
  c = normalize(c_base + c_correction)

  b = softplus(MLP_magnitude(θ))
  h = b · c
  x_hat = center + r_reconstructed · h
  ```

- [ ] **Sparsity Regularization**
  - Loss: L_total = L_recon + λ_angular · L_ang + λ_radial · L_rad + λ_sparse · ||c_correction||_1
  - L1 penalty encourages c_correction → 0 when not needed
  - In tail where manifold is asymptotically homogeneous, correction vanishes

- [ ] **Intuition**
  - Near origin: complex manifold → use correction
  - In tail: cone-like manifold → correction driven to zero by sparsity
  - Adaptive: learns where non-homogeneity is needed

- [ ] Add complete architecture diagram (Figure 2)

---

## SECTION 5: THEORETICAL ANALYSIS (1.5 pages)

### TODO List:

### 5.1: Encoder Tail Preservation (0.3 pages)
- [ ] **Theorem 1** (Encoder preserves tail measure)
  - Statement: If X is regularly varying with measure μ_X and encoder is p-homogeneous, then W is regularly varying with μ_W = encode_* μ_X
  - Proof sketch: Direct from homogeneity property
  - Implication: Can do EVT in latent space

### 5.2: Decoder Tail Index Preservation (0.5 pages)
- [ ] **Proposition 1** (Bounded scaling preserves tail index)
  - If b(θ) is bounded: b_min ≤ b(θ) ≤ b_max
  - Then: ||X_hat|| = r_w^(1/p) · b(θ) ~ r_w^(1/p) ~ r
  - Therefore: P(||X_hat|| > t) ~ t^(-α) (tail index preserved)

- [ ] **Analysis**
  - Radial behavior: controlled by bounded b
  - Angular behavior: c(θ, r_w) determines direction
  - Need: c_correction → 0 as r_w → ∞ for faithful generation

### 5.3: Sparsity Induces Asymptotic Homogeneity (0.4 pages)
- [ ] **Proposition 2** (Sparsity drives asymptotic homogeneity)
  - L1 penalty on c_correction
  - When c_base alone suffices: correction driven to zero
  - In tail (cone-like): c_base sufficient → ||c_correction|| → 0
  - Result: asymptotic homogeneity emerges

- [ ] **Trade-off Analysis**
  - Local flexibility vs asymptotic correctness
  - λ_sparse controls this trade-off
  - Manifold geometry determines optimal λ_sparse

### 5.4: Connection to Geometric Theory (0.3 pages)
- [ ] **Brief mention of ray-fibers** (defer details to appendix)
  - Immersion: decoder must span full manifold dimension
  - Injectivity: avoid collisions in representation
  - Our architecture addresses both via adaptive mechanism

- [ ] Forward reference to appendix for full geometric theory

---

## SECTION 6: EXPERIMENTS (3 pages)

### TODO List:

### 6.1: Controlled Synthetic Experiments (1 page)

#### Experiment 1: Perfect Cone (Homogeneous Manifold)
- [ ] **Setup**
  - Generate cone: {(r·cos(φ), r·sin(φ), r·h) : r ~ Pareto(α), φ ~ Uniform}
  - Train with various λ_sparse
  - Measure: sparsity, reconstruction error, tail index

- [ ] **Expected Result**
  - Low sparsity (c_correction ≈ 0)
  - Excellent tail preservation
  - Show λ_sparse has little effect (manifold already homogeneous)

- [ ] **Figure**: sparsity evolution + reconstruction quality

#### Experiment 2: Non-Homogeneous Surface
- [ ] **Setup**
  - Generate complex surface (current implementation)
  - Train with various λ_sparse
  - Compare with cone results

- [ ] **Expected Result**
  - Higher sparsity (c_correction active)
  - Still preserves tail index
  - λ_sparse affects trade-off

- [ ] **Figure**: compare sparsity vs reconstruction vs tail preservation

#### Experiment 3: Ablation Study
- [ ] **Variants**
  1. Full model (baseline + correction + sparsity)
  2. No correction (baseline only)
  3. No sparsity (correction always active)
  4. Standard autoencoder (no homogeneity)

- [ ] **Metrics**
  - Reconstruction error (MSE)
  - Tail index error (|α_hat - α_true|)
  - Generation quality (visual + quantitative)

- [ ] **Table**: Quantitative comparison

### 6.2: Visualization and Analysis (0.5 pages)
- [ ] **Figure**: c_base vs c_correction contributions
  - Show where correction is used
  - Plot ||c_correction|| vs r_w (should decrease for large r_w)

- [ ] **Figure**: Latent space visualization
  - 2D latent, color by radius
  - Show coverage and structure

- [ ] **Figure**: Generation from tail
  - Sample large r_w from Pareto
  - Decode and visualize
  - Compare with ground truth extremes

### 6.3: Real-World Heavy-Tailed Data (1.5 pages)

#### Dataset Selection
- [ ] **Choose 3 datasets** with documented heavy tails:
  1. Financial returns (S&P 500 or similar)
  2. Network traffic data
  3. Climate extremes OR insurance claims OR earthquake magnitudes

- [ ] **Preprocessing**
  - Embed in 3D if not already (PCA or similar)
  - Verify heavy tails (Hill estimator, QQ plots)
  - Train/val/test split

#### Baselines
- [ ] **Implement comparisons**:
  1. Standard autoencoder
  2. VAE
  3. β-VAE
  4. Our method (with tuned λ_sparse)

#### Metrics
- [ ] **Tail Index Preservation**
  - Estimate α on original data (Hill estimator)
  - Estimate α on latent data
  - Estimate α on reconstructed data
  - Report: |α_latent - α_original| and |α_recon - α_original|

- [ ] **Extreme Quantile Prediction**
  - Predict 99%, 99.9%, 99.99% quantiles
  - Compare predictions with held-out extreme values
  - Metric: relative error

- [ ] **Generation Quality**
  - Sample from tail in latent space
  - Decode to ambient space
  - Compare with real extreme samples (visual + quantitative)

- [ ] **Reconstruction Quality**
  - MSE on test set
  - Show our method maintains competitive reconstruction despite constraints

#### Results Format
- [ ] **Table 1**: Tail index preservation (3 datasets × 4 methods)
- [ ] **Table 2**: Extreme quantile errors
- [ ] **Figure**: Generated extremes vs real extremes
- [ ] **Figure**: Tail preservation (QQ plots)

---

## SECTION 7: DISCUSSION (0.5 pages)

### TODO List:
- [ ] **When Does It Work?**
  - Depends on manifold geometry
  - Works well when: manifold is asymptotically cone-like
  - Struggles when: manifold has persistent curvature at all scales

- [ ] **Hyperparameter Selection**
  - λ_sparse: tune based on manifold complexity
  - Could be adaptive (future work)
  - Rule of thumb: start with 0.01, increase if overfitting tail

- [ ] **Limitations**
  - Requires choosing center c (currently fixed)
  - Assumes manifold structure known/learned
  - 2D latent space (extension to higher dimensions needed)

- [ ] **Broader Impact**
  - Enables EVT in latent space
  - Applications: risk assessment, anomaly detection, climate modeling
  - Potential misuse: could be used to generate adversarial extremes (acknowledge this)

---

## SECTION 8: CONCLUSION (0.25 pages)

### TODO List:
- [ ] **Summary** (2-3 sentences)
  - Introduced homogeneous autoencoders with adaptive asymptotic homogeneity
  - Provably preserve tail structure while handling non-homogeneous manifolds
  - Validated on synthetic and real-world data

- [ ] **Impact** (1-2 sentences)
  - Enables extreme value analysis in latent space
  - Opens new possibilities for high-dimensional EVT

- [ ] **Future Work** (2-3 bullets)
  - Adaptive λ_sparse based on local manifold geometry
  - Extension to higher-dimensional latent spaces
  - Applications to specific domains (finance, climate, etc.)

---

## APPENDIX SECTIONS

### TODO List:

### A: Complete Geometric Theory
- [ ] Ray-fiber analysis
- [ ] Immersion criteria
- [ ] Injectivity conditions
- [ ] Proofs from main paper

### B: Implementation Details
- [ ] Network architectures (layer sizes, activations)
- [ ] Training procedure (optimizer, learning rate schedule)
- [ ] Hyperparameter ranges tested
- [ ] Computational requirements

### C: Additional Experiments
- [ ] More ablations
- [ ] Sensitivity analyses
- [ ] Additional datasets
- [ ] Failure cases

### D: Extreme Value Theory Background
- [ ] Regular variation definition
- [ ] Tail measures
- [ ] Hill estimator details
- [ ] Connection to homogeneity

---

## FIGURES NEEDED (≈6-8 figures)

### Figure 1: Architecture Diagram
- [ ] Show full encoder-decoder architecture
- [ ] Highlight three decoder components
- [ ] Show data flow with dimensions

### Figure 2: Decoder Components
- [ ] Visualize c_base, c_correction, b
- [ ] Show how they combine
- [ ] Illustrate sparsity effect

### Figure 3: Synthetic Results (Cone vs Surface)
- [ ] 2×2 grid: {cone, surface} × {sparsity, reconstruction}
- [ ] Show evolution during training

### Figure 4: Ablation Study
- [ ] Bar chart comparing 4 variants
- [ ] Metrics: reconstruction, tail index, generation

### Figure 5: Real-World Results - Tail Preservation
- [ ] QQ plots for 3 datasets
- [ ] Our method vs baselines
- [ ] Show tail index estimates

### Figure 6: Real-World Results - Generation
- [ ] Generated extremes vs real extremes
- [ ] Scatter plots or histograms
- [ ] 3 datasets

### Figure 7: Sparsity Analysis
- [ ] ||c_correction|| vs r_w
- [ ] Show asymptotic decrease
- [ ] Compare different λ_sparse values

### Figure 8 (optional): Failure Cases
- [ ] When does it not work well?
- [ ] Manifold examples where method struggles

---

## TABLES NEEDED (≈3-4 tables)

### Table 1: Synthetic Experiments
- [ ] Columns: Method, Reconstruction MSE, Tail Index Error, Sparsity
- [ ] Rows: Cone/Surface × various λ_sparse

### Table 2: Real-World Tail Preservation
- [ ] Columns: Dataset, Method, α_original, α_latent, α_recon, |Δα|
- [ ] Rows: 3 datasets × 4 methods

### Table 3: Extreme Quantile Prediction
- [ ] Columns: Dataset, Method, 99% error, 99.9% error, 99.99% error
- [ ] Rows: 3 datasets × 4 methods

### Table 4 (optional): Computational Cost
- [ ] Training time, inference time, memory
- [ ] Compare with baselines

---

## REFERENCES TO ADD

### EVT and Statistics
- [ ] Regular variation textbook (Resnick, de Haan & Ferreira)
- [ ] Hill estimator papers
- [ ] EVT in machine learning (if exists)

### Manifold Learning
- [ ] Autoencoder fundamentals
- [ ] VAE, β-VAE papers
- [ ] Geometric deep learning review

### Homogeneous Functions
- [ ] Euler's theorem
- [ ] Applications in economics/physics
- [ ] Scale-invariant representations

### Extreme Value Applications
- [ ] Finance (risk management)
- [ ] Climate (extreme events)
- [ ] Networks (anomaly detection)

---

## NEURIPS CHECKLIST

- [ ] Claims: abstract/intro match contributions? ✓
- [ ] Limitations: discussed in Section 7? ✓
- [ ] Theory: proofs in appendix? ✓
- [ ] Reproducibility: implementation details in appendix? ✓
- [ ] Code: will be released (add link before submission)? ✓
- [ ] Data: real-world datasets described? ✓
- [ ] Compute: mention in appendix? ✓
- [ ] Ethics: broader impact in discussion? ✓

---

## WRITING TIMELINE

### Phase 1: Structure (DONE)
- [x] This document

### Phase 2: Core Content (Week 1-2)
- [ ] Write all method section (Sec 4)
- [ ] Write theory section (Sec 5)
- [ ] Draft introduction and related work (Sec 1-2)

### Phase 3: Experiments (Week 3-4)
- [ ] Run all synthetic experiments
- [ ] Run real-world experiments
- [ ] Create all figures and tables

### Phase 4: Polish (Week 5)
- [ ] Fill in results in intro/abstract
- [ ] Write discussion and conclusion
- [ ] Complete appendices
- [ ] Internal review

### Phase 5: Final (Week 6)
- [ ] Address review feedback
- [ ] Final formatting
- [ ] Submit!
