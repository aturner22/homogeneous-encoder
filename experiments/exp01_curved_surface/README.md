# exp01 — curved surface in R^3

## Hypothesis

Headline sanity check. A HAE trained on a 2-D curved surface embedded
in R^3 should reconstruct the manifold while producing a latent code
whose radial distribution is Proposition-1 consistent. A size-matched
StandardAE reconstructs the same surface but breaks the radial law.

## Config

`lib.config.CurvedSurfaceConfig` (D=3, m=2, p=2, Student-t radii).
Single seed; no sweep.

## Run

From the repo root:

```bash
make exp01          # train + plot (cache-aware)
make exp01-plots    # replot from cached pickles
```

## Outputs

`results/` holds the single-panel PNGs used in the paper's hero
figure: 3-D scatters (`fig_curved_*`, `fig_hero.png`), latent Hill
curves (`fig_latent_hill.png`), encoder homogeneity scan
(`fig_homogeneity_scan.png`), and the HAE correction diagnostics
(`fig_hae_correction_*`). Cached per-model pickles live in
`results/seed0/`.
