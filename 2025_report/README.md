# `2025_report/` — NeurIPS 2025 submission

LaTeX source for the paper. Figures are *not* stored here — they are
pulled from `experiments/<exp>/results/` at compile time via
`\graphicspath`.

## Compile

From the repo root:

```bash
make paper
```

Or directly:

```bash
cd 2025_report
pdflatex -halt-on-error neurips_2025.tex
pdflatex -halt-on-error neurips_2025.tex   # second pass resolves refs
```

Two passes are required for the cross-references and the table of
contents. `neurips_2025.pdf` lands next to the source.

## Regenerating figures before compile

The `.tex` resolves figure paths via:

```latex
\graphicspath{{../experiments/}{./}}
```

So every `\includegraphics{exp02_flexible_toy_ablation/results/seeds5/fig_headline}`
reads the PDF living under `experiments/.../results/`. To refresh those
without retraining, run from the repo root:

```bash
make plots     # regenerate every figure from cached pickles
make paper     # recompile
```

`make clean-plots` wipes every `fig_*.{pdf,png}` (but keeps the
pickles), so `make clean-plots && make plots && make paper` is the
canonical "recompile with fresh figures" loop.

See the root `README.md` for the full reproduction workflow.

## Files

- `neurips_2025.tex` — the paper.
- `neurips_2025.sty` — NeurIPS 2025 style file.
- `references.bib` — BibTeX.
- `paper_agenda.md` — working notes on outstanding edits.
- `style_guide/` — writing conventions for this submission.
- `old.tex` — previous draft kept for reference; not compiled.
