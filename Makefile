# Reproduction orchestrator for the p-homogeneous encoder paper.
#
# Entry points:
#
#   make reproduce        train + evaluate + plot every experiment, then compile paper
#   make plots            regenerate every figure from cached pickles (no training)
#   make paper            pdflatex the NeurIPS submission twice
#   make exp02            run just exp02 (cache-aware: only retrains missing seeds)
#   make clean-plots      remove fig_*.{pdf,png}; keep cached pickles
#
# Cache-aware means: if a per-seed, per-model pickle already exists under
# a run's results/ tree, it is reused. To force a retrain for one
# experiment, pass ARGS=--force-retrain:
#
#   make exp02 ARGS=--force-retrain
#
# Climate runs download ERA5 outside the Makefile; see
# experiments/climate/README.md for data acquisition.

SHELL := /bin/bash
PY    := uv run python
E     := experiments
REP   := 2025_report

ARGS ?=

.PHONY: reproduce plots paper clean-plots \
        exp01 exp02 exp03 exp04 exp05 exp06 exp07 exp08 exp09 climate \
        exp01-plots exp02-plots exp03-plots exp04-plots exp05-plots \
        exp06-plots exp07-plots exp08-plots exp09-plots climate-plots

reproduce: exp01 exp02 exp03 exp04 exp05 exp06 exp07 exp08 exp09 climate paper

# -----------------------------------------------------------------------------
# Individual experiments (cache-aware)
# -----------------------------------------------------------------------------

exp01:
	cd $(E)/exp01_curved_surface         && $(PY) run.py $(ARGS)

exp02:
	cd $(E)/exp02_flexible_toy_ablation  && $(PY) run.py --output-subdir seeds5 --num-seeds 5 $(ARGS)

exp03:
	cd $(E)/exp03_ambient_dim_sweep      && $(PY) run.py $(ARGS)

exp04:
	cd $(E)/exp04_intrinsic_dim_sweep    && $(PY) run.py $(ARGS)

exp05:
	cd $(E)/exp05_tail_index_sweep       && $(PY) run.py $(ARGS)

exp06:
	cd $(E)/exp06_homogeneity_degree_sweep && $(PY) run.py $(ARGS)

exp07:
	cd $(E)/exp07_network_size_sweep     && $(PY) run.py $(ARGS)

exp08:
	cd $(E)/exp08_curved_surface_pareto_margins         && $(PY) run.py $(ARGS)

exp09:
	cd $(E)/exp09_curved_surface_pareto_margins_two_sided && $(PY) run.py $(ARGS)

climate:
	cd $(E)/climate                      && bash run_all.sh

# -----------------------------------------------------------------------------
# Plot-only fast path: regenerate figures from cached pickles, no training
# -----------------------------------------------------------------------------

plots: exp01-plots exp02-plots exp03-plots exp04-plots exp05-plots exp06-plots exp07-plots exp08-plots exp09-plots climate-plots

exp01-plots:
	cd $(E)/exp01_curved_surface         && $(PY) run.py --plot-only

exp02-plots:
	cd $(E)/exp02_flexible_toy_ablation  && $(PY) run.py --output-subdir seeds5 --num-seeds 5 --plot-only

exp03-plots:
	cd $(E)/exp03_ambient_dim_sweep      && $(PY) run.py --plot-only

exp04-plots:
	cd $(E)/exp04_intrinsic_dim_sweep    && $(PY) run.py --plot-only

exp05-plots:
	cd $(E)/exp05_tail_index_sweep       && $(PY) run.py --plot-only

exp06-plots:
	cd $(E)/exp06_homogeneity_degree_sweep && $(PY) run.py --plot-only

exp07-plots:
	cd $(E)/exp07_network_size_sweep     && $(PY) run.py --plot-only

exp08-plots:
	cd $(E)/exp08_curved_surface_pareto_margins         && $(PY) run.py --plot-only

exp09-plots:
	cd $(E)/exp09_curved_surface_pareto_margins_two_sided && $(PY) run.py --plot-only

# Climate covers both raw-margin runs (u10, tp, t2m, tensor) and the
# Pareto-margined / two-sided variants when caches exist. Each entry is
# (results_subdir, --var, tensor_path); the loop only runs when a
# seed0/ directory is present so we don't crash on missing pickles.
CLIMATE_VARIANTS := \
	u10:u10:data/era5_u10.pt \
	tp:tp:data/era5_tp.pt \
	t2m:t2m:data/era5_t2m.pt \
	tensor:tensor:data/era5_tensor.pt \
	u10_pareto:u10:data/era5_u10.pt \
	tp_pareto:tp:data/era5_tp.pt \
	tp_pareto_native:tp:data/era5_tp_native.pt

climate-plots:
	@for entry in $(CLIMATE_VARIANTS); do \
		subdir=$${entry%%:*}; rest=$${entry#*:}; \
		var=$${rest%%:*}; tensor=$${rest#*:}; \
		if [ -d "$(E)/climate/results/$$subdir/seed0" ]; then \
			echo ">>> climate-plots $$subdir (var=$$var)"; \
			( cd $(E)/climate && $(PY) run.py --var $$var --tensor $$tensor --results-subdir $$subdir --plot-only ) || exit 1; \
		fi; \
	done

# -----------------------------------------------------------------------------
# Paper compile
# -----------------------------------------------------------------------------

paper:
	cd $(REP) && pdflatex -halt-on-error neurips_2025.tex && pdflatex -halt-on-error neurips_2025.tex

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

clean-plots:
	find $(E) -type f \( -name 'fig_*.pdf' -o -name 'fig_*.png' \) -delete
	find $(E) -type f \( -name 'spatial_recon_error.*' -o -name 'sample_fields.*' \
		-o -name 'return_level.*' -o -name 'pca_scree.*' \
		-o -name 'marginal_distributions.*' -o -name 'tail_qq.*' \) -delete

# The cache is never touched by any `make` target other than --force-retrain.
# To nuke pickles yourself: find experiments -name '*.pkl' -delete
