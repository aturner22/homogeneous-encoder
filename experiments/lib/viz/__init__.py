"""Publication plots, split into themed submodules.

2-D figures use plotnine (ggplot2 grammar). 3-D scatters and a few
composite diagnostics stay in matplotlib because plotnine has no 3-D
support. Importing this package applies a sans-serif rcParams patch
globally — see ``_base._apply_mpl_aesthetic``.
"""

from __future__ import annotations

from ._base import (
    FIGSIZE_DOUBLE,
    FIGSIZE_HERO,
    FIGSIZE_SINGLE,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    THEME_PUBLICATION,
)
from .manifold_plots import (
    save_architecture_thumbnails,
    save_curved_surface_scatter,
    save_hero_curved_surface,
    save_latent_scatter_by_radius,
    save_overlay_reconstruction,
    save_overlay_reconstruction_panels,
)
from .marginal_diagnostics import (
    save_marginal_pareto_hill,
    save_marginal_pareto_histograms,
)
from .panel_sets import (
    plot_training_history,
    save_diagnostic_panel_set,
)
from .sweep_plots import (
    plot_sweep_grid,
    save_sweep_metric,
)
from .tail_plots import (
    save_binned_recon_error,
    save_correction_magnitude_scatter,
    save_extrapolation_curve,
    save_latent_hill_curves,
    save_latent_vs_ambient_radius,
)

__all__ = [
    "FIGSIZE_DOUBLE",
    "FIGSIZE_HERO",
    "FIGSIZE_SINGLE",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "MODEL_MARKERS",
    "THEME_PUBLICATION",
    "plot_sweep_grid",
    "plot_training_history",
    "save_architecture_thumbnails",
    "save_binned_recon_error",
    "save_correction_magnitude_scatter",
    "save_curved_surface_scatter",
    "save_diagnostic_panel_set",
    "save_extrapolation_curve",
    "save_hero_curved_surface",
    "save_latent_hill_curves",
    "save_latent_scatter_by_radius",
    "save_latent_vs_ambient_radius",
    "save_marginal_pareto_hill",
    "save_marginal_pareto_histograms",
    "save_overlay_reconstruction",
    "save_overlay_reconstruction_panels",
    "save_sweep_metric",
]
