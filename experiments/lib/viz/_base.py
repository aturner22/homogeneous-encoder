"""Shared aesthetic constants and private helpers for the viz package.

Importing this module applies a NeurIPS-ready sans-serif rcParams patch
to matplotlib — all sibling modules rely on the patch being active, so
the side-effect is deliberate. Public-looking names stay private
(`_color`, `_label`, `_save_gg`, ...) and are re-exported by
`viz/__init__.py` only where needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from plotnine import (
    element_blank,
    element_line,
    element_text,
    ggsave,
    theme,
    theme_bw,
)

# NeurIPS column widths in inches (single-column: 3.25", double: 6.75").
FIGSIZE_SINGLE: tuple[float, float] = (3.25, 2.5)
FIGSIZE_DOUBLE: tuple[float, float] = (6.75, 3.0)
FIGSIZE_HERO: tuple[float, float] = (6.75, 4.5)


def _apply_mpl_aesthetic() -> None:
    """NeurIPS-ready sans-serif style for all matplotlib figures."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.labelpad": 6.0,
        "axes.titlesize": 10,
        "axes.titlepad": 8.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 4.0,
        "ytick.major.pad": 4.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "figure.figsize": FIGSIZE_DOUBLE,
        "figure.dpi": 110,
        "figure.constrained_layout.use": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.10,
    })


_apply_mpl_aesthetic()


# Okabe-Ito colour-universal-design palette (CVD-safe, print-stable).
# HAE = strong blue, StandardAE = strong orange, PCA = neutral grey reference.
MODEL_COLORS: dict[str, str] = {
    "HomogeneousAE": "#0173B2",
    "StandardAE": "#DE8F05",
    "PCA": "#949494",
}
MODEL_LABELS: dict[str, str] = {
    "HomogeneousAE": "HAE",
    "StandardAE": "AE",
    "PCA": "PCA",
}
MODEL_MARKERS: dict[str, str] = {
    "HomogeneousAE": "o",
    "StandardAE": "s",
    "PCA": "^",
}

_DPI = 300

# Viridis truncated at the top so the brightest points read as a soft
# yellow-green rather than fully saturated yellow — keeps tail-vs-bulk
# separation visible without "everything in the tail is identical".
RADIUS_CMAP = LinearSegmentedColormap.from_list(
    "viridis_truncated", plt.cm.viridis(np.linspace(0.05, 0.85, 256))
)

# plotnine theme matching the matplotlib aesthetic. Margins on titles
# and axis labels keep them well clear of the panel and tick text.
THEME_PUBLICATION = (
    theme_bw()
    + theme(
        text=element_text(family="DejaVu Sans", size=9),
        plot_title=element_text(
            size=10, ha="center", margin={"b": 8, "units": "pt"},
        ),
        axis_title_x=element_text(
            size=9, margin={"t": 6, "units": "pt"},
        ),
        axis_title_y=element_text(
            size=9, margin={"r": 6, "units": "pt"},
        ),
        axis_text=element_text(size=8),
        legend_text=element_text(size=8),
        panel_grid_major=element_line(size=0.4, colour="#cccccc"),
        panel_grid_minor=element_blank(),
        legend_background=element_blank(),
        legend_key=element_blank(),
        plot_margin=0.02,
    )
)

_MODEL_ORDER = ["HAE", "AE", "PCA"]
_LABEL_COLORS: dict[str, str] = {
    "HAE": MODEL_COLORS["HomogeneousAE"],
    "AE": MODEL_COLORS["StandardAE"],
    "PCA": MODEL_COLORS["PCA"],
}
_LABEL_SHAPES: dict[str, str] = {"HAE": "o", "AE": "s", "PCA": "^"}


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, "#666666")


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _marker(name: str) -> str:
    return MODEL_MARKERS.get(name, "o")


def _ordered_labels(present) -> list:
    """Model labels present in data, in canonical order."""
    present = set(present)
    return [m for m in _MODEL_ORDER if m in present]


def _col_vals(labels: Sequence[str]) -> dict:
    return {m: _LABEL_COLORS.get(m, "#666666") for m in labels}


def _shp_vals(labels: Sequence[str]) -> dict:
    return {m: _LABEL_SHAPES.get(m, "o") for m in labels}


def _pdf_sibling(path: Path) -> Path:
    return path.with_suffix(".pdf")


def _save_gg(p, path: Path, width: float = 6.0, height: float = 4.0) -> None:
    path = Path(path)
    ggsave(p, filename=str(path), width=width, height=height,
           dpi=_DPI, verbose=False)
    if path.suffix.lower() == ".png":
        ggsave(p, filename=str(_pdf_sibling(path)),
               width=width, height=height, verbose=False)


def _finish(
    fig, output_path: Path, *, tight: bool = True, pad_inches: float = 0.25,
) -> None:
    """Save a matplotlib figure. Writes a .pdf sibling alongside .png for vector inclusion.

    ``pad_inches`` defaults to a slightly generous value (0.25) so 3-D
    axis labels — which can sit just outside the axes bbox — are not
    cropped by ``bbox_inches='tight'``.
    """
    output_path = Path(output_path)
    if tight:
        fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight",
                pad_inches=pad_inches)
    if output_path.suffix.lower() == ".png":
        fig.savefig(_pdf_sibling(output_path), bbox_inches="tight",
                    pad_inches=pad_inches)
    plt.close(fig)
