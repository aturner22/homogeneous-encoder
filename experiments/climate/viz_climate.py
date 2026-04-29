"""Climate-specific visualisations for ERA5 experiment.

These plots use lat/lon grid metadata and are not generic enough for
``lib/viz.py``. They are called from ``run.py`` after training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lib.extremes import fit_gpd_pot, return_level_ci, return_level_gpd
from lib.viz import FIGSIZE_DOUBLE, MODEL_COLORS, MODEL_LABELS

try:
    import cartopy.crs as ccrs  # type: ignore
    import cartopy.feature as cfeature  # type: ignore
    _HAS_CARTOPY = True
except ImportError:  # pragma: no cover
    _HAS_CARTOPY = False


def _save_with_pdf(fig, path: Path, *, dpi: int = 300, emit_pdf: bool = True) -> None:
    """Save figure to the given path and, when emit_pdf, a .pdf sibling for vector inclusion.

    Cartopy-heavy figures (coastline vector paths) should pass emit_pdf=False: the
    resulting PDF can be tens of megabytes even when the PNG is small. Those stay
    raster-only and \\includefig falls back to the PNG.
    """
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if emit_pdf and path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _add_map_features(ax) -> None:
    """Add coastlines, borders, and land/ocean fills to a GeoAxes."""
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.2, zorder=0)
    ax.coastlines(resolution="50m", linewidth=0.6, color="black", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", zorder=3)


def _reshape_to_grid(
    flat: np.ndarray,
    n_cells: int,
    n_vars: int,
    var_idx: int,
) -> np.ndarray:
    """Extract one variable's columns and reshape samples to (N, n_lat, n_lon).

    ``flat`` has shape (N, n_cells * n_vars). Variable ``var_idx`` occupies
    columns ``[var_idx * n_cells : (var_idx + 1) * n_cells]``.
    """
    col_start = var_idx * n_cells
    col_end = (var_idx + 1) * n_cells
    return flat[:, col_start:col_end]


def _infer_grid_shape(lats: np.ndarray, lons: np.ndarray, n_cells: int):
    """Return (n_lat, n_lon) from coordinate arrays."""
    n_lat = len(lats)
    n_lon = len(lons)
    if n_lat * n_lon != n_cells:
        # Fall back: assume square-ish
        side = int(np.sqrt(n_cells))
        return side, n_cells // side
    return n_lat, n_lon


def plot_spatial_recon_error(
    x: np.ndarray,
    x_hat: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    var_names: list[str],
    n_cells: int,
    path: Path,
) -> None:
    """Per-gridpoint MSE heatmap for each variable with coastline overlay."""
    n_vars = len(var_names)
    n_lat, n_lon = _infer_grid_shape(lats, lons, n_cells)

    projection = ccrs.PlateCarree() if _HAS_CARTOPY else None
    transform = ccrs.PlateCarree() if _HAS_CARTOPY else None
    subplot_kw = {"projection": projection} if _HAS_CARTOPY else {}

    fig, axes = plt.subplots(
        1, n_vars, figsize=(4 * n_vars, 3.5),
        squeeze=False, subplot_kw=subplot_kw,
    )
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    for vi, vname in enumerate(var_names):
        x_var = _reshape_to_grid(x, n_cells, n_vars, vi)
        xh_var = _reshape_to_grid(x_hat, n_cells, n_vars, vi)
        mse_per_cell = np.mean((x_var - xh_var) ** 2, axis=0)
        grid = mse_per_cell.reshape(n_lat, n_lon)

        ax = axes[0, vi]
        if _HAS_CARTOPY:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            im = ax.pcolormesh(
                lons, lats, grid, shading="auto", cmap="YlOrRd",
                transform=transform, alpha=0.85, zorder=2, rasterized=True,
            )
            _add_map_features(ax)
        else:
            im = ax.pcolormesh(
                lons, lats, grid, shading="auto", cmap="YlOrRd",
                rasterized=True,
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        ax.set_title(f"{vname.upper()} MSE")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save_with_pdf(fig, path, dpi=150, emit_pdf=not _HAS_CARTOPY)


def plot_sample_fields(
    x: np.ndarray,
    x_hat: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    var_names: list[str],
    n_cells: int,
    sample_indices: Sequence[int],
    path: Path,
) -> None:
    """Side-by-side original vs reconstructed fields for selected samples.

    Uses cartopy to overlay coastlines and country borders so the spatial
    fields are visually grounded in geography. Falls back to plain axes
    if cartopy is unavailable.
    """
    n_vars = len(var_names)
    n_lat, n_lon = _infer_grid_shape(lats, lons, n_cells)
    n_samples = len(sample_indices)

    projection = ccrs.PlateCarree() if _HAS_CARTOPY else None
    transform = ccrs.PlateCarree() if _HAS_CARTOPY else None
    subplot_kw = {"projection": projection} if _HAS_CARTOPY else {}

    fig, axes = plt.subplots(
        n_samples, 2 * n_vars,
        figsize=(3.2 * 2 * n_vars, 2.8 * n_samples),
        squeeze=False,
        subplot_kw=subplot_kw,
    )

    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    for si, idx in enumerate(sample_indices):
        for vi, vname in enumerate(var_names):
            x_var = _reshape_to_grid(x, n_cells, n_vars, vi)
            xh_var = _reshape_to_grid(x_hat, n_cells, n_vars, vi)
            orig = x_var[idx].reshape(n_lat, n_lon)
            recon = xh_var[idx].reshape(n_lat, n_lon)

            vmin = min(orig.min(), recon.min())
            vmax = max(orig.max(), recon.max())

            for col_offset, grid, label in (
                (0, orig, "original"),
                (1, recon, "reconstructed"),
            ):
                ax = axes[si, 2 * vi + col_offset]
                if _HAS_CARTOPY:
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    mesh = ax.pcolormesh(
                        lons, lats, grid,
                        shading="auto", vmin=vmin, vmax=vmax, cmap="RdBu_r",
                        transform=transform, alpha=0.85, zorder=2,
                        rasterized=True,
                    )
                    _add_map_features(ax)
                else:
                    mesh = ax.pcolormesh(
                        lons, lats, grid,
                        shading="auto", vmin=vmin, vmax=vmax, cmap="RdBu_r",
                        rasterized=True,
                    )
                if si == 0:
                    ax.set_title(f"{vname.upper()} {label}")
                if col_offset == 0 and not _HAS_CARTOPY:
                    ax.set_ylabel(f"Sample {idx}")
                if col_offset == 1:
                    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            if _HAS_CARTOPY:
                # GeoAxes don't show ylabel; annotate the row instead.
                axes[si, 0].text(
                    -0.15, 0.5, f"Sample {idx}",
                    transform=axes[si, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=10,
                )

    fig.tight_layout()
    _save_with_pdf(fig, path, dpi=150, emit_pdf=not _HAS_CARTOPY)


def plot_marginal_distributions(
    x: np.ndarray,
    x_hat: np.ndarray,
    var_names: list[str],
    n_cells: int,
    path: Path,
    n_bins: int = 100,
) -> None:
    """Per-variable histogram of original vs reconstructed (log y-axis)."""
    n_vars = len(var_names)

    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 3.5), squeeze=False)
    for vi, vname in enumerate(var_names):
        x_var = _reshape_to_grid(x, n_cells, n_vars, vi).ravel()
        xh_var = _reshape_to_grid(x_hat, n_cells, n_vars, vi).ravel()

        lo = min(np.percentile(x_var, 0.5), np.percentile(xh_var, 0.5))
        hi = max(np.percentile(x_var, 99.5), np.percentile(xh_var, 99.5))
        bins = np.linspace(lo, hi, n_bins + 1)

        ax = axes[0, vi]
        ax.hist(x_var, bins=bins, alpha=0.5, density=True, label="Original")
        ax.hist(xh_var, bins=bins, alpha=0.5, density=True, label="Reconstructed")
        ax.set_yscale("log")
        ax.set_title(vname.upper())
        ax.set_xlabel("Value (Standardised)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_with_pdf(fig, path, dpi=150)


def plot_tail_qq(
    x: np.ndarray,
    x_hat: np.ndarray,
    var_names: list[str],
    n_cells: int,
    path: Path,
    tail_fraction: float = 0.1,
) -> None:
    """QQ plot of original vs reconstructed per-variable norms (upper tail)."""
    n_vars = len(var_names)

    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 3.5), squeeze=False)
    for vi, vname in enumerate(var_names):
        x_var = _reshape_to_grid(x, n_cells, n_vars, vi)
        xh_var = _reshape_to_grid(x_hat, n_cells, n_vars, vi)

        # Per-sample norm across grid cells for this variable.
        r_orig = np.linalg.norm(x_var, axis=1)
        r_recon = np.linalg.norm(xh_var, axis=1)

        # Take upper tail.
        threshold = np.percentile(r_orig, 100 * (1 - tail_fraction))
        tail_mask = r_orig >= threshold
        q_orig = np.sort(r_orig[tail_mask])
        q_recon = np.sort(r_recon[tail_mask])

        # Match lengths (in case mask differs).
        n = min(len(q_orig), len(q_recon))
        q_orig = q_orig[:n]
        q_recon = q_recon[:n]

        ax = axes[0, vi]
        ax.scatter(q_orig, q_recon, s=8, alpha=0.5, edgecolors="none")
        lims = [min(q_orig.min(), q_recon.min()), max(q_orig.max(), q_recon.max())]
        ax.plot(lims, lims, "k--", linewidth=0.8, label="y = x")
        ax.set_title(f"{vname.upper()} Tail QQ")
        ax.set_xlabel("Original Quantile")
        ax.set_ylabel("Reconstructed Quantile")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_with_pdf(fig, path, dpi=150)


def plot_return_level_curves(
    series_by_label: Mapping[str, np.ndarray],
    path: Path,
    *,
    n_per_year: float,
    return_periods: np.ndarray | None = None,
    threshold_quantile: float = 0.975,
    n_boot: int = 300,
    title: str | None = None,
) -> None:
    """POT/GPD return-level curves with inline labels and bootstrap bands.

    ``series_by_label`` maps display label → 1-D array of magnitudes
    (e.g. per-sample norm of a variable, with the first entry being
    the empirical reference). Colours are drawn from the Okabe-Ito
    palette: the empirical curve is black.

    ``n_per_year`` is the observation cadence (ERA5 6-hourly = 1460; daily
    = 365). Gets a warning if the series covers fewer than three nominal
    years — the GPD extrapolation to 100-year levels is then speculative.
    """
    if return_periods is None:
        return_periods = np.array([1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0])

    any_series = next(iter(series_by_label.values()), None)
    if any_series is not None and len(any_series) / max(n_per_year, 1.0) < 3.0:
        import warnings
        warnings.warn(
            f"Return-level curve computed from only "
            f"{len(any_series) / n_per_year:.1f} nominal years of data "
            f"(n={len(any_series)}, n_per_year={n_per_year}). "
            f"100-year extrapolation is unreliable below ~3 years of data.",
            stacklevel=2,
        )

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)

    palette = {
        "empirical": "#222222",
        "HomogeneousAE": MODEL_COLORS["HomogeneousAE"],
        "StandardAE": MODEL_COLORS["StandardAE"],
        "HAE": MODEL_COLORS["HomogeneousAE"],
        "AE": MODEL_COLORS["StandardAE"],
    }
    for label, values in series_by_label.items():
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size < 100:
            continue
        try:
            params = fit_gpd_pot(x, threshold_quantile=threshold_quantile)
        except ValueError:
            continue
        rl = return_level_gpd(params, return_periods, n_per_year=n_per_year)
        colour = palette.get(label, "#666666")
        display = MODEL_LABELS.get(label, label)
        ax.plot(
            return_periods, rl,
            color=colour, linewidth=1.4, label=display,
        )
        if n_boot > 0:
            lo, hi = return_level_ci(
                x, return_periods,
                n_per_year=n_per_year,
                threshold_quantile=threshold_quantile,
                n_boot=n_boot,
                ci=0.90,
            )
            ax.fill_between(return_periods, lo, hi, color=colour,
                            alpha=0.15, linewidth=0)
        ax.text(
            return_periods[-1], rl[-1], f"  {display}",
            color=colour, fontsize=8, ha="left", va="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Return Period (Years)")
    ax.set_ylabel("Return Level")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    ax.margins(x=0.15)
    _save_with_pdf(fig, path, dpi=300)
