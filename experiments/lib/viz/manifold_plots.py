"""3-D and latent-space scatters. Matplotlib only — plotnine has no 3D."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.stats import rankdata

from ._base import (
    FIGSIZE_SINGLE,
    RADIUS_CMAP,
    _color,
    _finish,
    _label,
)


def _zoom_lims(
    *arrays: np.ndarray, q: float = 0.99, pad_frac: float = 0.40,
) -> tuple[float, float]:
    """Symmetric percentile-based limits across one or more coordinate arrays.

    ``pad_frac`` extends the limits by that fraction of the inter-quantile
    range on each side, so the axes have visible breathing room around
    the plotted cluster.
    """
    vals = np.concatenate([a.ravel() for a in arrays])
    lo = float(np.quantile(vals, 1.0 - q))
    hi = float(np.quantile(vals, q))
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _style_3d(ax) -> None:
    """Light gridlines, faint panes, and visible tick marks for a 3-D scatter."""
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#cccccc")
        axis.pane.set_alpha(0.4)
        axis._axinfo["grid"]["color"] = "#cccccc"
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.tick_params(axis="both", which="major", labelsize=7, pad=2)
    ax.tick_params(axis="z", which="major", labelsize=7, pad=2)


def _add_radius_colorbar(fig, ax, sc) -> None:
    """Compact horizontal colorbar in the top-right of the subplot.

    Sits just above the X3 annotation; no tick labels, just an
    'Ambient Radius' caption above. Purely a key for
    'darker = smaller radius, brighter = larger'.
    """
    cax = ax.inset_axes([0.82, 0.93, 0.28, 0.022])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("Ambient Radius", fontsize=7, labelpad=3)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.minorticks_off()
    cb.ax.tick_params(which="both", top=False, bottom=False,
                      labeltop=False, labelbottom=False)
    cb.set_ticks([])
    cb.outline.set_visible(False)


def _set_xyz_labels(ax) -> None:
    """X1/X2 via the standard API; X3 as an axes-relative text annotation.

    matplotlib's 3-D ``set_zlabel`` places the label in 3-D space along
    the z-axis tip — with the default camera and tight bbox cropping it
    often lands off-page. Anchoring X3 in axes-relative coordinates
    keeps it visible and well clear of the z-tick numbers.
    """
    ax.set_xlabel("X1", fontsize=8, labelpad=6)
    ax.set_ylabel("X2", fontsize=8, labelpad=6)
    ax.set_zlabel("")
    ax.text2D(
        1.12, 0.58, "X3",
        transform=ax.transAxes, fontsize=8, ha="left", va="center",
    )


def _log_color_range(color_by: np.ndarray) -> tuple[float, float]:
    """Positive (vmin, vmax) for a LogNorm over a colour-mapping array."""
    arr = np.asarray(color_by, dtype=np.float64)
    positive = arr[arr > 0]
    if positive.size == 0:
        return 1.0, 10.0
    vmin = float(positive.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        vmax = vmin * 10
    return vmin, vmax


def save_curved_surface_scatter(
    data: np.ndarray,
    output_path: Path,
    *,
    color_by: np.ndarray | None = None,
    title: str | None = None,
    max_points: int = 5000,
    cmap=None,
    zoom_quantile: float = 0.99,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """3-D scatter of a curved surface, coloured (log-scaled) by ambient radius.

    Pass explicit ``vmin``/``vmax`` to share a colour scale across multiple
    plots (e.g. truth and the per-model reconstructions).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.shape[1] != 3:
        raise ValueError(
            f"save_curved_surface_scatter requires D=3, got {data.shape[1]}")
    if color_by is None:
        color_by = np.linalg.norm(data, axis=1)
    rng = np.random.default_rng(0)
    if len(data) > max_points:
        pick = rng.choice(len(data), size=max_points, replace=False)
        data = data[pick]
        color_by = color_by[pick]

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = _log_color_range(color_by)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap_use = RADIUS_CMAP if cmap is None else cmap

    fig = plt.figure(figsize=(4.8, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        data[:, 0], data[:, 1], data[:, 2],
        c=color_by, cmap=cmap_use, norm=norm,
        s=8, alpha=0.7, linewidths=0, rasterized=True,
    )
    ax.set_xlim(*_zoom_lims(data[:, 0], q=zoom_quantile))
    ax.set_ylim(*_zoom_lims(data[:, 1], q=zoom_quantile))
    ax.set_zlim(*_zoom_lims(data[:, 2], q=zoom_quantile))
    _set_xyz_labels(ax)
    if title:
        ax.set_title(title, fontsize=10, loc="center", pad=16)
    _style_3d(ax)
    _add_radius_colorbar(fig, ax, sc)
    _finish(fig, output_path)


def save_overlay_reconstruction(
    x: np.ndarray,
    x_hat: np.ndarray,
    output_path: Path,
    *,
    title: str | None = None,
    max_points: int = 1500,
    zoom_quantile: float = 0.99,
    color_original: str = "#0B3D6E",
    color_recon: str = "#d62728",
) -> None:
    """3-D scatter overlaying original (blue) and reconstructed (red) points.

    Lets the reader read off pointwise reconstruction quality at a glance:
    matching points sit on top of each other, mismatches separate visibly.
    """
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    if x.shape[1] != 3 or x_hat.shape[1] != 3:
        raise ValueError("save_overlay_reconstruction requires D=3")
    n = min(len(x), len(x_hat))
    x = x[:n]
    x_hat = x_hat[:n]
    rng = np.random.default_rng(0)
    if n > max_points:
        pick = rng.choice(n, size=max_points, replace=False)
        x = x[pick]
        x_hat = x_hat[pick]

    fig = plt.figure(figsize=(4.8, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    # Disable matplotlib's automatic depth-sort so our manual zorder
    # (original behind, reconstruction in front) is respected — otherwise
    # 3-D depth ordering can flip the layers depending on camera angle.
    ax.computed_zorder = False
    # Original: slightly larger, darker, less see-through, drawn first (behind).
    ax.scatter(
        x[:, 0], x[:, 1], x[:, 2],
        c=color_original, s=14, alpha=0.45, linewidths=0,
        label="Original", rasterized=True, depthshade=False, zorder=1,
    )
    # Reconstruction: small dark dots, drawn last (in front).
    ax.scatter(
        x_hat[:, 0], x_hat[:, 1], x_hat[:, 2],
        c=color_recon, s=3, alpha=0.95, linewidths=0,
        label="Reconstruction", rasterized=True, depthshade=False, zorder=2,
    )
    ax.set_xlim(*_zoom_lims(x[:, 0], x_hat[:, 0], q=zoom_quantile))
    ax.set_ylim(*_zoom_lims(x[:, 1], x_hat[:, 1], q=zoom_quantile))
    ax.set_zlim(*_zoom_lims(x[:, 2], x_hat[:, 2], q=zoom_quantile))
    _set_xyz_labels(ax)
    if title:
        ax.set_title(title, fontsize=10, loc="center", pad=16)
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=color_original,
               markersize=5, alpha=0.7, label="Original"),
        Line2D([], [], marker="o", linestyle="", color=color_recon,
               markersize=4, alpha=1.0, label="Reconstruction"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              frameon=False)
    _style_3d(ax)
    _finish(fig, output_path)


def save_overlay_reconstruction_panels(
    x: np.ndarray,
    x_hat_by_model: Mapping[str, np.ndarray],
    output_path: Path,
    *,
    max_points: int = 1500,
    zoom_quantile: float = 0.99,
    color_original: str = "#0B3D6E",
    color_recon: str = "#d62728",
) -> None:
    """Side-by-side 3-D overlays (one panel per model) of original vs reconstruction."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[1] != 3:
        raise ValueError("save_overlay_reconstruction_panels requires D=3")
    items = [(n, np.asarray(v, dtype=np.float64))
             for n, v in x_hat_by_model.items()
             if v is not None and np.asarray(v).shape[1] == 3]
    if not items:
        return

    n = min(len(x), *(len(v) for _, v in items))
    x = x[:n]
    items = [(name, v[:n]) for name, v in items]
    rng = np.random.default_rng(0)
    if n > max_points:
        pick = rng.choice(n, size=max_points, replace=False)
        x = x[pick]
        items = [(name, v[pick]) for name, v in items]

    n_panels = len(items)
    fig = plt.figure(figsize=(4.6 * n_panels, 4.4))
    for idx, (name, x_hat) in enumerate(items):
        ax = fig.add_subplot(1, n_panels, idx + 1, projection="3d")
        ax.computed_zorder = False
        ax.scatter(
            x[:, 0], x[:, 1], x[:, 2],
            c=color_original, s=14, alpha=0.45, linewidths=0,
            label="Original", rasterized=True,
            depthshade=False, zorder=1,
        )
        ax.scatter(
            x_hat[:, 0], x_hat[:, 1], x_hat[:, 2],
            c=color_recon, s=3, alpha=0.95, linewidths=0,
            label="Reconstruction", rasterized=True,
            depthshade=False, zorder=2,
        )
        ax.set_xlim(*_zoom_lims(x[:, 0], x_hat[:, 0], q=zoom_quantile))
        ax.set_ylim(*_zoom_lims(x[:, 1], x_hat[:, 1], q=zoom_quantile))
        ax.set_zlim(*_zoom_lims(x[:, 2], x_hat[:, 2], q=zoom_quantile))
        _set_xyz_labels(ax)
        ax.set_title(_label(name), fontsize=10, loc="center", pad=16)
        if idx == 0:
            legend_handles = [
                Line2D([], [], marker="o", linestyle="", color=color_original,
                       markersize=5, alpha=0.7, label="Original"),
                Line2D([], [], marker="o", linestyle="", color=color_recon,
                       markersize=4, alpha=1.0, label="Reconstruction"),
            ]
            ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
                      frameon=False)
        _style_3d(ax)
    _finish(fig, output_path)


def save_hero_curved_surface(
    x_truth: np.ndarray,
    point: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    *,
    max_points: int = 6000,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Hero figure: 1×2 — heavy-tailed manifold (3-D) + latent vs ambient (log-log)."""
    fig = plt.figure(figsize=(7.8, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.45)

    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    rng = np.random.default_rng(0)
    if len(x_truth) > max_points:
        pick = rng.choice(len(x_truth), size=max_points, replace=False)
        x_plot = x_truth[pick]
    else:
        x_plot = x_truth
    radii = np.linalg.norm(x_plot, axis=1)
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = _log_color_range(radii)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax
    norm = LogNorm(vmin=vmin, vmax=vmax)
    sc_a = ax_a.scatter(
        x_plot[:, 0], x_plot[:, 1], x_plot[:, 2],
        c=radii, cmap=RADIUS_CMAP, norm=norm,
        s=4, alpha=0.85, linewidths=0, rasterized=True,
    )
    ax_a.set_xlim(*_zoom_lims(x_plot[:, 0]))
    ax_a.set_ylim(*_zoom_lims(x_plot[:, 1]))
    ax_a.set_zlim(*_zoom_lims(x_plot[:, 2]))
    _add_radius_colorbar(fig, ax_a, sc_a)
    ax_a.set_title("(a) Heavy-tailed Curved Manifold",
                   fontsize=10, loc="center", pad=16)
    _set_xyz_labels(ax_a)
    _style_3d(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    # Layer order on the latent-vs-ambient panel: AE drawn first, then HAE
    # on top, so the HAE cloud sits in front of the AE cloud.
    radius_data: list[tuple[str, np.ndarray, np.ndarray]] = []
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for name in ("StandardAE", "HomogeneousAE"):
        if name not in point:
            continue
        m = point[name]
        amb = m.get("_ambient_radii")
        lat = m.get("_latent_radii")
        if amb is None or lat is None:
            continue
        amb = np.asarray(amb, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        pos = (amb > 0) & (lat > 0)
        if pos.sum() > 3000:
            sub_idx = rng.choice(np.where(pos)[0], size=3000, replace=False)
        else:
            sub_idx = np.where(pos)[0]
        radius_data.append((name, amb[sub_idx], lat[sub_idx]))
        all_x.append(amb[sub_idx])
        all_y.append(lat[sub_idx])

    for name, x_pts, y_pts in radius_data:
        ax_b.scatter(
            x_pts, y_pts,
            c=_color(name), s=12, alpha=0.5, linewidths=0,
            label=_label(name), rasterized=True,
        )

    if all_x:
        x_concat = np.concatenate(all_x)
        y_concat = np.concatenate(all_y)
        x_lo = float(x_concat.min())
        x_hi = float(x_concat.max())
        y_lo = float(y_concat.min())
        y_hi = float(y_concat.max())

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    if all_x:
        ax_b.set_ylim(y_lo / 1.4, y_hi * 1.2)
        ax_b.set_xlim(x_lo / 1.2, x_hi * 1.2)
    ax_b.set_xlabel("Ambient Radius")
    ax_b.set_ylabel("Latent Radius")
    ax_b.set_title("(b) Latent vs Ambient Radius",
                   fontsize=10, loc="center", pad=16)
    ax_b.grid(False)
    ax_b.legend(loc="upper left", fontsize=8, frameon=False,
                markerscale=1.6, handletextpad=0.4)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.86,
                        bottom=0.17, wspace=0.45)
    _finish(fig, output_path, tight=False)


def save_latent_scatter_by_radius(
    latent_codes: np.ndarray,
    ambient_radii: np.ndarray,
    model_name: str,
    output_path: Path,
    *,
    max_points: int = 3000,
    quantile: float = 0.99,
) -> None:
    """Latent-space scatter coloured by ambient-radius rank.

    When the latent dimension is 2, plots the raw axes ``z[:,0]`` vs
    ``z[:,1]`` so the colour-by-radius reading remains interpretable —
    PCA can rotate/mix latent dimensions in a way that breaks the
    structural meaning of the colour. When the latent dimension is
    >2, falls back to a 2-D SVD projection (no clean alternative).
    Drops points whose plotted coordinates fall outside ``quantile``
    per axis so a handful of extremes don't blow up the plot scale.
    """
    z = np.asarray(latent_codes, dtype=np.float64)
    r = np.asarray(ambient_radii, dtype=np.float64)
    n = min(len(z), len(r))
    z = z[:n]
    r = r[:n]

    latent_dim = z.shape[1] if z.ndim > 1 else 1
    if latent_dim == 2:
        z2 = z[:, :2]
        x_label, y_label = "z₁", "z₂"
    elif latent_dim > 2:
        z_centered = z - z.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(z_centered, full_matrices=False)
        z2 = z_centered @ vt[:2].T
        x_label, y_label = "Latent PC1", "Latent PC2"
    else:
        z2 = np.column_stack([z[:, 0], np.zeros(n)])
        x_label, y_label = "z₁", ""

    lo_q = (1.0 - quantile) / 2.0
    hi_q = 1.0 - lo_q
    x_lo, x_hi = np.quantile(z2[:, 0], [lo_q, hi_q])
    y_lo, y_hi = np.quantile(z2[:, 1], [lo_q, hi_q])
    keep = (
        (z2[:, 0] >= x_lo) & (z2[:, 0] <= x_hi)
        & (z2[:, 1] >= y_lo) & (z2[:, 1] <= y_hi)
    )
    if keep.any():
        z2 = z2[keep]
        r = r[keep]
        n = int(keep.sum())

    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        z2 = z2[idx]
        r = r[idx]
        n = max_points

    rank = rankdata(r) / n

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    sc = ax.scatter(
        z2[:, 0], z2[:, 1],
        c=rank, cmap="viridis", s=4, alpha=0.6, linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(_label(model_name), fontsize=10, loc="center", pad=10)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.9)
    cb.set_label("Ambient Radius Rank", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _finish(fig, output_path)


def _strip_3d_chrome(ax) -> None:
    """Remove ticks, labels, and pane chrome for embeddable thumbnails."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#cccccc")
        axis._axinfo["grid"]["color"] = "#dddddd"
        axis._axinfo["grid"]["linewidth"] = 0.3


def save_architecture_thumbnails(
    x: np.ndarray,
    z: np.ndarray,
    x_hat: np.ndarray,
    output_dir: Path,
    *,
    max_points: int = 2500,
    prefix: str = "arch",
) -> None:
    """Three minimal thumbnails for the paper's architecture diagram.

    Writes ``<prefix>_input.{png,pdf}``, ``<prefix>_latent.{png,pdf}``,
    and ``<prefix>_output.{png,pdf}`` into ``output_dir``. The figures
    have no titles, axis labels, ticks, or colorbars — pure data,
    intended for embedding inside a TikZ figure. Colour mapping is
    shared (log-scaled ambient radius) across all three so the same
    point reads the same colour through encoder and decoder.
    """
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    if x.shape[1] != 3 or x_hat.shape[1] != 3:
        raise ValueError(
            f"save_architecture_thumbnails requires D=3, got {x.shape[1]}"
        )

    n = min(len(x), len(z), len(x_hat))
    x = x[:n]
    z = z[:n]
    x_hat = x_hat[:n]
    rng = np.random.default_rng(0)
    if n > max_points:
        pick = rng.choice(n, size=max_points, replace=False)
        x = x[pick]
        z = z[pick]
        x_hat = x_hat[pick]

    radii = np.linalg.norm(x, axis=1)
    pos = radii[radii > 0]
    if pos.size == 0:
        return
    norm = LogNorm(vmin=float(pos.min()),
                   vmax=float(np.quantile(radii, 0.97)))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Input: 3-D ambient sample.
    fig = plt.figure(figsize=(2.6, 2.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2],
               c=radii, cmap=RADIUS_CMAP, norm=norm,
               s=5, alpha=0.8, linewidths=0, rasterized=True)
    _strip_3d_chrome(ax)
    _finish(fig, out / f"{prefix}_input.png", pad_inches=0.05)

    # Latent: 2-D projection (raw if m=2, otherwise PCA-2).
    if z.shape[1] == 2:
        z2 = z
    else:
        zc = z - z.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(zc, full_matrices=False)
        z2 = zc @ vt[:2].T
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    ax.scatter(z2[:, 0], z2[:, 1],
               c=radii, cmap=RADIUS_CMAP, norm=norm,
               s=4, alpha=0.7, linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.6)
    _finish(fig, out / f"{prefix}_latent.png", pad_inches=0.05)

    # Output: 3-D reconstruction.
    fig = plt.figure(figsize=(2.6, 2.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2],
               c=radii, cmap=RADIUS_CMAP, norm=norm,
               s=5, alpha=0.8, linewidths=0, rasterized=True)
    _strip_3d_chrome(ax)
    _finish(fig, out / f"{prefix}_output.png", pad_inches=0.05)
