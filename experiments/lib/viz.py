"""Simple, single-panel publication plots.

Inspired by ``minimal_p_homogeneous_encoder.py``: no custom rc_context,
no module-level style wrapper, one figure per saved PNG, log axes where
they help, dashed reference lines where they clarify. Drivers import the
``save_*`` primitives directly and save one PNG per separation.

The only composite figures are:
- ``plot_curved_surface_diagnostic`` (exp01, inherently a 2x2 of 3-D views)
- ``plot_training_history`` (diagnostic only, not a paper figure)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _apply_aesthetic() -> None:
    """Quiet serif look. Called once on module import."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


_apply_aesthetic()


MODEL_COLORS: Dict[str, str] = {
    "HomogeneousAE": "#d62728",
    "StandardAE": "#1f77b4",
    "PCA": "#2ca02c",
}
MODEL_LABELS: Dict[str, str] = {
    "HomogeneousAE": "HAE",
    "StandardAE": "AE",
    "PCA": "PCA",
}
MODEL_MARKERS: Dict[str, str] = {
    "HomogeneousAE": "o",
    "StandardAE": "s",
    "PCA": "^",
}

_DPI = 220
_FIGSIZE = (6.0, 4.0)


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, "#666666")


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _marker(name: str) -> str:
    return MODEL_MARKERS.get(name, "o")


def _finish(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-panel primitives (one PNG each)
# ---------------------------------------------------------------------------


def save_homogeneity_scan(
    scan_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    precision_floor: float = 1e-10,
    title: Optional[str] = None,
) -> None:
    """Encoder homogeneity residual vs scale lambda (log-log).

    ``scan_by_model[name]`` is a dict with keys ``scales`` and
    ``relative_residual`` produced by ``lib.metrics.homogeneity_scan``.
    Only autoencoder models are plotted (PCA is analytically
    1-homogeneous and has no scan entry).
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for name, scan in scan_by_model.items():
        scales = np.asarray(scan["scales"], dtype=np.float64)
        residuals = np.asarray(scan["relative_residual"], dtype=np.float64)
        plotted = np.maximum(residuals, precision_floor * 0.1)
        ax.plot(
            scales,
            plotted,
            color=_color(name),
            marker=_marker(name),
            markersize=4.0,
            linewidth=1.8,
            label=_label(name),
        )
    ax.axhline(
        precision_floor,
        linestyle="--",
        color="#555555",
        linewidth=1.0,
        label="precision floor",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"scale $\lambda$")
    ax.set_ylabel(r"$\|f(\lambda x) - \lambda^p f(x)\|^2 / \|\lambda^p f(x)\|^2$")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    _finish(fig, output_path)


def save_latent_hill_curves(
    latent_curves_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    ambient_curve: Optional[Mapping[str, np.ndarray]] = None,
    alpha_ambient: Optional[float] = None,
    p: float = 1.0,
    title: Optional[str] = None,
) -> None:
    """Hill estimator curves on the latent radii, one line per model.

    ``latent_curves_by_model[name]`` is a dict with keys ``k`` and
    ``alpha_hat``. ``ambient_curve`` (optional) is plotted in black as
    an extra reference line. A horizontal dashed line is drawn at
    ``alpha_ambient / p`` when ``alpha_ambient`` is supplied — this is
    the Proposition 1 target computed from the data itself, no
    dependence on an unobservable true alpha.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for name, curve in latent_curves_by_model.items():
        k_values = np.asarray(curve["k"], dtype=np.float64)
        alpha_hat = np.asarray(curve["alpha_hat"], dtype=np.float64)
        ax.plot(
            k_values,
            alpha_hat,
            color=_color(name),
            linewidth=1.8,
            label=_label(name),
        )
    if ambient_curve is not None:
        k_amb = np.asarray(ambient_curve["k"], dtype=np.float64)
        alpha_amb = np.asarray(ambient_curve["alpha_hat"], dtype=np.float64)
        ax.plot(
            k_amb,
            alpha_amb,
            color="#222222",
            linestyle=":",
            linewidth=1.3,
            label="ambient",
        )
    if alpha_ambient is not None:
        target = float(alpha_ambient) / float(p)
        ax.axhline(
            target,
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            label=rf"$\hat\alpha_{{\mathrm{{amb}}}}/p = {target:.2f}$",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"order statistic $k$")
    ax.set_ylabel(r"latent Hill estimate $\hat\alpha_k$")
    if title:
        ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    _finish(fig, output_path)


def save_extrapolation_curve(
    extrap_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> None:
    """Reconstruction MSE at increasing scale multipliers (log-log)."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for name, extrap in extrap_by_model.items():
        scales = np.asarray(extrap["scales"], dtype=np.float64)
        mse = np.asarray(extrap["mse"], dtype=np.float64)
        ax.plot(
            scales,
            np.maximum(mse, 1e-12),
            color=_color(name),
            marker=_marker(name),
            markersize=4.0,
            linewidth=1.8,
            label=_label(name),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"scale multiplier $\lambda$")
    ax.set_ylabel(r"test MSE $\|x_\lambda - \hat{x}_\lambda\|^2$")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    _finish(fig, output_path)


def save_binned_recon_error(
    binned_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> None:
    """Binned reconstruction MSE as a function of ambient radius."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for name, binned in binned_by_model.items():
        centers = np.asarray(binned["centers"], dtype=np.float64)
        median = np.asarray(binned["median"], dtype=np.float64)
        counts = np.asarray(binned["counts"], dtype=np.int64)
        valid = (counts >= 5) & np.isfinite(median)
        if not np.any(valid):
            continue
        ax.plot(
            centers[valid],
            median[valid],
            color=_color(name),
            marker=_marker(name),
            markersize=4.0,
            linewidth=1.8,
            label=_label(name),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"ambient radius $\|x\|$")
    ax.set_ylabel(r"median $\|x - \hat{x}\|^2$ per bin")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    _finish(fig, output_path)


def save_sweep_metric(
    parameter_values: Sequence[float],
    series_by_model: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    output_path: Path,
    *,
    metric_key: str,
    xlabel: str,
    ylabel: str,
    title: Optional[str] = None,
    yscale: Optional[str] = None,
    xscale: Optional[str] = None,
    reference_curve: Optional[Tuple[Sequence[float], Sequence[float], str]] = None,
) -> None:
    """Single-panel sweep plot with optional shaded multi-seed bands.

    ``series_by_model[name][metric_key]`` must be a dict with ``mean``
    and ``std`` arrays whose length matches ``parameter_values``. When
    ``std`` is non-zero a shaded band is drawn.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    parameter_array = np.asarray(parameter_values, dtype=np.float64)
    for name, metric_series in series_by_model.items():
        cell = metric_series.get(metric_key)
        if cell is None:
            continue
        mean = np.asarray(cell["mean"], dtype=np.float64)
        std = np.asarray(cell["std"], dtype=np.float64)
        color = _color(name)
        ax.plot(
            parameter_array,
            mean,
            color=color,
            marker=_marker(name),
            markersize=4.5,
            linewidth=1.8,
            label=_label(name),
        )
        if np.any(std > 0.0):
            ax.fill_between(
                parameter_array,
                mean - std,
                mean + std,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
    if reference_curve is not None:
        x_ref, y_ref, label_ref = reference_curve
        ax.plot(
            np.asarray(x_ref, dtype=np.float64),
            np.asarray(y_ref, dtype=np.float64),
            color="#222222",
            linestyle="--",
            linewidth=1.1,
            label=label_ref,
        )
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    _finish(fig, output_path)


def save_latent_vs_ambient_radius(
    radii_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    p: float,
    title: Optional[str] = None,
    max_points: int = 4000,
) -> None:
    """Log-log scatter of latent radius vs ambient radius, one cloud per model.

    ``radii_by_model[name]`` is a dict with keys ``ambient`` and ``latent``.
    For an exactly ``p``-homogeneous encoder Proposition 1 predicts
    ``||z|| propto ||x||^p``, i.e. a straight line of slope ``p`` on log-log.
    A dashed reference line at that slope is drawn through the HAE bulk so
    deviations are visible at a glance.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    rng = np.random.default_rng(0)

    anchor_point: Optional[Tuple[float, float]] = None
    all_ambient: List[np.ndarray] = []

    for name, radii in radii_by_model.items():
        ambient = np.asarray(radii["ambient"], dtype=np.float64)
        latent = np.asarray(radii["latent"], dtype=np.float64)
        mask = (ambient > 0) & (latent > 0) & np.isfinite(ambient) & np.isfinite(latent)
        ambient = ambient[mask]
        latent = latent[mask]
        if ambient.size == 0:
            continue
        all_ambient.append(ambient)
        if ambient.size > max_points:
            pick = rng.choice(ambient.size, size=max_points, replace=False)
            ambient_plot = ambient[pick]
            latent_plot = latent[pick]
        else:
            ambient_plot = ambient
            latent_plot = latent
        ax.scatter(
            ambient_plot,
            latent_plot,
            s=6,
            alpha=0.25,
            color=_color(name),
            label=_label(name),
            linewidths=0,
            rasterized=True,
        )
        if name == "HomogeneousAE" and anchor_point is None:
            bulk_mask = ambient <= np.quantile(ambient, 0.5)
            if np.any(bulk_mask):
                anchor_point = (
                    float(np.median(ambient[bulk_mask])),
                    float(np.median(latent[bulk_mask])),
                )

    if all_ambient and anchor_point is not None:
        ambient_cat = np.concatenate(all_ambient)
        x_ref = np.geomspace(float(ambient_cat.min()), float(ambient_cat.max()), 100)
        x0, y0 = anchor_point
        y_ref = y0 * (x_ref / x0) ** float(p)
        ax.plot(
            x_ref,
            y_ref,
            color="#222222",
            linestyle="--",
            linewidth=1.1,
            label=rf"slope $p = {p:g}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"ambient radius $\|x\|$")
    ax.set_ylabel(r"latent radius $\|z\|$")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", markerscale=2.0)
    _finish(fig, output_path)


def save_correction_magnitude_scatter(
    latent_radii: np.ndarray,
    delta_norms: np.ndarray,
    output_path: Path,
    *,
    n_quantiles: int = 16,
    title: Optional[str] = None,
) -> None:
    """HAE-only: ``||delta||`` vs latent radius with a binned-mean overlay.

    Adapted from ``minimal_p_homogeneous_encoder.py:485-506``. Shows that
    the decoder correction term vanishes as the latent radius grows,
    which is the asymptotic-homogeneity condition in the paper.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    rho = np.asarray(latent_radii, dtype=np.float64)
    d = np.asarray(delta_norms, dtype=np.float64)
    mask = (rho > 0) & np.isfinite(rho) & np.isfinite(d)
    rho = rho[mask]
    d = d[mask]

    ax.scatter(
        rho,
        d,
        s=6,
        alpha=0.25,
        color=_color("HomogeneousAE"),
        linewidths=0,
        rasterized=True,
    )

    quantiles = np.quantile(rho, np.linspace(0.0, 1.0, n_quantiles))
    centers: List[float] = []
    means: List[float] = []
    for i in range(len(quantiles) - 1):
        lower = quantiles[i]
        upper = quantiles[i + 1]
        if i == len(quantiles) - 2:
            bin_mask = (rho >= lower) & (rho <= upper)
        else:
            bin_mask = (rho >= lower) & (rho < upper)
        if np.any(bin_mask):
            centers.append(float(np.sqrt(max(lower, 1e-12) * max(upper, 1e-12))))
            means.append(float(d[bin_mask].mean()))

    if centers:
        ax.plot(
            centers,
            means,
            color="#222222",
            linewidth=2.0,
            label="binned mean",
        )
        ax.legend(loc="best")

    ax.set_xscale("log")
    ax.set_xlabel(r"latent radius $\|z\|$")
    ax.set_ylabel(r"correction magnitude $\|\delta\|$")
    if title:
        ax.set_title(title)
    _finish(fig, output_path)


def save_correction_by_regime(
    latent_radii: np.ndarray,
    delta_norms: np.ndarray,
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> None:
    """HAE-only: bar chart of mean ``||delta||`` by latent-radius percentile.

    Adapted from ``minimal_p_homogeneous_encoder.py:508-537``. Bars should
    decrease from the bulk regime to the far tail.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    rho = np.asarray(latent_radii, dtype=np.float64)
    d = np.asarray(delta_norms, dtype=np.float64)
    mask = (rho > 0) & np.isfinite(rho) & np.isfinite(d)
    rho = rho[mask]
    d = d[mask]

    percentile_labels = [50, 75, 90, 95, 99]
    percentile_values = np.percentile(rho, percentile_labels)

    means: List[float] = []
    labels: List[str] = []

    bulk_mask = rho <= percentile_values[0]
    if np.any(bulk_mask):
        means.append(float(d[bulk_mask].mean()))
        labels.append("≤50%")

    for i in range(1, len(percentile_labels)):
        lower = percentile_values[i - 1]
        upper = percentile_values[i]
        regime_mask = (rho > lower) & (rho <= upper)
        if np.any(regime_mask):
            means.append(float(d[regime_mask].mean()))
            labels.append(f"{percentile_labels[i - 1]}–{percentile_labels[i]}%")

    tail_mask = rho > percentile_values[-1]
    if np.any(tail_mask):
        means.append(float(d[tail_mask].mean()))
        labels.append(">99%")

    positions = np.arange(len(means))
    ax.bar(
        positions,
        means,
        color=_color("HomogeneousAE"),
        alpha=0.85,
        edgecolor="#222222",
        linewidth=0.6,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(r"mean $\|\delta\|$")
    ax.set_xlabel("latent-radius percentile regime")
    if title:
        ax.set_title(title)
    _finish(fig, output_path)


def save_curved_surface_scatter(
    data: np.ndarray,
    output_path: Path,
    *,
    color_by: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    max_points: int = 2500,
    cmap: str = "viridis",
) -> None:
    """Single 3-D scatter coloured by ambient radius.

    ``color_by`` is an optional (N,) array — defaults to the norms of
    ``data``. Points are subsampled to ``max_points`` for rendering speed.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.shape[1] != 3:
        raise ValueError(f"save_curved_surface_scatter requires D=3, got {data.shape[1]}")
    if color_by is None:
        color_by = np.linalg.norm(data, axis=1)
    rng = np.random.default_rng(0)
    if len(data) > max_points:
        pick = rng.choice(len(data), size=max_points, replace=False)
        data = data[pick]
        color_by = color_by[pick]
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        data[:, 0], data[:, 1], data[:, 2],
        c=color_by, cmap=cmap, s=4, alpha=0.55, linewidths=0,
    )
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    _finish(fig, output_path)


def save_exp02_panel_set(
    point: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    *,
    p: float,
    prefix: str = "",
) -> List[Path]:
    """Write the full set of exp02 diagnostic panels for one training point.

    ``point`` is ``{model_name: metrics_dict}`` where the metrics dict
    contains the underscore-prefixed arrays from ``evaluate_model``.
    Returns the list of paths written (up to 6).
    """
    from .metrics import hill_curve, hill_estimate

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    model_names = [n for n in ("HomogeneousAE", "StandardAE", "PCA") if n in point]

    radii_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    for name in model_names:
        m = point[name]
        if "_ambient_radii" in m and "_latent_radii" in m:
            radii_by_model[name] = {
                "ambient": m["_ambient_radii"],
                "latent": m["_latent_radii"],
            }
    if radii_by_model:
        path = output_dir / f"{prefix}fig_latent_vs_ambient_radius.png"
        save_latent_vs_ambient_radius(radii_by_model, path, p=p)
        written.append(path)

    latent_curves: Dict[str, Dict[str, np.ndarray]] = {}
    ambient_curve: Optional[Dict[str, np.ndarray]] = None
    alpha_ambient: Optional[float] = None
    for name in model_names:
        m = point[name]
        if "_latent_radii" in m:
            try:
                k, alpha_hat = hill_curve(m["_latent_radii"])
                latent_curves[name] = {"k": k, "alpha_hat": alpha_hat}
            except ValueError:
                pass
        if ambient_curve is None and "_ambient_radii" in m:
            try:
                k_a, ah_a = hill_curve(m["_ambient_radii"])
                ambient_curve = {"k": k_a, "alpha_hat": ah_a}
                alpha_ambient = hill_estimate(m["_ambient_radii"])["alpha"]
            except ValueError:
                pass
    if latent_curves:
        path = output_dir / f"{prefix}fig_latent_hill.png"
        save_latent_hill_curves(
            latent_curves, path,
            ambient_curve=ambient_curve,
            alpha_ambient=alpha_ambient,
            p=p,
        )
        written.append(path)

    extrap_by_model: Dict[str, Any] = {}
    for name in model_names:
        extrap = point[name].get("_extrapolation")
        if extrap is not None:
            extrap_by_model[name] = extrap
    if extrap_by_model:
        path = output_dir / f"{prefix}fig_extrapolation.png"
        save_extrapolation_curve(extrap_by_model, path)
        written.append(path)

    binned_by_model: Dict[str, Any] = {}
    for name in model_names:
        binned = point[name].get("_binned_error")
        if binned is not None:
            binned_by_model[name] = binned
    if binned_by_model:
        path = output_dir / f"{prefix}fig_recon_vs_radius.png"
        save_binned_recon_error(binned_by_model, path)
        written.append(path)

    hae = point.get("HomogeneousAE")
    if hae is not None:
        delta_norm = hae.get("_delta_norm")
        latent_radii = hae.get("_latent_radii")
        if delta_norm is not None and latent_radii is not None:
            path = output_dir / f"{prefix}fig_hae_correction_vs_radius.png"
            save_correction_magnitude_scatter(latent_radii, delta_norm, path)
            written.append(path)
            path = output_dir / f"{prefix}fig_hae_correction_by_regime.png"
            save_correction_by_regime(latent_radii, delta_norm, path)
            written.append(path)

    return written


# ---------------------------------------------------------------------------
# Composite figures (only where multi-view is inherent)
# ---------------------------------------------------------------------------


def plot_training_history(
    histories: Mapping[str, Dict[str, List[float]]],
    output_path: Path,
) -> None:
    """Diagnostic training curves (not a paper figure)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    for name, history in histories.items():
        epochs = np.asarray(history["epoch"])
        color = _color(name)
        axes[0].plot(
            epochs, history["train_total"], color=color, linewidth=1.4,
            label=f"{_label(name)} train",
        )
        axes[0].plot(
            epochs, history["val_total"], color=color, linestyle=":",
            linewidth=1.4, label=f"{_label(name)} val",
        )
        axes[1].plot(
            epochs, history["train_reconstruction"], color=color,
            linewidth=1.4, label=_label(name),
        )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("total loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("reconstruction MSE")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    _finish(fig, output_path)


def plot_curved_surface_diagnostic(
    x: np.ndarray,
    x_hat_by_model: Mapping[str, np.ndarray],
    scan_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Exp01 continuity figure: three 3-D scatters + homogeneity scan."""
    if x.shape[1] != 3:
        raise ValueError("plot_curved_surface_diagnostic is only defined for D=3")

    fig = plt.figure(figsize=(10.0, 8.0))
    ax_truth = fig.add_subplot(2, 2, 1, projection="3d")
    ax_hae = fig.add_subplot(2, 2, 2, projection="3d")
    ax_standard = fig.add_subplot(2, 2, 3, projection="3d")
    ax_scan = fig.add_subplot(2, 2, 4)

    rng = np.random.default_rng(0)
    if len(x) > 2500:
        pick = rng.choice(len(x), size=2500, replace=False)
        x = x[pick]
        x_hat_by_model = {k: v[pick] for k, v in x_hat_by_model.items()}

    radii = np.linalg.norm(x, axis=1)
    panels = [
        (ax_truth, x, "truth"),
        (ax_hae, x_hat_by_model.get("HomogeneousAE"), "HomogeneousAE"),
        (ax_standard, x_hat_by_model.get("StandardAE"), "StandardAE"),
    ]
    for ax, data, name in panels:
        if data is None:
            continue
        ax.scatter(
            data[:, 0], data[:, 1], data[:, 2],
            c=radii, cmap="viridis", s=4, alpha=0.55, linewidths=0,
        )
        title_name = "truth" if name == "truth" else _label(name)
        ax.set_title(title_name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    for name, scan in scan_by_model.items():
        scales = np.asarray(scan["scales"], dtype=np.float64)
        residuals = np.asarray(scan["relative_residual"], dtype=np.float64)
        plotted = np.maximum(residuals, 1e-13)
        ax_scan.plot(
            scales, plotted,
            color=_color(name), marker=_marker(name), markersize=4.0,
            linewidth=1.6, label=_label(name),
        )
    ax_scan.axhline(1e-10, linestyle="--", color="#555555", linewidth=0.9)
    ax_scan.set_xscale("log")
    ax_scan.set_yscale("log")
    ax_scan.set_xlabel(r"scale $\lambda$")
    ax_scan.set_ylabel(r"rel. homogeneity residual")
    ax_scan.set_title("encoder homogeneity", fontsize=10)
    ax_scan.grid(True, alpha=0.3)
    ax_scan.legend(loc="best", fontsize=8)

    _finish(fig, output_path)
