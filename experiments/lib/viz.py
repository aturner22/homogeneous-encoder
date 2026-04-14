"""Simple, single-panel publication plots.

2D plots use plotnine (ggplot2 grammar of graphics). The two 3D scatter
functions remain in matplotlib because plotnine has no 3D support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    annotate,
    coord_cartesian,
    facet_wrap,
    geom_col,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    ggsave,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_shape_manual,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_bw,
    element_blank,
    element_line,
    element_text,
)


# ---------------------------------------------------------------------------
# Aesthetic constants
# ---------------------------------------------------------------------------


def _apply_mpl_aesthetic() -> None:
    """Quiet serif look for the matplotlib 3D plots."""
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


_apply_mpl_aesthetic()


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

# plotnine theme for publication-quality 2D plots
THEME_PUBLICATION = (
    theme_bw()
    + theme(
        text=element_text(family="serif", size=11),
        plot_title=element_text(size=12),
        panel_grid_major=element_line(size=0.5, colour="#cccccc"),
        panel_grid_minor=element_blank(),
        legend_background=element_blank(),
        legend_key=element_blank(),
    )
)

_MODEL_ORDER = ["HAE", "AE", "PCA"]
_LABEL_COLORS: Dict[str, str] = {"HAE": "#d62728", "AE": "#1f77b4", "PCA": "#2ca02c"}
_LABEL_SHAPES: Dict[str, str] = {"HAE": "o", "AE": "s", "PCA": "^"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _save_gg(p, path: Path, width: float = 6.0, height: float = 4.0) -> None:
    ggsave(p, filename=str(path), width=width, height=height,
           dpi=_DPI, verbose=False)


def _finish(fig, output_path: Path) -> None:
    """Save a matplotlib figure (3D plots only)."""
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2D plots (plotnine)
# ---------------------------------------------------------------------------


def save_homogeneity_scan(
    scan_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    precision_floor: float = 1e-10,
    title: Optional[str] = None,
) -> None:
    """Encoder homogeneity residual vs scale lambda (log-log)."""
    dfs: list[pd.DataFrame] = []
    for name, scan in scan_by_model.items():
        scales = np.asarray(scan["scales"], dtype=np.float64)
        residuals = np.asarray(scan["relative_residual"], dtype=np.float64)
        plotted = np.maximum(residuals, precision_floor * 0.1)
        dfs.append(pd.DataFrame({
            "scale": scales, "residual": plotted, "model": _label(name),
        }))

    df = pd.concat(dfs, ignore_index=True)
    labels = _ordered_labels(df["model"].unique())
    df["model"] = pd.Categorical(df["model"], categories=labels, ordered=True)

    p = (
        ggplot(df, aes("scale", "residual", color="model", shape="model"))
        + geom_line(size=1.8)
        + geom_point(size=3)
        + geom_hline(yintercept=precision_floor, linetype="dashed",
                     color="#555555", size=0.8)
        + scale_x_log10()
        + scale_y_log10()
        + scale_color_manual(values=_col_vals(labels))
        + scale_shape_manual(values=_shp_vals(labels))
        + labs(x=r"scale $\lambda$",
               y=(r"$\|f(\lambda x) - \lambda^p f(x)\|^2"
                  r" / \|\lambda^p f(x)\|^2$"),
               color="", shape="")
        + THEME_PUBLICATION
    )
    if title:
        p = p + labs(title=title)
    _save_gg(p, output_path)


def save_latent_hill_curves(
    latent_curves_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    ambient_curve: Optional[Mapping[str, np.ndarray]] = None,
    alpha_ambient: Optional[float] = None,
    p: float = 1.0,
    title: Optional[str] = None,
) -> None:
    """Hill estimator curves on the latent radii, one line per model."""
    dfs: list[pd.DataFrame] = []
    for name, curve in latent_curves_by_model.items():
        k_values = np.asarray(curve["k"], dtype=np.float64)
        alpha_hat = np.asarray(curve["alpha_hat"], dtype=np.float64)
        dfs.append(pd.DataFrame({
            "k": k_values, "alpha_hat": alpha_hat, "model": _label(name),
        }))

    if ambient_curve is not None:
        k_amb = np.asarray(ambient_curve["k"], dtype=np.float64)
        alpha_amb = np.asarray(ambient_curve["alpha_hat"], dtype=np.float64)
        dfs.append(pd.DataFrame({
            "k": k_amb, "alpha_hat": alpha_amb, "model": "ambient",
        }))

    df = pd.concat(dfs, ignore_index=True)
    all_labels = _ordered_labels(df["model"].unique())
    if "ambient" in df["model"].values:
        all_labels = all_labels + ["ambient"]
    df["model"] = pd.Categorical(df["model"], categories=all_labels, ordered=True)

    colors = _col_vals(all_labels)
    colors["ambient"] = "#222222"
    linetypes = {m: "solid" for m in all_labels}
    linetypes["ambient"] = "dotted"

    y_max = float(df["alpha_hat"].max()) * 1.15

    p_plot = (
        ggplot(df, aes("k", "alpha_hat", color="model", linetype="model"))
        + geom_line(size=1.5)
        + scale_x_log10()
        + scale_color_manual(values=colors)
        + scale_linetype_manual(values=linetypes)
        + coord_cartesian(ylim=(0, y_max))
        + labs(x=r"order statistic $k$",
               y=r"latent Hill estimate $\hat\alpha_k$",
               color="", linetype="")
        + THEME_PUBLICATION
    )

    if alpha_ambient is not None:
        target = float(alpha_ambient) / float(p)
        p_plot = (
            p_plot
            + geom_hline(yintercept=target, linetype="dashed",
                         color="#222222", size=0.8)
            + annotate("text", x=float(df["k"].max()), y=target,
                       label=rf"$\hat\alpha_{{\mathrm{{amb}}}}/p = {target:.2f}$",
                       ha="right", va="bottom", size=9, color="#444444")
        )
    if title:
        p_plot = p_plot + labs(title=title)
    _save_gg(p_plot, output_path)


def save_extrapolation_curve(
    extrap_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> None:
    """Reconstruction MSE at increasing scale multipliers (log-log)."""
    dfs: list[pd.DataFrame] = []
    for name, extrap in extrap_by_model.items():
        scales = np.asarray(extrap["scales"], dtype=np.float64)
        mse = np.maximum(np.asarray(extrap["mse"], dtype=np.float64), 1e-12)
        dfs.append(pd.DataFrame({
            "scale": scales, "mse": mse, "model": _label(name),
        }))

    df = pd.concat(dfs, ignore_index=True)
    labels = _ordered_labels(df["model"].unique())
    df["model"] = pd.Categorical(df["model"], categories=labels, ordered=True)

    p = (
        ggplot(df, aes("scale", "mse", color="model", shape="model"))
        + geom_line(size=1.8)
        + geom_point(size=3)
        + scale_x_log10()
        + scale_y_log10()
        + scale_color_manual(values=_col_vals(labels))
        + scale_shape_manual(values=_shp_vals(labels))
        + labs(x=r"scale multiplier $\lambda$",
               y=r"test MSE $\|x_\lambda - \hat{x}_\lambda\|^2$",
               color="", shape="")
        + THEME_PUBLICATION
    )
    if title:
        p = p + labs(title=title)
    _save_gg(p, output_path)


def save_binned_recon_error(
    binned_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> None:
    """Binned reconstruction MSE as a function of ambient radius."""
    dfs: list[pd.DataFrame] = []
    for name, binned in binned_by_model.items():
        centers = np.asarray(binned["centers"], dtype=np.float64)
        median = np.asarray(binned["median"], dtype=np.float64)
        counts = np.asarray(binned["counts"], dtype=np.int64)
        valid = (counts >= 5) & np.isfinite(median)
        if not np.any(valid):
            continue
        dfs.append(pd.DataFrame({
            "center": centers[valid], "median": median[valid],
            "model": _label(name),
        }))

    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)
    labels = _ordered_labels(df["model"].unique())
    df["model"] = pd.Categorical(df["model"], categories=labels, ordered=True)

    p = (
        ggplot(df, aes("center", "median", color="model", shape="model"))
        + geom_line(size=1.8)
        + geom_point(size=3)
        + scale_x_log10()
        + scale_y_log10()
        + scale_color_manual(values=_col_vals(labels))
        + scale_shape_manual(values=_shp_vals(labels))
        + labs(x=r"ambient radius $\|x\|$",
               y=r"median $\|x - \hat{x}\|^2$ per bin",
               color="", shape="")
        + THEME_PUBLICATION
    )
    if title:
        p = p + labs(title=title)
    _save_gg(p, output_path)


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
    """Single-panel sweep plot with optional shaded multi-seed bands."""
    parameter_array = np.asarray(parameter_values, dtype=np.float64)
    dfs: list[pd.DataFrame] = []
    for name, metric_series in series_by_model.items():
        cell = metric_series.get(metric_key)
        if cell is None:
            continue
        mean = np.asarray(cell["mean"], dtype=np.float64)
        std = np.asarray(cell["std"], dtype=np.float64)
        dfs.append(pd.DataFrame({
            "param": parameter_array, "mean": mean,
            "lower": mean - std, "upper": mean + std,
            "model": _label(name),
        }))

    df = pd.concat(dfs, ignore_index=True)
    labels = _ordered_labels(df["model"].unique())
    df["model"] = pd.Categorical(df["model"], categories=labels, ordered=True)

    p = (
        ggplot(df, aes("param", "mean", color="model"))
        + geom_ribbon(aes(ymin="lower", ymax="upper", fill="model"),
                      alpha=0.2, colour="none")
        + geom_line(size=1.8)
        + geom_point(aes(shape="model"), size=3.5)
        + scale_color_manual(values=_col_vals(labels))
        + scale_fill_manual(values=_col_vals(labels))
        + scale_shape_manual(values=_shp_vals(labels))
        + labs(x=xlabel, y=ylabel, color="", shape="", fill="")
        + THEME_PUBLICATION
    )
    if xscale == "log":
        p = p + scale_x_log10()
    if yscale == "log":
        p = p + scale_y_log10()

    if reference_curve is not None:
        x_ref, y_ref, label_ref = reference_curve
        ref_df = pd.DataFrame({
            "param": np.asarray(x_ref, dtype=np.float64),
            "mean": np.asarray(y_ref, dtype=np.float64),
            "reference": label_ref,
        })
        p = (p
             + geom_line(data=ref_df, mapping=aes("param", "mean",
                         linetype="reference"),
                         inherit_aes=False, color="#222222", size=1.0)
             + scale_linetype_manual(values={label_ref: "dashed"}, name=""))

    if title:
        p = p + labs(title=title)
    _save_gg(p, output_path)


def save_latent_vs_ambient_radius(
    radii_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    p: float,
    title: Optional[str] = None,
    max_points: int = 4000,
) -> None:
    """Log-log scatter of latent radius vs ambient radius, one cloud per model."""
    rng = np.random.default_rng(0)
    dfs: list[pd.DataFrame] = []
    anchor_point: Optional[Tuple[float, float]] = None
    all_ambient: List[np.ndarray] = []

    for name, radii in radii_by_model.items():
        ambient = np.asarray(radii["ambient"], dtype=np.float64)
        latent = np.asarray(radii["latent"], dtype=np.float64)
        mask = ((ambient > 0) & (latent > 0)
                & np.isfinite(ambient) & np.isfinite(latent))
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
        dfs.append(pd.DataFrame({
            "ambient": ambient_plot, "latent": latent_plot,
            "model": _label(name),
        }))
        if name == "HomogeneousAE" and anchor_point is None:
            bulk_mask = ambient <= np.quantile(ambient, 0.5)
            if np.any(bulk_mask):
                anchor_point = (
                    float(np.median(ambient[bulk_mask])),
                    float(np.median(latent[bulk_mask])),
                )

    df = pd.concat(dfs, ignore_index=True)
    labels = _ordered_labels(df["model"].unique())
    df["model"] = pd.Categorical(df["model"], categories=labels, ordered=True)

    p_plot = (
        ggplot(df, aes("ambient", "latent", color="model"))
        + geom_point(size=0.5, alpha=0.25, stroke=0)
        + scale_x_log10()
        + scale_y_log10()
        + scale_color_manual(values=_col_vals(labels))
        + labs(x=r"ambient radius $\|x\|$",
               y=r"latent radius $\|z\|$",
               color="")
        + THEME_PUBLICATION
    )

    if all_ambient and anchor_point is not None:
        ambient_cat = np.concatenate(all_ambient)
        x_ref = np.geomspace(float(ambient_cat.min()),
                             float(ambient_cat.max()), 100)
        x0, y0 = anchor_point
        y_ref = y0 * (x_ref / x0) ** float(p)
        ref_df = pd.DataFrame({"ambient": x_ref, "latent": y_ref})
        p_plot = p_plot + geom_line(
            data=ref_df, mapping=aes("ambient", "latent"),
            inherit_aes=False, color="#222222", linetype="dashed", size=0.9,
        )

    if title:
        p_plot = p_plot + labs(title=title)
    _save_gg(p_plot, output_path)


def save_correction_magnitude_scatter(
    latent_radii: np.ndarray,
    delta_norms: np.ndarray,
    output_path: Path,
    *,
    n_quantiles: int = 16,
    title: Optional[str] = None,
) -> None:
    """HAE-only: ||delta|| vs latent radius with a binned-mean overlay."""
    rho = np.asarray(latent_radii, dtype=np.float64)
    d = np.asarray(delta_norms, dtype=np.float64)
    mask = (rho > 0) & np.isfinite(rho) & np.isfinite(d)
    rho = rho[mask]
    d = d[mask]

    scatter_df = pd.DataFrame({"rho": rho, "delta": d})

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

    p = (
        ggplot(scatter_df, aes("rho", "delta"))
        + geom_point(size=0.5, alpha=0.25, color=_LABEL_COLORS["HAE"], stroke=0)
        + scale_x_log10()
        + labs(x=r"latent radius $\|z\|$",
               y=r"correction magnitude $\|\delta\|$")
        + THEME_PUBLICATION
    )

    if centers:
        binned_df = pd.DataFrame({"rho": centers, "delta": means})
        p = p + geom_line(data=binned_df, mapping=aes("rho", "delta"),
                          inherit_aes=False, color="#222222", size=1.8)

    if title:
        p = p + labs(title=title)
    _save_gg(p, output_path)


def save_correction_by_regime(
    latent_radii: np.ndarray,
    delta_norms: np.ndarray,
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> None:
    """HAE-only: bar chart of mean ||delta|| by latent-radius percentile."""
    rho = np.asarray(latent_radii, dtype=np.float64)
    d = np.asarray(delta_norms, dtype=np.float64)
    mask = (rho > 0) & np.isfinite(rho) & np.isfinite(d)
    rho = rho[mask]
    d = d[mask]

    percentile_labels = [50, 75, 90, 95, 99]
    percentile_values = np.percentile(rho, percentile_labels)

    bar_means: List[float] = []
    bar_labels: List[str] = []

    bulk_mask = rho <= percentile_values[0]
    if np.any(bulk_mask):
        bar_means.append(float(d[bulk_mask].mean()))
        bar_labels.append("\u226450%")

    for i in range(1, len(percentile_labels)):
        lower = percentile_values[i - 1]
        upper = percentile_values[i]
        regime_mask = (rho > lower) & (rho <= upper)
        if np.any(regime_mask):
            bar_means.append(float(d[regime_mask].mean()))
            bar_labels.append(
                f"{percentile_labels[i - 1]}\u2013{percentile_labels[i]}%")

    tail_mask = rho > percentile_values[-1]
    if np.any(tail_mask):
        bar_means.append(float(d[tail_mask].mean()))
        bar_labels.append(">99%")

    df = pd.DataFrame({"regime": bar_labels, "mean_delta": bar_means})
    df["regime"] = pd.Categorical(df["regime"], categories=bar_labels, ordered=True)

    p = (
        ggplot(df, aes("regime", "mean_delta"))
        + geom_col(fill=_LABEL_COLORS["HAE"], alpha=0.85,
                   colour="#222222", size=0.4)
        + labs(x="latent-radius percentile regime",
               y=r"mean $\|\delta\|$")
        + THEME_PUBLICATION
        + theme(axis_text_x=element_text(rotation=15, ha="right"))
    )
    if title:
        p = p + labs(title=title)
    _save_gg(p, output_path)


# ---------------------------------------------------------------------------
# 3D plots (matplotlib — plotnine has no 3D support)
# ---------------------------------------------------------------------------


def save_curved_surface_scatter(
    data: np.ndarray,
    output_path: Path,
    *,
    color_by: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    max_points: int = 2500,
    cmap: str = "viridis",
) -> None:
    """Single 3-D scatter coloured by ambient radius."""
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


def plot_curved_surface_diagnostic(
    x: np.ndarray,
    x_hat_by_model: Mapping[str, np.ndarray],
    scan_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Exp01 continuity figure: three 3-D scatters + homogeneity scan."""
    if x.shape[1] != 3:
        raise ValueError(
            "plot_curved_surface_diagnostic is only defined for D=3")

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


# ---------------------------------------------------------------------------
# Diagnostic (not paper figures)
# ---------------------------------------------------------------------------


def plot_training_history(
    histories: Mapping[str, Dict[str, List[float]]],
    output_path: Path,
) -> None:
    """Diagnostic training curves (not a paper figure)."""
    rows: list[dict] = []
    for name, history in histories.items():
        lbl = _label(name)
        epochs = history["epoch"]
        for i, ep in enumerate(epochs):
            rows.append({"epoch": ep, "value": history["train_total"][i],
                         "model": lbl, "split": "train",
                         "panel": "Total loss"})
            rows.append({"epoch": ep, "value": history["val_total"][i],
                         "model": lbl, "split": "val",
                         "panel": "Total loss"})
            rows.append({"epoch": ep,
                         "value": history["train_reconstruction"][i],
                         "model": lbl, "split": "train",
                         "panel": "Reconstruction MSE"})

    df = pd.DataFrame(rows)
    labels = _ordered_labels(df["model"].unique())
    df["model"] = pd.Categorical(df["model"], categories=labels, ordered=True)
    df["split"] = pd.Categorical(df["split"],
                                 categories=["train", "val"], ordered=True)
    df["panel"] = pd.Categorical(
        df["panel"],
        categories=["Total loss", "Reconstruction MSE"],
        ordered=True,
    )

    p = (
        ggplot(df, aes("epoch", "value", color="model", linetype="split"))
        + geom_line(size=1.2)
        + scale_y_log10()
        + scale_color_manual(values=_col_vals(labels))
        + scale_linetype_manual(values={"train": "solid", "val": "dotted"})
        + facet_wrap("~panel", scales="free_y")
        + labs(x="epoch", y="", color="", linetype="")
        + THEME_PUBLICATION
        + theme(legend_position="bottom")
    )
    _save_gg(p, output_path, width=12.0, height=4.5)


# ---------------------------------------------------------------------------
# Composite panel orchestrator
# ---------------------------------------------------------------------------


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

    model_names = [n for n in ("HomogeneousAE", "StandardAE", "PCA")
                   if n in point]

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
