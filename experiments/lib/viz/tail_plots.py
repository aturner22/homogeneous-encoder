"""Plotnine 2-D figures centred on tail / homogeneity behaviour."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_cartesian,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_shape_manual,
    scale_x_log10,
    scale_y_log10,
)

from ._base import (
    _LABEL_COLORS,
    THEME_PUBLICATION,
    _col_vals,
    _label,
    _ordered_labels,
    _save_gg,
    _shp_vals,
)


def save_latent_hill_curves(
    latent_curves_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    ambient_curve: Mapping[str, np.ndarray] | None = None,
    alpha_ambient: float | None = None,
    latent_estimates_by_model: Mapping[str, float] | None = None,
    p: float = 1.0,
    title: str | None = None,
) -> None:
    """Hill estimator curves on the latent radii, one line per model.

    When a curve dict carries optional ``alpha_hat_lo`` and
    ``alpha_hat_hi`` arrays (same shape as ``alpha_hat``), a shaded
    ribbon at those bounds is drawn behind the line.

    Per-model and ambient point estimates of the tail index are
    appended to the legend labels (e.g. ``HAE (α=3.79)``) rather than
    annotated on the panel — there is no spare room near the curves.
    """
    dfs: list[pd.DataFrame] = []
    for name, curve in latent_curves_by_model.items():
        k_values = np.asarray(curve["k"], dtype=np.float64)
        alpha_hat = np.asarray(curve["alpha_hat"], dtype=np.float64)
        dfs.append(pd.DataFrame({
            "k": k_values, "alpha_hat": alpha_hat * p, "model": _label(name),
        }))

    if ambient_curve is not None:
        k_amb = np.asarray(ambient_curve["k"], dtype=np.float64)
        alpha_amb = np.asarray(ambient_curve["alpha_hat"], dtype=np.float64)
        dfs.append(pd.DataFrame({
            "k": k_amb, "alpha_hat": alpha_amb, "model": "ambient",
        }))

    df = pd.concat(dfs, ignore_index=True)
    # Legend order: Ambient first, then HAE, PCA, AE (only those present).
    present = set(df["model"].unique())
    desired_order = ["ambient", "HAE", "PCA", "AE"]
    base_labels = [m for m in desired_order if m in present]

    # Per-model tail-index estimates → legend label suffixes.
    label_estimates: dict[str, float] = {}
    if latent_estimates_by_model:
        for raw_name, est in latent_estimates_by_model.items():
            if est is None or not np.isfinite(est):
                continue
            label_estimates[_label(raw_name)] = float(est) * p
    if (
        ambient_curve is not None
        and alpha_ambient is not None
        and np.isfinite(alpha_ambient)
    ):
        label_estimates["ambient"] = float(alpha_ambient)

    display_map: dict[str, str] = {}
    for lbl in base_labels:
        if lbl in label_estimates:
            display_map[lbl] = f"{lbl} (α={label_estimates[lbl]:.2f})"
        else:
            display_map[lbl] = lbl
    display_labels = [display_map[lbl] for lbl in base_labels]

    df["model"] = pd.Categorical(
        df["model"].map(display_map), categories=display_labels, ordered=True,
    )

    colors: dict[str, str] = {}
    linetypes: dict[str, str] = {}
    for base, disp in zip(base_labels, display_labels):
        if base == "ambient":
            # Muted blue: distinct from HAE (now black) and AE (orange);
            # reads as a calm reference rather than competing with the
            # foreground model curves.
            colors[disp] = "#4C78A8"
        elif base == "HAE":
            # HAE in black on this plot — the reader's eye should track
            # how closely it shadows the dark-grey ambient curve.
            colors[disp] = "#000000"
        else:
            colors[disp] = _col_vals([base])[base]
        linetypes[disp] = "solid"

    y_max = float(df["alpha_hat"].max()) * 1.15

    # Split ambient into its own (background, thicker) layer; everything
    # else stays in the foreground as thinner semi-transparent lines.
    ambient_disp = display_map.get("ambient")
    ambient_df = df[df["model"] == ambient_disp] if ambient_disp else None
    model_df = df[df["model"] != ambient_disp] if ambient_disp else df

    p_plot = ggplot(df, aes("k", "alpha_hat", color="model", linetype="model"))
    if ambient_df is not None and not ambient_df.empty:
        p_plot = p_plot + geom_line(
            data=ambient_df, mapping=aes("k", "alpha_hat", color="model"),
            size=2.8, alpha=0.6, inherit_aes=False,
        )
    if not model_df.empty:
        p_plot = p_plot + geom_line(
            data=model_df,
            mapping=aes("k", "alpha_hat", color="model", linetype="model"),
            size=0.8, alpha=0.85, inherit_aes=False,
        )
    p_plot = (
        p_plot
        + scale_x_log10()
        + scale_color_manual(values=colors)
        + scale_linetype_manual(values=linetypes)
        + coord_cartesian(ylim=(0, y_max))
        + labs(x="Order Statistic",
               y="Hill Estimate",
               color="", linetype="")
        + THEME_PUBLICATION
    )

    if title:
        p_plot = p_plot + labs(title=title)
    _save_gg(p_plot, output_path)


def save_extrapolation_curve(
    extrap_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    title: str | None = None,
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
        + geom_line(size=0.5, linetype="dotted", alpha=0.7)
        + geom_point(size=3)
        + scale_x_log10()
        + scale_y_log10()
        + scale_color_manual(values=_col_vals(labels))
        + scale_shape_manual(values=_shp_vals(labels))
        + labs(x="Scale Multiplier",
               y="Reconstruction MSE",
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
    title: str | None = None,
) -> None:
    """Binned reconstruction MSE as a function of ambient radius.

    Each model gets a median line connecting bin centres with a
    shaded q25–q75 ribbon for the within-bin IQR. Layered AE → PCA →
    HAE so HAE sits on top. Bins with fewer than five samples are
    dropped.
    """
    dfs: list[pd.DataFrame] = []
    for name, binned in binned_by_model.items():
        centers = np.asarray(binned["centers"], dtype=np.float64)
        median = np.asarray(binned["median"], dtype=np.float64)
        counts = np.asarray(binned["counts"], dtype=np.int64)
        q25 = np.asarray(binned.get("q25", median), dtype=np.float64)
        q75 = np.asarray(binned.get("q75", median), dtype=np.float64)
        valid = (counts >= 5) & np.isfinite(median) & (median > 0)
        if not np.any(valid):
            continue
        dfs.append(pd.DataFrame({
            "center": centers[valid],
            "median": median[valid],
            "q25": np.maximum(q25[valid], 1e-30),
            "q75": q75[valid],
            "model": _label(name),
        }))

    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)
    labels = _ordered_labels(df["model"].unique())
    df["model"] = pd.Categorical(df["model"], categories=labels, ordered=True)
    layer_order = [m for m in ("AE", "PCA", "HAE") if m in labels]
    colors = _col_vals(labels)

    p = ggplot(mapping=aes("center", "median"))
    for label in layer_order:
        sub = df[df["model"] == label]
        if sub.empty:
            continue
        p = (
            p
            + geom_ribbon(
                data=sub,
                mapping=aes(x="center", ymin="q25", ymax="q75", fill="model"),
                alpha=0.18, colour="none", inherit_aes=False,
            )
            + geom_line(
                data=sub,
                mapping=aes("center", "median", color="model"),
                size=1.0, alpha=0.9, inherit_aes=False,
            )
            + geom_point(
                data=sub,
                mapping=aes("center", "median", color="model", shape="model"),
                size=2.4, inherit_aes=False,
            )
        )
    p = (
        p
        + scale_x_log10()
        + scale_y_log10()
        + scale_color_manual(values=colors, breaks=layer_order[::-1])
        + scale_fill_manual(values=colors, guide=None)
        + scale_shape_manual(values=_shp_vals(labels), breaks=layer_order[::-1])
        + labs(x="Ambient Radius",
               y="Reconstruction Error (median, IQR)",
               color="", shape="")
        + THEME_PUBLICATION
    )
    if title:
        p = p + labs(title=title)
    _save_gg(p, output_path)


def save_latent_vs_ambient_radius(
    radii_by_model: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    p: float,
    title: str | None = None,
    max_points: int = 4000,
) -> None:
    """Log-log scatter of latent radius vs ambient radius, one cloud per model."""
    rng = np.random.default_rng(0)
    dfs: list[pd.DataFrame] = []

    for name, radii in radii_by_model.items():
        ambient = np.asarray(radii["ambient"], dtype=np.float64)
        latent = np.asarray(radii["latent"], dtype=np.float64)
        mask = ((ambient > 0) & (latent > 0)
                & np.isfinite(ambient) & np.isfinite(latent))
        ambient = ambient[mask]
        latent = latent[mask]
        if ambient.size == 0:
            continue
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

    df = pd.concat(dfs, ignore_index=True)
    labels = _ordered_labels(df["model"].unique())
    df["model"] = pd.Categorical(df["model"], categories=labels, ordered=True)

    # Layer order: AE bottom, PCA middle, HAE on top.
    layer_order = [m for m in ("AE", "PCA", "HAE") if m in labels]

    x_min = float(df["ambient"].min())
    x_max = float(df["ambient"].max())
    y_min = float(df["latent"].min())
    y_max = float(df["latent"].max())

    p_plot = ggplot(mapping=aes("ambient", "latent"))
    for label in layer_order:
        sub = df[df["model"] == label]
        if sub.empty:
            continue
        p_plot = p_plot + geom_point(
            data=sub,
            mapping=aes("ambient", "latent", color="model"),
            size=1.8, alpha=0.5, stroke=0, inherit_aes=False,
        )
    p_plot = (
        p_plot
        + scale_x_log10(limits=(x_min / 1.2, x_max * 1.2))
        + scale_y_log10(limits=(y_min / 1.4, y_max * 1.2))
        + scale_color_manual(values=_col_vals(labels), breaks=layer_order[::-1])
        + guides(color=guide_legend(override_aes={"size": 3, "alpha": 1}))
        + labs(x="Ambient Radius",
               y="Latent Radius",
               color="")
        + THEME_PUBLICATION
    )

    if title:
        p_plot = p_plot + labs(title=title)
    _save_gg(p_plot, output_path)


def save_correction_magnitude_scatter(
    latent_radii: np.ndarray,
    delta_norms: np.ndarray,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """HAE-only: ||delta|| vs latent radius (raw scatter)."""
    rho = np.asarray(latent_radii, dtype=np.float64)
    d = np.asarray(delta_norms, dtype=np.float64)
    mask = (rho > 0) & np.isfinite(rho) & np.isfinite(d)
    rho = rho[mask]
    d = d[mask]

    scatter_df = pd.DataFrame({"rho": rho, "delta": d})

    p = (
        ggplot(scatter_df, aes("rho", "delta"))
        + geom_point(size=1.0, alpha=0.5, color=_LABEL_COLORS["HAE"], stroke=0)
        + scale_x_log10()
        + labs(x="Latent Radius",
               y="Correction Magnitude")
        + THEME_PUBLICATION
    )

    if title:
        p = p + labs(title=title)
    _save_gg(p, output_path)


