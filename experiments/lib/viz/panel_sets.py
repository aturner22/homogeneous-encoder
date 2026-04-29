"""Composite panel orchestrator and diagnostic training-history plot."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    facet_wrap,
    geom_line,
    ggplot,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    scale_y_log10,
    theme,
)

from ..metrics import hill_curve, hill_estimate
from ._base import (
    THEME_PUBLICATION,
    _col_vals,
    _label,
    _ordered_labels,
    _save_gg,
)
from .manifold_plots import save_latent_scatter_by_radius
from .tail_plots import (
    save_binned_recon_error,
    save_correction_magnitude_scatter,
    save_extrapolation_curve,
    save_latent_hill_curves,
    save_latent_vs_ambient_radius,
)


def plot_training_history(
    histories: Mapping[str, dict[str, list[float]]],
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
        + labs(x="Epoch", y="", color="", linetype="")
        + THEME_PUBLICATION
        + theme(legend_position="bottom")
    )
    _save_gg(p, output_path, width=12.0, height=4.5)


def save_diagnostic_panel_set(
    point: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    *,
    p: float,
    prefix: str = "",
) -> list[Path]:
    """Write the full set of per-model diagnostic panels for one training point.

    ``point`` is ``{model_name: metrics_dict}`` where the metrics dict
    contains the underscore-prefixed arrays from ``_evaluate_model``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    model_names = [n for n in ("HomogeneousAE", "StandardAE", "PCA")
                   if n in point]

    radii_by_model: dict[str, dict[str, np.ndarray]] = {}
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

    latent_curves: dict[str, dict[str, np.ndarray]] = {}
    latent_estimates: dict[str, float] = {}
    ambient_curve: dict[str, np.ndarray] | None = None
    alpha_ambient: float | None = None
    for name in model_names:
        m = point[name]
        if "_latent_radii" in m:
            try:
                k, alpha_hat = hill_curve(m["_latent_radii"])
                latent_curves[name] = {"k": k, "alpha_hat": alpha_hat}
                latent_estimates[name] = float(
                    hill_estimate(m["_latent_radii"])["alpha"]
                )
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
            latent_estimates_by_model=latent_estimates,
            p=p,
        )
        written.append(path)

    extrap_by_model: dict[str, Any] = {}
    for name in model_names:
        extrap = point[name].get("_extrapolation")
        if extrap is not None:
            extrap_by_model[name] = extrap
    if extrap_by_model:
        path = output_dir / f"{prefix}fig_extrapolation.png"
        save_extrapolation_curve(extrap_by_model, path)
        written.append(path)

    binned_by_model: dict[str, Any] = {}
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

    for name in model_names:
        m = point[name]
        codes = m.get("_latent_codes")
        radii_amb = m.get("_ambient_radii")
        if codes is None or radii_amb is None:
            continue
        path = output_dir / f"{prefix}fig_latent_scatter_{name}.png"
        save_latent_scatter_by_radius(
            np.asarray(codes),
            np.asarray(radii_amb),
            name,
            path,
        )
        written.append(path)

    return written
