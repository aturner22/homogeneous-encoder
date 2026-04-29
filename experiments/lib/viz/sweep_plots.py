"""Sweep summary plots: single metric vs parameter and the appendix grid."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_shape_manual,
    scale_x_log10,
    scale_y_log10,
)

from ._base import (
    THEME_PUBLICATION,
    _col_vals,
    _color,
    _finish,
    _label,
    _ordered_labels,
    _save_gg,
    _shp_vals,
)


def save_sweep_metric(
    parameter_values: Sequence[float],
    series_by_model: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    output_path: Path,
    *,
    metric_key: str,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    yscale: str | None = None,
    xscale: str | None = None,
    reference_curve: tuple[Sequence[float], Sequence[float], str] | None = None,
    show_bands: bool = True,
) -> None:
    """Single-panel sweep plot.

    ``show_bands=True`` (default) draws a ±1σ ribbon using ``cell["std"]``;
    ``show_bands=False`` draws just the point series, suitable for the
    canonical single-seed variant where only ``cell["mean"]`` is meaningful.
    """
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

    p = ggplot(df, aes("param", "mean", color="model"))
    if show_bands:
        p = p + geom_ribbon(aes(ymin="lower", ymax="upper", fill="model"),
                            alpha=0.2, colour="none")
    p = (
        p
        + geom_line(size=0.5, linetype="dotted", alpha=0.7)
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


def plot_sweep_grid(
    sweep_specs: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    metric_key: str = "hill_drift_latent",
    ylabel: str = r"Hill drift $|p\,\hat\alpha_\ell - \hat\alpha_a|$",
    yscale: str = "log",
) -> None:
    """Horizontal strip of sweep panels for the appendix.

    ``sweep_specs`` is a list of dicts with ``path`` (to a sweep.json),
    ``xlabel``, and optional ``xscale``. Each panel plots ``metric_key``
    for HomogeneousAE and StandardAE as mean line + std band.
    """
    n = len(sweep_specs)
    panel_w = max(2.1, 6.75 / max(n, 1))
    fig, axes = plt.subplots(
        1, n,
        figsize=(panel_w * n, 2.6),
        sharey=True,
    )
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.15)

    for ax, spec in zip(axes, sweep_specs, strict=True):
        path = Path(spec["path"])
        with open(path, encoding="utf-8") as fh:
            sweep = json.load(fh)
        x = np.asarray(sweep["parameter_values"], dtype=np.float64)
        for name in ("HomogeneousAE", "StandardAE"):
            series = []
            for raw in sweep["raw"]:
                cell = raw.get(name, {}).get(metric_key)
                if cell is None:
                    series.append((np.nan, np.nan))
                else:
                    series.append((cell["mean"], cell["std"]))
            mean = np.array([s[0] for s in series], dtype=np.float64)
            std = np.array([s[1] for s in series], dtype=np.float64)
            colour = _color(name)
            ax.plot(x, mean, color=colour, linewidth=1.2,
                    label=_label(name))
            ax.fill_between(x, mean - std, mean + std,
                            color=colour, alpha=0.18, linewidth=0)
        ax.set_xlabel(spec["xlabel"])
        if spec.get("xscale") == "log":
            ax.set_xscale("log")
        if yscale == "log":
            ax.set_yscale("log")

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best", fontsize=7)
    _finish(fig, output_path, tight=False)
