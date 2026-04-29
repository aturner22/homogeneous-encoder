"""Diagnostics for the Pareto-margin pre-standardisation step.

Two figures, both arranged as a row of small subplots — one column per
ambient coordinate (or per sampled column when the dimension is large):

* ``save_marginal_pareto_histograms``: density histograms with log-y
  for tails. Two rows: original (top), Pareto-standardised (bottom).
* ``save_marginal_pareto_hill``: Hill-estimator curves on each marginal
  before vs after the transform; verifies that the post-transform tail
  index matches the configured target.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..metrics import hill_curve
from ._base import _finish

_RAW_COLOR = "#0173B2"
_STD_COLOR = "#DE8F05"


def _select_columns(d: int, max_cols: int) -> Sequence[int]:
    if d <= max_cols:
        return list(range(d))
    rng = np.random.default_rng(0)
    return sorted(rng.choice(d, size=max_cols, replace=False).tolist())


def _format_dim(j: int) -> str:
    return f"Dim {j + 1}"


def save_marginal_pareto_histograms(
    raw: np.ndarray,
    transformed: np.ndarray,
    output_path: Path,
    *,
    max_cols: int = 6,
    bins: int = 80,
    pareto_kind: str = "one_sided",
) -> None:
    """Per-marginal histograms before / after Pareto pre-standardisation.

    When ``pareto_kind="two_sided"`` the lower row uses a symlog x-scale
    so both heavy tails are visible without clipping the negatives.
    """
    raw = np.asarray(raw, dtype=np.float64)
    transformed = np.asarray(transformed, dtype=np.float64)
    if raw.shape != transformed.shape:
        raise ValueError(
            f"raw and transformed shape mismatch: {raw.shape} vs {transformed.shape}"
        )
    d = raw.shape[1]
    cols = _select_columns(d, max_cols)
    n_cols = len(cols)

    fig, axes = plt.subplots(
        2, n_cols, figsize=(2.4 * n_cols, 4.4),
        sharey="row", squeeze=False,
    )
    for k, j in enumerate(cols):
        axes[0, k].hist(raw[:, j], bins=bins, color=_RAW_COLOR,
                        alpha=0.85, linewidth=0)
        axes[0, k].set_yscale("log")
        axes[0, k].set_title(_format_dim(j), fontsize=9, loc="center")
        axes[0, k].grid(True, alpha=0.25)

        axes[1, k].hist(transformed[:, j], bins=bins, color=_STD_COLOR,
                        alpha=0.85, linewidth=0)
        axes[1, k].set_yscale("log")
        if pareto_kind == "two_sided":
            axes[1, k].set_xscale("symlog", linthresh=1.0)
        else:
            axes[1, k].set_xscale("log")
        axes[1, k].grid(True, alpha=0.25)

        axes[1, k].set_xlabel("Value", fontsize=8)
    axes[0, 0].set_ylabel("Original\nDensity", fontsize=9)
    axes[1, 0].set_ylabel("Pareto-margins\nDensity", fontsize=9)

    if n_cols < d:
        fig.suptitle(
            f"Marginal Histograms: {n_cols} of {d} Dimensions",
            fontsize=10, y=1.02,
        )
    _finish(fig, output_path)


def save_marginal_pareto_hill(
    raw: np.ndarray,
    transformed: np.ndarray,
    output_path: Path,
    *,
    max_cols: int = 6,
    target_alpha: float | None = None,
    pareto_kind: str = "one_sided",
) -> None:
    """Per-marginal Hill estimator curves: original vs transformed.

    For each plotted dimension we Hill-estimate the right tail of the
    coordinate values directly. Original: heavy if the data was heavy.
    Transformed: should hover near ``target_alpha``. For
    ``pareto_kind="two_sided"`` the Hill estimator runs on ``|x|`` so
    the right tail of ``|X|`` carries the index ``alpha``.
    """
    raw = np.asarray(raw, dtype=np.float64)
    transformed = np.asarray(transformed, dtype=np.float64)
    if raw.shape != transformed.shape:
        raise ValueError(
            f"raw and transformed shape mismatch: {raw.shape} vs {transformed.shape}"
        )
    d = raw.shape[1]
    cols = _select_columns(d, max_cols)
    n_cols = len(cols)

    fig, axes = plt.subplots(
        1, n_cols, figsize=(2.6 * n_cols, 2.8),
        sharey=True, squeeze=False,
    )
    axes = axes[0]
    y_top = 0.0

    for k, j in enumerate(cols):
        ax = axes[k]
        for label, arr, color, transform_marker in (
            ("Original", raw[:, j], _RAW_COLOR, False),
            ("Pareto Margins", transformed[:, j], _STD_COLOR, True),
        ):
            values = arr
            if transform_marker and pareto_kind == "two_sided":
                values = np.abs(values)
            positive = values[values > 0]
            if positive.size < 50:
                continue
            try:
                k_arr, alpha_hat = hill_curve(positive)
            except ValueError:
                continue
            ax.plot(k_arr, alpha_hat, color=color, linewidth=1.4, label=label)
            y_top = max(y_top, float(np.nanmax(alpha_hat)))
        if target_alpha is not None:
            ax.axhline(float(target_alpha), color="#222222",
                       linestyle="--", linewidth=0.9)
            y_top = max(y_top, float(target_alpha))
        ax.set_xscale("log")
        ax.set_title(_format_dim(j), fontsize=9, loc="center")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Order Statistic", fontsize=8)

    axes[0].set_ylabel("Hill Estimate", fontsize=9)
    axes[0].set_ylim(0, max(y_top * 1.15, 1.0))
    axes[0].legend(loc="best", fontsize=7, frameon=False)

    if target_alpha is not None:
        suffix = " (|x| for two-sided)" if pareto_kind == "two_sided" else ""
        fig.suptitle(
            f"Per-marginal Hill Curves (Target Pareto Index = "
            f"{target_alpha:g}){suffix}",
            fontsize=10, y=1.04,
        )
    _finish(fig, output_path)
