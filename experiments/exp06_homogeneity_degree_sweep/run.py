"""exp06: sweep the homogeneity degree p.

The tightest numerical check of Proposition 1: if the encoder is exactly
p-homogeneous, the latent tail index should track ``alpha_ambient / p``
for every p. We sweep p in {0.5, 1.0, 2.0}, train the three-model zoo
multi-seed at each point, and save one single-panel PNG:

- ``fig_latent_hill_vs_p.png`` — absolute latent Hill estimate vs p,
  with the ``alpha_ambient / p`` target drawn as a dashed reference
  curve (``alpha_ambient`` is itself a Hill estimate on the ambient data
  — so the target does not depend on the unobservable true alpha).

The worst homogeneity residual is still recorded in ``exp06_summary.json``
for the appendix, but is not plotted — encoder homogeneity is structural
and gets its single visual verification in exp01.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import FlexibleToyConfig, ensure_output_dir  # noqa: E402
from lib.sweep import MODEL_NAMES, run_flexible_toy_sweep, write_sweep_json  # noqa: E402
from lib.viz import save_exp02_panel_set, save_sweep_metric  # noqa: E402


def _extract_scalar_series(
    per_seed_points: List[Mapping[str, List[Mapping[str, Any]]]],
    path: tuple,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Collect mean/std across seeds of a scalar field nested inside metrics.

    ``path`` is a tuple of keys to follow inside each per-seed metrics dict
    (e.g. ``("hill_latent", "alpha")``). Models whose metrics lack the full
    path at any seed are dropped (e.g. PCA has no ``homogeneity_error`` key).
    """
    series: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name in MODEL_NAMES:
        means: List[float] = []
        stds: List[float] = []
        ok_for_model = True
        for point in per_seed_points:
            seed_list = point.get(model_name, [])
            values: List[float] = []
            for seed_metrics in seed_list:
                node: Any = seed_metrics
                missing = False
                for key in path:
                    if not isinstance(node, Mapping) or key not in node:
                        missing = True
                        break
                    node = node[key]
                if missing:
                    continue
                try:
                    values.append(float(node))
                except (TypeError, ValueError):
                    continue
            if not values:
                ok_for_model = False
                break
            values_arr = np.asarray(values, dtype=np.float64)
            means.append(float(np.nanmean(values_arr)))
            stds.append(float(np.nanstd(values_arr)))
        if ok_for_model:
            series[model_name] = {
                "mean": np.asarray(means, dtype=np.float64),
                "std": np.asarray(stds, dtype=np.float64),
            }
    return series


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "results"
    base_config = FlexibleToyConfig(output_dir=str(output_dir))
    ensure_output_dir(base_config)

    p_values = [0.5, 1.0, 2.0]
    result = run_flexible_toy_sweep(
        base_config=base_config,
        parameter_name="p_homogeneity",
        parameter_values=p_values,
    )

    output = Path(base_config.output_dir)
    write_sweep_json(result, output / "sweep.json")

    per_seed_points = result["per_seed_points"]

    latent_alpha_series = _extract_scalar_series(
        per_seed_points, path=("hill_latent", "alpha")
    )
    homogeneity_worst_series = _extract_scalar_series(
        per_seed_points, path=("homogeneity_error", "worst")
    )
    ambient_alpha_series = _extract_scalar_series(
        per_seed_points, path=("hill_ambient", "alpha")
    )

    # Ambient Hill is independent of p but the sweep recomputes it each
    # point; average everything down to one number for the target curve.
    any_model = next(iter(ambient_alpha_series.values()))
    alpha_ambient = float(np.nanmean(any_model["mean"]))
    print(f"Ambient Hill estimate (mean across sweep points): {alpha_ambient:.3f}")

    # Pack for save_sweep_metric: expects {model: {metric_key: {mean, std}}}.
    latent_hill_series = {
        name: {"hill_latent_alpha": series}
        for name, series in latent_alpha_series.items()
    }

    p_dense = np.linspace(min(p_values), max(p_values), 100)
    reference_curve = (p_dense, alpha_ambient / p_dense, r"$\hat\alpha_{\mathrm{amb}}/p$")

    save_sweep_metric(
        parameter_values=p_values,
        series_by_model=latent_hill_series,
        output_path=output / "fig_latent_hill_vs_p.png",
        metric_key="hill_latent_alpha",
        xlabel=r"homogeneity degree $p$",
        ylabel=r"latent Hill $\hat\alpha$",
        yscale="log",
        reference_curve=reference_curve,
    )
    print(f"Wrote fig_latent_hill_vs_p.png to {output}")

    summary_payload = {
        "parameter_name": "p_homogeneity",
        "parameter_values": p_values,
        "alpha_ambient": alpha_ambient,
        "latent_hill_alpha": {
            name: {
                "mean": series["mean"].tolist(),
                "std": series["std"].tolist(),
            }
            for name, series in latent_alpha_series.items()
        },
        "homogeneity_worst": {
            name: {
                "mean": series["mean"].tolist(),
                "std": series["std"].tolist(),
            }
            for name, series in homogeneity_worst_series.items()
        },
    }
    with open(output / "exp06_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)

    for value, point in zip(p_values, per_seed_points):
        subdir = output / f"point_p_homogeneity={value}"
        subdir.mkdir(exist_ok=True)
        save_exp02_panel_set(
            {name: point[name][0] for name in MODEL_NAMES},
            subdir,
            p=value,
        )

    print("\n=== exp06 summary ===")
    print(f"p values          : {p_values}")
    print(f"alpha_ambient     : {alpha_ambient:.3f}")
    print(f"alpha_ambient / p : {[alpha_ambient / p for p in p_values]}")
    for name in MODEL_NAMES:
        if name in latent_alpha_series:
            means = latent_alpha_series[name]["mean"]
            print(f"  {name:<14} latent alpha_hat = {means.tolist()}")
        if name in homogeneity_worst_series:
            means = homogeneity_worst_series[name]["mean"]
            print(f"  {name:<14} hom. worst       = {means.tolist()}")


if __name__ == "__main__":
    main()
