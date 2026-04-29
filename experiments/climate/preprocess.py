"""Preprocess downloaded ERA5 netCDFs into (N, D) torch tensors.

Supports regional vectorisation: each sample is one time step, D equals
num_gridpoints * num_variables. Also writes per-variable tensors for
univariate runs.

Pipeline: load netCDFs → jitter precip zeros → stack regionally →
drop NaN rows → z-score standardise → shuffle → save tensors + metadata.

Usage:
    python experiments/climate/preprocess.py --config config.yaml
    python experiments/climate/preprocess.py --config config.yaml --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is not installed.") from exc
    with open(path, encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def _apply_profile(config: dict[str, Any], profile: str | None) -> dict[str, Any]:
    """Overlay ``config['profiles'][profile]`` onto the base config.

    Returns a shallow-merged copy with the ``profiles`` block stripped.
    Keys absent from the selected profile keep their base value; keys
    present in both are taken from the profile.
    """
    profiles = config.pop("profiles", {}) or {}
    if profile is None:
        return config
    if profile not in profiles:
        raise SystemExit(
            f"Profile {profile!r} not defined in config. "
            f"Available: {sorted(profiles) or '(none)'}."
        )
    merged = dict(config)
    merged.update(profiles[profile])
    return merged


def _jitter_zeros(array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add tiny positive noise to exact zeros so ‖x‖ > 0 everywhere."""
    positive = array[array > 0]
    if positive.size == 0:
        eps = 1e-8
    else:
        eps = 1e-6 * float(np.nanmedian(positive))
    mask = array == 0.0
    array[mask] = rng.uniform(0.0, eps, size=int(mask.sum()))
    return array


def _stack_regional(data_arrays: list[np.ndarray]) -> np.ndarray:
    """Each sample = one time step, D = num_gridpoints * num_variables."""
    flattened = [arr.reshape(arr.shape[0], -1) for arr in data_arrays]
    return np.concatenate(flattened, axis=1)


def _short_name(variable: str) -> str:
    """Map CDS variable name to a short filename-friendly label."""
    mapping = {
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "total_precipitation": "tp",
        "2m_temperature": "t2m",
        "mean_sea_level_pressure": "msp",
    }
    return mapping.get(variable, variable.replace(" ", "_")[:10])


def preprocess(config: dict[str, Any], dry_run: bool) -> None:
    variables = config["variables"]
    n_vars = len(variables)
    data_dir = Path(__file__).resolve().parent / config.get("output_dir", "data")
    output_tensor_path = data_dir / config.get("tensor_name", "era5_tensor.pt")
    rng = np.random.default_rng(config.get("sample_seed", 42))

    if dry_run:
        fake_time = 100
        fake_cells = 221
        fake_arrays = [np.random.randn(fake_time, fake_cells) for _ in variables]
        stacked = _stack_regional(fake_arrays)
        print(
            f"[dry-run] joint tensor shape: {stacked.shape}  "
            f"(T={fake_time}, D={fake_cells * n_vars})"
        )
        for var_name in config.get("univariate_runs", []):
            print(f"[dry-run] univariate {_short_name(var_name)} shape: "
                  f"({fake_time}, {fake_cells})")
        return

    try:
        import xarray  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "xarray is not installed. `pip install xarray netCDF4` before real runs."
        ) from exc

    # Load raw arrays and extract grid metadata from the first variable.
    raw_arrays: list[np.ndarray] = []
    lats: np.ndarray | None = None
    lons: np.ndarray | None = None

    for variable in variables:
        nc_path = data_dir / f"{variable}.nc"
        if not nc_path.exists():
            raise FileNotFoundError(
                f"Expected netCDF at {nc_path}. Did you run download.py first?"
            )
        dataset = xarray.open_dataset(nc_path)
        data_var = next(iter(dataset.data_vars.values()))
        arr = data_var.values  # shape (T, lat, lon) or similar

        # Extract lat/lon from the first variable.
        if lats is None:
            for lat_name in ("latitude", "lat"):
                if lat_name in dataset.coords:
                    lats = dataset.coords[lat_name].values
                    break
            for lon_name in ("longitude", "lon"):
                if lon_name in dataset.coords:
                    lons = dataset.coords[lon_name].values
                    break

        # Jitter precipitation zeros.
        if config.get("precip_jitter") and "precipitation" in variable:
            arr = _jitter_zeros(arr.copy(), rng)

        # log1p for one-sided variables so the downstream z-score covers
        # reals symmetrically instead of a skewed half-line.
        if variable in config.get("log_transform_variables", []):
            arr = np.log1p(arr)
            print(f"Applied log1p to {variable}")

        raw_arrays.append(arr)
        dataset.close()

    n_cells = raw_arrays[0][0].size  # spatial cells per time step
    print(f"Loaded {n_vars} variables, {raw_arrays[0].shape[0]} time steps, "
          f"{n_cells} grid cells each")

    # Stack: (T, n_cells * n_vars)
    stacked = _stack_regional(raw_arrays)
    print(f"Stacked shape: {stacked.shape}")

    # Drop NaN rows.
    nan_mask = np.isnan(stacked).any(axis=1)
    n_dropped = int(nan_mask.sum())
    if n_dropped > 0:
        stacked = stacked[~nan_mask]
        print(f"Dropped {n_dropped} rows with NaN, {stacked.shape[0]} remaining")

    # Per-column z-score standardisation.
    if config.get("standardise", False):
        col_mean = stacked.mean(axis=0)
        col_std = stacked.std(axis=0)
        col_std[col_std == 0] = 1.0  # avoid division by zero for constant columns
        stacked = (stacked - col_mean) / col_std
        stats_path = data_dir / "era5_stats.npz"
        np.savez(stats_path, mean=col_mean, std=col_std)
        print(f"Standardised; saved stats to {stats_path}")

    # Shuffle rows (temporal order doesn't matter for ergodic distribution).
    rng.shuffle(stacked)

    # Save joint tensor.
    data_dir.mkdir(parents=True, exist_ok=True)
    tensor = torch.tensor(stacked, dtype=torch.float32)
    torch.save(tensor, output_tensor_path)
    print(f"Wrote joint tensor {tuple(tensor.shape)} to {output_tensor_path}")

    # Save per-variable univariate tensors.
    # ``univariate_suffix`` (default empty) gets appended before the .pt
    # extension so profiles can emit alternate variants without
    # overwriting the canonical era5_<short>.pt files.
    univariate_suffix = config.get("univariate_suffix", "") or ""
    var_index = {name: i for i, name in enumerate(variables)}
    for uni_var in config.get("univariate_runs", []):
        idx = var_index.get(uni_var)
        if idx is None:
            print(f"Warning: univariate variable {uni_var} not in variables list, skipping")
            continue
        col_start = idx * n_cells
        col_end = (idx + 1) * n_cells
        uni_tensor = tensor[:, col_start:col_end]
        short = _short_name(uni_var)
        uni_path = data_dir / f"era5_{short}{univariate_suffix}.pt"
        torch.save(uni_tensor, uni_path)
        print(f"Wrote univariate {short} tensor {tuple(uni_tensor.shape)} to {uni_path}")

    # Save grid metadata for spatial plotting.
    meta_path = data_dir / "era5_meta.npz"
    np.savez(
        meta_path,
        lats=lats if lats is not None else np.array([]),
        lons=lons if lons is not None else np.array([]),
        var_names=np.array([_short_name(v) for v in variables]),
        n_cells=np.array(n_cells),
    )
    print(f"Wrote grid metadata to {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--profile",
        default=None,
        help="Named profile under `profiles:` in the yaml to overlay on base config.",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    config = _load_yaml(Path(arguments.config))
    config = _apply_profile(config, arguments.profile)
    preprocess(config, dry_run=arguments.dry_run)


if __name__ == "__main__":
    main()
