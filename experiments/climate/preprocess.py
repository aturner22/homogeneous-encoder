"""Preprocess downloaded ERA5 netCDFs into a (T, D) torch tensor.

This stub supports two vectorisation modes: per-gridpoint multivariate
("one sample per grid cell per time") or regional vectorisation
("one sample per time, D = num_cells * num_variables"). The exact choice
is deferred to the user's later decision (see climate/README.md) and
lives in the yaml config passed to this script.

Usage:
    python experiments/climate/preprocess.py --config config.yaml
    python experiments/climate/preprocess.py --config config.yaml --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is not installed.") from exc
    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def _stack_per_gridpoint(data_arrays: List["xarray.DataArray"]) -> np.ndarray:
    """Each sample = one (time, lat, lon) with variables stacked on last axis."""
    stacked = np.stack([array.values for array in data_arrays], axis=-1)
    return stacked.reshape(-1, stacked.shape[-1])


def _stack_regional(data_arrays: List["xarray.DataArray"]) -> np.ndarray:
    """Each sample = one time step, D = num_gridpoints * num_variables."""
    flattened = [array.values.reshape(array.shape[0], -1) for array in data_arrays]
    return np.concatenate(flattened, axis=1)


def preprocess(config: Dict[str, Any], dry_run: bool) -> None:
    mode = config.get("vectorisation", "per_gridpoint")
    if mode not in ("per_gridpoint", "regional"):
        raise ValueError(f"Unknown vectorisation mode: {mode}")

    data_dir = Path(__file__).resolve().parent / config.get("output_dir", "data")
    output_tensor_path = data_dir / config.get("tensor_name", "era5_tensor.pt")

    if dry_run:
        fake_time = 24
        fake_cells = 50
        num_variables = len(config["variables"])
        if mode == "per_gridpoint":
            fake_tensor = np.random.randn(fake_time * fake_cells, num_variables)
        else:
            fake_tensor = np.random.randn(fake_time, fake_cells * num_variables)
        print(
            f"[dry-run] would write tensor of shape {fake_tensor.shape} "
            f"to {output_tensor_path}"
        )
        return

    try:
        import xarray  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "xarray is not installed. `pip install xarray netCDF4` before real runs."
        ) from exc

    data_arrays = []
    for variable in config["variables"]:
        nc_path = data_dir / f"{variable}.nc"
        if not nc_path.exists():
            raise FileNotFoundError(
                f"Expected netCDF at {nc_path}. Did you run download.py first?"
            )
        dataset = xarray.open_dataset(nc_path)
        data_arrays.append(next(iter(dataset.data_vars.values())))

    if mode == "per_gridpoint":
        stacked = _stack_per_gridpoint(data_arrays)
    else:
        stacked = _stack_regional(data_arrays)

    tensor = torch.tensor(stacked, dtype=torch.float32)
    output_tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_tensor_path)
    print(f"Wrote tensor of shape {tuple(tensor.shape)} to {output_tensor_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    config = _load_yaml(Path(arguments.config))
    preprocess(config, dry_run=arguments.dry_run)


if __name__ == "__main__":
    main()
