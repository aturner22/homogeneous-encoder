"""ERA5 downloader driven by a yaml request config.

Reads a request yaml that specifies variables, area, time, and
(optionally) pressure levels, then dispatches one cdsapi call per
variable. Writes netCDF into ``climate/data/``. Does nothing on import;
run as a script.

Usage:
    python experiments/climate/download.py --config config.yaml

The yaml format is intentionally a thin wrapper around the raw cdsapi
request so that choices are transparent and reproducible. Example:

    dataset: reanalysis-era5-single-levels
    output_dir: data
    years: [2000, 2001, ..., 2020]
    months: [6, 7, 8]
    days: all
    times: ["00:00", "06:00", "12:00", "18:00"]
    area: [70, -20, 30, 40]     # North, West, South, East (Europe)
    product_type: reanalysis
    format: netcdf
    variables:
      - 2m_temperature
      - total_precipitation
      - 10m_u_component_of_wind
      - 10m_v_component_of_wind
      - mean_sea_level_pressure

The user must create this yaml when the climate choices are made. Until
then, running this script will refuse to proceed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PyYAML is not installed. `pip install pyyaml` or add it to the project deps."
        ) from exc
    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def _expand(values: Any, default: List[str]) -> List[str]:
    if values == "all":
        return default
    if isinstance(values, list):
        return [str(v).zfill(2) if isinstance(v, int) else str(v) for v in values]
    raise ValueError(f"Unsupported yaml value: {values!r}")


def build_requests(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset = config["dataset"]
    variables = config["variables"]
    years = [str(y) for y in config["years"]]
    months = _expand(config.get("months", "all"), [f"{m:02d}" for m in range(1, 13)])
    days = _expand(config.get("days", "all"), [f"{d:02d}" for d in range(1, 32)])
    times = config.get("times", ["00:00", "06:00", "12:00", "18:00"])
    area = config.get("area")
    product_type = config.get("product_type", "reanalysis")
    file_format = config.get("format", "netcdf")
    pressure_levels = config.get("pressure_levels")

    requests: List[Dict[str, Any]] = []
    for variable in variables:
        request: Dict[str, Any] = {
            "dataset": dataset,
            "product_type": product_type,
            "variable": variable,
            "year": years,
            "month": months,
            "day": days,
            "time": times,
            "format": file_format,
        }
        if area is not None:
            request["area"] = area
        if pressure_levels is not None:
            request["pressure_level"] = [str(p) for p in pressure_levels]
        requests.append(request)
    return requests


def _dispatch(requests: List[Dict[str, Any]], output_dir: Path) -> None:
    try:
        import cdsapi  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "cdsapi is not installed. `pip install cdsapi` and ensure ~/.cdsapirc is set up."
        ) from exc

    client = cdsapi.Client()
    output_dir.mkdir(parents=True, exist_ok=True)
    for request in requests:
        dataset = request.pop("dataset")
        variable = request["variable"]
        target = output_dir / f"{variable}.nc"
        print(f"Fetching {variable} -> {target}")
        client.retrieve(dataset, request, str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to request yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded requests without calling cdsapi",
    )
    arguments = parser.parse_args()

    config_path = Path(arguments.config)
    if not config_path.exists():
        raise SystemExit(
            f"Config file {config_path} not found. See experiments/climate/README.md."
        )
    config = _load_yaml(config_path)
    requests = build_requests(config)

    output_dir = Path(__file__).resolve().parent / config.get("output_dir", "data")
    if arguments.dry_run:
        print(f"output_dir = {output_dir}")
        for request in requests:
            print(request)
        return

    _dispatch(requests, output_dir)


if __name__ == "__main__":
    main()
