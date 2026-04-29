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
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PyYAML is not installed. `pip install pyyaml` or add it to the project deps."
        ) from exc
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


def _expand(values: Any, default: list[str]) -> list[str]:
    if values == "all":
        return default
    if isinstance(values, list):
        return [str(v).zfill(2) if isinstance(v, int) else str(v) for v in values]
    raise ValueError(f"Unsupported yaml value: {values!r}")


def build_requests(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    variables = config["variables"]
    years = [str(y) for y in config["years"]]
    months = _expand(config.get("months", "all"), [f"{m:02d}" for m in range(1, 13)])
    days = _expand(config.get("days", "all"), [f"{d:02d}" for d in range(1, 32)])
    times = config.get("times", ["00:00", "06:00", "12:00", "18:00"])
    area = config.get("area")
    grid = config.get("grid")
    product_type = config.get("product_type", "reanalysis")
    file_format = config.get("format", "netcdf")
    pressure_levels = config.get("pressure_levels")

    batch_size = config.get("year_batch_size", 1)

    requests: list[dict[str, Any]] = []
    for variable in variables:
        for i in range(0, len(years), batch_size):
            batch = years[i : i + batch_size]
            request: dict[str, Any] = {
                "dataset": dataset,
                "product_type": product_type,
                "variable": variable,
                "year": batch,
                "month": months,
                "day": days,
                "time": times,
                "format": file_format,
            }
            if area is not None:
                request["area"] = area
            if grid is not None:
                request["grid"] = grid
            if pressure_levels is not None:
                request["pressure_level"] = [str(p) for p in pressure_levels]
            requests.append(request)
    return requests


def _dispatch(requests: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import cdsapi  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "cdsapi is not installed. `pip install cdsapi` and ensure ~/.cdsapirc is set up."
        ) from exc

    client = cdsapi.Client()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_files: dict[str, list[Path]] = defaultdict(list)

    for request in requests:
        dataset = request.pop("dataset")
        variable = request["variable"]
        years = request["year"]
        label = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"
        target = output_dir / f"{variable}_{label}.nc"
        if target.exists():
            print(f"  [skip] {target.name} already exists")
            batch_files[variable].append(target)
            continue
        print(f"Fetching {variable} {label} -> {target}")
        try:
            client.retrieve(dataset, request, str(target))
        except Exception as exc:  # cdsapi raises several undocumented subclasses
            raise SystemExit(
                f"CDS API request failed for {variable} {label}: {exc}\n"
                f"  - Check ~/.cdsapirc is present and the key is valid "
                f"(see https://cds.climate.copernicus.eu/api-how-to).\n"
                f"  - If the service is rate-limiting, wait and retry; "
                f"already-downloaded batches are cached on disk.\n"
                f"Aborting before merge to avoid partial output."
            ) from exc
        batch_files[variable].append(target)

    # Merge batch files into a single file per variable.
    for variable, paths in batch_files.items():
        if len(paths) <= 1:
            continue
        merged_path = output_dir / f"{variable}.nc"
        if merged_path.exists():
            print(f"  [skip] {merged_path.name} already exists")
            continue
        try:
            import xarray  # type: ignore
        except ImportError:
            print("xarray not installed — skipping merge. Batch files are in", output_dir)
            return
        print(f"Merging {len(paths)} batch files -> {merged_path.name}")
        datasets = [xarray.open_dataset(p) for p in sorted(paths)]
        # ERA5 files use "valid_time" as the time dimension.
        time_dim = "valid_time" if "valid_time" in datasets[0].dims else "time"
        ds = xarray.concat(datasets, dim=time_dim)
        ds.to_netcdf(merged_path)
        for d in datasets:
            d.close()
        ds.close()
        print(f"  wrote {merged_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to request yaml")
    parser.add_argument(
        "--profile",
        default=None,
        help="Named profile under `profiles:` in the yaml to overlay on base config.",
    )
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
    config = _apply_profile(config, arguments.profile)
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
