"""Shared CLI scaffolding for the experiment drivers.

Every ``experiments/exp0*/run.py`` has the same opening block: parse
``--plot-only`` / ``--force-retrain`` flags, resolve the ``results/``
output directory next to the script, instantiate the experiment's
config dataclass, write ``config.json``, lock in determinism, and
print the resolved config to stdout. This module centralises that
pattern so the drivers can skip straight to the experiment logic.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

from .config import TrainConfig, ensure_output_dir, save_config
from .determinism import enable_deterministic


def parse_standard_args(
    description: str | None = None,
    extra: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the caching + smoke-test flags every driver accepts.

    - ``--plot-only`` loads cached artifacts and regenerates plots.
    - ``--force-retrain`` ignores the cache and retrains everything.
    - ``--dry-run`` is a smoke-test escape hatch; drivers that don't
      implement it simply ignore ``args.dry_run``.

    Pass ``extra`` to register additional experiment-specific flags on
    the same parser (e.g. exp02's ``--num-seeds`` / ``--output-subdir``
    or climate's ``--config`` / ``--var``).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Load cached artifacts and regenerate plots only.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cached artifacts and retrain all models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with a small synthetic tensor; skip real data loading.",
    )
    if extra is not None:
        extra(parser)
    return parser.parse_args(argv)


def init_experiment[C: TrainConfig](
    script_path: Path,
    config_cls: type[C],
    *,
    subdir: str | None = None,
    **config_kwargs: Any,
) -> C:
    """Build + persist the experiment's config and prime determinism.

    - ``script_path`` is typically ``Path(__file__)``; the results
      directory lands at ``<script_path.parent>/results[/subdir]``.
    - ``config_kwargs`` override dataclass defaults.
    - Returns the fully-constructed config dataclass instance.
    """
    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls.__name__} is not a dataclass")

    results_root = Path(script_path).resolve().parent / "results"
    output_dir = results_root / subdir if subdir else results_root
    config = config_cls(output_dir=str(output_dir), **config_kwargs)

    ensure_output_dir(config)
    save_config(config)
    enable_deterministic(config.seed)

    print(f"Config: {config}")
    print(f"Device: {config.device}")
    return config
