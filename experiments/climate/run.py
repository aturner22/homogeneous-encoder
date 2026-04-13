"""Train and evaluate the three-model zoo on a preprocessed ERA5 tensor.

Loads a torch tensor produced by ``preprocess.py``, splits it into
train/val/test, and runs the same three-way comparison as ``exp02`` with
the same metrics. Device-agnostic - runs on CPU for smoke tests and
scales to GPU when one is available.

Usage:
    python experiments/climate/run.py --tensor data/era5_tensor.pt
    python experiments/climate/run.py --dry-run     # uses a synthetic tensor

The dry-run path does not touch disk and exists so the pipeline can be
smoke-tested before real ERA5 data is downloaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import FlexibleToyConfig, TrainConfig, ensure_output_dir  # noqa: E402
from lib.data import generate_flexible_toy  # noqa: E402
from lib.evaluation import train_and_evaluate, write_metrics_json  # noqa: E402
from lib.models import (  # noqa: E402
    HomogeneousAutoencoder,
    PCABaseline,
    StandardAutoencoder,
)


def _split(tensor: torch.Tensor, fractions=(0.7, 0.15, 0.15)):
    total = len(tensor)
    generator = torch.Generator().manual_seed(0)
    permutation = torch.randperm(total, generator=generator)
    n_train = int(fractions[0] * total)
    n_val = int(fractions[1] * total)
    train_indices = permutation[:n_train]
    val_indices = permutation[n_train : n_train + n_val]
    test_indices = permutation[n_train + n_val :]
    return tensor[train_indices], tensor[val_indices], tensor[test_indices]


def _load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"No tensor at {path}. Run preprocess.py first.")
    tensor = torch.load(path)
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected torch.Tensor in {path}, got {type(tensor)}")
    return tensor.to(torch.float32)


def _synthetic_tensor() -> torch.Tensor:
    # Small synthetic ERA5-shaped tensor (few thousand samples, D = 8 variables).
    # Re-use the flexible toy generator so the dry-run exercises the same metrics.
    return generate_flexible_toy(
        sample_count=4000,
        D=8,
        m=3,
        alpha=2.5,
        kappa=0.3,
        curvature_rank=6,
        embedding_seed=17,
        sample_seed=17,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensor",
        type=str,
        default="data/era5_tensor.pt",
        help="Path (relative to climate/) to a preprocessed tensor",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a small synthetic tensor instead of loading from disk",
    )
    parser.add_argument(
        "--alpha-assumed",
        type=float,
        default=2.0,
        help="Assumed ambient tail index for evaluation (Hill target); "
             "for real ERA5 you should estimate this from the data first.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (kept low by default for CPU smoke tests)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override; defaults to FlexibleToyConfig.batch_size",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of seeds (only used by multi-seed sweeps; climate runs "
             "single-seed but the field is kept for parity with other experiments).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Width override for the encoder/decoder MLPs",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=None,
        help="Depth override for the encoder/decoder MLPs",
    )
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = root / "results"

    if arguments.dry_run:
        print("[dry-run] using synthetic tensor")
        tensor = _synthetic_tensor()
    else:
        tensor = _load_tensor(root / arguments.tensor)

    print(f"Tensor shape: {tuple(tensor.shape)}")
    D = int(tensor.shape[1])
    m = min(8, max(2, D // 4))

    config_overrides = dict(
        output_dir=str(output_dir),
        D=D,
        m=m,
        alpha=arguments.alpha_assumed,
        epochs=arguments.epochs,
    )
    if arguments.batch_size is not None:
        config_overrides["batch_size"] = arguments.batch_size
    if arguments.num_seeds is not None:
        config_overrides["num_seeds"] = arguments.num_seeds
    if arguments.hidden_dim is not None:
        config_overrides["hidden_dim"] = arguments.hidden_dim
    if arguments.hidden_layers is not None:
        config_overrides["hidden_layers"] = arguments.hidden_layers
    config = FlexibleToyConfig(**config_overrides)
    ensure_output_dir(config)

    train_data, val_data, test_data = _split(tensor)

    homogeneous_model = HomogeneousAutoencoder(
        D=D, m=m,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        p_homogeneity=config.p_homogeneity,
    )
    standard_model = StandardAutoencoder(
        D=D, m=m,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
    )
    pca_model = PCABaseline(D=D, m=m)

    common = dict(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
        alpha_true=config.alpha,
        p_for_hill=config.p_homogeneity,
    )
    metrics_by_model = {
        "HomogeneousAE": train_and_evaluate("HomogeneousAE", homogeneous_model, **common),
        "StandardAE": train_and_evaluate("StandardAE", standard_model, **common),
        "PCA": train_and_evaluate("PCA", pca_model, **common),
    }

    write_metrics_json(metrics_by_model, Path(config.output_dir) / "metrics.json")
    print("Done. Metrics written to", Path(config.output_dir) / "metrics.json")


if __name__ == "__main__":
    main()
