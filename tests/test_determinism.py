"""Verify HAE training under enable_deterministic is bit-reproducible on CPU."""

from __future__ import annotations

import torch
from lib.config import TrainConfig
from lib.data import generate_flexible_toy
from lib.determinism import enable_deterministic
from lib.evaluation import train_and_evaluate
from lib.models import HomogeneousAutoencoder


def _train_once() -> float:
    enable_deterministic(42)
    config = TrainConfig(
        n_train=800, n_val=200, n_test=400, batch_size=64,
        epochs=10, recon_patience=50, penalty_patience=50,
        warmup_max_epochs=20, device="cpu", seed=42,
        output_dir="/tmp",  # unused by train_and_evaluate directly
    )
    def _gen(n: int, k: int) -> torch.Tensor:
        return generate_flexible_toy(
            n, D=4, m=2, alpha=2.0, kappa=0.5,
            curvature_rank=2, embedding_seed=1234, sample_seed=config.seed + k,
        )

    train = _gen(config.n_train, 1)
    val = _gen(config.n_val, 2)
    test = _gen(config.n_test, 3)
    model = HomogeneousAutoencoder(
        D=4, m=2, hidden_dim=16, hidden_layers=2, p_homogeneity=1.0,
    )
    metrics = train_and_evaluate(
        model_name="HAE",
        model=model,
        train_data=train,
        val_data=val,
        test_data=test,
        config=config,
        alpha_true=2.0,
        p_for_hill=1.0,
        artifact_path=None,
    )
    return float(metrics["reconstruction_mse"])


def test_hae_training_bit_reproducible() -> None:
    torch.set_num_threads(1)
    mse_a = _train_once()
    mse_b = _train_once()
    assert mse_a == mse_b, f"Deterministic run diverged: {mse_a!r} vs {mse_b!r}"
