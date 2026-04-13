"""Shared training loop.

Device-agnostic. Works for both ``HomogeneousAutoencoder`` and
``StandardAutoencoder`` via the ``loss_fn`` parameter. ``PCABaseline``
uses ``fit_pca_baseline`` instead, which runs a single SVD pass and
returns a history dict compatible with the neural path.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn

from .config import TrainConfig
from .models import (
    HomogeneousAutoencoder,
    PCABaseline,
    StandardAutoencoder,
    homogeneous_loss,
    standard_loss,
)


LossFn = Callable[[torch.Tensor, Dict[str, torch.Tensor], float], Dict[str, torch.Tensor]]


def _select_loss_fn(model: nn.Module) -> LossFn:
    if isinstance(model, HomogeneousAutoencoder):
        return homogeneous_loss
    if isinstance(model, StandardAutoencoder):
        return standard_loss
    raise TypeError(f"No loss function registered for {type(model).__name__}")


def _run_epoch(
    model: nn.Module,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    loss_fn: LossFn,
    *,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    totals: List[float] = []
    recons: List[float] = []
    penalties: List[float] = []

    with torch.set_grad_enabled(train):
        indices = torch.randperm(len(data)) if train else torch.arange(len(data))
        for start in range(0, len(data), config.batch_size):
            batch = data[indices[start : start + config.batch_size]].to(config.device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            forward_pass = model(batch)
            losses = loss_fn(batch, forward_pass, config.lambda_cor)
            if train:
                losses["total"].backward()
                optimizer.step()
            totals.append(float(losses["total"].item()))
            recons.append(float(losses["reconstruction"].item()))
            penalties.append(float(losses["penalty"].item()))

    return {
        "total": float(np.mean(totals)),
        "reconstruction": float(np.mean(recons)),
        "penalty": float(np.mean(penalties)),
    }


def train(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainConfig,
    *,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train a neural autoencoder with Adam and best-val checkpointing."""
    loss_fn = _select_loss_fn(model)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_total": [],
        "train_reconstruction": [],
        "train_penalty": [],
        "val_total": [],
        "val_reconstruction": [],
        "val_penalty": [],
    }

    best_val_total = float("inf")
    best_state = None
    epochs_without_improvement = 0
    patience = int(getattr(config, "patience", 0))

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model, train_data, optimizer, config, loss_fn, train=True
        )
        val_metrics = _run_epoch(
            model, val_data, optimizer, config, loss_fn, train=False
        )

        history["epoch"].append(epoch)
        history["train_total"].append(train_metrics["total"])
        history["train_reconstruction"].append(train_metrics["reconstruction"])
        history["train_penalty"].append(train_metrics["penalty"])
        history["val_total"].append(val_metrics["total"])
        history["val_reconstruction"].append(val_metrics["reconstruction"])
        history["val_penalty"].append(val_metrics["penalty"])

        if val_metrics["total"] < best_val_total:
            best_val_total = val_metrics["total"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == config.epochs):
            print(
                f"  epoch {epoch:3d} | "
                f"train {train_metrics['total']:.5f} "
                f"(recon {train_metrics['reconstruction']:.5f}, "
                f"pen {train_metrics['penalty']:.5f}) | "
                f"val {val_metrics['total']:.5f} "
                f"(recon {val_metrics['reconstruction']:.5f}, "
                f"pen {val_metrics['penalty']:.5f})"
            )

        if patience > 0 and epochs_without_improvement >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (patience {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def fit_pca_baseline(
    model: PCABaseline,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainConfig,
) -> Dict[str, List[float]]:
    """One-shot SVD fit returning a history-dict compatible with train(...)."""
    model.to(config.device)
    with torch.no_grad():
        model.fit(train_data.to(config.device))
        train_forward = model(train_data.to(config.device))
        val_forward = model(val_data.to(config.device))
        train_recon = float(torch.mean((train_data.to(config.device) - train_forward["x_hat"]) ** 2).item())
        val_recon = float(torch.mean((val_data.to(config.device) - val_forward["x_hat"]) ** 2).item())

    return {
        "epoch": [1],
        "train_total": [train_recon],
        "train_reconstruction": [train_recon],
        "train_penalty": [0.0],
        "val_total": [val_recon],
        "val_reconstruction": [val_recon],
        "val_penalty": [0.0],
    }
