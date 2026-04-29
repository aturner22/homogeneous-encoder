"""Shared training loop.

Device-agnostic. Works for both ``HomogeneousAutoencoder`` and
``StandardAutoencoder`` via the ``loss_fn`` parameter. ``PCABaseline``
uses ``fit_pca_baseline`` instead, which runs a single SVD pass and
returns a history dict compatible with the neural path.
"""

from __future__ import annotations

from collections.abc import Callable

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

LossFn = Callable[[torch.Tensor, dict[str, torch.Tensor], float], dict[str, torch.Tensor]]


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
    lambda_cor_override: float | None = None,
) -> dict[str, float]:
    """Run one train/eval pass, returning loss aggregates."""
    model.train(train)
    totals: list[float] = []
    recons: list[float] = []
    penalties: list[float] = []

    effective_lambda = lambda_cor_override if lambda_cor_override is not None else config.lambda_cor

    with torch.set_grad_enabled(train):
        indices = torch.randperm(len(data)) if train else torch.arange(len(data))
        for start in range(0, len(data), config.batch_size):
            batch = data[indices[start : start + config.batch_size]].to(config.device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            forward_pass = model(batch)
            losses = loss_fn(batch, forward_pass, effective_lambda)
            if train:
                losses["total"].backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=config.grad_clip
                    )
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
) -> dict[str, list[float]]:
    """Train a neural autoencoder with Adam and best-val checkpointing."""
    loss_fn = _select_loss_fn(model)
    model.to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate,
        weight_decay=getattr(config, "weight_decay", 0.0),
    )

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_total": [],
        "train_reconstruction": [],
        "train_penalty": [],
        "val_total": [],
        "val_reconstruction": [],
        "val_penalty": [],
        "lambda_eff": [],
    }

    # Checkpoint metric: "val_total" (legacy) or "val_reconstruction"
    checkpoint_key = "reconstruction" if config.checkpoint_metric == "val_reconstruction" else "total"

    # Two-phase schedule for HAE, single phase for StandardAE:
    #   HAE warmup — λ=0. Track best val_reconstruction; exit when it
    #     stops improving for `recon_patience` epochs, or hit
    #     `warmup_max_epochs`. At exit, freeze an EMA baseline.
    #   HAE penalty — ratchet λ up/down against the baseline; early
    #     stop on `penalty_patience` against the checkpoint metric.
    #   StandardAE — no warmup / no penalty. Use `recon_patience` as
    #     its sole early-stopping budget.
    is_hae = isinstance(model, HomogeneousAutoencoder) and config.adaptive_lambda

    in_warmup = is_hae
    lambda_eff = 0.0 if is_hae else config.lambda_cor
    baseline_recon: float | None = None
    ema_recon: float | None = None

    best_val_recon = float("inf")
    best_val_metric = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model, train_data, optimizer, config, loss_fn,
            train=True, lambda_cor_override=lambda_eff,
        )
        val_metrics = _run_epoch(
            model, val_data, optimizer, config, loss_fn,
            train=False, lambda_cor_override=lambda_eff,
        )

        patience_budget = 0
        if in_warmup:
            raw_recon = val_metrics["reconstruction"]
            alpha_ema = config.ratchet_ema
            ema_recon = raw_recon if ema_recon is None else (
                alpha_ema * raw_recon + (1 - alpha_ema) * ema_recon
            )

            if raw_recon < best_val_recon:
                best_val_recon = raw_recon
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            converged = epochs_without_improvement >= config.recon_patience
            hit_cap = epoch >= config.warmup_max_epochs
            if converged or hit_cap:
                baseline_recon = ema_recon
                lambda_eff = 1e-4
                in_warmup = False
                best_val_metric = float("inf")
                epochs_without_improvement = 0
                if verbose:
                    reason = "converged" if converged else "max cap"
                    print(
                        f"  [warmup exit at epoch {epoch} ({reason})] "
                        f"baseline_recon={baseline_recon:.5f}, "
                        f"starting penalty phase"
                    )
        else:
            if is_hae:
                assert baseline_recon is not None and ema_recon is not None
                raw_recon = val_metrics["reconstruction"]
                alpha_ema = config.ratchet_ema
                ema_recon = alpha_ema * raw_recon + (1 - alpha_ema) * ema_recon
                if ema_recon <= baseline_recon * (1 + config.recon_tolerance):
                    lambda_eff = min(config.lambda_max,
                                     lambda_eff * config.lambda_growth)
                else:
                    lambda_eff = lambda_eff * config.lambda_decay
                patience_budget = config.penalty_patience
            else:
                patience_budget = config.recon_patience

            val_checkpoint = val_metrics[checkpoint_key]
            if val_checkpoint < best_val_metric:
                best_val_metric = val_checkpoint
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        history["epoch"].append(epoch)
        history["train_total"].append(train_metrics["total"])
        history["train_reconstruction"].append(train_metrics["reconstruction"])
        history["train_penalty"].append(train_metrics["penalty"])
        history["val_total"].append(val_metrics["total"])
        history["val_reconstruction"].append(val_metrics["reconstruction"])
        history["val_penalty"].append(val_metrics["penalty"])
        history["lambda_eff"].append(lambda_eff)

        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == config.epochs):
            lam_str = f", λ={lambda_eff:.2e}" if is_hae else ""
            print(
                f"  epoch {epoch:3d} | "
                f"train {train_metrics['total']:.5f} "
                f"(recon {train_metrics['reconstruction']:.5f}, "
                f"pen {train_metrics['penalty']:.5f}) | "
                f"val {val_metrics['total']:.5f} "
                f"(recon {val_metrics['reconstruction']:.5f}, "
                f"pen {val_metrics['penalty']:.5f})"
                f"{lam_str}"
            )

        if patience_budget > 0 and epochs_without_improvement >= patience_budget:
            if verbose:
                print(f"  early stopping at epoch {epoch} (patience {patience_budget})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def fit_pca_baseline(
    model: PCABaseline,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainConfig,
) -> dict[str, list[float]]:
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
        "lambda_eff": [0.0],
    }
