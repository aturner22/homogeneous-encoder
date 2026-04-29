"""Per-(seed, model) training artifact persistence.

Every experiment produces one pickle per trained model containing
everything needed to regenerate its plots without retraining:

- ``metrics``: the full evaluation dict, including underscore-prefixed
  raw arrays (``_ambient_radii``, ``_latent_radii``, ``_binned_error``,
  ``_history``, ...) that the viz layer consumes.
- ``state_dict``: torch state_dict for neural models, or the fitted
  PCA components tensor for the linear baseline.
- ``model_class``: string name so ``rebuild_model`` can dispatch.
- ``config_snapshot``: dataclasses.asdict of the TrainConfig that was
  in effect, so a reload can reconstruct the architecture.

This module has two responsibilities:

1. Save / load the pickle pair atomically.
2. Rebuild a model instance from a saved state_dict + config.

Pickle format is fine because the payload contains only numpy arrays,
python scalars, and torch state_dicts — no torch.device handles, no
live model objects. Pickle is portable across CPU/GPU machines.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

from .models import (
    HomogeneousAutoencoder,
    PCABaseline,
    StandardAutoencoder,
)

ARTIFACT_VERSION = 1


def _extract_state(model: nn.Module) -> dict[str, Any]:
    """Return a CPU-side state_dict (torch tensors) for any of the three model types."""
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    extra: dict[str, Any] = {}
    if isinstance(model, PCABaseline):
        extra["_fitted"] = bool(model._fitted)
    return {"state_dict": state, "extra": extra}


def save_run_artifact(
    path: Path,
    *,
    metrics: Mapping[str, Any],
    model: nn.Module,
    config: Any | None = None,
) -> Path:
    """Pickle the full training artifact for one (seed, model) to ``path``.

    Creates parent directories. Writes atomically via a ``.tmp`` sidecar
    so a ``KeyboardInterrupt`` mid-write cannot leave a half-written pickle
    behind and mask a missing cache entry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = _extract_state(model)
    config_snapshot: dict[str, Any] | None = None
    if config is not None and is_dataclass(config):
        config_snapshot = asdict(config)

    payload = {
        "version": ARTIFACT_VERSION,
        "model_class": type(model).__name__,
        "state": state,
        "metrics": dict(metrics),
        "config_snapshot": config_snapshot,
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def load_run_artifact(path: Path) -> dict[str, Any]:
    """Load a pickle produced by :func:`save_run_artifact`.

    Returns the raw payload dict. Callers that only need the metrics
    should index ``["metrics"]`` directly.
    """
    path = Path(path)
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if payload.get("version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Artifact at {path} has version {payload.get('version')!r}; "
            f"expected {ARTIFACT_VERSION}. Regenerate with --force-retrain."
        )
    return payload


def rebuild_model_from_artifact(payload: Mapping[str, Any]) -> nn.Module:
    """Reconstruct a model instance from a loaded artifact payload.

    Uses ``config_snapshot`` to recover architecture hyperparameters
    (hidden_dim, hidden_layers, p_homogeneity, D, m) and loads the
    saved state_dict. For PCA the ``_fitted`` flag is restored so
    ``encode`` / ``decode`` don't assert.
    """
    model_class = payload["model_class"]
    snapshot = payload.get("config_snapshot") or {}
    state_block = payload["state"]
    state_dict = state_block["state_dict"]
    extra = state_block.get("extra", {})

    if model_class == "HomogeneousAutoencoder":
        model = HomogeneousAutoencoder(
            D=int(snapshot["D"]),
            m=int(snapshot["m"]),
            hidden_dim=int(snapshot["hidden_dim"]),
            hidden_layers=int(snapshot["hidden_layers"]),
            p_homogeneity=float(snapshot["p_homogeneity"]),
            learnable_centre=bool(snapshot.get("learnable_centre", False)),
        )
    elif model_class == "StandardAutoencoder":
        # The saved config snapshot always refers to the HAE-side hidden_dim;
        # the matched StdAE hidden_dim is implied by the state_dict shape.
        decoder_first_weight = state_dict["decoder.net.0.weight"]
        stdae_hidden = int(decoder_first_weight.shape[0])
        model = StandardAutoencoder(
            D=int(snapshot["D"]),
            m=int(snapshot["m"]),
            hidden_dim=stdae_hidden,
            hidden_layers=int(snapshot["hidden_layers"]),
        )
    elif model_class == "PCABaseline":
        model = PCABaseline(
            D=int(snapshot["D"]),
            m=int(snapshot["m"]),
        )
        model._fitted = bool(extra.get("_fitted", True))
    else:
        raise ValueError(f"Unknown model_class in artifact: {model_class!r}")

    # strict=False so older saved state_dicts (no `centre` buffer) load
    # cleanly; the constructor's default zero centre is the right value
    # in that case.
    model.load_state_dict(state_dict, strict=False)
    return model


def artifact_path(seed_dir: Path, model_name: str) -> Path:
    """Canonical per-(seed, model) pickle path: ``<seed_dir>/<model>.pkl``."""
    return Path(seed_dir) / f"{model_name}.pkl"
