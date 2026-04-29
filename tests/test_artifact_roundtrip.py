"""Save + reload each of the three model types; verify state round-trips exactly."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from lib.artifacts import (
    load_run_artifact,
    rebuild_model_from_artifact,
    save_run_artifact,
)
from lib.models import HomogeneousAutoencoder, PCABaseline, StandardAutoencoder


@dataclass
class _ConfigSnapshot:
    D: int = 5
    m: int = 2
    hidden_dim: int = 16
    hidden_layers: int = 2
    p_homogeneity: float = 2.0
    extras: dict[str, float] = field(default_factory=dict)


def _assert_state_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys()
    for k, a_val in a.items():
        assert torch.equal(a_val, b[k]), f"state_dict mismatch for {k!r}"


def test_homogeneous_autoencoder_roundtrip(tmp_path: Path) -> None:
    cfg = _ConfigSnapshot()
    model = HomogeneousAutoencoder(
        D=cfg.D, m=cfg.m,
        hidden_dim=cfg.hidden_dim, hidden_layers=cfg.hidden_layers,
        p_homogeneity=cfg.p_homogeneity,
    )
    metrics = {"reconstruction_mse": 0.123, "_latent_radii": [1.0, 2.0, 3.0]}

    path = save_run_artifact(
        tmp_path / "HomogeneousAE.pkl", metrics=metrics, model=model, config=cfg,
    )
    payload = load_run_artifact(path)
    assert payload["metrics"] == metrics

    restored = rebuild_model_from_artifact(payload)
    assert isinstance(restored, HomogeneousAutoencoder)
    _assert_state_equal(model.state_dict(), restored.state_dict())


def test_standard_autoencoder_roundtrip(tmp_path: Path) -> None:
    cfg = _ConfigSnapshot()
    model = StandardAutoencoder(
        D=cfg.D, m=cfg.m,
        hidden_dim=cfg.hidden_dim, hidden_layers=cfg.hidden_layers,
    )
    metrics = {"reconstruction_mse": 0.456}
    path = save_run_artifact(tmp_path / "StandardAE.pkl", metrics=metrics, model=model, config=cfg)

    payload = load_run_artifact(path)
    restored = rebuild_model_from_artifact(payload)
    assert isinstance(restored, StandardAutoencoder)
    _assert_state_equal(model.state_dict(), restored.state_dict())


def test_pca_roundtrip(tmp_path: Path) -> None:
    cfg = _ConfigSnapshot()
    model = PCABaseline(D=cfg.D, m=cfg.m)
    torch.manual_seed(0)
    x = torch.randn(200, cfg.D)
    model.fit(x)

    metrics = {"reconstruction_mse": 0.789}
    path = save_run_artifact(tmp_path / "PCA.pkl", metrics=metrics, model=model, config=cfg)

    payload = load_run_artifact(path)
    restored = rebuild_model_from_artifact(payload)
    assert isinstance(restored, PCABaseline)
    assert restored._fitted
    _assert_state_equal(model.state_dict(), restored.state_dict())
