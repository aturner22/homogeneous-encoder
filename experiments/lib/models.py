"""Model implementations.

Three models sharing a common interface:

- ``HomogeneousAutoencoder``: exact paper architecture. The encoder is
  ``p``-homogeneous by construction and the decoder is asymptotically
  homogeneous via the bounded angular correction
  ``delta(eta, rho) = Delta_psi(eta, s(rho)) - Delta_psi(eta, 0)``
  with ``s(rho) = 1/(1+rho)``.

- ``StandardAutoencoder``: deterministic MLP autoencoder with MSE loss
  (no homogeneity, no correction, no penalty). Same hidden sizes as the
  homogeneous model.

- ``PCABaseline``: uncentered top-m SVD. Exactly 1-homogeneous by
  construction, so it perfectly transports regular variation; used as
  the "linear baseline that wins on tails but loses on reconstruction".

All three expose ``encode(x) -> dict``, ``decode(z) -> dict``,
``forward(x) -> dict`` and are trained/evaluated through the shared
training loop in ``train.py``.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def unit_vector(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.linalg.norm(x, dim=dim, keepdim=True) + eps)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# Homogeneous autoencoder (paper's current architecture)
# -----------------------------------------------------------------------------


class HomogeneousAutoencoder(nn.Module):
    """Exact paper architecture.

    Encoder:
        theta = x / ||x||_2
        r     = ||x||_2
        a(theta) = softplus(A_phi(theta))           (scalar)
        e(theta) = normalize(E_phi(theta))          (unit vector in R^m)
        f(x) = r^p * a(theta) * e(theta)

    Decoder:
        rho       = ||z||_2
        eta       = z / rho
        a_tilde   = softplus(A_psi(eta))
        e_tilde   = normalize(E_psi(eta))
        s         = 1 / (1 + rho)
        delta     = Delta_psi(eta, s) - Delta_psi(eta, 0)
        c_tilde   = normalize(e_tilde + delta)
        h(z)      = rho^(1/p) * a_tilde * c_tilde
    """

    def __init__(
        self,
        D: int,
        m: int,
        hidden_dim: int,
        hidden_layers: int,
        p_homogeneity: float,
    ) -> None:
        super().__init__()
        self.D = int(D)
        self.m = int(m)
        self.p = float(p_homogeneity)

        self.A_phi = MLP(D, 1, hidden_dim, hidden_layers)
        self.E_phi = MLP(D, m, hidden_dim, hidden_layers)

        self.A_psi = MLP(m, 1, hidden_dim, hidden_layers)
        self.E_psi = MLP(m, D, hidden_dim, hidden_layers)
        self.Delta_psi = MLP(m + 1, D, hidden_dim, hidden_layers)

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        r = torch.linalg.norm(x, dim=1)
        theta = unit_vector(x, dim=1)
        a = F.softplus(self.A_phi(theta).squeeze(-1))
        e = unit_vector(self.E_phi(theta), dim=1)
        r_p = r ** self.p
        z = r_p[:, None] * a[:, None] * e
        return {"z": z, "r": r, "theta": theta, "a": a, "e": e}

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        rho = torch.linalg.norm(z, dim=1)
        eta = unit_vector(z, dim=1)
        a_tilde = F.softplus(self.A_psi(eta).squeeze(-1))
        e_tilde = unit_vector(self.E_psi(eta), dim=1)

        s = 1.0 / (1.0 + rho)
        delta_input_at_s = torch.cat([eta, s[:, None]], dim=1)
        delta_input_at_zero = torch.cat([eta, torch.zeros_like(s)[:, None]], dim=1)
        delta = self.Delta_psi(delta_input_at_s) - self.Delta_psi(delta_input_at_zero)

        e_plus_delta = e_tilde + delta
        c_tilde = unit_vector(e_plus_delta, dim=1)

        radial_scale = rho ** (1.0 / self.p)
        x_hat = radial_scale[:, None] * a_tilde[:, None] * c_tilde
        return {
            "x_hat": x_hat,
            "rho": rho,
            "eta": eta,
            "a_tilde": a_tilde,
            "e_tilde": e_tilde,
            "delta": delta,
            "c_tilde": c_tilde,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode(x)
        decoded = self.decode(encoded["z"])
        return {**encoded, **decoded}


def homogeneous_loss(
    x: torch.Tensor,
    forward_pass: Dict[str, torch.Tensor],
    lambda_cor: float,
) -> Dict[str, torch.Tensor]:
    x_hat = forward_pass["x_hat"]
    rho = forward_pass["rho"]
    e_tilde = forward_pass["e_tilde"]
    c_tilde = forward_pass["c_tilde"]

    reconstruction = torch.mean((x - x_hat) ** 2)
    weight = torch.log1p(rho)
    per_sample_penalty = torch.linalg.norm(c_tilde - e_tilde, dim=1)
    penalty = torch.mean(weight * per_sample_penalty)
    total = reconstruction + lambda_cor * penalty
    return {"total": total, "reconstruction": reconstruction, "penalty": penalty}


# -----------------------------------------------------------------------------
# Standard deterministic autoencoder baseline
# -----------------------------------------------------------------------------


class StandardAutoencoder(nn.Module):
    def __init__(
        self,
        D: int,
        m: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        self.D = int(D)
        self.m = int(m)
        self.encoder = MLP(D, m, hidden_dim, hidden_layers)
        self.decoder = MLP(m, D, hidden_dim, hidden_layers)

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {"z": z}

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_hat = self.decoder(z)
        return {"x_hat": x_hat}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode(x)
        decoded = self.decode(encoded["z"])
        return {**encoded, **decoded}


def standard_loss(
    x: torch.Tensor,
    forward_pass: Dict[str, torch.Tensor],
    lambda_cor: float,
) -> Dict[str, torch.Tensor]:
    del lambda_cor  # kept for signature parity with homogeneous_loss
    reconstruction = torch.mean((x - forward_pass["x_hat"]) ** 2)
    zero = torch.zeros((), device=reconstruction.device)
    return {"total": reconstruction, "reconstruction": reconstruction, "penalty": zero}


# -----------------------------------------------------------------------------
# PCA baseline
# -----------------------------------------------------------------------------


class PCABaseline(nn.Module):
    """Uncentered top-m PCA as a model with the same interface as the neural nets.

    Uncentered means encode is a pure linear map ``x -> x @ V``, so the
    baseline is *exactly* 1-homogeneous and therefore exactly transports
    regular variation by Proposition 1 of the paper. This is the cleanest
    linear tail-preserving baseline.
    """

    def __init__(self, D: int, m: int) -> None:
        super().__init__()
        self.D = int(D)
        self.m = int(m)
        self.register_buffer("V", torch.zeros(D, m))
        self._fitted = False

    def fit(self, x: torch.Tensor) -> None:
        x = x.to(self.V.device)
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        self.V.copy_(vh[: self.m].T)
        self._fitted = True

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert self._fitted, "PCABaseline.fit must be called before encode"
        z = x @ self.V
        return {"z": z}

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert self._fitted, "PCABaseline.fit must be called before decode"
        x_hat = z @ self.V.T
        return {"x_hat": x_hat}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode(x)
        decoded = self.decode(encoded["z"])
        return {**encoded, **decoded}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_matched_hidden_dim(
    target_params: int, D: int, m: int, hidden_layers: int
) -> int:
    """Compute the StdAE hidden_dim whose total params match *target_params*.

    StdAE has two MLPs — encoder(D→m) and decoder(m→D) — each with the
    same hidden_dim *h* and *hidden_layers* layers.  Total parameters:

        2*(L-1)*h^2 + (2*D + 2*m + 2*L)*h + (D + m)

    We solve the quadratic in *h* and round to the nearest integer.
    """
    import math

    L = hidden_layers
    a = 2 * (L - 1)
    b = 2 * D + 2 * m + 2 * L
    c = (D + m) - target_params
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        raise ValueError(
            f"No solution for matched hidden_dim: discriminant={discriminant}"
        )
    h = (-b + math.sqrt(discriminant)) / (2 * a)
    return max(1, int(round(h)))
