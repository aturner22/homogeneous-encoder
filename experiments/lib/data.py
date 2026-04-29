"""Toy data generators with analytical tail structure.

Two families of datasets:

1. Curved surface in R^3 for exp01 as an attractive model for visualisation  

2. Flexible-dimension toy manifold in R^D with intrinsic dimension m and
   closed-form tail index alpha, built as

       r ~ Pareto(alpha)
       u ~ Uniform(S^{m-1})
       y = r * u                                  in R^m
       phi(y) = A y + kappa * B @ tanh(C y)       in R^D
       x = phi(y)

   The embedding phi has rank m for small kappa, is smooth and globally
   injective with probability 1 over the random draw of (A, B, C), and is
   asymptotically linear:  phi(lambda y)/lambda -> A y as lambda -> inf.
   Consequently x is regularly varying in R^D with tail index exactly
   alpha.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Curved surface (D = 3, m = 2)
# -----------------------------------------------------------------------------


def _sample_correlated_student_t(
    sample_count: int,
    degrees_of_freedom: float,
    correlation: float,
    scale_t: float,
    scale_v: float,
    rng: np.random.Generator,
) -> np.ndarray:
    covariance = np.array(
        [
            [scale_t ** 2, correlation * scale_t * scale_v],
            [correlation * scale_t * scale_v, scale_v ** 2],
        ],
        dtype=np.float64,
    )
    gaussian = rng.multivariate_normal(
        mean=np.zeros(2, dtype=np.float64), cov=covariance, size=sample_count
    )
    chi_square = rng.chisquare(df=degrees_of_freedom, size=sample_count)
    scaling = np.sqrt(degrees_of_freedom / chi_square)[:, None]
    return gaussian * scaling


def _unit_vector_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def _stereo_inverse_np(z: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(z * z, axis=1, keepdims=True)
    denominator = 1.0 + squared_norm
    x12 = 2.0 * z / denominator
    x3 = (squared_norm - 1.0) / denominator
    return _unit_vector_np(np.concatenate([x12, x3], axis=1))


def _smooth_radial_profile(t: np.ndarray, offset: float, smooth: float) -> np.ndarray:
    return offset + np.sqrt(t * t + smooth * smooth)


def generate_curved_surface(
    sample_count: int,
    seed: int,
    *,
    student_df_t: float = 2.8,
    student_scale_t: float = 1.0,
    latent_v_scale: float = 0.9,
    latent_correlation: float = 0.75,
    angular_scale_1: float = 0.90,
    angular_scale_2: float = 0.70,
    radial_offset: float = 1.25,
    radial_smooth: float = 0.75,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    latent = _sample_correlated_student_t(
        sample_count=sample_count,
        degrees_of_freedom=float(student_df_t),
        correlation=float(latent_correlation),
        scale_t=float(student_scale_t),
        scale_v=float(latent_v_scale),
        rng=rng,
    )
    t = latent[:, 0]
    v = latent[:, 1]

    z1 = float(angular_scale_1) * np.tanh(t)
    z2 = float(angular_scale_2) * np.tanh(v)
    angular_chart = np.stack([z1, z2], axis=1)
    angular_direction = _stereo_inverse_np(angular_chart)
    radial = _smooth_radial_profile(
        t=t, offset=float(radial_offset), smooth=float(radial_smooth)
    )
    ambient = radial[:, None] * angular_direction
    return torch.tensor(ambient, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Flexible toy manifold in R^D with intrinsic dimension m
# -----------------------------------------------------------------------------


class FlexibleToyEmbedding(nn.Module):
    """Fixed smooth embedding phi: R^m -> R^D, phi(y) = A y + kappa B tanh(C y).

    All three matrices are buffers so the embedding is deterministic across
    splits and models when the same `embedding_seed` is used.
    """

    def __init__(
        self,
        D: int,
        m: int,
        kappa: float,
        curvature_rank: int,
        embedding_seed: int,
    ) -> None:
        super().__init__()
        if D <= m:
            raise ValueError(f"Need D > m, got D={D}, m={m}")

        generator = torch.Generator().manual_seed(int(embedding_seed))

        random_matrix = torch.randn(D, m, generator=generator, dtype=torch.float64)
        q_matrix, _ = torch.linalg.qr(random_matrix, mode="reduced")
        linear_frame = q_matrix.to(torch.float32)

        curvature_basis = torch.randn(
            D, curvature_rank, generator=generator, dtype=torch.float32
        ) / float(np.sqrt(curvature_rank))
        curvature_projection = torch.randn(
            curvature_rank, m, generator=generator, dtype=torch.float32
        ) / float(np.sqrt(m))

        self.register_buffer("linear_frame", linear_frame)
        self.register_buffer("curvature_basis", curvature_basis)
        self.register_buffer("curvature_projection", curvature_projection)
        self.kappa = float(kappa)
        self.D = int(D)
        self.m = int(m)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        linear_part = y @ self.linear_frame.T
        curvature_argument = y @ self.curvature_projection.T
        curvature_part = torch.tanh(curvature_argument) @ self.curvature_basis.T
        return linear_part + self.kappa * curvature_part


def _sample_pareto_radius(
    sample_count: int, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """Pareto tail on [1, inf): P(R > t) = t^{-alpha}."""
    uniform_samples = rng.uniform(low=0.0, high=1.0, size=sample_count)
    return (1.0 - uniform_samples) ** (-1.0 / float(alpha))


def _sample_uniform_sphere(
    sample_count: int, m: int, rng: np.random.Generator
) -> np.ndarray:
    raw = rng.standard_normal(size=(sample_count, m))
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norm, 1e-12)


def generate_flexible_toy(
    sample_count: int,
    *,
    D: int,
    m: int,
    alpha: float,
    kappa: float,
    curvature_rank: int,
    embedding_seed: int,
    sample_seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(int(sample_seed))
    r = _sample_pareto_radius(sample_count, alpha, rng)
    u = _sample_uniform_sphere(sample_count, m, rng)
    y_np = r[:, None] * u
    y = torch.tensor(y_np, dtype=torch.float32)

    embedding = FlexibleToyEmbedding(
        D=D,
        m=m,
        kappa=kappa,
        curvature_rank=curvature_rank,
        embedding_seed=embedding_seed,
    )
    with torch.no_grad():
        x = embedding(y)
    return x