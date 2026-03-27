#!/usr/bin/env python3
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass(frozen=True)
class GlobalConfig:
    seed: int = 37
    output_root: str = "p_homogeneous_autoencoder_runs"
    device: str = "mps" if torch.mps.is_available() else "cpu"
    dtype: str = "float32"

    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0

    q_norm: str = "2"
    p_homogeneity: float = 2.0

    n_train: int = 16000
    n_val: int = 4000
    n_test: int = 4000

    batch_size: int = 1024
    epochs: int = 1000
    learning_rate: float = 1.5e-4
    weight_decay: float = 1.0e-6
    grad_clip_norm: float = 20.0

    hidden_width: int = 256
    hidden_depth: int = 5

    # Loss weights for decomposed architecture
    lambda_angular: float = 1.0  # Angular reconstruction loss weight
    lambda_radial: float = 1.0   # Radial reconstruction loss weight
    lambda_sparse: float = 0.01  # Sparsity penalty on c_correction

    student_df_t: float = 2.8
    student_scale_t: float = 1.0
    latent_v_scale: float = 0.9

    angular_scale_1: float = 0.90
    angular_scale_2: float = 0.70

    radial_offset: float = 1.25
    radial_smooth: float = 0.75

    pair_subset_size: int = 1400
    x_separation_eps: float = 2.0e-2
    z_collision_eps: float = 2.5e-3

    symmetry_probe_count: int = 2000
    n_show: int = 3500
    plot_radial_percentile: float = 0.95  # Filter out extreme radial outliers for plotting

    hill_quantile: float = 0.95
    hill_k_min: int = 150


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    manifold_dim: int
    latent_dim: int  # Angular latent dimension only (NOT full latent size)


def configure_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("p_homogeneous_autoencoder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03dZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, "run.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(obj, file_handle, indent=2, sort_keys=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float64":
        return torch.float64
    return torch.float32


def center_vector(config: GlobalConfig) -> np.ndarray:
    return np.array([config.center_x, config.center_y, config.center_z], dtype=np.float64)


def unit_vector_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def unit_vector_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    return x / torch.clamp(norm, min=eps)


def q_norm_np(x: np.ndarray, q_norm: str) -> np.ndarray:
    if q_norm == "1":
        return np.sum(np.abs(x), axis=-1)
    if q_norm == "2":
        return np.linalg.norm(x, axis=-1)
    if q_norm == "inf":
        return np.max(np.abs(x), axis=-1)
    raise ValueError(f"Unsupported q_norm={q_norm}")


def q_norm_torch(x: torch.Tensor, q_norm: str) -> torch.Tensor:
    if q_norm == "1":
        return torch.sum(torch.abs(x), dim=-1)
    if q_norm == "2":
        return torch.linalg.norm(x, dim=-1)
    if q_norm == "inf":
        return torch.max(torch.abs(x), dim=-1).values
    raise ValueError(f"Unsupported q_norm={q_norm}")


def hill_estimator(samples: np.ndarray, k: int) -> float:
    positive_samples = np.asarray(samples, dtype=np.float64)
    positive_samples = positive_samples[np.isfinite(positive_samples)]
    positive_samples = positive_samples[positive_samples > 0.0]
    if positive_samples.size < 3:
        return float("nan")
    sorted_descending = np.sort(positive_samples)[::-1]
    k = int(min(max(2, k), sorted_descending.size - 1))
    top = sorted_descending[:k]
    kth = sorted_descending[k - 1]
    return float(np.mean(np.log(top) - np.log(kth)))


def sample_student_t(sample_count: int, degrees_of_freedom: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_t(df=degrees_of_freedom, size=sample_count) * scale).astype(np.float64)


def stereo_inverse_np(z: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(z * z, axis=1, keepdims=True)
    denominator = 1.0 + squared_norm
    x12 = 2.0 * z / denominator
    x3 = (squared_norm - 1.0) / denominator
    return unit_vector_np(np.concatenate([x12, x3], axis=1))


def smooth_radial_profile(t: np.ndarray, radial_offset: float, radial_smooth: float) -> np.ndarray:
    return radial_offset + np.sqrt(t * t + radial_smooth * radial_smooth)


def generate_latent_data(
    sample_count: int,
    config: GlobalConfig,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = sample_student_t(
        sample_count=sample_count,
        degrees_of_freedom=float(config.student_df_t),
        scale=float(config.student_scale_t),
        rng=rng,
    )
    v = (float(config.latent_v_scale) * rng.standard_normal(size=sample_count)).astype(np.float64)

    return {"t": t, "v": v}


def manifold_from_latent(
    latent_data: Dict[str, np.ndarray],
    config: GlobalConfig,
) -> Dict[str, np.ndarray]:
    t = latent_data["t"]
    v = latent_data["v"]

    z1 = float(config.angular_scale_1) * np.tanh(t)
    z2 = float(config.angular_scale_2) * np.tanh(v)

    angular_chart = np.stack([z1, z2], axis=1)
    angular_direction = stereo_inverse_np(angular_chart)
    radial_profile = smooth_radial_profile(
        t=t,
        radial_offset=float(config.radial_offset),
        radial_smooth=float(config.radial_smooth),
    )

    center = center_vector(config)
    ambient_points = center[None, :] + radial_profile[:, None] * angular_direction

    return {
        "t": t,
        "v": v,
        "chart": angular_chart,
        "u": angular_direction,
        "rho_euclidean": radial_profile,
        "x": ambient_points,
    }


class ManifoldDataset(torch.utils.data.Dataset):
    def __init__(self, data: Dict[str, np.ndarray], device: torch.device, dtype: torch.dtype) -> None:
        self.x = torch.tensor(data["x"], device=device, dtype=dtype)
        self.u = torch.tensor(data["u"], device=device, dtype=dtype)
        self.t = torch.tensor(data["t"], device=device, dtype=dtype)
        self.v = torch.tensor(data["v"], device=device, dtype=dtype)
        self.chart = torch.tensor(data["chart"], device=device, dtype=dtype)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "u": self.u[index],
            "t": self.t[index],
            "v": self.v[index],
            "chart": self.chart[index],
        }


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_width: int, hidden_depth: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = int(input_dim)
        for _ in range(int(hidden_depth)):
            layers.append(nn.Linear(current_dim, int(hidden_width)))
            layers.append(nn.GELU())
            current_dim = int(hidden_width)
        layers.append(nn.Linear(current_dim, int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FixedRadialAngularAutoencoder(nn.Module):
    """Fixed-radial autoencoder with 2D latent: (angular, radial).

    Full latent representation = angular_latent (learned) + radial_scalar (preserved)
    - angular_latent: learned compression of directions on S^2
    - radial_scalar: preserved distance from center (NOT learned)

    Args:
        latent_dim: Angular latent dimension (NOT full latent size!)
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_width: int,
        hidden_depth: int,
        q_norm: str,
        p_homogeneity: float,
        center: torch.Tensor,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)  # Angular latent dim only
        self.q_norm = str(q_norm)
        self.p_homogeneity = float(p_homogeneity)
        self.register_buffer("center", center.reshape(1, 3))

        # Decomposed encoder: g(u) = a(u) · e(u) where u ∈ S²
        # e: S² → S¹ (angular component - will be normalized)
        self.encoder_angular = MultiLayerPerceptron(
            input_dim=3,
            output_dim=2,
            hidden_width=hidden_width,
            hidden_depth=hidden_depth,
        )
        # a: S² → ℝ⁺ (magnitude scaling component)
        self.encoder_magnitude = MultiLayerPerceptron(
            input_dim=3,
            output_dim=1,
            hidden_width=hidden_width,
            hidden_depth=hidden_depth,
        )

        # Three-component decoder: n(θ, r_w) = b(θ) · [c_base(θ) + c_correction(θ, r_w)]
        # Note: θ ∈ S¹ (1D circle) represented as 2D vector in ℝ²

        # c_base: S¹ → S² (homogeneous baseline angular component)
        self.decoder_angular_base = MultiLayerPerceptron(
            input_dim=2,  # θ (1D circle as 2D vector)
            output_dim=3,
            hidden_width=hidden_width,
            hidden_depth=hidden_depth,
        )

        # c_correction: S¹ × ℝ → ℝ³ (sparse non-homogeneous correction)
        self.decoder_angular_correction = MultiLayerPerceptron(
            input_dim=3,  # [θ (1D circle as 2D vector), log(r_w) (1D scalar)]
            output_dim=3,
            hidden_width=hidden_width,
            hidden_depth=hidden_depth,
        )

        # b: S¹ → ℝ⁺ (magnitude scaling component)
        self.decoder_magnitude = MultiLayerPerceptron(
            input_dim=2,  # θ (1D circle as 2D vector)
            output_dim=1,
            hidden_width=hidden_width,
            hidden_depth=hidden_depth,
        )

    def encode_fixed_radial(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose x into radial + angular components."""
        shifted = x - self.center
        angular_direction = unit_vector_torch(shifted)  # Direction on S^2
        radial_norm = q_norm_torch(shifted, q_norm=self.q_norm)
        radial_scalar = radial_norm ** self.p_homogeneity  # Preserved radial
        return shifted, angular_direction, radial_scalar

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode to product-form latent: w = r^p · z where z = a(u) · e(u).

        Decomposed encoding:
        - u = π(x): direction on S²
        - e(u): S² → S¹ (normalized angular encoding)
        - a(u): S² → ℝ⁺ (magnitude scaling)
        - z = a(u) · e(u): combined angular encoding
        - w = r^p · z: final latent

        Returns:
            w: 2D latent
            r: Original radius ||x - center||
            r_p: r^p (radius raised to p_homogeneity power)
            z: Angular encoding z = a · e
            e: Normalized angular component on S¹
            a: Magnitude scaling
            u: Original direction on S²
        """
        shifted, angular_direction, radial_scalar = self.encode_fixed_radial(x)

        # Decomposed angular encoding: z = a(u) · e(u)
        e_raw = self.encoder_angular(angular_direction)  # S² → ℝ²
        e = unit_vector_torch(e_raw)  # Normalize to S¹
        a = torch.nn.functional.softplus(self.encoder_magnitude(angular_direction).squeeze(-1))  # S² → ℝ⁺

        z = a[:, None] * e  # z = a · e (scalar times unit vector)
        w = radial_scalar[:, None] * z  # w = r^p · z

        return {
            "w": w,
            "r": torch.linalg.norm(shifted, dim=1),  # Original radius
            "r_p": radial_scalar,  # r^p
            "z": z,
            "e": e,
            "a": a,
            "u": angular_direction,
        }

    def decode(self, w: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode from product-form latent w using three-component decoder.

        Three-component decoding:
        1. Extract ||w||^(1/p) (inverted radius)
        2. Extract θ = w/||w|| ∈ S¹ (angular direction on 1D circle, represented as 2D vector)
        3. Decode c_base(θ): S¹ → S² (homogeneous baseline direction)
        4. Decode c_correction(θ, r_w): S¹ × ℝ → ℝ³ (sparse non-homogeneous correction)
        5. Combine: c = normalize(c_base + c_correction)
        6. Decode b(θ): S¹ → ℝ⁺ (angle-dependent magnitude scaling)
        7. h = b(θ) · c: combined output
        8. Reconstruct x = center + ||w||^(1/p) · h

        Note: θ is a point on the 1D circle S¹, represented as a 2D unit vector in ℝ²
        Sparsity penalty on c_correction encourages asymptotic homogeneity

        Args:
            w: 2D latent

        Returns:
            x_hat: Reconstructed ambient point
            r_w: ||w||
            r_reconstructed: ||w||^(1/p) (inverted radius)
            theta: w/||w|| (point on S¹ as 2D vector)
            c: Combined decoded angular direction on S²
            c_base: Homogeneous baseline direction
            c_correction_raw: Sparse correction term (for loss computation)
            b: Decoded magnitude scaling
            h: Combined decoder output b · c
        """
        # Extract radius with p_homogeneity inversion
        r_w = torch.linalg.norm(w, dim=1)
        r_reconstructed = r_w ** (1.0 / self.p_homogeneity)

        # Extract angular direction on S¹
        theta = unit_vector_torch(w)

        # Three-component decoding: h = b(θ) · [c_base(θ) + c_correction(θ, r_w)]

        # 1. Homogeneous baseline: c_base(θ)
        c_base_raw = self.decoder_angular_base(theta)  # S¹ → ℝ³
        c_base = unit_vector_torch(c_base_raw)  # Normalize to S²

        # 2. Sparse non-homogeneous correction: c_correction(θ, r_w)
        log_r_w = torch.log(r_w + 1e-8)  # Add epsilon for numerical stability
        c_corr_input = torch.cat([theta, log_r_w[:, None]], dim=1)
        c_correction_raw = self.decoder_angular_correction(c_corr_input)  # S¹ × ℝ → ℝ³

        # 3. Combine and normalize to S²
        c_combined_raw = c_base + c_correction_raw
        c = unit_vector_torch(c_combined_raw)  # Normalize to S²

        # 4. Angle-dependent magnitude scaling
        b = torch.nn.functional.softplus(self.decoder_magnitude(theta).squeeze(-1))  # S¹ → ℝ⁺

        h = b[:, None] * c  # h = b · c (scalar times unit vector)

        # Reconstruct ambient point
        x_hat = self.center + r_reconstructed[:, None] * h

        return {
            "x_hat": x_hat,
            "r_w": r_w,
            "r_reconstructed": r_reconstructed,
            "theta": theta,
            "c": c,
            "c_base": c_base,
            "c_correction_raw": c_correction_raw,  # For sparsity loss
            "b": b,
            "h": h,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode(x)
        decoded = self.decode(encoded["w"])
        return {
            **encoded,
            **decoded,
        }


def direction_geodesic_squared(u: torch.Tensor, u_hat: torch.Tensor) -> torch.Tensor:
    dot_product = torch.sum(u * u_hat, dim=-1).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    angle = torch.arccos(dot_product)
    return torch.mean(angle * angle)


def latent_variance_penalty(z: torch.Tensor) -> torch.Tensor:
    variance = torch.var(z, dim=0, unbiased=False)
    return torch.mean(1.0 / torch.clamp(variance, min=1.0e-4))



def train_epoch(
    model: FixedRadialAngularAutoencoder,
    data_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    config: GlobalConfig,
) -> Dict[str, float]:
    model.train()

    batch_losses_total: List[float] = []
    batch_losses_angular: List[float] = []
    batch_losses_radial: List[float] = []
    batch_losses_full: List[float] = []
    batch_losses_sparse: List[float] = []

    for batch in data_loader:
        x = batch["x"]
        forward_pass = model(x)

        # Full reconstruction loss
        full_reconstruction = torch.mean(torch.sum((forward_pass["x_hat"] - x) ** 2, dim=1))

        # Angular reconstruction loss: geodesic distance on S² between c and u
        angular_loss = direction_geodesic_squared(forward_pass["u"], forward_pass["c"])

        # Radial reconstruction loss: |r_reconstructed * b - r_original|
        radial_reconstructed = forward_pass["r_reconstructed"] * forward_pass["b"]
        radial_loss = torch.mean((radial_reconstructed - forward_pass["r"]) ** 2)

        # Sparsity loss: L1 penalty on c_correction
        sparsity_loss = torch.mean(torch.abs(forward_pass["c_correction_raw"]))

        # Total loss
        total_loss = (
            full_reconstruction +
            config.lambda_angular * angular_loss +
            config.lambda_radial * radial_loss +
            config.lambda_sparse * sparsity_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.grad_clip_norm))
        optimizer.step()

        batch_losses_total.append(float(total_loss.detach().cpu().item()))
        batch_losses_angular.append(float(angular_loss.detach().cpu().item()))
        batch_losses_radial.append(float(radial_loss.detach().cpu().item()))
        batch_losses_full.append(float(full_reconstruction.detach().cpu().item()))
        batch_losses_sparse.append(float(sparsity_loss.detach().cpu().item()))

    return {
        "total": float(np.mean(batch_losses_total)),
        "angular": float(np.mean(batch_losses_angular)),
        "radial": float(np.mean(batch_losses_radial)),
        "full_reconstruction": float(np.mean(batch_losses_full)),
        "sparse": float(np.mean(batch_losses_sparse)),
    }


@torch.no_grad()
def evaluate_model(
    model: FixedRadialAngularAutoencoder,
    data_loader: torch.utils.data.DataLoader,
    config: GlobalConfig,
) -> Dict[str, float]:
    model.eval()

    losses_total: List[float] = []
    losses_angular: List[float] = []
    losses_radial: List[float] = []
    losses_full: List[float] = []
    losses_sparse: List[float] = []

    for batch in data_loader:
        x = batch["x"]
        forward_pass = model(x)

        # Full reconstruction loss
        full_reconstruction = torch.mean(torch.sum((forward_pass["x_hat"] - x) ** 2, dim=1))

        # Angular reconstruction loss
        angular_loss = direction_geodesic_squared(forward_pass["u"], forward_pass["c"])

        # Radial reconstruction loss
        radial_reconstructed = forward_pass["r_reconstructed"] * forward_pass["b"]
        radial_loss = torch.mean((radial_reconstructed - forward_pass["r"]) ** 2)

        # Sparsity loss
        sparsity_loss = torch.mean(torch.abs(forward_pass["c_correction_raw"]))

        # Total loss
        total_loss = (
            full_reconstruction +
            config.lambda_angular * angular_loss +
            config.lambda_radial * radial_loss +
            config.lambda_sparse * sparsity_loss
        )

        losses_total.append(float(total_loss.cpu().item()))
        losses_angular.append(float(angular_loss.cpu().item()))
        losses_radial.append(float(radial_loss.cpu().item()))
        losses_full.append(float(full_reconstruction.cpu().item()))
        losses_sparse.append(float(sparsity_loss.cpu().item()))

    return {
        "total": float(np.mean(losses_total)),
        "angular": float(np.mean(losses_angular)),
        "radial": float(np.mean(losses_radial)),
        "full_reconstruction": float(np.mean(losses_full)),
        "sparse": float(np.mean(losses_sparse)),
    }


@torch.no_grad()
def collect_outputs(
    model: FixedRadialAngularAutoencoder,
    dataset: ManifoldDataset,
) -> Dict[str, np.ndarray]:
    model.eval()

    x = dataset.x
    forward_pass = model(x)

    return {
        "x": x.detach().cpu().numpy().astype(np.float64),
        "u": forward_pass["u"].detach().cpu().numpy().astype(np.float64),
        "z": forward_pass["z"].detach().cpu().numpy().astype(np.float64),
        "w": forward_pass["w"].detach().cpu().numpy().astype(np.float64),
        "x_hat": forward_pass["x_hat"].detach().cpu().numpy().astype(np.float64),
        "c": forward_pass["c"].detach().cpu().numpy().astype(np.float64),  # Decoded direction on S²
        "r": forward_pass["r"].detach().cpu().numpy().astype(np.float64),
        "r_reconstructed": forward_pass["r_reconstructed"].detach().cpu().numpy().astype(np.float64),
    }


def pairwise_latent_collision_statistics(
    x: np.ndarray,
    z: np.ndarray,
    subset_size: int,
    x_separation_eps: float,
    z_collision_eps: float,
) -> Dict[str, float]:
    subset_size = int(min(subset_size, x.shape[0]))
    x_subset = np.asarray(x[:subset_size], dtype=np.float64)
    z_subset = np.asarray(z[:subset_size], dtype=np.float64)

    x_squared_norms = np.sum(x_subset * x_subset, axis=1, keepdims=True)
    z_squared_norms = np.sum(z_subset * z_subset, axis=1, keepdims=True)

    x_squared_distances = x_squared_norms + x_squared_norms.T - 2.0 * (x_subset @ x_subset.T)
    z_squared_distances = z_squared_norms + z_squared_norms.T - 2.0 * (z_subset @ z_subset.T)

    x_squared_distances = np.maximum(x_squared_distances, 0.0)
    z_squared_distances = np.maximum(z_squared_distances, 0.0)

    upper_triangle_mask = np.triu(np.ones((subset_size, subset_size), dtype=bool), k=1)

    x_distances = np.sqrt(x_squared_distances[upper_triangle_mask])
    z_distances = np.sqrt(z_squared_distances[upper_triangle_mask])

    separated = x_distances > float(x_separation_eps)
    collided = z_distances < float(z_collision_eps)
    dangerous = separated & collided

    return {
        "subset_size": float(subset_size),
        "pair_count": float(x_distances.size),
        "x_distance_mean": float(np.mean(x_distances)),
        "z_distance_mean": float(np.mean(z_distances)),
        "dangerous_pair_count": float(np.sum(dangerous)),
        "dangerous_pair_fraction": float(np.mean(dangerous)),
        "minimum_z_distance_over_separated_pairs": float(np.min(z_distances[separated])) if np.any(separated) else float("nan"),
    }


def symmetry_probe(
    model: FixedRadialAngularAutoencoder,
    config: GlobalConfig,
) -> Dict[str, float]:
    probe_count = int(config.symmetry_probe_count)
    rng = np.random.default_rng(config.seed + 701)

    t = sample_student_t(
        sample_count=probe_count,
        degrees_of_freedom=float(config.student_df_t),
        scale=float(config.student_scale_t),
        rng=rng,
    )
    v = (float(config.latent_v_scale) * rng.standard_normal(size=probe_count)).astype(np.float64)
    v_symmetric = -v

    latent_1 = {"t": t, "v": v}
    latent_2 = {"t": -t, "v": v_symmetric}

    data_1 = manifold_from_latent(latent_1, config)
    data_2 = manifold_from_latent(latent_2, config)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    x_1 = torch.tensor(data_1["x"], device=device, dtype=dtype)
    x_2 = torch.tensor(data_2["x"], device=device, dtype=dtype)

    with torch.no_grad():
        z_1 = model.encode(x_1)["z"].detach().cpu().numpy().astype(np.float64)
        z_2 = model.encode(x_2)["z"].detach().cpu().numpy().astype(np.float64)

    x_distance = np.linalg.norm(data_1["x"] - data_2["x"], axis=1)
    z_distance = np.linalg.norm(z_1 - z_2, axis=1)

    return {
        "probe_count": float(probe_count),
        "mean_x_distance": float(np.mean(x_distance)),
        "mean_z_distance": float(np.mean(z_distance)),
        "median_z_distance": float(np.median(z_distance)),
        "min_z_distance": float(np.min(z_distance)),
        "max_z_distance": float(np.max(z_distance)),
    }


def summarize_split(
    data: Dict[str, np.ndarray],
    outputs: Dict[str, np.ndarray],
    config: GlobalConfig,
) -> Dict[str, float]:
    center = center_vector(config)[None, :]

    raw_q_radius = q_norm_np(data["x"] - center, q_norm=config.q_norm)
    encoded_fixed_radial = outputs["r"]
    encoded_recovered_radius = encoded_fixed_radial ** (1.0 / float(config.p_homogeneity))

    q_threshold = float(np.quantile(raw_q_radius, float(config.hill_quantile)))
    k_value = min(max(int(config.hill_k_min), int(np.sum(raw_q_radius > q_threshold))), raw_q_radius.size - 1)

    x_rmse = float(np.sqrt(np.mean(np.sum((outputs["x_hat"] - data["x"]) ** 2, axis=1))))
    u_dot = np.sum(outputs["u"] * outputs["c"], axis=1)
    u_dot = np.clip(u_dot, -1.0 + 1.0e-10, 1.0 - 1.0e-10)
    angular_error = np.arccos(u_dot)

    summary = {
        "sample_count": float(data["x"].shape[0]),
        "x_rmse": x_rmse,
        "mean_direction_error": float(np.mean(angular_error)),
        "median_direction_error": float(np.median(angular_error)),
        "max_direction_error": float(np.max(angular_error)),
        "mean_log_raw_q_radius": float(np.mean(np.log(raw_q_radius + 1.0e-12))),
        "mean_log_encoded_fixed_radius": float(np.mean(np.log(encoded_recovered_radius + 1.0e-12))),
        "hill_raw_q_radius": float(hill_estimator(raw_q_radius, k_value)),
        "hill_encoded_fixed_radius": float(hill_estimator(encoded_recovered_radius, k_value)),
        "mean_absolute_q_radial_reconstruction_error": float(np.mean(np.abs(raw_q_radius - q_norm_np(outputs["x_hat"] - center, q_norm=config.q_norm)))),
    }

    pairwise_summary = pairwise_latent_collision_statistics(
        x=data["x"],
        z=outputs["z"],
        subset_size=int(config.pair_subset_size),
        x_separation_eps=float(config.x_separation_eps),
        z_collision_eps=float(config.z_collision_eps),
    )

    for key, value in pairwise_summary.items():
        summary[f"pair_{key}"] = float(value)

    return summary


def scatter_3d(axis, x, y, z, c=None, size: float = 2.0, alpha: float = 0.35, title: str = ""):
    scatter = axis.scatter(x, y, z, c=c, s=size, alpha=alpha, depthshade=False)
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_zlabel("x3")
    axis.set_title(title)
    return scatter


def plot_unit_sphere(axis, resolution: int = 28) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    axis.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, linewidth=0.25, alpha=0.15)


def plot_training_history(history: Dict[str, List[float]], output_path: str) -> None:
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(history["epoch"], history["train_loss"], label="train loss")
    axis.plot(history["epoch"], history["val_loss"], label="val loss")
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("reconstruction loss")
    axis.set_title("Training history")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_reconstructions(
    experiment_dir: str,
    split_name: str,
    data: Dict[str, np.ndarray],
    outputs: Dict[str, np.ndarray],
    config: GlobalConfig,
) -> None:
    # Filter by radial distance to avoid extreme outliers stretching the plot
    radial_distances = q_norm_np(data["x"] - center_vector(config)[None, :], q_norm=config.q_norm)
    radial_threshold = float(np.quantile(radial_distances, float(config.plot_radial_percentile)))
    within_threshold = radial_distances <= radial_threshold

    # Apply both radial filter and n_show limit
    valid_indices = np.where(within_threshold)[0]
    n_show = min(int(config.n_show), len(valid_indices))
    selected_indices = valid_indices[:n_show]

    x = data["x"][selected_indices]
    x_hat = outputs["x_hat"][selected_indices]
    z = outputs["z"][selected_indices]
    t = data["t"][selected_indices]

    figure = plt.figure(figsize=(18, 10))

    axis_1 = figure.add_subplot(2, 3, 1, projection="3d")
    scatter = scatter_3d(
        axis_1,
        x[:, 0],
        x[:, 1],
        x[:, 2],
        c=t,
        title=f"Original ambient data ({split_name})",
    )
    figure.colorbar(scatter, ax=axis_1, shrink=0.6)

    axis_2 = figure.add_subplot(2, 3, 2, projection="3d")
    scatter_3d(
        axis_2,
        x_hat[:, 0],
        x_hat[:, 1],
        x_hat[:, 2],
        c=t,
        title=f"Reconstruction ({split_name})",
    )

    axis_3 = figure.add_subplot(2, 3, 3, projection="3d")
    scatter_3d(
        axis_3,
        x[:, 0],
        x[:, 1],
        x[:, 2],
        c=None,
        alpha=0.18,
        title=f"Original vs reconstruction ({split_name})",
    )
    scatter_3d(
        axis_3,
        x_hat[:, 0],
        x_hat[:, 1],
        x_hat[:, 2],
        c=None,
        alpha=0.18,
        title=f"Original vs reconstruction ({split_name})",
    )

    axis_4 = figure.add_subplot(2, 3, 4)
    reconstruction_error = np.linalg.norm(x_hat - x, axis=1)
    axis_4.hist(reconstruction_error, bins=80, density=True, alpha=0.8)
    axis_4.set_xlabel("||x_hat - x||_2")
    axis_4.set_ylabel("density")
    axis_4.set_title("Reconstruction error")

    axis_5 = figure.add_subplot(2, 3, 5)
    axis_5.scatter(t, z[:, 0], s=4.0, alpha=0.35)
    axis_5.set_xlabel("t")
    axis_5.set_ylabel("z")
    axis_5.set_title("Latent coordinate versus heavy-tail variable")

    axis_6 = figure.add_subplot(2, 3, 6)
    raw_q_radius = q_norm_np(x - center_vector(config)[None, :], q_norm=config.q_norm)
    reconstructed_q_radius = q_norm_np(x_hat - center_vector(config)[None, :], q_norm=config.q_norm)
    axis_6.scatter(raw_q_radius, reconstructed_q_radius, s=4.0, alpha=0.35)
    lower = float(min(np.min(raw_q_radius), np.min(reconstructed_q_radius)))
    upper = float(max(np.max(raw_q_radius), np.max(reconstructed_q_radius)))
    axis_6.plot([lower, upper], [lower, upper], linewidth=1.0, alpha=0.8)
    axis_6.set_xlabel("raw q-radius")
    axis_6.set_ylabel("reconstructed q-radius")
    axis_6.set_title("Fixed radial consistency")

    figure.tight_layout()
    output_path = os.path.join(experiment_dir, f"reconstructions_{split_name}.png")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_direction_reconstruction(
    experiment_dir: str,
    split_name: str,
    outputs: Dict[str, np.ndarray],
) -> None:
    n_show = min(3000, outputs["u"].shape[0])
    u = outputs["u"][:n_show]
    u_hat = outputs["c"][:n_show]

    figure = plt.figure(figsize=(12, 5))

    axis_1 = figure.add_subplot(1, 2, 1, projection="3d")
    plot_unit_sphere(axis_1)
    scatter_3d(
        axis_1,
        u[:, 0],
        u[:, 1],
        u[:, 2],
        c=None,
        alpha=0.22,
        title=f"True directions ({split_name})",
    )

    axis_2 = figure.add_subplot(1, 2, 2, projection="3d")
    plot_unit_sphere(axis_2)
    scatter_3d(
        axis_2,
        u_hat[:, 0],
        u_hat[:, 1],
        u_hat[:, 2],
        c=None,
        alpha=0.22,
        title=f"Decoded directions ({split_name})",
    )

    figure.tight_layout()
    output_path = os.path.join(experiment_dir, f"directions_{split_name}.png")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def train_single_experiment(
    experiment_spec: ExperimentSpec,
    global_config: GlobalConfig,
    root_output_dir: str,
    logger: logging.Logger,
) -> Dict:
    experiment_dir = os.path.join(root_output_dir, experiment_spec.name)
    os.makedirs(experiment_dir, exist_ok=True)

    device = torch.device(global_config.device)
    dtype = torch_dtype(global_config.dtype)

    total_sample_count = int(global_config.n_train + global_config.n_val + global_config.n_test)

    latent_data = generate_latent_data(
        sample_count=total_sample_count,
        config=global_config,
        seed=global_config.seed + 1000 + 37 * experiment_spec.manifold_dim + 101 * experiment_spec.latent_dim,
    )
    manifold_data = manifold_from_latent(
        latent_data=latent_data,
        config=global_config,
    )

    train_indices = np.arange(0, global_config.n_train)
    val_indices = np.arange(global_config.n_train, global_config.n_train + global_config.n_val)
    test_indices = np.arange(global_config.n_train + global_config.n_val, total_sample_count)

    train_data = {key: value[train_indices] for key, value in manifold_data.items()}
    val_data = {key: value[val_indices] for key, value in manifold_data.items()}
    test_data = {key: value[test_indices] for key, value in manifold_data.items()}

    train_dataset = ManifoldDataset(train_data, device=device, dtype=dtype)
    val_dataset = ManifoldDataset(val_data, device=device, dtype=dtype)
    test_dataset = ManifoldDataset(test_data, device=device, dtype=dtype)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(global_config.batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(global_config.batch_size),
        shuffle=False,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(global_config.batch_size),
        shuffle=False,
        drop_last=False,
    )

    model = FixedRadialAngularAutoencoder(
        latent_dim=int(experiment_spec.latent_dim),
        hidden_width=int(global_config.hidden_width),
        hidden_depth=int(global_config.hidden_depth),
        q_norm=str(global_config.q_norm),
        p_homogeneity=float(global_config.p_homogeneity),
        center=torch.tensor(center_vector(global_config), device=device, dtype=dtype),
    ).to(device=device, dtype=dtype)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(global_config.learning_rate),
        weight_decay=float(global_config.weight_decay),
    )

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    best_validation_total = float("inf")
    best_state_dict = None

    for epoch in range(1, int(global_config.epochs) + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, global_config)
        val_metrics = evaluate_model(model, val_loader, global_config)

        validation_loss = val_metrics["total"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["total"])
        history["val_loss"].append(validation_loss)

        if validation_loss < best_validation_total:
            best_validation_total = validation_loss
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == int(global_config.epochs):
            logger.info(
                f"[{experiment_spec.name}] "
                f"epoch={epoch:03d} | "
                f"train_total={train_metrics['total']:.6e} | "
                f"val_total={validation_loss:.6e} | "
                f"train_ang={train_metrics['angular']:.6e} | "
                f"val_ang={val_metrics['angular']:.6e} | "
                f"train_rad={train_metrics['radial']:.6e} | "
                f"val_rad={val_metrics['radial']:.6e} | "
                f"train_sparse={train_metrics['sparse']:.6e} | "
                f"val_sparse={val_metrics['sparse']:.6e}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    save_json(os.path.join(experiment_dir, "history.json"), history)
    plot_training_history(history, os.path.join(experiment_dir, "training_history.png"))

    train_outputs = collect_outputs(model, train_dataset)
    val_outputs = collect_outputs(model, val_dataset)
    test_outputs = collect_outputs(model, test_dataset)

    plot_reconstructions(experiment_dir, "train", train_data, train_outputs, global_config)
    plot_reconstructions(experiment_dir, "test", test_data, test_outputs, global_config)
    plot_direction_reconstruction(experiment_dir, "train", train_outputs)
    plot_direction_reconstruction(experiment_dir, "test", test_outputs)

    train_summary = summarize_split(train_data, train_outputs, global_config)
    val_summary = summarize_split(val_data, val_outputs, global_config)
    test_summary = summarize_split(test_data, test_outputs, global_config)
    symmetry_summary = symmetry_probe(model, global_config)

    result = {
        "experiment_spec": asdict(experiment_spec),
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
        "symmetry_probe": symmetry_summary,
        "best_validation_total": best_validation_total,
    }

    save_json(os.path.join(experiment_dir, "summary.json"), result)
    torch.save(model.state_dict(), os.path.join(experiment_dir, "model.pt"))

    logger.info(f"[{experiment_spec.name}] train summary:")
    logger.info(json.dumps(train_summary, indent=2, sort_keys=True))
    logger.info(f"[{experiment_spec.name}] val summary:")
    logger.info(json.dumps(val_summary, indent=2, sort_keys=True))
    logger.info(f"[{experiment_spec.name}] test summary:")
    logger.info(json.dumps(test_summary, indent=2, sort_keys=True))
    logger.info(f"[{experiment_spec.name}] symmetry probe:")
    logger.info(json.dumps(symmetry_summary, indent=2, sort_keys=True))

    return result


def main() -> None:
    global_config = GlobalConfig()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root_output_dir = f"{global_config.output_root}_{timestamp}"
    logger = configure_logging(root_output_dir)

    set_seed(int(global_config.seed))

    logger.info("Global configuration:")
    logger.info(json.dumps(asdict(global_config), indent=2, sort_keys=True))

    experiment_spec = ExperimentSpec(
        name="surface_m2_latent1_compressed",
        manifold_dim=2,
        latent_dim=1,
    )

    logger.info(
        f"Starting experiment {experiment_spec.name} "
        f"(manifold_dim={experiment_spec.manifold_dim}, latent_dim={experiment_spec.latent_dim})."
    )

    result = train_single_experiment(
        experiment_spec=experiment_spec,
        global_config=global_config,
        root_output_dir=root_output_dir,
        logger=logger,
    )

    save_json(os.path.join(root_output_dir, "result.json"), result)
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()