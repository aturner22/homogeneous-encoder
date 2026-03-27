#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

@dataclass
class Config:
    n_train: int = 10000
    n_val: int = 2000
    n_test: int = 2000

    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 1e-3

    hidden_dim: int = 128
    hidden_layers: int = 2

    p_homogeneity: float = 2.0
    lambda_sparse: float = 0.1 # key hyperparameter

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    output_dir: str = "results"

    plot_max_points: int = 5000
    plot_radial_percentile: float = 0.99

    student_df_t: float = 2.8
    student_scale_t: float = 1.0
    latent_v_scale: float = 0.9
    latent_correlation: float = 0.75

    angular_scale_1: float = 0.90
    angular_scale_2: float = 0.70

    radial_offset: float = 1.25
    radial_smooth: float = 0.75


def unit_vector(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.linalg.norm(x, dim=dim, keepdim=True) + eps)


def unit_vector_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PHomogeneousAutoencoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.p = config.p_homogeneity
        self.center = nn.Parameter(torch.zeros(3), requires_grad=False)

        self.encoder_angular = MLP(3, 2, config.hidden_dim, config.hidden_layers)
        self.encoder_magnitude = MLP(3, 1, config.hidden_dim, config.hidden_layers)

        self.decoder_angular_base = MLP(2, 3, config.hidden_dim, config.hidden_layers)
        self.decoder_angular_correction = MLP(3, 3, config.hidden_dim, config.hidden_layers)
        self.decoder_magnitude = MLP(2, 1, config.hidden_dim, config.hidden_layers)

    def encode(self, x: torch.Tensor) -> dict:
        y = x - self.center
        r = torch.linalg.norm(y, dim=1)
        u = unit_vector(y, dim=1)
        r_p = r ** self.p

        e = unit_vector(self.encoder_angular(u), dim=1)
        a = torch.nn.functional.softplus(self.encoder_magnitude(u).squeeze(-1))

        w = r_p[:, None] * a[:, None] * e

        return {"w": w, "r": r, "u": u, "r_p": r_p, "e": e, "a": a}

    def decode(self, w: torch.Tensor) -> dict:
        r_w = torch.linalg.norm(w, dim=1)
        r_reconstructed = r_w ** (1.0 / self.p)
        theta = unit_vector(w, dim=1)

        c_base = unit_vector(self.decoder_angular_base(theta), dim=1)

        log_r_w = torch.log(r_w + 1e-8)
        c_correction_input = torch.cat([theta, log_r_w[:, None]], dim=1)
        c_correction_raw = self.decoder_angular_correction(c_correction_input)

        c = unit_vector(c_base + c_correction_raw, dim=1)

        c = unit_vector(c_base + c_correction_raw, dim=1)

        b = torch.nn.functional.softplus(self.decoder_magnitude(theta).squeeze(-1))
        h = b[:, None] * c
        x_hat = self.center + r_reconstructed[:, None] * h

        return {
            "x_hat": x_hat,
            "r_w": r_w,
            "r_reconstructed": r_reconstructed,
            "theta": theta,
            "c_base": c_base,
            "c_correction_raw": c_correction_raw,
            "c": c,
            "b": b,
            "h": h,
        }

    def forward(self, x: torch.Tensor) -> dict:
        encoded = self.encode(x)
        decoded = self.decode(encoded["w"])
        return {**encoded, **decoded}


def compute_loss(x: torch.Tensor, forward_pass: dict, config: Config) -> dict:
    x_hat = forward_pass["x_hat"]
    c_correction_raw = forward_pass["c_correction_raw"]
    radius = forward_pass["r"]

    reconstruction_loss = torch.mean((x - x_hat) ** 2)
    sparsity_per_sample = torch.mean(torch.abs(c_correction_raw), dim=1)
    sparsity_loss = torch.mean(radius * sparsity_per_sample)
    total_loss = reconstruction_loss + config.lambda_sparse * sparsity_loss

    return {
        "total": total_loss,
        "reconstruction": reconstruction_loss,
        "sparsity": sparsity_loss,
    }


def sample_correlated_student_t(
    sample_count: int,
    degrees_of_freedom: float,
    correlation: float,
    scale_t: float,
    scale_v: float,
    rng: np.random.Generator,
) -> np.ndarray:
    covariance_matrix = np.array(
        [
            [scale_t ** 2, correlation * scale_t * scale_v],
            [correlation * scale_t * scale_v, scale_v ** 2],
        ],
        dtype=np.float64,
    )

    gaussian_samples = rng.multivariate_normal(
        mean=np.zeros(2, dtype=np.float64),
        cov=covariance_matrix,
        size=sample_count,
    )

    chi_square_samples = rng.chisquare(df=degrees_of_freedom, size=sample_count)
    scaling = np.sqrt(degrees_of_freedom / chi_square_samples)[:, None]

    return gaussian_samples * scaling


def stereo_inverse_np(z: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(z * z, axis=1, keepdims=True)
    denominator = 1.0 + squared_norm
    x12 = 2.0 * z / denominator
    x3 = (squared_norm - 1.0) / denominator
    return unit_vector_np(np.concatenate([x12, x3], axis=1))


def smooth_radial_profile(t: np.ndarray, radial_offset: float, radial_smooth: float) -> np.ndarray:
    return radial_offset + np.sqrt(t * t + radial_smooth * radial_smooth)


def generate_surface_data(
    sample_count: int,
    config: Config,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)

    latent_samples = sample_correlated_student_t(
        sample_count=sample_count,
        degrees_of_freedom=float(config.student_df_t),
        correlation=float(config.latent_correlation),
        scale_t=float(config.student_scale_t),
        scale_v=float(config.latent_v_scale),
        rng=rng,
    )

    t = latent_samples[:, 0]
    v = latent_samples[:, 1]

    z1 = float(config.angular_scale_1) * np.tanh(t)
    z2 = float(config.angular_scale_2) * np.tanh(v)

    angular_chart = np.stack([z1, z2], axis=1)
    angular_direction = stereo_inverse_np(angular_chart)
    radial_profile = smooth_radial_profile(
        t=t,
        radial_offset=float(config.radial_offset),
        radial_smooth=float(config.radial_smooth),
    )

    ambient_points = radial_profile[:, None] * angular_direction
    return torch.tensor(ambient_points, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: Config,
) -> dict:
    model.train()

    total_losses = []
    reconstruction_losses = []
    sparsity_losses = []

    indices = torch.randperm(len(data))
    for start_index in range(0, len(data), config.batch_size):
        batch_indices = indices[start_index:start_index + config.batch_size]
        batch = data[batch_indices].to(config.device)

        optimizer.zero_grad(set_to_none=True)
        forward_pass = model(batch)
        losses = compute_loss(batch, forward_pass, config)

        losses["total"].backward()
        optimizer.step()

        total_losses.append(losses["total"].item())
        reconstruction_losses.append(losses["reconstruction"].item())
        sparsity_losses.append(losses["sparsity"].item())

    return {
        "total": float(np.mean(total_losses)),
        "reconstruction": float(np.mean(reconstruction_losses)),
        "sparsity": float(np.mean(sparsity_losses)),
    }


@torch.no_grad()
def evaluate(model: nn.Module, data: torch.Tensor, config: Config) -> dict:
    model.eval()

    data_device = data.to(config.device)
    forward_pass = model(data_device)
    losses = compute_loss(data_device, forward_pass, config)

    return {
        "total": float(losses["total"].item()),
        "reconstruction": float(losses["reconstruction"].item()),
        "sparsity": float(losses["sparsity"].item()),
        "rmse": float(torch.sqrt(losses["reconstruction"]).item()),
    }


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)



def select_plot_subset(
    x: np.ndarray,
    max_points: int,
    radial_percentile: float,
) -> np.ndarray:
    radius = np.linalg.norm(x, axis=1)
    radius_threshold = float(np.quantile(radius, radial_percentile))
    valid_indices = np.where(radius <= radius_threshold)[0]
    if len(valid_indices) > max_points:
        valid_indices = valid_indices[:max_points]
    return valid_indices


def plot_training_history(history: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    epoch = np.asarray(history["epoch"])

    axes[0].plot(epoch, history["train_total"], label="train", linewidth=2)
    axes[0].plot(epoch, history["val_total"], label="val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss")
    axes[0].set_title("Total loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epoch, history["train_reconstruction"], label="train", linewidth=2)
    axes[1].plot(epoch, history["val_reconstruction"], label="val", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reconstruction loss")
    axes[1].set_title("Reconstruction")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epoch, history["train_sparsity"], label="train", linewidth=2)
    axes[2].plot(epoch, history["val_sparsity"], label="val", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel(r"Mean $|c_{\mathrm{correction}}|$")
    axes[2].set_title("Sparsity term")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def plot_reconstruction_diagnostics(
    model: nn.Module,
    data: torch.Tensor,
    config: Config,
    output_dir: Path,
    split_name: str,
) -> dict:
    model.eval()
    forward_pass = model(data.to(config.device))

    x = data.cpu().numpy()
    x_hat = forward_pass["x_hat"].cpu().numpy()
    w = forward_pass["w"].cpu().numpy()

    reconstruction_error = np.linalg.norm(x - x_hat, axis=1)
    true_radius = np.linalg.norm(x, axis=1)
    reconstructed_radius = np.linalg.norm(x_hat, axis=1)

    selected_indices = select_plot_subset(
        x=x,
        max_points=config.plot_max_points,
        radial_percentile=config.plot_radial_percentile,
    )

    x_plot = x[selected_indices]
    x_hat_plot = x_hat[selected_indices]
    radius_plot = true_radius[selected_indices]

    figure = plt.figure(figsize=(16, 9))

    axis_1 = figure.add_subplot(2, 3, 1, projection="3d")
    scatter_1 = axis_1.scatter(
        x_plot[:, 0], x_plot[:, 1], x_plot[:, 2],
        c=radius_plot, cmap="viridis", s=8, alpha=0.35, depthshade=False,
    )
    axis_1.set_title(f"Original data ({split_name})")
    axis_1.set_xlabel("x")
    axis_1.set_ylabel("y")
    axis_1.set_zlabel("z")
    figure.colorbar(scatter_1, ax=axis_1, shrink=0.65)

    axis_2 = figure.add_subplot(2, 3, 2, projection="3d")
    scatter_2 = axis_2.scatter(
        x_hat_plot[:, 0], x_hat_plot[:, 1], x_hat_plot[:, 2],
        c=radius_plot, cmap="viridis", s=8, alpha=0.35, depthshade=False,
    )
    axis_2.set_title(f"Reconstruction ({split_name})")
    axis_2.set_xlabel("x")
    axis_2.set_ylabel("y")
    axis_2.set_zlabel("z")
    figure.colorbar(scatter_2, ax=axis_2, shrink=0.65)

    axis_3 = figure.add_subplot(2, 3, 3, projection="3d")
    axis_3.scatter(
        x_plot[:, 0], x_plot[:, 1], x_plot[:, 2],
        s=8, alpha=0.15, depthshade=False, label="x",
    )
    axis_3.scatter(
        x_hat_plot[:, 0], x_hat_plot[:, 1], x_hat_plot[:, 2],
        s=8, alpha=0.15, depthshade=False, label=r"$\hat{x}$",
    )
    axis_3.set_title(f"Original vs reconstruction ({split_name})")
    axis_3.set_xlabel("x")
    axis_3.set_ylabel("y")
    axis_3.set_zlabel("z")
    axis_3.legend()

    axis_4 = figure.add_subplot(2, 3, 4)
    axis_4.hist(reconstruction_error, bins=60, density=True, alpha=0.8, edgecolor="black")
    axis_4.axvline(
        reconstruction_error.mean(),
        linestyle="--",
        linewidth=1.5,
        label=f"mean={reconstruction_error.mean():.4f}",
    )
    axis_4.axvline(
        np.median(reconstruction_error),
        linestyle=":",
        linewidth=1.5,
        label=f"median={np.median(reconstruction_error):.4f}",
    )
    axis_4.set_xlabel(r"$\|x-\hat{x}\|_2$")
    axis_4.set_ylabel("Density")
    axis_4.set_title("Reconstruction error")
    axis_4.grid(True, alpha=0.3)
    axis_4.legend()

    axis_5 = figure.add_subplot(2, 3, 5)
    axis_5.scatter(true_radius, reconstructed_radius, s=8, alpha=0.25)
    lower = float(min(true_radius.min(), reconstructed_radius.min()))
    upper = float(max(true_radius.max(), reconstructed_radius.max()))
    axis_5.plot([lower, upper], [lower, upper], linewidth=1.5)
    axis_5.set_xlabel("True radius")
    axis_5.set_ylabel("Reconstructed radius")
    axis_5.set_title("Radial consistency")
    axis_5.grid(True, alpha=0.3)

    axis_6 = figure.add_subplot(2, 3, 6)
    latent_radius = np.linalg.norm(w, axis=1)
    axis_6.scatter(true_radius, latent_radius, s=8, alpha=0.25)
    axis_6.set_xlabel("True radius")
    axis_6.set_ylabel(r"$\|w\|_2$")
    axis_6.set_title("Latent radius versus ambient radius")
    axis_6.set_yscale("log")
    axis_6.grid(True, alpha=0.3)

    figure.tight_layout()
    output_path = output_dir / f"reconstruction_diagnostics_{split_name}.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return {
        "mean_error": float(reconstruction_error.mean()),
        "median_error": float(np.median(reconstruction_error)),
        "rmse": float(np.sqrt(np.mean(reconstruction_error ** 2))),
        "mean_radial_error": float(np.mean(np.abs(true_radius - reconstructed_radius))),
    }


@torch.no_grad()
def plot_latent_and_homogeneity_diagnostics(
    model: nn.Module,
    data: torch.Tensor,
    config: Config,
    output_dir: Path,
    split_name: str,
) -> dict:
    model.eval()
    forward_pass = model(data.to(config.device))

    x = data.cpu().numpy()
    w = forward_pass["w"].cpu().numpy()
    c_correction_raw = forward_pass["c_correction_raw"].cpu().numpy()

    true_radius = np.linalg.norm(x, axis=1)
    latent_radius = np.linalg.norm(w, axis=1)
    correction_norm = np.linalg.norm(c_correction_raw, axis=1)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))

    scatter_1 = axes[0, 0].scatter(
        w[:, 0], w[:, 1], c=true_radius, cmap="viridis", s=8, alpha=0.35
    )
    axes[0, 0].set_xlabel(r"$w_1$")
    axes[0, 0].set_ylabel(r"$w_2$")
    axes[0, 0].set_title("Latent space")
    axes[0, 0].axis("equal")
    figure.colorbar(scatter_1, ax=axes[0, 0])

    axes[0, 1].scatter(latent_radius, correction_norm, s=8, alpha=0.25)

    latent_radius_quantiles = np.quantile(latent_radius, np.linspace(0.0, 1.0, 16))
    binned_centers = []
    binned_means = []
    for index in range(len(latent_radius_quantiles) - 1):
        lower = latent_radius_quantiles[index]
        upper = latent_radius_quantiles[index + 1]
        if index == len(latent_radius_quantiles) - 2:
            mask = (latent_radius >= lower) & (latent_radius <= upper)
        else:
            mask = (latent_radius >= lower) & (latent_radius < upper)
        if np.any(mask):
            binned_centers.append(0.5 * (lower + upper))
            binned_means.append(float(correction_norm[mask].mean()))

    axes[0, 1].plot(binned_centers, binned_means, linewidth=2)
    axes[0, 1].set_xlabel(r"$\|w\|_2$")
    axes[0, 1].set_ylabel(r"$\|c_{\mathrm{correction}}\|_2$")
    axes[0, 1].set_title("Correction magnitude vs latent radius")
    axes[0, 1].set_xscale("log")
    axes[0, 1].grid(True, alpha=0.3)

    percentile_labels = [50, 75, 90, 95, 99]
    percentile_values = np.percentile(latent_radius, percentile_labels)

    mean_corrections = []
    bar_labels = []

    bulk_mask = latent_radius <= percentile_values[0]
    if np.any(bulk_mask):
        mean_corrections.append(float(correction_norm[bulk_mask].mean()))
        bar_labels.append("≤50%")

    for index in range(1, len(percentile_labels)):
        lower = percentile_values[index - 1]
        upper = percentile_values[index]
        mask = (latent_radius > lower) & (latent_radius <= upper)
        if np.any(mask):
            mean_corrections.append(float(correction_norm[mask].mean()))
            bar_labels.append(f"{percentile_labels[index - 1]}–{percentile_labels[index]}%")

    tail_mask_bar = latent_radius > percentile_values[-1]
    if np.any(tail_mask_bar):
        mean_corrections.append(float(correction_norm[tail_mask_bar].mean()))
        bar_labels.append(">99%")

    axes[1, 0].bar(np.arange(len(mean_corrections)), mean_corrections, alpha=0.8)
    axes[1, 0].set_xticks(np.arange(len(mean_corrections)))
    axes[1, 0].set_xticklabels(bar_labels, rotation=20)
    axes[1, 0].set_ylabel(r"Mean $\|c_{\mathrm{correction}}\|_2$")
    axes[1, 0].set_title("Correction by latent-radius regime")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    x_probe = data[:min(500, len(data))].to(config.device)
    scales = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    homogeneity_errors = []
    encoded_probe = model.encode(x_probe)["w"]

    for scale in scales:
        encoded_scaled_input = model.encode(scale * x_probe)["w"]
        scaled_encoded_input = (scale ** config.p_homogeneity) * encoded_probe
        homogeneity_error = torch.mean((encoded_scaled_input - scaled_encoded_input) ** 2).item()
        homogeneity_errors.append(float(homogeneity_error))

    axes[1, 1].plot(scales, homogeneity_errors, marker="o", linewidth=2)
    axes[1, 1].axhline(1.0e-8, linestyle="--", alpha=0.6)
    axes[1, 1].set_xlabel(r"Scale $\lambda$")
    axes[1, 1].set_ylabel(r"$\|\mathrm{enc}(\lambda x)-\lambda^p \mathrm{enc}(x)\|^2$")
    axes[1, 1].set_title(f"Encoder homogeneity test (p={config.p_homogeneity})")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    figure.tight_layout()
    output_path = output_dir / f"latent_homogeneity_diagnostics_{split_name}.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    bulk_mask_summary = latent_radius <= np.percentile(latent_radius, 50)
    tail_mask_summary = latent_radius >= np.percentile(latent_radius, 95)

    return {
        "mean_sparsity": float(correction_norm.mean()),
        "bulk_sparsity": float(correction_norm[bulk_mask_summary].mean()),
        "tail_sparsity": float(correction_norm[tail_mask_summary].mean()),
        "max_homogeneity_violation": float(np.max(homogeneity_errors)),
    }


@torch.no_grad()
def analyze_and_visualize(
    model: nn.Module,
    data: torch.Tensor,
    config: Config,
    output_dir: Path,
    split_name: str = "test",
) -> dict:
    reconstruction_summary = plot_reconstruction_diagnostics(
        model=model,
        data=data,
        config=config,
        output_dir=output_dir,
        split_name=split_name,
    )

    homogeneity_summary = plot_latent_and_homogeneity_diagnostics(
        model=model,
        data=data,
        config=config,
        output_dir=output_dir,
        split_name=split_name,
    )

    return {
        **reconstruction_summary,
        **homogeneity_summary,
    }


def main():
    config = Config()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Config: {config}")
    print(f"Device: {config.device}")
    print(f"Output directory: {output_dir}\n")

    print("Generating data...")
    train_data = generate_surface_data(config.n_train, config, config.seed + 1)
    val_data = generate_surface_data(config.n_val, config, config.seed + 2)
    test_data = generate_surface_data(config.n_test, config, config.seed + 3)

    print("Creating model...")
    model = PHomogeneousAutoencoder(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    number_of_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {number_of_parameters:,}\n")

    print("Training...")

    history = {
        "epoch": [],
        "train_total": [],
        "train_reconstruction": [],
        "train_sparsity": [],
        "val_total": [],
        "val_reconstruction": [],
        "val_sparsity": [],
    }

    best_validation_total = float("inf")
    best_state_dict = None

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(model, train_data, optimizer, config)
        val_metrics = evaluate(model, val_data, config)

        history["epoch"].append(epoch)
        history["train_total"].append(train_metrics["total"])
        history["train_reconstruction"].append(train_metrics["reconstruction"])
        history["train_sparsity"].append(train_metrics["sparsity"])
        history["val_total"].append(val_metrics["total"])
        history["val_reconstruction"].append(val_metrics["reconstruction"])
        history["val_sparsity"].append(val_metrics["sparsity"])

        if val_metrics["total"] < best_validation_total:
            best_validation_total = val_metrics["total"]
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['total']:.4f} "
                f"(recon: {train_metrics['reconstruction']:.4f}, "
                f"sparse: {train_metrics['sparsity']:.4f}) | "
                f"Val Loss: {val_metrics['total']:.4f} "
                f"(recon: {val_metrics['reconstruction']:.4f}, "
                f"sparse: {val_metrics['sparsity']:.4f})"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print("\nPlotting training history...")
    plot_training_history(history, output_dir)
    print(f"Saved training history to: {output_dir / 'training_history.png'}")

    save_json(output_dir / "history.json", history)
    torch.save(model.state_dict(), output_dir / "model.pt")

    analyze_and_visualize(model, val_data, config, output_dir, "val")
    analyze_and_visualize(model, test_data, config, output_dir, "test")


if __name__ == "__main__":
    main()