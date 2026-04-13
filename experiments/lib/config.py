from dataclasses import dataclass, field
from pathlib import Path

import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    n_train: int = 25000
    n_val: int = 5000
    n_test: int = 5000

    batch_size: int = 512
    epochs: int = 500
    patience: int = 50
    learning_rate: float = 1e-3

    hidden_dim: int = 256
    hidden_layers: int = 4

    p_homogeneity: float = 1.0
    lambda_cor: float = 0.1

    num_seeds: int = 1

    device: str = field(default_factory=default_device)
    seed: int = 42
    output_dir: str = "results"


@dataclass
class CurvedSurfaceConfig(TrainConfig):
    student_df_t: float = 2.8
    student_scale_t: float = 1.0
    latent_v_scale: float = 0.9
    latent_correlation: float = 0.75
    angular_scale_1: float = 0.90
    angular_scale_2: float = 0.70
    radial_offset: float = 1.25
    radial_smooth: float = 0.75

    # smaller problem; slightly smaller net and fewer samples are fine
    n_train: int = 20000
    n_val: int = 4000
    n_test: int = 4000
    hidden_dim: int = 192
    hidden_layers: int = 3
    p_homogeneity: float = 2.0
    lambda_cor: float = 0.1


@dataclass
class FlexibleToyConfig(TrainConfig):
    D: int = 10
    m: int = 3
    # alpha=1.8: infinite-variance regime, so StandardAE sees fewer tail
    # samples in training and the separation actually shows up.
    alpha: float = 1.8
    # kappa=1.0: strong curvature so PCA clearly fails reconstruction.
    kappa: float = 1.0
    curvature_rank: int = 8
    embedding_seed: int = 1234

    p_homogeneity: float = 1.0
    lambda_cor: float = 0.1


def ensure_output_dir(config: TrainConfig) -> Path:
    path = Path(config.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
