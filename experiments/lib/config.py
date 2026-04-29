import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    """Shared training knobs for every experiment.

    Patience fields control early stopping in two phases:

    - ``recon_patience`` — used while the HAE is in its recon-only
      warmup phase, and as the *sole* patience budget for
      ``StandardAutoencoder`` (which has no penalty phase).
    - ``penalty_patience`` — used once the HAE has exited warmup and
      is training the full objective with the ratcheting correction
      penalty.
    - ``warmup_max_epochs`` — hard cap on the warmup phase. The HAE
      exits warmup at ``min(recon_patience convergence, warmup_max_epochs)``.

    The ``lambda_*`` / ``recon_tolerance`` / ``ratchet_ema`` fields
    drive the post-warmup ratchet and are read only when
    ``adaptive_lambda=True``.
    """

    n_train: int = 25000
    n_val: int = 5000
    n_test: int = 5000

    batch_size: int = 512
    epochs: int = 2000
    recon_patience: int = 100
    penalty_patience: int = 600
    warmup_max_epochs: int = 1000
    learning_rate: float = 1e-3

    hidden_dim: int = 256
    hidden_layers: int = 4

    p_homogeneity: float = 1.0
    lambda_cor: float = 0.1

    # Optional ambient recentring c. The encoder operates on x - c and the
    # decoder re-adds c at the output. When learnable_centre is True the
    # vector is an nn.Parameter trained jointly with the network; when
    # False it is a fixed zero buffer (no recentring).
    learnable_centre: bool = False

    # Optional pre-standardisation of the data marginals to standard Pareto.
    # When True, each ambient coordinate is mapped through a per-margin GPD-tail
    # PIT (rank-based bulk + GPD-fitted tail above pit_threshold_quantile) and
    # then the standard Pareto inverse CDF F^{-1}(u)=(1-u)^{-1/alpha} with
    # alpha=pareto_target_alpha. Train-set fit is reused on val/test.
    pre_standardize_to_pareto_margins: bool = False
    pareto_target_alpha: float = 1.0
    pit_threshold_quantile: float = 0.975
    # Selects the Pareto target family in the inverse-CDF step:
    #   "one_sided" -> standard Pareto on [1, inf), tail index alpha;
    #   "two_sided" -> continuous symmetric Lomax on R, both tails alpha.
    pareto_target_kind: str = "one_sided"

    # Adaptive lambda ratchet (False = fixed lambda, no warmup phase)
    adaptive_lambda: bool = False
    lambda_growth: float = 1.10
    lambda_decay: float = 0.95
    lambda_max: float = 1.0
    recon_tolerance: float = 0.50
    ratchet_ema: float = 0.1
    checkpoint_metric: str = "val_total"

    # Gradient clipping (0 = disabled)
    grad_clip: float = 1.0

    # Regularisation
    weight_decay: float = 0.0

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
    p_homogeneity: float = 1.0
    lambda_cor: float = 0.1

    adaptive_lambda: bool = True
    learnable_centre: bool = True


@dataclass
class CurvedSurfaceParetoMarginsConfig(CurvedSurfaceConfig):
    """Curved-surface variant whose ambient marginals are pre-standardised
    to standard Pareto via a per-margin GPD-tail PIT (Coles 2001 §4).

    Inherits every curved-surface knob from ``CurvedSurfaceConfig`` and
    only flips the pre-standardisation flag on. The PIT threshold and
    target Pareto tail index are configurable via ``TrainConfig``.
    """

    pre_standardize_to_pareto_margins: bool = True


@dataclass
class CurvedSurfaceParetoMarginsTwoSidedConfig(CurvedSurfaceParetoMarginsConfig):
    """Two-sided variant of ``CurvedSurfaceParetoMarginsConfig``.

    Uses a continuous symmetric Lomax target so each ambient coordinate
    keeps its sign (support is ``ℝ`` rather than ``[1, ∞)``), preserving
    the original manifold's geometric symmetry around the origin.
    """

    pareto_target_kind: str = "two_sided"


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

    adaptive_lambda: bool = True
    learnable_centre: bool = False

    # If set, the top fraction (1 - tail_holdout_quantile) of samples by
    # ambient radius becomes the test set; train/val are drawn from the
    # bulk. This gives max(train_radius) < min(test_radius), a strict
    # extrapolation regime for the paper's homogeneity hypothesis.
    tail_holdout_quantile: float | None = None


def ensure_output_dir(config: TrainConfig) -> Path:
    path = Path(config.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(config: TrainConfig, path: Path | None = None) -> Path:
    if path is None:
        path = Path(config.output_dir) / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)
    return path
