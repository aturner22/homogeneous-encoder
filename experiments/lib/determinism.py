"""Seed + deterministic-algorithm helpers.

Every ``run.py`` calls :func:`enable_deterministic` near the top to
lock all four RNG sources (``random``, ``numpy``, ``torch``,
``torch.cuda``) and request deterministic kernels from cuDNN. With
``warn_only=True`` the pipeline still runs on ops that have no
deterministic implementation — only a warning is emitted.

CPU runs with these flags set are bit-reproducible across machines
with the same library versions. GPU runs are deterministic to within
the warnings cuDNN emits for the specific ops in the model.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def enable_deterministic(seed: int) -> None:
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        torch.use_deterministic_algorithms(True)
