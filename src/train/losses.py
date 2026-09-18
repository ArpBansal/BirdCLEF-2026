from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def build_class_frequency_weights(labels: np.ndarray, cap: float = 10.0) -> Tensor:
    positive_count = labels.sum(axis=0).astype(np.float32) + 1.0
    frequency = positive_count / labels.shape[0]
    weights = np.clip(1.0 / np.sqrt(frequency), 1.0, cap)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def species_focal_loss(
    logits: Tensor,
    targets: Tensor,
    class_weights: Tensor,
    gamma: float = 2.5,
    label_smoothing: float = 0.03,
) -> Tensor:
    targets = targets * (1 - label_smoothing) + label_smoothing / 2.0
    binary_cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probability_true = torch.exp(-binary_cross_entropy)
    focal = (1 - probability_true).pow(gamma) * binary_cross_entropy
    return (focal * class_weights.to(logits.device).unsqueeze(0)).mean()


def focal_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    pos_weight: Tensor | None = None,
) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    probability = torch.sigmoid(logits)
    probability_true = targets * probability + (1 - targets) * (1 - probability)
    return (((1 - probability_true) ** gamma) * bce).mean()


def mixup_files(
    embeddings: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray | None = None,
    alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if alpha <= 0 or len(embeddings) < 2:
        return embeddings, logits, labels, families
    coefficient = np.random.beta(alpha, alpha)
    coefficient = max(coefficient, 1.0 - coefficient)
    permutation = np.random.permutation(len(embeddings))
    mixed_families = (
        coefficient * families + (1 - coefficient) * families[permutation]
        if families is not None
        else None
    )
    return (
        coefficient * embeddings + (1 - coefficient) * embeddings[permutation],
        coefficient * logits + (1 - coefficient) * logits[permutation],
        coefficient * labels + (1 - coefficient) * labels[permutation],
        mixed_families,
    )
