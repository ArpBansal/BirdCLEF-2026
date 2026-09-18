"""Probability transformations used by the active notebook branches."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def file_confidence_scale(
    probabilities: np.ndarray, n_windows: int = 12, top_k: int = 2, power: float = 0.4
) -> np.ndarray:
    view = probabilities.reshape(-1, n_windows, probabilities.shape[1])
    top = np.sort(view, axis=1)[:, -top_k:, :].mean(axis=1)
    scale = np.power(np.clip(top, 1e-7, 1.0), power)
    return (view * scale[:, None, :]).reshape(probabilities.shape)


def rank_aware_scaling(
    probabilities: np.ndarray, n_windows: int = 12, power: float = 0.5
) -> np.ndarray:
    view = probabilities.reshape(-1, n_windows, probabilities.shape[1])
    file_max = view.max(axis=1, keepdims=True)
    return (view * np.power(file_max, power)).reshape(probabilities.shape)


def adaptive_delta_smooth(
    probabilities: np.ndarray, n_windows: int = 12, base_alpha: float = 0.20
) -> np.ndarray:
    result = probabilities.copy()
    view = probabilities.reshape(-1, n_windows, probabilities.shape[1])
    output = result.reshape(view.shape)
    for index in range(n_windows):
        confidence = view[:, index, :].max(axis=-1, keepdims=True)
        alpha = base_alpha * (1.0 - confidence)
        if index == 0:
            neighbor = (view[:, index, :] + view[:, index + 1, :]) / 2.0
        elif index == n_windows - 1:
            neighbor = (view[:, index - 1, :] + view[:, index, :]) / 2.0
        else:
            neighbor = (view[:, index - 1, :] + view[:, index + 1, :]) / 2.0
        output[:, index, :] = (1.0 - alpha) * view[:, index, :] + alpha * neighbor
    return result


def apply_per_class_thresholds(
    scores: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    scaled = scores.copy()
    for class_index, threshold in enumerate(thresholds):
        above = scores[:, class_index] > threshold
        scaled[above, class_index] = 0.5 + 0.5 * (
            scores[above, class_index] - threshold
        ) / (1 - threshold + 1e-8)
        scaled[~above, class_index] = (
            0.5 * scores[~above, class_index] / (threshold + 1e-8)
        )
    return np.clip(scaled, 0.0, 1.0)


def taxonomy_smoothing(
    submission: pd.DataFrame,
    taxonomy: pd.DataFrame,
    genus_alpha: float = 0.15,
    class_alpha: float = 0.05,
) -> pd.DataFrame:
    """The notebook's v221 genus pass followed by class pass."""
    species_to_genus: dict[str, str] = {}
    species_to_class: dict[str, str] = {}
    for _, row in taxonomy.iterrows():
        label = str(row["primary_label"])
        scientific_name = str(row.get("scientific_name", ""))
        species_to_genus[label] = scientific_name.split(" ")[0]
        species_to_class[label] = str(row.get("class_name", ""))

    columns = list(submission.columns)
    genus_groups: dict[str, list[str]] = {}
    class_groups: dict[str, list[str]] = {}
    for column in columns:
        genus_groups.setdefault(species_to_genus.get(column, column), []).append(column)
        class_name = species_to_class.get(column, "")
        if class_name:
            class_groups.setdefault(class_name, []).append(column)

    probabilities = submission.to_numpy(np.float32, copy=True)
    column_index = {column: index for index, column in enumerate(columns)}
    for members in genus_groups.values():
        if len(members) > 1:
            indices = [column_index[member] for member in members]
            mean = probabilities[:, indices].mean(axis=1, keepdims=True)
            probabilities[:, indices] = (
                (1 - genus_alpha) * probabilities[:, indices] + genus_alpha * mean
            )
    for members in class_groups.values():
        if len(members) > 1:
            indices = [column_index[member] for member in members]
            mean = probabilities[:, indices].mean(axis=1, keepdims=True)
            probabilities[:, indices] = (
                (1 - class_alpha) * probabilities[:, indices] + class_alpha * mean
            )
    return pd.DataFrame(probabilities, index=submission.index, columns=columns)
