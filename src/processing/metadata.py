from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .audio import soundscape_metadata


def primary_labels(sample_submission: pd.DataFrame) -> list[str]:
    return sample_submission.columns[1:].tolist()


def build_label_matrix(
    rows: pd.DataFrame, labels: list[str], label_column: str = "label_list"
) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(rows), len(labels)), dtype=np.uint8)
    for row_index, row_labels in enumerate(rows[label_column]):
        for label in row_labels:
            if label in label_to_index:
                matrix[row_index, label_to_index[label]] = 1
    return matrix


def union_labels(values: Iterable[object]) -> list[str]:
    labels: set[str] = set()
    for value in values:
        if pd.notna(value):
            labels.update(part.strip() for part in str(value).split(";") if part.strip())
    return sorted(labels)


def build_soundscape_rows(labels_csv: str) -> pd.DataFrame:
    raw = pd.read_csv(labels_csv)
    rows = (
        raw.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    rows["end_sec"] = pd.to_timedelta(rows["end"]).dt.total_seconds().astype(int)
    rows["row_id"] = (
        rows["filename"].str.replace(".ogg", "", regex=False)
        + "_"
        + rows["end_sec"].astype(str)
    )
    parsed = rows["filename"].apply(soundscape_metadata).apply(pd.Series)
    return pd.concat([rows, parsed.drop(columns="filename")], axis=1)
