from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import PROJECT_ROOT
from src.processing.postprocess import taxonomy_smoothing


def build_ensemble(config_path: str | Path = "config/ensemble.yaml") -> pd.DataFrame:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = config["models"]
    weights = np.array([model["weight"] for model in models], np.float64)
    weights /= weights.sum()
    frames = []
    for model in models:
        submission_path = PROJECT_ROOT / model["submission"]
        frame = pd.read_csv(submission_path).set_index("row_id")
        if not frame.index.is_unique:
            raise ValueError(f"Duplicate row_id in {submission_path}")
        frames.append(frame)
    base_index, base_columns = frames[0].index, frames[0].columns
    for frame in frames[1:]:
        if set(frame.index) != set(base_index) or set(frame.columns) != set(base_columns):
            raise ValueError("Submission schemas do not match")
    blended = sum(
        weight * frame.loc[base_index, base_columns]
        for weight, frame in zip(weights, frames)
    )
    taxonomy_path = Path(config["taxonomy"])
    if not taxonomy_path.is_absolute():
        taxonomy_path = PROJECT_ROOT / taxonomy_path
    blended = taxonomy_smoothing(
        blended,
        pd.read_csv(taxonomy_path),
        config["genus_alpha"],
        config["class_alpha"],
    )
    output = PROJECT_ROOT / config["output"]
    blended.to_csv(output, index=True)
    print(f"Wrote {output} with shape {blended.shape}")
    return blended
