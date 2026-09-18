from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from inference.perch import PerchExtractor
from src.config import load_config
from src.processing.metadata import build_label_matrix, build_soundscape_rows


def taxonomy_groups(taxonomy: pd.DataFrame, labels: list[str]):
    group_column = next(
        (column for column in ("family", "order", "class_name") if column in taxonomy),
        None,
    )
    mapping = (
        taxonomy.set_index("primary_label")[group_column].to_dict()
        if group_column
        else {label: "Unknown" for label in labels}
    )
    groups = sorted({str(value) for value in mapping.values()})
    group_to_index = {group: index for index, group in enumerate(groups)}
    class_to_group = np.array(
        [group_to_index.get(str(mapping.get(label, "Unknown")), 0) for label in labels],
        np.int64,
    )
    return groups, class_to_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the notebook Perch training cache")
    parser.add_argument("--model", choices=["model_22", "model_51"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config or f"config/{args.model}.yaml")
    competition = config["paths"]["competition_dir"]
    sample = pd.read_csv(competition / "sample_submission.csv")
    taxonomy = pd.read_csv(competition / "taxonomy.csv")
    labels = sample.columns[1:].tolist()
    rows = build_soundscape_rows(str(competition / "train_soundscapes_labels.csv"))
    counts = rows.groupby("filename").size()
    full_files = sorted(counts[counts == config["windows_per_file"]].index)
    paths = [competition / "train_soundscapes" / filename for filename in full_files]
    paths = [path for path in paths if path.exists()]
    extractor = PerchExtractor(
        config["paths"]["perch_onnx"],
        config["paths"]["perch_labels"],
        taxonomy,
        labels,
        config["sample_rate"],
        config["windows_per_file"],
        config["window_seconds"],
    )
    metadata, logits, embeddings = extractor.run(paths)
    rows = rows.set_index("row_id").loc[metadata["row_id"]].reset_index()
    targets = build_label_matrix(rows, labels)
    file_count = len(paths)
    windows = config["windows_per_file"]
    site_names = sorted(metadata["site"].astype(str).unique())
    site_to_index = {site: index + 1 for index, site in enumerate(site_names)}
    files = metadata.drop_duplicates("filename")
    site_ids = np.array(
        [
            min(site_to_index.get(str(site), 0), config["proto_ssm"]["n_sites"] - 1)
            for site in files["site"]
        ],
        np.int64,
    )
    hours = files["hour_utc"].to_numpy(np.int64) % 24
    groups, class_to_group = taxonomy_groups(taxonomy, labels)
    file_labels = targets.reshape(file_count, windows, -1).max(axis=1)
    families = np.zeros((file_count, len(groups)), np.float32)
    for class_index, group_index in enumerate(class_to_group):
        families[:, group_index] = np.maximum(
            families[:, group_index], file_labels[:, class_index]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings.reshape(file_count, windows, -1),
        logits=logits.reshape(file_count, windows, -1),
        labels=targets.reshape(file_count, windows, -1),
        site_ids=site_ids,
        hours=hours,
        families=families,
        class_to_family=class_to_group,
        primary_labels=np.asarray(labels),
    )
    print(f"Wrote {args.output}: {file_count} files, {len(labels)} classes")


if __name__ == "__main__":
    main()
