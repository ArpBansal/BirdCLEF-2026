from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.processing.audio import read_soundscape, soundscape_metadata


class PerchExtractor:
    """ONNX Perch v2 inference and BirdCLEF taxonomy mapping."""

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        taxonomy: pd.DataFrame,
        primary_labels: list[str],
        sample_rate: int = 32_000,
        windows_per_file: int = 12,
        window_seconds: int = 5,
    ) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.sample_rate = sample_rate
        self.windows_per_file = windows_per_file
        self.window_seconds = window_seconds
        self.primary_labels = primary_labels

        perch_labels = pd.read_csv(labels_path).reset_index().rename(
            columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}
        )
        if "scientific_name" not in perch_labels:
            for candidate in ("label", "labels", "name"):
                if candidate in perch_labels:
                    perch_labels = perch_labels.rename(columns={candidate: "scientific_name"})
                    break
        mapping = taxonomy.merge(
            perch_labels[["bc_index", "scientific_name"]], on="scientific_name", how="left"
        )
        missing_index = len(perch_labels)
        mapping["bc_index"] = mapping["bc_index"].fillna(missing_index).astype(int)
        label_to_perch = mapping.set_index("primary_label")["bc_index"]
        indices = np.array(
            [int(label_to_perch.get(label, missing_index)) for label in primary_labels],
            dtype=np.int32,
        )
        self.mapped_mask = indices != missing_index
        self.mapped_positions = np.where(self.mapped_mask)[0].astype(np.int32)
        self.mapped_perch_indices = indices[self.mapped_mask]

        class_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
        self.proxy_map: dict[int, np.ndarray] = {}
        for position in np.where(~self.mapped_mask)[0]:
            label = primary_labels[position]
            row = taxonomy[taxonomy["primary_label"] == label]
            if row.empty or class_map.get(label) not in {"Amphibia", "Insecta", "Aves"}:
                continue
            genus = str(row.iloc[0]["scientific_name"]).split()[0]
            hits = perch_labels[
                perch_labels["scientific_name"].astype(str).str.match(
                    rf"^{re.escape(genus)}\s", na=False
                )
            ]
            if not hits.empty:
                self.proxy_map[position] = hits["bc_index"].to_numpy(np.int32)

    def _output(self, outputs: list[np.ndarray], name: str, width: int) -> np.ndarray:
        if name in self.output_names:
            return outputs[self.output_names.index(name)].astype(np.float32)
        candidates = [value for value in outputs if value.ndim == 2 and value.shape[1] == width]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Cannot identify Perch {name!r} output; outputs are "
                f"{[(n, a.shape) for n, a in zip(self.output_names, outputs)]}"
            )
        return candidates[0].astype(np.float32)

    def run(self, paths: list[Path]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rows: list[dict[str, object]] = []
        score_batches: list[np.ndarray] = []
        embedding_batches: list[np.ndarray] = []
        for path in paths:
            chunks, ends = read_soundscape(
                path,
                sample_rate=self.sample_rate,
                window_seconds=self.window_seconds,
                windows_per_file=self.windows_per_file,
            )
            outputs = self.session.run(None, {self.input_name: chunks})
            embedding = self._output(outputs, "embedding", 1536)
            # Perch class count differs between exported checkpoints; named output is preferred.
            if "label" in self.output_names:
                logits = outputs[self.output_names.index("label")].astype(np.float32)
            else:
                logits = max(
                    (value for value in outputs if value.ndim == 2 and value.shape[1] != 1536),
                    key=lambda value: value.shape[1],
                ).astype(np.float32)
            scores = np.zeros((self.windows_per_file, len(self.primary_labels)), np.float32)
            scores[:, self.mapped_positions] = logits[:, self.mapped_perch_indices]
            for position, perch_indices in self.proxy_map.items():
                scores[:, position] = logits[:, perch_indices].max(axis=1)
            metadata = soundscape_metadata(path)
            for end in ends:
                rows.append(
                    {
                        "row_id": f"{path.stem}_{int(end)}",
                        "filename": path.name,
                        "site": metadata["site"],
                        "hour_utc": metadata["hour_utc"],
                    }
                )
            score_batches.append(scores)
            embedding_batches.append(embedding)
        return (
            pd.DataFrame(rows),
            np.concatenate(score_batches),
            np.concatenate(embedding_batches),
        )
