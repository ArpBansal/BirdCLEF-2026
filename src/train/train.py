from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.model import LightProtoSSM, ProtoSSM
from src.train.trainer import (
    TrainingArrays,
    build_proto_model,
    save_weights,
    train_light_proto_ssm,
    train_proto_ssm,
)


def _arrays(archive: np.lib.npyio.NpzFile, prefix: str = "") -> TrainingArrays:
    families_key = f"{prefix}families"
    return TrainingArrays(
        embeddings=archive[f"{prefix}embeddings"],
        logits=archive[f"{prefix}logits"],
        labels=archive[f"{prefix}labels"],
        site_ids=archive[f"{prefix}site_ids"],
        hours=archive[f"{prefix}hours"],
        families=archive[families_key] if families_key in archive else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the notebook's ProtoSSM head")
    parser.add_argument("--model", choices=["model_22", "model_51"], required=True)
    parser.add_argument("--features", type=Path, required=True, help="Prepared Perch NPZ cache")
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config or f"config/{args.model}.yaml")
    archive = np.load(args.features)
    train = _arrays(archive)
    validation = _arrays(archive, "val_") if "val_embeddings" in archive else None
    model = build_proto_model(config, args.model)
    flat_embeddings = np.asarray(train.embeddings).reshape(-1, config["embedding_dim"])
    flat_labels = np.asarray(train.labels).reshape(-1, config["num_classes"])
    if isinstance(model, LightProtoSSM):
        model.init_prototypes(
            torch.tensor(flat_embeddings, dtype=torch.float32),
            torch.tensor(flat_labels, dtype=torch.float32),
        )
        model, history = train_light_proto_ssm(model, train, config)
    else:
        assert isinstance(model, ProtoSSM)
        if "class_to_family" in archive:
            class_to_family = archive["class_to_family"].astype(int).tolist()
            model.init_family_head(train.families.shape[1], class_to_family)
        model.init_prototypes_from_data(
            torch.tensor(flat_embeddings, dtype=torch.float32),
            torch.tensor(flat_labels, dtype=torch.float32),
        )
        model, history = train_proto_ssm(model, train, config, validation)
    output = config["paths"]["proto_weights"]
    save_weights(model, output)
    history_path = output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved weights to {output}")


if __name__ == "__main__":
    main()
