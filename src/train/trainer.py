from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.model import LightProtoSSM, ProtoSSM, ResidualSSM
from src.train.losses import focal_bce_with_logits, mixup_files


@dataclass
class TrainingArrays:
    embeddings: np.ndarray
    logits: np.ndarray
    labels: np.ndarray
    site_ids: np.ndarray
    hours: np.ndarray
    families: np.ndarray | None = None


def build_proto_model(config: dict, model_name: str) -> nn.Module:
    model_config = config["proto_ssm"]
    shared = dict(
        d_input=config["embedding_dim"],
        d_model=model_config["d_model"],
        d_state=model_config["d_state"],
        n_classes=config["num_classes"],
        n_windows=config["windows_per_file"],
        dropout=model_config["dropout"],
        n_sites=model_config["n_sites"],
        meta_dim=model_config["meta_dim"],
        use_cross_attn=model_config["use_cross_attn"],
        cross_attn_heads=model_config["cross_attn_heads"],
    )
    if model_name == "model_51":
        return LightProtoSSM(**shared)
    return ProtoSSM(n_ssm_layers=model_config["n_ssm_layers"], **shared)


def _tensor(value: np.ndarray, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.tensor(value, dtype=dtype)


def train_proto_ssm(
    model: nn.Module,
    train: TrainingArrays,
    config: dict,
    validation: TrainingArrays | None = None,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Notebook training loop: mixup, focal BCE, distillation, early stop, SWA."""
    settings = config["training"]
    labels = train.labels.copy()
    smoothing = settings["label_smoothing"]
    labels = labels * (1.0 - smoothing) + smoothing / 2.0
    label_tensor = _tensor(labels)
    positive = label_tensor.sum(dim=(0, 1))
    total = label_tensor.shape[0] * label_tensor.shape[1]
    positive_weight = ((total - positive) / (positive + 1)).clamp(
        max=settings["pos_weight_cap"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=settings["lr"],
        epochs=settings["n_epochs"],
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    best_loss = float("inf")
    best_state = None
    wait = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    swa_state = None
    swa_count = 0
    swa_start = int(settings["n_epochs"] * settings["swa_start_frac"])

    for epoch in range(settings["n_epochs"]):
        if settings["mixup_alpha"] > 0 and epoch > 5:
            emb, logits, epoch_labels, families = mixup_files(
                train.embeddings,
                train.logits,
                labels,
                train.families,
                settings["mixup_alpha"],
            )
        else:
            emb, logits, epoch_labels, families = (
                train.embeddings,
                train.logits,
                labels,
                train.families,
            )
        model.train()
        output = model(
            _tensor(emb),
            _tensor(logits),
            _tensor(train.site_ids, torch.long),
            _tensor(train.hours, torch.long),
        )
        species_output, family_output = (
            (output[0], output[1]) if isinstance(output, tuple) else (output, None)
        )
        target = _tensor(epoch_labels)
        loss = focal_bce_with_logits(
            species_output,
            target,
            gamma=settings["focal_gamma"],
            pos_weight=positive_weight[None, None, :],
        )
        loss = loss + settings["distill_weight"] * F.mse_loss(
            species_output, _tensor(logits)
        )
        if family_output is not None and families is not None:
            loss = loss + 0.1 * F.binary_cross_entropy_with_logits(
                family_output, _tensor(families)
            )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch >= swa_start:
            if swa_state is None:
                swa_state = copy.deepcopy(model.state_dict())
            else:
                for key, value in model.state_dict().items():
                    swa_state[key] += value
            swa_count += 1

        model.eval()
        with torch.no_grad():
            if validation is None:
                validation_loss = loss.detach()
            else:
                validation_output = model(
                    _tensor(validation.embeddings),
                    _tensor(validation.logits),
                    _tensor(validation.site_ids, torch.long),
                    _tensor(validation.hours, torch.long),
                )
                if isinstance(validation_output, tuple):
                    validation_output = validation_output[0]
                validation_loss = F.binary_cross_entropy_with_logits(
                    validation_output,
                    _tensor(validation.labels),
                    pos_weight=positive_weight[None, None, :],
                )
        history["train_loss"].append(float(loss))
        history["val_loss"].append(float(validation_loss))
        if validation_loss < best_loss:
            best_loss = float(validation_loss)
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= settings["patience"]:
                break

    if swa_state is not None and swa_count >= 3:
        model.load_state_dict({key: value / swa_count for key, value in swa_state.items()})
    elif best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def train_light_proto_ssm(
    model: LightProtoSSM, train: TrainingArrays, config: dict
) -> tuple[LightProtoSSM, dict[str, list[float]]]:
    """Exact Model_51 loop: weighted BCE + distillation, OneCycle, then SWA."""
    settings = config["training"]
    embeddings = _tensor(train.embeddings)
    logits = _tensor(train.logits)
    labels = _tensor(train.labels)
    sites = _tensor(train.site_ids, torch.long)
    hours = _tensor(train.hours, torch.long)
    positive = labels.sum(dim=(0, 1))
    total = labels.shape[0] * labels.shape[1]
    positive_weight = ((total - positive) / (positive + 1)).clamp(
        max=settings["pos_weight_cap"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=settings["lr"],
        epochs=settings["n_epochs"],
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    averaged = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(settings["n_epochs"] * settings["swa_start_frac"])
    swa_scheduler = torch.optim.swa_utils.SWALR(
        optimizer, swa_lr=settings["swa_lr"]
    )
    best_loss, best_state, wait = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    last_epoch = 0
    for epoch in range(settings["n_epochs"]):
        last_epoch = epoch
        model.train()
        output = model(embeddings, logits, sites, hours)
        loss = F.binary_cross_entropy_with_logits(
            output, labels, pos_weight=positive_weight[None, None, :]
        ) + settings["distill_weight"] * F.mse_loss(output, logits)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch >= swa_start:
            averaged.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        value = float(loss)
        history["train_loss"].append(value)
        history["val_loss"].append(value)
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= settings["patience"]:
                break
    if last_epoch >= swa_start:
        model.load_state_dict(averaged.module.state_dict())
    elif best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def train_residual_ssm(
    embeddings: np.ndarray,
    first_pass: np.ndarray,
    labels: np.ndarray,
    site_ids: np.ndarray,
    hours: np.ndarray,
    config: dict,
) -> ResidualSSM:
    settings = config["residual_ssm"]
    model = ResidualSSM(
        d_input=config["embedding_dim"],
        d_scores=config["num_classes"],
        d_model=settings["d_model"],
        d_state=settings["d_state"],
        n_classes=config["num_classes"],
        n_windows=config["windows_per_file"],
        dropout=settings["dropout"],
        n_sites=config["proto_ssm"]["n_sites"],
    )
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(first_pass, -30, 30)))
    residual_target = labels - probabilities
    count = len(embeddings)
    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(count, generator=generator).numpy()
    validation_count = max(1, int(count * 0.15))
    validation_indices, train_indices = permutation[:validation_count], permutation[validation_count:]
    emb, scores, target = _tensor(embeddings), _tensor(first_pass), _tensor(residual_target)
    sites, hour_values = _tensor(site_ids, torch.long), _tensor(hours, torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=settings["lr"],
        epochs=settings["n_epochs"],
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(settings["n_epochs"]):
        model.train()
        correction = model(
            emb[train_indices], scores[train_indices], sites[train_indices], hour_values[train_indices]
        )
        loss = F.mse_loss(correction, target[train_indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_loss = F.mse_loss(
                model(
                    emb[validation_indices],
                    scores[validation_indices],
                    sites[validation_indices],
                    hour_values[validation_indices],
                ),
                target[validation_indices],
            )
        if validation_loss < best_loss:
            best_loss = float(validation_loss)
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= settings["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def save_weights(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
