from __future__ import annotations

import glob
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d

from inference.perch import PerchExtractor
from src.model import ProtoSSM, ResidualSSM
from src.processing.audio import audio_to_mel, read_soundscape
from src.processing.postprocess import (
    adaptive_delta_smooth,
    apply_per_class_thresholds,
    file_confidence_scale,
    rank_aware_scaling,
    sigmoid,
)
from src.train.trainer import build_proto_model


def _state_dict(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


def _metadata_ids(metadata: pd.DataFrame, site_to_index: dict[str, int], n_sites: int):
    files = metadata.drop_duplicates("filename")
    sites = np.array(
        [min(site_to_index.get(str(site), 0), n_sites - 1) for site in files["site"]],
        dtype=np.int64,
    )
    hours = files["hour_utc"].to_numpy(np.int64) % 24
    return torch.tensor(sites), torch.tensor(hours)


def temporal_tta(model, embeddings, scores, site_ids, hours, shifts):
    model.eval()
    embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    predictions = []
    with torch.no_grad():
        for shift in shifts:
            shifted_embedding = torch.roll(embedding_tensor, shift, dims=1) if shift else embedding_tensor
            shifted_scores = torch.roll(score_tensor, shift, dims=1) if shift else score_tensor
            output = model(shifted_embedding, shifted_scores, site_ids, hours)
            if isinstance(output, tuple):
                output = output[0]
            output = output.numpy()
            predictions.append(np.roll(output, -shift, axis=1) if shift else output)
        flipped = model(
            embedding_tensor.flip(1), score_tensor.flip(1), site_ids, hours
        )
        if isinstance(flipped, tuple):
            flipped = flipped[0]
        predictions.append(flipped.numpy()[:, ::-1, :].copy())
    return np.mean(predictions, axis=0)


def _apply_prior(scores, sites, hours, tables, weight):
    if tables is None:
        return scores
    epsilon = 1e-4
    prior = np.tile(tables["global_p"], (len(scores), 1))
    for index, hour in enumerate(hours):
        hour = int(hour)
        if hour in tables["hour_to_i"]:
            table_index = tables["hour_to_i"][hour]
            count = tables["hour_n"][table_index]
            shrink = count / (count + 8.0)
            prior[index] = shrink * tables["hour_p"][table_index] + (1 - shrink) * tables["global_p"]
    for index, site in enumerate(sites):
        site = str(site)
        if site in tables["site_to_i"]:
            table_index = tables["site_to_i"][site]
            count = tables["site_n"][table_index]
            shrink = count / (count + 8.0)
            prior[index] = shrink * tables["site_p"][table_index] + (1 - shrink) * prior[index]
    if "sh_to_i" in tables:
        for index, (site, hour) in enumerate(zip(sites, hours)):
            key = (str(site), int(hour))
            if key in tables["sh_to_i"]:
                table_index = tables["sh_to_i"][key]
                count = tables["sh_n"][table_index]
                shrink = count / (count + 4.0)
                prior[index] = shrink * tables["sh_p"][table_index] + (1 - shrink) * prior[index]
    prior = np.clip(prior, epsilon, 1 - epsilon)
    return (scores + weight * (np.log(prior) - np.log1p(-prior))).astype(np.float32)


def _apply_probe_bundle(embeddings, scores, bundle):
    if not bundle or not bundle.get("probe_models"):
        return scores
    scaler, pca = bundle["scaler"], bundle["pca"]
    transformed = pca.transform(scaler.transform(embeddings)).astype(np.float32)
    result = scores.copy()
    windows = bundle.get("windows_per_file", 12)
    for class_index, classifier in bundle["probe_models"].items():
        column = scores[:, int(class_index)]
        view = column.reshape(-1, windows)
        previous = np.concatenate([view[:, :1], view[:, :-1]], axis=1).reshape(-1)
        following = np.concatenate([view[:, 1:], view[:, -1:]], axis=1).reshape(-1)
        features = np.hstack(
            [
                transformed,
                column[:, None],
                previous[:, None],
                following[:, None],
                np.repeat(view.mean(1), windows)[:, None],
                np.repeat(view.max(1), windows)[:, None],
                np.repeat(view.std(1), windows)[:, None],
            ]
        )
        prediction = classifier.predict_proba(features)[:, 1].astype(np.float32)
        # Notebook probe output is converted back to logits before blending.
        prediction = np.log(prediction + 1e-7) - np.log(1 - prediction + 1e-7)
        alpha = bundle.get("alpha_blend", 0.4)
        result[:, int(class_index)] = (1 - alpha) * column + alpha * prediction
    return result


def _model_51_rank_blend(
    proto_submission: pd.DataFrame,
    sed_submission: pd.DataFrame,
    taxonomy: pd.DataFrame,
    weights: list[float],
) -> pd.DataFrame:
    columns = [column for column in proto_submission if column != "row_id"]
    sed_submission = (
        sed_submission.set_index("row_id").loc[proto_submission["row_id"]].reset_index()
    )
    epsilon = 1e-5
    proto_values = np.clip(
        proto_submission[columns].to_numpy(np.float32), epsilon, 1 - epsilon
    )
    sed_values = np.clip(
        sed_submission[columns].to_numpy(np.float32), epsilon, 1 - epsilon
    )
    proto_rank = pd.DataFrame(proto_values).rank(axis=0, pct=True).to_numpy(np.float32)
    sed_rank = pd.DataFrame(sed_values).rank(axis=0, pct=True).to_numpy(np.float32)
    proto_weight, sed_weight = np.asarray(weights, np.float32)
    proto_weight, sed_weight = (
        proto_weight / (proto_weight + sed_weight),
        sed_weight / (proto_weight + sed_weight),
    )
    prediction = proto_weight * proto_rank + sed_weight * sed_rank

    row_ids = proto_submission["row_id"].astype(str).to_numpy()
    file_ids = np.array(["_".join(row_id.split("_")[:-1]) for row_id in row_ids])
    fake_only = (proto_values > 0.50) & (sed_values < 0.05)
    prediction = np.where(
        fake_only, 0.92 * prediction + 0.08 * proto_rank, prediction
    )
    offsets = np.arange(-3, 4, dtype=np.float32)
    kernel = (1.0 + (offsets / 1.20) ** 2 / 2.0) ** (-1.5)
    kernel = (kernel / kernel.sum()).astype(np.float32)
    context = proto_values.copy()
    for file_id in pd.unique(file_ids):
        mask = file_ids == file_id
        values = proto_values[mask]
        if len(values) > 1:
            padded = np.pad(values, ((3, 3), (0, 0)), mode="edge")
            context[mask] = sum(kernel[i] * padded[i : i + len(values)] for i in range(7))
    context_rank = pd.DataFrame(context).rank(axis=0, pct=True).to_numpy(np.float32)
    continuous = (
        (context_rank > 0.88)
        & (proto_rank > 0.75)
        & (sed_values < 0.12)
        & (~fake_only)
    )
    prediction = np.where(
        continuous,
        0.85 * prediction + 0.15 * np.maximum(proto_rank, context_rank),
        prediction,
    )
    sed_only = (
        (sed_rank > 0.95)
        & (proto_rank < 0.80)
        & (~fake_only)
        & (~continuous)
    )
    prediction = np.where(sed_only, 0.88 * prediction + 0.12 * sed_rank, prediction)

    result = proto_submission.copy()
    result[columns] = prediction.astype(np.float32)
    mirror_groups = (
        ("47158son15", "47158son16"),
        ("47158son09", "47158son12"),
        ("47158son02", "47158son14"),
        ("47158son13", "47158son21", "47158son22", "47158son23"),
    )
    for group in mirror_groups:
        valid = [species for species in group if species in columns]
        if len(valid) >= 2:
            maximum = result[valid].max(axis=1)
            result[valid] = np.repeat(maximum.to_numpy()[:, None], len(valid), axis=1)
    class_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    for species in columns:
        if class_map.get(species) in {"Amphibia", "Mammalia", "Reptilia"}:
            values = result[species].to_numpy(np.float32)
            threshold = values.mean() + 0.05
            result[species] = np.where(values < threshold, values * 0.9, values)
    return result


def run_sed(paths: list[Path], labels: list[str], config: dict) -> pd.DataFrame:
    pattern = config["paths"]["sed_glob"]
    if not Path(pattern).is_absolute():
        pattern = str(Path(__file__).resolve().parents[1] / pattern)
    model_paths = sorted(glob.glob(str(pattern)))
    if not model_paths:
        raise FileNotFoundError(f"No SED weights match {pattern}")
    sessions = [ort.InferenceSession(path, providers=["CPUExecutionProvider"]) for path in model_paths]
    predictions, row_ids = [], []
    sed = config["sed"]

    def sigmoid_sed(values: np.ndarray) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))).astype(np.float32)

    for path in paths:
        chunks, ends = read_soundscape(
            path, config["sample_rate"], config["window_seconds"], config["windows_per_file"]
        )
        mel = audio_to_mel(
            chunks,
            config["sample_rate"],
            sed["n_mels"],
            sed["n_fft"],
            sed["hop_length"],
            sed["fmin"],
            sed["fmax"],
            sed["top_db"],
        )
        total = np.zeros((len(chunks), len(labels)), np.float32)
        for session in sessions:
            outputs = session.run(None, {session.get_inputs()[0].name: mel})
            clip_logits, frame_logits = outputs[0], outputs[1]
            total += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(
                frame_logits.max(axis=1)
            )
        mean = total / len(sessions)
        if len(mean) > 1:
            mean = gaussian_filter1d(mean, sigma=sed["temporal_sigma"], axis=0, mode="nearest")
        predictions.append(mean.astype(np.float32))
        row_ids.extend(f"{path.stem}_{int(end)}" for end in ends)
    result = pd.DataFrame(np.clip(np.concatenate(predictions), 0, 1), columns=labels)
    result.insert(0, "row_id", row_ids)
    return result


def run_paths(
    config: dict, model_name: str, paths: list[Path]
) -> pd.DataFrame:
    """Run one notebook model for an explicit list of 60-second audio files."""
    competition = config["paths"]["competition_dir"]
    sample = pd.read_csv(competition / "sample_submission.csv")
    taxonomy = pd.read_csv(competition / "taxonomy.csv")
    labels = sample.columns[1:].tolist()
    if not paths:
        raise ValueError("At least one audio path is required")

    extractor = PerchExtractor(
        config["paths"]["perch_onnx"],
        config["paths"]["perch_labels"],
        taxonomy,
        labels,
        config["sample_rate"],
        config["windows_per_file"],
        config["window_seconds"],
    )
    metadata, scores, embeddings = extractor.run(paths)
    bundle_path = config["paths"]["probe_bundle"]
    bundle = joblib.load(bundle_path) if bundle_path.exists() else {}
    site_to_index = bundle.get("site_to_index", {})
    site_ids, hours = _metadata_ids(
        metadata, site_to_index, config["proto_ssm"]["n_sites"]
    )
    file_count = len(paths)
    embedding_files = embeddings.reshape(file_count, config["windows_per_file"], -1)
    score_files = scores.reshape(file_count, config["windows_per_file"], -1)

    proto = build_proto_model(config, model_name)
    if isinstance(proto, ProtoSSM):
        group_column = next(
            (column for column in ("family", "order", "class_name") if column in taxonomy),
            None,
        )
        group_map = (
            taxonomy.set_index("primary_label")[group_column].to_dict()
            if group_column
            else {label: "Unknown" for label in labels}
        )
        groups = sorted({str(value) for value in group_map.values()})
        group_to_index = {group: index for index, group in enumerate(groups)}
        class_to_group = [
            group_to_index.get(str(group_map.get(label, "Unknown")), 0) for label in labels
        ]
        proto.init_family_head(len(groups), class_to_group)
    proto.load_state_dict(_state_dict(config["paths"]["proto_weights"]))
    proto_scores = temporal_tta(
        proto,
        embedding_files,
        score_files,
        site_ids,
        hours,
        config["postprocess"]["tta_shifts"],
    ).reshape(-1, len(labels))

    adjusted = scores
    if model_name == "model_51":
        adjusted = _apply_prior(
            adjusted,
            metadata["site"].to_numpy(),
            metadata["hour_utc"].to_numpy(),
            bundle.get("prior_tables"),
            config["fusion"]["lambda_prior"],
        )
    adjusted = _apply_probe_bundle(embeddings, adjusted, bundle)
    if model_name == "model_51":
        per_class_weight = np.where(
            extractor.mapped_mask,
            config["fusion"]["mapped_proto_weight"],
            config["fusion"]["unmapped_proto_weight"],
        ).astype(np.float32)
    else:
        per_class_weight = np.full(len(labels), bundle.get("proto_weight", 0.5), np.float32)
    first_pass = (
        per_class_weight[None, :] * proto_scores
        + (1 - per_class_weight[None, :]) * adjusted
    )

    residual_settings = config["residual_ssm"]
    residual = ResidualSSM(
        d_input=config["embedding_dim"],
        d_scores=len(labels),
        d_model=residual_settings["d_model"],
        d_state=residual_settings["d_state"],
        n_classes=len(labels),
        n_windows=config["windows_per_file"],
        dropout=residual_settings["dropout"],
        n_sites=config["proto_ssm"]["n_sites"],
    )
    residual.load_state_dict(_state_dict(config["paths"]["residual_weights"]))
    residual.eval()
    with torch.no_grad():
        correction = residual(
            torch.tensor(embedding_files),
            torch.tensor(first_pass.reshape(score_files.shape)),
            site_ids,
            hours,
        ).numpy().reshape(first_pass.shape)
    final_scores = first_pass + residual_settings["correction_weight"] * correction

    class_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    temperatures = np.array(
        [
            config["postprocess"]["temperature_texture"]
            if class_map.get(label) in {"Amphibia", "Insecta"}
            else config["postprocess"]["temperature_aves"]
            for label in labels
        ],
        np.float32,
    )
    probabilities = sigmoid(final_scores / temperatures[None, :])
    power = config["postprocess"].get("file_confidence_power", 1.0)
    probabilities = file_confidence_scale(
        probabilities,
        config["windows_per_file"],
        config["postprocess"]["file_level_top_k"],
        power,
    )
    probabilities = rank_aware_scaling(
        probabilities,
        config["windows_per_file"],
        config["postprocess"]["rank_aware_power"],
    )
    probabilities = adaptive_delta_smooth(
        probabilities,
        config["windows_per_file"],
        config["postprocess"]["delta_shift_alpha"],
    )
    thresholds = bundle.get("thresholds")
    threshold_path = config["paths"].get("thresholds")
    if thresholds is None and threshold_path is not None and threshold_path.exists():
        thresholds = np.load(threshold_path)
    if thresholds is None:
        thresholds = np.full(len(labels), 0.5, np.float32)
    probabilities = apply_per_class_thresholds(np.clip(probabilities, 0, 1), thresholds)
    result = pd.DataFrame(probabilities.astype(np.float32), columns=labels)
    result.insert(0, "row_id", metadata["row_id"].to_numpy())

    if model_name == "model_51":
        sed = run_sed(paths, labels, config)
        result = _model_51_rank_blend(
            result, sed, taxonomy, config["fusion"]["proto_sed_weights"]
        )
    return result


def run_model(config: dict, model_name: str) -> pd.DataFrame:
    """Run a model on the competition test directory."""
    competition = config["paths"]["competition_dir"]
    paths = sorted((competition / "test_soundscapes").glob("*.ogg"))
    if not paths:
        raise FileNotFoundError(f"No test .ogg files in {competition / 'test_soundscapes'}")
    return run_paths(config, model_name, paths)
