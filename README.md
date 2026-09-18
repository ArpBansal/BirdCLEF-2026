# BirdCLEF 2026 taxonomy ensemble

This repository is a readable Python conversion of
`hierarchical-taxonomy-post-processing-birdclef-2.ipynb`. It keeps the active
notebook run unchanged at the algorithm level:

- Model 22: Perch embeddings/logits → ProtoSSM → MLP/probability fusion →
  ResidualSSM → temporal post-processing.
- Model 51: Perch → lightweight ProtoSSM + priors/probes → ResidualSSM, blended
  with the five-fold Distilled-SED predictions and the notebook's five gates.
- Final ensemble: `0.05 * Model_22 + 0.95 * Model_51`, followed by genus
  smoothing (`0.15`) and class smoothing (`0.05`).

All numerical settings live in [`config/`](config/). The original notebook is
left untouched as the reference implementation.

## Layout

```text
config/                 Model and ensemble YAML files
src/processing/         Audio, metadata, and post-processing
src/model/              ProtoSSM and ResidualSSM architectures
src/train/              Losses and training entry point
inference/              Perch/SED inference and ensemble entry points
dashboard/              Streamlit audio inspection and inference UI
models/                 Add downloaded/trained weights here
```

## Expected data and weights

```text
data/birdclef-2026/
  sample_submission.csv
  taxonomy.csv
  train_soundscapes_labels.csv
  test_soundscapes/*.ogg

models/
  labels.csv
  perch_v2.onnx
  perch_v2_no_dft.onnx
  sed_fold0.onnx ... sed_fold4.onnx
  model_22/proto_ssm_best.pt
  model_22/residual_ssm_best.pt
  model_22/probes.joblib              # optional probe/prior/calibration bundle
  model_51/proto_ssm.pt
  model_51/residual_ssm.pt
  model_51/probes.joblib              # optional probe/prior/calibration bundle
  model_51/thresholds.npy             # optional when bundled above
```

The optional joblib bundle can contain `probe_models`, `scaler`, `pca`,
`alpha_blend`, `site_to_index`, `prior_tables`, `thresholds`, and (for Model 22)
`proto_weight`. Missing optional calibration artifacts use the notebook's neutral
defaults; model checkpoints and Perch/SED files are required.

## Setup and inference

The project keeps the Python 3.13 environment selected during `uv init` and uses
the PyTorch 2.9 release line. Other dependency versions remain flexible.

```bash
uv sync
uv run birdclef-prepare --model model_22 --output data/model_22_train.npz
uv run birdclef-infer --model model_22
uv run birdclef-infer --model model_51
uv run birdclef-infer --model ensemble
```

The commands write `subm_22.csv`, `subm_51.csv`, then `submission.csv`.

## Interactive dashboard

The dashboard runs inference in the Streamlit process. This keeps model loading,
configuration, and preprocessing on one machine and avoids an unnecessary API
service for a single-user inspection tool.

```bash
uv run birdclef-dashboard
# Equivalent: uv run streamlit run dashboard/app.py
```

Upload WAV, OGG, FLAC, or MP3 audio to inspect the waveform and slide through
all twelve five-second normalized log-mel windows using the persistent signal
explorer. The dashboard also shows artifact readiness, top birds over the
recording, and the three strongest predictions in every window. The raw
prediction table is downloadable as CSV.

## Training

Training consumes the Perch cache as an NPZ to avoid recomputing the expensive
backbone pass. Arrays use file-major shapes: `embeddings` `(F, 12, 1536)`,
`logits` and `labels` `(F, 12, 234)`, and `site_ids`/`hours` `(F,)`.
Optional validation arrays use the same names prefixed with `val_`.

```bash
uv run birdclef-train --model model_22 --features data/model_22_train.npz
uv run birdclef-train --model model_51 --features data/model_51_train.npz
```

The loss, mixup, distillation weight, optimizer, scheduler, early stopping, and
SWA behavior are the notebook versions; adjust paths only in YAML.
