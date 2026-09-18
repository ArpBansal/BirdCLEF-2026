"""The notebook's 60-second/12-window audio preparation."""

from __future__ import annotations

import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


FILENAME_PATTERN = re.compile(
    r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg"
)


def soundscape_metadata(path: str | Path) -> dict[str, str | int]:
    path = Path(path)
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return {"filename": path.name, "site": "unknown", "hour_utc": -1}
    _, site, _, hms = match.groups()
    return {"filename": path.name, "site": site, "hour_utc": int(hms[:2])}


def read_soundscape(
    path: str | Path,
    sample_rate: int = 32_000,
    window_seconds: int = 5,
    windows_per_file: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Read mono audio, resample, then pad/truncate to exactly 60 seconds."""
    waveform, original_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if original_rate != sample_rate:
        waveform = librosa.resample(
            waveform, orig_sr=original_rate, target_sr=sample_rate
        )
    window_samples = sample_rate * window_seconds
    file_samples = windows_per_file * window_samples
    if len(waveform) < file_samples:
        waveform = np.pad(waveform, (0, file_samples - len(waveform)))
    else:
        waveform = waveform[:file_samples]
    chunks = waveform.reshape(windows_per_file, window_samples).astype(np.float32)
    ends = np.arange(1, windows_per_file + 1) * window_seconds
    return chunks, ends


def audio_to_mel(
    chunks: np.ndarray,
    sample_rate: int = 32_000,
    n_mels: int = 256,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: int = 20,
    fmax: int = 16_000,
    top_db: int = 80,
) -> np.ndarray:
    """Distilled-SED mel preprocessing, unchanged from Model_51."""
    mels = []
    for chunk in chunks:
        mel = librosa.feature.melspectrogram(
            y=chunk,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=2.0,
        )
        mel = librosa.power_to_db(mel, top_db=top_db)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mels.append(mel)
    return np.stack(mels)[:, None].astype(np.float32)
