from __future__ import annotations

import glob
import hashlib
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from inference.pipeline import run_paths
from src.config import PROJECT_ROOT, load_config
from src.logging_config import get_logger
from src.processing.audio import audio_to_mel, read_soundscape


MODEL_LABELS = {
    "Model 51 · ProtoSSM + Distilled-SED": "model_51",
    "Model 22 · ProtoSSM v5": "model_22",
}

logger = get_logger("dashboard")


def _required_artifacts(config: dict, model_name: str) -> list[Path]:
    paths = config["paths"]
    required = [
        paths["competition_dir"] / "sample_submission.csv",
        paths["competition_dir"] / "taxonomy.csv",
        paths["perch_onnx"],
        paths["perch_labels"],
        paths["proto_weights"],
        paths["residual_weights"],
    ]
    if model_name == "model_51" and not glob.glob(str(paths["sed_glob"])):
        sed_pattern = Path(str(paths["sed_glob"]))
        if not sed_pattern.is_absolute():
            sed_pattern = PROJECT_ROOT / sed_pattern
        if not glob.glob(str(sed_pattern)):
            required.append(sed_pattern)
    return [path for path in required if not path.exists()]


def _audio_summary(path: Path) -> tuple[np.ndarray, int, float]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return waveform, int(sample_rate), len(waveform) / sample_rate


def _waveform_frame(waveform: np.ndarray, sample_rate: int) -> pd.DataFrame:
    maximum_points = 5_000
    step = max(1, len(waveform) // maximum_points)
    samples = waveform[::step]
    return pd.DataFrame(
        {
            "Time (s)": np.arange(len(samples), dtype=np.float32) * step / sample_rate,
            "Amplitude": samples,
        }
    ).set_index("Time (s)")


def _spectrogram_frame(mel: np.ndarray, duration: float = 5.0) -> pd.DataFrame:
    # Limit visualization density only; inference still receives the complete tensor.
    values = mel[::2, ::2]
    frequency_bins, time_bins = values.shape
    return pd.DataFrame(
        {
            "Time (s)": np.tile(
                np.linspace(0, duration, time_bins, dtype=np.float32), frequency_bins
            ),
            "Mel bin": np.repeat(np.arange(0, mel.shape[0], 2), time_bins),
            "Relative energy (σ)": values.reshape(-1),
        }
    )


def _taxonomy_names(taxonomy: pd.DataFrame) -> dict[str, str]:
    name_column = next(
        (
            column
            for column in ("common_name", "english_name", "species", "scientific_name")
            if column in taxonomy
        ),
        None,
    )
    if name_column is None:
        return {}
    return taxonomy.set_index("primary_label")[name_column].astype(str).to_dict()


def _prediction_tables(
    prediction: pd.DataFrame, taxonomy: pd.DataFrame, top_k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability_columns = [column for column in prediction if column != "row_id"]
    values = prediction[probability_columns]
    names = _taxonomy_names(taxonomy)
    aggregate = values.max(axis=0).sort_values(ascending=False).head(top_k)
    top = pd.DataFrame(
        {
            "Bird": [names.get(label, label) for label in aggregate.index],
            "Label": aggregate.index,
            "Peak score": aggregate.to_numpy(),
        }
    )
    window_rows: list[dict[str, object]] = []
    for row_index, row in values.iterrows():
        best = row.nlargest(min(3, len(row)))
        end_seconds = int(str(prediction.loc[row_index, "row_id"]).rsplit("_", 1)[-1])
        for rank, (label, score) in enumerate(best.items(), start=1):
            window_rows.append(
                {
                    "Window": f"{max(0, end_seconds - 5)}–{end_seconds}s",
                    "Rank": rank,
                    "Bird": names.get(label, label),
                    "Label": label,
                    "Score": float(score),
                }
            )
    return top, pd.DataFrame(window_rows)


def _save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(uploaded_file.getbuffer())
        path = Path(handle.name)
    logger.info(
        "Saved upload for inference: name=%s bytes=%d temporary_path=%s",
        uploaded_file.name,
        uploaded_file.size,
        path,
    )
    return path


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">ACOUSTIC SPECIES IDENTIFICATION</div>
          <h1>BirdCLEF Audio Inspector</h1>
          <p>Explore the model-ready audio, move through every five-second window,
             and inspect species predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0e1117; }
        .block-container { max-width: 1480px; padding-top: 2.2rem; padding-bottom: 4rem; }
        .hero { margin-bottom: 1.35rem; }
        .hero .eyebrow { color: #59d0a5; font-size: .72rem; font-weight: 750;
                         letter-spacing: .16em; margin-bottom: .45rem; }
        .hero h1 { font-size: 2.55rem; line-height: 1.05; margin: 0; letter-spacing: -.035em; }
        .hero p { color: #98a2b3; font-size: 1rem; margin: .65rem 0 0; }
        div[data-testid="stFileUploader"] section {
            min-height: 78px; border: 1px dashed #465064; background: #151a23;
        }
        div[data-testid="stMetric"] {
            background: #151a23; border: 1px solid #262e3d; border-radius: 12px;
            padding: .7rem 1rem;
        }
        .spectrogram-empty {
            min-height: 430px; border: 1px solid #2a3445; border-radius: 12px;
            display: flex; align-items: center; justify-content: center; text-align: center;
            background-color: #101722;
            background-image:
              linear-gradient(rgba(75, 92, 117, .10) 1px, transparent 1px),
              linear-gradient(90deg, rgba(75, 92, 117, .10) 1px, transparent 1px),
              radial-gradient(circle at 50% 50%, rgba(40, 184, 134, .10), transparent 48%);
            background-size: 48px 48px, 48px 48px, auto;
        }
        .spectrogram-empty .icon { font-size: 2.2rem; margin-bottom: .8rem; }
        .spectrogram-empty strong { display: block; color: #e5e7eb; font-size: 1.05rem; }
        .spectrogram-empty span { display: block; color: #778195; margin-top: .35rem; }
        .section-kicker { color: #59d0a5; font-size: .7rem; font-weight: 750;
                          letter-spacing: .13em; }
        .window-title { font-size: 1.08rem; font-weight: 700; margin: .12rem 0 .1rem; }
        .window-subtitle { color: #8d98a9; font-size: .82rem; margin-bottom: .5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_spectrogram() -> None:
    st.markdown(
        """
        <div class="spectrogram-empty">
          <div>
            <div class="icon">〰</div>
            <strong>Mel spectrogram workspace</strong>
            <span>Upload a recording to explore its 5-second windows</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_time(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def _render_mel(mel: np.ndarray, start_seconds: int, window_seconds: int) -> None:
    frame = _spectrogram_frame(mel, window_seconds)
    frame["Time (s)"] += start_seconds
    st.vega_lite_chart(
        frame,
        {
            "mark": {"type": "rect", "tooltip": True},
            "encoding": {
                "x": {
                    "field": "Time (s)",
                    "type": "quantitative",
                    "title": "Recording time (seconds)",
                    "scale": {
                        "domain": [start_seconds, start_seconds + window_seconds]
                    },
                },
                "y": {
                    "field": "Mel bin",
                    "type": "ordinal",
                    "sort": "descending",
                    "title": "Frequency (low → high)",
                    "axis": {"labels": False, "ticks": False},
                },
                "color": {
                    "field": "Relative energy (σ)",
                    "type": "quantitative",
                    "scale": {"scheme": "viridis"},
                    "legend": {"title": "Standard deviations from window mean"},
                },
            },
            "height": 430,
            "config": {"view": {"stroke": None}},
        },
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="BirdCLEF Audio Inspector", page_icon="🦜", layout="wide"
    )
    _inject_styles()
    _render_header()

    with st.sidebar:
        st.markdown("### Inference settings")
        selected_label = st.selectbox("Model", list(MODEL_LABELS))
        model_name = MODEL_LABELS[selected_label]
        config = load_config(f"config/{model_name}.yaml")
        top_k = st.slider("Top predictions", 3, 20, 10)
        st.divider()
        st.markdown("#### Active pipeline")
        if model_name == "model_22":
            st.write("Perch → ProtoSSM v5 → ResidualSSM → temporal calibration")
        else:
            st.write("Perch → ProtoSSM → ResidualSSM + Distilled-SED → gated rank blend")

    upload_column, note_column = st.columns([3, 1], vertical_alignment="bottom")
    with upload_column:
        uploaded = st.file_uploader(
            "Audio recording",
            type=["wav", "ogg", "flac", "mp3"],
            help="Audio is converted to mono, resampled to 32 kHz, and padded or truncated to 60 seconds.",
        )
    with note_column:
        st.caption("Model input · mono · 32 kHz · first 60 seconds")

    st.markdown('<div class="section-kicker">SIGNAL EXPLORER</div>', unsafe_allow_html=True)
    st.markdown('<div class="window-title">Mel spectrogram</div>', unsafe_allow_html=True)
    if uploaded is None:
        _render_empty_spectrogram()
        st.caption(
            "The explorer stays here throughout the workflow. After upload, use the "
            "timeline to slide across all twelve model windows."
        )
        return

    temporary_path = _save_upload(uploaded)
    try:
        try:
            waveform, original_rate, duration = _audio_summary(temporary_path)
        except Exception as error:
            logger.exception(
                "Audio decoding failed: upload=%s content_type=%s",
                uploaded.name,
                uploaded.type,
            )
            st.error("The uploaded file could not be decoded as audio.")
            st.exception(error)
            return
        chunks, _ = read_soundscape(
            temporary_path,
            config["sample_rate"],
            config["window_seconds"],
            config["windows_per_file"],
        )
        window_seconds = config["window_seconds"]
        window_starts = list(
            range(0, window_seconds * config["windows_per_file"], window_seconds)
        )
        if st.session_state.get("spectrogram_audio") != uploaded.name:
            st.session_state["spectrogram_audio"] = uploaded.name
            st.session_state["spectrogram_start"] = 0

        control_left, control_center, control_right = st.columns([1, 8, 1])
        with control_left:
            previous = st.button(
                "← Prev", use_container_width=True, disabled=st.session_state["spectrogram_start"] == 0
            )
        if previous:
            st.session_state["spectrogram_start"] = max(
                0, st.session_state["spectrogram_start"] - window_seconds
            )
        with control_right:
            following = st.button(
                "Next →",
                use_container_width=True,
                disabled=st.session_state["spectrogram_start"] == window_starts[-1],
            )
        if following:
            st.session_state["spectrogram_start"] = min(
                window_starts[-1],
                st.session_state["spectrogram_start"] + window_seconds,
            )
        with control_center:
            start_seconds = st.select_slider(
                "Recording timeline",
                options=window_starts,
                key="spectrogram_start",
                format_func=lambda value: _format_time(value),
                label_visibility="collapsed",
            )

        segment = start_seconds // window_seconds
        st.markdown(
            f'<div class="window-subtitle">Window {segment + 1} of '
            f'{config["windows_per_file"]} · {_format_time(start_seconds)}–'
            f'{_format_time(start_seconds + window_seconds)}</div>',
            unsafe_allow_html=True,
        )
        explorer_column, details_column = st.columns([3, 1], gap="large")
        with explorer_column:
            sed = config.get("sed", {})
            mel = audio_to_mel(
                chunks[segment : segment + 1],
                config["sample_rate"],
                sed.get("n_mels", 256),
                sed.get("n_fft", 2048),
                sed.get("hop_length", 512),
                sed.get("fmin", 20),
                sed.get("fmax", 16_000),
                sed.get("top_db", 80),
            )[0, 0]
            _render_mel(mel, start_seconds, window_seconds)
            if model_name == "model_22":
                st.caption(
                    "Diagnostic SED-compatible log-mel view. Model 22 sends the raw "
                    "five-second waveform to Perch; this image is not an extra model input."
                )
            else:
                st.caption("This is the normalized log-mel preprocessing used by the SED branch.")
        with details_column:
            st.audio(uploaded.getvalue(), format=uploaded.type)
            first, second = st.columns(2)
            first.metric("Duration", f"{duration:.1f}s")
            second.metric("Source rate", f"{original_rate / 1000:.1f}k")
            st.metric("Current window", f"{_format_time(start_seconds)}–{_format_time(start_seconds + 5)}")
            if duration < 60:
                st.info(f"The final {60 - duration:.1f}s is zero-padded for inference.")
            elif duration > 60:
                st.warning(f"Only the first 60s is used; {duration - 60:.1f}s is outside model input.")

        with st.expander("Waveform and model diagnostics"):
            st.line_chart(_waveform_frame(waveform, original_rate), height=220)
            missing = _required_artifacts(config, model_name)
            if missing:
                st.error("Inference cannot start until the missing artifacts below are added.")
                st.code("\n".join(str(path) for path in missing))
            else:
                st.success("All required model and metadata artifacts are available.")
            st.json(
                {
                    "model": model_name,
                    "sample_rate": config["sample_rate"],
                    "window_seconds": config["window_seconds"],
                    "windows_per_file": config["windows_per_file"],
                    "num_classes": config["num_classes"],
                    "device": config["device"],
                }
            )

        missing = _required_artifacts(config, model_name)
        st.divider()
        st.markdown('<div class="section-kicker">MODEL OUTPUT</div>', unsafe_allow_html=True)
        prediction_heading, prediction_action = st.columns([3, 1], vertical_alignment="center")
        with prediction_heading:
            st.markdown("### Species predictions")
            st.caption("Scores use the complete 60-second, twelve-window inference path.")
        with prediction_action:
            run_clicked = st.button(
                "Run inference", type="primary", disabled=bool(missing), use_container_width=True
            )
        if missing:
            missing_text = "\n".join(f"- `{path}`" for path in missing)
            st.error(
                "Inference is disabled because required artifacts are missing:\n\n"
                f"{missing_text}"
            )
            missing_signature = f"{model_name}:" + "|".join(map(str, missing))
            if st.session_state.get("logged_missing_artifacts") != missing_signature:
                logger.error(
                    "Inference disabled: model=%s missing_artifacts=%s",
                    model_name,
                    [str(path) for path in missing],
                )
                st.session_state["logged_missing_artifacts"] = missing_signature
        upload_digest = hashlib.sha256(uploaded.getvalue()).hexdigest()
        cache_key = f"{model_name}:{upload_digest}"
        if run_clicked:
            started = time.perf_counter()
            logger.info(
                "Inference requested: model=%s upload=%s sha256=%s",
                model_name,
                uploaded.name,
                upload_digest,
            )
            with st.status("Running inference…", expanded=True) as status:
                st.write("Loading Perch and trained heads")
                try:
                    prediction = run_paths(config, model_name, [temporary_path])
                    original_stem = Path(uploaded.name).stem
                    prediction["row_id"] = prediction["row_id"].str.replace(
                        temporary_path.stem, original_stem, regex=False
                    )
                except Exception as error:
                    logger.exception(
                        "Inference failed: model=%s upload=%s temporary_path=%s",
                        model_name,
                        uploaded.name,
                        temporary_path,
                    )
                    status.update(label="Inference failed", state="error")
                    st.exception(error)
                else:
                    elapsed = time.perf_counter() - started
                    logger.info(
                        "Inference completed: model=%s upload=%s rows=%d elapsed_seconds=%.3f",
                        model_name,
                        uploaded.name,
                        len(prediction),
                        elapsed,
                    )
                    st.session_state["prediction"] = prediction
                    st.session_state["prediction_key"] = cache_key
                    st.session_state["prediction_elapsed"] = elapsed
                    status.update(label=f"Inference complete in {elapsed:.1f}s", state="complete")

        if st.session_state.get("prediction_key") == cache_key:
            prediction = st.session_state["prediction"]
            taxonomy = pd.read_csv(config["paths"]["competition_dir"] / "taxonomy.csv")
            top, by_window = _prediction_tables(prediction, taxonomy, top_k)
            winner = top.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Top bird", winner["Bird"])
            col2.metric("Peak score", f"{winner['Peak score']:.4f}")
            col3.metric("Inference time", f"{st.session_state['prediction_elapsed']:.1f}s")
            chart_column, table_column = st.columns([1, 1], gap="large")
            with chart_column:
                st.markdown("#### Top birds across the recording")
                st.bar_chart(top.set_index("Bird")["Peak score"], horizontal=True)
            with table_column:
                st.markdown("#### Ranked results")
                st.dataframe(
                    top.style.format({"Peak score": "{:.5f}"}),
                    hide_index=True,
                    use_container_width=True,
                )
            st.markdown("#### Top detections by five-second window")
            st.dataframe(
                by_window.style.format({"Score": "{:.5f}"}),
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                "Download prediction CSV",
                prediction.to_csv(index=False),
                file_name=f"{Path(uploaded.name).stem}_{model_name}.csv",
                mime="text/csv",
            )
        else:
            st.markdown(
                """
                <div class="spectrogram-empty" style="min-height:180px">
                  <div><strong>No inference result yet</strong>
                  <span>Run the selected model to populate species predictions.</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled dashboard error")
        raise
