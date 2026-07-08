#!/usr/bin/env python3
"""Streamlit UI for the raw IQ RF signal pipeline."""

from __future__ import annotations

import csv
import json
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rf_pipeline.data import IQMetadata, read_iq_segment
from rf_pipeline.inference import PipelineConfig, run_pipeline, run_waterfall_batch_pipeline
from rf_pipeline.preprocessing import SpectrogramConfig, iq_to_spectrogram, save_spectrogram


APP_RUN_ROOT = ROOT / "runs" / "ui_pipeline"


def main() -> None:
    st.set_page_config(page_title="RF IQ Pipeline Demo", layout="wide")
    st.title("RF IQ Pipeline Demo")
    init_session()
    st.caption(
        "Step through raw IQ upload, spectrogram/waterfall rendering, model inference, "
        "parameter estimation, and export."
    )

    render_stepper()
    st.divider()
    step_input()
    st.divider()
    step_preprocessing()
    st.divider()
    step_inference()
    st.divider()
    step_export()


def init_session() -> None:
    defaults = {
        "step": 1,
        "run_dir": None,
        "iq_path": None,
        "metadata": None,
        "preprocess": None,
        "preview": None,
        "result": None,
        "archive_path": None,
        "pipeline_mode": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_stepper() -> None:
    labels = ["1. IQ input", "2. Spectrogram", "3. Detection", "4. Export"]
    cols = st.columns(4)
    current = int(st.session_state.step)
    for index, (col, label) in enumerate(zip(cols, labels), start=1):
        status = "Done" if index < current else "Active" if index == current else "Locked"
        col.metric(label, status)
    if st.button("Start new run"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def step_input() -> None:
    st.subheader("Step 1 - Load raw IQ")
    input_mode = st.radio(
        "Input source",
        ["Upload file", "Use local file path"],
        horizontal=True,
        disabled=st.session_state.step > 1,
    )
    iq_upload = None
    local_iq_path = ""
    if input_mode == "Upload file":
        iq_upload = st.file_uploader("Raw IQ file", type=["iq", "dat", "bin"], disabled=st.session_state.step > 1)
        st.caption("For very large files, local path is usually faster and avoids browser upload limits.")
    else:
        local_iq_path = st.text_input(
            "Local IQ path",
            value="",
            placeholder=r"D:\data\sample.iq",
            disabled=st.session_state.step > 1,
        )
    c1, c2, c3 = st.columns(3)
    with c1:
        dtype = st.selectbox("IQ dtype", ["float32", "int16", "complex64"], index=0, disabled=st.session_state.step > 1)
    with c2:
        sample_rate = st.number_input(
            "Sample rate (Hz)",
            min_value=1.0,
            value=100_000_000.0,
            step=1_000_000.0,
            disabled=st.session_state.step > 1,
        )
    with c3:
        center_frequency = st.number_input(
            "Center frequency (Hz)",
            min_value=0.0,
            value=2_400_000_000.0,
            step=1_000_000.0,
            disabled=st.session_state.step > 1,
        )

    st.markdown("**Trim IQ segment**")
    trim_enabled = st.checkbox(
        "Use only a smaller time segment",
        value=True,
        disabled=st.session_state.step > 1,
        help="Recommended for large IQ files. The app copies only this segment to the run folder.",
    )
    t1, t2 = st.columns(2)
    with t1:
        trim_start_sec = st.number_input(
            "Start time (sec)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.6f",
            disabled=st.session_state.step > 1 or not trim_enabled,
        )
    with t2:
        trim_duration_sec = st.number_input(
            "Duration (sec)",
            min_value=0.000001,
            value=0.03,
            step=0.01,
            format="%.6f",
            disabled=st.session_state.step > 1 or not trim_enabled,
        )

    if st.session_state.iq_path:
        st.success(f"Loaded: {st.session_state.iq_path}")
        return
    source_ready = iq_upload is not None if input_mode == "Upload file" else bool(local_iq_path.strip())
    if st.button("Save IQ and continue", type="primary", disabled=not source_ready):
        run_dir = APP_RUN_ROOT / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = IQMetadata(
            sample_rate_hz=float(sample_rate),
            center_frequency_hz=float(center_frequency),
            dtype=dtype,
        )
        try:
            if input_mode == "Upload file":
                source_path = save_upload(iq_upload, run_dir / "input" / iq_upload.name)
            else:
                source_path = Path(local_iq_path.strip())
                if not source_path.is_file():
                    raise FileNotFoundError(f"IQ file not found: {source_path}")
            iq_path = materialize_iq_segment(
                source_path=source_path,
                target_dir=run_dir / "input",
                metadata=metadata,
                trim_enabled=trim_enabled,
                start_sec=float(trim_start_sec),
                duration_sec=float(trim_duration_sec),
            )
        except Exception as exc:
            st.exception(exc)
            return
        st.session_state.run_dir = run_dir
        st.session_state.iq_path = iq_path
        st.session_state.metadata = asdict(metadata)
        st.session_state.step = 2
        st.rerun()


def step_preprocessing() -> None:
    st.subheader("Step 2 - Render spectrogram or waterfall")
    if st.session_state.step < 2:
        st.info("Complete Step 1 first.")
        return

    c1, c2 = st.columns([1.0, 1.2])
    with c1:
        output_mode = st.radio("Render output", ["Static image", "Waterfall video + static image"], index=0)
        render_mode = st.radio(
            "Render mode",
            ["Train-compatible Matplotlib", "Fast OpenCV preview"],
            index=0,
            help="Use Matplotlib/hot for inference if the model was trained with the RFUAV scripts.",
        )
        stft_point = st.select_slider(
            "STFT point",
            options=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
            value=16384,
        )
        dynamic_range = st.slider(
            "Dynamic range (dB)",
            20.0,
            120.0,
            70.0,
            5.0,
            help="Power range mapped to image brightness. Lower values increase contrast; higher values preserve weak details.",
        )
        default_cmap_index = 0 if render_mode.startswith("Train-compatible") else 1
        colormap = st.selectbox("Colormap", ["hot", "jet", "turbo", "viridis", "gray"], index=default_cmap_index)
        p1, p2 = st.columns(2)
        with p1:
            image_width = st.number_input("Image width", min_value=128, value=960, step=32)
        with p2:
            image_height = st.number_input("Image height", min_value=128, value=720, step=32)
        video_window_sec = st.number_input(
            "Segment duration (sec)",
            min_value=0.0001,
            value=0.03,
            step=0.001,
            format="%.4f",
            help="Length of each IQ segment. Each segment is rendered into one spectrogram frame for inference/video.",
        )
        video_hop_sec = st.number_input(
            "Segment step / hop (sec)",
            min_value=0.0001,
            value=0.03,
            step=0.0005,
            format="%.4f",
            help="Use the same value as segment duration for equal non-overlapping chunks. Smaller values create overlapping chunks.",
        )
        with st.expander("What do these parameters mean?"):
            st.markdown(
                """
                - **Dynamic range**: the dB span compressed into the image color scale. Lower values make strong signals pop; higher values keep weaker signals visible.
                - **Segment duration**: length of each IQ chunk. Each chunk becomes one spectrogram image and one waterfall frame.
                - **Segment step / hop**: distance between chunk starts. Set equal to duration for non-overlapping equal chunks.
                """
            )

        if st.button("Generate preview", type="primary"):
            try:
                with st.spinner("Rendering spectrogram/waterfall..."):
                    preview = render_preprocessing_preview(
                        st.session_state.iq_path,
                        IQMetadata(**st.session_state.metadata),
                        st.session_state.run_dir,
                        output_mode,
                        int(stft_point),
                        float(dynamic_range),
                        "matplotlib" if render_mode.startswith("Train-compatible") else "opencv",
                        colormap,
                        int(image_width) or None,
                        int(image_height) or None,
                        float(video_window_sec),
                        float(video_hop_sec),
                    )
                st.session_state.preprocess = {
                    "output_mode": output_mode,
                    "stft_point": int(stft_point),
                    "dynamic_range": float(dynamic_range),
                    "colormap": colormap,
                    "render_mode": "matplotlib" if render_mode.startswith("Train-compatible") else "opencv",
                    "image_width": int(image_width),
                    "image_height": int(image_height),
                    "video_window_sec": float(video_window_sec),
                    "video_hop_sec": float(video_hop_sec),
                }
                st.session_state.preview = preview
                st.session_state.step = max(st.session_state.step, 3)
                st.rerun()
            except Exception as exc:
                st.exception(exc)

    with c2:
        preview = st.session_state.preview
        if preview:
            st.image(preview["spectrogram_path"], caption="Static spectrogram", use_container_width=True)
            if preview.get("waterfall_video_path") and Path(preview["waterfall_video_path"]).is_file():
                st.video(preview["waterfall_video_path"])
        else:
            st.info("Preview will appear here after rendering.")


def step_inference() -> None:
    st.subheader("Step 3 - Detection, classification, and estimation")
    if st.session_state.step < 3:
        st.info("Complete Step 2 first.")
        return

    c1, c2 = st.columns([1.0, 1.2])
    with c1:
        pipeline_mode = st.radio(
            "Inference mode",
            ["Detection 2 classes", "Detection 1 class + classification"],
            index=0,
        )
        detector_upload = st.file_uploader("Detector .pt", type=["pt"], key="detector_upload")
        detector_path_text = st.text_input("Or detector path", value="")
        classifier_upload = None
        classifier_path_text = ""
        if pipeline_mode == "Detection 1 class + classification":
            classifier_upload = st.file_uploader("Classifier .pt", type=["pt"], key="classifier_upload")
            classifier_path_text = st.text_input("Or classifier path", value="")
        architecture = st.selectbox("Detector architecture", ["auto", "yolo", "rtdetr"], index=0)
        p1, p2, p3 = st.columns(3)
        with p1:
            imgsz = st.select_slider("Image size", options=[320, 512, 640, 768, 1024], value=640)
        with p2:
            conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
        with p3:
            iou = st.slider("IoU", 0.0, 1.0, 0.70, 0.01)
        device = st.text_input("Device", value="", placeholder="empty, cpu, 0")
        preprocess = st.session_state.preprocess
        inference_source = "Static spectrogram"
        waterfall_overlay_mode = "map_static"
        if preprocess and preprocess["output_mode"].startswith("Waterfall"):
            inference_source = st.radio(
                "Inference source",
                ["Static spectrogram", "Segment spectrograms -> stitched waterfall"],
                index=1,
                help=(
                    "Cuts the IQ into equal segments, renders each segment to a spectrogram, "
                    "runs batched inference on those images, then stitches frames into a complete waterfall video."
                ),
            )
            waterfall_overlay_choice = st.radio(
                "Waterfall detection overlay",
                ["Map static detections to video", "Detect every waterfall frame"],
                index=0,
                help=(
                    "This applies only to Static spectrogram mode. Segment mode detects each segment spectrogram and draws it on the matching video frame."
                ),
            )
            waterfall_overlay_mode = "per_frame" if waterfall_overlay_choice.startswith("Detect") else "map_static"
        batch_size = st.number_input("Inference batch size", min_value=1, max_value=256, value=16, step=1)

        if st.button("Run detection and estimate", type="primary"):
            try:
                run_dir = st.session_state.run_dir
                detector_weights = resolve_model_file(detector_upload, detector_path_text, run_dir / "models", "detector.pt")
                if detector_weights is None:
                    st.warning("No detector was provided. The UI will use a heuristic signal detector for demo only.")
                classifier_weights = None
                if pipeline_mode == "Detection 1 class + classification":
                    classifier_weights = resolve_model_file(classifier_upload, classifier_path_text, run_dir / "models", "classifier.pt")
                    if classifier_weights is None:
                        st.warning("No classifier was provided. Detector class names will be used as fallback.")

                metadata = IQMetadata(**st.session_state.metadata)
                config = PipelineConfig(
                    output_dir=run_dir / "output",
                    detector_weights=detector_weights,
                    classifier_weights=classifier_weights,
                    architecture=architecture,
                    stft_point=preprocess["stft_point"],
                    dynamic_range_db=preprocess["dynamic_range"],
                    colormap=preprocess["colormap"],
                    render_mode=preprocess["render_mode"],
                    image_width=preprocess["image_width"],
                    image_height=preprocess["image_height"],
                    imgsz=int(imgsz),
                    conf=float(conf),
                    iou=float(iou),
                    batch=int(batch_size),
                    device=device.strip() or None,
                    save_video=preprocess["output_mode"].startswith("Waterfall"),
                    waterfall_detection_mode=waterfall_overlay_mode,
                    video_window_samples=max(int(preprocess["video_window_sec"] * metadata.sample_rate_hz), preprocess["stft_point"]),
                    video_hop_samples=(
                        max(int(preprocess["video_window_sec"] * metadata.sample_rate_hz), preprocess["stft_point"])
                        if inference_source.startswith("Segment")
                        else max(int(preprocess["video_hop_sec"] * metadata.sample_rate_hz), 1)
                    ),
                )
                with st.spinner("Running model and estimating signal parameters..."):
                    if inference_source.startswith("Segment"):
                        result = run_waterfall_batch_pipeline(st.session_state.iq_path, metadata, config)
                    else:
                        result = run_pipeline(st.session_state.iq_path, metadata, config)
                    write_outputs(result, run_dir, pipeline_mode, waterfall_overlay_mode)
                    archive_path = make_archive(run_dir)
                st.session_state.result = result
                st.session_state.archive_path = archive_path
                st.session_state.pipeline_mode = pipeline_mode
                st.session_state.step = 4
                st.rerun()
            except Exception as exc:
                st.exception(exc)

    with c2:
        result = st.session_state.result
        if result:
            st.image(result.overlay_path, caption="Detection overlay", use_container_width=True)
            detection_video = getattr(result, "waterfall_detection_video_path", None)
            if detection_video and Path(detection_video).is_file():
                st.video(detection_video)
            st.metric("Detections", len(result.detections))
            show_performance_metrics(result)
        else:
            st.info("Detection overlay and count will appear here.")


def step_export() -> None:
    st.subheader("Step 4 - Review and export")
    if st.session_state.step < 4 or not st.session_state.result:
        st.info("Complete Step 3 first.")
        return
    show_results(st.session_state.result, st.session_state.run_dir, st.session_state.archive_path)


def render_preprocessing_preview(
    iq_path: Path,
    metadata: IQMetadata,
    run_dir: Path,
    output_mode: str,
    stft_point: int,
    dynamic_range: float,
    render_mode: str,
    colormap: str,
    image_width: int | None,
    image_height: int | None,
    video_window_sec: float,
    video_hop_sec: float,
) -> dict[str, str | int | None]:
    preview_dir = run_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    segment_samples = max(int(video_window_sec * metadata.sample_rate_hz), stft_point)
    hop_samples = max(int(video_hop_sec * metadata.sample_rate_hz), 1)
    iq = read_iq_segment(iq_path, metadata.dtype, start_sample=0, sample_count=segment_samples)
    spectrogram_config = SpectrogramConfig(
        sample_rate_hz=metadata.sample_rate_hz,
        center_frequency_hz=metadata.center_frequency_hz,
        stft_point=stft_point,
        dynamic_range_db=dynamic_range,
        colormap=colormap,
        render_mode=render_mode,
        image_width=image_width,
        image_height=image_height,
    )
    frame = iq_to_spectrogram(iq, spectrogram_config)
    spectrogram_path = preview_dir / "spectrogram.png"
    width, height = save_spectrogram(frame, spectrogram_path)

    waterfall_video_path = None
    frame_count = 0
    if output_mode.startswith("Waterfall"):
        candidate = preview_dir / "waterfall.mp4"
        frame_count = save_preview_waterfall(
            iq_path=iq_path,
            metadata=metadata,
            path=candidate,
            spectrogram_config=spectrogram_config,
            segment_samples=segment_samples,
            hop_samples=hop_samples,
            max_frames=8,
        )
        if frame_count > 0:
            waterfall_video_path = str(candidate)

    return {
        "spectrogram_path": str(spectrogram_path),
        "waterfall_video_path": waterfall_video_path,
        "image_width": width,
        "image_height": height,
        "waterfall_frames": frame_count,
    }


def save_preview_waterfall(
    iq_path: Path,
    metadata: IQMetadata,
    path: Path,
    spectrogram_config: SpectrogramConfig,
    segment_samples: int,
    hop_samples: int,
    max_frames: int,
) -> int:
    from rf_pipeline.preprocessing import BrowserVideoWriter

    with BrowserVideoWriter(path) as writer:
        for frame_index in range(max_frames):
            start_sample = frame_index * hop_samples
            iq = read_iq_segment(iq_path, metadata.dtype, start_sample=start_sample, sample_count=segment_samples)
            if iq.size < segment_samples:
                break
            writer.write(iq_to_spectrogram(iq, spectrogram_config).image)
        return writer.frame_count


def materialize_iq_segment(
    source_path: Path,
    target_dir: Path,
    metadata: IQMetadata,
    trim_enabled: bool,
    start_sec: float,
    duration_sec: float,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    if not trim_enabled:
        return source_path

    bytes_per_sample = complex_sample_bytes(metadata.dtype)
    start_sample = int(start_sec * metadata.sample_rate_hz)
    sample_count = max(1, int(duration_sec * metadata.sample_rate_hz))
    byte_offset = start_sample * bytes_per_sample
    byte_count = sample_count * bytes_per_sample
    file_size = source_path.stat().st_size
    if byte_offset >= file_size:
        raise ValueError("Trim start is beyond the end of the IQ file.")
    byte_count = min(byte_count, file_size - byte_offset)
    byte_count -= byte_count % bytes_per_sample
    if byte_count <= 0:
        raise ValueError("Selected segment is empty.")

    target_path = target_dir / f"{source_path.stem}_trim_{start_sec:g}s_{duration_sec:g}s{source_path.suffix}"
    with source_path.open("rb") as src, target_path.open("wb") as dst:
        src.seek(byte_offset)
        remaining = byte_count
        chunk_size = 8 * 1024 * 1024
        while remaining > 0:
            chunk = src.read(min(chunk_size, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    return target_path


def complex_sample_bytes(dtype: str) -> int:
    if dtype == "int16":
        return 4
    if dtype in {"float32", "complex64"}:
        return 8
    raise ValueError(f"Unsupported IQ dtype: {dtype}")


def save_upload(uploaded_file, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(uploaded_file.getbuffer())
    return target


def resolve_model_file(uploaded_file, text_value: str, target_dir: Path, default_name: str) -> Path | None:
    text_value = text_value.strip()
    if uploaded_file is not None:
        return save_upload(uploaded_file, target_dir / uploaded_file.name)
    if text_value:
        path = Path(text_value)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path
    return None


def write_outputs(result, run_dir: Path, pipeline_mode: str, waterfall_overlay_mode: str | None = None) -> None:
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["pipeline_mode"] = pipeline_mode
    payload["waterfall_overlay_mode"] = waterfall_overlay_mode
    (output_dir / "pipeline_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output_dir / "signal_estimates.csv"
    rows = []
    for index, item in enumerate(result.detections, start=1):
        row = {
            "index": index,
            "frame_index": item.get("frame_index", ""),
            "frame_start_sec": item.get("frame_start_sec", ""),
            "frame_end_sec": item.get("frame_end_sec", ""),
            "det_class_id": item["detector"]["class_id"],
            "det_class_name": item["detector"]["class_name"],
            "det_confidence": item["detector"]["confidence"],
            "cls_class_id": item["classification"]["class_id"],
            "cls_class_name": item["classification"]["class_name"],
            "cls_confidence": item["classification"]["confidence"],
            "x1": item["detector"]["xyxy"][0],
            "y1": item["detector"]["xyxy"][1],
            "x2": item["detector"]["xyxy"][2],
            "y2": item["detector"]["xyxy"][3],
            **item["parameters"],
        }
        rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else [
        "index",
        "frame_index",
        "frame_start_sec",
        "frame_end_sec",
        "det_class_id",
        "det_class_name",
        "det_confidence",
        "cls_class_id",
        "cls_class_name",
        "cls_confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "time_start_sec",
        "time_end_sec",
        "duration_sec",
        "freq_low_hz",
        "freq_high_hz",
        "center_frequency_hz",
        "bandwidth_hz",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_archive(run_dir: Path) -> Path:
    archive_path = run_dir / "output_bundle.zip"
    output_dir = run_dir / "output"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir))
    return archive_path


def show_results(result, run_dir: Path, archive_path: Path) -> None:
    left, right = st.columns([1.2, 1.0])
    with left:
        st.subheader("Spectrogram")
        st.image(result.spectrogram_path, use_container_width=True)
        st.subheader("Detections")
        st.caption("Box colors: OFDM = yellow/cyan, FHSS = magenta, other = green.")
        st.image(result.overlay_path, use_container_width=True)
        detection_video = getattr(result, "waterfall_detection_video_path", None)
        if detection_video and Path(detection_video).is_file():
            st.subheader("Waterfall detections")
            st.video(detection_video)
        if result.waterfall_video_path and Path(result.waterfall_video_path).is_file():
            with st.expander("Raw waterfall video"):
                st.video(result.waterfall_video_path)
    with right:
        st.subheader("Signal Parameters")
        csv_path = run_dir / "output" / "signal_estimates.csv"
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.metric("Detections", len(result.detections))
        show_performance_metrics(result)
        segment_dir = getattr(result, "segment_spectrogram_dir", None)
        if segment_dir:
            segment_count = len(list(Path(segment_dir).glob("*.png"))) if Path(segment_dir).is_dir() else 0
            st.caption(f"Segment spectrograms saved: {segment_count} image(s) in {segment_dir}")

        json_path = run_dir / "output" / "pipeline_result.json"
        st.download_button(
            "Download JSON",
            json_path.read_bytes(),
            file_name="pipeline_result.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "Download CSV",
            csv_path.read_bytes(),
            file_name="signal_estimates.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download all outputs",
            archive_path.read_bytes(),
            file_name="rf_pipeline_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with st.expander("Raw JSON"):
        st.json(json.loads((run_dir / "output" / "pipeline_result.json").read_text(encoding="utf-8")))


def show_performance_metrics(result) -> None:
    perf = getattr(result, "performance", {}) or {}
    if not perf:
        return
    fps = perf.get("inference_fps")
    elapsed = perf.get("inference_elapsed_sec")
    frames = perf.get("inference_frames")
    c1, c2, c3 = st.columns(3)
    c1.metric("Infer FPS", f"{fps:.2f}" if isinstance(fps, (int, float)) else "-")
    c2.metric("Infer time", f"{elapsed:.3f}s" if isinstance(elapsed, (int, float)) else "-")
    c3.metric("Infer frames", frames if frames is not None else "-")


if __name__ == "__main__":
    main()
