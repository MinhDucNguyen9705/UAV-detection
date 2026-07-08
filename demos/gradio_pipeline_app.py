#!/usr/bin/env python3
"""Gradio UI for the raw IQ RF signal pipeline.

Run locally:
  python demos/gradio_pipeline_app.py

Run on Kaggle/remote notebook:
  python demos/gradio_pipeline_app.py --share
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rf_pipeline.data import IQMetadata, read_iq_segment
from rf_pipeline.inference import PipelineConfig, run_pipeline, run_waterfall_batch_pipeline
from rf_pipeline.preprocessing import BrowserVideoWriter, SpectrogramConfig, iq_to_spectrogram, save_spectrogram


APP_RUN_ROOT = ROOT / "runs" / "gradio_pipeline"


def resolve_local_iq_path(raw_value: str) -> Path:
    text = raw_value.strip().strip("\"'")
    candidates: list[Path] = []

    def add_candidate(value: str) -> None:
        if not value:
            return
        path = Path(value).expanduser()
        if path not in candidates:
            candidates.append(path)

    add_candidate(text)
    normalized = text.replace("\\", "/")
    add_candidate(normalized)
    if normalized.startswith("kaggle/"):
        add_candidate(f"/{normalized}")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    hint = ""
    if os.name == "nt" and normalized.startswith("/kaggle/"):
        hint = (
            "\nYou entered a Kaggle/Linux path, but this Gradio server is running on Windows. "
            "Use the Windows file path or run this app inside Kaggle with --share.\n"
        )
    tried = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"IQ file not found.{hint}\nCurrent working directory: {Path.cwd()}\nTried:\n{tried}")


def copy_upload(upload_path: str | Path, target: Path) -> Path:
    source = Path(upload_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


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

    from rf_pipeline.data import complex_sample_bytes

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


def load_iq_step(
    input_source: str,
    iq_upload: str | None,
    local_iq_path: str,
    dtype: str,
    sample_rate_hz: float,
    center_frequency_hz: float,
    trim_enabled: bool,
    trim_start_sec: float,
    trim_duration_sec: float,
) -> tuple[dict, str]:
    run_dir = APP_RUN_ROOT / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = IQMetadata(
        sample_rate_hz=float(sample_rate_hz),
        center_frequency_hz=float(center_frequency_hz),
        dtype=dtype,
    )

    if input_source == "Upload file":
        if not iq_upload:
            raise gr.Error("Please upload an IQ file.")
        source_path = copy_upload(iq_upload, run_dir / "input" / Path(iq_upload).name)
    else:
        source_path = resolve_local_iq_path(local_iq_path)

    iq_path = materialize_iq_segment(
        source_path=source_path,
        target_dir=run_dir / "input",
        metadata=metadata,
        trim_enabled=trim_enabled,
        start_sec=float(trim_start_sec),
        duration_sec=float(trim_duration_sec),
    )
    state = {
        "run_dir": str(run_dir),
        "iq_path": str(iq_path),
        "metadata": asdict(metadata),
    }
    return state, f"Loaded IQ: {iq_path}"


def save_preview_waterfall(
    iq_path: Path,
    metadata: IQMetadata,
    path: Path,
    spectrogram_config: SpectrogramConfig,
    segment_samples: int,
    hop_samples: int,
    fps: float,
    max_frames: int = 8,
) -> int:
    with BrowserVideoWriter(path, fps=fps) as writer:
        for frame_index in range(max_frames):
            start_sample = frame_index * hop_samples
            iq = read_iq_segment(iq_path, metadata.dtype, start_sample=start_sample, sample_count=segment_samples)
            if iq.size < segment_samples:
                break
            writer.write(iq_to_spectrogram(iq, spectrogram_config).image)
        return writer.frame_count


def render_preview_step(
    state: dict,
    output_mode: str,
    render_mode_label: str,
    stft_point: int,
    dynamic_range_db: float,
    colormap: str,
    image_width: int,
    image_height: int,
    segment_duration_sec: float,
    segment_hop_sec: float,
    output_video_fps: float,
) -> tuple[dict, str, str | None, str | None]:
    if not state or not state.get("iq_path"):
        raise gr.Error("Load IQ first.")

    run_dir = Path(state["run_dir"])
    iq_path = Path(state["iq_path"])
    metadata = IQMetadata(**state["metadata"])
    render_mode = "matplotlib" if render_mode_label.startswith("Train-compatible") else "opencv"
    preview_dir = run_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    segment_samples = max(int(segment_duration_sec * metadata.sample_rate_hz), int(stft_point))
    hop_samples = max(int(segment_hop_sec * metadata.sample_rate_hz), 1)
    iq = read_iq_segment(iq_path, metadata.dtype, start_sample=0, sample_count=segment_samples)
    if iq.size < segment_samples:
        raise gr.Error("IQ file/segment is shorter than the configured segment duration.")

    spectrogram_config = SpectrogramConfig(
        sample_rate_hz=metadata.sample_rate_hz,
        center_frequency_hz=metadata.center_frequency_hz,
        stft_point=int(stft_point),
        dynamic_range_db=float(dynamic_range_db),
        colormap=colormap,
        render_mode=render_mode,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    frame = iq_to_spectrogram(iq, spectrogram_config)
    spectrogram_path = preview_dir / "spectrogram.png"
    save_spectrogram(frame, spectrogram_path)

    video_path = None
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
            fps=float(output_video_fps),
        )
        if frame_count:
            video_path = str(candidate)

    state["preprocess"] = {
        "output_mode": output_mode,
        "stft_point": int(stft_point),
        "dynamic_range": float(dynamic_range_db),
        "colormap": colormap,
        "render_mode": render_mode,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "segment_duration_sec": float(segment_duration_sec),
        "segment_hop_sec": float(segment_hop_sec),
    }
    return state, f"Preview rendered. Waterfall preview frames: {frame_count}", str(spectrogram_path), video_path


def resolve_model_file(upload_path: str | None, text_path: str) -> Path | None:
    if upload_path:
        return Path(upload_path)
    text_path = text_path.strip().strip("\"'")
    if text_path:
        path = Path(text_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path
    return None


def write_outputs(result, run_dir: Path, pipeline_mode: str, inference_source: str) -> tuple[Path, Path, Path, pd.DataFrame]:
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["pipeline_mode"] = pipeline_mode
    payload["inference_source"] = inference_source
    json_path = output_dir / "pipeline_result.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for index, item in enumerate(result.detections, start=1):
        rows.append(
            {
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
        )
    df = pd.DataFrame(rows)
    csv_path = output_dir / "signal_estimates.csv"
    df.to_csv(csv_path, index=False)

    archive_path = run_dir / "output_bundle.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir))
    return json_path, csv_path, archive_path, df


def inference_step(
    state: dict,
    pipeline_mode: str,
    detector_upload: str | None,
    detector_path_text: str,
    classifier_upload: str | None,
    classifier_path_text: str,
    architecture: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    inference_source: str,
    waterfall_overlay_mode_label: str,
    batch_size: int,
    output_video_fps: float,
) -> tuple[dict, str, str | None, str | None, pd.DataFrame, dict, str | None, str | None, str | None]:
    if not state or not state.get("iq_path"):
        raise gr.Error("Load IQ first.")
    if not state.get("preprocess"):
        raise gr.Error("Render preview first.")

    run_dir = Path(state["run_dir"])
    metadata = IQMetadata(**state["metadata"])
    preprocess = state["preprocess"]
    detector_weights = resolve_model_file(detector_upload, detector_path_text)
    classifier_weights = None
    if pipeline_mode == "Detection 1 class + classification":
        classifier_weights = resolve_model_file(classifier_upload, classifier_path_text)

    segment_samples = max(int(preprocess["segment_duration_sec"] * metadata.sample_rate_hz), preprocess["stft_point"])
    hop_samples = max(int(preprocess["segment_hop_sec"] * metadata.sample_rate_hz), 1)
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
        waterfall_detection_mode="per_frame" if waterfall_overlay_mode_label.startswith("Detect") else "map_static",
        video_window_samples=segment_samples,
        video_hop_samples=hop_samples,
        video_fps=float(output_video_fps),
    )

    if inference_source.startswith("Segment"):
        result = run_waterfall_batch_pipeline(Path(state["iq_path"]), metadata, config)
    else:
        result = run_pipeline(Path(state["iq_path"]), metadata, config)

    json_path, csv_path, archive_path, df = write_outputs(result, run_dir, pipeline_mode, inference_source)
    state["result"] = asdict(result)
    perf = result.performance
    status = (
        f"Done. Detections: {len(result.detections)}. "
        f"Infer FPS: {perf.get('inference_fps', 0):.2f}, "
        f"frames: {perf.get('inference_frames')}, time: {perf.get('inference_elapsed_sec', 0):.3f}s."
    )
    return (
        state,
        status,
        result.overlay_path,
        result.waterfall_detection_video_path,
        df,
        perf,
        str(json_path),
        str(csv_path),
        str(archive_path),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RF IQ Pipeline Demo") as app:
        state = gr.State({})
        gr.Markdown(
            "# RF IQ Pipeline Demo\n"
            "Gradio UI: upload or reference IQ, render segment spectrograms/waterfall, "
            "run batched detection, estimate parameters, and export outputs."
        )

        with gr.Tab("1. IQ Input"):
            input_source = gr.Radio(["Upload file", "Use local file path"], value="Upload file", label="Input source")
            iq_upload = gr.File(label="Raw IQ file", file_types=[".iq", ".dat", ".bin"], type="filepath")
            local_iq_path = gr.Textbox(label="Local IQ path", placeholder="/kaggle/working/sample.iq")
            with gr.Row():
                dtype = gr.Dropdown(["float32", "int16", "complex64"], value="float32", label="IQ dtype")
                sample_rate = gr.Number(value=100_000_000, label="Sample rate (Hz)")
                center_frequency = gr.Number(value=2_400_000_000, label="Center frequency (Hz)")
            trim_enabled = gr.Checkbox(value=True, label="Use only a smaller time segment")
            with gr.Row():
                trim_start = gr.Number(value=0.0, label="Start time (sec)")
                trim_duration = gr.Number(value=0.03, label="Duration (sec)")
            load_button = gr.Button("Save IQ and continue", variant="primary")
            load_status = gr.Markdown()

        with gr.Tab("2. Spectrogram / Waterfall Preview"):
            output_mode = gr.Radio(["Static image", "Waterfall video + static image"], value="Waterfall video + static image", label="Render output")
            render_mode = gr.Radio(["Train-compatible Matplotlib", "Fast OpenCV preview"], value="Train-compatible Matplotlib", label="Render mode")
            with gr.Row():
                stft_point = gr.Dropdown([128, 256, 512, 1024, 2048, 4096, 8192, 16384], value=16384, label="STFT point")
                dynamic_range = gr.Slider(20, 120, value=70, step=5, label="Dynamic range (dB)")
                colormap = gr.Dropdown(["hot", "jet", "turbo", "viridis", "gray"], value="hot", label="Colormap")
            with gr.Row():
                image_width = gr.Number(value=960, label="Image width")
                image_height = gr.Number(value=720, label="Image height")
            with gr.Row():
                segment_duration = gr.Number(value=0.03, label="Segment duration (sec)")
                segment_hop = gr.Number(value=0.01, label="Segment step / hop (sec)")
                output_video_fps = gr.Number(value=24, label="Output video FPS")
            gr.Markdown(
                "- Segment duration: each IQ chunk becomes one spectrogram and one video frame.\n"
                "- Segment step / hop: use a smaller value than duration for overlapping chunks and smoother video.\n"
                "- Output video FPS: playback FPS of the stitched video; it does not change inference windows."
            )
            preview_button = gr.Button("Generate preview", variant="primary")
            preview_status = gr.Markdown()
            with gr.Row():
                preview_image = gr.Image(label="Spectrogram preview", type="filepath")
                preview_video = gr.Video(label="Waterfall preview")

        with gr.Tab("3. Inference"):
            pipeline_mode = gr.Radio(["Detection 2 classes", "Detection 1 class + classification"], value="Detection 2 classes", label="Inference mode")
            with gr.Row():
                detector_upload = gr.File(label="Detector .pt", file_types=[".pt"], type="filepath")
                detector_path = gr.Textbox(label="Or detector path")
            with gr.Row():
                classifier_upload = gr.File(label="Classifier .pt", file_types=[".pt"], type="filepath")
                classifier_path = gr.Textbox(label="Or classifier path")
            inference_source = gr.Radio(
                ["Static spectrogram", "Segment spectrograms -> stitched waterfall"],
                value="Segment spectrograms -> stitched waterfall",
                label="Inference source",
            )
            waterfall_overlay_mode = gr.Radio(
                ["Map static detections to video", "Detect every waterfall frame"],
                value="Map static detections to video",
                label="Waterfall overlay mode for static inference",
            )
            with gr.Row():
                architecture = gr.Dropdown(["auto", "yolo", "rtdetr"], value="auto", label="Detector architecture")
                imgsz = gr.Dropdown([320, 512, 640, 768, 1024], value=640, label="Image size")
                batch_size = gr.Number(value=16, precision=0, label="Inference batch size")
            with gr.Row():
                conf = gr.Slider(0, 1, value=0.25, step=0.01, label="Confidence")
                iou = gr.Slider(0, 1, value=0.70, step=0.01, label="IoU")
                device = gr.Textbox(label="Device", placeholder="empty, cpu, 0")
            infer_button = gr.Button("Run detection and estimate", variant="primary")

        with gr.Tab("4. Results / Export"):
            infer_status = gr.Markdown()
            with gr.Row():
                overlay_image = gr.Image(label="Detection overlay", type="filepath")
                detection_video = gr.Video(label="Waterfall detections")
            estimates = gr.Dataframe(label="Signal estimates")
            perf_json = gr.JSON(label="Performance")
            with gr.Row():
                json_file = gr.File(label="Download JSON")
                csv_file = gr.File(label="Download CSV")
                zip_file = gr.File(label="Download all outputs")

        load_button.click(
            load_iq_step,
            inputs=[input_source, iq_upload, local_iq_path, dtype, sample_rate, center_frequency, trim_enabled, trim_start, trim_duration],
            outputs=[state, load_status],
        )
        preview_button.click(
            render_preview_step,
            inputs=[
                state,
                output_mode,
                render_mode,
                stft_point,
                dynamic_range,
                colormap,
                image_width,
                image_height,
                segment_duration,
                segment_hop,
                output_video_fps,
            ],
            outputs=[state, preview_status, preview_image, preview_video],
        )
        infer_button.click(
            inference_step,
            inputs=[
                state,
                pipeline_mode,
                detector_upload,
                detector_path,
                classifier_upload,
                classifier_path,
                architecture,
                imgsz,
                conf,
                iou,
                device,
                inference_source,
                waterfall_overlay_mode,
                batch_size,
                output_video_fps,
            ],
            outputs=[state, infer_status, overlay_image, detection_video, estimates, perf_json, json_file, csv_file, zip_file],
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Gradio RF IQ pipeline UI.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio URL. Useful on Kaggle.")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_app().queue().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )
