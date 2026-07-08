"""End-to-end raw IQ inference pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rf_pipeline.data import IQMetadata, iter_iq_windows, load_iq_record
from rf_pipeline.model import HeuristicSpectrogramDetector, ImageClassifier, NullClassifier, UltralyticsDetector
from rf_pipeline.preprocessing import (
    SpectrogramConfig,
    WaterfallConfig,
    iq_to_spectrogram,
    save_frames_video,
    save_spectrogram,
    save_waterfall_video,
    waterfall_frames,
)

from .estimation import estimate_parameters


@dataclass(slots=True)
class PipelineConfig:
    output_dir: Path
    detector_weights: Path | None = None
    classifier_weights: Path | None = None
    architecture: str = "auto"
    stft_point: int = 16384
    dynamic_range_db: float = 70.0
    colormap: str = "hot"
    render_mode: str = "matplotlib"
    image_width: int | None = 960
    image_height: int | None = 720
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    batch: int = 16
    device: str | None = None
    save_video: bool = False
    save_waterfall_detection_video: bool = True
    waterfall_detection_mode: str = "map_static"
    video_window_samples: int = 32768
    video_hop_samples: int = 8192
    video_fps: float = 24.0


@dataclass(slots=True)
class PipelineResult:
    iq_path: str
    spectrogram_path: str
    overlay_path: str
    segment_spectrogram_dir: str | None
    waterfall_video_path: str | None
    waterfall_detection_video_path: str | None
    image_width: int
    image_height: int
    performance: dict[str, Any]
    detections: list[dict[str, Any]]


def run_pipeline(iq_path: Path, metadata: IQMetadata, config: PipelineConfig) -> PipelineResult:
    """Run rawIQ -> spectrogram -> detection/classification -> parameter estimation."""

    _guard_static_memory(iq_path, metadata, config)
    iq, metadata = load_iq_record(iq_path, metadata)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    spectrogram_config = SpectrogramConfig(
        sample_rate_hz=metadata.sample_rate_hz,
        center_frequency_hz=metadata.center_frequency_hz,
        stft_point=config.stft_point,
        dynamic_range_db=config.dynamic_range_db,
        colormap=config.colormap,
        render_mode=config.render_mode,
        image_width=config.image_width,
        image_height=config.image_height,
    )
    frame = iq_to_spectrogram(iq, spectrogram_config)
    spectrogram_path = config.output_dir / "spectrogram.png"
    image_width, image_height = save_spectrogram(frame, spectrogram_path)

    video_path: Path | None = None
    waterfall_config = WaterfallConfig(
        window_samples=config.video_window_samples,
        hop_samples=config.video_hop_samples,
        fps=config.video_fps,
    )
    if config.save_video:
        candidate_video_path = config.output_dir / "waterfall.mp4"
        frame_count = save_waterfall_video(
            iq,
            candidate_video_path,
            spectrogram_config,
            waterfall_config,
        )
        if frame_count > 0:
            video_path = candidate_video_path

    if config.detector_weights:
        detector = UltralyticsDetector(
            config.detector_weights,
            architecture=config.architecture,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            batch=config.batch,
        )
    else:
        detector = HeuristicSpectrogramDetector(class_name="signal")

    classifier = ImageClassifier(config.classifier_weights, device=config.device) if config.classifier_weights else NullClassifier()
    detections = []
    infer_start = time.perf_counter()
    raw_detections = detector.predict([spectrogram_path])[spectrogram_path]
    infer_elapsed = time.perf_counter() - infer_start
    crop_dir = config.output_dir / "classification_crops"
    for detection_index, detection in enumerate(raw_detections):
        crop_path = _save_detection_crop(spectrogram_path, crop_dir / f"static_det_{detection_index:04d}.png", detection.xyxy)
        classification = classifier.classify_crop(
            spectrogram_path,
            detection.xyxy,
            detection.class_name,
            detection.class_id,
            detection.confidence,
        )
        estimate = estimate_parameters(
            detection.xyxy,
            image_width=image_width,
            image_height=image_height,
            segment_start_sec=metadata.start_time_sec,
            segment_end_sec=metadata.start_time_sec + frame.duration_sec,
            freq_min_hz=frame.freq_min_hz,
            freq_max_hz=frame.freq_max_hz,
        )
        detections.append(
            {
                "detector": asdict(detection),
                "classification": asdict(classification),
                "classification_crop_path": str(crop_path) if crop_path else None,
                "parameters": asdict(estimate),
            }
        )

    overlay_path = config.output_dir / "detections_overlay.png"
    _save_overlay(spectrogram_path, overlay_path, detections)
    waterfall_detection_video_path: Path | None = None
    if config.save_video and config.save_waterfall_detection_video:
        candidate_detection_video = config.output_dir / "waterfall_detections.mp4"
        if config.waterfall_detection_mode == "per_frame":
            frame_count = _save_waterfall_detection_video(
                iq=iq,
                path=candidate_detection_video,
                spectrogram_config=spectrogram_config,
                waterfall_config=waterfall_config,
                detector=detector,
                classifier=classifier,
            )
        else:
            frame_count = _save_mapped_waterfall_detection_video(
                iq=iq,
                path=candidate_detection_video,
                spectrogram_config=spectrogram_config,
                waterfall_config=waterfall_config,
                detections=detections,
                segment_start_sec=metadata.start_time_sec,
                full_duration_sec=frame.duration_sec,
            )
        if frame_count > 0:
            waterfall_detection_video_path = candidate_detection_video

    return PipelineResult(
        iq_path=str(iq_path),
        spectrogram_path=str(spectrogram_path),
        overlay_path=str(overlay_path),
        segment_spectrogram_dir=None,
        waterfall_video_path=str(video_path) if video_path else None,
        waterfall_detection_video_path=str(waterfall_detection_video_path) if waterfall_detection_video_path else None,
        image_width=image_width,
        image_height=image_height,
        detections=detections,
        performance={
            "mode": "static_spectrogram",
            "inference_frames": 1,
            "inference_elapsed_sec": infer_elapsed,
            "inference_fps": (1.0 / infer_elapsed) if infer_elapsed > 0 else None,
            "num_detections": len(detections),
            "batch_size": config.batch,
        },
    )


def run_waterfall_batch_pipeline(iq_path: Path, metadata: IQMetadata, config: PipelineConfig) -> PipelineResult:
    """Cut IQ into equal-duration segments, infer spectrograms in batches, and stitch video."""

    import cv2

    config.output_dir.mkdir(parents=True, exist_ok=True)
    spectrogram_config = SpectrogramConfig(
        sample_rate_hz=metadata.sample_rate_hz,
        center_frequency_hz=metadata.center_frequency_hz,
        stft_point=config.stft_point,
        dynamic_range_db=config.dynamic_range_db,
        colormap=config.colormap,
        render_mode=config.render_mode,
        image_width=config.image_width,
        image_height=config.image_height,
    )
    waterfall_config = WaterfallConfig(
        window_samples=config.video_window_samples,
        hop_samples=config.video_hop_samples,
        fps=config.video_fps,
    )

    if config.detector_weights:
        detector = UltralyticsDetector(
            config.detector_weights,
            architecture=config.architecture,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            batch=config.batch,
        )
    else:
        detector = HeuristicSpectrogramDetector(class_name="signal")
    classifier = ImageClassifier(config.classifier_weights, device=config.device) if config.classifier_weights else NullClassifier()

    from rf_pipeline.preprocessing import BrowserVideoWriter

    raw_video_path = config.output_dir / "waterfall.mp4"
    detection_video_path = config.output_dir / "waterfall_detections.mp4"
    spectrogram_path = config.output_dir / "spectrogram.png"
    overlay_path = config.output_dir / "detections_overlay.png"
    segment_dir = config.output_dir / "segment_spectrograms"
    crop_dir = config.output_dir / "classification_crops"
    segment_dir.mkdir(parents=True, exist_ok=True)

    detections: list[dict[str, Any]] = []
    image_width = int(config.image_width or 0)
    image_height = int(config.image_height or 0)
    first_frame_saved = False
    inference_elapsed_sec = 0.0
    inference_frame_count = 0

    def flush_batch(records: list[dict[str, Any]], raw_writer, annotated_writer) -> None:
        nonlocal image_width, image_height, first_frame_saved, inference_elapsed_sec, inference_frame_count
        if not records:
            return
        paths = [record["path"] for record in records]
        batch_start = time.perf_counter()
        predictions = detector.predict(paths)
        inference_elapsed_sec += time.perf_counter() - batch_start
        inference_frame_count += len(records)
        for record in records:
            frame_image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
            if frame_image is None:
                continue
            raw_writer.write(frame_image)
            annotated = frame_image.copy()
            height, width = annotated.shape[:2]
            image_width = width
            image_height = height
            frame_start_sec = metadata.start_time_sec + record["start_sample"] / metadata.sample_rate_hz
            frame_end_sec = frame_start_sec + waterfall_config.window_samples / metadata.sample_rate_hz
            for detection in predictions.get(record["path"], []):
                crop_path = _save_detection_crop(
                    record["path"],
                    crop_dir / f"frame_{record['frame_index']:06d}_det_{len(detections):06d}.png",
                    detection.xyxy,
                )
                classification = classifier.classify_crop(
                    record["path"],
                    detection.xyxy,
                    detection.class_name,
                    detection.class_id,
                    detection.confidence,
                )
                estimate = estimate_parameters(
                    detection.xyxy,
                    image_width=width,
                    image_height=height,
                    segment_start_sec=frame_start_sec,
                    segment_end_sec=frame_end_sec,
                    freq_min_hz=record["frame_meta"].freq_min_hz,
                    freq_max_hz=record["frame_meta"].freq_max_hz,
                )
                item = {
                    "frame_index": record["frame_index"],
                    "frame_start_sec": frame_start_sec,
                    "frame_end_sec": frame_end_sec,
                    "detector": asdict(detection),
                    "classification": asdict(classification),
                    "classification_crop_path": str(crop_path) if crop_path else None,
                    "parameters": asdict(estimate),
                }
                detections.append(item)
                _draw_detection(annotated, item)
            annotated_writer.write(annotated)
            if not first_frame_saved:
                cv2.imwrite(str(spectrogram_path), frame_image)
                cv2.imwrite(str(overlay_path), annotated)
                first_frame_saved = True

    frame_index = 0
    batch_records: list[dict[str, Any]] = []
    with BrowserVideoWriter(raw_video_path, fps=waterfall_config.fps, codec=waterfall_config.codec) as raw_writer, BrowserVideoWriter(
        detection_video_path, fps=waterfall_config.fps, codec=waterfall_config.codec
    ) as annotated_writer:
        for start_sample, iq_window in iter_iq_windows(iq_path, metadata, waterfall_config.window_samples, waterfall_config.hop_samples):
            frame_meta = iq_to_spectrogram(iq_window, spectrogram_config)
            frame_path = segment_dir / f"segment_{frame_index:06d}.png"
            if not cv2.imwrite(str(frame_path), frame_meta.image):
                raise RuntimeError(f"Failed to write waterfall frame: {frame_path}")
            batch_records.append(
                {
                    "frame_index": frame_index,
                    "start_sample": start_sample,
                    "path": frame_path,
                    "frame_meta": frame_meta,
                }
            )
            frame_index += 1
            if len(batch_records) >= config.batch:
                flush_batch(batch_records, raw_writer, annotated_writer)
                batch_records = []
        flush_batch(batch_records, raw_writer, annotated_writer)

    if not first_frame_saved:
        raise ValueError("No waterfall frames were generated. Increase segment duration or reduce window size.")

    return PipelineResult(
        iq_path=str(iq_path),
        spectrogram_path=str(spectrogram_path),
        overlay_path=str(overlay_path),
        segment_spectrogram_dir=str(segment_dir),
        waterfall_video_path=str(raw_video_path),
        waterfall_detection_video_path=str(detection_video_path),
        image_width=image_width,
        image_height=image_height,
        performance={
            "mode": "segmented_spectrograms",
            "inference_frames": inference_frame_count,
            "inference_elapsed_sec": inference_elapsed_sec,
            "inference_fps": (inference_frame_count / inference_elapsed_sec) if inference_elapsed_sec > 0 else None,
            "num_detections": len(detections),
            "batch_size": config.batch,
        },
        detections=detections,
    )


def _save_overlay(image_path: Path, overlay_path: Path, detections: list[dict[str, Any]]) -> None:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    for item in detections:
        _draw_detection(image, item)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(overlay_path), image):
        raise RuntimeError(f"Failed to write overlay: {overlay_path}")


def _save_detection_crop(image_path: Path, crop_path: Path, xyxy: tuple[float, float, float, float]) -> Path | None:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = min(max(x1, 0), width)
    x2 = min(max(x2, 0), width)
    y1 = min(max(y1, 0), height)
    y2 = min(max(y2, 0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(crop_path), image[y1:y2, x1:x2]):
        raise RuntimeError(f"Failed to write classification crop: {crop_path}")
    return crop_path


def _draw_detection(image, item: dict[str, Any]) -> None:
    import cv2

    box = item["detector"]["xyxy"]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    label = item["classification"]["class_name"]
    confidence = item["classification"]["confidence"]
    color = _box_color(label, item["classification"]["class_id"])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        f"{label} {confidence:.2f}",
        (x1, max(12, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _save_waterfall_detection_video(
    iq,
    path: Path,
    spectrogram_config: SpectrogramConfig,
    waterfall_config: WaterfallConfig,
    detector,
    classifier,
) -> int:
    import cv2

    annotated_frames = []
    temp_dir = path.parent / "_waterfall_detection_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for index, frame in enumerate(waterfall_frames(iq, spectrogram_config, waterfall_config)):
        frame_path = temp_dir / f"frame_{index:05d}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"Failed to write waterfall frame: {frame_path}")
        frame_paths.append(frame_path)
    if not frame_paths:
        return 0

    predictions = detector.predict(frame_paths)
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        for detection in predictions.get(frame_path, []):
            classification = classifier.classify_crop(
                frame_path,
                detection.xyxy,
                detection.class_name,
                detection.class_id,
                detection.confidence,
            )
            _draw_detection(
                frame,
                {
                    "detector": asdict(detection),
                    "classification": asdict(classification),
                },
            )
        annotated_frames.append(frame)

    frame_count = save_frames_video(annotated_frames, path, fps=waterfall_config.fps, codec=waterfall_config.codec)
    for frame_path in frame_paths:
        frame_path.unlink(missing_ok=True)
    try:
        temp_dir.rmdir()
    except OSError:
        pass
    return frame_count


def _save_mapped_waterfall_detection_video(
    iq,
    path: Path,
    spectrogram_config: SpectrogramConfig,
    waterfall_config: WaterfallConfig,
    detections: list[dict[str, Any]],
    segment_start_sec: float,
    full_duration_sec: float,
) -> int:
    import cv2

    annotated_frames = []
    window_sec = waterfall_config.window_samples / spectrogram_config.sample_rate_hz
    hop_sec = waterfall_config.hop_samples / spectrogram_config.sample_rate_hz
    for frame_index, frame in enumerate(waterfall_frames(iq, spectrogram_config, waterfall_config)):
        height, width = frame.shape[:2]
        frame_start = frame_index * hop_sec
        frame_end = frame_start + window_sec
        for item in detections:
            params = item["parameters"]
            det_start = max(0.0, params["time_start_sec"] - segment_start_sec)
            det_end = min(full_duration_sec, params["time_end_sec"] - segment_start_sec)
            overlap_start = max(det_start, frame_start)
            overlap_end = min(det_end, frame_end)
            if overlap_end <= overlap_start:
                continue

            _, y1_static, _, y2_static = item["detector"]["xyxy"]
            x1 = int(round(((overlap_start - frame_start) / window_sec) * width))
            x2 = int(round(((overlap_end - frame_start) / window_sec) * width))
            y1 = int(round(y1_static))
            y2 = int(round(y2_static))
            x1 = min(max(x1, 0), width - 1)
            x2 = min(max(x2, 0), width - 1)
            y1 = min(max(y1, 0), height - 1)
            y2 = min(max(y2, 0), height - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            label = item["classification"]["class_name"]
            confidence = item["classification"]["confidence"]
            color = _box_color(label, item["classification"]["class_id"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        annotated_frames.append(frame)
    return save_frames_video(annotated_frames, path, fps=waterfall_config.fps, codec=waterfall_config.codec)


def _box_color(class_name: str, class_id: int) -> tuple[int, int, int]:
    name = str(class_name).lower()
    if "ofdm" in name:
        return (0, 255, 255)
    if "fhss" in name:
        return (255, 0, 255)
    palette = [
        (0, 255, 255),
        (255, 0, 255),
        (80, 220, 120),
        (255, 180, 40),
        (80, 160, 255),
        (220, 120, 255),
        (40, 220, 255),
        (180, 255, 80),
    ]
    return palette[int(class_id) % len(palette)]


def _guard_static_memory(iq_path: Path, metadata: IQMetadata, config: PipelineConfig) -> None:
    from rf_pipeline.data import complex_sample_bytes

    total_samples = Path(iq_path).stat().st_size // complex_sample_bytes(metadata.dtype)
    overlap = int(config.stft_point * 0.5)
    hop = max(1, config.stft_point - overlap)
    approx_frames = max(1, total_samples // hop)
    approx_bytes = approx_frames * config.stft_point * 16
    limit_bytes = 1_500_000_000
    if approx_bytes > limit_bytes:
        raise MemoryError(
            "Static spectrogram would allocate too much memory. "
            "Use Step 3 -> 'Segment spectrograms -> stitched waterfall' for large IQ files, "
            "or trim a shorter segment in Step 1."
        )
