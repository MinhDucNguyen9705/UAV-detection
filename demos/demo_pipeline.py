#!/usr/bin/env python3
"""Demo rawIQ -> spectrogram/waterfall -> detection -> classification -> RF estimates.

Examples:
  python demos/demo_pipeline.py --iq sample.iq --detector best.pt --out runs/demo
  python demos/demo_pipeline.py --make-synthetic --out runs/demo_synthetic --save-video
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rf_pipeline.data import IQMetadata, write_iq
from rf_pipeline.inference import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iq", type=Path, default=None, help="Raw IQ file. Interleaved float32 by default.")
    parser.add_argument("--make-synthetic", action="store_true", help="Generate a simple synthetic IQ signal first.")
    parser.add_argument("--out", type=Path, default=Path("runs/demo_pipeline"))
    parser.add_argument("--dtype", choices=["float32", "int16", "complex64"], default="float32")
    parser.add_argument("--sample-rate", type=float, default=100e6)
    parser.add_argument("--center-frequency", type=float, default=2.4e9)
    parser.add_argument("--detector", type=Path, default=None, help="YOLO/RT-DETR checkpoint. Omit for heuristic demo.")
    parser.add_argument("--classifier", type=Path, default=None, help="Optional crop classifier checkpoint.")
    parser.add_argument("--architecture", choices=["auto", "yolo", "rtdetr"], default="auto")
    parser.add_argument("--stft-point", type=int, default=16384)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def make_synthetic_iq(path: Path, sample_rate: float, seconds: float = 0.02) -> Path:
    """Create a two-tone burst IQ sample for smoke-testing the pipeline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = int(sample_rate * seconds)
    time = np.arange(n, dtype=np.float32) / sample_rate
    iq = 0.025 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    burst = (time > seconds * 0.25) & (time < seconds * 0.75)
    iq[burst] += 0.8 * np.exp(1j * 2.0 * np.pi * 8e6 * time[burst])
    iq[burst] += 0.35 * np.exp(1j * 2.0 * np.pi * -15e6 * time[burst])
    write_iq(path, iq.astype(np.complex64), dtype="float32")
    return path


def main() -> None:
    args = parse_args()
    iq_path = args.iq
    if args.make_synthetic:
        iq_path = make_synthetic_iq(args.out / "synthetic.iq", args.sample_rate)
    if iq_path is None:
        raise ValueError("Provide --iq or use --make-synthetic.")
    if not iq_path.is_file():
        raise FileNotFoundError(f"IQ file not found: {iq_path}")

    metadata = IQMetadata(
        sample_rate_hz=args.sample_rate,
        center_frequency_hz=args.center_frequency,
        dtype=args.dtype,
    )
    config = PipelineConfig(
        output_dir=args.out,
        detector_weights=args.detector,
        classifier_weights=args.classifier,
        architecture=args.architecture,
        stft_point=args.stft_point,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_video=args.save_video,
    )
    result = run_pipeline(iq_path, metadata, config)
    output_json = args.out / "pipeline_result.json"
    output_json.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    print(f"Spectrogram: {result.spectrogram_path}")
    print(f"Overlay: {result.overlay_path}")
    if result.waterfall_video_path:
        print(f"Waterfall: {result.waterfall_video_path}")
    if result.waterfall_detection_video_path:
        print(f"Waterfall detections: {result.waterfall_detection_video_path}")
    print(f"Detections: {len(result.detections)}")
    fps = result.performance.get("inference_fps")
    elapsed = result.performance.get("inference_elapsed_sec")
    frames = result.performance.get("inference_frames")
    if fps is not None:
        print(f"Inference FPS: {fps:.2f} ({frames} frame(s) in {elapsed:.3f}s)")
    print(f"Result JSON: {output_json}")


if __name__ == "__main__":
    main()
