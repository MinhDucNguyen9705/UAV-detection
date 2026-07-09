from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from label.model_pseudo_label import ModelPseudoLabelConfig, run_model_pseudo_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a YOLO or RT-DETR model on spectrogram images and save YOLO-format pseudo labels."
    )
    parser.add_argument("--model", required=True, help="Model checkpoint, e.g. best.pt.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Image folder, or OUT_DIR created by dataset/hf_rfuav_spectrogram_manifest.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root. For RFUAV output, labels default to OUT_DIR/spectrograms/labels.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="RFUAV split to infer, using OUT_DIR/splits/<split>.txt.",
    )
    parser.add_argument("--architecture", choices=("auto", "yolo", "rtdetr"), default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Examples: 0, 0,1, cpu.")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="Optional class IDs to keep.")
    parser.add_argument("--with-confidence", action="store_true", help="Append confidence to each label row.")
    parser.add_argument("--save-images", action="store_true", help="Also save images with predicted boxes.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not scan source subfolders.")
    parser.add_argument("--skip-empty", action="store_true", help="Do not create empty label files.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of image paths per model.predict chunk.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on CUDA.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ModelPseudoLabelConfig:
    return ModelPseudoLabelConfig(
        model=args.model,
        source=args.source,
        output=args.output,
        split=args.split,
        architecture=args.architecture,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
        classes=args.classes,
        with_confidence=args.with_confidence,
        save_images=args.save_images,
        recursive=not args.no_recursive,
        skip_empty=args.skip_empty,
        chunk_size=args.chunk_size,
        half=args.half,
        max_det=args.max_det,
    )


def main() -> int:
    result = run_model_pseudo_label(config_from_args(parse_args()))
    print("Inference done")
    print(f"Processed images: {result.processed_images}")
    print(f"Saved detections: {result.saved_detections}")
    print(f"Labels directory: {result.labels_dir}")
    if result.images_dir.exists():
        print(f"Visualization directory: {result.images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

