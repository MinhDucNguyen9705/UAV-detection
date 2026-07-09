from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from rf_pipeline.inference import estimate_parameters
from stress.utils import find_images, image_size, load_ultralytics_model, read_csv


@dataclass(slots=True)
class ImageEstimate:
    sample_id: str
    image_path: str
    image_width: int
    image_height: int
    sample_rate_hz: float
    center_frequency_hz: float
    freq_min_hz: float
    freq_max_hz: float
    segment_start_sec: float
    segment_end_sec: float
    detections: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate RF parameters from detector boxes on spectrograms.")
    parser.add_argument("--source", type=Path, required=True, help="Image, image folder, or dataset root.")
    parser.add_argument("--model", default=None, help="YOLO/RT-DETR checkpoint. Mutually exclusive with --labels-dir.")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Existing YOLO labels to convert.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional samples_manifest.csv.")
    parser.add_argument("--architecture", choices=("auto", "yolo", "rtdetr"), default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--class-names", nargs="*", default=["uav_signal"])
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--save-overlay-dir", type=Path, default=None)
    parser.add_argument("--default-sample-rate", type=float, default=100e6)
    parser.add_argument("--default-center-frequency", type=float, default=2.4e9)
    parser.add_argument("--default-duration", type=float, default=0.03)
    parser.add_argument("--min-conf", type=float, default=None)
    args = parser.parse_args()
    if (args.model is None) == (args.labels_dir is None):
        raise ValueError("Provide exactly one of --model or --labels-dir.")
    return args


def as_float(row: dict[str, str], key: str, default: float) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def resolve_manifest(source: Path, manifest: Path | None) -> Path | None:
    if manifest:
        return manifest.resolve()
    candidate = source / "manifests" / "samples_manifest.csv"
    return candidate.resolve() if candidate.is_file() else None


def manifest_rows_by_image(manifest: Path | None, source: Path) -> dict[Path, dict[str, str]]:
    if manifest is None:
        return {}
    rows = {}
    for row in read_csv(manifest):
        value = row.get("image_path", "")
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = source / path
        rows[path.resolve()] = row
    return rows


def resolve_images(args: argparse.Namespace) -> tuple[list[Path], dict[Path, dict[str, str]]]:
    source = args.source.resolve()
    rows = manifest_rows_by_image(resolve_manifest(source, args.manifest), source if source.is_dir() else source.parent)
    if rows:
        images = [path for path, row in rows.items() if path.is_file() and (args.split == "all" or row.get("split") == args.split)]
        return sorted(images), rows
    return find_images(source, recursive=args.recursive), {}


def class_name_for(class_id: int, class_names: list[str]) -> str:
    return class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)


def yolo_to_xyxy(values: list[float], width: int, height: int) -> list[float]:
    xc, yc, bw, bh = values
    return [
        (xc - bw / 2.0) * width,
        (yc - bh / 2.0) * height,
        (xc + bw / 2.0) * width,
        (yc + bh / 2.0) * height,
    ]


def xyxy_to_xywhn(xyxy: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = xyxy
    return [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height, (x2 - x1) / width, (y2 - y1) / height]


def mapping_for(row: dict[str, str], width: int, height: int, args: argparse.Namespace) -> dict[str, float]:
    sample_rate = as_float(row, "sample_rate_hz", args.default_sample_rate)
    center = as_float(row, "center_frequency_hz", args.default_center_frequency)
    start = as_float(row, "segment_start_sec", 0.0)
    end = as_float(row, "segment_end_sec", start + as_float(row, "duration_sec", args.default_duration))
    return {
        "sample_rate_hz": sample_rate,
        "center_frequency_hz": center,
        "freq_min_hz": as_float(row, "freq_min_hz", center - sample_rate / 2),
        "freq_max_hz": as_float(row, "freq_max_hz", center + sample_rate / 2),
        "segment_start_sec": start,
        "segment_end_sec": end,
        "image_width": as_float(row, "image_width", float(width)),
        "image_height": as_float(row, "image_height", float(height)),
    }


def labels_for_image(image: Path, image_root: Path, labels_dir: Path, width: int, height: int, min_conf: float | None):
    try:
        rel = image.relative_to(image_root)
    except ValueError:
        rel = Path(image.name)
    label_path = (labels_dir / rel).with_suffix(".txt")
    if not label_path.is_file():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        confidence = float(parts[5]) if len(parts) > 5 else None
        if min_conf is not None and confidence is not None and confidence < min_conf:
            continue
        rows.append((class_id, yolo_to_xyxy([float(v) for v in parts[1:5]], width, height), confidence))
    return rows


def make_result(image: Path, raw_detections, row: dict[str, str], args: argparse.Namespace) -> ImageEstimate:
    width, height = image_size(image)
    mapping = mapping_for(row, width, height, args)
    detections = []
    for class_id, xyxy, confidence in raw_detections:
        estimate = estimate_parameters(
            tuple(float(v) for v in xyxy),
            image_width=width,
            image_height=height,
            segment_start_sec=mapping["segment_start_sec"],
            segment_end_sec=mapping["segment_end_sec"],
            freq_min_hz=mapping["freq_min_hz"],
            freq_max_hz=mapping["freq_max_hz"],
        )
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name_for(class_id, args.class_names),
                "confidence": confidence,
                "bbox_xyxy_px": [float(v) for v in xyxy],
                "bbox_xywhn": xyxy_to_xywhn([float(v) for v in xyxy], width, height),
                **asdict(estimate),
            }
        )
    return ImageEstimate(
        sample_id=row.get("sample_id", image.stem),
        image_path=str(image),
        image_width=width,
        image_height=height,
        sample_rate_hz=mapping["sample_rate_hz"],
        center_frequency_hz=mapping["center_frequency_hz"],
        freq_min_hz=mapping["freq_min_hz"],
        freq_max_hz=mapping["freq_max_hz"],
        segment_start_sec=mapping["segment_start_sec"],
        segment_end_sec=mapping["segment_end_sec"],
        detections=detections,
    )


def save_overlay(image: Path, result: ImageEstimate, output_dir: Path, image_root: Path) -> None:
    frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if frame is None:
        return
    try:
        rel = image.relative_to(image_root)
    except ValueError:
        rel = Path(image.name)
    target = output_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    for det in result.detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy_px"]]
        label = f"{det['class_name']} {det['confidence']:.2f}" if det["confidence"] is not None else det["class_name"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(target), frame)


def main() -> int:
    args = parse_args()
    images, manifest_rows = resolve_images(args)
    if not images:
        raise FileNotFoundError(f"No images found from source: {args.source}")
    source = args.source.resolve()
    image_root = (source / "spectrograms" / "images").resolve() if source.is_dir() and (source / "spectrograms" / "images").is_dir() else (source if source.is_dir() else source.parent)

    results: list[ImageEstimate] = []
    if args.model:
        model = load_ultralytics_model(args.model, args.architecture)
        predictions = model.predict(
            source=[str(path) for path in images],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            batch=args.batch,
            device=args.device,
            classes=args.classes,
            stream=True,
            verbose=False,
        )
        for image, pred in zip(images, predictions):
            raw = []
            if pred.boxes is not None:
                for class_id, xyxy, confidence in zip(pred.boxes.cls.tolist(), pred.boxes.xyxy.tolist(), pred.boxes.conf.tolist()):
                    if args.min_conf is None or float(confidence) >= args.min_conf:
                        raw.append((int(class_id), [float(v) for v in xyxy], float(confidence)))
            result = make_result(image, raw, manifest_rows.get(image, {}), args)
            results.append(result)
            if args.save_overlay_dir:
                save_overlay(image, result, args.save_overlay_dir, image_root)
    else:
        labels_dir = args.labels_dir.resolve()
        for image in images:
            width, height = image_size(image)
            raw = labels_for_image(image, image_root, labels_dir, width, height, args.min_conf)
            result = make_result(image, raw, manifest_rows.get(image, {}), args)
            results.append(result)
            if args.save_overlay_dir:
                save_overlay(image, result, args.save_overlay_dir, image_root)

    payload = {
        "source": str(args.source.resolve()),
        "model": args.model,
        "labels_dir": str(args.labels_dir.resolve()) if args.labels_dir else None,
        "num_images": len(results),
        "num_detections": sum(len(item.detections) for item in results),
        "results": [asdict(item) for item in results],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Images: {payload['num_images']}")
    print(f"Detections: {payload['num_detections']}")
    print(f"Output JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
