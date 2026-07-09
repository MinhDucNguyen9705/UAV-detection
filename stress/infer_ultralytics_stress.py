from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stress.utils import (
    find_images,
    load_ultralytics_model,
    normalize_class_names,
    read_yaml_split_images,
    relative_to_dataset,
    scenario_for_image,
)


@dataclass(slots=True)
class PredictionRow:
    image_id: int
    image_path: str
    relative_image_path: str
    sample_id: str
    scenario: str
    class_id: int
    class_name: str
    confidence: float
    x_center_norm: float
    y_center_norm: float
    width_norm: float
    height_norm: float
    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float
    width_px: float
    height_px: float
    image_width: int
    image_height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer stress benchmark images and export metric-ready predictions.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--benchmark-yaml", type=Path, default=None)
    parser.add_argument("--list-file", type=Path, default=None)
    parser.add_argument("--scenarios", nargs="*", default=None, choices=["low_snr", "mix2", "near_far", "noise_only", "clean_single"])
    parser.add_argument("--architecture", choices=("auto", "yolo", "rtdetr"), default="auto")
    parser.add_argument("--mode", choices=("eval", "deploy", "both"), default="eval")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--eval-conf", type=float, default=0.001)
    parser.add_argument("--deploy-conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--skip-empty-labels", action="store_true")
    parser.add_argument("--class-names", nargs="*", default=None)
    return parser.parse_args()


def image_list_from_txt(list_file: Path, dataset_dir: Path) -> list[Path]:
    images = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        image = Path(text)
        if not image.is_absolute():
            image = dataset_dir / image
        if image.is_file():
            images.append(image.resolve())
    return images


def resolve_images(args: argparse.Namespace) -> list[Path]:
    dataset_dir = args.dataset_dir.resolve()
    if args.list_file:
        list_file = args.list_file if args.list_file.is_absolute() else dataset_dir / args.list_file
        images = image_list_from_txt(list_file.resolve(), dataset_dir)
    elif args.benchmark_yaml:
        yaml_path = args.benchmark_yaml if args.benchmark_yaml.is_absolute() else dataset_dir / args.benchmark_yaml
        images = read_yaml_split_images(yaml_path.resolve(), "test")
    else:
        default_yaml = dataset_dir / "benchmark_yamls" / "stress_all.yaml"
        images = read_yaml_split_images(default_yaml, "test") if default_yaml.is_file() else find_images(dataset_dir / "stress_yolo" / "images")

    unique = []
    seen = set()
    for image in images:
        if image not in seen:
            unique.append(image)
            seen.add(image)
    if args.scenarios:
        wanted = set(args.scenarios)
        unique = [image for image in unique if scenario_for_image(image) in wanted]
    return unique


def iter_chunks(items: list[Path], chunk_size: int):
    for start in range(0, len(items), max(1, chunk_size)):
        yield items[start : start + max(1, chunk_size)]


def free_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def confidence_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def resolve_mode_conf(args: argparse.Namespace, mode: str) -> float:
    return args.conf if args.conf is not None else (args.eval_conf if mode == "eval" else args.deploy_conf)


def mode_output_dir(args: argparse.Namespace, mode: str, conf: float) -> Path:
    if args.output_dir is None:
        return (args.dataset_dir / "predictions" / Path(args.weights).stem / f"{mode}_conf{confidence_tag(conf)}").resolve()
    if args.mode == "both":
        return (args.output_dir / f"{mode}_conf{confidence_tag(conf)}").resolve()
    return args.output_dir.resolve()


def prediction_label_path(output_dir: Path, relative_image: Path) -> Path:
    parts = list(relative_image.parts)
    if parts and parts[0].lower() == "stress_yolo":
        parts = parts[1:]
    if parts and parts[0].lower() == "images":
        parts[0] = "labels"
    else:
        parts = ["labels", *parts]
    return (output_dir / Path(*parts)).with_suffix(".txt")


def write_csv(path: Path, rows: list[PredictionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(PredictionRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_inference(args: argparse.Namespace, model: Any, class_names: dict[int, str], images: list[Path], mode: str) -> Path:
    import cv2

    conf = resolve_mode_conf(args, mode)
    output_dir = mode_output_dir(args, mode, conf)
    labels_root = output_dir / "labels"
    overlays_root = output_dir / "overlays"
    labels_root.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_root.mkdir(parents=True, exist_ok=True)

    rows: list[PredictionRow] = []
    coco_images = []
    coco_predictions = []
    detections_saved = 0

    image_ids = {image: idx for idx, image in enumerate(images, start=1)}
    for chunk in iter_chunks(images, args.chunk_size):
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=args.imgsz,
            conf=conf,
            iou=args.iou,
            batch=args.batch,
            device=args.device,
            classes=args.classes,
            stream=True,
            verbose=False,
            save=False,
            half=args.half,
            max_det=args.max_det,
        )
        for image, result in zip(chunk, results):
            image_id = image_ids[image]
            rel = relative_to_dataset(image, args.dataset_dir.resolve())
            scenario = scenario_for_image(image)
            sample_id = image.stem
            height, width = result.orig_shape[:2]
            coco_images.append({"id": image_id, "file_name": rel.as_posix(), "width": int(width), "height": int(height), "scenario": scenario, "sample_id": sample_id})
            label_rows = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for class_id_raw, conf_raw, xywhn, xyxy in zip(
                    boxes.cls.detach().cpu().tolist(),
                    boxes.conf.detach().cpu().tolist(),
                    boxes.xywhn.detach().cpu().tolist(),
                    boxes.xyxy.detach().cpu().tolist(),
                ):
                    class_id = int(class_id_raw)
                    confidence = float(conf_raw)
                    x_center, y_center, box_w, box_h = [float(v) for v in xywhn]
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    width_px = max(0.0, x2 - x1)
                    height_px = max(0.0, y2 - y1)
                    label_rows.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f} {confidence:.6f}")
                    rows.append(
                        PredictionRow(
                            image_id=image_id,
                            image_path=str(image),
                            relative_image_path=rel.as_posix(),
                            sample_id=sample_id,
                            scenario=scenario,
                            class_id=class_id,
                            class_name=class_names.get(class_id, str(class_id)),
                            confidence=confidence,
                            x_center_norm=x_center,
                            y_center_norm=y_center,
                            width_norm=box_w,
                            height_norm=box_h,
                            x1_px=x1,
                            y1_px=y1,
                            x2_px=x2,
                            y2_px=y2,
                            width_px=width_px,
                            height_px=height_px,
                            image_width=int(width),
                            image_height=int(height),
                        )
                    )
                    coco_predictions.append({"image_id": image_id, "category_id": class_id + 1, "bbox": [x1, y1, width_px, height_px], "score": confidence})

            label_path = prediction_label_path(output_dir, rel)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if label_rows or not args.skip_empty_labels:
                label_path.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
            detections_saved += len(label_rows)
            if args.save_overlays:
                overlay_path = overlays_root / rel
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(overlay_path), result.plot())
        del results
        free_accelerator_cache()

    write_csv(output_dir / "predictions.csv", rows)
    categories = [{"id": class_id + 1, "name": name, "supercategory": ""} for class_id, name in sorted(class_names.items())]
    (output_dir / "predictions_coco.json").write_text(json.dumps({"images": coco_images, "annotations": coco_predictions, "categories": categories}, indent=2), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(args.dataset_dir.resolve()),
                "weights": args.weights,
                "architecture": args.architecture,
                "inference_mode": mode,
                "imgsz": args.imgsz,
                "conf": conf,
                "iou": args.iou,
                "num_images": len(images),
                "num_detections": detections_saved,
                "class_names": class_names,
                "labels_dir": str(labels_root),
                "predictions_csv": str(output_dir / "predictions.csv"),
                "predictions_coco": str(output_dir / "predictions_coco.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{mode}] Images: {len(images)} Detections: {detections_saved} Output: {output_dir}")
    return output_dir


def main() -> int:
    args = parse_args()
    images = resolve_images(args)
    if not images:
        raise FileNotFoundError("No stress benchmark images matched the requested source/filter.")
    model = load_ultralytics_model(args.weights, args.architecture)
    class_names = normalize_class_names(getattr(model, "names", None), args.class_names)
    modes = ["eval", "deploy"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_inference(args, model, class_names, images, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
