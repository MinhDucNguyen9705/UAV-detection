from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stress.utils import load_ultralytics_model, read_yaml_split_images, write_resolved_ultralytics_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and stress-test an Ultralytics detector.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--benchmark-yaml-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--architecture", choices=("auto", "yolo", "rtdetr"), default="auto")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--project", default="runs/ultralytics_rfuav_stress")
    parser.add_argument("--name", default="yolo_rfuav_stress")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--false-alarm-conf", type=float, default=0.25)
    return parser.parse_args()


def box_metric(metrics, attr: str) -> float:
    value = getattr(metrics.box, attr, None)
    return float(value) if value is not None else float("nan")


def parse_benchmark_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    for scenario in ("low_snr", "near_far", "mix2", "noise_only"):
        prefix = f"{scenario}_"
        if stem == scenario:
            return scenario, "all"
        if stem.startswith(prefix):
            return scenario, stem[len(prefix) :]
    if stem == "stress_all":
        return "stress", "all"
    return stem, "all"


def yaml_sort_key(path: Path) -> tuple[int, str]:
    order = {"stress": 0, "low_snr": 1, "mix2": 2, "near_far": 3, "noise_only": 4}
    scenario, _ = parse_benchmark_name(path)
    return (order.get(scenario, 9), path.stem)


def count_false_alarms(model, yaml_path: Path, args: argparse.Namespace) -> dict:
    images = read_yaml_split_images(yaml_path, "test")
    total_detections = 0
    images_with_detections = 0
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.false_alarm_conf,
        batch=args.batch,
        device=args.device,
        verbose=False,
        stream=True,
    )
    for result in results:
        detections = 0 if result.boxes is None else len(result.boxes)
        total_detections += detections
        images_with_detections += int(detections > 0)
    total = len(images)
    return {
        "num_images": total,
        "total_detections": total_detections,
        "images_with_detections": images_with_detections,
        "image_false_alarm_rate": images_with_detections / total if total else 0.0,
        "false_alarms_per_image": total_detections / total if total else 0.0,
    }


def main() -> int:
    args = parse_args()
    if not args.train_data.exists():
        raise FileNotFoundError(f"Train data YAML not found: {args.train_data}")
    if not args.benchmark_yaml_dir.exists():
        raise FileNotFoundError(f"Benchmark YAML dir not found: {args.benchmark_yaml_dir}")

    resolved_dir = Path(args.project) / args.name / "_resolved_data"
    train_data = write_resolved_ultralytics_yaml(args.train_data.resolve(), resolved_dir, "train_data")
    checkpoint = args.weights

    if not args.skip_train and checkpoint is None:
        model = load_ultralytics_model(args.model, args.architecture)
        results = model.train(
            data=str(train_data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            project=args.project,
            name=args.name,
            seed=args.seed,
        )
        checkpoint = Path(getattr(results, "save_dir", Path(args.project) / args.name)) / "weights" / "best.pt"
    if checkpoint is None:
        checkpoint = Path(args.project) / args.name / "weights" / "best.pt"
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    rows = []
    for yaml_path in sorted(args.benchmark_yaml_dir.glob("*.yaml"), key=yaml_sort_key):
        scenario, level = parse_benchmark_name(yaml_path)
        resolved_yaml = write_resolved_ultralytics_yaml(yaml_path.resolve(), resolved_dir, f"bench_{yaml_path.stem}")
        model = load_ultralytics_model(checkpoint, args.architecture)
        row = {
            "scenario": scenario,
            "level": level,
            "metric_type": "detection",
            "yaml": str(yaml_path),
            "checkpoint": str(checkpoint),
            "map50_95": "",
            "map50": "",
            "map75": "",
            "mp": "",
            "mr": "",
            "num_images": "",
            "total_detections": "",
            "images_with_detections": "",
            "image_false_alarm_rate": "",
            "false_alarms_per_image": "",
            "false_alarm_conf": "",
        }
        if scenario == "noise_only":
            fa = count_false_alarms(model, resolved_yaml, args)
            row.update({"metric_type": "false_alarm", **fa, "false_alarm_conf": args.false_alarm_conf})
        else:
            metrics = model.val(
                data=str(resolved_yaml),
                split="test",
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                project=args.project,
                name=f"{args.name}_bench_{yaml_path.stem}",
            )
            row.update(
                {
                    "map50_95": box_metric(metrics, "map"),
                    "map50": box_metric(metrics, "map50"),
                    "map75": box_metric(metrics, "map75"),
                    "mp": box_metric(metrics, "mp"),
                    "mr": box_metric(metrics, "mr"),
                }
            )
        rows.append(row)
        print(f"{scenario}/{level}: {row}")

    output_csv = args.output_csv or Path(args.project) / args.name / "stress_metrics.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote stress metrics: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
