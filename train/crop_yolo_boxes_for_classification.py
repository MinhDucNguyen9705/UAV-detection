from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rf_pipeline.model import (  # noqa: E402
    CLASSIFICATION_MODEL_CHOICES,
    TORCHVISION_CLASSIFIERS,
    ULTRALYTICS_CLASSIFIERS,
    build_torchvision_classifier,
    train_torchvision_classifier,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_FILENAME_REGEX = r"^(?P<class>.+?)_\d{5}_pack"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read YOLO txt annotations, crop each bounding box, and write crops in "
            "torchvision ImageFolder layout: out/{split}/{class_name}/*.jpg."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="YOLO dataset root.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output ImageFolder root.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help=(
            "Split names to export. With --data-yaml, reads split entries from that YAML. "
            "Otherwise looks for <split>.txt first, then images/<split>."
        ),
    )
    parser.add_argument(
        "--class-source",
        choices=("label", "filename"),
        default="label",
        help="Use YOLO class id from labels, or extract the class from image filename.",
    )
    parser.add_argument(
        "--filename-regex",
        default=DEFAULT_FILENAME_REGEX,
        help=(
            "Regex used when --class-source filename. Must contain a named group "
            "'class', or the first capture group is used."
        ),
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help=(
            "Optional Ultralytics data.yaml path. For the RFUAV stress dataset use "
            "<dataset-dir>/source_grouped_splits/data_with_stress.yaml."
        ),
    )
    parser.add_argument(
        "--train-classifier",
        action="store_true",
        help="After cropping, train a classification model on --out-dir.",
    )
    parser.add_argument(
        "--classifier-model",
        choices=CLASSIFICATION_MODEL_CHOICES,
        default="yolo11n_cls",
        help="Classifier architecture. Torchvision models train with the local trainer; YOLO cls models use Ultralytics.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Backward-compatible alias/custom Ultralytics .pt path. If set, it overrides --classifier-model "
            "for Ultralytics training."
        ),
    )
    parser.add_argument(
        "--classifier-backend",
        choices=("auto", "ultralytics", "torchvision"),
        default="auto",
        help="Training backend. Auto uses Ultralytics for yolo*_cls and torchvision otherwise.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use torchvision pretrained ImageNet weights. Requires weights to be available/downloadable.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for torchvision classifiers.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/rfuav_signal_classification")
    parser.add_argument("--name", default="fhss_ofdm_cls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", type=float, default=0.0, help="Relative padding around each bbox, e.g. 0.05.")
    parser.add_argument("--min-size", type=int, default=2, help="Skip crops smaller than this width or height.")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality for saved crops.")
    parser.add_argument("--overwrite", action="store_true", help="Delete out-dir before writing.")
    parser.add_argument(
        "--skip-split-crop",
        action="store_true",
        help="Skip normal train/val/test crop export. Useful when only evaluating benchmark crops.",
    )
    parser.add_argument(
        "--eval-benchmark-crops",
        action="store_true",
        help="Crop and evaluate classification metrics for every YAML under --benchmark-yaml-dir.",
    )
    parser.add_argument(
        "--benchmark-yaml-dir",
        type=Path,
        default=None,
        help="Directory containing per-benchmark YAML files, e.g. <dataset-dir>/benchmark_yamls.",
    )
    parser.add_argument(
        "--classifier-weights",
        type=Path,
        default=None,
        help="Classifier checkpoint for benchmark evaluation. Defaults to the just-trained best.pt when available.",
    )
    parser.add_argument(
        "--eval-crops-dir",
        type=Path,
        default=None,
        help="Temporary/output directory for per-benchmark classification crops.",
    )
    parser.add_argument(
        "--eval-output-csv",
        type=Path,
        default=None,
        help="CSV path for per-benchmark classification metrics.",
    )
    parser.add_argument(
        "--keep-eval-crops",
        action="store_true",
        help="Keep per-benchmark crop folders after evaluation.",
    )
    return parser.parse_args()


def load_names(data_yaml: Path) -> list[str]:
    if not data_yaml.exists():
        return []

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        indexed_names = {int(index): str(name) for index, name in names.items()}
        return [indexed_names[index] for index in sorted(indexed_names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def load_yaml_data(data_yaml: Path) -> dict:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML file: {data_yaml}")
    return data


def resolve_dataset_root(data_yaml: Path, data: dict, fallback: Path) -> Path:
    raw_root = Path(data.get("path", fallback))
    if raw_root.is_absolute():
        return raw_root.resolve()
    return (data_yaml.parent / raw_root).resolve()


def read_split_entries_from_yaml(data_yaml: Path, split: str, fallback_root: Path) -> list[Path]:
    if not data_yaml.exists():
        return []
    data = load_yaml_data(data_yaml)
    if split not in data:
        return []

    dataset_root = resolve_dataset_root(data_yaml, data, fallback_root)
    values = data[split] if isinstance(data[split], list) else [data[split]]
    images: list[Path] = []
    for value in values:
        path = Path(str(value))
        if not path.is_absolute():
            path = dataset_root / path

        if path.is_file() and path.suffix.lower() == ".txt":
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip().replace("\\", "/")
                if not line:
                    continue
                image_path = Path(line)
                if not image_path.is_absolute():
                    image_path = dataset_root / image_path
                images.append(image_path)
        elif path.is_dir():
            images.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        else:
            images.append(path)
    return images


def read_split_file(dataset_dir: Path, split: str) -> list[Path]:
    split_file = dataset_dir / f"{split}.txt"
    if not split_file.exists():
        return []

    images: list[Path] = []
    for raw in split_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip().replace("\\", "/")
        if not line:
            continue
        path = Path(line)
        if not path.is_absolute():
            path = dataset_dir / path
        if not path.exists() and line.startswith("data/"):
            path = dataset_dir / line.removeprefix("data/")
        images.append(path)
    return images


def discover_split_images(dataset_dir: Path, split: str, data_yaml: Path | None) -> list[Path]:
    if data_yaml is not None and data_yaml.exists():
        from_yaml = read_split_entries_from_yaml(data_yaml, split, dataset_dir)
        if from_yaml:
            return from_yaml

    from_txt = read_split_file(dataset_dir, split)
    if from_txt:
        return from_txt

    split_dir = dataset_dir / "images" / split
    if not split_dir.exists():
        return []
    return sorted(path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for_image(dataset_dir: Path, image_path: Path) -> Path:
    try:
        rel = image_path.resolve().relative_to(dataset_dir.resolve())
    except ValueError:
        rel = image_path

    parts = list(rel.parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"
        return dataset_dir / Path(*parts).with_suffix(".txt")
    if len(parts) >= 3 and parts[0] in {"source_yolo", "stress_yolo"} and parts[1] == "images":
        parts[1] = "labels"
        return dataset_dir / Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def class_from_label(class_id: int, names: list[str]) -> str:
    if 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)


def class_from_filename(image_path: Path, pattern: re.Pattern[str]) -> str:
    match = pattern.search(image_path.stem)
    if not match:
        raise ValueError(f"Cannot extract class from filename: {image_path.name}")
    if "class" in match.groupdict():
        return match.group("class")
    return match.group(1)


def sanitize_class_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name.strip())
    return cleaned.rstrip(". ") or "unknown"


def yolo_to_xyxy(
    width: int,
    height: int,
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    padding: float,
) -> tuple[int, int, int, int]:
    bw = box_width * width
    bh = box_height * height
    cx = x_center * width
    cy = y_center * height
    pad_x = bw * padding
    pad_y = bh * padding

    x1 = max(0, int(round(cx - bw / 2 - pad_x)))
    y1 = max(0, int(round(cy - bh / 2 - pad_y)))
    x2 = min(width, int(round(cx + bw / 2 + pad_x)))
    y2 = min(height, int(round(cy + bh / 2 + pad_y)))
    return x1, y1, x2, y2


def export_split(
    dataset_dir: Path,
    out_dir: Path,
    split: str,
    data_yaml: Path | None,
    names: list[str],
    class_source: str,
    filename_pattern: re.Pattern[str],
    padding: float,
    min_size: int,
    jpg_quality: int,
) -> Counter:
    counts: Counter = Counter()
    images = discover_split_images(dataset_dir, split, data_yaml)

    for image_path in images:
        if not image_path.exists():
            counts["missing_images"] += 1
            continue

        label_path = label_path_for_image(dataset_dir, image_path)
        if not label_path.exists():
            counts["missing_labels"] += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            counts["unreadable_images"] += 1
            continue
        height, width = image.shape[:2]

        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for box_index, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                counts["bad_labels"] += 1
                continue

            class_id = int(float(parts[0]))
            x_center, y_center, box_width, box_height = map(float, parts[1:5])
            class_name = (
                class_from_label(class_id, names)
                if class_source == "label"
                else class_from_filename(image_path, filename_pattern)
            )
            class_name = sanitize_class_name(class_name)

            x1, y1, x2, y2 = yolo_to_xyxy(width, height, x_center, y_center, box_width, box_height, padding)
            if x2 - x1 < min_size or y2 - y1 < min_size:
                counts["small_crops"] += 1
                continue

            crop = image[y1:y2, x1:x2]
            crop_dir = out_dir / split / class_name
            crop_dir.mkdir(parents=True, exist_ok=True)
            try:
                rel_for_hash = image_path.resolve().relative_to(dataset_dir.resolve()).as_posix()
            except ValueError:
                rel_for_hash = image_path.resolve().as_posix()
            digest = hashlib.sha1(rel_for_hash.encode("utf-8")).hexdigest()[:10]
            crop_name = f"{image_path.stem}_{digest}_box{box_index:03d}.jpg"
            crop_path = crop_dir / crop_name
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            counts[(split, class_name)] += 1
            counts["crops"] += 1

    counts["images"] += len(images)
    return counts


def resolve_classifier_backend(args: argparse.Namespace) -> str:
    if args.classifier_backend != "auto":
        return args.classifier_backend
    if args.model is not None:
        return "ultralytics"
    if args.classifier_model in ULTRALYTICS_CLASSIFIERS:
        return "ultralytics"
    return "torchvision"


def resolve_ultralytics_model(args: argparse.Namespace) -> str:
    if args.model is not None:
        return args.model
    if args.classifier_model not in ULTRALYTICS_CLASSIFIERS:
        raise ValueError(f"{args.classifier_model} is not an Ultralytics classifier")
    return ULTRALYTICS_CLASSIFIERS[args.classifier_model]


def train_ultralytics_classifier(args: argparse.Namespace, out_dir: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: install with `pip install ultralytics`.") from exc

    model = YOLO(resolve_ultralytics_model(args))
    results = model.train(
        data=str(out_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
    )
    print(f"Classification training finished. Save dir: {getattr(results, 'save_dir', '')}")
    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    return save_dir / "weights" / "best.pt"


def train_classifier(args: argparse.Namespace, out_dir: Path) -> Path:
    backend = resolve_classifier_backend(args)
    print(f"Classifier model: {args.model or args.classifier_model}")
    print(f"Classifier backend: {backend}")
    if backend == "ultralytics":
        return train_ultralytics_classifier(args, out_dir)

    if args.classifier_model not in TORCHVISION_CLASSIFIERS:
        raise ValueError(f"{args.classifier_model} is not a torchvision classifier")
    result = train_torchvision_classifier(
        model_name=args.classifier_model,
        data_dir=out_dir,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        run_name=args.name,
        seed=args.seed,
        pretrained=args.pretrained,
        lr=args.lr,
    )
    print(f"Classification training finished. Save dir: {result.save_dir}")
    print(f"Best checkpoint: {result.best_checkpoint}")
    return result.best_checkpoint


def benchmark_sort_key(path: Path) -> tuple[int, str]:
    if path.stem == "stress_all":
        return (0, path.stem)
    order = {"low_snr": 1, "mix2": 2, "near_far": 3, "noise_only": 4}
    scenario, _level = parse_benchmark_name(path)
    return (order.get(scenario, 9), path.stem)


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


def list_imagefolder_samples(root: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    if not root.exists():
        return samples
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            samples.append((image_path, class_dir.name))
    return samples


def classification_metrics(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    row: dict[str, float | int] = {}
    total = len(y_true)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    row["num_crops"] = total
    row["accuracy"] = correct / total if total else 0.0

    macro_f1 = 0.0
    weighted_f1 = 0.0
    for class_name in classes:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == class_name and pred == class_name)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != class_name and pred == class_name)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == class_name and pred != class_name)
        support = sum(1 for true in y_true if true == class_name)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row[f"support_{class_name}"] = support
        row[f"precision_{class_name}"] = precision
        row[f"recall_{class_name}"] = recall
        row[f"f1_{class_name}"] = f1
        macro_f1 += f1
        weighted_f1 += f1 * support
    row["macro_f1"] = macro_f1 / len(classes) if classes else 0.0
    row["weighted_f1"] = weighted_f1 / total if total else 0.0
    return row


def predict_torchvision_classifier(
    weights: Path,
    samples: list[tuple[Path, str]],
    imgsz: int,
    batch: int,
    device: str | None,
    workers: int,
) -> tuple[list[str], list[str], list[str]]:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    class SampleDataset(Dataset):
        def __init__(self, items: list[tuple[Path, str]]):
            self.items = items
            self.transform = transforms.Compose(
                [
                    transforms.Resize((imgsz, imgsz)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int):
            image_path, label = self.items[index]
            image = Image.open(image_path).convert("RGB")
            return self.transform(image), label

    checkpoint = torch.load(weights, map_location="cpu")
    classes = [str(name) for name in checkpoint["classes"]]
    model_name = checkpoint["model_name"]
    model = build_torchvision_classifier(model_name, len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cpu":
        device_obj = torch.device("cpu")
    elif str(device).isdigit():
        device_obj = torch.device(f"cuda:{device}")
    else:
        device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    loader = DataLoader(SampleDataset(samples), batch_size=batch, shuffle=False, num_workers=workers)
    y_true: list[str] = []
    y_pred: list[str] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device_obj)
            logits = model(images)
            pred_ids = logits.argmax(dim=1).cpu().tolist()
            y_true.extend([str(label) for label in labels])
            y_pred.extend(classes[index] for index in pred_ids)
    return y_true, y_pred, classes


def predict_ultralytics_classifier(
    weights: Path,
    samples: list[tuple[Path, str]],
    imgsz: int,
    batch: int,
    device: str | None,
) -> tuple[list[str], list[str], list[str]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    image_paths = [str(path) for path, _label in samples]
    y_true = [label for _path, label in samples]
    y_pred: list[str] = []
    classes = sorted(set(y_true))

    results = model.predict(
        source=image_paths,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        stream=True,
    )
    for result in results:
        names = getattr(result, "names", None) or getattr(model, "names", {})
        pred_id = int(result.probs.top1)
        if isinstance(names, dict):
            y_pred.append(str(names[pred_id]))
        else:
            y_pred.append(str(names[pred_id]))
    classes = sorted(set(classes) | set(y_pred))
    return y_true, y_pred, classes


def evaluate_classifier_on_crops(
    args: argparse.Namespace,
    weights: Path,
    crop_root: Path,
) -> dict:
    samples = list_imagefolder_samples(crop_root / "test")
    if not samples:
        return {"num_crops": 0, "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    backend = resolve_classifier_backend(args)
    if backend == "torchvision":
        y_true, y_pred, classes = predict_torchvision_classifier(
            weights=weights,
            samples=samples,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
        )
    else:
        y_true, y_pred, classes = predict_ultralytics_classifier(
            weights=weights,
            samples=samples,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    return classification_metrics(y_true, y_pred, sorted(classes))


def evaluate_benchmark_crops(args: argparse.Namespace, dataset_dir: Path, weights: Path) -> Path:
    benchmark_yaml_dir = (args.benchmark_yaml_dir or dataset_dir / "benchmark_yamls").resolve()
    if not benchmark_yaml_dir.exists():
        raise FileNotFoundError(f"Benchmark YAML directory not found: {benchmark_yaml_dir}")
    if not weights.exists():
        raise FileNotFoundError(f"Classifier weights not found: {weights}")

    eval_crops_dir = (args.eval_crops_dir or Path(args.project) / args.name / "benchmark_crops").resolve()
    output_csv = (args.eval_output_csv or Path(args.project) / args.name / "benchmark_classification_metrics.csv").resolve()
    if eval_crops_dir.exists() and not args.keep_eval_crops:
        shutil.rmtree(eval_crops_dir)
    eval_crops_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    yaml_paths = sorted(benchmark_yaml_dir.glob("*.yaml"), key=benchmark_sort_key)
    for yaml_path in yaml_paths:
        scenario, level = parse_benchmark_name(yaml_path)
        if scenario == "noise_only":
            print(f"Skipping {yaml_path.name}: noise-only has no classification labels.")
            continue

        crop_root = eval_crops_dir / yaml_path.stem
        if crop_root.exists():
            shutil.rmtree(crop_root)
        counts = export_split(
            dataset_dir=dataset_dir,
            out_dir=crop_root,
            split="test",
            data_yaml=yaml_path,
            names=load_names(yaml_path),
            class_source="label",
            filename_pattern=re.compile(args.filename_regex),
            padding=args.padding,
            min_size=args.min_size,
            jpg_quality=args.jpg_quality,
        )
        if counts["crops"] == 0:
            print(f"Skipping {yaml_path.name}: no crops.")
            continue

        metrics = evaluate_classifier_on_crops(args, weights, crop_root)
        row = {
            "benchmark": yaml_path.stem,
            "scenario": scenario,
            "level": level,
            "yaml": str(yaml_path),
            "weights": str(weights),
            **metrics,
        }
        rows.append(row)
        print(
            f"{yaml_path.stem}: crops={metrics['num_crops']}, "
            f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
        )

        if not args.keep_eval_crops:
            shutil.rmtree(crop_root)

    if not rows:
        raise RuntimeError("No benchmark classification metrics were produced.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Benchmark classification metrics: {output_csv}")
    return output_csv


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    data_yaml = (args.data_yaml.resolve() if args.data_yaml else dataset_dir / "data.yaml")
    if args.data_yaml is None and not data_yaml.exists():
        data_yaml = dataset_dir / "source_grouped_splits" / "data_with_stress.yaml"
    data_yaml = data_yaml.resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if args.padding < 0:
        raise ValueError("--padding must be >= 0")
    if args.min_size < 1:
        raise ValueError("--min-size must be >= 1")
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = load_names(data_yaml)
    filename_pattern = re.compile(args.filename_regex)

    if not args.skip_split_crop:
        total: Counter = Counter()
        split_totals: dict[str, Counter] = {}
        for split in args.splits:
            split_counts = export_split(
                dataset_dir=dataset_dir,
                out_dir=out_dir,
                split=split,
                data_yaml=data_yaml if data_yaml.exists() else None,
                names=names,
                class_source=args.class_source,
                filename_pattern=filename_pattern,
                padding=args.padding,
                min_size=args.min_size,
                jpg_quality=args.jpg_quality,
            )
            split_totals[split] = split_counts
            total.update(split_counts)

        print(f"Dataset: {dataset_dir}")
        print(f"Output: {out_dir}")
        print(f"Class source: {args.class_source}")
        print(f"Crops written: {total['crops']}")
        for split in args.splits:
            split_counts = split_totals[split]
            class_counts = {
                key[1]: value
                for key, value in split_counts.items()
                if isinstance(key, tuple) and key[0] == split
            }
            print(f"{split}: {sum(class_counts.values())} crops from {split_counts['images']} discovered images")
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count}")
        for key in ("missing_images", "missing_labels", "unreadable_images", "bad_labels", "small_crops"):
            if total[key]:
                print(f"{key}: {total[key]}")
    else:
        print("Skipping normal split crop export.")

    trained_weights: Path | None = None
    if args.train_classifier:
        trained_weights = train_classifier(args, out_dir)
    if args.eval_benchmark_crops:
        eval_weights = args.classifier_weights or trained_weights
        if eval_weights is None:
            raise ValueError("--eval-benchmark-crops requires --classifier-weights unless --train-classifier ran first")
        evaluate_benchmark_crops(args, dataset_dir, eval_weights.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
