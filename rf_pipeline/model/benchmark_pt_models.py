from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from .classification import ImageClassifier
    from .detection import UltralyticsDetector
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rf_pipeline.model.classification import ImageClassifier
    from rf_pipeline.model.detection import UltralyticsDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSIFIER_HINTS = ("classifier", "classify", "classification", "mobilenet", "efficientnet", "convnext")
SINGLE_DETECTOR_HINTS = ("single", "single_class", "1class", "one_class")


@dataclass(slots=True)
class ModelGroups:
    two_class_detectors: list[Path]
    single_class_detectors: list[Path]
    classifiers: list[Path]


@dataclass(slots=True)
class BenchmarkRow:
    benchmark: str
    detector: str
    classifier: str
    images: int
    detections: int
    classifications: int
    warmup_runs: int
    repeat_runs: int
    elapsed_sec: float
    fps: float
    detector_fps: float | None = None
    classifier_fps: float | None = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch .pt detector and detector+classifier pipelines.")
    parser.add_argument("--models-dir", type=Path, required=True, help="Folder containing .pt models.")
    parser.add_argument("--images-dir", type=Path, default=None, help="Folder containing spectrogram/images used for benchmarking.")
    parser.add_argument("--synthetic-images", type=int, default=0, help="Generate random benchmark images when --images-dir is not provided.")
    parser.add_argument("--synthetic-width", type=int, default=960, help="Synthetic image width.")
    parser.add_argument("--synthetic-height", type=int, default=720, help="Synthetic image height.")
    parser.add_argument("--synthetic-seed", type=int, default=2026, help="Random seed for synthetic images.")
    parser.add_argument("--out", type=Path, default=Path("runs/pt_benchmark"), help="Output directory for CSV/JSON reports.")
    parser.add_argument("--device", default="0", help="Device for inference: 0/cuda:0 for GPU, or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Detector image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="Detector IoU threshold.")
    parser.add_argument("--batch", type=int, default=16, help="Detector batch size.")
    parser.add_argument("--warmup-runs", type=int, default=2, help="Warmup passes before timing.")
    parser.add_argument("--repeat-runs", type=int, default=5, help="Timed passes over the image set.")
    parser.add_argument("--limit-images", type=int, default=0, help="Limit number of benchmark images. 0 uses all images.")
    parser.add_argument("--detector-two-pattern", default="", help="Optional glob/name fragment for 2-class detector .pt files.")
    parser.add_argument("--detector-single-pattern", default="", help="Optional glob/name fragment for single-class detector .pt files.")
    parser.add_argument("--classifier-pattern", default="", help="Optional glob/name fragment for classifier .pt files.")
    parser.add_argument("--classifier-backend", default="auto", choices=["auto", "ultralytics", "torchvision"], help="Classifier loader backend.")
    parser.add_argument("--classify-full-frame-if-empty", action="store_true", help="Run classifier on the full image when detector returns no boxes.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first model error instead of recording it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.images_dir and not args.classify_full_frame_if_empty:
        args.classify_full_frame_if_empty = True

    model_groups = discover_models(
        args.models_dir,
        two_pattern=args.detector_two_pattern,
        single_pattern=args.detector_single_pattern,
        classifier_pattern=args.classifier_pattern,
    )
    args.out.mkdir(parents=True, exist_ok=True)

    with image_source(args) as images:
        if not images:
            source = args.images_dir if args.images_dir else "synthetic images"
            raise FileNotFoundError(f"No benchmark images found from {source}")

        rows: list[BenchmarkRow] = []
        print(f"Images: {len(images)}")
        print(f"2-class detectors: {len(model_groups.two_class_detectors)}")
        print(f"Single-class detectors: {len(model_groups.single_class_detectors)}")
        print(f"Classifiers: {len(model_groups.classifiers)}")

        for detector_path in model_groups.two_class_detectors:
            rows.append(
                run_guarded(
                    lambda p=detector_path: benchmark_detector(p, images=images, args=args, benchmark_name="detector_two_class"),
                    args.fail_fast,
                    detector=detector_path.name,
                    classifier="",
                    benchmark="detector_two_class",
                )
            )
            print_row(rows[-1])

        for detector_path in model_groups.single_class_detectors:
            for classifier_path in model_groups.classifiers:
                rows.append(
                    run_guarded(
                        lambda d=detector_path, c=classifier_path: benchmark_detector_classifier(d, c, images=images, args=args),
                        args.fail_fast,
                        detector=detector_path.name,
                        classifier=classifier_path.name,
                        benchmark="single_detector_plus_classifier",
                    )
                )
                print_row(rows[-1])

        write_reports(rows, args.out)

    print(f"Reports: {args.out / 'pt_benchmark.csv'}")
    return 0


def run_guarded(fn, fail_fast: bool, detector: str, classifier: str, benchmark: str) -> BenchmarkRow:
    try:
        return fn()
    except Exception as exc:
        if fail_fast:
            raise
        return BenchmarkRow(
            benchmark=benchmark,
            detector=detector,
            classifier=classifier,
            images=0,
            detections=0,
            classifications=0,
            warmup_runs=0,
            repeat_runs=0,
            elapsed_sec=0.0,
            fps=0.0,
            error=f"{exc.__class__.__name__}: {exc}",
        )


def benchmark_detector(detector_path: Path, images: list[Path], args: argparse.Namespace, benchmark_name: str) -> BenchmarkRow:
    detector = UltralyticsDetector(
        detector_path,
        architecture="auto",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
    )
    for _ in range(max(0, args.warmup_runs)):
        detector.predict(images)

    total_detections = 0
    start = time.perf_counter()
    for _ in range(max(1, args.repeat_runs)):
        predictions = detector.predict(images)
        total_detections += sum(len(items) for items in predictions.values())
    elapsed = time.perf_counter() - start
    frames = len(images) * max(1, args.repeat_runs)
    return BenchmarkRow(
        benchmark=benchmark_name,
        detector=detector_path.name,
        classifier="",
        images=frames,
        detections=total_detections,
        classifications=0,
        warmup_runs=max(0, args.warmup_runs),
        repeat_runs=max(1, args.repeat_runs),
        elapsed_sec=elapsed,
        fps=frames / elapsed if elapsed > 0 else 0.0,
    )


def benchmark_detector_classifier(detector_path: Path, classifier_path: Path, images: list[Path], args: argparse.Namespace) -> BenchmarkRow:
    detector = UltralyticsDetector(
        detector_path,
        architecture="auto",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
    )
    classifier = ImageClassifier(classifier_path, device=args.device, backend=args.classifier_backend)

    for _ in range(max(0, args.warmup_runs)):
        classify_predictions(detector.predict(images), classifier, args.classify_full_frame_if_empty)

    total_detections = 0
    total_classifications = 0
    detector_elapsed = 0.0
    classifier_elapsed = 0.0
    start = time.perf_counter()
    for _ in range(max(1, args.repeat_runs)):
        detector_start = time.perf_counter()
        predictions = detector.predict(images)
        detector_elapsed += time.perf_counter() - detector_start
        total_detections += sum(len(items) for items in predictions.values())

        classifier_start = time.perf_counter()
        total_classifications += classify_predictions(predictions, classifier, args.classify_full_frame_if_empty)
        classifier_elapsed += time.perf_counter() - classifier_start
    elapsed = time.perf_counter() - start

    frames = len(images) * max(1, args.repeat_runs)
    return BenchmarkRow(
        benchmark="single_detector_plus_classifier",
        detector=detector_path.name,
        classifier=classifier_path.name,
        images=frames,
        detections=total_detections,
        classifications=total_classifications,
        warmup_runs=max(0, args.warmup_runs),
        repeat_runs=max(1, args.repeat_runs),
        elapsed_sec=elapsed,
        fps=frames / elapsed if elapsed > 0 else 0.0,
        detector_fps=frames / detector_elapsed if detector_elapsed > 0 else None,
        classifier_fps=total_classifications / classifier_elapsed if classifier_elapsed > 0 else None,
    )


def classify_predictions(predictions, classifier: ImageClassifier, classify_full_frame_if_empty: bool) -> int:
    count = 0
    for image_path, detections in predictions.items():
        if not detections and classify_full_frame_if_empty:
            classifier.classify_crop(image_path, full_image_box(image_path), fallback_name="signal", fallback_class_id=0)
            count += 1
            continue
        for detection in detections:
            classifier.classify_crop(image_path, detection.xyxy, detection.class_name, detection.class_id, detection.confidence)
            count += 1
    return count


def full_image_box(path: Path) -> tuple[float, float, float, float]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    height, width = image.shape[:2]
    return 0.0, 0.0, float(width), float(height)


def discover_images(images_dir: Path, limit: int) -> list[Path]:
    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    return images[:limit] if limit and limit > 0 else images


class image_source:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> list[Path]:
        if self.args.images_dir:
            return discover_images(self.args.images_dir, self.args.limit_images)
        count = self.args.synthetic_images or max(self.args.batch, 32)
        self._temp_dir = tempfile.TemporaryDirectory(prefix="pt_benchmark_images_")
        return generate_synthetic_images(
            Path(self._temp_dir.name),
            count=count,
            width=self.args.synthetic_width,
            height=self.args.synthetic_height,
            seed=self.args.synthetic_seed,
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()


def generate_synthetic_images(image_dir: Path, count: int, width: int, height: int, seed: int) -> list[Path]:
    import cv2
    import numpy as np

    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for index in range(max(1, count)):
        image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        path = image_dir / f"synthetic_{index:05d}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write synthetic image: {path}")
        paths.append(path)
    return paths


def discover_models(models_dir: Path, two_pattern: str, single_pattern: str, classifier_pattern: str) -> ModelGroups:
    models = sorted(models_dir.rglob("*.pt"))
    classifiers: list[Path] = []
    single_detectors: list[Path] = []
    two_detectors: list[Path] = []

    for path in models:
        if matches_filter(path, classifier_pattern):
            classifiers.append(path)
            continue
        if matches_filter(path, single_pattern):
            single_detectors.append(path)
            continue
        if matches_filter(path, two_pattern):
            two_detectors.append(path)
            continue

        kind = infer_model_kind(path)
        if kind == "classifier":
            classifiers.append(path)
        elif kind == "single_detector":
            single_detectors.append(path)
        else:
            two_detectors.append(path)

    return ModelGroups(
        two_class_detectors=dedupe(two_detectors),
        single_class_detectors=dedupe(single_detectors),
        classifiers=dedupe(classifiers),
    )


def matches_filter(path: Path, pattern: str) -> bool:
    pattern = pattern.strip()
    if not pattern:
        return False
    return path.match(pattern) or pattern.lower() in path.name.lower()


def infer_model_kind(path: Path) -> str:
    name = path.stem.lower()
    if any(hint in name for hint in CLASSIFIER_HINTS):
        return "classifier"
    if any(hint in name for hint in SINGLE_DETECTOR_HINTS):
        return "single_detector"
    return "two_detector"


def dedupe(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def write_reports(rows: list[BenchmarkRow], out_dir: Path) -> None:
    csv_path = out_dir / "pt_benchmark.csv"
    json_path = out_dir / "pt_benchmark.json"
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(BenchmarkRow.__dataclass_fields__)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")


def print_row(row: BenchmarkRow) -> None:
    classifier = f" + {row.classifier}" if row.classifier else ""
    if row.error:
        print(f"[ERROR] {row.benchmark}: {row.detector}{classifier}: {row.error}")
        return
    detail = f"fps={row.fps:.2f}, images={row.images}, det={row.detections}, cls={row.classifications}"
    if row.detector_fps is not None:
        detail += f", detector_fps={row.detector_fps:.2f}"
    if row.classifier_fps is not None:
        detail += f", classifier_fps={row.classifier_fps:.2f}"
    print(f"[OK] {row.benchmark}: {row.detector}{classifier}: {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
