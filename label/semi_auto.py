from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MaskMethod = Literal["percentile", "adaptive_row", "cfar_track"]
Box = tuple[int, int, int, int, int]


@dataclass(slots=True)
class SemiAutoLabelConfig:
    out_dir: Path
    manifest: Path | None = None
    image_root: Path | None = None
    method: MaskMethod = "cfar_track"
    percentile: float = 94.0
    row_margin: float = 35.0
    cfar_percentile: float = 60.0
    cfar_margin: float = 28.0
    cfar_time_smooth: int = 31
    fallback_percentile: float = 0.0
    time_dilate: int = 1
    freq_dilate: int = 1
    band_min_height: int = 4
    band_merge_gap: int = 3
    max_band_height_ratio: float = 0.75
    track_overlap: float = 0.2
    track_gap: int = 3
    track_min_frames: int = 2
    blur_kernel: int = 5
    morph_kernel: int = 11
    morph_iterations: int = 1
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.85
    max_box_height_ratio: float = 0.95
    max_box_width_ratio: float = 0.95
    ignore_edge_ratio: float = 0.0
    edge_small_height_ratio: float = 0.08
    min_width: int = 4
    min_height: int = 4
    max_boxes: int = 12
    pad_ratio: float = 0.0
    tighten_boxes: bool = False
    tighten_percentile: float = 85.0
    tighten_min_pixels: int = 8
    class_id: int = 0
    class_name: str = "uav_signal"
    max_images: int | None = None
    random_sample: bool = False
    balanced_by_class: bool = False
    class_column: str = "class_name"
    seed: int = 42
    export_yolo: bool = False
    export_cvat_coco: bool = False
    overwrite: bool = False

    def validate(self) -> None:
        if self.manifest is None and self.image_root is None:
            raise ValueError("Provide manifest or image_root.")
        if self.method not in {"percentile", "adaptive_row", "cfar_track"}:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.blur_kernel < 1:
            raise ValueError("blur_kernel must be >= 1.")
        if self.morph_kernel < 1:
            raise ValueError("morph_kernel must be >= 1.")
        if self.max_images is not None and self.max_images < 1:
            raise ValueError("max_images must be positive when set.")


@dataclass(slots=True)
class LabelStats:
    sample_id: str
    image_path: str
    label_path: str
    overlay_path: str
    split: str
    class_name: str
    num_boxes: int
    status: str


def run_semi_auto_label(config: SemiAutoLabelConfig) -> list[LabelStats]:
    """Generate labels, overlays, manifests, and optional export formats."""

    config.validate()
    rows = read_manifest(config.manifest) if config.manifest else discover_images(config.image_root)
    rows = select_rows(rows, config)
    if not rows:
        raise RuntimeError("No images to label.")

    write_selected_rows(config.out_dir / "manifests" / "selected_input_manifest.csv", rows)
    stats: list[LabelStats] = []
    for row in rows:
        stats.append(label_one(row, config))

    write_stats(config.out_dir / "manifests" / "pseudo_label_manifest.csv", stats)
    write_summary(stats, config.out_dir)

    stats_by_sample = {item.sample_id: item for item in stats}
    if config.export_yolo:
        export_yolo(rows, stats_by_sample, config)
    if config.export_cvat_coco:
        export_cvat_coco(rows, stats_by_sample, config)
    return stats


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def discover_images(root: Path | None) -> list[dict[str, str]]:
    if root is None:
        return []
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rows.append(
                {
                    "sample_id": path.stem,
                    "image_path": str(path),
                    "split": "train",
                    "class_name": path.parent.name,
                }
            )
    return rows


def select_rows(rows: list[dict[str, str]], config: SemiAutoLabelConfig) -> list[dict[str, str]]:
    if config.max_images is None:
        return rows
    if config.balanced_by_class:
        return select_balanced_rows(rows, config.max_images, config.class_column, config.seed)
    if config.random_sample and len(rows) > config.max_images:
        return random.Random(config.seed).sample(rows, config.max_images)
    return rows[: config.max_images]


def select_balanced_rows(rows: list[dict[str, str]], max_images: int, class_column: str, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        class_value = row.get(class_column) or row.get("class_name") or "unknown"
        groups.setdefault(class_value, []).append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    selected: list[dict[str, str]] = []
    class_names = sorted(groups)
    while len(selected) < max_images:
        added = False
        rng.shuffle(class_names)
        for class_name in class_names:
            if len(selected) >= max_images:
                break
            if groups[class_name]:
                selected.append(groups[class_name].pop())
                added = True
        if not added:
            break
    rng.shuffle(selected)
    return selected


def write_selected_rows(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_channel(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.max() == gray.min():
        return gray
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def make_mask(gray: np.ndarray, config: SemiAutoLabelConfig) -> np.ndarray:
    if config.blur_kernel > 1:
        kernel_size = config.blur_kernel if config.blur_kernel % 2 else config.blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    if config.method == "percentile":
        threshold = np.percentile(gray, config.percentile)
        mask = (gray >= threshold).astype(np.uint8) * 255
    elif config.method == "adaptive_row":
        row_median = np.median(gray, axis=1, keepdims=True)
        mask = (gray.astype(np.float32) >= row_median.astype(np.float32) + config.row_margin).astype(np.uint8) * 255
    else:
        mask = cfar_track_mask(gray, config)

    if config.morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.morph_kernel, config.morph_kernel))
        for _ in range(config.morph_iterations):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def cfar_track_mask(gray: np.ndarray, config: SemiAutoLabelConfig) -> np.ndarray:
    power = gray.astype(np.float32)
    height, width = power.shape
    noise_floor = np.percentile(power, config.cfar_percentile, axis=1, keepdims=True)
    excess = power - noise_floor

    if config.cfar_time_smooth > 1:
        kernel_size = config.cfar_time_smooth if config.cfar_time_smooth % 2 else config.cfar_time_smooth + 1
        excess = cv2.medianBlur(np.clip(excess, 0, 255).astype(np.uint8), kernel_size).astype(np.float32)

    detections = excess >= config.cfar_margin
    if config.fallback_percentile and config.fallback_percentile > 0:
        detections |= power >= np.percentile(power, config.fallback_percentile)

    if config.time_dilate > 1 or config.freq_dilate > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(1, config.time_dilate), max(1, config.freq_dilate)),
        )
        detections = cv2.dilate(detections.astype(np.uint8) * 255, kernel) > 0

    frame_bands = [
        [
            band
            for band in bands_from_column(detections[:, x], config.band_min_height, config.band_merge_gap)
            if (band[1] - band[0] + 1) <= height * config.max_band_height_ratio
        ]
        for x in range(width)
    ]
    tracks = track_frequency_bands(frame_bands, config.track_overlap, config.track_gap)
    mask = np.zeros((height, width), dtype=np.uint8)
    for track in tracks:
        if len(track["frames"]) < config.track_min_frames:
            continue
        for x, y1, y2 in track["observations"]:
            mask[max(0, int(y1)) : min(height, int(y2) + 1), max(0, int(x)) : min(width, int(x) + 1)] = 255
    return mask


def bands_from_column(column: np.ndarray, min_height: int, merge_gap: int) -> list[tuple[int, int]]:
    ys = np.flatnonzero(column)
    if ys.size == 0:
        return []
    bands = []
    start = int(ys[0])
    prev = int(ys[0])
    for y_value in ys[1:]:
        y = int(y_value)
        if y <= prev + 1 + merge_gap:
            prev = y
            continue
        if prev - start + 1 >= min_height:
            bands.append((start, prev))
        start = prev = y
    if prev - start + 1 >= min_height:
        bands.append((start, prev))
    return bands


def interval_overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    if inter == 0:
        return 0.0
    denom = min(a[1] - a[0] + 1, b[1] - b[0] + 1)
    return inter / max(denom, 1)


def track_frequency_bands(
    frame_bands: list[list[tuple[int, int]]],
    min_overlap: float,
    max_gap: int,
) -> list[dict[str, object]]:
    active: list[dict[str, object]] = []
    closed: list[dict[str, object]] = []
    for x, bands in enumerate(frame_bands):
        assigned_tracks: set[int] = set()
        for band in bands:
            best_idx = None
            best_score = 0.0
            for idx, track in enumerate(active):
                if x - int(track["last_x"]) > max_gap + 1:
                    continue
                score = interval_overlap_ratio(band, (int(track["last_y1"]), int(track["last_y2"])))
                if score > best_score:
                    best_idx = idx
                    best_score = score

            if best_idx is not None and best_score >= min_overlap and best_idx not in assigned_tracks:
                track = active[best_idx]
                track["frames"].append(x)
                track["observations"].append((x, band[0], band[1]))
                track["last_x"] = x
                track["last_y1"], track["last_y2"] = band
                track["y1"] = min(int(track["y1"]), band[0])
                track["y2"] = max(int(track["y2"]), band[1])
                assigned_tracks.add(best_idx)
                continue
            active.append(
                {
                    "frames": [x],
                    "observations": [(x, band[0], band[1])],
                    "last_x": x,
                    "last_y1": band[0],
                    "last_y2": band[1],
                    "y1": band[0],
                    "y2": band[1],
                }
            )

        still_active = []
        for track in active:
            if x - int(track["last_x"]) > max_gap:
                closed.append(track)
            else:
                still_active.append(track)
        active = still_active
    closed.extend(active)
    return closed


def component_boxes(mask: np.ndarray, image_shape: tuple[int, int], config: SemiAutoLabelConfig) -> list[Box]:
    height, width = image_shape
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(1, int(width * height * config.min_area_ratio))
    max_area = int(width * height * config.max_area_ratio)
    pad_x = int(width * config.pad_ratio)
    pad_y = int(height * config.pad_ratio)

    boxes: list[Box] = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < min_area or area > max_area:
            continue
        if w < config.min_width or h < config.min_height:
            continue
        if h > height * config.max_box_height_ratio or w > width * config.max_box_width_ratio:
            continue
        if config.ignore_edge_ratio > 0 and h < height * config.edge_small_height_ratio:
            edge_px = height * config.ignore_edge_ratio
            if y < edge_px or (y + h) > height - edge_px:
                continue
        x1 = max(0, int(x) - pad_x)
        y1 = max(0, int(y) - pad_y)
        x2 = min(width - 1, int(x + w) + pad_x)
        y2 = min(height - 1, int(y + h) + pad_y)
        boxes.append((int(area), x1, y1, x2, y2))
    boxes.sort(reverse=True)
    return boxes[: config.max_boxes]


def tighten_boxes(boxes: list[Box], gray: np.ndarray, config: SemiAutoLabelConfig) -> list[Box]:
    if not config.tighten_boxes:
        return boxes

    height, width = gray.shape[:2]
    tightened: list[Box] = []
    for area, x1, y1, x2, y2 in boxes:
        roi = gray[y1 : y2 + 1, x1 : x2 + 1]
        if roi.size == 0:
            continue
        threshold = np.percentile(roi, config.tighten_percentile)
        ys, xs = np.where(roi >= threshold)
        if xs.size < config.tighten_min_pixels:
            tightened.append((area, x1, y1, x2, y2))
            continue

        nx1 = max(0, x1 + int(xs.min()))
        nx2 = min(width - 1, x1 + int(xs.max()))
        ny1 = max(0, y1 + int(ys.min()))
        ny2 = min(height - 1, y1 + int(ys.max()))
        if nx2 - nx1 + 1 < config.min_width or ny2 - ny1 + 1 < config.min_height:
            tightened.append((area, x1, y1, x2, y2))
            continue
        tightened.append((int((nx2 - nx1 + 1) * (ny2 - ny1 + 1)), nx1, ny1, nx2, ny2))
    tightened.sort(reverse=True)
    return tightened[: config.max_boxes]


def write_yolo_label(label_path: Path, boxes: list[Box], image_shape: tuple[int, int], config: SemiAutoLabelConfig) -> None:
    height, width = image_shape
    ensure_dir(label_path.parent)
    with label_path.open("w", encoding="utf-8") as f:
        for _, x1, y1, x2, y2 in boxes:
            xc = ((x1 + x2) / 2.0) / width
            yc = ((y1 + y2) / 2.0) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            f.write(f"{config.class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n")


def save_overlay(overlay_path: Path, image: np.ndarray, boxes: list[Box]) -> None:
    overlay = image.copy()
    for _, x1, y1, x2, y2 in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ensure_dir(overlay_path.parent)
    cv2.imwrite(str(overlay_path), overlay)


def label_one(row: dict[str, str], config: SemiAutoLabelConfig) -> LabelStats:
    image_path = Path(row["image_path"])
    sample_id = row.get("sample_id") or image_path.stem
    split = row.get("split") or "train"
    class_name = row.get("class_name") or image_path.parent.name
    label_path = config.out_dir / "labels" / split / f"{sample_id}.txt"
    overlay_path = config.out_dir / "overlays" / split / f"{sample_id}.jpg"

    if label_path.exists() and overlay_path.exists() and not config.overwrite:
        num_boxes = len([line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        return LabelStats(sample_id, str(image_path), str(label_path), str(overlay_path), split, class_name, num_boxes, "cached")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        ensure_dir(label_path.parent)
        label_path.write_text("", encoding="utf-8")
        return LabelStats(sample_id, str(image_path), str(label_path), str(overlay_path), split, class_name, 0, "missing_image")

    gray = normalize_channel(image)
    mask = make_mask(gray, config)
    boxes = component_boxes(mask, image.shape[:2], config)
    boxes = tighten_boxes(boxes, gray, config)
    write_yolo_label(label_path, boxes, image.shape[:2], config)
    save_overlay(overlay_path, image, boxes)
    return LabelStats(sample_id, str(image_path), str(label_path), str(overlay_path), split, class_name, len(boxes), "pseudo")


def write_stats(path: Path, stats: list[LabelStats]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(LabelStats.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in stats:
            writer.writerow(asdict(item))


def export_yolo(rows: list[dict[str, str]], stats_by_sample: dict[str, LabelStats], config: SemiAutoLabelConfig) -> None:
    yolo_root = config.out_dir / "yolo"
    for row in rows:
        image_path = Path(row["image_path"])
        sample_id = row.get("sample_id") or image_path.stem
        split = row.get("split") or "train"
        stat = stats_by_sample.get(sample_id)
        if stat is None:
            continue

        dst_img = yolo_root / "images" / split / f"{sample_id}{image_path.suffix.lower()}"
        dst_label = yolo_root / "labels" / split / f"{sample_id}.txt"
        ensure_dir(dst_img.parent)
        ensure_dir(dst_label.parent)
        shutil.copy2(image_path, dst_img)
        shutil.copy2(stat.label_path, dst_label)

    (yolo_root / "data.yaml").write_text(
        "\n".join(
            [
                f"train: {as_posix(yolo_root / 'images' / 'train')}",
                f"val: {as_posix(yolo_root / 'images' / 'val')}",
                f"test: {as_posix(yolo_root / 'images' / 'test')}",
                "nc: 1",
                f"names: ['{config.class_name}']",
                "",
            ]
        ),
        encoding="utf-8",
    )


def as_posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def read_yolo_boxes(label_path: Path, image_width: int, image_height: int) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc = float(parts[1]) * image_width
            yc = float(parts[2]) * image_height
            bw = float(parts[3]) * image_width
            bh = float(parts[4]) * image_height
        except ValueError:
            continue
        x = max(0.0, xc - bw / 2.0)
        y = max(0.0, yc - bh / 2.0)
        bw = max(0.0, min(bw, image_width - x))
        bh = max(0.0, min(bh, image_height - y))
        if bw > 0 and bh > 0:
            boxes.append((class_id, x, y, bw, bh))
    return boxes


def export_cvat_coco(rows: list[dict[str, str]], stats_by_sample: dict[str, LabelStats], config: SemiAutoLabelConfig) -> None:
    coco_root = config.out_dir / "cvat_coco"
    image_root = coco_root / "images"
    annotation_dir = coco_root / "annotations"
    ensure_dir(image_root)
    ensure_dir(annotation_dir)

    images = []
    annotations = []
    used_names: set[str] = set()
    annotation_id = 1
    image_id = 1

    for row in rows:
        image_path = Path(row["image_path"])
        sample_id = row.get("sample_id") or image_path.stem
        stat = stats_by_sample.get(sample_id)
        if stat is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        file_name = f"{sample_id}{image_path.suffix.lower()}"
        if file_name in used_names:
            file_name = f"{row.get('split') or 'train'}_{sample_id}{image_path.suffix.lower()}"
        used_names.add(file_name)
        shutil.copy2(image_path, image_root / file_name)
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

        for class_id, x, y, bw, bh in read_yolo_boxes(Path(stat.label_path), width, height):
            category_id = 1 if class_id == config.class_id else int(class_id) + 1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [round(x, 3), round(y, 3), round(bw, 3), round(bh, 3)],
                    "area": round(bw * bh, 3),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1
        image_id += 1

    coco = {
        "info": {"description": "RFUAV semi-auto pseudo labels exported for CVAT", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": config.class_name, "supercategory": "signal"}],
    }
    (annotation_dir / "instances_default.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")


def write_summary(stats: list[LabelStats], out_dir: Path) -> None:
    summary = {
        "num_images": len(stats),
        "num_labeled_images": sum(1 for item in stats if item.num_boxes > 0),
        "num_empty_images": sum(1 for item in stats if item.num_boxes == 0),
        "total_boxes": sum(item.num_boxes for item in stats),
        "by_split": {},
        "by_class": {},
        "by_status": {},
    }
    for item in stats:
        summary["by_split"][item.split] = summary["by_split"].get(item.split, 0) + 1
        summary["by_class"][item.class_name] = summary["by_class"].get(item.class_name, 0) + 1
        summary["by_status"][item.status] = summary["by_status"].get(item.status, 0) + 1
    ensure_dir(out_dir / "manifests")
    (out_dir / "manifests" / "label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
