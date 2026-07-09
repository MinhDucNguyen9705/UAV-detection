from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class ModelPseudoLabelConfig:
    model: str
    source: Path
    output: Path | None = None
    split: str = "all"
    architecture: str = "auto"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    batch: int = 16
    device: str | None = None
    classes: list[int] | None = None
    with_confidence: bool = False
    save_images: bool = False
    recursive: bool = True
    skip_empty: bool = False
    chunk_size: int = 256
    half: bool = False
    max_det: int = 300


@dataclass(slots=True)
class ModelPseudoLabelResult:
    image_root: Path
    labels_dir: Path
    images_dir: Path
    processed_images: int
    saved_detections: int


def run_model_pseudo_label(config: ModelPseudoLabelConfig) -> ModelPseudoLabelResult:
    source = config.source.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Image folder not found: {source}")

    image_root, images, labels_dir, images_dir = resolve_layout(
        source=source,
        output=config.output,
        split=config.split,
        recursive=config.recursive,
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {source}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    if config.save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(config.model, config.architecture)
    processed_count = 0
    detection_count = 0

    torch = _try_import_torch()
    for chunk in iter_chunks(images, config.chunk_size):
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            batch=config.batch,
            device=config.device,
            classes=config.classes,
            stream=True,
            verbose=False,
            save=False,
            half=config.half,
            max_det=config.max_det,
        )

        for image_path, result in zip(chunk, results):
            relative = _relative_image_path(image_path, image_root)
            rows = yolo_rows_from_result(result, config.with_confidence)
            write_label_file(labels_dir, relative, rows, config.skip_empty)
            if config.save_images:
                save_plotted_image(images_dir, relative, result)
            processed_count += 1
            detection_count += len(rows)

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ModelPseudoLabelResult(
        image_root=image_root,
        labels_dir=labels_dir,
        images_dir=images_dir,
        processed_images=processed_count,
        saved_detections=detection_count,
    )


def load_model(checkpoint: str, architecture: str):
    from ultralytics import RTDETR, YOLO

    if architecture == "auto":
        architecture = "rtdetr" if Path(checkpoint).name.lower().startswith("rtdetr") else "yolo"
    return RTDETR(checkpoint) if architecture == "rtdetr" else YOLO(checkpoint)


def find_images(source: Path, recursive: bool) -> list[Path]:
    iterator = source.rglob("*") if recursive else source.glob("*")
    return sorted(path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def relocate_manifest_image(path: Path, image_root: Path) -> Path:
    """Relocate an absolute manifest path after the RFUAV output folder was moved."""

    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index].lower() == "spectrograms" and parts[index + 1].lower() == "images":
            return image_root.joinpath(*parts[index + 2 :]).resolve()
    return path


def iter_chunks(items: list[Path], chunk_size: int) -> Iterator[list[Path]]:
    size = max(1, chunk_size)
    for index in range(0, len(items), size):
        yield items[index : index + size]


def resolve_layout(
    source: Path,
    output: Path | None,
    split: str,
    recursive: bool,
) -> tuple[Path, list[Path], Path, Path]:
    """Resolve a plain image folder or an RFUAV dataset builder output directory."""

    rfuav_image_root = source / "spectrograms" / "images"
    if rfuav_image_root.is_dir():
        image_root = rfuav_image_root.resolve()
        images = _images_from_rfuav_output(source, image_root, split)
        if output is None:
            labels_dir = source / "spectrograms" / "labels"
            plotted_dir = source / "spectrograms" / "predictions"
        else:
            labels_dir = output.resolve() / "labels"
            plotted_dir = output.resolve() / "images"
        return image_root, images, labels_dir, plotted_dir

    if split != "all":
        raise ValueError("--split is only supported when --source is an RFUAV dataset builder out-dir.")

    image_root = source.resolve()
    images = find_images(image_root, recursive=recursive)
    output_root = output.resolve() if output else source.parent / f"{source.name}_predictions"
    return image_root, images, output_root / "labels", output_root / "images"


def yolo_rows_from_result(result, with_confidence: bool) -> list[str]:
    rows: list[str] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return rows

    cls = boxes.cls.detach().cpu().numpy()
    xywhn = boxes.xywhn.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    for class_id, box_xywhn, confidence in zip(cls, xywhn, conf):
        values = [str(int(class_id)), *(f"{float(value):.6f}" for value in box_xywhn)]
        if with_confidence:
            values.append(f"{float(confidence):.6f}")
        rows.append(" ".join(values))
    return rows


def write_label_file(labels_dir: Path, relative_image: Path, rows: list[str], skip_empty: bool) -> Path | None:
    if not rows and skip_empty:
        return None
    label_file = (labels_dir / relative_image).with_suffix(".txt")
    label_file.parent.mkdir(parents=True, exist_ok=True)
    label_file.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return label_file


def save_plotted_image(images_dir: Path, relative_image: Path, result) -> Path:
    import cv2

    plotted_file = images_dir / relative_image
    plotted_file.parent.mkdir(parents=True, exist_ok=True)
    plotted = result.plot()
    cv2.imwrite(str(plotted_file), plotted)
    return plotted_file


def _images_from_rfuav_output(source: Path, image_root: Path, split: str) -> list[Path]:
    if split == "all":
        return find_images(image_root, recursive=True)

    split_file = source / "splits" / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"RFUAV split file not found: {split_file}")

    images: list[Path] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        image = Path(value)
        if not image.is_absolute():
            image = source / image
        image = image.resolve()
        if not image.is_file():
            image = relocate_manifest_image(image, image_root)
        if image.is_file():
            images.append(image)
        else:
            raise FileNotFoundError(f"Image listed in {split_file} not found: {image}")
    return images


def _relative_image_path(image_path: Path, image_root: Path) -> Path:
    try:
        return image_path.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(f"Image is outside the spectrogram image root: {image_path}") from exc


def _try_import_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None

