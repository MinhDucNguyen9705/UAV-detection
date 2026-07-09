from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import yaml


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    height, width = image.shape[:2]
    return width, height


def find_images(source: Path, recursive: bool = True) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {source}")
        return [source.resolve()]
    iterator = source.rglob("*") if recursive else source.glob("*")
    return sorted(path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def resolve_dataset_root(yaml_path: Path, data: dict[str, Any]) -> Path:
    raw_root = Path(str(data.get("path", yaml_path.parent)))
    if raw_root.is_absolute():
        return raw_root.resolve()
    return (yaml_path.parent / raw_root).resolve()


def read_yaml_split_images(yaml_path: Path, split: str = "test") -> list[Path]:
    data = load_yaml(yaml_path)
    if split not in data:
        return []
    root = resolve_dataset_root(yaml_path, data)
    values = data[split] if isinstance(data[split], list) else [data[split]]
    images: list[Path] = []
    for value in values:
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        if path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text:
                    continue
                image = Path(text)
                if not image.is_absolute():
                    image = root / image
                images.append(image.resolve())
        elif path.is_dir():
            images.extend(find_images(path, recursive=True))
        else:
            images.append(path.resolve())
    return images


def write_resolved_ultralytics_yaml(yaml_path: Path, output_dir: Path, prefix: str) -> Path:
    data = load_yaml(yaml_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "path": str(resolve_dataset_root(yaml_path, data)),
        "names": data.get("names", {0: "uav_signal"}),
    }
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        list_path = output_dir / f"{prefix}_{split}.txt"
        with list_path.open("w", encoding="utf-8") as f:
            for image in read_yaml_split_images(yaml_path, split):
                f.write(f"{image.as_posix()}\n")
        resolved[split] = str(list_path.resolve())
    resolved_yaml = output_dir / f"{prefix}.yaml"
    with resolved_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)
    return resolved_yaml


def infer_architecture(model_or_weights: str | Path, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "rtdetr" if Path(str(model_or_weights)).name.lower().startswith("rtdetr") else "yolo"


def load_ultralytics_model(model_or_weights: str | Path, architecture: str = "auto"):
    from ultralytics import RTDETR, YOLO

    resolved = infer_architecture(model_or_weights, architecture)
    return RTDETR(str(model_or_weights)) if resolved == "rtdetr" else YOLO(str(model_or_weights))


def normalize_class_names(names: Any, override: list[str] | None = None) -> dict[int, str]:
    if override:
        return {index: name for index, name in enumerate(override)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {}


def scenario_for_image(image: Path) -> str:
    parts = image.parts
    for index, part in enumerate(parts[:-1]):
        if part.lower() == "images" and index + 1 < len(parts):
            return parts[index + 1]
    parent = image.parent.name
    return parent if parent in {"low_snr", "mix2", "near_far", "noise_only", "clean_single"} else ""


def relative_to_dataset(image: Path, dataset_dir: Path) -> Path:
    try:
        return image.relative_to(dataset_dir)
    except ValueError:
        return Path(image.name)
