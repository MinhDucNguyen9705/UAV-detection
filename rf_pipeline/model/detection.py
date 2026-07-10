from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class DetectionBox:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class Detector(Protocol):
    def predict(self, image_paths: list[Path]) -> dict[Path, list[DetectionBox]]:
        ...


class UltralyticsDetector:
    """YOLO/RT-DETR wrapper with the same auto-selection used by old scripts."""

    def __init__(
        self,
        weights: str | Path,
        architecture: str = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str | None = None,
        classes: list[int] | None = None,
        batch: int = 16,
    ) -> None:
        from ultralytics import RTDETR, YOLO

        self.weights = str(weights)
        self.architecture = architecture
        if architecture == "auto":
            architecture = "rtdetr" if Path(weights).name.lower().startswith("rtdetr") else "yolo"
        self.model = RTDETR(self.weights) if architecture == "rtdetr" else YOLO(self.weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.classes = classes
        self.batch = batch
        is_onnx = Path(self.weights).suffix.lower() == ".onnx"
        if is_onnx:
            _validate_onnx_runtime_device(device)
        self.names = _load_onnx_names(Path(self.weights)) if is_onnx else {}

    def predict(self, image_paths: list[Path]) -> dict[Path, list[DetectionBox]]:
        results = self.model.predict(
            source=[str(path) for path in image_paths],
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            batch=self.batch,
            stream=True,
            verbose=False,
        )
        out: dict[Path, list[DetectionBox]] = {}
        names = self.names or getattr(self.model, "names", {}) or {}
        for image_path, result in zip(image_paths, results):
            boxes: list[DetectionBox] = []
            if result.boxes is not None and len(result.boxes) > 0:
                for class_id, xyxy, confidence in zip(
                    result.boxes.cls.detach().cpu().tolist(),
                    result.boxes.xyxy.detach().cpu().tolist(),
                    result.boxes.conf.detach().cpu().tolist(),
                ):
                    idx = int(class_id)
                    class_name = names.get(idx, str(idx)) if isinstance(names, dict) else str(idx)
                    boxes.append(
                        DetectionBox(
                            class_id=idx,
                            class_name=class_name,
                            confidence=float(confidence),
                            xyxy=tuple(float(v) for v in xyxy),
                        )
                    )
            out[image_path] = boxes
        return out


class HeuristicSpectrogramDetector:
    """Fallback detector for demos when trained weights are not available.

    It thresholds bright connected components on the rendered spectrogram. This
    is not a replacement for YOLO, but it keeps the whole pipeline runnable.
    """

    def __init__(self, class_name: str = "signal", percentile: float = 99.0, min_area: int = 64) -> None:
        self.class_name = class_name
        self.percentile = percentile
        self.min_area = min_area

    def predict(self, image_paths: list[Path]) -> dict[Path, list[DetectionBox]]:
        import cv2

        out: dict[Path, list[DetectionBox]] = {}
        for path in image_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read image: {path}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold = np.percentile(gray, self.percentile)
            mask = (gray >= threshold).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            boxes: list[DetectionBox] = []
            height, width = gray.shape[:2]
            for label_id in range(1, num_labels):
                x, y, w, h, area = stats[label_id]
                if int(area) < self.min_area:
                    continue
                confidence = min(0.99, float(area) / float(width * height) * 20.0)
                boxes.append(
                    DetectionBox(
                        class_id=0,
                        class_name=self.class_name,
                        confidence=max(0.05, confidence),
                        xyxy=(float(x), float(y), float(x + w), float(y + h)),
                    )
                )
            out[path] = boxes
        return out


def _load_onnx_names(path: Path) -> dict[int, str]:
    for candidate in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
        if not candidate.is_file():
            continue
        try:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        classes = metadata.get("classes") or metadata.get("names") or {}
        if isinstance(classes, dict):
            try:
                return {int(key): str(value) for key, value in classes.items()}
            except (TypeError, ValueError):
                return {}
        if isinstance(classes, list):
            return {index: str(value) for index, value in enumerate(classes)}
    return {}


def _validate_onnx_runtime_device(device: str | None) -> None:
    wants_cuda = bool(device and str(device).strip().lower() != "cpu")
    if not wants_cuda:
        return
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX detector inference requires `onnxruntime` or `onnxruntime-gpu`.") from exc
    if "CUDAExecutionProvider" not in set(ort.get_available_providers()):
        raise RuntimeError(
            "ONNX detector was asked to run on GPU, but ONNX Runtime does not expose CUDAExecutionProvider. "
            "Install onnxruntime-gpu in this environment, or set Device to cpu."
        )
