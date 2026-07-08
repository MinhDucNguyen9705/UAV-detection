"""Optional second-stage classification for detected crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ClassificationResult:
    class_id: int
    class_name: str
    confidence: float


class NullClassifier:
    """Returns the detector class when no second-stage classifier is configured."""

    def classify_crop(self, image_path: Path, xyxy: tuple[float, float, float, float], fallback_name: str) -> ClassificationResult:
        return ClassificationResult(class_id=0, class_name=fallback_name, confidence=1.0)


class ImageClassifier:
    """Ultralytics image-classification wrapper for detection crops."""

    def __init__(self, weights: str | Path, imgsz: int = 224, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.device = device

    def classify_crop(self, image_path: Path, xyxy: tuple[float, float, float, float], fallback_name: str) -> ClassificationResult:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        x1 = min(max(x1, 0), width)
        x2 = min(max(x2, 0), width)
        y1 = min(max(y1, 0), height)
        y2 = min(max(y2, 0), height)
        if x2 <= x1 or y2 <= y1:
            return ClassificationResult(class_id=0, class_name=fallback_name, confidence=0.0)
        crop = image[y1:y2, x1:x2]
        result = self.model.predict(crop, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        probs = getattr(result, "probs", None)
        if probs is None:
            return ClassificationResult(class_id=0, class_name=fallback_name, confidence=0.0)
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        names = getattr(self.model, "names", {}) or {}
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
        return ClassificationResult(class_id=class_id, class_name=class_name, confidence=confidence)
