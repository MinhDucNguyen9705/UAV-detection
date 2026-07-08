"""Optional second-stage classification for detected crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    """Auto classifier wrapper.

    Supports:
    - Ultralytics classification checkpoints.
    - Torchvision checkpoints/state_dicts for convnexttiny, efficientnetb0,
      mobilenetv3_large, and mobilenetv3_small.
    """

    def __init__(
        self,
        weights: str | Path,
        imgsz: int = 224,
        device: str | None = None,
        backend: str = "auto",
    ) -> None:
        self.weights = Path(weights)
        self.imgsz = imgsz
        self.device = device
        self.backend = backend
        self.model: Any = None
        self.names: dict[int, str] = {}
        self._torch_transform = None

        if backend in {"auto", "ultralytics"}:
            try:
                self._init_ultralytics()
                self.backend = "ultralytics"
                return
            except Exception as exc:
                if backend == "ultralytics":
                    raise
                self._ultralytics_error = exc

        if backend in {"auto", "torchvision"}:
            try:
                self._init_torchvision()
                self.backend = "torchvision"
                return
            except Exception as exc:
                detail = f"Torchvision load failed: {exc}"
                if hasattr(self, "_ultralytics_error"):
                    detail = f"Ultralytics load failed: {self._ultralytics_error}\n{detail}"
                raise RuntimeError(
                    f"Could not load classifier checkpoint: {self.weights}\n"
                    "If this is a Torchvision checkpoint, make sure the filename or checkpoint "
                    "metadata contains one supported architecture: convnexttiny, efficientnetb0, "
                    "mobilenetv3_large, or mobilenetv3_small.\n"
                    f"{detail}"
                ) from exc

    def _init_ultralytics(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(self.weights))
        names = getattr(self.model, "names", {}) or {}
        self.names = {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else {}

    def _init_torchvision(self) -> None:
        import torch
        from torchvision import transforms

        checkpoint = _torch_load(self.weights, self.device)
        state_dict, metadata_model, names = _extract_checkpoint_parts(checkpoint)
        model_name = _infer_model_name(self.weights, metadata_model)
        num_classes = _infer_num_classes(state_dict, names)
        self.model = _build_torchvision_model(model_name, num_classes)
        missing, unexpected = self.model.load_state_dict(_clean_state_dict(state_dict), strict=False)
        if missing and unexpected:
            raise RuntimeError(f"State dict mismatch. Missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.names = names or {index: str(index) for index in range(num_classes)}

        device = _resolve_torch_device(self.device)
        self.model.to(device)
        self.model.eval()
        self._torch_device = device
        self._torch_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.imgsz, self.imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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
        if self.backend == "torchvision":
            return self._classify_torchvision(crop, fallback_name)
        return self._classify_ultralytics(crop, fallback_name)

    def _classify_ultralytics(self, crop, fallback_name: str) -> ClassificationResult:
        result = self.model.predict(crop, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        probs = getattr(result, "probs", None)
        if probs is None:
            return ClassificationResult(class_id=0, class_name=fallback_name, confidence=0.0)
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        return ClassificationResult(class_id=class_id, class_name=self.names.get(class_id, str(class_id)), confidence=confidence)

    def _classify_torchvision(self, crop, fallback_name: str) -> ClassificationResult:
        import cv2
        import torch

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self._torch_transform(rgb).unsqueeze(0).to(self._torch_device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, class_id = torch.max(probs, dim=0)
        index = int(class_id.item())
        return ClassificationResult(
            class_id=index,
            class_name=self.names.get(index, fallback_name if index == 0 else str(index)),
            confidence=float(confidence.item()),
        )


def _torch_load(path: Path, device: str | None):
    import torch

    map_location = _resolve_torch_device(device)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_torch_device(device: str | None):
    import torch

    if device:
        if device.isdigit():
            return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _extract_checkpoint_parts(checkpoint) -> tuple[dict[str, Any], str | None, dict[int, str]]:
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict(), checkpoint.__class__.__name__.lower(), {}
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported Torch checkpoint object.")

    names = _extract_names(checkpoint)
    model_name = checkpoint.get("model_name") or checkpoint.get("arch") or checkpoint.get("architecture")
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if hasattr(value, "state_dict"):
            return value.state_dict(), model_name or value.__class__.__name__.lower(), names
        if isinstance(value, dict):
            return value, model_name, names
    if all(hasattr(value, "shape") for value in checkpoint.values()):
        return checkpoint, model_name, names
    raise RuntimeError("Could not find state_dict in checkpoint.")


def _extract_names(checkpoint: dict[str, Any]) -> dict[int, str]:
    for key in ("classes", "class_names", "names"):
        value = checkpoint.get(key)
        if isinstance(value, list):
            return {index: str(name) for index, name in enumerate(value)}
        if isinstance(value, dict):
            try:
                return {int(k): str(v) for k, v in value.items()}
            except (TypeError, ValueError):
                pass
    class_to_idx = checkpoint.get("class_to_idx")
    if isinstance(class_to_idx, dict):
        return {int(index): str(name) for name, index in class_to_idx.items()}
    return {}


def _infer_model_name(path: Path, metadata_model: str | None) -> str:
    text = f"{metadata_model or ''} {path.stem}".lower().replace("-", "_")
    compact = text.replace("_", "")
    if "convnexttiny" in compact:
        return "convnext_tiny"
    if "mobilenet" in text and ("v3" in text or "mobilenetv3" in text):
        return "mobilenet_v3_small" if "small" in text else "mobilenet_v3_large"
    if "efficientnetb0" in compact or "efficientnet_b0" in text:
        return "efficientnet_b0"
    raise RuntimeError(f"Cannot infer torchvision architecture from {path.name}.")


def _infer_num_classes(state_dict: dict[str, Any], names: dict[int, str]) -> int:
    if names:
        return max(names) + 1
    for key in ("classifier.3.weight", "classifier.2.weight", "classifier.1.weight"):
        value = state_dict.get(key)
        if value is not None and hasattr(value, "shape"):
            return int(value.shape[0])
    for key, value in state_dict.items():
        if key.endswith(".weight") and hasattr(value, "ndim") and value.ndim == 2:
            return int(value.shape[0])
    raise RuntimeError("Cannot infer number of classes from checkpoint.")


def _build_torchvision_model(model_name: str, num_classes: int):
    from torchvision import models

    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = _linear_like(model.classifier[3], num_classes)
        return model
    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = _linear_like(model.classifier[3], num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = _linear_like(model.classifier[1], num_classes)
        return model
    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = _linear_like(model.classifier[2], num_classes)
        return model
    raise RuntimeError(f"Unsupported torchvision architecture: {model_name}")


def _linear_like(old_layer, out_features: int):
    import torch

    return torch.nn.Linear(old_layer.in_features, out_features)


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.").removeprefix("model.")
        cleaned[key] = value
    return cleaned
