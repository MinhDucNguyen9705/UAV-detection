from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ClassificationResult:
    class_id: int
    class_name: str
    confidence: float
    scores: dict[str, float] | None = None


@dataclass(slots=True)
class ClassificationCrop:
    image_path: Path
    xyxy: tuple[float, float, float, float]
    fallback_name: str
    fallback_class_id: int = 0
    fallback_confidence: float = 0.0
    image: Any | None = None


class NullClassifier:
    """Returns the detector class when no second-stage classifier is configured."""

    def classify_crop(
        self,
        image_path: Path,
        xyxy: tuple[float, float, float, float],
        fallback_name: str,
        fallback_class_id: int = 0,
        fallback_confidence: float = 1.0,
    ) -> ClassificationResult:
        return ClassificationResult(
            class_id=fallback_class_id,
            class_name=fallback_name,
            confidence=fallback_confidence,
            scores={fallback_name: fallback_confidence},
        )

    def classify_crops(self, crops: list[ClassificationCrop]) -> list[ClassificationResult]:
        return [
            self.classify_crop(
                crop.image_path,
                crop.xyxy,
                crop.fallback_name,
                crop.fallback_class_id,
                crop.fallback_confidence,
            )
            for crop in crops
        ]


class ImageClassifier:
    """Auto classifier wrapper.

    Supports:
    - Ultralytics classification checkpoints.
    - Torchvision checkpoints/state_dicts for convnexttiny, efficientnetb0,
      efficientnet_v2_s, mobilenetv3_large, and mobilenetv3_small.
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
        self._onnx_session = None
        self._onnx_input_name = ""
        self._onnx_output_name = ""

        if self.weights.suffix.lower() == ".onnx":
            if backend not in {"auto", "onnx"}:
                raise RuntimeError(f"Backend {backend!r} cannot load ONNX classifier: {self.weights}")
            self._init_onnx()
            self.backend = "onnx"
            return

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
        if isinstance(checkpoint, dict) and checkpoint.get("imgsz"):
            self.imgsz = int(checkpoint["imgsz"])
        self.model = _build_torchvision_model(model_name, num_classes)
        missing, unexpected = self.model.load_state_dict(_clean_state_dict(state_dict), strict=False)
        if missing and unexpected:
            raise RuntimeError(f"State dict mismatch. Missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.names = names or {index: str(index) for index in range(num_classes)}

        device = _resolve_torch_device(self.device)
        self.model.to(device)
        self._torch_use_half = _use_half_precision(self.device)
        if self._torch_use_half:
            self.model.half()
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

    def _init_onnx(self) -> None:
        import onnxruntime as ort

        metadata = _load_onnx_metadata(self.weights)
        if metadata.get("imgsz"):
            self.imgsz = int(metadata["imgsz"])
        classes = metadata.get("classes") or metadata.get("names") or []
        if isinstance(classes, dict):
            self.names = {int(k): str(v) for k, v in classes.items()}
        elif isinstance(classes, list):
            self.names = {index: str(name) for index, name in enumerate(classes)}
        else:
            self.names = {}

        providers, provider_options = _onnx_provider_config(self.device)
        self._onnx_session = ort.InferenceSession(str(self.weights), providers=providers, provider_options=provider_options)
        self._onnx_input_name = str(metadata.get("input_name") or self._onnx_session.get_inputs()[0].name)
        outputs = self._onnx_session.get_outputs()
        self._onnx_output_name = str(metadata.get("output_name") or outputs[0].name)

    def classify_crop(
        self,
        image_path: Path,
        xyxy: tuple[float, float, float, float],
        fallback_name: str,
        fallback_class_id: int = 0,
        fallback_confidence: float = 0.0,
    ) -> ClassificationResult:
        return self.classify_crops(
            [
                ClassificationCrop(
                    image_path=image_path,
                    xyxy=xyxy,
                    fallback_name=fallback_name,
                    fallback_class_id=fallback_class_id,
                    fallback_confidence=fallback_confidence,
                )
            ]
        )[0]

    def classify_crops(self, crops: list[ClassificationCrop]) -> list[ClassificationResult]:
        if not crops:
            return []
        fallback_results, valid_crops = self._prepare_crops(crops)
        if not valid_crops:
            return fallback_results
        if self.backend == "onnx":
            return self._classify_onnx_batch(fallback_results, valid_crops)
        if self.backend == "torchvision":
            return self._classify_torchvision_batch(fallback_results, valid_crops)
        return self._classify_ultralytics_batch(fallback_results, valid_crops)

    def _prepare_crops(self, crops: list[ClassificationCrop]):
        import cv2

        image_cache = {}
        fallback_results = [
            ClassificationResult(
                class_id=crop.fallback_class_id,
                class_name=crop.fallback_name,
                confidence=crop.fallback_confidence,
                scores={crop.fallback_name: crop.fallback_confidence},
            )
            for crop in crops
        ]
        valid_crops = []
        for index, crop in enumerate(crops):
            image = crop.image if crop.image is not None else image_cache.get(crop.image_path)
            if image is None:
                image = cv2.imread(str(crop.image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot read image: {crop.image_path}")
                image_cache[crop.image_path] = image
            height, width = image.shape[:2]
            x1, y1, x2, y2 = [int(round(v)) for v in crop.xyxy]
            x1 = min(max(x1, 0), width)
            x2 = min(max(x2, 0), width)
            y1 = min(max(y1, 0), height)
            y2 = min(max(y2, 0), height)
            if x2 <= x1 or y2 <= y1:
                continue
            valid_crops.append((index, image[y1:y2, x1:x2], crop))
        return fallback_results, valid_crops

    def _classify_onnx(
        self,
        crop,
        fallback_name: str,
        fallback_class_id: int,
        fallback_confidence: float,
    ) -> ClassificationResult:
        import cv2
        import numpy as np

        if self._onnx_session is None:
            return ClassificationResult(
                class_id=fallback_class_id,
                class_name=fallback_name,
                confidence=fallback_confidence,
                scores={fallback_name: fallback_confidence},
            )
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = ((resized - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)
        logits = self._onnx_session.run([self._onnx_output_name], {self._onnx_input_name: tensor})[0]
        probs = _softmax(logits[0])
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        scores = {self.names.get(index, str(index)): float(score) for index, score in enumerate(probs.tolist())}
        return ClassificationResult(
            class_id=class_id,
            class_name=self.names.get(class_id, fallback_name if class_id == fallback_class_id else str(class_id)),
            confidence=confidence,
            scores=scores,
        )

    def _classify_onnx_batch(self, results: list[ClassificationResult], valid_crops) -> list[ClassificationResult]:
        import numpy as np

        if self._onnx_session is None:
            return results
        tensors = [self._preprocess_onnx_crop(crop) for _, crop, _ in valid_crops]
        logits = self._onnx_session.run([self._onnx_output_name], {self._onnx_input_name: np.concatenate(tensors, axis=0)})[0]
        if logits.ndim == 1:
            logits = logits[None, :]
        for row, (index, _, request) in zip(logits, valid_crops):
            probs = _softmax(row)
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            scores = {self.names.get(i, str(i)): float(score) for i, score in enumerate(probs.tolist())}
            results[index] = ClassificationResult(
                class_id=class_id,
                class_name=self.names.get(class_id, request.fallback_name if class_id == request.fallback_class_id else str(class_id)),
                confidence=confidence,
                scores=scores,
            )
        return results

    def _preprocess_onnx_crop(self, crop):
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        return ((resized - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)

    def _classify_ultralytics(
        self,
        crop,
        fallback_name: str,
        fallback_class_id: int,
        fallback_confidence: float,
    ) -> ClassificationResult:
        result = self.model.predict(crop, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        probs = getattr(result, "probs", None)
        if probs is None:
            return ClassificationResult(
                class_id=fallback_class_id,
                class_name=fallback_name,
                confidence=fallback_confidence,
                scores={fallback_name: fallback_confidence},
            )
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        scores = {}
        data = getattr(probs, "data", None)
        if data is not None:
            scores = {self.names.get(index, str(index)): float(score) for index, score in enumerate(data.detach().cpu().tolist())}
        return ClassificationResult(
            class_id=class_id,
            class_name=self.names.get(class_id, str(class_id)),
            confidence=confidence,
            scores=scores or None,
        )

    def _classify_ultralytics_batch(self, results: list[ClassificationResult], valid_crops) -> list[ClassificationResult]:
        try:
            predictions = self.model.predict(
                [crop for _, crop, _ in valid_crops],
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception:
            for index, crop, request in valid_crops:
                results[index] = self._classify_ultralytics(
                    crop,
                    request.fallback_name,
                    request.fallback_class_id,
                    request.fallback_confidence,
                )
            return results
        for result, (index, _, request) in zip(predictions, valid_crops):
            probs = getattr(result, "probs", None)
            if probs is None:
                continue
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
            scores = {}
            data = getattr(probs, "data", None)
            if data is not None:
                scores = {self.names.get(i, str(i)): float(score) for i, score in enumerate(data.detach().cpu().tolist())}
            results[index] = ClassificationResult(
                class_id=class_id,
                class_name=self.names.get(class_id, str(class_id)),
                confidence=confidence,
                scores=scores or None,
            )
        return results

    def _classify_torchvision(
        self,
        crop,
        fallback_name: str,
        fallback_class_id: int,
        fallback_confidence: float,
    ) -> ClassificationResult:
        import cv2
        import torch

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self._torch_transform(rgb).unsqueeze(0).to(self._torch_device)
        if getattr(self, "_torch_use_half", False):
            tensor = tensor.half()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, class_id = torch.max(probs, dim=0)
        index = int(class_id.item())
        scores = {self.names.get(i, str(i)): float(score) for i, score in enumerate(probs.detach().cpu().tolist())}
        return ClassificationResult(
            class_id=index,
            class_name=self.names.get(index, fallback_name if index == fallback_class_id else str(index)),
            confidence=float(confidence.item()),
            scores=scores,
        )

    def _classify_torchvision_batch(self, results: list[ClassificationResult], valid_crops) -> list[ClassificationResult]:
        import cv2
        import torch

        tensors = []
        for _, crop, _ in valid_crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self._torch_transform(rgb))
        batch = torch.stack(tensors, dim=0).to(self._torch_device)
        if getattr(self, "_torch_use_half", False):
            batch = batch.half()
        with torch.no_grad():
            logits = self.model(batch)
            probs_batch = torch.softmax(logits, dim=1)
            confidences, class_ids = torch.max(probs_batch, dim=1)
        for row, confidence, class_id, (index, _, request) in zip(probs_batch, confidences, class_ids, valid_crops):
            class_index = int(class_id.item())
            scores = {self.names.get(i, str(i)): float(score) for i, score in enumerate(row.detach().cpu().tolist())}
            results[index] = ClassificationResult(
                class_id=class_index,
                class_name=self.names.get(class_index, request.fallback_name if class_index == request.fallback_class_id else str(class_index)),
                confidence=float(confidence.item()),
                scores=scores,
            )
        return results


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


def _use_half_precision(device: str | None) -> bool:
    if not device or str(device).strip().lower() == "cpu":
        return False
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


def _load_onnx_metadata(path: Path) -> dict[str, Any]:
    candidates = [
        path.with_suffix(path.suffix + ".json"),
        path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _onnx_provider_config(device: str | None) -> tuple[list[str], list[dict[str, str]]]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX classifier inference requires `onnxruntime`. Install it before loading .onnx weights.") from exc

    available = set(ort.get_available_providers())
    wants_cuda = bool(device and str(device).strip().lower() != "cpu")
    if wants_cuda and "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "ONNX classifier was asked to run on GPU, but ONNX Runtime does not expose CUDAExecutionProvider. "
            "Install onnxruntime-gpu in this environment, or set Device to cpu."
        )
    if device != "cpu" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{"device_id": str(_onnx_device_id(device))}, {}]
    return ["CPUExecutionProvider"], [{}]


def _onnx_device_id(device: str | None) -> int:
    if not device:
        return 0
    text = str(device).strip().lower()
    if text.isdigit():
        return int(text)
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return 0


def _softmax(logits):
    import numpy as np

    values = np.asarray(logits, dtype=np.float32)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.maximum(np.sum(exp), 1e-12)


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
    if "efficientnetv2s" in compact or "efficientnet_v2_s" in text:
        return "efficientnet_v2_s"
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
    if model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
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
