"""Model wrappers for detection and optional classification."""

from .classification import ClassificationResult, ImageClassifier, NullClassifier
from .classification_training import (
    CLASSIFICATION_MODEL_CHOICES,
    TORCHVISION_CLASSIFIERS,
    ULTRALYTICS_CLASSIFIERS,
    TorchvisionTrainResult,
    build_torchvision_classifier,
    train_torchvision_classifier,
)
from .detection import DetectionBox, HeuristicSpectrogramDetector, UltralyticsDetector

__all__ = [
    "CLASSIFICATION_MODEL_CHOICES",
    "ClassificationResult",
    "ImageClassifier",
    "NullClassifier",
    "TORCHVISION_CLASSIFIERS",
    "TorchvisionTrainResult",
    "ULTRALYTICS_CLASSIFIERS",
    "DetectionBox",
    "HeuristicSpectrogramDetector",
    "UltralyticsDetector",
    "build_torchvision_classifier",
    "train_torchvision_classifier",
]
