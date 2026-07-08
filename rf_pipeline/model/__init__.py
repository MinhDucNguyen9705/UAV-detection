"""Model wrappers for detection and optional classification."""

from .classification import ClassificationResult, ImageClassifier, NullClassifier
from .detection import DetectionBox, HeuristicSpectrogramDetector, UltralyticsDetector

__all__ = [
    "ClassificationResult",
    "ImageClassifier",
    "NullClassifier",
    "DetectionBox",
    "HeuristicSpectrogramDetector",
    "UltralyticsDetector",
]
