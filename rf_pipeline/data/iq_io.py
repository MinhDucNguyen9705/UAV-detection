"""Raw IQ file IO.

Supported layouts:
- complex64: native numpy complex64 samples.
- float32/int16: interleaved I,Q scalar values.

The default mirrors the existing scripts in this repo: interleaved float32 IQ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(slots=True)
class IQMetadata:
    """Minimal RF metadata needed to map spectrogram pixels back to RF units."""

    sample_rate_hz: float = 100e6
    center_frequency_hz: float = 2.4e9
    dtype: str = "float32"
    start_time_sec: float = 0.0


def read_iq(path: str | Path, dtype: str = "float32") -> np.ndarray:
    """Read a raw IQ file as complex64 samples."""

    path = Path(path)
    if dtype == "complex64":
        return np.fromfile(path, dtype=np.complex64)
    if dtype not in {"float32", "int16"}:
        raise ValueError(f"Unsupported IQ dtype: {dtype}")

    scalar_dtype = np.float32 if dtype == "float32" else np.int16
    raw = np.fromfile(path, dtype=scalar_dtype)
    if raw.size < 2:
        return np.array([], dtype=np.complex64)
    if raw.size % 2:
        raw = raw[:-1]
    if dtype == "int16":
        raw = raw.astype(np.float32) / 32768.0
    i = raw[0::2].astype(np.float32, copy=False)
    q = raw[1::2].astype(np.float32, copy=False)
    return (i + 1j * q).astype(np.complex64)


def complex_sample_bytes(dtype: str) -> int:
    if dtype == "int16":
        return 4
    if dtype in {"float32", "complex64"}:
        return 8
    raise ValueError(f"Unsupported IQ dtype: {dtype}")


def read_iq_segment(path: str | Path, dtype: str, start_sample: int, sample_count: int) -> np.ndarray:
    """Read a complex IQ segment without loading the full file."""

    path = Path(path)
    if start_sample < 0 or sample_count < 1:
        raise ValueError("start_sample must be >= 0 and sample_count must be >= 1.")
    with path.open("rb") as f:
        f.seek(start_sample * complex_sample_bytes(dtype))
        if dtype == "complex64":
            return np.fromfile(f, dtype=np.complex64, count=sample_count)
        if dtype not in {"float32", "int16"}:
            raise ValueError(f"Unsupported IQ dtype: {dtype}")
        scalar_dtype = np.float32 if dtype == "float32" else np.int16
        raw = np.fromfile(f, dtype=scalar_dtype, count=sample_count * 2)
    if raw.size < 2:
        return np.array([], dtype=np.complex64)
    if raw.size % 2:
        raw = raw[:-1]
    if dtype == "int16":
        raw = raw.astype(np.float32) / 32768.0
    return (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)).astype(np.complex64)


def iter_iq_windows(
    path: str | Path,
    metadata: IQMetadata,
    window_samples: int,
    hop_samples: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield sliding IQ windows as ``(start_sample, iq_window)``."""

    path = Path(path)
    if window_samples < 1 or hop_samples < 1:
        raise ValueError("window_samples and hop_samples must be >= 1.")
    total_samples = path.stat().st_size // complex_sample_bytes(metadata.dtype)
    if total_samples < window_samples:
        return
    for start_sample in range(0, total_samples - window_samples + 1, hop_samples):
        yield start_sample, read_iq_segment(path, metadata.dtype, start_sample, window_samples)


def write_iq(path: str | Path, iq: np.ndarray, dtype: str = "float32") -> None:
    """Write complex samples as raw IQ."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iq = np.asarray(iq, dtype=np.complex64)
    if dtype == "complex64":
        iq.tofile(path)
        return
    if dtype not in {"float32", "int16"}:
        raise ValueError(f"Unsupported IQ dtype: {dtype}")

    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real
    interleaved[1::2] = iq.imag
    if dtype == "int16":
        interleaved = np.clip(interleaved, -1.0, 1.0)
        interleaved = (interleaved * 32767.0).astype(np.int16)
    interleaved.tofile(path)


def load_iq_record(path: str | Path, metadata: IQMetadata | None = None) -> tuple[np.ndarray, IQMetadata]:
    """Read IQ and return it with explicit metadata."""

    metadata = metadata or IQMetadata()
    return read_iq(path, metadata.dtype), metadata
