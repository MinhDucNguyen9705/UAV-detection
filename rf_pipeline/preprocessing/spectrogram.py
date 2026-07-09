from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import stft, windows


@dataclass(slots=True)
class SpectrogramConfig:
    sample_rate_hz: float = 100e6
    center_frequency_hz: float = 2.4e9
    stft_point: int = 16384
    overlap_ratio: float = 0.5
    boundary: str | None = "zeros"
    padded: bool = True
    dynamic_range_db: float = 70.0
    colormap: str = "hot"
    render_mode: str = "matplotlib"
    image_width: int | None = None
    image_height: int | None = None


@dataclass(slots=True)
class SpectrogramFrame:
    image: np.ndarray
    power_db: np.ndarray
    time_axis_sec: np.ndarray
    frequency_axis_hz: np.ndarray
    duration_sec: float
    freq_min_hz: float
    freq_max_hz: float


def iq_to_spectrogram(iq: np.ndarray, config: SpectrogramConfig) -> SpectrogramFrame:
    """Convert complex IQ to a BGR uint8 spectrogram image."""

    if iq.size < config.stft_point:
        raise ValueError(f"IQ length ({iq.size}) must be >= stft_point ({config.stft_point}).")
    overlap = int(config.stft_point * config.overlap_ratio)
    overlap = min(max(overlap, 0), config.stft_point - 1)
    iq = np.asarray(iq, dtype=np.complex64)
    window = windows.hamming(config.stft_point).astype(np.float32)
    freq, time, zxx = stft(
        iq,
        fs=config.sample_rate_hz,
        window=window,
        nperseg=config.stft_point,
        noverlap=overlap,
        return_onesided=False,
        boundary=config.boundary,
        padded=config.padded,
    )
    freq = np.fft.fftshift(freq)
    zxx = np.fft.fftshift(zxx, axes=0)
    if config.render_mode == "matplotlib":
        power_db = (10.0 * np.log10(np.abs(zxx).astype(np.float32, copy=False) + 1e-12)).astype(np.float32)
        image = _render_matplotlib(power_db, freq, time, config)
    elif config.render_mode == "opencv":
        power_db = (20.0 * np.log10(np.abs(zxx).astype(np.float32, copy=False) + 1e-12)).astype(np.float32)
        high = float(np.percentile(power_db, 99.5))
        low = high - config.dynamic_range_db
        normalized = np.clip((power_db - low) / max(high - low, 1e-6), 0.0, 1.0)
        gray = np.flipud((normalized * 255.0).astype(np.uint8))
        image = _apply_colormap(gray, config.colormap)
    else:
        raise ValueError("render_mode must be 'matplotlib' or 'opencv'.")
    image = _resize_if_requested(image, config.image_width, config.image_height)
    absolute_freq = freq + config.center_frequency_hz
    return SpectrogramFrame(
        image=image,
        power_db=power_db,
        time_axis_sec=time,
        frequency_axis_hz=absolute_freq,
        duration_sec=float(iq.size / config.sample_rate_hz),
        freq_min_hz=float(absolute_freq.min()),
        freq_max_hz=float(absolute_freq.max()),
    )


def save_spectrogram(frame: SpectrogramFrame, path: str | Path) -> tuple[int, int]:
    """Save spectrogram image and return width, height."""

    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame.image):
        raise RuntimeError(f"Failed to write spectrogram image: {path}")
    height, width = frame.image.shape[:2]
    return width, height


def _apply_colormap(gray: np.ndarray, colormap: str) -> np.ndarray:
    import cv2

    maps = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "gray": None,
    }
    key = colormap.lower()
    if key not in maps:
        raise ValueError(f"Unsupported colormap {colormap!r}; choose one of {sorted(maps)}.")
    if maps[key] is None:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gray, maps[key])


def _render_matplotlib(power_db: np.ndarray, freq: np.ndarray, time: np.ndarray, config: SpectrogramConfig) -> np.ndarray:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [time.min(), time.max(), freq.min(), freq.max()]
    fig = plt.figure(figsize=(6.4, 4.8), dpi=300)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(power_db, extent=extent, aspect="auto", origin="lower", cmap=config.colormap)
    ax.axis("off")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    rgb = rgba[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _resize_if_requested(image: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    import cv2

    target_width = width or image.shape[1]
    target_height = height or image.shape[0]
    if target_width == image.shape[1] and target_height == image.shape[0]:
        return image
    interpolation = cv2.INTER_AREA if target_width < image.shape[1] or target_height < image.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)
