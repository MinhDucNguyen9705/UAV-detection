from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .spectrogram import SpectrogramConfig, iq_to_spectrogram


@dataclass(slots=True)
class WaterfallConfig:
    window_samples: int
    hop_samples: int
    fps: float = 12.0
    codec: str = "mp4v"


class BrowserVideoWriter:
    """Streaming BGR frame writer that finalizes to browser-friendly MP4."""

    def __init__(self, path: str | Path, fps: float = 12.0, codec: str = "mp4v") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.codec = codec
        self.ffmpeg = shutil.which("ffmpeg")
        self.write_path = (
            self.path.with_name(f"{self.path.stem}.opencv{self.path.suffix}")
            if self.ffmpeg and self.path.suffix.lower() == ".mp4"
            else self.path
        )
        self.writer = None
        self.frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        import cv2

        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(str(self.write_path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.write_path}")
        self.writer.write(frame)
        self.frame_count += 1

    def close(self) -> int:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.frame_count > 0 and self.write_path != self.path:
            _transcode_browser_mp4(self.write_path, self.path, self.fps)
        return self.frame_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def save_waterfall_video(
    iq: np.ndarray,
    path: str | Path,
    spectrogram_config: SpectrogramConfig,
    waterfall_config: WaterfallConfig,
) -> int:
    """Save a sliding-window waterfall video and return the number of frames."""

    import cv2

    if waterfall_config.window_samples < spectrogram_config.stft_point:
        raise ValueError("window_samples must be >= stft_point.")
    if waterfall_config.hop_samples < 1:
        raise ValueError("hop_samples must be >= 1.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    starts = range(0, max(1, iq.size - waterfall_config.window_samples + 1), waterfall_config.hop_samples)
    with BrowserVideoWriter(path, fps=waterfall_config.fps, codec=waterfall_config.codec) as writer:
        for start in starts:
            segment = iq[start : start + waterfall_config.window_samples]
            if segment.size < waterfall_config.window_samples:
                break
            frame = iq_to_spectrogram(segment, spectrogram_config).image
            writer.write(frame)
        return writer.frame_count


def waterfall_frames(
    iq: np.ndarray,
    spectrogram_config: SpectrogramConfig,
    waterfall_config: WaterfallConfig,
) -> Iterable[np.ndarray]:
    """Yield rendered waterfall frames as BGR images."""

    if waterfall_config.window_samples < spectrogram_config.stft_point:
        raise ValueError("window_samples must be >= stft_point.")
    if waterfall_config.hop_samples < 1:
        raise ValueError("hop_samples must be >= 1.")
    starts = range(0, max(1, iq.size - waterfall_config.window_samples + 1), waterfall_config.hop_samples)
    for start in starts:
        segment = iq[start : start + waterfall_config.window_samples]
        if segment.size < waterfall_config.window_samples:
            break
        yield iq_to_spectrogram(segment, spectrogram_config).image


def save_frames_video(frames: Iterable[np.ndarray], path: str | Path, fps: float = 12.0, codec: str = "mp4v") -> int:
    """Save BGR frames to browser-friendly video and return frame count."""

    import cv2

    with BrowserVideoWriter(path, fps=fps, codec=codec) as writer:
        for frame in frames:
            writer.write(frame)
        return writer.frame_count


def _transcode_browser_mp4(source: Path, target: Path, fps: float) -> None:
    """Transcode OpenCV output to browser-friendly H.264 MP4."""

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-r",
        f"{fps:g}",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(target),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        source.unlink(missing_ok=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        if not target.exists() and source.exists():
            source.replace(target)
        print(f"Warning: FFmpeg MP4 transcode failed; kept OpenCV video. {exc}")
