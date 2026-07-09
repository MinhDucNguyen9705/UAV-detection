from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ParameterEstimate:
    time_start_sec: float
    time_end_sec: float
    duration_sec: float
    freq_low_hz: float
    freq_high_hz: float
    center_frequency_hz: float
    bandwidth_hz: float


def estimate_parameters(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    segment_start_sec: float,
    segment_end_sec: float,
    freq_min_hz: float,
    freq_max_hz: float,
) -> ParameterEstimate:
    """Convert image-space xyxy box to RF time/frequency estimates."""

    x1, y1, x2, y2 = xyxy
    x1 = min(max(x1, 0.0), float(image_width))
    x2 = min(max(x2, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    time_span = segment_end_sec - segment_start_sec
    freq_span = freq_max_hz - freq_min_hz
    time_start = segment_start_sec + (x1 / image_width) * time_span
    time_end = segment_start_sec + (x2 / image_width) * time_span

    # Image y=0 is top, while lower RF frequency is rendered at the bottom.
    freq_high = freq_max_hz - (y1 / image_height) * freq_span
    freq_low = freq_max_hz - (y2 / image_height) * freq_span
    return ParameterEstimate(
        time_start_sec=time_start,
        time_end_sec=time_end,
        duration_sec=max(0.0, time_end - time_start),
        freq_low_hz=freq_low,
        freq_high_hz=freq_high,
        center_frequency_hz=(freq_low + freq_high) / 2.0,
        bandwidth_hz=max(0.0, freq_high - freq_low),
    )
