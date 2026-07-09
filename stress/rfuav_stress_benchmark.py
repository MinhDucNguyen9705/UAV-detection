from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.signal import stft, windows

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stress.utils import read_csv


@dataclass(slots=True)
class BenchmarkRow:
    sample_id: str
    scenario: str
    split: str
    iq_path: str
    spectrogram_path: str
    metadata_path: str
    source_raw_paths: str
    class_names: str
    sample_rate_hz: float
    center_frequency_hz: float
    stft_point: int
    duration_sec: float
    num_sources: int
    snr_db: str
    sir_db: str
    time_shift_sec: str
    freq_shift_hz: str
    image_width: int
    image_height: int
    freq_min_hz: float
    freq_max_hz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create local RFUAV stress benchmark IQ/spectrogram samples.")
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="*", default=["clean_single", "low_snr", "mix2", "near_far", "noise_only"], choices=["clean_single", "low_snr", "mix2", "near_far", "noise_only"])
    parser.add_argument("--num-clean", type=int, default=200)
    parser.add_argument("--num-low-snr-per-level", type=int, default=100)
    parser.add_argument("--num-mix2", type=int, default=300)
    parser.add_argument("--num-near-far", type=int, default=200)
    parser.add_argument("--num-noise-only", type=int, default=100)
    parser.add_argument("--snr-levels", type=float, nargs="*", default=[-10, -5, 0, 5, 10])
    parser.add_argument("--mix2-sir-levels", type=float, nargs="*", default=[-10, 0, 10])
    parser.add_argument("--near-far-sir-levels", type=float, nargs="*", default=[10, 20, 30])
    parser.add_argument("--max-time-shift-ms", type=float, default=20.0)
    parser.add_argument("--max-freq-shift-mhz", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.03)
    parser.add_argument("--stft-point", type=int, default=16384)
    parser.add_argument("--default-sample-rate", type=float, default=100e6)
    parser.add_argument("--default-center-frequency", type=float, default=2.4e9)
    parser.add_argument("--render-spectrogram", action="store_true")
    parser.add_argument("--colormap", default="hot")
    parser.add_argument("--image-format", default="png", choices=["png", "jpg"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sources", type=int, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def float_field(row: dict[str, str], name: str, default: float) -> float:
    value = row.get(name, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_sources(path: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = []
    for row in read_csv(path):
        raw_path = Path(row.get("raw_path", ""))
        if raw_path.is_file():
            rows.append(row)
    if args.max_sources:
        rows = rows[: args.max_sources]
    if not rows:
        raise RuntimeError("No usable raw_path rows found in raw manifest.")
    return rows


def read_iq(path: Path, dtype_name: str) -> np.ndarray:
    dtype = np.float32 if dtype_name == "float32" else np.int16
    raw = np.fromfile(path, dtype=dtype)
    if raw.size % 2:
        raw = raw[:-1]
    if dtype_name == "int16":
        raw = raw.astype(np.float32) / 32768.0
    return (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)).astype(np.complex64)


def write_iq(path: Path, iq: np.ndarray) -> None:
    ensure_dir(path.parent)
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    interleaved.tofile(path)


def sample_segment(iq: np.ndarray, segment_len: int, rng: random.Random) -> tuple[np.ndarray, int]:
    if iq.size < segment_len:
        raise ValueError("IQ source is shorter than requested segment length.")
    start = rng.randint(0, iq.size - segment_len)
    return iq[start : start + segment_len].copy(), start


def power(iq: np.ndarray) -> float:
    return float(np.mean(np.abs(iq) ** 2) + 1e-12)


def add_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise_power = power(iq) / (10 ** (snr_db / 10.0))
    noise = (rng.normal(size=iq.size).astype(np.float32) + 1j * rng.normal(size=iq.size).astype(np.float32)) * np.sqrt(noise_power / 2.0)
    return (iq + noise).astype(np.complex64)


def scale_to_sir(reference: np.ndarray, interferer: np.ndarray, sir_db: float) -> np.ndarray:
    scale = np.sqrt((power(reference) / (10 ** (sir_db / 10.0))) / power(interferer))
    return (interferer * scale).astype(np.complex64)


def time_shift(iq: np.ndarray, shift_samples: int) -> np.ndarray:
    if shift_samples == 0:
        return iq.copy()
    out = np.zeros_like(iq)
    if shift_samples > 0:
        out[shift_samples:] = iq[:-shift_samples]
    else:
        out[:shift_samples] = iq[-shift_samples:]
    return out


def freq_shift(iq: np.ndarray, shift_hz: float, sample_rate: float) -> np.ndarray:
    if shift_hz == 0:
        return iq.copy()
    n = np.arange(iq.size, dtype=np.float32)
    rot = np.exp(1j * 2.0 * np.pi * shift_hz * n / sample_rate).astype(np.complex64)
    return (iq * rot).astype(np.complex64)


def render_spectrogram(iq: np.ndarray, image_path: Path, sample_rate: float, stft_point: int, cmap: str) -> tuple[int, int]:
    freq, time, zxx = stft(iq, sample_rate, return_onesided=False, window=windows.hamming(stft_point), nperseg=stft_point)
    freq = np.fft.fftshift(freq)
    zxx = np.fft.fftshift(zxx, axes=0)
    power_db = 10 * np.log10(np.abs(zxx) + 1e-12)
    ensure_dir(image_path.parent)
    plt.figure()
    plt.imshow(power_db, extent=[time.min(), time.max(), freq.min(), freq.max()], aspect="auto", origin="lower", cmap=cmap)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
    plt.savefig(str(image_path), dpi=300)
    plt.close()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to render spectrogram: {image_path}")
    h, w = image.shape[:2]
    return w, h


def source_params(source: dict[str, str], args: argparse.Namespace) -> tuple[float, float, int]:
    sample_rate = float_field(source, "sample_rate_hz", args.default_sample_rate)
    center = float_field(source, "center_frequency_hz", args.default_center_frequency)
    return sample_rate, center, int(sample_rate * args.duration)


def make_paths(out_dir: Path, scenario: str, sample_id: str, image_format: str) -> tuple[Path, Path, Path]:
    return (
        out_dir / "raw_iq" / scenario / f"{sample_id}.iq",
        out_dir / "metadata" / scenario / f"{sample_id}.json",
        out_dir / "spectrograms" / "images" / scenario / f"{sample_id}.{image_format}",
    )


def save_sample(iq: np.ndarray, sample_id: str, scenario: str, sources: list[dict[str, str]], metadata: dict, args: argparse.Namespace, sample_rate: float, center: float, num_sources: int, snr_db: str = "", sir_db: str = "", time_shift_sec: str = "", freq_shift_hz: str = "") -> BenchmarkRow:
    iq_path, meta_path, img_path = make_paths(args.out_dir, scenario, sample_id, args.image_format)
    write_iq(iq_path, iq)
    ensure_dir(meta_path.parent)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    image_width = image_height = 0
    image_path = ""
    if args.render_spectrogram:
        image_width, image_height = render_spectrogram(iq, img_path, sample_rate, args.stft_point, args.colormap)
        image_path = str(img_path)
    return BenchmarkRow(
        sample_id=sample_id,
        scenario=scenario,
        split=sources[0].get("split") or "train",
        iq_path=str(iq_path),
        spectrogram_path=image_path,
        metadata_path=str(meta_path),
        source_raw_paths=";".join(source.get("raw_path", "") for source in sources),
        class_names=";".join(source.get("class_name", "") for source in sources),
        sample_rate_hz=sample_rate,
        center_frequency_hz=center,
        stft_point=args.stft_point,
        duration_sec=args.duration,
        num_sources=num_sources,
        snr_db=snr_db,
        sir_db=sir_db,
        time_shift_sec=time_shift_sec,
        freq_shift_hz=freq_shift_hz,
        image_width=image_width,
        image_height=image_height,
        freq_min_hz=center - sample_rate / 2,
        freq_max_hz=center + sample_rate / 2,
    )


def choose_pair(sources: list[dict[str, str]], rng: random.Random) -> tuple[dict[str, str], dict[str, str]]:
    return tuple(rng.sample(sources, 2)) if len(sources) > 1 else (sources[0], sources[0])


def write_manifest(path: Path, rows: list[BenchmarkRow]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchmarkRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def generate(args: argparse.Namespace) -> list[BenchmarkRow]:
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    sources = load_sources(args.raw_manifest, args)
    rows: list[BenchmarkRow] = []
    selected = set(args.scenarios)

    def source_iq(source: dict[str, str]) -> np.ndarray:
        dtype = source.get("effective_dtype") or source.get("dtype") or "float32"
        return read_iq(Path(source["raw_path"]), dtype)

    if "clean_single" in selected:
        for index in range(args.num_clean):
            source = rng.choice(sources)
            sr, cf, segment_len = source_params(source, args)
            segment, start = sample_segment(source_iq(source), segment_len, rng)
            rows.append(save_sample(segment, f"clean_single_{index:06d}", "clean_single", [source], {"scenario": "clean_single", "source_start_sample": start, "source": source}, args, sr, cf, 1))

    if "low_snr" in selected:
        idx = 0
        for snr in args.snr_levels:
            for _ in range(args.num_low_snr_per_level):
                source = rng.choice(sources)
                sr, cf, segment_len = source_params(source, args)
                segment, start = sample_segment(source_iq(source), segment_len, rng)
                noisy = add_awgn(segment, snr, np_rng)
                rows.append(save_sample(noisy, f"low_snr_{snr:g}db_{idx:06d}".replace("-", "m"), "low_snr", [source], {"scenario": "low_snr", "source_start_sample": start, "snr_db": snr, "source": source}, args, sr, cf, 1, snr_db=str(snr)))
                idx += 1

    for scenario, count, levels in [("mix2", args.num_mix2, args.mix2_sir_levels), ("near_far", args.num_near_far, args.near_far_sir_levels)]:
        if scenario not in selected:
            continue
        for index in range(count):
            first, second = choose_pair(sources, rng)
            sr, cf, segment_len = source_params(first, args)
            x1, start1 = sample_segment(source_iq(first), segment_len, rng)
            x2, start2 = sample_segment(source_iq(second), segment_len, rng)
            sir = rng.choice(levels)
            shift_sec = rng.uniform(0, args.max_time_shift_ms / 1000.0)
            shift_hz = rng.uniform(-args.max_freq_shift_mhz, args.max_freq_shift_mhz) * 1e6
            second_mod = freq_shift(time_shift(scale_to_sir(x1, x2, sir), int(shift_sec * sr)), shift_hz, sr)
            mixed = (x1 + second_mod).astype(np.complex64)
            rows.append(save_sample(mixed, f"{scenario}_{index:06d}", scenario, [first, second], {"scenario": scenario, "source_start_samples": [start1, start2], "sir_db": sir, "time_shift_sec": shift_sec, "freq_shift_hz": shift_hz, "sources": [first, second]}, args, sr, cf, 2, sir_db=str(sir), time_shift_sec=f"{shift_sec:.9f}", freq_shift_hz=f"{shift_hz:.3f}"))

    if "noise_only" in selected:
        for index in range(args.num_noise_only):
            source = rng.choice(sources)
            sr, cf, segment_len = source_params(source, args)
            noise = (np_rng.normal(size=segment_len).astype(np.float32) + 1j * np_rng.normal(size=segment_len).astype(np.float32)) * np.sqrt(0.5)
            rows.append(save_sample(noise.astype(np.complex64), f"noise_only_{index:06d}", "noise_only", [source], {"scenario": "noise_only", "num_sources": 0, "reference_source": source}, args, sr, cf, 0))
    return rows


def main() -> int:
    args = parse_args()
    rows = generate(args)
    write_manifest(args.out_dir / "manifests" / "benchmark_manifest.csv", rows)
    summary = {"num_samples": len(rows), "by_scenario": {}}
    for row in rows:
        summary["by_scenario"][row.scenario] = summary["by_scenario"].get(row.scenario, 0) + 1
    ensure_dir(args.out_dir / "manifests")
    (args.out_dir / "manifests" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Generated {len(rows)} samples in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
