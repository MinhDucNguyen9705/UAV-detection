"""Download RFUAV archives from Hugging Face and build spectrogram manifests.

The Hugging Face RFUAV dataset stores large per-drone archives. Archives may
contain nested folders with raw IQ files and XML metadata. This script:

1. Lists and downloads selected archives from a Hugging Face dataset repo.
2. Extracts .tar/.tar.gz/.tgz/.zip/.rar archives.
3. Finds .iq files and nearby .xml metadata.
4. Generates spectrogram PNG files from raw interleaved IQ.
5. Writes raw and sample manifests for reproducible training splits.

RAR support requires the optional Python package `rarfile` and an external
unrar/bsdtar/7z executable available on PATH.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from dataset.manifest import (
    RawManifestRow,
    SampleManifestRow,
    assign_splits,
    copy_split_index,
    read_dataclass_csv,
    write_csv,
    write_summary,
)


ARCHIVE_EXTS = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".zip",
    ".rar",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RFUAV spectrograms from Hugging Face archives.")
    parser.add_argument("--repo-id", default="kitofrank/RFUAV", help="Hugging Face dataset repo id.")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None, help="HF token if the repo is private. Falls back to HF_TOKEN env var.")
    parser.add_argument("--prompt-token", action="store_true", help="Prompt for a Hugging Face token before downloading.")
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--include",
        nargs="*",
        default=["*.tar", "*.tar.gz", "*.tgz", "*.zip", "*.rar"],
        help="fnmatch patterns for archive paths in the HF repo.",
    )
    parser.add_argument("--archive-name", nargs="*", default=None, help="Specific archive basenames to process.")
    parser.add_argument("--max-archives", type=int, default=None)
    parser.add_argument(
        "--max-archive-size-gb",
        type=float,
        default=None,
        help="Skip Hugging Face archives larger than this size before downloading.",
    )
    parser.add_argument("--raw-exts", default=".iq,.dat,.bin")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "int16"])
    parser.add_argument("--default-sample-rate", type=float, default=100e6)
    parser.add_argument("--default-center-frequency", type=float, default=2.4e9)
    parser.add_argument("--stft-point", type=int, default=1024)
    parser.add_argument("--duration", type=float, default=0.1)
    parser.add_argument("--hop-ratio", type=float, default=1.0)
    parser.add_argument("--dynamic-range-db", type=float, default=70.0)
    parser.add_argument(
        "--render-mode",
        default="matplotlib",
        choices=["matplotlib", "opencv"],
        help="matplotlib matches graphic/RawDataProcessor.py; opencv saves the native STFT array.",
    )
    parser.add_argument("--colormap", default="hot", help="Matplotlib colormap for spectrogram rendering.")
    parser.add_argument("--image-format", default="jpg", choices=["jpg", "png"], help="Saved spectrogram image format.")
    parser.add_argument(
        "--image-width",
        type=int,
        default=0,
        help="Resize saved spectrogram image to this width. 0 keeps native STFT frame count.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=0,
        help="Resize saved spectrogram image to this height. 0 keeps native STFT frequency bins.",
    )
    parser.add_argument("--max-raw-files", type=int, default=None)
    parser.add_argument("--max-segments-per-raw", type=int, default=None)
    parser.add_argument("--split-key", default="raw_file_id", choices=["raw_file_id", "session_id", "class_name"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true", help="Use archives already present in --download-dir.")
    parser.add_argument("--skip-extract", action="store_true", help="Use files already extracted in --extract-dir.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint/manifest and skip processed archives.")
    parser.add_argument(
        "--delete-archive-after-extract",
        action="store_true",
        help="Delete each downloaded/local archive only after it has been extracted successfully.",
    )
    parser.add_argument(
        "--delete-extracted-after-convert",
        action="store_true",
        help="Delete the extracted folder for each archive after its spectrograms have been generated.",
    )
    parser.add_argument(
        "--delete-iq-after-convert",
        action="store_true",
        help="Delete each raw IQ file immediately after its spectrogram segments have been generated.",
    )
    parser.add_argument(
        "--stream-extract-convert",
        action="store_true",
        help="Extract one raw IQ file at a time, convert it, then optionally delete it before extracting the next IQ file.",
    )
    parser.add_argument("--overwrite-images", action="store_true")
    args = parser.parse_args()

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        parser.error("Split ratios must sum to 1.")
    return args


def resolve_hf_token(args: argparse.Namespace) -> None:
    if args.token:
        return
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        args.token = env_token
        print("Using Hugging Face token from environment variable.")
        return
    if args.skip_download and not args.prompt_token:
        return
    if args.prompt_token:
        try:
            token = getpass.getpass("Hugging Face token/API key (press Enter for public access): ").strip()
        except (EOFError, KeyboardInterrupt):
            token = ""
        args.token = token or None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_id(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def has_archive_ext(path: str | Path) -> bool:
    name = str(path).lower()
    return any(name.endswith(ext) for ext in ARCHIVE_EXTS)


def max_archive_size_bytes(args: argparse.Namespace) -> int | None:
    limit_gb = getattr(args, "max_archive_size_gb", None)
    if limit_gb is None:
        return None
    return int(limit_gb * 1024**3)


def format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    return f"{value:.2f} {unit}"


def hf_repo_file_sizes(args: argparse.Namespace) -> dict[str, int]:
    from huggingface_hub import HfApi

    api = HfApi(token=getattr(args, "token", None))
    info = api.repo_info(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        files_metadata=True,
    )
    sizes: dict[str, int] = {}
    for sibling in getattr(info, "siblings", []) or []:
        name = getattr(sibling, "rfilename", None)
        if not name:
            continue
        size = getattr(sibling, "size", None)
        lfs = getattr(sibling, "lfs", None)
        if size is None and isinstance(lfs, dict):
            size = lfs.get("size")
        if size is not None:
            sizes[name] = int(size)
    return sizes


def hf_archive_size(args: argparse.Namespace, repo_path: str) -> int | None:
    try:
        return hf_repo_file_sizes(args).get(repo_path)
    except Exception as exc:
        print(f"Warning: could not read Hugging Face file size metadata: {exc}")
        return None


def archive_stem(path: Path) -> str:
    name = path.name
    for ext in sorted(ARCHIVE_EXTS, key=len, reverse=True):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return path.stem


def list_hf_archives(args: argparse.Namespace) -> list[str]:
    from fnmatch import fnmatch

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    files = api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type, revision=args.revision)
    archives = [f for f in files if has_archive_ext(f) and any(fnmatch(f, p) for p in args.include)]
    if args.archive_name:
        wanted = set(args.archive_name)
        archives = [f for f in archives if Path(f).name in wanted or f in wanted]
    archives = sorted(archives)
    size_limit = max_archive_size_bytes(args)
    if size_limit is not None:
        try:
            sizes = hf_repo_file_sizes(args)
        except Exception as exc:
            print(f"Warning: could not read Hugging Face file size metadata; size filter disabled: {exc}")
            sizes = {}
        filtered_archives = []
        for repo_path in archives:
            size = sizes.get(repo_path)
            if size is not None and size > size_limit:
                print(
                    "Skip archive over size limit: "
                    f"{repo_path} ({format_bytes(size)} > {format_bytes(size_limit)})"
                )
                continue
            if size is None:
                print(f"Warning: size metadata unavailable for {repo_path}; keeping it.")
            filtered_archives.append(repo_path)
        archives = filtered_archives
    if args.max_archives:
        archives = archives[: args.max_archives]
    return archives


def download_archives(args: argparse.Namespace) -> list[Path]:
    ensure_dir(args.download_dir)
    if args.skip_download:
        archives = [p for p in args.download_dir.rglob("*") if p.is_file() and has_archive_ext(p)]
        if args.archive_name:
            wanted = set(args.archive_name)
            archives = [p for p in archives if p.name in wanted or str(p) in wanted]
        return sorted(archives)

    from huggingface_hub import hf_hub_download

    archive_paths = []
    archive_names = list_hf_archives(args)
    if not archive_names:
        raise RuntimeError("No matching archives found in the Hugging Face repo.")

    for repo_path in archive_names:
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            filename=repo_path,
            token=args.token,
            local_dir=args.download_dir,
        )
        archive_paths.append(Path(local_path))
        print(f"Downloaded: {repo_path}")
    return archive_paths


def local_archives(args: argparse.Namespace) -> list[Path]:
    archives = [p for p in args.download_dir.rglob("*") if p.is_file() and has_archive_ext(p)]
    if args.archive_name:
        wanted = set(args.archive_name)
        archives = [p for p in archives if p.name in wanted or str(p) in wanted]
    if args.max_archives:
        archives = archives[: args.max_archives]
    return sorted(archives)


def download_one_archive(args: argparse.Namespace, repo_path: str) -> Path:
    from huggingface_hub import hf_hub_download

    ensure_dir(args.download_dir)
    size_limit = max_archive_size_bytes(args)
    if size_limit is not None:
        size = hf_archive_size(args, repo_path)
        if size is not None and size > size_limit:
            raise RuntimeError(
                "Archive exceeds --max-archive-size-gb: "
                f"{repo_path} ({format_bytes(size)} > {format_bytes(size_limit)})"
            )
    local_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        filename=repo_path,
        token=args.token,
        local_dir=args.download_dir,
    )
    print(f"Downloaded: {repo_path}")
    return Path(local_path)


def safe_extract_tar(archive: Path, target_dir: Path) -> None:
    with tarfile.open(archive) as tf:
        target_root = target_dir.resolve()
        for member in tf.getmembers():
            member_path = (target_dir / member.name).resolve()
            if not str(member_path).startswith(str(target_root)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        try:
            tf.extractall(target_dir, filter="data")
        except TypeError:
            tf.extractall(target_dir)


def extract_archive(archive: Path, extract_root: Path, skip_extract: bool, delete_after_extract: bool) -> Path:
    target_dir = extract_root / archive_stem(archive)
    done_flag = target_dir / ".extract_done"
    if skip_extract or done_flag.exists():
        return target_dir

    ensure_dir(target_dir)
    lower = archive.name.lower()
    print(f"Extracting: {archive.name}")
    if lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        safe_extract_tar(archive, target_dir)
    elif lower.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
    elif lower.endswith(".rar"):
        try:
            import rarfile
        except ImportError as exc:
            raise RuntimeError("Install `rarfile` and an unrar/bsdtar/7z executable to extract .rar files.") from exc
        with rarfile.RarFile(archive) as rf:
            rf.extractall(target_dir)
    else:
        raise RuntimeError(f"Unsupported archive type: {archive}")

    done_flag.write_text("ok\n", encoding="utf-8")
    if delete_after_extract and archive.exists():
        archive.unlink()
        print(f"Deleted archive after successful extraction: {archive}")
    return target_dir


def parse_number(value: str | None, default: float) -> float:
    if value is None:
        return default
    text = value.strip().replace(",", "")
    if not text:
        return default
    multipliers = {
        "ghz": 1e9,
        "mhz": 1e6,
        "khz": 1e3,
        "hz": 1.0,
        "g": 1e9,
        "m": 1e6,
        "k": 1e3,
    }
    lower = text.lower().replace(" ", "")
    for suffix, mult in multipliers.items():
        if lower.endswith(suffix):
            return float(lower[: -len(suffix)]) * mult
    return float(lower)


def infer_iq_dtype(meta: dict[str, str], cli_dtype: str) -> str:
    if cli_dtype != "auto":
        return cli_dtype

    data_type = xml_value(meta, "DataType", "datatype").lower()
    if "int16" in data_type or "short" in data_type:
        return "int16"
    if "float" in data_type or "complex float" in data_type:
        return "float32"
    return "float32"


def parse_iq_time_range_from_name(path: Path) -> tuple[float, float]:
    """Parse names like pack2_0-1s.iq or pack2_9-10s.iq."""
    match = re.search(r"_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)s$", path.stem, flags=re.IGNORECASE)
    if not match:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2))


def xml_matches_raw_prefix(xml_path: Path, raw_path: Path) -> bool:
    xml_stem = xml_path.stem.lower()
    raw_stem = raw_path.stem.lower()
    return raw_stem == xml_stem or raw_stem.startswith(xml_stem + "_") or raw_stem.startswith(xml_stem + "-")


def xml_member_matches_raw_prefix(xml_name: str, raw_name: str) -> bool:
    xml_stem = Path(norm_archive_name(xml_name)).stem.lower()
    raw_stem = Path(norm_archive_name(raw_name)).stem.lower()
    return raw_stem == xml_stem or raw_stem.startswith(xml_stem + "_") or raw_stem.startswith(xml_stem + "-")


def parse_xml(xml_path: Path | None) -> dict[str, str]:
    if xml_path is None or not xml_path.exists():
        return {}
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return {}

    values: dict[str, str] = {}
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].strip()
        text = (elem.text or "").strip()
        if tag and text:
            values[tag.lower()] = text
    return values


def parse_xml_text(xml_text: str) -> dict[str, str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}

    values: dict[str, str] = {}
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].strip()
        text = (elem.text or "").strip()
        if tag and text:
            values[tag.lower()] = text
    return values


def xml_value(meta: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = meta.get(name.lower())
        if value:
            return value
    return default


def find_nearest_xml(raw_path: Path, extracted_root: Path) -> Path | None:
    nearby_xmls = sorted(raw_path.parent.glob("*.xml"))
    for xml_path in nearby_xmls:
        meta = parse_xml(xml_path)
        iq_file_name = xml_value(meta, "IQFileName", "iq_file_name")
        if iq_file_name and Path(iq_file_name).name.lower() == raw_path.name.lower():
            return xml_path

    same_stem = list(raw_path.parent.glob(raw_path.stem + ".xml"))
    if same_stem:
        return same_stem[0]
    for xml_path in nearby_xmls:
        if xml_matches_raw_prefix(xml_path, raw_path):
            return xml_path
    if nearby_xmls:
        return nearby_xmls[0]
    ancestors = [raw_path.parent, *raw_path.parents]
    for parent in ancestors:
        if extracted_root not in parent.parents and parent != extracted_root:
            continue
        xmls = sorted(parent.glob("*.xml"))
        if xmls:
            return xmls[0]
        if parent == extracted_root:
            break
    all_xml = sorted(extracted_root.rglob("*.xml"))
    return all_xml[0] if len(all_xml) == 1 else None


def read_iq(path: Path, dtype_name: str) -> np.ndarray:
    import numpy as np

    dtype = np.float32 if dtype_name == "float32" else np.int16
    raw = np.fromfile(path, dtype=dtype)
    if raw.size < 2:
        return np.array([], dtype=np.complex64)
    if raw.size % 2:
        raw = raw[:-1]
    if dtype_name == "int16":
        raw = raw.astype(np.float32) / 32768.0
    return (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)).astype(np.complex64)


def spectrogram_uint8(iq: np.ndarray, sample_rate: float, stft_point: int, dynamic_range_db: float) -> np.ndarray:
    import cv2
    import numpy as np
    from scipy.signal import stft, windows

    _, _, zxx = stft(
        iq,
        fs=sample_rate,
        window=windows.hamming(stft_point),
        nperseg=stft_point,
        noverlap=stft_point // 2,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    zxx = np.fft.fftshift(zxx, axes=0)
    power_db = 20.0 * np.log10(np.abs(zxx) + 1e-12)
    high = np.percentile(power_db, 99.5)
    low = high - dynamic_range_db
    normalized = np.clip((power_db - low) / max(high - low, 1e-6), 0.0, 1.0)
    gray = np.flipud((normalized * 255).astype(np.uint8))
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def stft_for_rfuav_render(
    iq: np.ndarray,
    sample_rate: float,
    stft_point: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np
    from scipy.signal import stft, windows

    f, t, zxx = stft(
        iq,
        sample_rate,
        return_onesided=False,
        window=windows.hamming(stft_point),
        nperseg=stft_point,
    )
    f = np.fft.fftshift(f)
    zxx = np.fft.fftshift(zxx, axes=0)
    aug = 10 * np.log10(np.abs(zxx) + 1e-12)
    return f, t, aug


def save_spectrogram_matplotlib(
    iq: np.ndarray,
    image_path: Path,
    sample_rate: float,
    stft_point: int,
    colormap: str,
) -> tuple[int, int]:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, t, aug = stft_for_rfuav_render(iq, sample_rate, stft_point)
    extent = [t.min(), t.max(), f.min(), f.max()]

    plt.figure()
    plt.imshow(aug, extent=extent, aspect="auto", origin="lower", cmap=colormap)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
    ensure_dir(image_path.parent)
    plt.savefig(str(image_path), dpi=300)
    plt.close()

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read saved spectrogram: {image_path}")
    height, width = image.shape[:2]
    return width, height


def resize_spectrogram_for_output(image: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    target_width = width if width and width > 0 else image.shape[1]
    target_height = height if height and height > 0 else image.shape[0]
    if target_width == image.shape[1] and target_height == image.shape[0]:
        return image
    interpolation = cv2.INTER_AREA if target_width < image.shape[1] or target_height < image.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def save_spectrogram_opencv(
    iq: np.ndarray,
    image_path: Path,
    sample_rate: float,
    stft_point: int,
    dynamic_range_db: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    image = spectrogram_uint8(iq, sample_rate, stft_point, dynamic_range_db)
    image = resize_spectrogram_for_output(image, image_width, image_height)
    ensure_dir(image_path.parent)
    import cv2

    cv2.imwrite(str(image_path), image)
    height, width = image.shape[:2]
    return width, height


def save_spectrogram_image(
    iq: np.ndarray,
    image_path: Path,
    args: argparse.Namespace,
    sample_rate: float,
) -> tuple[int, int]:
    if args.render_mode == "matplotlib":
        return save_spectrogram_matplotlib(iq, image_path, sample_rate, args.stft_point, args.colormap)
    return save_spectrogram_opencv(
        iq=iq,
        image_path=image_path,
        sample_rate=sample_rate,
        stft_point=args.stft_point,
        dynamic_range_db=args.dynamic_range_db,
        image_width=args.image_width,
        image_height=args.image_height,
    )


def discover_raw_files(extracted_dirs: Iterable[Path], raw_exts: set[str]) -> list[tuple[Path, Path]]:
    results = []
    for root in extracted_dirs:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in raw_exts:
                results.append((root, path))
    return results


def clean_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    return cleaned.strip("_") or "unknown"


def metadata_from_xml(meta: dict[str, str], archive_file: str, raw_path: Path, args: argparse.Namespace) -> dict[str, object]:
    drone = xml_value(meta, "Drone", "drone", default=archive_file)
    serial = xml_value(meta, "SerialNumber", "serial_number", "Serial", default=raw_path.stem)
    sample_rate = parse_number(xml_value(meta, "SampleRate", "sample_rate"), args.default_sample_rate)
    center_freq = parse_number(
        xml_value(meta, "CenterFrequency", "MiddleFrequency", "center_frequency"),
        args.default_center_frequency,
    )
    return {
        "drone": drone,
        "serial": serial,
        "data_type": xml_value(meta, "DataType", "datatype", default=args.dtype),
        "iq_file_name": xml_value(meta, "IQFileName", "iq_file_name"),
        "sample_count_xml": xml_value(meta, "SampleCount", "sample_count"),
        "effective_dtype": infer_iq_dtype(meta, args.dtype),
        "sample_rate": sample_rate,
        "center_freq": center_freq,
        "class_name": clean_name(drone or archive_file),
        "session_id": clean_name(serial or raw_path.parent.name),
    }


def process_raw_file(
    args: argparse.Namespace,
    raw_path: Path,
    xml_path: Path | None,
    meta: dict[str, str],
    archive_file: str,
    archive_path: str,
) -> tuple[RawManifestRow, list[SampleManifestRow]]:
    info = metadata_from_xml(meta, archive_file, raw_path, args)
    class_name = str(info["class_name"])
    session_id = str(info["session_id"])
    sample_rate = float(info["sample_rate"])
    center_freq = float(info["center_freq"])
    raw_file_start_sec, raw_file_end_sec = parse_iq_time_range_from_name(raw_path)
    raw_file_id = f"{class_name}_{session_id}_{raw_path.stem}_{stable_id(str(raw_path))}"

    effective_dtype = str(info["effective_dtype"])
    iq = read_iq(raw_path, effective_dtype)
    raw_row = RawManifestRow(
        raw_file_id=raw_file_id,
        archive_file=archive_file,
        archive_path=archive_path,
        raw_path=str(raw_path),
        xml_path=str(xml_path or ""),
        class_name=class_name,
        session_id=session_id,
        device_type=xml_value(meta, "DeviceType", "device_type"),
        drone=str(info["drone"]),
        serial_number=str(info["serial"]),
        data_type=str(info["data_type"]),
        reference_snr_db=xml_value(meta, "ReferenceSNRLevel", "ReferenceSNR", "snr"),
        center_frequency_hz=center_freq,
        sample_rate_hz=sample_rate,
        if_bandwidth_hz=xml_value(meta, "IFBandwidth", "Bandwidth", "if_bandwidth"),
        scale_factor_db=xml_value(meta, "ScaleFactor", "scale_factor"),
        iq_file_name=str(info["iq_file_name"]),
        sample_count_xml=str(info["sample_count_xml"]),
        effective_dtype=effective_dtype,
        raw_file_start_sec=raw_file_start_sec,
        raw_file_end_sec=raw_file_end_sec,
        num_iq_samples=int(iq.size),
    )

    sample_rows: list[SampleManifestRow] = []
    segment_len = int(sample_rate * args.duration)
    hop_len = max(1, int(segment_len * args.hop_ratio))
    if iq.size < segment_len:
        return raw_row, sample_rows
    if segment_len < args.stft_point:
        raise ValueError(
            f"duration is too short for STFT: sample_rate*duration={segment_len} samples "
            f"< stft_point={args.stft_point}. Increase --duration or reduce --stft-point."
        )

    image_root = args.out_dir / "spectrograms" / "images"
    class_dir = image_root / class_name
    ensure_dir(class_dir)
    freq_min = center_freq - sample_rate / 2.0
    freq_max = center_freq + sample_rate / 2.0

    segment_count = 0
    for start in range(0, iq.size - segment_len + 1, hop_len):
        if args.max_segments_per_raw is not None and segment_count >= args.max_segments_per_raw:
            break
        end = start + segment_len
        sample_id = f"{raw_file_id}_seg{segment_count:06d}"
        image_suffix = f".{args.image_format}"
        image_path = class_dir / f"{sample_id}{image_suffix}"

        if args.overwrite_images or not image_path.exists():
            width, height = save_spectrogram_image(iq[start:end], image_path, args, sample_rate)
        else:
            import cv2

            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]

        sample_rows.append(
            SampleManifestRow(
                sample_id=sample_id,
                split="",
                image_path=str(image_path),
                raw_file_id=raw_file_id,
                archive_file=archive_file,
                raw_path=str(raw_path),
                xml_path=str(xml_path or ""),
                class_name=class_name,
                session_id=session_id,
                sample_rate_hz=sample_rate,
                center_frequency_hz=center_freq,
                stft_point=args.stft_point,
                duration_sec=args.duration,
                segment_index=segment_count,
                segment_start_sec=raw_file_start_sec + start / sample_rate,
                segment_end_sec=raw_file_start_sec + end / sample_rate,
                raw_file_start_sec=raw_file_start_sec,
                raw_file_end_sec=raw_file_end_sec,
                image_width=width,
                image_height=height,
                freq_min_hz=freq_min,
                freq_max_hz=freq_max,
            )
        )
        segment_count += 1

    if args.delete_iq_after_convert and segment_count > 0 and raw_path.exists():
        raw_path.unlink()
        print(f"Deleted IQ after conversion: {raw_path}")

    return raw_row, sample_rows


def archive_id(source: str | Path) -> str:
    return Path(str(source)).name


def load_resume_state(args: argparse.Namespace) -> tuple[list[RawManifestRow], list[SampleManifestRow], set[str], list[str]]:
    if not args.resume:
        return [], [], set(), []

    manifest_dir = args.out_dir / "manifests"
    raw_rows = read_dataclass_csv(manifest_dir / "raw_manifest.csv", RawManifestRow)
    sample_rows = read_dataclass_csv(manifest_dir / "samples_manifest.csv", SampleManifestRow)

    processed_archives: list[str] = []
    progress_path = manifest_dir / "progress_checkpoint.json"
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            processed_archives = list(progress.get("processed_archives", []))
        except json.JSONDecodeError:
            processed_archives = []

    processed_ids = {archive_id(item) for item in processed_archives}
    processed_ids.update(row.archive_file for row in raw_rows if row.archive_file)
    processed_ids.update(row.archive_file for row in sample_rows if row.archive_file)

    if raw_rows or sample_rows or processed_ids:
        print(
            f"Resume enabled: loaded {len(raw_rows)} raw rows, {len(sample_rows)} sample rows, "
            f"{len(processed_ids)} processed archive ids."
        )
    else:
        print("Resume enabled: no previous checkpoint/manifest found; starting fresh.")

    return raw_rows, sample_rows, processed_ids, processed_archives


def build_dataset(args: argparse.Namespace, archives: list[Path]) -> tuple[list[RawManifestRow], list[SampleManifestRow]]:
    ensure_dir(args.extract_dir)
    extracted_dirs = [
        extract_archive(path, args.extract_dir, args.skip_extract, delete_after_extract=False)
        for path in archives
    ]
    archive_by_extract_dir = {args.extract_dir / archive_stem(path): path for path in archives}
    raw_exts = {e.strip().lower() for e in args.raw_exts.split(",") if e.strip()}
    raw_pairs = discover_raw_files(extracted_dirs, raw_exts)
    if args.max_raw_files:
        raw_pairs = raw_pairs[: args.max_raw_files]

    image_root = args.out_dir / "spectrograms" / "images"
    ensure_dir(image_root)
    raw_rows: list[RawManifestRow] = []
    sample_rows: list[SampleManifestRow] = []

    for extracted_root, raw_path in raw_pairs:
        archive = archive_by_extract_dir.get(extracted_root)
        archive_file = archive.name if archive else extracted_root.name
        archive_path = str(archive or "")
        xml_path = find_nearest_xml(raw_path, extracted_root)
        meta = parse_xml(xml_path)
        raw_row, rows = process_raw_file(args, raw_path, xml_path, meta, archive_file, archive_path)
        raw_rows.append(raw_row)
        sample_rows.extend(rows)

    assign_splits(sample_rows, args.split_key, args.train_ratio, args.val_ratio, args.seed)
    return raw_rows, sample_rows


def write_checkpoint_outputs(
    args: argparse.Namespace,
    raw_rows: list[RawManifestRow],
    sample_rows: list[SampleManifestRow],
    processed_archives: list[str],
) -> None:
    assign_splits(sample_rows, args.split_key, args.train_ratio, args.val_ratio, args.seed)
    raw_fields = list(RawManifestRow.__dataclass_fields__.keys())
    sample_fields = list(SampleManifestRow.__dataclass_fields__.keys())

    manifest_dir = args.out_dir / "manifests"
    ensure_dir(manifest_dir)
    write_csv(manifest_dir / "raw_manifest.csv", raw_rows, raw_fields)
    write_csv(manifest_dir / "samples_manifest.csv", sample_rows, sample_fields)
    write_csv(manifest_dir / "raw_manifest_checkpoint.csv", raw_rows, raw_fields)
    write_csv(manifest_dir / "samples_manifest_checkpoint.csv", sample_rows, sample_fields)
    copy_split_index(sample_rows, args.out_dir)
    write_summary(raw_rows, sample_rows, args.out_dir)

    progress = {
        "status": "running",
        "processed_archives": processed_archives,
        "num_processed_archives": len(processed_archives),
        "num_raw_files": len(raw_rows),
        "num_samples": len(sample_rows),
    }
    progress_path = manifest_dir / "progress_checkpoint.json"
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    tmp_path.replace(progress_path)
    print(f"Checkpoint saved: {manifest_dir / 'samples_manifest.csv'} ({len(sample_rows)} samples)")


def norm_archive_name(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def safe_member_target(target_root: Path, member_name: str) -> Path:
    target_root_resolved = target_root.resolve()
    target_path = (target_root / norm_archive_name(member_name)).resolve()
    if not str(target_path).startswith(str(target_root_resolved)):
        raise RuntimeError(f"Unsafe archive member path: {member_name}")
    return target_path


def find_external_rar_extractor() -> tuple[str, str] | None:
    for exe in ("7z", "7zz", "7za"):
        path = shutil.which(exe)
        if path:
            return "7z", path
    for exe in ("unrar", "UnRAR"):
        path = shutil.which(exe)
        if path:
            return "unrar", path
    return None


def extract_rar_member_external(archive: Path, member_name: str, target_path: Path) -> None:
    extractor = find_external_rar_extractor()
    if extractor is None:
        raise RuntimeError(
            "rarfile could not read this RAR member and no external extractor was found. "
            "Install 7-Zip and add 7z.exe to PATH, or install UnRAR."
        )

    kind, exe = extractor
    if kind == "7z":
        cmd = [exe, "x", "-y", "-so", str(archive), member_name]
    else:
        cmd = [exe, "p", "-inul", str(archive), member_name]

    with target_path.open("wb") as dst:
        proc = subprocess.run(cmd, stdout=dst, stderr=subprocess.PIPE)

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        if target_path.exists():
            target_path.unlink()
        raise RuntimeError(f"External RAR extraction failed for {member_name}: {stderr}")


def read_rar_member_bytes_external(archive: Path, member_name: str) -> bytes:
    extractor = find_external_rar_extractor()
    if extractor is None:
        raise RuntimeError("No external RAR extractor found.")

    kind, exe = extractor
    if kind == "7z":
        cmd = [exe, "x", "-y", "-so", str(archive), member_name]
    else:
        cmd = [exe, "p", "-inul", str(archive), member_name]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"External RAR read failed for {member_name}: {stderr}")
    return proc.stdout


def nearest_xml_member(raw_name: str, xml_text_by_name: dict[str, str]) -> str | None:
    xml_names = sorted(xml_text_by_name)
    raw = Path(norm_archive_name(raw_name))
    raw_basename = raw.name.lower()

    for xml_name, xml_text in xml_text_by_name.items():
        meta = parse_xml_text(xml_text)
        iq_file_name = xml_value(meta, "IQFileName", "iq_file_name")
        if iq_file_name and Path(iq_file_name).name.lower() == raw_basename:
            xml_parent = Path(norm_archive_name(xml_name)).parent.as_posix()
            if xml_parent == raw.parent.as_posix():
                return xml_name

    for xml_name, xml_text in xml_text_by_name.items():
        meta = parse_xml_text(xml_text)
        iq_file_name = xml_value(meta, "IQFileName", "iq_file_name")
        if iq_file_name and Path(iq_file_name).name.lower() == raw_basename:
            return xml_name

    same_stem = raw.with_suffix(".xml").as_posix()
    for xml_name in xml_names:
        if norm_archive_name(xml_name) == same_stem:
            return xml_name

    for xml_name in xml_names:
        if xml_member_matches_raw_prefix(xml_name, raw_name):
            return xml_name

    raw_parent = raw.parent.as_posix()
    parent_matches = [x for x in xml_names if Path(norm_archive_name(x)).parent.as_posix() == raw_parent]
    if parent_matches:
        return sorted(parent_matches, key=len)[0]

    for parent in raw.parents:
        parent_posix = parent.as_posix()
        parent_matches = [x for x in xml_names if Path(norm_archive_name(x)).parent.as_posix() == parent_posix]
        if parent_matches:
            return sorted(parent_matches, key=len)[0]

    return xml_names[0] if len(xml_names) == 1 else None


def read_archive_index(archive: Path, raw_exts: set[str]) -> tuple[list[str], dict[str, str]]:
    lower = archive.name.lower()
    raw_names: list[str] = []
    xml_text_by_name: dict[str, str] = {}

    if lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = norm_archive_name(member.name)
                suffix = Path(name).suffix.lower()
                if suffix in raw_exts:
                    raw_names.append(member.name)
                elif suffix == ".xml":
                    try:
                        extracted = tf.extractfile(member)
                        if extracted is not None:
                            xml_text_by_name[member.name] = extracted.read().decode("utf-8", errors="ignore")
                    except Exception as exc:
                        print(f"Warning: could not read XML metadata {member.name} in {archive.name}: {exc}")
    elif lower.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = norm_archive_name(info.filename)
                suffix = Path(name).suffix.lower()
                if suffix in raw_exts:
                    raw_names.append(info.filename)
                elif suffix == ".xml":
                    try:
                        xml_text_by_name[info.filename] = zf.read(info).decode("utf-8", errors="ignore")
                    except Exception as exc:
                        print(f"Warning: could not read XML metadata {info.filename} in {archive.name}: {exc}")
    elif lower.endswith(".rar"):
        try:
            import rarfile
        except ImportError as exc:
            raise RuntimeError("Install `rarfile` and an unrar/bsdtar/7z executable to stream .rar files.") from exc
        with rarfile.RarFile(archive) as rf:
            for info in rf.infolist():
                if info.isdir():
                    continue
                name = norm_archive_name(info.filename)
                suffix = Path(name).suffix.lower()
                if suffix in raw_exts:
                    raw_names.append(info.filename)
                elif suffix == ".xml":
                    try:
                        xml_text_by_name[info.filename] = rf.read(info).decode("utf-8", errors="ignore")
                    except Exception as exc:
                        try:
                            xml_text_by_name[info.filename] = read_rar_member_bytes_external(
                                archive, info.filename
                            ).decode("utf-8", errors="ignore")
                        except Exception as fallback_exc:
                            print(
                                f"Warning: could not read XML metadata {info.filename} in {archive.name}: "
                                f"{exc}; external fallback also failed: {fallback_exc}"
                            )
    else:
        raise RuntimeError(f"Unsupported archive type: {archive}")

    return sorted(raw_names), xml_text_by_name


def extract_one_member(archive: Path, member_name: str, target_root: Path) -> Path:
    lower = archive.name.lower()
    target_path = safe_member_target(target_root, member_name)
    ensure_dir(target_path.parent)

    if lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        with tarfile.open(archive) as tf:
            member = tf.getmember(member_name)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Cannot extract member: {member_name}")
            with target_path.open("wb") as f:
                shutil.copyfileobj(extracted, f)
    elif lower.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            with zf.open(member_name) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    elif lower.endswith(".rar"):
        import rarfile

        try:
            with rarfile.RarFile(archive) as rf:
                with rf.open(member_name) as src, target_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        except Exception as exc:
            if target_path.exists():
                target_path.unlink()
            print(f"Warning: rarfile failed for {member_name}: {exc}. Trying external extractor.")
            extract_rar_member_external(archive, member_name, target_path)
    else:
        raise RuntimeError(f"Unsupported archive type: {archive}")

    return target_path


def write_xml_sidecar_if_needed(target_root: Path, xml_name: str | None, xml_text: str | None) -> Path | None:
    if not xml_name or xml_text is None:
        return None
    xml_path = safe_member_target(target_root, xml_name)
    if not xml_path.exists():
        ensure_dir(xml_path.parent)
        xml_path.write_text(xml_text, encoding="utf-8")
    return xml_path


def build_dataset_streaming(
    args: argparse.Namespace,
    archives: list[Path],
) -> tuple[list[RawManifestRow], list[SampleManifestRow]]:
    ensure_dir(args.extract_dir)
    raw_exts = {e.strip().lower() for e in args.raw_exts.split(",") if e.strip()}
    raw_rows: list[RawManifestRow] = []
    sample_rows: list[SampleManifestRow] = []
    processed_raw = 0

    for archive in archives:
        archive_target_root = args.extract_dir / archive_stem(archive)
        ensure_dir(archive_target_root)
        raw_names, xml_text_by_name = read_archive_index(archive, raw_exts)
        print(f"Streaming archive: {archive.name} ({len(raw_names)} raw files)")

        for raw_name in raw_names:
            if args.max_raw_files is not None and processed_raw >= args.max_raw_files:
                break

            raw_path = extract_one_member(archive, raw_name, archive_target_root)
            xml_name = nearest_xml_member(raw_name, xml_text_by_name)
            xml_text = xml_text_by_name.get(xml_name or "")
            xml_path = write_xml_sidecar_if_needed(archive_target_root, xml_name, xml_text)
            meta = parse_xml_text(xml_text) if xml_text else {}

            raw_row, rows = process_raw_file(args, raw_path, xml_path, meta, archive.name, str(archive))
            raw_rows.append(raw_row)
            sample_rows.extend(rows)
            processed_raw += 1

        if args.max_raw_files is not None and processed_raw >= args.max_raw_files:
            break

    assign_splits(sample_rows, args.split_key, args.train_ratio, args.val_ratio, args.seed)
    return raw_rows, sample_rows


def cleanup_archive_outputs(args: argparse.Namespace, archive: Path) -> None:
    if args.delete_archive_after_extract and archive.exists():
        archive.unlink()
        print(f"Deleted archive after conversion: {archive}")

    if args.delete_extracted_after_convert:
        extracted_dir = args.extract_dir / archive_stem(archive)
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)
            print(f"Deleted extracted folder after conversion: {extracted_dir}")


def process_one_archive(
    args: argparse.Namespace,
    archive: Path,
) -> tuple[list[RawManifestRow], list[SampleManifestRow]]:
    if args.stream_extract_convert:
        raw_rows, sample_rows = build_dataset_streaming(args, [archive])
    else:
        raw_rows, sample_rows = build_dataset(args, [archive])
    cleanup_archive_outputs(args, archive)
    return raw_rows, sample_rows


def process_archives_sequentially(args: argparse.Namespace) -> tuple[list[RawManifestRow], list[SampleManifestRow]]:
    if args.skip_download:
        archive_sources: list[str | Path] = local_archives(args)
    else:
        archive_sources = list_hf_archives(args)

    if not archive_sources:
        raise RuntimeError("No archives to process.")

    all_raw_rows, all_sample_rows, processed_ids, processed_archives = load_resume_state(args)
    if args.resume and processed_ids:
        before = len(archive_sources)
        archive_sources = [source for source in archive_sources if archive_id(source) not in processed_ids]
        skipped = before - len(archive_sources)
        print(f"Resume skip: {skipped} archives already processed, {len(archive_sources)} remaining.")

    if not archive_sources:
        print("No remaining archives to process after resume filtering.")
        assign_splits(all_sample_rows, args.split_key, args.train_ratio, args.val_ratio, args.seed)
        return all_raw_rows, all_sample_rows

    for index, source in enumerate(archive_sources, start=1):
        print(f"\n=== Archive {index}/{len(archive_sources)}: {source} ===")
        archive = source if isinstance(source, Path) else download_one_archive(args, source)
        raw_rows, sample_rows = process_one_archive(args, archive)
        all_raw_rows.extend(raw_rows)
        all_sample_rows.extend(sample_rows)
        processed_archives.append(str(source))
        processed_ids.add(archive_id(source))
        write_checkpoint_outputs(args, all_raw_rows, all_sample_rows, processed_archives)

    assign_splits(all_sample_rows, args.split_key, args.train_ratio, args.val_ratio, args.seed)
    return all_raw_rows, all_sample_rows


def main() -> None:
    args = parse_args()
    resolve_hf_token(args)
    ensure_dir(args.out_dir)
    ensure_dir(args.out_dir / "manifests")

    raw_rows, sample_rows = process_archives_sequentially(args)
    raw_fields = list(RawManifestRow.__dataclass_fields__.keys())
    sample_fields = list(SampleManifestRow.__dataclass_fields__.keys())
    write_csv(args.out_dir / "manifests" / "raw_manifest.csv", raw_rows, raw_fields)
    write_csv(args.out_dir / "manifests" / "samples_manifest.csv", sample_rows, sample_fields)
    copy_split_index(sample_rows, args.out_dir)
    write_summary(raw_rows, sample_rows, args.out_dir)
    progress_path = args.out_dir / "manifests" / "progress_checkpoint.json"
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            progress = {}
    else:
        progress = {}
    progress.update(
        {
            "status": "complete",
            "num_raw_files": len(raw_rows),
            "num_samples": len(sample_rows),
        }
    )
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    print(f"Done. Raw files: {len(raw_rows)}. Spectrogram samples: {len(sample_rows)}.")
    print(f"Raw manifest: {args.out_dir / 'manifests' / 'raw_manifest.csv'}")
    print(f"Sample manifest: {args.out_dir / 'manifests' / 'samples_manifest.csv'}")


if __name__ == "__main__":
    main()
