from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RawManifestRow:
    raw_file_id: str
    archive_file: str
    archive_path: str
    raw_path: str
    xml_path: str
    class_name: str
    session_id: str
    device_type: str
    drone: str
    serial_number: str
    data_type: str
    reference_snr_db: str
    center_frequency_hz: float
    sample_rate_hz: float
    if_bandwidth_hz: str
    scale_factor_db: str
    iq_file_name: str
    sample_count_xml: str
    effective_dtype: str
    raw_file_start_sec: float
    raw_file_end_sec: float
    num_iq_samples: int


@dataclass
class SampleManifestRow:
    sample_id: str
    split: str
    image_path: str
    raw_file_id: str
    archive_file: str
    raw_path: str
    xml_path: str
    class_name: str
    session_id: str
    sample_rate_hz: float
    center_frequency_hz: float
    stft_point: int
    duration_sec: float
    segment_index: int
    segment_start_sec: float
    segment_end_sec: float
    raw_file_start_sec: float
    raw_file_end_sec: float
    image_width: int
    image_height: int
    freq_min_hz: float
    freq_max_hz: float


def assign_splits(rows: list[SampleManifestRow], key: str, train: float, val: float, seed: int) -> None:
    groups: dict[str, list[SampleManifestRow]] = {}
    for row in rows:
        groups.setdefault(str(getattr(row, key)), []).append(row)

    group_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_train = int(round(len(group_ids) * train))
    n_val = int(round(len(group_ids) * val))
    train_groups = set(group_ids[:n_train])
    val_groups = set(group_ids[n_train : n_train + n_val])

    for group_id, group_rows in groups.items():
        split = "train" if group_id in train_groups else "val" if group_id in val_groups else "test"
        for row in group_rows:
            row.split = split


def write_csv(path: Path, rows: list[object], fieldnames: list[str]) -> None:
    _ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    tmp_path.replace(path)


def read_dataclass_csv(path: Path, cls) -> list:
    if not path.exists():
        return []
    field_defs = cls.__dataclass_fields__
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kwargs = {}
            for name, field_def in field_defs.items():
                kwargs[name] = _coerce_dataclass_value(row.get(name, ""), field_def.type)
            rows.append(cls(**kwargs))
    return rows


def copy_split_index(sample_rows: list[SampleManifestRow], out_dir: Path) -> None:
    split_dir = out_dir / "splits"
    _ensure_dir(split_dir)
    for split in ("train", "val", "test"):
        lines = [row.image_path for row in sample_rows if row.split == split]
        path = split_dir / f"{split}.txt"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        tmp_path.replace(path)


def write_summary(raw_rows: list[RawManifestRow], sample_rows: list[SampleManifestRow], out_dir: Path) -> None:
    summary = {
        "num_raw_files": len(raw_rows),
        "num_samples": len(sample_rows),
        "splits": {},
        "classes": {},
        "archives": {},
    }
    for row in sample_rows:
        summary["splits"][row.split] = summary["splits"].get(row.split, 0) + 1
        summary["classes"][row.class_name] = summary["classes"].get(row.class_name, 0) + 1
        summary["archives"][row.archive_file] = summary["archives"].get(row.archive_file, 0) + 1
    path = out_dir / "manifests" / "summary.json"
    _ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _coerce_dataclass_value(value: str, target_type):
    if target_type is int:
        return int(float(value or 0))
    if target_type is float:
        return float(value or 0.0)
    return value


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
