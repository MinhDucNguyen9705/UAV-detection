from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class Box:
    image_key: str
    sample_id: str
    scenario: str
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


@dataclass
class Match:
    image_key: str
    sample_id: str
    scenario: str
    gt_index: int
    pred_index: int
    iou: float
    confidence: float
    gt_box: Box
    pred_box: Box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO/RT-DETR predictions on RFUAV stress benchmark."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Stress dataset root containing YOLO images/labels and benchmark metadata.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="Prediction output directories, each containing predictions.csv and predictions_coco.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        default=None,
        help="Optional raw_manifest.csv used for parameter mapping/provenance.",
    )
    parser.add_argument("--benchmark-manifest", type=Path, default=None)
    parser.add_argument(
        "--mix2-raw-annotations",
        type=Path,
        default=None,
        help="Optional component-level COCO annotations for mix2 source-aware metrics.",
    )
    parser.add_argument(
        "--near-far-raw-annotations",
        type=Path,
        default=None,
        help="Optional component-level COCO annotations for near-far weak/strong metrics.",
    )
    parser.add_argument("--duration", type=float, default=0.03)
    parser.add_argument("--default-sample-rate", type=float, default=100e6)
    parser.add_argument("--default-center-frequency", type=float, default=2.4e9)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--map-thresholds",
        type=float,
        nargs="*",
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Threshold for point metrics such as precision/recall/counting.",
    )
    parser.add_argument(
        "--single-class",
        action="store_true",
        default=True,
        help="Collapse all GT/pred classes to class 0 uav_signal.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def image_key(path_or_relative: str) -> str:
    value = path_or_relative.replace("\\", "/")
    marker = "stress_yolo/images/"
    if marker in value:
        return value[value.index(marker) :]
    return value.lstrip("./")


def scenario_from_key(key: str) -> str:
    parts = key.split("/")
    for idx, part in enumerate(parts[:-1]):
        if part == "images" and idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def sample_id_from_key(key: str) -> str:
    return Path(key).stem


def yolo_to_xyxy(parts: list[str], width: float, height: float) -> tuple[float, float, float, float]:
    xc, yc, bw, bh = [float(item) for item in parts[:4]]
    x1 = (xc - bw / 2.0) * width
    y1 = (yc - bh / 2.0) * height
    x2 = (xc + bw / 2.0) * width
    y2 = (yc + bh / 2.0) * height
    return x1, y1, x2, y2


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = box_area((ix1, iy1, ix2, iy2))
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def load_eval_images(pred_dir: Path) -> dict[str, dict[str, Any]]:
    coco_path = pred_dir / "predictions_coco.json"
    if not coco_path.is_file():
        raise FileNotFoundError(f"Missing predictions_coco.json: {coco_path}")
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images: dict[str, dict[str, Any]] = {}
    for item in data.get("images", []):
        key = image_key(item.get("file_name", ""))
        images[key] = {
            "width": int(item.get("width", 0)),
            "height": int(item.get("height", 0)),
            "scenario": item.get("scenario") or scenario_from_key(key),
            "sample_id": item.get("sample_id") or sample_id_from_key(key),
        }
    return images


def load_gt_boxes(dataset_dir: Path, eval_images: dict[str, dict[str, Any]], single_class: bool) -> dict[str, list[Box]]:
    labels_root = dataset_dir / "stress_yolo" / "labels"
    gt_by_image: dict[str, list[Box]] = {}
    for key, meta in eval_images.items():
        parts = key.split("/")
        label_parts = parts[:]
        try:
            idx = label_parts.index("images")
            label_parts[idx] = "labels"
        except ValueError:
            pass
        label_path = dataset_dir / Path(*label_parts)
        label_path = label_path.with_suffix(".txt")
        boxes: list[Box] = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                row = line.lstrip("\ufeff").strip().split()
                if len(row) < 5:
                    continue
                class_id = 0 if single_class else int(float(row[0]))
                boxes.append(
                    Box(
                        image_key=key,
                        sample_id=meta["sample_id"],
                        scenario=meta["scenario"],
                        class_id=class_id,
                        xyxy=yolo_to_xyxy(row[1:5], meta["width"], meta["height"]),
                    )
                )
        gt_by_image[key] = boxes
    return gt_by_image


def load_predictions(pred_dir: Path, single_class: bool) -> dict[str, list[Box]]:
    rows = read_csv(pred_dir / "predictions.csv")
    by_image: dict[str, list[Box]] = {}
    for row in rows:
        key = image_key(row.get("relative_image_path") or row.get("image_path", ""))
        class_id = 0 if single_class else int(float(row.get("class_id", 0)))
        box = Box(
            image_key=key,
            sample_id=row.get("sample_id") or sample_id_from_key(key),
            scenario=row.get("scenario") or scenario_from_key(key),
            class_id=class_id,
            confidence=as_float(row.get("confidence"), 0.0),
            xyxy=(
                as_float(row.get("x1_px")),
                as_float(row.get("y1_px")),
                as_float(row.get("x2_px")),
                as_float(row.get("y2_px")),
            ),
        )
        by_image.setdefault(key, []).append(box)
    for boxes in by_image.values():
        boxes.sort(key=lambda item: item.confidence, reverse=True)
    return by_image


def greedy_match(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_threshold: float,
    conf_threshold: float | None = None,
) -> tuple[list[Match], list[int], list[int]]:
    preds = [
        (idx, box)
        for idx, box in enumerate(pred_boxes)
        if conf_threshold is None or box.confidence >= conf_threshold
    ]
    preds.sort(key=lambda item: item[1].confidence, reverse=True)
    used_gt: set[int] = set()
    matches: list[Match] = []
    unmatched_pred: list[int] = []
    for pred_idx, pred in preds:
        best_gt = -1
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in used_gt or gt.class_id != pred.class_id:
                continue
            value = iou(gt.xyxy, pred.xyxy)
            if value > best_iou:
                best_iou = value
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt)
            matches.append(
                Match(
                    image_key=pred.image_key,
                    sample_id=pred.sample_id,
                    scenario=pred.scenario,
                    gt_index=best_gt,
                    pred_index=pred_idx,
                    iou=best_iou,
                    confidence=pred.confidence,
                    gt_box=gt_boxes[best_gt],
                    pred_box=pred,
                )
            )
        else:
            unmatched_pred.append(pred_idx)
    unmatched_gt = [idx for idx in range(len(gt_boxes)) if idx not in used_gt]
    return matches, unmatched_gt, unmatched_pred


def precision_recall_ap(gt_by_image: dict[str, list[Box]], pred_by_image: dict[str, list[Box]], threshold: float) -> dict[str, float]:
    total_gt = sum(len(items) for items in gt_by_image.values())
    all_preds: list[tuple[str, int, Box]] = []
    for key, boxes in pred_by_image.items():
        for idx, box in enumerate(boxes):
            all_preds.append((key, idx, box))
    all_preds.sort(key=lambda item: item[2].confidence, reverse=True)

    used: dict[str, set[int]] = {key: set() for key in gt_by_image}
    tp: list[int] = []
    fp: list[int] = []
    for key, _, pred in all_preds:
        gt_boxes = gt_by_image.get(key, [])
        best_gt = -1
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in used.setdefault(key, set()) or gt.class_id != pred.class_id:
                continue
            value = iou(gt.xyxy, pred.xyxy)
            if value > best_iou:
                best_iou = value
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= threshold:
            used[key].add(best_gt)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    if total_gt == 0:
        false_positives = sum(fp)
        return {
            "ap": 0.0,
            "precision_final": 0.0 if false_positives else 1.0,
            "recall_final": 0.0,
            "total_gt": 0,
            "total_pred": len(all_preds),
        }

    cum_tp = []
    cum_fp = []
    running_tp = 0
    running_fp = 0
    for t, f in zip(tp, fp):
        running_tp += t
        running_fp += f
        cum_tp.append(running_tp)
        cum_fp.append(running_fp)

    precisions = [
        cum_tp[i] / max(cum_tp[i] + cum_fp[i], 1)
        for i in range(len(cum_tp))
    ]
    recalls = [cum_tp[i] / total_gt for i in range(len(cum_tp))]
    ap = 0.0
    for recall_threshold in [x / 100 for x in range(0, 101)]:
        candidates = [p for p, r in zip(precisions, recalls) if r >= recall_threshold]
        ap += (max(candidates) if candidates else 0.0) / 101.0
    return {
        "ap": ap,
        "precision_final": precisions[-1] if precisions else 0.0,
        "recall_final": recalls[-1] if recalls else 0.0,
        "total_gt": total_gt,
        "total_pred": len(all_preds),
    }


def load_benchmark_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    rows = read_csv(path)
    return {row.get("sample_id", ""): row for row in rows if row.get("sample_id")}


def load_raw_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    rows = read_csv(path)
    return {row.get("raw_file_id", ""): row for row in rows if row.get("raw_file_id")}


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def component_source_index(file_name: str) -> int | None:
    lower = file_name.lower()
    if "_source0." in lower:
        return 0
    if "_source1_shifted_scaled." in lower:
        return 1
    return None


def component_sample_id(file_name: str) -> str:
    return Path(file_name).parts[0] if len(Path(file_name).parts) > 1 else Path(file_name).stem


def load_source_gt_annotations(
    path: Path | None,
    scenario: str,
    single_class: bool,
) -> dict[str, dict[int, list[Box]]]:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    images = {
        int(item["id"]): item
        for item in data.get("images", [])
    }
    out: dict[str, dict[int, list[Box]]] = {}
    for ann in data.get("annotations", []):
        image = images.get(int(ann.get("image_id", -1)))
        if not image:
            continue
        file_name = image.get("file_name", "")
        source_index = component_source_index(file_name)
        if source_index is None:
            continue
        sample_id = component_sample_id(file_name)
        class_id = 0 if single_class else int(ann.get("category_id", 1)) - 1
        out.setdefault(sample_id, {}).setdefault(source_index, []).append(
            Box(
                image_key=f"stress_yolo/images/{scenario}/{sample_id}{Path(file_name).suffix}",
                sample_id=sample_id,
                scenario=scenario,
                class_id=class_id,
                xyxy=xywh_to_xyxy([float(v) for v in ann.get("bbox", [0, 0, 0, 0])]),
            )
        )
    return out


def load_all_source_gt(args: argparse.Namespace) -> dict[str, dict[int, list[Box]]]:
    merged: dict[str, dict[int, list[Box]]] = {}
    for scenario, path in (
        ("mix2", args.mix2_raw_annotations),
        ("near_far", args.near_far_raw_annotations),
    ):
        scenario_rows = load_source_gt_annotations(path, scenario, args.single_class)
        for sample_id, sources in scenario_rows.items():
            merged[sample_id] = sources
    return merged


def group_key(meta: dict[str, str], field: str) -> str:
    return meta.get(field, "") or "unknown"


def physical_params(box: Box, image_meta: dict[str, Any], benchmark_row: dict[str, str], raw_rows: dict[str, dict[str, str]], args: argparse.Namespace) -> dict[str, float]:
    raw_id = benchmark_row.get("source_raw_file_ids", "").split(";")[0].strip()
    raw = raw_rows.get(raw_id, {})
    sample_rate = as_float(raw.get("sample_rate_hz"), args.default_sample_rate)
    center_freq = as_float(raw.get("center_frequency_hz"), args.default_center_frequency)
    freq_min = center_freq - sample_rate / 2.0
    freq_max = center_freq + sample_rate / 2.0
    duration = args.duration
    width = as_float(image_meta.get("width"), 1920)
    height = as_float(image_meta.get("height"), 1440)
    x1, y1, x2, y2 = box.xyxy
    t0 = (x1 / width) * duration
    t1 = (x2 / width) * duration
    f_high = freq_max - (y1 / height) * (freq_max - freq_min)
    f_low = freq_max - (y2 / height) * (freq_max - freq_min)
    return {
        "center_frequency_hz": (f_low + f_high) / 2.0,
        "bandwidth_hz": max(0.0, f_high - f_low),
        "duration_sec": max(0.0, t1 - t0),
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mae": math.nan, "mean": math.nan, "count": 0}
    return {"mae": mean(abs(v) for v in values), "mean": mean(values), "count": len(values)}


def evaluate_model(pred_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    eval_images = load_eval_images(pred_dir)
    gt_by_image = load_gt_boxes(args.dataset_dir, eval_images, args.single_class)
    pred_by_image = load_predictions(pred_dir, args.single_class)
    benchmark_rows = load_benchmark_rows(args.benchmark_manifest)
    raw_rows = load_raw_rows(args.raw_manifest)
    source_gt_by_sample = load_all_source_gt(args)

    ap_by_threshold = {
        f"{thr:.2f}": precision_recall_ap(gt_by_image, pred_by_image, thr)
        for thr in args.map_thresholds
    }
    map_50 = ap_by_threshold.get("0.50", {}).get("ap", 0.0)
    map_5095 = mean(item["ap"] for item in ap_by_threshold.values()) if ap_by_threshold else 0.0

    totals = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
    scenario_stats: dict[str, dict[str, float]] = {}
    snr_stats: dict[str, dict[str, float]] = {}
    sir_stats: dict[str, dict[str, float]] = {}
    count_abs_errors: list[float] = []
    exact_count = 0
    false_alarm_counts: list[int] = []
    merge_candidates = 0
    duplicate_fp = 0
    source_stats: dict[str, dict[str, float]] = {}
    weak_stats: dict[str, dict[str, float]] = {}
    source_count_abs_errors: list[float] = []
    exact_source_count_images = 0
    cross_source_merge_predictions = 0
    param_errors = {
        "center_frequency_hz": [],
        "bandwidth_hz": [],
        "duration_sec": [],
    }

    for key, image_meta in eval_images.items():
        gt_boxes = gt_by_image.get(key, [])
        pred_boxes_all = pred_by_image.get(key, [])
        pred_boxes = [box for box in pred_boxes_all if box.confidence >= args.conf_threshold]
        matches, unmatched_gt, unmatched_pred = greedy_match(gt_boxes, pred_boxes, args.iou_threshold)
        sample_id = image_meta["sample_id"]
        scenario = image_meta["scenario"]
        benchmark_row = benchmark_rows.get(sample_id, {})

        totals["gt"] += len(gt_boxes)
        totals["pred"] += len(pred_boxes)
        totals["tp"] += len(matches)
        totals["fp"] += len(unmatched_pred)
        totals["fn"] += len(unmatched_gt)

        stat = scenario_stats.setdefault(scenario, {"images": 0, "gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0})
        stat["images"] += 1
        stat["gt"] += len(gt_boxes)
        stat["pred"] += len(pred_boxes)
        stat["tp"] += len(matches)
        stat["fp"] += len(unmatched_pred)
        stat["fn"] += len(unmatched_gt)

        if scenario == "low_snr":
            snr_key = group_key(benchmark_row, "snr_db")
            stat = snr_stats.setdefault(snr_key, {"images": 0, "gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0})
            stat["images"] += 1
            stat["gt"] += len(gt_boxes)
            stat["pred"] += len(pred_boxes)
            stat["tp"] += len(matches)
            stat["fp"] += len(unmatched_pred)
            stat["fn"] += len(unmatched_gt)
        if scenario in {"mix2", "near_far"}:
            sir_key = f"{scenario}:{group_key(benchmark_row, 'sir_db')}"
            stat = sir_stats.setdefault(sir_key, {"images": 0, "gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0})
            stat["images"] += 1
            stat["gt"] += len(gt_boxes)
            stat["pred"] += len(pred_boxes)
            stat["tp"] += len(matches)
            stat["fp"] += len(unmatched_pred)
            stat["fn"] += len(unmatched_gt)

            source_gt = source_gt_by_sample.get(sample_id, {})
            if source_gt:
                image_source_exact = True
                for source_index in sorted(source_gt):
                    source_boxes = source_gt[source_index]
                    source_matches, source_unmatched_gt, _ = greedy_match(
                        source_boxes,
                        pred_boxes,
                        args.iou_threshold,
                    )
                    source_key = f"{scenario}:source{source_index}"
                    stat = source_stats.setdefault(
                        source_key,
                        {"images": 0, "gt": 0, "matched_gt": 0, "missed_gt": 0},
                    )
                    stat["images"] += 1
                    stat["gt"] += len(source_boxes)
                    stat["matched_gt"] += len(source_matches)
                    stat["missed_gt"] += len(source_unmatched_gt)
                    source_count_abs_errors.append(abs(len(source_matches) - len(source_boxes)))
                    if len(source_matches) != len(source_boxes):
                        image_source_exact = False

                    if scenario == "near_far":
                        weak_index = int(as_float(benchmark_row.get("weak_source_index"), 1))
                        role = "weak" if source_index == weak_index else "strong"
                        sir_role_key = f"{role}:sir_{group_key(benchmark_row, 'sir_db')}"
                        stat = weak_stats.setdefault(
                            sir_role_key,
                            {"images": 0, "gt": 0, "matched_gt": 0, "missed_gt": 0, "images_with_any_match": 0},
                        )
                        stat["images"] += 1
                        stat["gt"] += len(source_boxes)
                        stat["matched_gt"] += len(source_matches)
                        stat["missed_gt"] += len(source_unmatched_gt)
                        if source_matches:
                            stat["images_with_any_match"] += 1
                if image_source_exact:
                    exact_source_count_images += 1

                for pred in pred_boxes:
                    overlapped_sources = 0
                    for source_index, source_boxes in source_gt.items():
                        if any(iou(pred.xyxy, gt.xyxy) >= args.iou_threshold for gt in source_boxes):
                            overlapped_sources += 1
                    if overlapped_sources >= 2:
                        cross_source_merge_predictions += 1

        count_abs_errors.append(abs(len(pred_boxes) - len(gt_boxes)))
        if len(pred_boxes) == len(gt_boxes):
            exact_count += 1
        if scenario == "noise_only":
            false_alarm_counts.append(len(pred_boxes))

        for pred_idx in unmatched_pred:
            pred = pred_boxes[pred_idx]
            overlaps = [iou(pred.xyxy, gt.xyxy) for gt in gt_boxes]
            if sum(value >= args.iou_threshold for value in overlaps) >= 2:
                merge_candidates += 1
            elif any(value >= args.iou_threshold for value in overlaps):
                duplicate_fp += 1

        for match in matches:
            gt_params = physical_params(match.gt_box, image_meta, benchmark_row, raw_rows, args)
            pred_params = physical_params(match.pred_box, image_meta, benchmark_row, raw_rows, args)
            for name in param_errors:
                param_errors[name].append(pred_params[name] - gt_params[name])

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    def finalize_group(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        out = {}
        for key, stat in sorted(stats.items()):
            p = stat["tp"] / max(stat["tp"] + stat["fp"], 1)
            r = stat["tp"] / max(stat["tp"] + stat["fn"], 1)
            out[key] = {**stat, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)}
        return out

    def finalize_source_stats(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        out = {}
        for key, stat in sorted(stats.items()):
            recall_value = stat["matched_gt"] / max(stat["gt"], 1)
            out[key] = {**stat, "recall": recall_value}
        return out

    def finalize_weak_stats(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        out = {}
        for key, stat in sorted(stats.items()):
            recall_value = stat["matched_gt"] / max(stat["gt"], 1)
            image_recall = stat["images_with_any_match"] / max(stat["images"], 1)
            out[key] = {**stat, "recall": recall_value, "image_any_recall": image_recall}
        return out

    return {
        "model_name": pred_dir.name,
        "prediction_dir": str(pred_dir),
        "num_images": len(eval_images),
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
        "totals": totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP@0.5": map_50,
        "mAP@0.5:0.95": map_5095,
        "ap_by_threshold": ap_by_threshold,
        "by_scenario": finalize_group(scenario_stats),
        "by_snr_db": finalize_group(snr_stats),
        "by_sir_db": finalize_group(sir_stats),
        "count_accuracy": exact_count / max(len(eval_images), 1),
        "mean_abs_count_error": mean(count_abs_errors) if count_abs_errors else math.nan,
        "false_alarm_per_noise_image": mean(false_alarm_counts) if false_alarm_counts else math.nan,
        "noise_images": len(false_alarm_counts),
        "merge_candidate_fp": merge_candidates,
        "duplicate_candidate_fp": duplicate_fp,
        "cross_source_merge_predictions": cross_source_merge_predictions,
        "source_aware_count_accuracy": exact_source_count_images / max(
            sum(1 for key, meta in eval_images.items() if meta["scenario"] in {"mix2", "near_far"} and meta["sample_id"] in source_gt_by_sample),
            1,
        ),
        "source_aware_mean_abs_count_error": mean(source_count_abs_errors) if source_count_abs_errors else math.nan,
        "source_recall": finalize_source_stats(source_stats),
        "near_far_weak_strong_recall_by_sir": finalize_weak_stats(weak_stats),
        "parameter_mae": {
            name: summarize_values(values)
            for name, values in param_errors.items()
        },
        "limitations": [
            "Single-class evaluation collapses fhss/ofdm into uav_signal; class confusion is not measured.",
            "Weak/strong near_far recall uses component-level raw annotations for source0/source1_shifted_scaled.",
            "Cross-source merge counts predictions overlapping GT boxes from both sources; split errors remain heuristic.",
        ],
    }


def write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name",
        "num_images",
        "precision",
        "recall",
        "f1",
        "mAP@0.5",
        "mAP@0.5:0.95",
        "count_accuracy",
        "mean_abs_count_error",
        "false_alarm_per_noise_image",
        "center_frequency_mae_hz",
        "bandwidth_mae_hz",
        "duration_mae_sec",
        "source_aware_count_accuracy",
        "source_aware_mean_abs_count_error",
        "cross_source_merge_predictions",
        "weak_recall_sir10",
        "weak_recall_sir20",
        "weak_recall_sir30",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model_name": result["model_name"],
                    "num_images": result["num_images"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                    "mAP@0.5": result["mAP@0.5"],
                    "mAP@0.5:0.95": result["mAP@0.5:0.95"],
                    "count_accuracy": result["count_accuracy"],
                    "mean_abs_count_error": result["mean_abs_count_error"],
                    "false_alarm_per_noise_image": result["false_alarm_per_noise_image"],
                    "center_frequency_mae_hz": result["parameter_mae"]["center_frequency_hz"]["mae"],
                    "bandwidth_mae_hz": result["parameter_mae"]["bandwidth_hz"]["mae"],
                    "duration_mae_sec": result["parameter_mae"]["duration_sec"]["mae"],
                    "source_aware_count_accuracy": result["source_aware_count_accuracy"],
                    "source_aware_mean_abs_count_error": result["source_aware_mean_abs_count_error"],
                    "cross_source_merge_predictions": result["cross_source_merge_predictions"],
                    "weak_recall_sir10": result["near_far_weak_strong_recall_by_sir"].get("weak:sir_10.0", {}).get("recall", ""),
                    "weak_recall_sir20": result["near_far_weak_strong_recall_by_sir"].get("weak:sir_20.0", {}).get("recall", ""),
                    "weak_recall_sir30": result["near_far_weak_strong_recall_by_sir"].get("weak:sir_30.0", {}).get("recall", ""),
                }
            )


def main() -> None:
    args = parse_args()
    if args.benchmark_manifest is None:
        args.benchmark_manifest = args.dataset_dir / "benchmark_manifest.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for pred_dir in args.predictions:
        result = evaluate_model(pred_dir.resolve(), args)
        results.append(result)
        out_file = args.output_dir / f"{pred_dir.name}_metrics.json"
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{pred_dir.name}: mAP@0.5={result['mAP@0.5']:.4f}, recall={result['recall']:.4f}, precision={result['precision']:.4f}")

    combined = {"models": results}
    (args.output_dir / "all_metrics.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    write_summary_csv(args.output_dir / "summary.csv", results)
    print(f"Metrics written to: {args.output_dir}")


if __name__ == "__main__":
    main()
