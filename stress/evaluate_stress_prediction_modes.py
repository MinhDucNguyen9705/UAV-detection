from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stress predictions while keeping mAP and deploy metrics separate.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--eval-predictions", type=Path, nargs="*", default=[])
    parser.add_argument("--deploy-predictions", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluator-script",
        type=Path,
        default=Path(__file__).with_name("evaluate_stress_predictions.py"),
        help="Script exposing evaluate_model(prediction_dir, args). Defaults to stress/evaluate_stress_predictions.py.",
    )
    parser.add_argument("--raw-manifest", type=Path, default=None)
    parser.add_argument("--benchmark-manifest", type=Path, default=None)
    parser.add_argument("--mix2-raw-annotations", type=Path, default=None)
    parser.add_argument("--near-far-raw-annotations", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=0.03)
    parser.add_argument("--default-sample-rate", type=float, default=100e6)
    parser.add_argument("--default-center-frequency", type=float, default=2.4e9)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--map-thresholds", type=float, nargs="*", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--single-class", action="store_true", default=True)
    return parser.parse_args()


def load_evaluator(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Evaluator script not found: {path}")
    spec = importlib.util.spec_from_file_location("rfuav_stress_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import evaluator script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluator_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        raw_manifest=args.raw_manifest,
        benchmark_manifest=args.benchmark_manifest or args.dataset_dir / "benchmark_manifest.csv",
        mix2_raw_annotations=args.mix2_raw_annotations,
        near_far_raw_annotations=args.near_far_raw_annotations,
        duration=args.duration,
        default_sample_rate=args.default_sample_rate,
        default_center_frequency=args.default_center_frequency,
        iou_threshold=args.iou_threshold,
        map_thresholds=args.map_thresholds,
        conf_threshold=args.conf_threshold,
        single_class=args.single_class,
    )


def model_name_from_prediction_dir(pred_dir: Path) -> str:
    if pred_dir.name.startswith(("eval_conf", "deploy_conf")) and pred_dir.parent.name:
        return pred_dir.parent.name
    return pred_dir.name


def evaluate_with_model_name(evaluator: Any, pred_dir: Path, ev_args: SimpleNamespace) -> dict[str, Any]:
    result = evaluator.evaluate_model(pred_dir.resolve(), ev_args)
    result["model_name"] = model_name_from_prediction_dir(pred_dir.resolve())
    return result


def compact_eval_metrics(result: dict[str, Any]) -> dict[str, Any]:
    keys = ["model_name", "prediction_dir", "num_images", "mAP@0.5", "mAP@0.5:0.95", "ap_by_threshold", "by_scenario", "by_snr_db", "by_sir_db", "source_recall", "near_far_weak_strong_recall_by_sir"]
    return {key: result.get(key) for key in keys}


def compact_deploy_metrics(result: dict[str, Any]) -> dict[str, Any]:
    keys = ["model_name", "prediction_dir", "num_images", "conf_threshold", "iou_threshold", "precision", "recall", "f1", "by_scenario", "by_snr_db", "by_sir_db", "count_accuracy", "mean_abs_count_error", "false_alarm_per_noise_image", "merge_candidate_fp", "duplicate_candidate_fp", "cross_source_merge_predictions", "source_aware_count_accuracy", "source_aware_mean_abs_count_error", "source_recall", "near_far_weak_strong_recall_by_sir", "parameter_mae"]
    return {key: result.get(key) for key in keys}


def write_role_summary(path: Path, eval_results: list[dict[str, Any]], deploy_results: list[dict[str, Any]]) -> None:
    fields = ["role", "model_name", "prediction_dir", "num_images", "mAP@0.5", "mAP@0.5:0.95", "precision", "recall", "f1", "count_accuracy", "false_alarm_per_noise_image", "center_frequency_mae_hz", "bandwidth_mae_hz", "duration_mae_sec", "source_aware_count_accuracy", "source_aware_mean_abs_count_error", "cross_source_merge_predictions"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in eval_results:
            writer.writerow({"role": "eval_map", "model_name": result.get("model_name"), "prediction_dir": result.get("prediction_dir"), "num_images": result.get("num_images"), "mAP@0.5": result.get("mAP@0.5"), "mAP@0.5:0.95": result.get("mAP@0.5:0.95")})
        for result in deploy_results:
            mae = result.get("parameter_mae") or {}
            writer.writerow(
                {
                    "role": "deploy_point",
                    "model_name": result.get("model_name"),
                    "prediction_dir": result.get("prediction_dir"),
                    "num_images": result.get("num_images"),
                    "precision": result.get("precision"),
                    "recall": result.get("recall"),
                    "f1": result.get("f1"),
                    "count_accuracy": result.get("count_accuracy"),
                    "false_alarm_per_noise_image": result.get("false_alarm_per_noise_image"),
                    "center_frequency_mae_hz": (mae.get("center_frequency_hz") or {}).get("mae"),
                    "bandwidth_mae_hz": (mae.get("bandwidth_hz") or {}).get("mae"),
                    "duration_mae_sec": (mae.get("duration_sec") or {}).get("mae"),
                    "source_aware_count_accuracy": result.get("source_aware_count_accuracy"),
                    "source_aware_mean_abs_count_error": result.get("source_aware_mean_abs_count_error"),
                    "cross_source_merge_predictions": result.get("cross_source_merge_predictions"),
                }
            )


def main() -> int:
    args = parse_args()
    if not args.eval_predictions and not args.deploy_predictions:
        raise ValueError("Provide at least one of --eval-predictions or --deploy-predictions.")
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = load_evaluator(args.evaluator_script.resolve())
    ev_args = evaluator_args(args)
    eval_results = [evaluate_with_model_name(evaluator, pred_dir, ev_args) for pred_dir in args.eval_predictions]
    deploy_results = [evaluate_with_model_name(evaluator, pred_dir, ev_args) for pred_dir in args.deploy_predictions]

    for result in eval_results:
        (args.output_dir / f"{result['model_name']}_eval_map_metrics.json").write_text(json.dumps(compact_eval_metrics(result), indent=2), encoding="utf-8")
    for result in deploy_results:
        (args.output_dir / f"{result['model_name']}_deploy_point_metrics.json").write_text(json.dumps(compact_deploy_metrics(result), indent=2), encoding="utf-8")
    combined = {
        "notes": [
            "eval_map metrics should be computed from low-confidence inference outputs.",
            "deploy_point metrics should be computed from operational-threshold inference outputs.",
            "Do not compare deploy mAP directly when deploy inference used a high confidence cutoff.",
        ],
        "eval_map": [compact_eval_metrics(result) for result in eval_results],
        "deploy_point": [compact_deploy_metrics(result) for result in deploy_results],
    }
    (args.output_dir / "all_metrics_by_role.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    write_role_summary(args.output_dir / "summary_by_role.csv", eval_results, deploy_results)
    print(f"Metrics written to: {args.output_dir}")
    for result in eval_results:
        print(
            f"[eval/mAP] {result['model_name']}: "
            f"mAP@0.5={result.get('mAP@0.5', 0.0):.4f}, "
            f"mAP@0.5:0.95={result.get('mAP@0.5:0.95', 0.0):.4f}"
        )
    for result in deploy_results:
        print(
            f"[deploy] {result['model_name']}: "
            f"precision={result.get('precision', 0.0):.4f}, "
            f"recall={result.get('recall', 0.0):.4f}, "
            f"f1={result.get('f1', 0.0):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
