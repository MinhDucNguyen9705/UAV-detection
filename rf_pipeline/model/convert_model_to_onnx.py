from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

try:
    from .classification import (
        _build_torchvision_model,
        _clean_state_dict,
        _extract_checkpoint_parts,
        _infer_model_name,
        _infer_num_classes,
        _resolve_torch_device,
        _torch_load,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rf_pipeline.model.classification import (
        _build_torchvision_model,
        _clean_state_dict,
        _extract_checkpoint_parts,
        _infer_model_name,
        _infer_num_classes,
        _resolve_torch_device,
        _torch_load,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert detector or classifier checkpoints to ONNX.")
    parser.add_argument(
        "--kind",
        choices=["auto", "classifier", "detector"],
        default="classifier",
        help="Model kind to export. Default keeps the old classifier-only behavior.",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Input .pt checkpoint.")
    parser.add_argument("--out", type=Path, required=True, help="Output .onnx path.")
    parser.add_argument("--imgsz", type=int, default=None, help="Input image size. Defaults to checkpoint metadata or 224.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--device", default="cpu", help="Export device, usually cpu or cuda:0.")
    parser.add_argument("--dynamic-batch", action="store_true", help="Export dynamic batch axis.")
    parser.add_argument("--architecture", choices=["auto", "yolo", "rtdetr"], default="auto", help="Detector architecture.")
    parser.add_argument("--simplify", action="store_true", help="Simplify detector ONNX graph when supported by Ultralytics.")
    parser.add_argument("--half", action="store_true", help="Export detector FP16 ONNX. Requires GPU export in Ultralytics.")
    parser.add_argument("--nms", action="store_true", help="Include NMS in exported detector ONNX when supported by Ultralytics.")
    parser.add_argument("--check", action="store_true", help="Run ONNX Runtime parity check after classifier export.")
    return parser.parse_args()


def convert_classifier_to_onnx(
    weights: Path,
    out: Path,
    imgsz: int | None = None,
    opset: int = 17,
    device: str = "cpu",
    dynamic_batch: bool = False,
    check: bool = False,
) -> Path:
    import torch

    checkpoint = _torch_load(weights, device)
    state_dict, metadata_model, names = _extract_checkpoint_parts(checkpoint)
    model_name = _infer_model_name(weights, metadata_model)
    num_classes = _infer_num_classes(state_dict, names)
    if isinstance(checkpoint, dict):
        imgsz = int(imgsz or checkpoint.get("imgsz") or 224)
    else:
        imgsz = int(imgsz or 224)

    model = _build_torchvision_model(model_name, num_classes)
    missing, unexpected = model.load_state_dict(_clean_state_dict(state_dict), strict=False)
    if missing and unexpected:
        raise RuntimeError(f"State dict mismatch. Missing={missing[:5]}, unexpected={unexpected[:5]}")

    device_obj = _resolve_torch_device(device)
    model.to(device_obj)
    model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device_obj)

    out = out.with_suffix(".onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
    )

    metadata = {
        "type": "torchvision_classifier",
        "source_weights": str(weights),
        "model_name": model_name,
        "classes": [names[index] for index in sorted(names)] if names else [str(index) for index in range(num_classes)],
        "imgsz": imgsz,
        "opset": opset,
        "input_name": "input",
        "output_name": "logits",
        "preprocessing": {
            "color": "RGB",
            "resize": [imgsz, imgsz],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    metadata_path = onnx_metadata_path(out)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if check:
        _check_onnx_parity(model, dummy, out)

    return out


def convert_detector_to_onnx(
    weights: Path,
    out: Path,
    imgsz: int | None = None,
    opset: int = 17,
    device: str = "cpu",
    architecture: str = "auto",
    dynamic_batch: bool = False,
    simplify: bool = False,
    half: bool = False,
    nms: bool = False,
) -> Path:
    from ultralytics import RTDETR, YOLO

    weights = Path(weights)
    source_weights = weights
    out = out.with_suffix(".onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    if architecture == "auto":
        architecture = "rtdetr" if weights.name.lower().startswith("rtdetr") else "yolo"

    with tempfile.TemporaryDirectory(prefix="onnx_export_", dir=out.parent) as temp_dir:
        export_weights = Path(temp_dir) / weights.name
        shutil.copyfile(weights, export_weights)
        model = RTDETR(str(export_weights)) if architecture == "rtdetr" else YOLO(str(export_weights))
        exported = Path(
            model.export(
                format="onnx",
                imgsz=int(imgsz or 640),
                opset=opset,
                device=device,
                dynamic=dynamic_batch,
                simplify=simplify,
                half=half,
                nms=nms,
            )
        )

        if exported.resolve() != out.resolve():
            shutil.copyfile(exported, out)

        names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        classes = {int(key): str(value) for key, value in names.items()}
    else:
        classes = {}
    metadata = {
        "type": "ultralytics_detector",
        "source_weights": str(source_weights),
        "architecture": architecture,
        "classes": classes,
        "imgsz": int(imgsz or 640),
        "opset": opset,
        "dynamic_batch": bool(dynamic_batch),
        "nms": bool(nms),
    }
    onnx_metadata_path(out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def onnx_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _check_onnx_parity(model, dummy, onnx_path: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    import torch

    with torch.no_grad():
        torch_logits = model(dummy).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_logits = session.run(None, {input_name: dummy.detach().cpu().numpy()})[0]
    max_diff = float(np.max(np.abs(torch_logits - ort_logits)))
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX parity check failed. Max abs diff: {max_diff:.6g}")
    print(f"ONNX parity check passed. Max abs diff: {max_diff:.6g}")


def main() -> int:
    args = parse_args()
    kind = args.kind
    if kind == "auto":
        try:
            out = convert_classifier_to_onnx(
                weights=args.weights,
                out=args.out,
                imgsz=args.imgsz,
                opset=args.opset,
                device=args.device,
                dynamic_batch=args.dynamic_batch,
                check=args.check,
            )
            print(f"ONNX classifier: {out}")
            print(f"Metadata: {onnx_metadata_path(out)}")
            return 0
        except Exception as exc:
            print(f"Classifier export failed, trying detector export instead: {exc}")
            kind = "detector"

    if kind == "detector":
        out = convert_detector_to_onnx(
            weights=args.weights,
            out=args.out,
            imgsz=args.imgsz,
            opset=args.opset,
            device=args.device,
            architecture=args.architecture,
            dynamic_batch=args.dynamic_batch,
            simplify=args.simplify,
            half=args.half,
            nms=args.nms,
        )
        print(f"ONNX detector: {out}")
    else:
        out = convert_classifier_to_onnx(
            weights=args.weights,
            out=args.out,
            imgsz=args.imgsz,
            opset=args.opset,
            device=args.device,
            dynamic_batch=args.dynamic_batch,
            check=args.check,
        )
        print(f"ONNX classifier: {out}")
    print(f"Metadata: {onnx_metadata_path(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
