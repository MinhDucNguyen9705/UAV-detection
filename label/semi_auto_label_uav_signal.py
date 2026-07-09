from __future__ import annotations

import argparse
from pathlib import Path

from .semi_auto import SemiAutoLabelConfig, run_semi_auto_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo YOLO labels for UAV signal spectrograms.")
    parser.add_argument("--manifest", type=Path, default=None, help="Input samples_manifest.csv.")
    parser.add_argument("--image-root", type=Path, default=None, help="Input image folder if no manifest is available.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", default="cfar_track", choices=["percentile", "adaptive_row", "cfar_track"])
    parser.add_argument("--percentile", type=float, default=94.0)
    parser.add_argument("--row-margin", type=float, default=35.0, help="Intensity margin for adaptive_row.")
    parser.add_argument("--cfar-percentile", type=float, default=60.0, help="Noise floor percentile per frequency row.")
    parser.add_argument("--cfar-margin", type=float, default=28.0, help="Detection margin above row noise floor.")
    parser.add_argument("--cfar-time-smooth", type=int, default=31, help="Median smoothing along time; odd number.")
    parser.add_argument("--fallback-percentile", type=float, default=0.0, help="Optional global bright-pixel fallback percentile. 0 disables it.")
    parser.add_argument("--time-dilate", type=int, default=1, help="Dilate detections along time before tracking.")
    parser.add_argument("--freq-dilate", type=int, default=1, help="Dilate detections along frequency before tracking.")
    parser.add_argument("--band-min-height", type=int, default=4, help="Minimum frequency-band height in one time frame.")
    parser.add_argument("--band-merge-gap", type=int, default=3, help="Merge frequency bands separated by <= this many pixels.")
    parser.add_argument("--max-band-height-ratio", type=float, default=0.75, help="Ignore single-frame bands taller than this image-height ratio.")
    parser.add_argument("--track-overlap", type=float, default=0.2, help="Minimum frequency overlap ratio to continue a track.")
    parser.add_argument("--track-gap", type=int, default=3, help="Max missing time frames before a track is closed.")
    parser.add_argument("--track-min-frames", type=int, default=2, help="Minimum active frames for a valid track.")
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--morph-kernel", type=int, default=11)
    parser.add_argument("--morph-iterations", type=int, default=1)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.85)
    parser.add_argument("--max-box-height-ratio", type=float, default=0.95)
    parser.add_argument("--max-box-width-ratio", type=float, default=0.95)
    parser.add_argument("--ignore-edge-ratio", type=float, default=0.0, help="Ignore small boxes near top/bottom image edges. 0 disables.")
    parser.add_argument("--edge-small-height-ratio", type=float, default=0.08, help="Only edge boxes shorter than this are ignored.")
    parser.add_argument("--min-width", type=int, default=4)
    parser.add_argument("--min-height", type=int, default=4)
    parser.add_argument("--max-boxes", type=int, default=12)
    parser.add_argument("--pad-ratio", type=float, default=0.0)
    parser.add_argument("--tighten-boxes", action="store_true", help="Shrink each bbox to bright pixels inside the component box.")
    parser.add_argument("--tighten-percentile", type=float, default=85.0)
    parser.add_argument("--tighten-min-pixels", type=int, default=8)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--class-name", default="uav_signal")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--balanced-by-class", action="store_true", help="Select up to --max-images with a near-even count per class.")
    parser.add_argument("--class-column", default="class_name", help="Manifest column used for balanced sampling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-yolo", action="store_true")
    parser.add_argument("--export-cvat-coco", action="store_true", help="Export a CVAT-friendly COCO dataset.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.manifest is None and args.image_root is None:
        parser.error("Provide --manifest or --image-root.")
    return args


def config_from_args(args: argparse.Namespace) -> SemiAutoLabelConfig:
    return SemiAutoLabelConfig(
        manifest=args.manifest,
        image_root=args.image_root,
        out_dir=args.out_dir,
        method=args.method,
        percentile=args.percentile,
        row_margin=args.row_margin,
        cfar_percentile=args.cfar_percentile,
        cfar_margin=args.cfar_margin,
        cfar_time_smooth=args.cfar_time_smooth,
        fallback_percentile=args.fallback_percentile,
        time_dilate=args.time_dilate,
        freq_dilate=args.freq_dilate,
        band_min_height=args.band_min_height,
        band_merge_gap=args.band_merge_gap,
        max_band_height_ratio=args.max_band_height_ratio,
        track_overlap=args.track_overlap,
        track_gap=args.track_gap,
        track_min_frames=args.track_min_frames,
        blur_kernel=args.blur_kernel,
        morph_kernel=args.morph_kernel,
        morph_iterations=args.morph_iterations,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        max_box_height_ratio=args.max_box_height_ratio,
        max_box_width_ratio=args.max_box_width_ratio,
        ignore_edge_ratio=args.ignore_edge_ratio,
        edge_small_height_ratio=args.edge_small_height_ratio,
        min_width=args.min_width,
        min_height=args.min_height,
        max_boxes=args.max_boxes,
        pad_ratio=args.pad_ratio,
        tighten_boxes=args.tighten_boxes,
        tighten_percentile=args.tighten_percentile,
        tighten_min_pixels=args.tighten_min_pixels,
        class_id=args.class_id,
        class_name=args.class_name,
        max_images=args.max_images,
        random_sample=args.random_sample,
        balanced_by_class=args.balanced_by_class,
        class_column=args.class_column,
        seed=args.seed,
        export_yolo=args.export_yolo,
        export_cvat_coco=args.export_cvat_coco,
        overwrite=args.overwrite,
    )


def main() -> int:
    config = config_from_args(parse_args())
    stats = run_semi_auto_label(config)
    print(f"Done. Images: {len(stats)}. Boxes: {sum(item.num_boxes for item in stats)}")
    print(f"Overlays: {config.out_dir / 'overlays'}")
    print(f"Labels: {config.out_dir / 'labels'}")
    if config.export_cvat_coco:
        print(f"CVAT COCO: {config.out_dir / 'cvat_coco' / 'annotations' / 'instances_default.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
