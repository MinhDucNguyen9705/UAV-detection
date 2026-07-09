from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class OverlayRow:
    image_path: str
    label_path: str
    overlay_path: str
    scenario: str
    sample_id: str
    image_width: int
    image_height: int
    boxes: int
    max_confidence: float | None
    has_label_file: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO bbox labels onto RFUAV stress benchmark images.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Root image directory, e.g. <stress_dataset>/stress_yolo/images.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Root YOLO label directory, e.g. <prediction_run>/labels.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <labels-run>/overlays_from_labels.")
    parser.add_argument("--class-names", nargs="*", default=["uav_signal", "uav_signal"])
    parser.add_argument("--include-empty", action="store_true", help="Write overlays for images without a label file too.")
    parser.add_argument("--scenario", nargs="*", default=None, help="Optional scenario folder filter, e.g. low_snr mix2.")
    parser.add_argument("--reference-dir", type=Path, default=None, help="Only draw images whose relative image path exists under this directory.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.55)
    parser.add_argument("--hide-conf", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Do not redraw overlays that already exist.")
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def label_for_image(image: Path, images_dir: Path, labels_dir: Path) -> Path:
    rel = image.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def output_for_image(image: Path, images_dir: Path, output_dir: Path) -> Path:
    rel = image.relative_to(images_dir)
    return output_dir / rel


def find_images(source: Path, recursive: bool = True) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {source}")
        return [source.resolve()]
    iterator = source.rglob("*") if recursive else source.glob("*")
    return sorted(path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def parse_yolo_label(path: Path) -> list[tuple[int, float, float, float, float, float | None]]:
    boxes = []
    if not path.is_file():
        return boxes
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) not in {5, 6}:
            raise ValueError(f"Invalid YOLO row in {path}:{line_number}: expected 5 or 6 columns, got {len(parts)}")
        class_id = int(float(parts[0]))
        x_center, y_center, width, height = (float(value) for value in parts[1:5])
        confidence = float(parts[5]) if len(parts) == 6 else None
        boxes.append((class_id, x_center, y_center, width, height, confidence))
    return boxes


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (35, 198, 255),
        (80, 220, 110),
        (255, 170, 45),
        (215, 95, 255),
        (70, 130, 255),
    ]
    return palette[class_id % len(palette)]


def class_name_for(class_id: int, class_names: list[str]) -> str:
    return class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"


def color_for_display_class(class_id: int, class_names: list[str]) -> tuple[int, int, int]:
    display_name = class_name_for(class_id, class_names)
    for index, name in enumerate(class_names):
        if name == display_name:
            return color_for_class(index)
    return color_for_class(class_id)


def draw_boxes_cv2(
    image,
    boxes: list[tuple[int, float, float, float, float, float | None]],
    class_names: list[str],
    line_thickness: int,
    font_scale: float,
    hide_conf: bool,
) -> None:
    height, width = image.shape[:2]
    for class_id, x_center, y_center, box_w, box_h, confidence in boxes:
        x1 = int(round((x_center - box_w / 2.0) * width))
        y1 = int(round((y_center - box_h / 2.0) * height))
        x2 = int(round((x_center + box_w / 2.0) * width))
        y2 = int(round((y_center + box_h / 2.0) * height))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        color = color_for_display_class(class_id, class_names)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, max(1, line_thickness))

        class_name = class_name_for(class_id, class_names)
        label = class_name if hide_conf or confidence is None else f"{class_name} {confidence:.2f}"
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y1 = max(0, y1 - text_size[1] - baseline - 4)
        label_y2 = label_y1 + text_size[1] + baseline + 4
        label_x2 = min(width - 1, x1 + text_size[0] + 8)
        cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 4, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def draw_boxes_pillow(
    image,
    boxes: list[tuple[int, float, float, float, float, float | None]],
    class_names: list[str],
    line_thickness: int,
    font_scale: float,
    hide_conf: bool,
):
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    width, height = image.size
    font_size = max(10, int(round(font_scale * 22)))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for class_id, x_center, y_center, box_w, box_h, confidence in boxes:
        x1 = int(round((x_center - box_w / 2.0) * width))
        y1 = int(round((y_center - box_h / 2.0) * height))
        x2 = int(round((x_center + box_w / 2.0) * width))
        y2 = int(round((y_center + box_h / 2.0) * height))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        b, g, r = color_for_display_class(class_id, class_names)
        color = (r, g, b)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(1, line_thickness))

        class_name = class_name_for(class_id, class_names)
        label = class_name if hide_conf or confidence is None else f"{class_name} {confidence:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y1 = max(0, y1 - text_height - 8)
        label_y2 = label_y1 + text_height + 8
        label_x2 = min(width - 1, x1 + text_width + 8)
        draw.rectangle((x1, label_y1, label_x2, label_y2), fill=color)
        draw.text((x1 + 4, label_y1 + 4), label, fill=(0, 0, 0), font=font)

    return image


def write_summary(path: Path, rows: list[OverlayRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(OverlayRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_cv2_image(path: Path, image) -> None:
    encoded, buffer = cv2.imencode(path.suffix, image)
    if not encoded:
        raise ValueError(f"Cannot encode overlay: {path}")
    path.write_bytes(buffer.tobytes())


def reference_relative_paths(reference_dir: Path) -> set[Path]:
    return {path.relative_to(reference_dir) for path in find_images(reference_dir)}


def main() -> int:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else labels_dir.parent / "overlays_from_labels"
    summary_csv = args.summary_csv.resolve() if args.summary_csv else output_dir / "overlay_summary.csv"

    images = find_images(images_dir)
    if args.scenario:
        wanted = set(args.scenario)
        images = [image for image in images if image.relative_to(images_dir).parts[0] in wanted]
    if args.reference_dir:
        reference_dir = args.reference_dir.resolve()
        wanted_paths = reference_relative_paths(reference_dir)
        images = [image for image in images if image.relative_to(images_dir) in wanted_paths]
    if args.max_images is not None:
        images = images[: max(0, args.max_images)]

    rows: list[OverlayRow] = []
    written = 0
    skipped_empty = 0
    for image_path in images:
        label_path = label_for_image(image_path, images_dir, labels_dir)
        boxes = parse_yolo_label(label_path)
        if not boxes and not args.include_empty:
            skipped_empty += 1
            continue

        overlay_path = output_for_image(image_path, images_dir, output_dir)
        if args.skip_existing and overlay_path.is_file():
            continue

        if cv2 is not None:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            draw_boxes_cv2(image, boxes, args.class_names, args.line_thickness, args.font_scale, args.hide_conf)
            height, width = image.shape[:2]
        else:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            draw_boxes_pillow(image, boxes, args.class_names, args.line_thickness, args.font_scale, args.hide_conf)
            width, height = image.size

        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        if cv2 is not None:
            write_cv2_image(overlay_path, image)
        else:
            image.save(overlay_path)

        confidences = [confidence for *_coords, confidence in boxes if confidence is not None]
        rel_parts = image_path.relative_to(images_dir).parts
        rows.append(
            OverlayRow(
                image_path=str(image_path),
                label_path=str(label_path),
                overlay_path=str(overlay_path),
                scenario=rel_parts[0] if rel_parts else "",
                sample_id=image_path.stem,
                image_width=width,
                image_height=height,
                boxes=len(boxes),
                max_confidence=max(confidences) if confidences else None,
                has_label_file=label_path.is_file(),
            )
        )
        written += 1

    write_summary(summary_csv, rows)
    print(f"Images scanned: {len(images)}")
    print(f"Overlays written: {written}")
    print(f"Images skipped without boxes: {skipped_empty}")
    print(f"Output dir: {output_dir}")
    print(f"Summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
