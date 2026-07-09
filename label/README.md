# Gán Nhãn Bán Tự Động RFUAV

Folder này chứa các workflow tạo pseudo-label cho bài toán detection trên ảnh spectrogram.

Hiện có hai hướng:

- Xử lý tín hiệu/ảnh: CFAR, threshold theo percentile, adaptive theo hàng tần số.
- Model-assisted: dùng detector YOLO/RT-DETR đã train để infer nhãn YOLO cho tập ảnh mới.

## Nhãn Giả Bằng CFAR/Threshold

Dùng khi chưa có detector tốt hoặc muốn tạo nhãn ban đầu:

```powershell
python -m label.semi_auto_label_uav_signal `
  --manifest path\to\samples_manifest.csv `
  --out-dir runs\pseudo_labels_cfar `
  --method cfar_track `
  --export-yolo `
  --export-cvat-coco
```

Kết quả đầu ra chính:

```text
runs\pseudo_labels_cfar\
  labels\
  overlays\
  manifests\
  yolo\
  cvat_coco\
```

`overlays/` dùng để kiểm tra nhanh chất lượng bbox. `cvat_coco/` có thể dùng để đưa sang CVAT và review thủ công.

## Nhãn Giả Bằng Model

Dùng detector đã train để infer nhãn trên folder spectrogram hoặc output của dataset builder:

```powershell
python -m label.infer_model_to_yolo `
  --model runs\ultralytics_rfuav_stress\yolo_rfuav_stress\weights\best.pt `
  --source path\to\RFUAV_hf\dataset_v1 `
  --split train `
  --conf 0.25 `
  --save-images
```

Với một folder ảnh thường:

```powershell
python -m label.infer_model_to_yolo `
  --model path\to\best.pt `
  --source path\to\spectrogram_images `
  --output runs\model_pseudo_labels
```

Khi `--source` là output của dataset builder, nhãn mặc định được ghi vào:

```text
<out-dir>\spectrograms\labels
```

Ảnh visualization mặc định được ghi vào:

```text
<out-dir>\spectrograms\predictions
```

## Gợi Ý Quy Trình

1. Sinh pseudo-label ban đầu bằng `cfar_track`.
2. Review một subset nhỏ bằng CVAT.
3. Train detector baseline.
4. Dùng detector baseline infer lại pseudo-label bằng `label.infer_model_to_yolo`.
5. Lặp lại review/train nếu cần.
