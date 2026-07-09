# Script Huấn Luyện RFUAV

Folder này chứa các script phục vụ huấn luyện model và chuẩn bị dữ liệu sát với bước huấn luyện.

Phân chia trách nhiệm:

- `train/`: train detector, crop dữ liệu cho classifier, train/evaluate classifier.
- `stress/`: tạo benchmark stress-test và evaluate prediction.
- `rf_pipeline/`: core inference/model wrapper dùng lại trong demo và script.

## Huấn Luyện Detector Và Validate Trên Stress Benchmark

```powershell
python -m train.train_eval_ultralytics_stress `
  --train-data path\to\data.yaml `
  --benchmark-yaml-dir path\to\benchmark_yamls `
  --model yolo11n.pt `
  --project runs\ultralytics_rfuav_stress `
  --name yolo_rfuav_stress
```

Script này có thể:

- Train YOLO/RT-DETR detector bằng Ultralytics.
- Dùng checkpoint đã train để validate trên nhiều YAML benchmark.
- Ghi `stress_metrics.csv`.
- Tính false alarm riêng cho `noise_only`.

Nếu đã có weights:

```powershell
python -m train.train_eval_ultralytics_stress `
  --train-data path\to\data.yaml `
  --benchmark-yaml-dir path\to\benchmark_yamls `
  --weights path\to\best.pt `
  --skip-train
```

## Cắt YOLO Boxes Cho Classification

Tạo dataset crop dạng ImageFolder từ ảnh + YOLO labels:

```powershell
python -m train.crop_yolo_boxes_for_classification `
  --dataset-dir path\to\yolo_dataset `
  --out-dir runs\classification_crops `
  --data-yaml path\to\data.yaml `
  --splits train val test
```

Kết quả đầu ra:

```text
runs\classification_crops\
  train\
    <class_name>\
      *.jpg
  val\
    <class_name>\
      *.jpg
  test\
    <class_name>\
      *.jpg
```

## Huấn Luyện Classifier Sau Khi Crop

```powershell
python -m train.crop_yolo_boxes_for_classification `
  --dataset-dir path\to\yolo_dataset `
  --out-dir runs\classification_crops `
  --train-classifier `
  --classifier-model mobilenet_v3_small `
  --classifier-backend torchvision `
  --epochs 50
```

Các classifier hỗ trợ được định nghĩa trong `rf_pipeline.model`, gồm torchvision model và Ultralytics classification checkpoint.

## Đánh Giá Classifier Trên Benchmark Crop

```powershell
python -m train.crop_yolo_boxes_for_classification `
  --dataset-dir path\to\yolo_dataset `
  --out-dir runs\classification_crops `
  --skip-split-crop `
  --eval-benchmark-crops `
  --benchmark-yaml-dir path\to\benchmark_yamls `
  --classifier-weights path\to\classifier.pt
```

Metric gồm accuracy, macro F1, weighted F1 và precision/recall/F1 theo từng class.
