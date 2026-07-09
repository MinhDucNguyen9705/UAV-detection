# RFUAV UAV Signal Pipeline

Repo này tổ chức pipeline nghiên cứu cho bài toán phát hiện, phân loại và ước lượng tham số tín hiệu UAV/Drone trên ảnh phổ RF sinh từ raw IQ.

Luồng chính:

```text
Raw IQ / RFUAV archives
  -> spectrogram + manifest + leakage-safe split
  -> pseudo label / human review
  -> detector + optional crop classifier
  -> stress benchmark
  -> inference + RF parameter estimation
  -> metrics + analysis
```

## Cấu Trúc Code

```text
src\
  dataset\        Tạo dataset từ RFUAV/Hugging Face archives, sinh spectrogram, manifest, split.
  label\          Gán nhãn bán tự động bằng CFAR/threshold hoặc model-assisted pseudo label.
  train\          Train detector và chuẩn bị/train classifier từ crop YOLO boxes.
  stress\         Tạo benchmark stress-test, infer/evaluate predictions, metric nâng cao.
  rf_pipeline\    Core pipeline đọc IQ, sinh spectrogram/waterfall, detector/classifier wrapper, estimate RF params.
  demos\          Gradio UI cho pipeline end-to-end.
  weights\        Gợi ý nơi đặt model weights local.
  runs\           Output chạy thử/thực nghiệm. Được gitignore.
```

## 0. Cài Đặt Môi Trường

Clone repo và cd vào đúng thư mục project:

```powershell
git clone https://github.com/MinhDucNguyen9705/UAV-detection.git
cd UAV-detection
```

Khuyến nghị dùng Python 3.10 hoặc 3.11. Tạo môi trường bằng Conda:

```powershell
conda create -n rfuav python=3.11 -y
conda activate rfuav
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra nhanh các thư viện chính:

```powershell
python -c "import cv2, gradio, numpy, scipy, torch, torchvision, ultralytics; print('OK')"
```

Ghi chú:

- `requirements.txt` vẫn được cài bằng `pip` bên trong môi trường Conda để giữ đúng dependency của project.
- `torch` trong `requirements.txt` cài bản mặc định từ PyPI. Nếu cần CUDA cụ thể, có thể cài PyTorch bằng lệnh Conda/Pip theo hướng dẫn chính thức phù hợp GPU trước hoặc sau khi cài requirements.
- Nếu cần xuất video MP4 tương thích trình duyệt, nên cài `ffmpeg` và thêm vào `PATH`.
- Nếu xử lý archive `.rar`, ngoài package `rarfile` cần có `7z`, `unrar` hoặc công cụ giải nén tương đương trong `PATH`.
- Model weights không commit vào git. Đặt các file `.pt` trong `weights/` hoặc truyền đường dẫn qua tham số CLI/UI.

## Cấu Trúc Data Khuyến Nghị

Dataset sau khi build nên có dạng:

```text
dataset_v1\
  spectrograms\
    images\
      <class_name>\
        <sample_id>.jpg
    labels\
      <class_name>\
        <sample_id>.txt
    predictions\
      <class_name>\
        <sample_id>.jpg
  manifests\
    raw_manifest.csv
    samples_manifest.csv
    summary.json
    progress_checkpoint.json
  splits\
    train.txt
    val.txt
    test.txt
```

`raw_manifest.csv` lưu thông tin theo file IQ gốc: archive, raw path, XML path, drone/session, sample rate, center frequency, dtype, số sample.

`samples_manifest.csv` lưu thông tin theo từng spectrogram: `sample_id`, `split`, `raw_file_id`, `session_id`, thời gian cắt, STFT config, kích thước ảnh, dải tần. Split phải theo `raw_file_id` hoặc `session_id`, không split ngẫu nhiên từng ảnh để tránh leakage.

## Dataset Tạo Sẵn

| Tên | Nội dung | Link |
|---|---|---|
| RFUAV spectrogram dataset | Spectrogram + `raw_manifest.csv` + `samples_manifest.csv` + split train/val/test | https://www.kaggle.com/datasets/minhducnguyen9705/uav-spectrogram-hot |
| RFUAV pseudo-label YOLO | Ảnh spectrogram + YOLO labels sinh bán tự động | https://www.kaggle.com/datasets/minhducnguyen9705/uav-signal-dataset |
| RFUAV stress benchmark | Low-SNR, mix2, near-far, noise-only và metadata benchmark | Low-SNR, mix2, noise-only: https://drive.google.com/drive/u/1/folders/1RPS3kfxF0SO-t1AW3hD6fbzeD1Z1X-AO; Near-far: https://drive.google.com/drive/u/1/folders/1nzUVVyXHoqZOJXmbFgjksWurTVJipZU3 |
| Model weights | Detector/classifier weights dùng cho demo UI và inference | https://www.kaggle.com/models/minhducnguyen9705/uav-signal/PyTorch/default/1 |

Sau khi tải dataset tạo sẵn, kiểm tra cấu trúc thư mục khớp phần **Cấu Trúc Data Khuyến Nghị** ở trên. Nếu đặt dataset ở vị trí khác, truyền đúng đường dẫn qua các tham số như `--source`, `--manifest`, `--raw-manifest`, `--dataset-dir`.

## 1. Tạo Dataset Từ RFUAV

Xem chi tiết tại [dataset/README.md](dataset/README.md).

Smoke run với một archive nhỏ:

```powershell
python -m dataset.hf_rfuav_spectrogram_manifest `
  --repo-id kitofrank/RFUAV `
  --download-dir path\to\RFUAV_hf\downloads `
  --extract-dir path\to\RFUAV_hf\extracted `
  --out-dir path\to\RFUAV_hf\dataset_v1 `
  --archive-name "DAUTEL EVO NANO.rar" `
  --default-sample-rate 100000000 `
  --default-center-frequency 2400000000 `
  --stft-point 16384 `
  --duration 0.03 `
  --max-raw-files 2 `
  --max-segments-per-raw 20
```

Chạy tiết kiệm ổ đĩa:

```powershell
python -m dataset.hf_rfuav_spectrogram_manifest `
  --repo-id kitofrank/RFUAV `
  --download-dir path\to\RFUAV_hf\downloads `
  --extract-dir path\to\RFUAV_hf\extracted `
  --out-dir path\to\RFUAV_hf\dataset_v1 `
  --stream-extract-convert `
  --delete-iq-after-convert `
  --delete-archive-after-extract `
  --delete-extracted-after-convert `
  --stft-point 16384 `
  --duration 0.03 `
  --resume
```

## 2. Gán Nhãn Bán Tự Động

Có hai nhánh chính trong [label](label).

### CFAR/Threshold Pseudo Label

```powershell
python -m label.semi_auto_label_uav_signal `
  --manifest path\to\RFUAV_hf\dataset_v1\manifests\samples_manifest.csv `
  --out-dir runs\pseudo_labels_cfar `
  --method cfar_track `
  --export-yolo `
  --export-cvat-coco
```

Output gồm `labels/`, `overlays/`, `manifests/`, và nếu bật export thì có `yolo/` hoặc `cvat_coco/`.

### Model-Assisted Pseudo Label

Dùng detector đã train để infer nhãn YOLO:

```powershell
python -m label.infer_model_to_yolo `
  --model runs\ultralytics_rfuav_stress\yolo_rfuav_stress\weights\best.pt `
  --source path\to\RFUAV_hf\dataset_v1 `
  --split train `
  --conf 0.25 `
  --save-images
```

Với dataset builder output, labels mặc định được ghi vào:

```text
dataset_v1\spectrograms\labels
```

## 3. Train Detector Và Classifier

Xem thêm [train/README.md](train/README.md).

Train detector YOLO/RT-DETR và đánh giá nhanh trên các YAML stress:

```powershell
python -m train.train_eval_ultralytics_stress `
  --train-data path\to\data.yaml `
  --benchmark-yaml-dir path\to\benchmark_yamls `
  --model yolo26n.pt `
  --project runs\ultralytics_rfuav_stress `
  --name yolo_rfuav_stress
```

Tạo crop dataset cho classifier:

```powershell
python -m train.crop_yolo_boxes_for_classification `
  --dataset-dir path\to\yolo_dataset `
  --out-dir runs\classification_crops `
  --data-yaml path\to\data.yaml `
  --splits train val test
```

Train classifier sau khi crop:

```powershell
python -m train.crop_yolo_boxes_for_classification `
  --dataset-dir path\to\yolo_dataset `
  --out-dir runs\classification_crops `
  --train-classifier `
  --classifier-model mobilenet_v3_small `
  --classifier-backend torchvision `
  --epochs 50
```

Model registry/classifier helpers nằm trong `rf_pipeline.model`.

## 4. Tạo Stress Benchmark

Script chính:

```powershell
python -m stress.rfuav_stress_benchmark `
  --raw-manifest path\to\RFUAV_hf\dataset_v1\manifests\raw_manifest.csv `
  --out-dir path\to\RFUAV_hf\stress_v1 `
  --scenarios clean_single low_snr mix2 near_far noise_only `
  --render-spectrogram `
  --duration 0.03 `
  --stft-point 16384
```

Các scenario hiện có:

- `clean_single`: một nguồn sạch.
- `low_snr`: thêm AWGN theo SNR.
- `mix2`: trộn hai nguồn với SIR, dịch thời gian/tần số.
- `near_far`: nguồn mạnh/yếu.
- `noise_only`: ảnh không có UAV để đo false alarm.

Lưu ý: stress generator hiện tạo IQ/spectrogram/metadata. Nếu cần train/evaluate mAP đầy đủ, cần có YOLO labels/YAML tương ứng cho stress dataset.

## 5. Inference Và Evaluate Stress

Infer detector trên stress images:

```powershell
python -m stress.infer_ultralytics_stress `
  --dataset-dir path\to\stress_dataset `
  --weights path\to\best.pt `
  --mode both `
  --eval-conf 0.001 `
  --deploy-conf 0.25 `
  --save-overlays
```

Evaluate tách hai vai trò:

```powershell
python -m stress.evaluate_stress_prediction_modes `
  --dataset-dir path\to\stress_dataset `
  --eval-predictions path\to\eval_conf0p001 `
  --deploy-predictions path\to\deploy_conf0p25 `
  --output-dir runs\stress_metrics
```

Metric hiện có:

- mAP@0.5, mAP@0.5:0.95.
- precision, recall, F1.
- count accuracy, mean absolute count error.
- false alarm per noise image.
- weak/source-aware recall nếu có annotation nguồn.
- MAE của center frequency, bandwidth, duration.

## 6. Demo UI End-to-End

Repo chỉ giữ demo có giao diện Gradio:

```powershell
python demos\gradio_pipeline_app.py
```

Trên notebook remote có thể bật share link:

```powershell
python demos\gradio_pipeline_app.py --share
```

UI nhận default weight path từ launch arguments. Các weight này chỉ được load khi thực sự chạy inference:

- Detector: `--detector-weight yolo26n.pt`
- Classifier: `--classifier-weight mobilenetv3_small.pt`

Ví dụ dùng custom weights:

```powershell
python demos\gradio_pipeline_app.py `
  --detector-weight path\to\weights\yolo26n.pt `
  --classifier-weight path\to\weights\mobilenetv3_small.pt
```

Output chính gồm:

```text
spectrogram.png
detections_overlay.png
pipeline_result.json
waterfall.mp4
waterfall_detections.mp4
segment_spectrograms\
classification_crops\
```

Classifier filename/architecture đang hỗ trợ:

- `convnexttiny.pt`
- `efficientnetb0.pt`
- `mobilenetv3_large.pt`
- `mobilenetv3_small.pt`

Classifier weights có thể là Ultralytics classification checkpoint hoặc Torchvision checkpoint/state_dict cho các architecture trên.

Với IQ dài, nên dùng chế độ `Segment spectrograms -> stitched waterfall` trong UI. Chế độ này cắt IQ thành segment, render từng spectrogram, infer theo batch, vẽ detection lên frame tương ứng và ghép thành waterfall video.

Các thông số UI quan trọng:

- `STFT point`: mặc định thường dùng `16384`.
- `Colormap`: mặc định `hot`, tương thích với renderer train/stress.
- `Dynamic range`: dải dB dùng để map màu trong chế độ OpenCV preview.
- `Segment duration`: thời lượng IQ cho mỗi spectrogram/frame.
- `Segment step / hop`: bước nhảy giữa hai segment liên tiếp.
- `Output video FPS`: FPS của video xuất ra, không thay đổi cửa sổ IQ dùng inference.

Với file IQ lớn, nên dùng `Use local file path` và bật `Use only a smaller time segment` để chỉ copy đoạn byte cần xử lý vào run folder.

## 7. Core Pipeline

`rf_pipeline/` là phần lõi dùng lại giữa demo và script:

```text
rf_pipeline\
  data\iq_io.py                    Đọc/ghi IQ, đọc segment, iterate windows.
  preprocessing\spectrogram.py     IQ -> spectrogram + time/frequency mapping.
  preprocessing\waterfall.py       Waterfall video.
  model\detection.py               Ultralytics detector wrapper + heuristic detector.
  model\classification.py          Classifier inference wrapper.
  model\classification_training.py Classifier training helper.
  inference\estimation.py          Bbox -> RF parameters.
  inference\pipeline.py            End-to-end orchestration.
```

## 8. Quy Ước Thực Nghiệm

- Không split theo từng ảnh spectrogram nếu các ảnh cùng sinh từ một raw/session.
- Luôn lưu manifest và config STFT để truy vết kết quả.
- Với large IQ, ưu tiên segmented/waterfall pipeline thay vì full static spectrogram.
- Với pseudo label, nên review thủ công một subset nhỏ và dùng nó làm human-verified set.
- Khi báo cáo stress benchmark, tách metric theo SNR, SIR, scenario, số nguồn và false alarm.
