# Bộ Tạo Dataset RFUAV

Folder này chứa phần tạo dataset cho project RFUAV. Phần này được tách riêng khỏi `rf_pipeline/` vì đây là pipeline chuẩn bị dữ liệu, không phải pipeline inference/demo.

Chức năng chính:

- Tải archive RFUAV từ Hugging Face hoặc dùng archive đã có sẵn trên máy.
- Giải nén `.tar`, `.tar.gz`, `.tgz`, `.zip`, `.rar`.
- Tìm raw IQ và XML metadata.
- Sinh spectrogram từ raw IQ.
- Ghi `raw_manifest.csv`, `samples_manifest.csv`.
- Chia tập theo `raw_file_id` hoặc `session_id` để tránh leakage.
- Hỗ trợ chế độ stream từng file IQ để tiết kiệm ổ đĩa.

## Lệnh Chính

```powershell
python -m dataset.hf_rfuav_spectrogram_manifest --help
```

Wrapper tương đương:

```powershell
python dataset\build_hf_rfuav_dataset.py --help
```

## Chạy Thử Với Một Archive

Nên chạy thử một archive nhỏ trước khi build toàn bộ RFUAV:

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

## Chế Độ Tiết Kiệm Ổ Đĩa

Với archive lớn, nên dùng stream extraction:

```powershell
python -m dataset.hf_rfuav_spectrogram_manifest `
  --repo-id kitofrank/RFUAV `
  --download-dir path\to\RFUAV_hf\downloads `
  --extract-dir path\to\RFUAV_hf\extracted `
  --out-dir path\to\RFUAV_hf\dataset_v1 `
  --include "*.tar" "*.tar.gz" "*.tgz" "*.rar" "*.zip" `
  --stream-extract-convert `
  --delete-iq-after-convert `
  --delete-archive-after-extract `
  --delete-extracted-after-convert `
  --default-sample-rate 100000000 `
  --default-center-frequency 2400000000 `
  --stft-point 16384 `
  --duration 0.03 `
  --resume
```

## Kết Quả Đầu Ra

```text
dataset_v1\
  spectrograms\
    images\
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

`raw_manifest.csv` có một dòng cho mỗi file raw IQ. File này lưu archive nguồn, đường dẫn raw/XML, metadata RF, dtype suy luận, số sample và khoảng thời gian của raw file.

`samples_manifest.csv` có một dòng cho mỗi spectrogram segment. File này lưu split chống leakage, `raw_file_id`, `session_id`, thời gian segment, cấu hình STFT, kích thước ảnh và mapping tần số.

Mặc định split theo `raw_file_id`, nên mọi spectrogram sinh từ cùng một raw IQ nằm trong cùng train/val/test. Dùng `--split-key session_id` nếu muốn tách nghiêm ngặt hơn theo phiên thu.
