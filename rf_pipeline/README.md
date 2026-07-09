# Pipeline RF

Folder này chứa core pipeline dùng lại cho demo UI và các script khác. Đây là phần xử lý/inference chính, tách khỏi các bước build dataset, gán nhãn, huấn luyện và đánh giá benchmark.

Hướng dẫn chạy demo UI Gradio được đặt ở [README.md](../README.md).

## Module Chính

- `data/`: đọc/ghi raw IQ, đọc từng segment, iterate sliding windows.
- `preprocessing/`: sinh spectrogram tĩnh và waterfall video.
- `model/`: detector wrapper, classifier wrapper, helper train classifier.
- `inference/`: orchestration end-to-end và ước lượng tham số RF.

## Output Của Pipeline

Các hàm inference có thể sinh những artifact sau, tùy cấu hình chạy:

```text
spectrogram.png
detections_overlay.png
pipeline_result.json
waterfall.mp4
waterfall_detections.mp4
segment_spectrograms\
classification_crops\
```

Ý nghĩa:

- `spectrogram.png`: ảnh phổ tĩnh.
- `detections_overlay.png`: ảnh phổ có bbox detection.
- `pipeline_result.json`: detection, classification, tham số RF và tốc độ inference.
- `waterfall.mp4`: video waterfall raw.
- `waterfall_detections.mp4`: waterfall kèm bbox detection.
- `segment_spectrograms/`: các spectrogram segment khi xử lý IQ dài.
- `classification_crops/`: crop dùng cho classifier tầng hai.

## Xử Lý IQ Dài

Với IQ dài, không nên render full-file static spectrogram vì STFT có thể dùng rất nhiều RAM. Pipeline có chế độ segment:

1. Cắt IQ thành các segment cùng duration.
2. Render mỗi segment thành một ảnh spectrogram.
3. Chạy detector theo batch trên các ảnh segment.
4. Vẽ detection lên từng frame.
5. Ghép thành raw waterfall video và detection waterfall video.

Waterfall detection overlay có thể map detection theo khoảng thời gian overlap giữa detection và frame waterfall. Cách này thường ổn định hơn chạy detector độc lập trên từng frame ngắn.

