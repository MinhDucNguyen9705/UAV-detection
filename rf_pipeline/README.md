# RF Pipeline Refactor

This folder is a non-destructive refactor layer over the original scripts.
Old files are kept as-is; reusable logic is organized into four parts:

- `data/`: raw IQ read/write and minimal metadata.
- `preprocessing/`: IQ to static spectrogram and waterfall video.
- `model/`: detector wrappers, heuristic demo detector, optional crop classifier.
- `inference/`: end-to-end orchestration and RF parameter estimation.

## Demo

Run without trained weights using a synthetic IQ sample and heuristic detector:

```powershell
python demos\demo_pipeline.py --make-synthetic --out runs\demo_pipeline_synthetic --save-video
```

Run with a YOLO or RT-DETR detector:

```powershell
python demos\demo_pipeline.py `
  --iq path\to\sample.iq `
  --detector path\to\best.pt `
  --out runs\demo_real `
  --sample-rate 100000000 `
  --center-frequency 2400000000 `
  --dtype float32 `
  --save-video
```

Add a second-stage classifier if you use detection-one-class plus classification:

```powershell
python demos\demo_pipeline.py `
  --iq path\to\sample.iq `
  --detector path\to\detector.pt `
  --classifier path\to\classifier.pt `
  --out runs\demo_detect_classify
```

Outputs:

- `spectrogram.png`: static spectrogram.
- `waterfall.mp4`: optional sliding-window waterfall video.
- `waterfall_detections.mp4`: optional waterfall video with per-frame detection boxes.
- `detections_overlay.png`: spectrogram with detection boxes.
- `pipeline_result.json`: detection, classification, and estimated RF parameters.

Waterfall detection overlay defaults to mapping static spectrogram detections
onto each video frame by time overlap. This is usually more stable than running
the detector independently on every waterfall frame, because each waterfall
frame has a shorter time window and different visual context from the training
spectrogram.

For complete long IQ files, choose `Segment spectrograms -> stitched waterfall`
in Step 3. That mode cuts the IQ into equal-duration segments, renders each
segment as a spectrogram image, runs detector inference on those images in
batches, draws detections on the matching waterfall frame, and stitches all
frames into complete raw and detection-overlay waterfall videos. Segment images
are saved in `segment_spectrograms/`.

Avoid `Static spectrogram` on long/full IQ files. A full-file STFT can allocate
multiple GB of RAM; the UI preview therefore renders only the first few segment
frames, and the static pipeline raises a clear memory warning for large files.

Inference speed is reported as `Infer FPS`, `Infer time`, and `Infer frames` in
the UI and in `pipeline_result.json`. For `Segment spectrograms -> stitched
waterfall`, FPS is computed only over batched detector inference time, not IQ
disk IO, rendering, video encoding, or CSV/JSON export.

## UI Demo

```powershell
streamlit run demos\ui_pipeline_app.py
```

The UI is a step-by-step wizard: upload `.iq/.dat/.bin`, render a static
spectrogram or waterfall preview, choose either two-class detection or one-class
detection plus classification, then review and export JSON/CSV/ZIP outputs.

Spectrogram controls:

- Default inference preset follows the stress benchmark training scripts:
  `STFT=16384`, `colormap=hot`, Matplotlib-style rendering, and `960x720`
  output to preserve the original 4:3 spectrogram aspect ratio.
- `Dynamic range`: dB span mapped to the image color scale. Lower values boost
  contrast for strong signals; higher values keep weaker signal detail visible.
  This only affects the fast OpenCV preview mode; Matplotlib mode follows the
  RFUAV training renderer.
- `Segment duration`: duration of each IQ chunk rendered into one spectrogram
  image and one video frame.
- `Segment step / hop`: time shift between segment starts. Set it equal to
  segment duration for non-overlapping equal chunks.

For large IQ files, prefer `Use local file path` in Step 1 and enable `Use only
a smaller time segment`. This copies only the selected byte range into the run
folder before rendering. Browser upload can also be increased if needed:

```powershell
streamlit run demos\ui_pipeline_app.py --server.maxUploadSize 1024
```
