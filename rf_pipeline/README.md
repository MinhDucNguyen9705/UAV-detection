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

The Gradio UI receives default weight paths from launch arguments. These paths
are only loaded when inference runs:

- Detector default argument: `--detector-weight yolo26n.pt`
- Classifier default argument: `--classifier-weight mobilenetv3_small.pt`

Example:

```bash
python demos/gradio_pipeline_app.py --share \
  --detector-weight /kaggle/working/weights/yolo26n.pt \
  --classifier-weight /kaggle/working/weights/mobilenetv3_small.pt
```

Supported classifier filenames/architectures are limited to:

- `convnexttiny.pt`
- `efficientnetb0.pt`
- `mobilenetv3_large.pt`
- `mobilenetv3_small.pt`

Classifier weights can be either Ultralytics classification checkpoints or
Torchvision checkpoints/state_dicts for those four architectures.

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

## Gradio UI Demo

```powershell
python demos\gradio_pipeline_app.py
```

On Kaggle or another remote notebook, use Gradio share mode and open the public
URL printed by Gradio:

```bash
python demos/gradio_pipeline_app.py --share
```

The UI is a step-by-step workflow: upload `.iq/.dat/.bin`, render a static
spectrogram or waterfall preview, choose either two-class detection or one-class
detection plus classification, then review and export JSON/CSV/ZIP outputs.
Inputs are shown progressively, so classifier controls only appear in
`Detection 1 class + classification`, local-path controls only appear when
`Use local file path` is selected, and waterfall controls only appear for
waterfall output.

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
- `Segment step / hop`: time shift between segment starts. Use a smaller value
  than segment duration for overlapping chunks and smoother video. The Gradio
  default is `duration=0.03s`, `hop=0.01s`.
- `Output video FPS`: playback frame rate of the stitched video. It does not
  change the IQ windows used for inference. The default is `24`.

For large IQ files, prefer `Use local file path` in Step 1 and enable `Use only
a smaller time segment`. This copies only the selected byte range into the run
folder before rendering.

On Kaggle/Linux, use forward-slash paths such as
`/kaggle/working/DJI MAVIC3 PRO/.../pack1_0-1s.iq`. The UI also normalizes
common copied Windows-style paths like `\kaggle\working\...`.
