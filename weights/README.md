Optional place for local model weights.

Default detector:
- `yolo26n.pt`

Supported classifier weights:
- `convnexttiny.pt`
- `efficientnetb0.pt`
- `mobilenetv3_large.pt`
- `mobilenetv3_small.pt`

The Gradio UI does not load these files at startup. Pass defaults when launching:

```bash
python demos/gradio_pipeline_app.py \
  --detector-weight weights/yolo26n.pt \
  --classifier-weight weights/mobilenetv3_small.pt
```
