# Depth Pro 2

Depth Pro 2 predicts per-pixel metric depth for photos, video, and live RGB
frames. The converted MLX checkpoints include Apple's shared focal-length
estimator and load through `mlx_vlm.load`.

## Supported models

| Variant | MLX checkpoint | Source weights |
| --- | --- | --- |
| S | [nativ-community/DepthPro2-S](https://huggingface.co/nativ-community/DepthPro2-S) | [apple/DepthPro2-S](https://huggingface.co/apple/DepthPro2-S) |
| B | [nativ-community/DepthPro2-B](https://huggingface.co/nativ-community/DepthPro2-B) | [apple/DepthPro2-B](https://huggingface.co/apple/DepthPro2-B) |
| L | [nativ-community/DepthPro2-L](https://huggingface.co/nativ-community/DepthPro2-L) | [apple/DepthPro2-L](https://huggingface.co/apple/DepthPro2-L) |
| H+ | [nativ-community/DepthPro2-H](https://huggingface.co/nativ-community/DepthPro2-H) | [apple/DepthPro2-H](https://huggingface.co/apple/DepthPro2-H) |

## Photo or short clip

```python
from mlx_vlm import load
from mlx_vlm.models.depth_pro_2.generate import DepthPro2Predictor, read_image

model, processor = load("nativ-community/DepthPro2-S")  # Or a local converted folder.
predictor = DepthPro2Predictor(model, processor)

result = predictor.infer(read_image("image.jpg"))
depth_m = result["depth"]  # (H, W), depth along the camera's viewing axis
focal_px = result["focallength_px"]
```

Pass a `(T, H, W, 3)` array to `infer` to attend jointly over a short clip.
Inputs are channel-last RGB in uint8 or floating point in [0, 1]. Depth maps
keep the input resolution. Pixels with zero inverse depth have infinite depth.

Pass a calibrated focal length in pixels as `f_px` when available. The predictor
uses the bundled [focal estimator](https://huggingface.co/apple/DepthPro2-focal)
when you omit it.

## Video and live frames

`infer_video` takes an iterable of same-size RGB frames and yields one result per
frame. Buffered mode uses overlapping clips with an inverse-depth crossfade.
When a focal length is omitted, video inference estimates a single focal length
from the first window.

```python
from mlx_vlm.models.video_depth_anything.generate import read_video_frames

frames, fps = read_video_frames("input.mp4")
for result in predictor.infer_video(frames, window=16, overlap=4):
    depth_m = result["depth"]
```

For a live source, create a stream with
`predictor.stream(window=8, mode="unbuffered")` and pass each RGB frame to
`stream.step(frame)`. Unbuffered mode returns a prediction immediately using
a rolling attention cache. Buffered mode may return zero or more predictions
per step. Call `stream.flush()` at the end to retrieve any remaining predictions.

Apple distributes the reference code under its [software license](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE)
and the weights under the [Apple Machine Learning Research Model license](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE_MODEL).
