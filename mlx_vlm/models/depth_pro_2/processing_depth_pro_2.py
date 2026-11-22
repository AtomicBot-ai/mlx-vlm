"""ImageNet normalization for channel-last Depth Pro 2 input."""

import mlx.core as mx

from ..base import install_auto_processor_patch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DepthPro2Processor:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def __call__(self, image):
        pixels = mx.array(image)
        if pixels.ndim not in (3, 4) or pixels.shape[-1] != 3 or not pixels.size:
            raise ValueError(
                "Expected a non-empty RGB image (H, W, 3) or clip (T, H, W, 3)"
            )
        if pixels.dtype == mx.uint8:
            pixels = pixels.astype(mx.float32) / 255.0
        elif mx.issubdtype(pixels.dtype, mx.floating):
            pixels = pixels.astype(mx.float32)
        else:
            raise ValueError(
                "Use uint8 RGB pixels or floating-point RGB pixels in [0, 1]"
            )
        return {
            "pixel_values": (pixels - mx.array(IMAGENET_MEAN)) / mx.array(IMAGENET_STD)
        }


install_auto_processor_patch("depth_pro_2", DepthPro2Processor)
