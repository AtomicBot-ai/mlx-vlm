from dataclasses import dataclass

from ..base import BaseModelConfig

# Width, depth, attention heads, and whether to use SwiGLU.
BACKBONES = {
    "vit_small_patch16_dinov3": (384, 12, 6, False),
    "vit_base_patch16_dinov3": (768, 12, 12, False),
    "vit_large_patch16_dinov3.lvd1689m": (1024, 24, 16, False),
    "vit_huge_plus_patch16_dinov3": (1280, 32, 20, True),
}


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "depth_pro_2"
    backbone_preset: str = "vit_small_patch16_dinov3"
    reassembled_blocks: tuple[int, ...] = (2, 5, 8, 11)
    cross_frame_blocks: tuple[int, ...] = (5, 7, 9, 11)
    dims_reassemble: tuple[int, ...] = (48, 96, 192, 384)
    dim_decoder: int = 64
    fov: bool = False

    def __post_init__(self):
        if self.backbone_preset not in BACKBONES:
            raise ValueError(f"Unknown Depth Pro 2 backbone: {self.backbone_preset}")
        _, depth, _, _ = BACKBONES[self.backbone_preset]
        taps = tuple(self.reassembled_blocks)
        if (
            len(taps) != 4
            or tuple(sorted(set(taps))) != taps
            or not (0 <= taps[0] < taps[-1] < depth)
        ):
            raise ValueError(
                "Depth Pro 2 requires four ordered feature taps within the backbone"
            )
        if any(not 0 <= block < depth for block in self.cross_frame_blocks):
            raise ValueError(
                "Depth Pro 2 cross-frame blocks must be within the backbone"
            )
        if len(self.dims_reassemble) != 4 or min(self.dims_reassemble) < 1:
            raise ValueError("Depth Pro 2 requires four positive feature widths")
        if self.dim_decoder < 2:
            raise ValueError("Depth Pro 2 decoder width must be at least 2")
