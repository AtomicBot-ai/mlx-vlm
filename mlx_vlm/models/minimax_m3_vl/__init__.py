from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .minimax_m3_vl import Model
from .processing_minimax_m3_vl import (
    MiniMaxM3VLImageProcessor,
    MiniMaxM3VLProcessor,
    MiniMaxM3VLVideoProcessor,
    MiniMaxVLProcessor,
)
from .vision import VisionModel

__all__ = [
    "LanguageModel",
    "MiniMaxM3VLImageProcessor",
    "MiniMaxM3VLProcessor",
    "MiniMaxM3VLVideoProcessor",
    "MiniMaxVLProcessor",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]


def validate_sharding(config, world_size):
    """Complete routed experts are partitioned; attention remains replicated."""
    text = config.get("text_config") or config
    experts = text.get("num_local_experts", 128)
    if world_size < 1 or experts % world_size:
        raise ValueError(
            f"Routed expert count {experts} must be divisible by {world_size}"
        )
