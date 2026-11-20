from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "granite_docling_text"
    hidden_size: int = 1024
    intermediate_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = 4  # None: same as num_attention_heads
    rms_norm_eps: float = 1e-5
    vocab_size: int = 133800
    max_position_embeddings: int = 32768
    attention_bias: bool = False
    mlp_bias: bool = False
    embedding_multiplier: float = 12.0  # Scales all input embeddings
    logits_scaling: float = 4.0  # Divides the output logits
    residual_multiplier: float = 0.263  # Scales each residual branch
    attention_multiplier: float = 0.015625  # Attention scale, not 1/sqrt(head_dim)
    rope_theta: float = 10000.0
    rope_parameters: Optional[Dict[str, Union[float, str]]] = None  # Transformers 5
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        # Transformers 5 stores the RoPE base and scaling in rope_parameters.
        if self.rope_parameters:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
            if self.rope_parameters.get("rope_type", "default") != "default":
                self.rope_scaling = self.rope_parameters


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "granite_docling_vision"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 512  # Tile size in pixels
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    out_hidden_size: int = 1024  # Text hidden size that the connector projects to
    scale_factor: int = 4  # Pixel shuffle factor of the coarse connector path
    # Encoder layers (0-based) whose outputs become DeepStack features
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [3, 7, 10])
    use_fine_route: bool = True  # Build the fine path (scale_factor // 2, 4x tokens)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "granite_docling"
    image_token_id: int = 100337
    density_router_hidden_size: Optional[int] = None  # None: no density router
    density_router_threshold: float = 0.4  # Router probability for the fine path
    eos_token_id: Optional[Union[int, List[int]]] = None

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"vision_config.out_hidden_size ({self.vision_config.out_hidden_size}) "
                f"must equal text_config.hidden_size ({self.text_config.hidden_size})."
            )
        if (
            self.density_router_hidden_size is not None
            and not self.vision_config.use_fine_route
        ):
            raise ValueError(
                "The density router selects the fine connector path; it needs "
                "vision_config.use_fine_route=True."
            )
