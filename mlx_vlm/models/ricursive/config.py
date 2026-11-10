# Copyright © 2026 Neywa Labs Inc.

from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "ricursive"
    vocab_size: int = 100352
    hidden_size: int = 5120
    intermediate_size: int = 40960
    num_hidden_layers: int = 80
    num_attention_heads: int = 80
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 2097152
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    sliding_window: Optional[int] = 1024
    layer_types: Optional[List[str]] = None
    use_gated_attention: bool = True
    gate_full_precision: bool = True
    use_valres: bool = False
    ffn_clamp_threshold: Optional[float] = None
    embed_scale: Optional[float] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 5 == 4 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have one entry per hidden layer")
        if set(self.layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError("Unsupported Ricursive layer type")
        if self.hidden_act != "silu":
            raise ValueError("Ricursive only supports the silu activation")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Attention heads must be divisible by KV heads")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be positive or None")
