from dataclasses import dataclass
from typing import List, Optional, Union

from ..bailing_moe.config import ModelConfig as BailingConfig


@dataclass
class ModelConfig(BailingConfig):
    eos_token_id: Union[int, List[int]] = 156895
    pad_token_id: Optional[int] = 156892
    score_function: str = "sigmoid"
    head_dim: int = 128
    q_lora_rank: Optional[int] = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    layer_group_size: int = 8
    short_conv_kernel_size: int = 4
    no_kda_lora: bool = True
    kda_safe_gate: bool = True
    kda_lower_bound: float = -5.0
    rope_interleave: bool = True
    gated_attention_proj_granularity_type: Optional[str] = "head_wise"
    expert_swiglu_limit_list: Optional[List[float]] = None
    share_expert_swiglu_limit_list: Optional[List[float]] = None
    num_nextn_predict_layers: int = 0

    def is_linear(self, layer_idx: int) -> bool:
        return (
            (layer_idx + 1) % self.layer_group_size != 0
            and layer_idx
            < self.num_hidden_layers // self.layer_group_size * self.layer_group_size
        )
