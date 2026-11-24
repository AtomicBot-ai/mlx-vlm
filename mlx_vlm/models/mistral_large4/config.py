from dataclasses import InitVar, asdict, dataclass, field
from typing import Optional

from ..base import BaseModelConfig
from ..mistral3.config import ModelConfig as Mistral3Config
from ..pixtral import VisionConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "mistral_large4_text"
    hidden_size: int = 6144
    intermediate_size: int = 24576
    moe_intermediate_size: int = 6144
    num_hidden_layers: int = 72
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    rms_norm_eps: float = 1e-5
    vocab_size: int = 204800
    max_position_embeddings: int = 1048576
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    v_head_dim: int = 128
    n_routed_experts: int = 128
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 1
    n_group: int = 1
    topk_group: int = 1
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    final_logit_softcapping: float = 30.0
    rope_parameters: Optional[dict] = None
    no_rope_layers: Optional[list] = None
    kv_source_layers: Optional[list] = None
    indexer_types: Optional[list] = None
    nvfp4: bool = False
    expert_start: int = 0
    expert_end: Optional[int] = None
    mtp_config: Optional[dict] = None

    def __post_init__(self):
        n = self.num_hidden_layers
        if self.rope_parameters is None:
            self.rope_parameters = dict(
                rope_theta=10000.0,
                factor=2048.0,
                original_max_position_embeddings=512,
                beta_fast=32.0,
                beta_slow=1.0,
            )
        if self.no_rope_layers is None:
            self.no_rope_layers = [int((i + 1) % 4 != 0) for i in range(n)]
        if self.kv_source_layers is None:
            self.kv_source_layers = list(range(n))
        if self.indexer_types is None:
            self.indexer_types = ["full"] * n
        if self.expert_end is None:
            self.expert_end = self.n_routed_experts
        if not 0 <= self.expert_start < self.expert_end <= self.n_routed_experts:
            raise ValueError("Invalid expert partition")
        for name in ("no_rope_layers", "kv_source_layers", "indexer_types"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have {n} entries")
        for i, source in enumerate(self.kv_source_layers):
            if not 0 <= source <= i or self.kv_source_layers[source] != source:
                raise ValueError(f"Invalid KV source {source} for layer {i}")
            if self.no_rope_layers[source] != self.no_rope_layers[i]:
                raise ValueError("KV sharing requires the same rotary layout")


@dataclass
class ModelConfig(Mistral3Config):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "mistral_large4"
    image_token_index: int = 10
    quantization: Optional[dict] = None
    quantization_config: Optional[dict] = None
    native_config: InitVar[Optional[dict]] = None

    @classmethod
    def from_dict(cls, params):
        if "dim" in params and "n_layers" in params and not params.get("text_config"):
            params = {**params, "native_config": params}
        return super().from_dict(params)

    def __post_init__(self, native_config=None):
        super().__post_init__()
        if native_config is None:
            return

        # Native params.json describes the same architecture using Mistral's
        # names. Normalize here, before constructing modules or converting weights.
        p = native_config
        required = {
            "dim",
            "n_layers",
            "hidden_dim",
            "n_heads",
            "norm_eps",
            "vocab_size",
            "moe",
            "dsa",
        }
        missing = required - p.keys()
        if missing:
            raise ValueError(
                f"Missing native Mistral Large 4 fields: {sorted(missing)}"
            )
        text = dict(p)
        for source, target in {
            "dim": "hidden_size",
            "hidden_dim": "intermediate_size",
            "n_layers": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "n_kv_heads": "num_key_value_heads",
            "norm_eps": "rms_norm_eps",
            "softmax_tanh_gating": "final_logit_softcapping",
            "mtp": "mtp_config",
        }.items():
            if source in p:
                text[target] = p[source]
        text["model_type"] = "mistral_large4_text"
        text.setdefault("num_key_value_heads", p["n_heads"])
        for source, target in {
            "num_experts": "n_routed_experts",
            "num_shared_experts": "n_shared_experts",
            "expert_hidden_dim": "moe_intermediate_size",
            "num_experts_per_tok": "num_experts_per_tok",
            "num_expert_groups": "n_group",
            "num_expert_groups_per_tok": "topk_group",
            "routed_scale": "routed_scaling_factor",
            "first_k_dense_replace": "first_k_dense_replace",
        }.items():
            if source in p["moe"]:
                text[target] = p["moe"][source]
        dsa = p["dsa"]
        for source, target in {
            "topk": "index_topk",
            "head_dim": "index_head_dim",
            "n_heads": "index_n_heads",
        }.items():
            if source in dsa:
                text[target] = dsa[source]

        def pattern(value, choices):
            parts = value.split(",")
            if len(parts) > p["n_layers"] or any(part not in choices for part in parts):
                raise ValueError(f"Invalid Mistral layer pattern: {value!r}")
            return [parts[i % len(parts)] for i in range(p["n_layers"])]

        if p.get("nope") is not None:
            # Despite its name, no_rope_layers uses 1 for RoPE and 0 for NoPE.
            text["no_rope_layers"] = [
                int(x == "rope") for x in pattern(p["nope"], {"rope", "nope"})
            ]
        if dsa.get("sharing") is not None:
            text["indexer_types"] = pattern(dsa["sharing"], {"full", "shared"})
        if p.get("kv_sharing") is not None:
            text["kv_source_layers"] = [
                int(x) for x in p["kv_sharing"]["source_layers_str"].split(",")
            ]
        yarn = p.get("yarn") or {}
        text["rope_parameters"] = {
            "rope_theta": p.get("rope_theta", 10000.0),
            "factor": yarn.get("factor", 1.0),
            "original_max_position_embeddings": yarn.get(
                "original_max_position_embeddings",
                p.get("max_position_embeddings", 512),
            ),
            "beta_fast": yarn.get("beta", 32.0),
            "beta_slow": yarn.get("alpha", 1.0),
        }
        quantization = p.get("quantization_config") or {}
        text["nvfp4"] = quantization.get("quant_algo") == "NVFP4" or any(
            ".experts" in name and value.get("quant_algo") == "NVFP4"
            for name, value in quantization.get("quantized_layers", {}).items()
        )
        native_quant = p.get("quantization") or {}
        if "qformat_weight" in native_quant:
            if (
                native_quant["qformat_weight"] != "fp8_e4m3"
                or native_quant.get("qscheme_weight") != "BLOCK"
                or native_quant.get("block_size_out") != 128
                or native_quant.get("block_size_in") != 128
            ):
                raise ValueError(
                    "Native Mistral FP8 weights require E4M3 128x128 blocks"
                )
            # This is a source checkpoint format, not an MLX quantization recipe.
            self.quantization = None
            self.quantization_config = {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
                "scale_key": "qscale_weight",
            }
        self.text_config = TextConfig.from_dict(text)
        self.vocab_size = self.text_config.vocab_size
        vision = p.get("vision_encoder") or {}
        if vision:
            vision = {
                **vision,
                "head_dim": vision.get(
                    "head_dim", vision["hidden_size"] // vision["num_attention_heads"]
                ),
            }
            self.vision_config = VisionConfig.from_dict(vision)
            self.image_token_id = self.image_token_index = vision.get(
                "image_token_id", 10
            )
            self.spatial_merge_size = vision.get("spatial_merge_size", 2)
            self.multimodal_projector_bias = vision.get("adapter_bias", False)


def config_from_params(params: dict) -> dict:
    """Normalize native metadata before saving config.json for the shared loader."""
    return {
        key: value
        for key, value in {**params, **asdict(ModelConfig.from_dict(params))}.items()
        if value is not None
    }
