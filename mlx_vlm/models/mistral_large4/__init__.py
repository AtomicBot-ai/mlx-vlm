from ..base import install_auto_processor_patch
from ..mistral3.processing_mistral3 import Mistral3Processor
from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .mistral_large4 import Model

install_auto_processor_patch("mistral_large4", Mistral3Processor)


def validate_sharding(config, world_size):
    """Expert parallelism permits uneven partitions, unlike tensor parallelism."""
    text = config.get("text_config") or config
    experts = text.get("n_routed_experts", TextConfig.n_routed_experts)
    if world_size < 1 or world_size > experts:
        raise ValueError(
            f"Select between 1 and {experts} workers for this expert-parallel model"
        )
