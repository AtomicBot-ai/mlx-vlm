# Copyright © 2026 Neywa Labs Inc.

from ....models.bailing_hybrid.config import ModelConfig
from ....models.bailing_hybrid.language import sanitize_weights
from ..mtp_split import MTPSplitter


class BailingHybridMTPSplitter(MTPSplitter):
    output_model_type = "bailing_hybrid_mtp"

    def select_keys(self, key, text_config):
        return key.removeprefix("language_model.").startswith(
            f"model.layers.{text_config['num_hidden_layers']}."
        )

    def rename(self, tensors, text_config):
        prefix = f"model.layers.{text_config['num_hidden_layers']}."
        return {
            "mtp." + key.removeprefix("language_model.").removeprefix(prefix): value
            for key, value in tensors.items()
        }

    def run_sanitize(self, tensors, text_config):
        return sanitize_weights(
            tensors, ModelConfig.from_dict(text_config), [("mtp", False)]
        )

    def should_quantize_key(self, key):
        return super().should_quantize_key(key) and not key.endswith(
            ("attention.g_proj.weight", "eh_proj.weight")
        )

    def quantization_from_source(self, tensors, source_config):
        if any(key.endswith(".scales") for key in tensors):
            return source_config.get("quantization")
        return None
