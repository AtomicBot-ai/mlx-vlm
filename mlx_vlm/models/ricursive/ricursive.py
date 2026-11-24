# Copyright © 2026 Neywa Labs Inc.

import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights):
        # Accept both upstream and already-converted MLX names on every load.
        result = {}
        for key, value in weights.items():
            if not key.startswith("language_model."):
                key = f"language_model.{key}"
            if (
                self.config.tie_word_embeddings
                and key == "language_model.lm_head.weight"
            ):
                continue
            result[key] = value
        return result

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def layers(self):
        return self.language_model.layers
