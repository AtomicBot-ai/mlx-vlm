# Copyright © 2026 Neywa Labs Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .echo import EchoWeights, extract_echo_weights
from .language import LanguageModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)
        self.echo = None

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights):
        auxiliary = extract_echo_weights(weights, self.config)
        if auxiliary:
            self.echo = EchoWeights(self.config)
        backbone = {
            k: v
            for k, v in weights.items()
            if not k.removeprefix("language_model.").startswith("echo.")
        }
        out = {
            f"language_model.{k}": v
            for k, v in self.language_model.sanitize(backbone).items()
        }
        out.update({f"echo.{k}": v for k, v in auxiliary.items()})
        return out

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    def make_cache(self):
        return self.language_model.make_cache()

    def shard(self, group=None):
        self.language_model.shard(group)
