import mlx.nn as nn

from ..bailing_moe.bailing_moe import Model as BailingModel
from .config import ModelConfig
from .language import LanguageModel


class Model(BailingModel):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)

    def sanitize(self, weights):
        return {
            f"language_model.{key}": value
            for key, value in self.language_model.sanitize(weights).items()
        }

    def shard(self, group=None):
        self.language_model.shard(group)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
