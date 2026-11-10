# Copyright © 2026 Neywa Labs Inc.

from dataclasses import dataclass
from typing import Optional

from ....models.bailing_hybrid.config import ModelConfig as TextConfig
from ....models.base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "bailing_hybrid_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 2

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.text_config is None:
            raise ValueError("Ling MTP requires text_config")
        if self.text_config.num_nextn_predict_layers != 1:
            raise ValueError("Ling MTP requires one NEXTN prediction layer")
