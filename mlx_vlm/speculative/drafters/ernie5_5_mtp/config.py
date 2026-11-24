# Copyright © 2026 Neywa Labs Inc.

from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.ernie5_5_moe.config import ModelConfig as TextConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "ernie5_5_mtp"
    text_config: Optional[TextConfig] = None
    # Two draft tokens plus the target bonus token.
    block_size: int = 3

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.text_config is None:
            raise ValueError("ECHO requires text_config")
        if not self.text_config.mtp_shared_last_layer:
            raise ValueError("ECHO requires mtp_shared_last_layer=true")
        if self.text_config.num_nextn_predict_layers != 1:
            raise ValueError("ECHO currently requires one shared prediction layer")
        if self.block_size < 2:
            raise ValueError("ECHO block_size includes a bonus and must be at least 2")
