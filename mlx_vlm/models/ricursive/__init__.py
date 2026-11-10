# Copyright © 2026 Neywa Labs Inc.

from transformers import PreTrainedTokenizerFast

from ..base import install_auto_processor_patch
from .config import ModelConfig
from .ricursive import Model

# This text-only checkpoint ships a standard tokenizer and a custom HF config.
# Load its tokenizer directly, without executing the upstream modeling code.
install_auto_processor_patch("ricursive", PreTrainedTokenizerFast)

__all__ = ["Model", "ModelConfig"]
