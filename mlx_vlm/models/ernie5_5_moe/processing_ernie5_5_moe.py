# Copyright © 2026 Neywa Labs Inc.

from transformers import PreTrainedTokenizerFast

from ..base import install_auto_processor_patch

# This is a base language model: preserve its tokenizer and plain completion
# prompts without inventing an instruction/chat template.
install_auto_processor_patch("ernie5_5_moe", PreTrainedTokenizerFast)
