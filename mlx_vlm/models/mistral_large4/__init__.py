from ..base import install_auto_processor_patch
from ..mistral3.processing_mistral3 import Mistral3Processor
from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .mistral_large4 import Model

install_auto_processor_patch("mistral_large4", Mistral3Processor)
