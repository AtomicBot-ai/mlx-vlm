from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import mlx.core as mx

from ...generate.edit_image import ImageEditRequest
from ...generate.image import (
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
)
from ...generate.image_defaults import ImageSamplingDefaults
from .download import SOURCE_REPO_ID, download_model
from .pipeline import MiniMaxH3ImagePipeline


@dataclass(slots=True)
class MiniMaxH3ImageGenerationModel(ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    is_image_edit_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "minimax_h3_image"
    pipeline: MiniMaxH3ImagePipeline
    model_id: str
    family: str = "minimax_h3_image"

    @property
    def default_sampling(self):
        return ImageSamplingDefaults(6, 1.0)

    @classmethod
    def resolve_defaults(cls, model, *, model_path=None):
        return ImageSamplingDefaults(6, 1.0)

    @classmethod
    def supports_model(cls, model):
        path = Path(model).expanduser()
        if not path.exists():
            return model.lower().rstrip("/") in (
                SOURCE_REPO_ID.lower(),
                "minimax-h3-image",
                "minimax_h3_image",
            )
        try:
            metadata = json.loads((path / "model_index.json").read_text())
        except (OSError, ValueError):
            return False
        return isinstance(metadata, dict) and (
            metadata.get("model_type") == cls.model_type
            or metadata.get("_class_name") == "MiniMaxH3ImagePipeline"
        )

    @classmethod
    def from_model_id(cls, model, **kwargs):
        path = kwargs.pop("model_path", None)
        revision = kwargs.pop("revision", None)
        force_download = kwargs.pop("force_download", False)
        evict = kwargs.pop("evict_text_encoder", True)
        expansion = {
            name: kwargs.pop(name)
            for name in ("upsample_prompt", "prompt_expansion_model", "evict_upsampler")
            if name in kwargs
        }
        if kwargs:
            raise TypeError(
                f"Unexpected H3-Image load arguments: {', '.join(sorted(kwargs))}"
            )
        if path is None:
            source = (
                SOURCE_REPO_ID
                if model in ("minimax-h3-image", "minimax_h3_image")
                else model
            )
            path = download_model(
                source, revision=revision, force_download=force_download
            )
        return cls(
            MiniMaxH3ImagePipeline(path, evict_text_encoder=evict, **expansion),
            str(model),
        )

    def _run(self, request, image_paths=()):
        allowed = {
            "min_generation_area",
            "max_sequence_length",
            "upsample_prompt",
            "prompt_expansion_model",
        }
        unexpected = set(request.extra) - allowed
        if unexpected:
            raise ValueError(
                f"Unsupported H3-Image options: {', '.join(sorted(unexpected))}"
            )
        seed = 0 if request.seed is None else request.seed
        steps = request.resolve_steps(6)
        guidance = request.resolve_guidance(1.0)
        width = 4096 if request.width is None else request.width
        height = 4096 if request.height is None else request.height
        array = self.pipeline.generate_array(
            request.prompt,
            image_paths=image_paths,
            seed=seed,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
            **request.extra,
        )
        return ImageGenerationResult(
            array=array,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            model=self.model_id,
            family=self.family,
            guidance=guidance,
            prompt_tokens=self.pipeline.prompt_tokens,
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata={
                "model_path": str(self.pipeline.model_path),
                "reference_images": len(image_paths),
                **getattr(self.pipeline, "last_prompt_metadata", {}),
            },
        )

    def generate(self, request: ImageGenerationRequest):
        return self._run(request)

    def edit(self, request: ImageEditRequest):
        if not request.image_paths:
            raise ValueError(
                "H3-Image editing requires one to sixteen reference images"
            )
        return self._run(request, request.image_paths)
