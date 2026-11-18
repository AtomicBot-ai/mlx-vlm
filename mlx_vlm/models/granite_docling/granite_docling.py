from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class DensityRouter(nn.Module):
    """Predicts from the vision features of a sample's tiles if it needs the fine path."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_size = config.vision_config.hidden_size
        self.att = nn.Linear(hidden_size, 1)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, config.density_router_hidden_size)
        self.fc2 = nn.Linear(config.density_router_hidden_size, 1)

    def __call__(self, hidden_states: mx.array, tiles_per_sample: list) -> mx.array:
        # Attention-pool the patches of every tile, then average the tiles of every sample.
        weights = mx.softmax(self.att(hidden_states), axis=1)
        tile_features = (hidden_states * weights).sum(axis=1)
        offsets = np.cumsum([0] + list(tiles_per_sample))
        sample_features = mx.stack(
            [
                tile_features[int(s) : int(e)].mean(axis=0)
                for s, e in zip(offsets[:-1], offsets[1:])
            ]
        )
        hidden = nn.gelu(self.fc1(self.norm(sample_features)))
        return self.fc2(hidden).squeeze(-1)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        if config.density_router_hidden_size is not None:
            self.density_router = DensityRouter(config)

    @staticmethod
    def _real_tiles(pixel_values: mx.array, pixel_attention_mask: Optional[mx.array]):
        """Flatten (batch, tiles, C, H, W) to the real tiles, channels last."""
        if pixel_values.ndim == 4:
            pixel_values = pixel_values[None]
        batch_size, num_tiles = pixel_values.shape[:2]
        if pixel_attention_mask is None:
            tile_mask = np.ones((batch_size, num_tiles), dtype=bool)
        else:
            tile_mask = np.array(pixel_attention_mask).astype(bool)
            if tile_mask.ndim > 2:
                # Per-pixel masks: a tile is real if any of its pixels is.
                tile_mask = tile_mask.reshape(batch_size, num_tiles, -1).any(-1)
        flat = pixel_values.reshape(batch_size * num_tiles, *pixel_values.shape[2:])
        tiles = flat[mx.array(np.flatnonzero(tile_mask), dtype=mx.uint32)]
        return tiles.transpose(0, 2, 3, 1), tile_mask.sum(-1).tolist()

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        tiles, _ = self._real_tiles(pixel_values, kwargs.get("pixel_attention_mask"))
        # The processor returns tile_fine_mask only when it was called with fine_route=True.
        fine = kwargs.get("tile_fine_mask") is not None
        image_features, deepstack_features = self.vision_model(tiles, fine=fine)

        image_mask = input_ids == self.config.image_token_id
        mask_np = np.array(image_mask)
        n_image_tokens = int(mask_np.sum())
        if n_image_tokens != image_features.shape[0]:
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )
        positions = mx.array(np.flatnonzero(mask_np), dtype=mx.uint32)
        B, L, D = inputs_embeds.shape
        flat = inputs_embeds.reshape(B * L, D)
        flat[positions] = image_features.astype(flat.dtype)
        inputs_embeds = flat.reshape(B, L, D)

        deepstack_visual_embeds = [
            f.astype(inputs_embeds.dtype) for f in deepstack_features
        ]
        if not self.training:
            mx.eval(deepstack_visual_embeds)
        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def predict_fine_route(
        self, pixel_values: mx.array, pixel_attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """True for the samples that need the fine connector path.

        Route them by calling the processor again with fine_route=True.
        """
        if "density_router" not in self:
            raise ValueError("This checkpoint has no density router.")
        tiles, tiles_per_sample = self._real_tiles(pixel_values, pixel_attention_mask)
        hidden_states, _ = self.vision_model.encode(
            tiles.astype(self.vision_model.embeddings.patch_embedding.weight.dtype)
        )
        logits = self.density_router(hidden_states, tiles_per_sample)
        return mx.sigmoid(logits) >= self.config.density_router_threshold

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            mask=mask,
            cache=cache,
            visual_pos_masks=features.visual_pos_masks,
            deepstack_visual_embeds=features.deepstack_visual_embeds,
            logits_to_keep=kwargs.get("logits_to_keep", 0),
        )

    def sanitize(self, weights):
        prefixes = (
            ("model.text_model.", "language_model.model."),
            ("model.vision_model.", "vision_model."),
            ("model.density_router.", "density_router."),
            ("lm_head.", "language_model.lm_head."),
        )
        sanitized = {}
        for k, v in weights.items():
            # Multi-token prediction heads for serving engines; not part of this model.
            if k.startswith("mtp."):
                continue
            for source, target in prefixes:
                if k.startswith(source):
                    k = target + k[len(source) :]
                    break
            if (
                k.startswith("language_model.lm_head.")
                and self.config.text_config.tie_word_embeddings
            ):
                continue
            sanitized[k] = v
        return sanitized
