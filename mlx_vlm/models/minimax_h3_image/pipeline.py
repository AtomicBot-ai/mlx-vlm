from __future__ import annotations

import gc
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from ..minimax_h3.packing import patchify_video_latents, unpatchify_video_tokens
from ..minimax_h3.processing import decode_image, process_qwen_images, resize_lanczos
from .packing import (
    bound_short_edge,
    build_image_layout,
    canvas_size,
    sampling_schedule,
)
from .upsampler import DEFAULT_UPSAMPLER, MiniMaxH3PromptUpsampler
from .weights import load_conditioner, load_transformer, load_vae

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


class MiniMaxH3ImagePipeline:
    def __init__(
        self,
        model_path,
        *,
        evict_text_encoder=True,
        upsample_prompt=True,
        prompt_expansion_model=DEFAULT_UPSAMPLER,
        evict_upsampler=True,
    ):
        self.model_path = Path(model_path)
        self.evict_text_encoder = evict_text_encoder
        self._conditioner = self._transformer = self._vae = None
        self._adaln_caches = None
        self.prompt_tokens = 0
        self.upsample_prompt = upsample_prompt
        self.prompt_expansion_model = str(prompt_expansion_model or DEFAULT_UPSAMPLER)
        self.evict_upsampler = evict_upsampler
        self.upsampler = None
        self.upsampled_prompts = {}
        self.last_prompt_metadata = {}

    def _prepare_prompt(
        self,
        prompt,
        images,
        *,
        width,
        height,
        seed,
        upsample_prompt=None,
        prompt_expansion_model=None,
    ):
        enabled = self.upsample_prompt if upsample_prompt is None else upsample_prompt
        self.last_prompt_metadata = {"input_prompt": prompt, "upsampled": False}
        if not enabled:
            return prompt
        model_id = str(prompt_expansion_model or self.prompt_expansion_model)
        divisor = math.gcd(width, height)
        ratio = f"{width // divisor}:{height // divisor}"
        key = (prompt, model_id, ratio, seed)
        expansion = self.upsampled_prompts.get(key) if not images else None
        if expansion is None:
            if self.evict_upsampler:
                # A new transformer has a new AdaLN cache owner. Invalidate both
                # together when staging PE ahead of the diffusion components.
                self._conditioner = self._transformer = self._vae = None
                self._adaln_caches = None
                gc.collect()
                mx.clear_cache()
            if self.upsampler is None or self.upsampler.model_id != model_id:
                if self.upsampler is not None:
                    self.upsampler.unload()
                self.upsampler = MiniMaxH3PromptUpsampler(
                    model_id, evict_model=self.evict_upsampler
                )
            references = [Image.fromarray(np.array(image)) for image in images]
            expansion = self.upsampler(
                prompt,
                references,
                aspect_ratio=ratio if images else "adaptive",
                seed=seed,
            )
            if not images:
                if len(self.upsampled_prompts) >= 4:
                    self.upsampled_prompts.pop(next(iter(self.upsampled_prompts)))
                self.upsampled_prompts[key] = expansion
        self.last_prompt_metadata.update(
            upsampled=True,
            revised_prompt=expansion.expanded_caption,
            prompt_expansion_model=model_id,
            suggested_aspect_ratio=expansion.aspect_ratio,
        )
        return expansion.expanded_caption

    @property
    def vae(self):
        if self._vae is None:
            self._vae = load_vae(self.model_path)
        return self._vae

    @property
    def transformer(self):
        if self._transformer is None:
            self._transformer = load_transformer(self.model_path)
        return self._transformer

    def _encode_prompt(self, prompt, images, max_sequence_length):
        if self._conditioner is None:
            self._conditioner = load_conditioner(self.model_path)
        conditioner = self._conditioner
        tokenizer = conditioner.tokenizer
        pixels = grid = None
        ids, tags = [], []
        if images:
            vision_images = []
            for image in images:
                width, height = bound_short_edge(
                    image.shape[1], image.shape[0], 1024, 1024
                )
                vision_images.append(resize_lanczos(image, height, width))
            pixels, grid = process_qwen_images(vision_images)
            for index, shape in enumerate(grid.tolist()):
                label = tokenizer.encode(
                    f"Picture {index + 1}: ", add_special_tokens=False
                )[:max_sequence_length]
                vision = (
                    [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [tokenizer.convert_tokens_to_ids("<|image_pad|>")]
                    * (shape[0] * shape[1] * shape[2] // 4)
                    + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                ids.extend(label + vision)
                tags.extend([1] * len(label) + [0] * len(vision))
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[
            :max_sequence_length
        ]
        ids.extend(prompt_ids)
        tags.extend([1] * len(prompt_ids))
        if not ids:
            raise ValueError("Prompt must encode to at least one token")
        output = conditioner._encode(
            ids, tags, pixel_values=pixels, image_grid_thw=grid
        )
        hidden = output.hidden_states
        mx.eval(hidden)
        self.prompt_tokens = len(ids)
        if self.evict_text_encoder:
            self._conditioner = None
            del conditioner, output
            gc.collect()
            mx.clear_cache()
        return hidden, mx.array(tags, mx.int32)

    def generate_array(
        self,
        prompt,
        *,
        image_paths=(),
        seed=0,
        steps=6,
        guidance=1.0,
        width=4096,
        height=4096,
        min_generation_area=4194304,
        max_sequence_length=4096,
        upsample_prompt=None,
        prompt_expansion_model=None,
    ):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("H3-Image requires a non-empty prompt")
        if steps != 6 or guidance != 1:
            raise ValueError(
                "H3-Image is distilled for exactly six steps and guidance 1"
            )
        if not 1 <= max_sequence_length <= 4096:
            raise ValueError("max_sequence_length must be between 1 and 4096")
        if len(image_paths) > 16:
            raise ValueError("H3-Image supports at most 16 reference images")
        canvas_width, canvas_height = canvas_size(width, height, min_generation_area)
        images = [decode_image(image) for image in image_paths]
        # Validate references before loading the large conditioner.
        for image in images:
            bound_short_edge(image.shape[1], image.shape[0], 1024, 2048)
        prompt = self._prepare_prompt(
            prompt,
            images,
            width=width,
            height=height,
            seed=seed,
            upsample_prompt=upsample_prompt,
            prompt_expansion_model=prompt_expansion_model,
        )
        hidden, text_tags = self._encode_prompt(prompt, images, max_sequence_length)
        key = mx.random.key(seed)

        def random_key():
            nonlocal key
            key, result = mx.random.split(key)
            return result

        reference_latents = []
        if images:
            vae = self.vae
            ratio = vae.config.spatial_compression_ratio
            mean = mx.array(vae.config.latents_mean).reshape(1, -1, 1, 1)
            std = mx.array(vae.config.latents_std).reshape(1, -1, 1, 1)
            for image in images:
                w, h = bound_short_edge(image.shape[1], image.shape[0], 1024, 2048)
                image = resize_lanczos(image, h, w)[
                    : h // ratio * ratio, : w // ratio * ratio
                ]
                pixels = image.astype(mx.float32) / 255
                pixels = (
                    (pixels - mx.array(PIXEL_MEAN)) / mx.array(PIXEL_STD)
                ).transpose(2, 0, 1)[None]
                z = (
                    vae.encode(pixels)
                    .sample(key=random_key())
                    .astype(mx.float16)
                    .astype(mx.float32)
                )
                z = z[..., : z.shape[-2] // 2 * 2, : z.shape[-1] // 2 * 2]
                z = (z - mean) / std
                mx.eval(z)
                reference_latents.append(z)
        transformer = self.transformer
        config = transformer.config
        # The released image VAE compresses each spatial axis by sixteen.
        lh, lw = canvas_height // 16, canvas_width // 16
        layout = build_image_layout(
            text_tags,
            [z.shape[-2:] for z in reference_latents],
            lh,
            lw,
            config.patch_size,
        )
        positions, tags, image_indices, text_indices, reference_rows = layout
        noise = mx.random.normal((1, config.in_channels, 1, lh, lw), key=random_key())
        target = patchify_video_latents(noise, config.patch_size)
        references = []
        for z in reference_latents:
            noise = mx.random.normal(z.shape, key=random_key())
            t = mx.array(0.999, mx.float32)
            references.append(
                patchify_video_latents(
                    (t * z + (1 - t) * noise)[:, :, None], config.patch_size
                )
            )
        rows = mx.concatenate([*references, target], axis=0)
        timesteps, sigmas = sampling_schedule()
        tables = [mx.stack([t, mx.array(0.999, mx.float32)]) for t in timesteps]
        if self._adaln_caches is None:
            # The schedule never changes. Materialize the tiny timestep tables once,
            # then release ~26 GB of raw AdaLN projection weights.
            self._adaln_caches = [
                transformer.build_adaln_cache(table) for table in tables
            ]
            transformer.drop_adaln_weights()
            gc.collect()
            mx.clear_cache()
        row_indices = mx.zeros((tags.shape[0],), mx.int32)
        if reference_rows:
            row_indices = row_indices.at[image_indices[:reference_rows]].add(1)
        for step in range(6):
            velocity = transformer(
                rows[None],
                hidden,
                tables[step],
                row_indices,
                tags,
                positions,
                image_indices,
                text_indices,
                step,
                self._adaln_caches[step],
            )[0]
            target = rows[reference_rows:] + (
                sigmas[step] - sigmas[step + 1]
            ) * velocity[reference_rows:].astype(mx.float32)
            rows = mx.concatenate([rows[:reference_rows], target], axis=0)
            mx.eval(rows)
        latents = unpatchify_video_tokens(
            rows[reference_rows:], 1, lh, lw, config.in_channels, config.patch_size
        )[:, :, 0]
        vae = self.vae
        mean = mx.array(vae.config.latents_mean).reshape(1, -1, 1, 1)
        std = mx.array(vae.config.latents_std).reshape(1, -1, 1, 1)
        decoded = vae.decode(latents * std + mean).sample[0].transpose(1, 2, 0)
        image = mx.clip(
            (decoded.astype(mx.float32) * mx.array(PIXEL_STD) + mx.array(PIXEL_MEAN))
            * 255,
            0,
            255,
        ).astype(mx.uint8)
        image = resize_lanczos(image, height, width)
        mx.eval(image)
        return image
