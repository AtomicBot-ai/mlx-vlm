"""Granite for Docling page tiling and prompt expansion on the PIL backend."""

from functools import lru_cache
from itertools import accumulate

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_backends import PilBackend
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    infer_channel_dimension_format,
    is_valid_image,
    make_nested_list_of_images,
)
from transformers.processing_utils import ImagesKwargs, MultiModalData, ProcessorMixin

from ..base import install_auto_processor_patch


class GraniteDoclingImageKwargs(ImagesKwargs, total=False):
    crop_to_patches: bool
    min_patches: int
    max_patches: int
    dense_max_patches: int
    dense_max_scale: float
    dense_min_aspect: float
    dense_max_aspect: float
    fine_route: bool


@lru_cache(maxsize=32)
def _tile_grids(minimum, maximum):
    # The checkpoint has row/column markers for positions 1 through 16.
    sides = range(1, min(16, maximum) + 1)
    return sorted(
        (
            (rows, cols)
            for cols in sides
            for rows in sides
            if minimum <= rows * cols <= maximum
        ),
        key=lambda grid: grid[0] * grid[1],
    )


def _closest_grid(height, width, size, minimum, maximum):
    best, distance = (1, 1), float("inf")
    tile_area = size["height"] * size["width"]
    for rows, cols in _tile_grids(minimum, maximum):
        delta = abs(width / height - cols / rows)
        if delta < distance or (
            delta == distance and 2 * width * height > rows * cols * tile_area
        ):
            best, distance = (rows, cols), delta
    return best


class GraniteDoclingImageProcessor(PilBackend):
    valid_kwargs = GraniteDoclingImageKwargs
    model_input_names = ["pixel_values", "pixel_attention_mask"]
    size = {"height": 512, "width": 512}
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = do_resize = do_rescale = do_normalize = True
    crop_to_patches = True
    min_patches = 1
    max_patches = 32
    dense_max_patches = 0
    dense_max_scale = dense_min_aspect = dense_max_aspect = 0.0
    fine_route = False

    def process_image(
        self, image, do_convert_rgb=None, input_data_format=None, **kwargs
    ):
        if not do_convert_rgb:
            return super().process_image(image, False, input_data_format, **kwargs)
        # Transformers 5.14 drops alpha. Docling uses a white page background.
        if isinstance(image, Image.Image):
            if "A" in image.getbands() or image.info.get("transparency") is not None:
                image = Image.alpha_composite(
                    Image.new("RGBA", image.size, "white"), image.convert("RGBA")
                )
            return np.asarray(image.convert("RGB")).transpose(2, 0, 1)
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[None]
        else:
            layout = input_data_format or infer_channel_dimension_format(
                image, num_channels=(1, 3, 4)
            )
            if layout == ChannelDimension.LAST:
                image = np.moveaxis(image, -1, 0)
        if image.shape[0] == 1:
            return np.repeat(image, 3, axis=0)
        if image.shape[0] == 4:
            integer = np.issubdtype(image.dtype, np.integer)
            if not integer and (image.min() < 0 or image.max() > 1):
                raise ValueError("Floating-point RGBA channels must be in [0, 1].")
            white = 255 if integer else 1.0
            alpha = image[3:] / white
            rgb = alpha * image[:3] + (1 - alpha) * white
            return (np.rint(rgb) if integer else rgb).astype(image.dtype)
        return image

    def _prepare_images_structure(self, images, expected_ndims=3):
        return make_nested_list_of_images(
            self.fetch_images(images), expected_ndims=expected_ndims
        )

    def get_tile_grid(self, height, width, images_kwargs=None):
        options = images_kwargs or {}
        size = options.get("size", self.size)
        minimum = options.get("min_patches", self.min_patches)
        maximum = options.get("max_patches", self.max_patches)
        if not options.get("crop_to_patches", self.crop_to_patches) or maximum <= 1:
            return 1, 1
        grid = _closest_grid(height, width, size, minimum, maximum)
        dense = options.get("dense_max_patches", self.dense_max_patches)
        scale = options.get("dense_max_scale", self.dense_max_scale)
        lower = options.get("dense_min_aspect", self.dense_min_aspect)
        upper = options.get("dense_max_aspect", self.dense_max_aspect)
        aspect = width / height
        if (
            dense
            and (scale > 0 or lower > 0)
            and (lower <= 0 or aspect > lower)
            and (upper <= 0 or aspect <= upper)
            and (scale <= 0 or grid[1] * size["width"] / width < scale)
        ):
            return _closest_grid(height, width, size, minimum, dense)
        return grid

    def get_number_of_image_patches(self, height, width, images_kwargs=None):
        rows, cols = self.get_tile_grid(height, width, images_kwargs)
        return rows * cols + int(rows * cols > 1)

    def _tiles(self, image, rows, cols, size, resample):
        height, width = size.height, size.width
        canvas = self.resize(
            image, SizeDict(height=rows * height, width=cols * width), resample
        )
        for row in range(rows):
            for col in range(cols):
                yield canvas[
                    :,
                    row * height : (row + 1) * height,
                    col * width : (col + 1) * width,
                ]
        if rows * cols > 1:
            yield self.resize(image, size, resample)

    def _preprocess(
        self,
        images,
        size,
        resample,
        do_rescale,
        rescale_factor,
        do_normalize,
        image_mean,
        image_std,
        fine_route,
        return_tensors,
        **kwargs,
    ):
        grids = [
            [
                self.get_tile_grid(*page.shape[-2:], {**kwargs, "size": size})
                for page in sample
            ]
            for sample in images
        ]
        tile_counts = [
            sum(r * c + int(r * c > 1) for r, c in sample) for sample in grids
        ]
        pixels = np.zeros(
            (len(images), max(tile_counts), 3, size.height, size.width),
            dtype=np.float32,
        )
        for batch_idx, (sample, sample_grids) in enumerate(zip(images, grids)):
            index = 0
            for page, (rows, cols) in zip(sample, sample_grids):
                for tile in self._tiles(page, rows, cols, size, resample):
                    if do_rescale:
                        tile = self.rescale(tile, rescale_factor)
                    if do_normalize:
                        tile = self.normalize(tile, image_mean, image_std)
                    pixels[batch_idx, index] = tile
                    index += 1
        data = {
            "pixel_values": pixels,
            "pixel_attention_mask": np.arange(pixels.shape[1])
            < np.array(tile_counts)[:, None],
            "rows": [[r for r, c in sample] for sample in grids],
            "cols": [[c for r, c in sample] for sample in grids],
        }
        if fine_route:
            data["tile_fine_mask"] = np.ones(len(images), dtype=bool)
        return BatchFeature(
            data, tensor_type=return_tensors, skip_tensor_conversion=["rows", "cols"]
        )


class GraniteDoclingProcessor(ProcessorMixin):
    unused_input_names = ["rows", "cols"]

    def __init__(
        self, image_processor, tokenizer, image_seq_len=64, chat_template=None, **kwargs
    ):
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.fake_image_token = "<fake_token_around_image>"
        self.fake_image_token_id = tokenizer.convert_tokens_to_ids(
            self.fake_image_token
        )
        self.global_image_tag = "<global-img>"
        self.image_seq_len = image_seq_len
        super().__init__(
            image_processor, tokenizer, chat_template=chat_template, **kwargs
        )

    @classmethod
    def _get_arguments_from_pretrained(
        cls, pretrained_model_name_or_path, processor_dict=None, **kwargs
    ):
        return [
            GraniteDoclingImageProcessor.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            ),
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs),
        ]

    def __call__(self, images=None, text=None, *, fine_route=None, **kwargs):
        if fine_route is not None:
            kwargs["fine_route"] = fine_route
        return super().__call__(images=images, text=text, **kwargs)

    def prepare_inputs_layout(
        self, images=None, text=None, videos=None, audio=None, **kwargs
    ):
        if text is not None:
            # The template expander replaces image markers in the list in place.
            text = [text] if isinstance(text, str) else list(text)
        if images is not None:
            images = self.image_processor.fetch_images(images)
            if is_valid_image(images):
                images = [[images]]
            elif images and is_valid_image(images[0]):
                if text is None:
                    images = [images]
                else:
                    counts = [prompt.count(self.image_token) for prompt in text]
                    if sum(counts) != len(images):
                        raise ValueError(
                            "The number of images must match the image placeholders."
                        )
                    offsets = list(accumulate(counts, initial=0))
                    images = [
                        images[start:end] for start, end in zip(offsets, offsets[1:])
                    ]
            if text is not None and [len(sample) for sample in images] != [
                prompt.count(self.image_token) for prompt in text
            ]:
                raise ValueError(
                    "Each prompt must have one image placeholder per page."
                )
        return images, text, videos, audio

    def _page_prompt(self, rows, cols, fine_route):
        marker = self.fake_image_token
        tokens = self.image_token * (self.image_seq_len * (4 if fine_route else 1))
        prompt = "".join(
            "".join(f"{marker}<row_{r + 1}_col_{c + 1}>{tokens}" for c in range(cols))
            + "\n"
            for r in range(rows)
        )
        if rows * cols > 1:
            prompt += f"\n{marker}{self.global_image_tag}{tokens}{marker}"
        return prompt

    def replace_image_token(self, image_inputs, image_idx, **kwargs):
        rows = [r for sample in image_inputs["rows"] for r in sample]
        cols = [c for sample in image_inputs["cols"] for c in sample]
        return self._page_prompt(
            rows[image_idx], cols[image_idx], "tile_fine_mask" in image_inputs
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        if image_sizes is None:
            return MultiModalData()
        fine = kwargs.get("fine_route", self.image_processor.fine_route)
        counts, patches = [], []
        for height, width in image_sizes:
            rows, cols = self.image_processor.get_tile_grid(height, width, kwargs)
            prompt = self._page_prompt(rows, cols, fine)
            counts.append(
                len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            )
            patches.append(rows * cols + int(rows * cols > 1))
        return MultiModalData(num_image_tokens=counts, num_image_patches=patches)


install_auto_processor_patch("granite_docling", GraniteDoclingProcessor)
