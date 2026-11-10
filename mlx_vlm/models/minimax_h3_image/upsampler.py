"""MiniMax's released prompt-expansion requests, using the native MLX VLM loader."""

from __future__ import annotations

import gc
import json
import math
import re
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import mlx.core as mx
from PIL import Image, ImageOps

DEFAULT_UPSAMPLER = "MiniMaxAI/MiniMax-H3-Image-PE"
MAX_TOKENS = 4096
MAX_REFERENCES = 16


@dataclass(frozen=True, slots=True)
class PromptExpansionResult:
    expanded_caption: str
    aspect_ratio: str
    reasoning: str = field(default="", repr=False)


def validate_aspect_ratio(value):
    match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", value)
    if match is None:
        raise ValueError(f"Expected a positive integer W:H aspect ratio, got {value!r}")
    return value


def prepare_reference(image):
    """Keep original display geometry in the request, and bound the VLM preview."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as decoded:
            image = ImageOps.exif_transpose(decoded).convert("RGB")
    else:
        image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if min(width, height) < 1 or max(width, height) / min(width, height) > 200:
        raise ValueError(
            "H3 prompt-expander references need positive dimensions and an aspect ratio within 200:1"
        )
    divisor = math.gcd(width, height)
    ratio = f"{width // divisor}:{height // divisor}"
    # Geometry metadata describes the original, before this preview resize.
    preview = image.copy()
    preview.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    return preview, ratio


def load_prompt_assets(model_path):
    root = Path(model_path) / "prompt"
    root = root if root.is_dir() else files(__package__).joinpath("prompts")
    return {
        "system": root.joinpath("t2i/system_prompt.txt")
        .read_text(encoding="utf-8")
        .rstrip("\n"),
        "t2i": root.joinpath("t2i/task_prompt.txt").read_text(encoding="utf-8"),
        "edit": root.joinpath("edit/task_prompt.txt").read_text(encoding="utf-8"),
        "params": json.loads(
            root.joinpath("t2i/request_params.json").read_text(encoding="utf-8")
        ),
    }


def build_messages(prompt, references, assets, aspect_ratio="adaptive"):
    if not references:
        return [
            {"role": "system", "content": assets["system"]},
            {"role": "user", "content": assets["t2i"].format(prompt)},
        ]
    if aspect_ratio != "adaptive":
        validate_aspect_ratio(aspect_ratio)
    count = len(references)
    content = [
        {"type": "text", "text": assets["edit"]},
        {
            "type": "text",
            "text": f"Reference Image Count:\n{count} input image(s), labeled <Picture 1> through <Picture {count}>.",
        },
    ]
    for index, (_, ratio) in enumerate(references, 1):
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"Reference Media:\n- <Picture {index}> [aspect_ratio {ratio}]",
                },
                {"type": "image"},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": (
                "\nUser Editing Instruction (JSON string):\n"
                f"{json.dumps(prompt, ensure_ascii=False)}\n\n"
                f"User Aspect Ratio Selection:\n{aspect_ratio}\n\n"
                "Apply the instructions above to the supplied inputs and reference images. "
                "Output only the required two-field JSON object."
            ),
        }
    )
    return [{"role": "user", "content": content}]


def response_schema(assets, *, edit=False, aspect_ratio="adaptive"):
    schema = json.loads(
        json.dumps(assets["params"]["response_format"]["json_schema"]["schema"])
    )
    if edit:
        schema["properties"].pop("aspect_ratio")
        schema["properties"]["target_aspect_ratio"] = (
            {"type": "string", "pattern": "^[1-9][0-9]*:[1-9][0-9]*$"}
            if aspect_ratio == "adaptive"
            else {"type": "string", "enum": [validate_aspect_ratio(aspect_ratio)]}
        )
        schema["required"] = ["expanded_caption", "target_aspect_ratio"]
    return schema


def parse_expansion(text, *, schema, edit=False):
    reasoning = ""
    # The checkpoint's chat template pre-opens <think>. Keep the delimiters
    # during decoding and strip reasoning before parsing the complete answer.
    if not text.lstrip().startswith("{") and "</think>" in text:
        reasoning, _, text = text.partition("</think>")
        reasoning = reasoning.removeprefix("<think>").strip()
    try:
        value = json.loads(text.strip())
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "H3 prompt expander did not return a complete JSON caption"
        ) from exc
    ratio_key = "target_aspect_ratio" if edit else "aspect_ratio"
    if not isinstance(value, dict) or set(value) != {"expanded_caption", ratio_key}:
        raise ValueError(
            "H3 prompt expansion must contain only the caption and aspect ratio"
        )
    caption, ratio = value["expanded_caption"], value[ratio_key]
    if (
        not isinstance(caption, str)
        or not caption.strip()
        or not isinstance(ratio, str)
    ):
        raise ValueError(
            "H3 expanded caption and aspect ratio must be non-empty strings"
        )
    validate_aspect_ratio(ratio)
    allowed = schema["properties"][ratio_key].get("enum")
    if allowed is not None and ratio not in allowed:
        raise ValueError(
            f"H3 prompt expander returned an unsupported aspect ratio: {ratio}"
        )
    return PromptExpansionResult(caption.strip(), ratio, reasoning)


class MiniMaxH3PromptUpsampler:
    def __init__(
        self,
        model=DEFAULT_UPSAMPLER,
        *,
        evict_model=True,
        max_tokens=MAX_TOKENS,
        enable_thinking=True,
        reasoning_effort="xhigh",
    ):
        if max_tokens < 1:
            raise ValueError("Prompt expansion max_tokens must be positive")
        if reasoning_effort not in {"xhigh", "medium", "low"}:
            raise ValueError("reasoning_effort must be xhigh, medium, or low")
        self.model_id = str(model)
        self.evict_model = evict_model
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.model = self.processor = self.prompts = None

    def _load(self):
        if self.model is not None:
            return
        from ...utils import get_model_path, load

        path = get_model_path(self.model_id)
        self.prompts = load_prompt_assets(path)
        self.model, self.processor = load(str(path), trust_remote_code=False)

    def unload(self):
        self.model = self.processor = None
        gc.collect()
        mx.clear_cache()

    def __call__(self, prompt, images=(), *, aspect_ratio="adaptive", seed=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        if len(images) > MAX_REFERENCES:
            raise ValueError("H3 prompt expansion accepts at most 16 reference images")
        if images and aspect_ratio != "adaptive":
            validate_aspect_ratio(aspect_ratio)
        references = [prepare_reference(image) for image in images]
        # Loading initializes MLX modules before replacing their weights, which
        # can also consume random keys. Preserve the caller's state around both
        # initialization and sampling when a request supplies its own seed.
        state = [mx.array(key) for key in mx.random.state] if seed is not None else None
        try:
            self._load()
            messages = build_messages(prompt, references, self.prompts, aspect_ratio)
            formatted = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                preserve_thinking=True,
                reasoning_effort=self.reasoning_effort,
            )
            schema = response_schema(
                self.prompts, edit=bool(images), aspect_ratio=aspect_ratio
            )
            from ...generate.dispatch import generate
            from ...structured import (
                ThinkingAwareLogitsProcessor,
                build_json_schema_logits_processor,
            )

            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            grammar = build_json_schema_logits_processor(tokenizer, schema)
            grammar = ThinkingAwareLogitsProcessor(
                grammar, tokenizer, enable_thinking=self.enable_thinking
            )
            if seed is not None:
                mx.random.seed(seed)
            params = self.prompts["params"]
            result = generate(
                self.model,
                self.processor,
                formatted,
                image=[image for image, _ in references] or None,
                max_tokens=self.max_tokens,
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=params["top_k"],
                min_p=params["min_p"],
                presence_penalty=params["presence_penalty"],
                repetition_penalty=params["repetition_penalty"],
                logits_processors=[grammar],
                enable_thinking=self.enable_thinking,
                skip_special_tokens=False,
                verbose=False,
            )
            if result.finish_reason == "length":
                raise RuntimeError(
                    "H3 prompt expander reached its token limit before finishing the caption"
                )
            if result.finish_reason != "stop":
                raise RuntimeError(
                    f"H3 prompt expansion did not finish: {result.finish_reason}"
                )
            return parse_expansion(result.text, schema=schema, edit=bool(images))
        finally:
            if state is not None:
                for index, key in enumerate(state):
                    mx.random.state[index][:] = key
            if self.evict_model:
                self.unload()
