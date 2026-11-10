"""Released H3 prompt contracts and image integration, without model downloads."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx_vlm.generate.image import ImageGenerationRequest
from mlx_vlm.models.minimax_h3_image.model import MiniMaxH3ImageGenerationModel
from mlx_vlm.models.minimax_h3_image.pipeline import MiniMaxH3ImagePipeline
from mlx_vlm.models.minimax_h3_image.upsampler import (
    DEFAULT_UPSAMPLER,
    MAX_TOKENS,
    MiniMaxH3PromptUpsampler,
    PromptExpansionResult,
    build_messages,
    load_prompt_assets,
    parse_expansion,
    prepare_reference,
    response_schema,
)
from mlx_vlm.structured import ThinkingAwareLogitsProcessor

dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
pipeline_module = importlib.import_module("mlx_vlm.models.minimax_h3_image.pipeline")
structured = importlib.import_module("mlx_vlm.structured")


@pytest.fixture
def mocked_upsampler(tmp_path, monkeypatch):
    upsampler = MiniMaxH3PromptUpsampler()
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [1 if text == "<think>" else 2]
    )
    processor = SimpleNamespace(
        tokenizer=tokenizer, apply_chat_template=Mock(return_value="formatted")
    )
    assets = load_prompt_assets(tmp_path)

    def load():
        upsampler.model, upsampler.processor = object(), processor
        upsampler.prompts = assets

    monkeypatch.setattr(upsampler, "_load", load)
    generate = Mock(
        return_value=SimpleNamespace(
            text='Plan the scene.\n</think>\n{"expanded_caption":"A glossy green teapot.","aspect_ratio":"3:2"}',
            finish_reason="stop",
        )
    )
    monkeypatch.setattr(dispatch, "generate", generate)
    grammar = Mock()
    monkeypatch.setattr(structured, "build_json_schema_logits_processor", grammar)
    return upsampler, processor, generate, grammar


@pytest.mark.parametrize("edit", [False, True])
def test_released_messages_sampling_and_thinking_aware_grammar(mocked_upsampler, edit):
    upsampler, processor, generate, grammar = mocked_upsampler
    images = (
        [Image.new("RGB", (300, 200)), Image.new("RGB", (100, 200))] if edit else []
    )
    if edit:
        generate.return_value.text = 'Consider the sources.</think>{"expanded_caption":"Make <Picture 1> green.","target_aspect_ratio":"16:9"}'
    result = upsampler("a green teapot", images, aspect_ratio="16:9", seed=42)
    assert result.aspect_ratio == ("16:9" if edit else "3:2")
    assert result.reasoning
    messages = processor.apply_chat_template.call_args.args[0]
    template_kwargs = processor.apply_chat_template.call_args.kwargs
    assert template_kwargs["enable_thinking"] is True
    assert template_kwargs["preserve_thinking"] is True
    assert template_kwargs["reasoning_effort"] == "xhigh"
    if edit:
        assert len(messages) == 1 and messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert content[0]["text"].startswith("You are an expert")
        assert (
            content[2]["text"] == "Reference Media:\n- <Picture 1> [aspect_ratio 3:2]"
        )
        assert content[3] == {"type": "image"}
        assert (
            content[4]["text"] == "Reference Media:\n- <Picture 2> [aspect_ratio 1:2]"
        )
        assert '"a green teapot"' in content[-1]["text"]
        assert "User Aspect Ratio Selection:\n16:9" in content[-1]["text"]
    else:
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith("Reason in the thinking section")
        assert "input_text: a green teapot\noutput_text:" in messages[1]["content"]
    settings = generate.call_args.kwargs
    assert settings["max_tokens"] == MAX_TOKENS
    assert (settings["temperature"], settings["top_p"], settings["top_k"]) == (
        1,
        0.95,
        20,
    )
    assert settings["min_p"] == 0 and settings["presence_penalty"] == 0
    assert settings["repetition_penalty"] == 1
    assert settings["skip_special_tokens"] is False
    assert isinstance(settings["logits_processors"][0], ThinkingAwareLogitsProcessor)
    assert settings["logits_processors"][0].enable_thinking
    schema = grammar.call_args.args[1]
    assert schema["additionalProperties"] is False
    if edit:
        assert schema["properties"]["target_aspect_ratio"]["enum"] == ["16:9"]
        assert len(settings["image"]) == 2
    else:
        assert len(schema["properties"]["aspect_ratio"]["enum"]) == 10
        assert settings["image"] is None
    assert upsampler.model is upsampler.processor is None


@pytest.mark.parametrize(
    "text",
    [
        '{"expanded_caption":"", "aspect_ratio":"1:1"}',
        '{"expanded_caption":"cat", "aspect_ratio":"0:1"}',
        '{"expanded_caption":"cat", "aspect_ratio":"7:3"}',
        '{"expanded_caption":"cat", "aspect_ratio":"1:1", "extra":1}',
        "unclosed reasoning",
        '```json\n{"expanded_caption":"cat", "aspect_ratio":"1:1"}\n```',
    ],
)
def test_invalid_response_releases_model(mocked_upsampler, text):
    upsampler, _, generate, _ = mocked_upsampler
    generate.return_value.text = text
    with pytest.raises(ValueError):
        upsampler("a cat")
    assert upsampler.model is upsampler.processor is None


@pytest.mark.parametrize("failure", ["length", "exception"])
def test_generation_failure_releases_model(mocked_upsampler, failure):
    upsampler, _, generate, _ = mocked_upsampler
    if failure == "length":
        generate.return_value.finish_reason = "length"
    else:
        generate.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError):
        upsampler("a cat")
    assert upsampler.model is upsampler.processor is None


def test_seed_does_not_change_callers_random_state(mocked_upsampler):
    upsampler, _, generate, _ = mocked_upsampler
    result = generate.return_value
    load = upsampler._load

    def initialize():
        mx.eval(mx.random.uniform(shape=(4,)))
        load()

    def sample(*args, **kwargs):
        mx.eval(mx.random.normal((4,)))
        return result

    generate.side_effect = sample
    upsampler._load = initialize
    before = [np.array(key) for key in mx.random.state]
    upsampler("a cat", seed=123)
    for key, expected in zip(mx.random.state, before):
        np.testing.assert_array_equal(np.array(key), expected)


def test_reference_ratio_is_original_oriented_geometry():
    image = Image.new("RGB", (3000, 1000))
    preview, ratio = prepare_reference(image)
    assert ratio == "3:1" and max(preview.size) == 2048
    rotated = Image.new("RGB", (100, 200))
    rotated.getexif()[274] = 6
    preview, ratio = prepare_reference(rotated)
    assert ratio == "2:1" and preview.size == (200, 100)


def test_input_validation_precedes_model_load(monkeypatch):
    upsampler = MiniMaxH3PromptUpsampler()
    load = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(upsampler, "_load", load)
    for prompt, images, ratio in [
        ("", [], "adaptive"),
        ("cat", [object()] * 17, "adaptive"),
        ("edit", [Image.new("RGB", (64, 64))], "0:1"),
    ]:
        with pytest.raises(ValueError):
            upsampler(prompt, images, aspect_ratio=ratio)
    load.assert_not_called()


def test_bundled_prompt_assets_and_caption_literals(tmp_path):
    assets = load_prompt_assets(tmp_path)
    assert len(assets["t2i"]) > 15000 and len(assets["edit"]) > 38000
    query = '猫 {a} "quoted"'
    assert f"input_text: {query}" in build_messages(query, [], assets)[1]["content"]
    schema = response_schema(assets)
    caption = "Render the literal text </think> on a sign."
    result = parse_expansion(
        json.dumps({"expanded_caption": caption, "aspect_ratio": "1:1"}), schema=schema
    )
    assert result.expanded_caption == caption


def test_expansion_cache_and_weight_eviction(monkeypatch):
    pipeline = MiniMaxH3ImagePipeline("/not-loaded")
    pipeline._conditioner = pipeline._transformer = pipeline._vae = object()
    pipeline._adaln_caches = [object()]
    calls = []

    class Upsampler:
        def __init__(self, model, **kwargs):
            self.model_id = model

        def __call__(self, prompt, images, **kwargs):
            assert (
                pipeline._conditioner is pipeline._transformer is pipeline._vae is None
            )
            assert pipeline._adaln_caches is None
            calls.append((prompt, images, kwargs))
            return PromptExpansionResult("Expanded caption.", "3:2")

        def unload(self):
            pass

    monkeypatch.setattr(pipeline_module, "MiniMaxH3PromptUpsampler", Upsampler)
    args = {"width": 2048, "height": 2048, "seed": 42}
    assert pipeline._prepare_prompt("a cat", [], **args) == "Expanded caption."
    assert pipeline._prepare_prompt("a cat", [], **args) == "Expanded caption."
    assert len(calls) == 1
    assert pipeline.last_prompt_metadata["revised_prompt"] == "Expanded caption."
    assert pipeline.last_prompt_metadata["suggested_aspect_ratio"] == "3:2"
    pipeline._prepare_prompt("a cat", [], **(args | {"seed": 43}))
    assert len(calls) == 2
    for index in range(6):
        pipeline._prepare_prompt(str(index), [], **args)
    assert len(pipeline.upsampled_prompts) == 4
    assert (
        pipeline._prepare_prompt("raw caption", [], **args, upsample_prompt=False)
        == "raw caption"
    )
    assert pipeline.last_prompt_metadata == {
        "input_prompt": "raw caption",
        "upsampled": False,
    }


def test_edit_expansion_is_not_cached_and_uses_requested_ratio(monkeypatch):
    pipeline = MiniMaxH3ImagePipeline("/not-loaded")
    call = Mock(return_value=PromptExpansionResult("Edited caption.", "2:1"))
    factory = Mock(return_value=Mock(model_id=DEFAULT_UPSAMPLER, side_effect=call))
    monkeypatch.setattr(pipeline_module, "MiniMaxH3PromptUpsampler", factory)
    images = [mx.zeros((64, 96, 3), mx.uint8)]
    for _ in range(2):
        pipeline._prepare_prompt(
            "make it blue", images, width=2048, height=1024, seed=42
        )
    assert call.call_count == 2
    assert call.call_args.kwargs["aspect_ratio"] == "2:1"
    assert call.call_args.args[1][0].size == (96, 64)


def test_image_wrapper_forwards_expansion_controls_and_metadata(tmp_path):
    pipeline = SimpleNamespace(
        model_path=tmp_path,
        prompt_tokens=10,
        last_prompt_metadata={"revised_prompt": "expanded", "upsampled": True},
        generate_array=Mock(return_value=mx.zeros((32, 32, 3), mx.uint8)),
    )
    model = MiniMaxH3ImageGenerationModel(pipeline, "test")
    result = model.generate(
        ImageGenerationRequest(
            "short",
            width=32,
            height=32,
            extra={"upsample_prompt": True, "prompt_expansion_model": "local/pe"},
        )
    )
    assert pipeline.generate_array.call_args.kwargs["upsample_prompt"]
    assert (
        pipeline.generate_array.call_args.kwargs["prompt_expansion_model"] == "local/pe"
    )
    assert result.metadata["revised_prompt"] == "expanded"
