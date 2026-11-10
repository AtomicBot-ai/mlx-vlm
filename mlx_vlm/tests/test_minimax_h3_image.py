"""H3-Image loading, fixed-schedule sampling, conversion, and image API contracts."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from numpy.testing import assert_allclose, assert_array_equal

from mlx_vlm.generate import generate_image, resolve_image_defaults
from mlx_vlm.generate.edit_image import ImageEditRequest, image_edit_model_class
from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    image_generation_model_class,
    is_image_generation_model,
    load_image_generation_model,
)
from mlx_vlm.models.minimax_h3_image import (
    MiniMaxH3ImageTransformer,
    MiniMaxH3ImageTransformerConfig,
    MiniMaxH3ImageVAE,
    MiniMaxH3ImageVAEConfig,
)
from mlx_vlm.models.minimax_h3_image.convert import convert_minimax_h3_image, model_card
from mlx_vlm.models.minimax_h3_image.model import MiniMaxH3ImageGenerationModel
from mlx_vlm.models.minimax_h3_image.packing import (
    SAMPLE_SIGMAS,
    bound_short_edge,
    build_image_layout,
    canvas_size,
    sampling_schedule,
)
from mlx_vlm.models.minimax_h3_image.pipeline import MiniMaxH3ImagePipeline
from mlx_vlm.models.minimax_h3_image.weights import (
    load_conditioner,
    load_transformer,
    load_vae,
)


def _transformer_config():
    return MiniMaxH3ImageTransformerConfig(
        hidden_size=32,
        num_layers=2,
        num_refiner_layers=1,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        in_channels=2,
        text_dim=32,
        freq_dim=8,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=1,
    )


def _vae_config():
    return MiniMaxH3ImageVAEConfig(
        block_out_channels=(4, 4, 4, 4, 4, 4),
        layers_per_block=1,
        norm_num_groups=2,
        latent_channels=2,
        latents_mean=(0.2, -0.3),
        latents_std=(0.8, 1.2),
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
    )


def _inputs():
    return {
        "hidden_states": mx.arange(32, dtype=mx.float32).reshape(1, 4, 8) / 50,
        "encoder_hidden_states": mx.ones((1, 2, 32)) / 20,
        "timestep": mx.array([0.25, 0.999]),
        "timestep_indices": mx.array([0, 0, 1, 0, 0, 0]),
        "token_tags": mx.array([1, 0, 0, 0, 0, 0]),
        "position_ids": mx.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 16], [3, 16, 0], [3, 16, 16]]
        ),
        "image_indices": mx.arange(2, 6),
        "text_indices": mx.arange(2),
    }


def _write_component(path, config, weights):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "model.safetensors"), weights)


def _write_source(root):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    from mlx_vlm.models.qwen3_vl.config import ModelConfig
    from mlx_vlm.models.qwen3_vl.qwen3_vl import Model

    root.mkdir()
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ImagePipeline"})
    )
    transformer = MiniMaxH3ImageTransformer(_transformer_config())
    _write_component(
        root / "transformer_diffusers",
        asdict(transformer.config)
        | {"_class_name": "MiniMaxH3ImageTransformer2DModel"},
        dict(tree_flatten(transformer.parameters())),
    )
    vae = MiniMaxH3ImageVAE(_vae_config())
    weights = {
        k: (
            v.transpose(0, 4, 1, 2, 3)
            if v.ndim == 5
            else v.transpose(0, 3, 1, 2) if v.ndim == 4 else v
        )
        for k, v in tree_flatten(vae.parameters())
    }
    _write_component(
        root / "vae_diffusers",
        asdict(vae.config) | {"_class_name": "AutoencoderKLMiniMaxH3Image"},
        weights,
    )
    config = {
        "model_type": "qwen3_vl",
        "image_token_id": 2,
        "vision_start_token_id": 1,
        "vision_end_token_id": 3,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 51,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "rope_theta": 10000,
            "max_position_embeddings": 4096,
            "rope_scaling": {"type": "default", "mrope_section": [1, 1, 2]},
        },
        "vision_config": {
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "out_hidden_size": 32,
            "num_heads": 2,
            "patch_size": 16,
            "spatial_patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
    }
    qwen = Model(ModelConfig.from_dict(config))
    official = {}
    for key, value in tree_flatten(qwen.parameters()):
        if key.startswith("vision_tower."):
            key = key.replace("vision_tower", "model.visual", 1)
            if key == "model.visual.patch_embed.proj.weight":
                value = value.transpose(0, 4, 1, 2, 3)
        elif key.startswith("language_model.model."):
            key = key.replace("language_model.model", "model.language_model", 1)
        else:
            key = key.replace("language_model.lm_head", "lm_head", 1)
        official[key] = value
    _write_component(root / "text_encoder", config, official)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel(
                {
                    "<unk>": 0,
                    "<|vision_start|>": 1,
                    "<|image_pad|>": 2,
                    "<|vision_end|>": 3,
                },
                unk_token="<unk>",
            )
        ),
        unk_token="<unk>",
    )
    tokenizer.chat_template = "unused source chat template"
    tokenizer.save_pretrained(root / "tokenizer")
    (root / "tokenizer" / "unused-source.txt").write_text("unused tokenizer sidecar")
    (root / "processor").mkdir()
    (root / "processor" / "preprocessor_config.json").write_text("{}")
    (root / "LICENSE").write_text("source license must not be copied")


@pytest.mark.parametrize(
    "metadata",
    [{"_class_name": "MiniMaxH3ImagePipeline"}, {"model_type": "minimax_h3_image"}],
)
def test_metadata_discovery_and_defaults(tmp_path, monkeypatch, metadata):
    (tmp_path / "model_index.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(
        mx, "load", Mock(side_effect=AssertionError("Must not load weights"))
    )
    assert image_generation_model_class(str(tmp_path)) is MiniMaxH3ImageGenerationModel
    assert image_edit_model_class(str(tmp_path)) is MiniMaxH3ImageGenerationModel
    assert is_image_generation_model(str(tmp_path))
    for model in (str(tmp_path), "MiniMaxAI/MiniMax-H3-Image"):
        for task in ("generate", "edit"):
            defaults = resolve_image_defaults(model, task)
            assert (defaults.steps, defaults.guidance) == (6, 1.0)
    assert not is_image_generation_model("MiniMaxAI/MiniMax-H3")


def test_selective_download(tmp_path, monkeypatch):
    from mlx_vlm.models.minimax_h3_image import download

    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ImagePipeline"})
    )
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "text_encoder" / "config.json").write_text("{}")
    snapshot = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(download, "snapshot_download", snapshot)
    assert (
        download.download_model("MiniMaxAI/MiniMax-H3-Image", revision="abc")
        == tmp_path
    )
    patterns = snapshot.call_args.kwargs["allow_patterns"]
    assert "transformer_diffusers/*" in patterns
    assert "vae_diffusers/*" in patterns
    assert "transformer/*" not in patterns
    assert "vae/*" not in patterns
    assert all("LICENSE" not in pattern for pattern in patterns)
    assert snapshot.call_args.kwargs["revision"] == "abc"
    (tmp_path / "model_index.json").write_text(
        json.dumps({"mlx_vlm_format": "minimax_h3_image"})
    )
    download.download_model("org/converted")
    assert "transformer/*" in snapshot.call_args.kwargs["allow_patterns"]


def test_custom_repo_uses_metadata_before_weight_download(tmp_path, monkeypatch):
    import importlib

    from mlx_vlm.models.minimax_h3_image import download

    image_module = importlib.import_module("mlx_vlm.generate.image")
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ImagePipeline"})
    )
    metadata = Mock(return_value=tmp_path)
    selected = Mock(return_value=tmp_path)
    monkeypatch.setattr(image_module, "get_model_path", metadata)
    monkeypatch.setattr(download, "download_model", selected)
    model = load_image_generation_model("example/custom-package", revision="abc")
    assert isinstance(model, MiniMaxH3ImageGenerationModel)
    assert all(
        not pattern.endswith("safetensors")
        for pattern in metadata.call_args.kwargs["allow_patterns"]
    )
    selected.assert_called_once_with(
        "example/custom-package", revision="abc", force_download=False
    )


def test_prompt_expander_is_not_an_image_generator(tmp_path, monkeypatch):
    import importlib

    image_module = importlib.import_module("mlx_vlm.generate.image")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.setattr(image_module, "get_model_path", Mock(return_value=tmp_path))
    monkeypatch.setattr(
        mx, "load", Mock(side_effect=AssertionError("Must not load weights"))
    )
    model = "MiniMaxAI/MiniMax-H3-Image-PE"
    assert not is_image_generation_model(model)
    assert image_generation_model_class(model) is None


def test_external_conditioner_pointer_and_local_precedence(tmp_path, monkeypatch):
    from mlx_vlm.models.minimax_h3_image import download

    root, shared = tmp_path / "image", tmp_path / "shared"
    root.mkdir()
    (shared / "text_encoder").mkdir(parents=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "text_encoder": [
                    "transformers",
                    "Qwen3VLForConditionalGeneration",
                    {
                        "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
                        "subfolder": "text_encoder",
                        "revision": "abc",
                    },
                ]
            }
        )
    )
    snapshot = Mock(return_value=str(shared))
    monkeypatch.setattr(download, "snapshot_download", snapshot)
    assert download.text_encoder_path(root) == shared / "text_encoder"
    snapshot.assert_called_once_with(
        "MiniMaxAI/MiniMax-H3", revision="abc", allow_patterns=["text_encoder/*"]
    )
    (root / "text_encoder").mkdir()
    (root / "text_encoder" / "config.json").write_text("{}")
    assert download.text_encoder_path(root) == root / "text_encoder"
    assert snapshot.call_count == 1


def test_schedule_and_reference_layout():
    timesteps, sigmas = sampling_schedule()
    expected = np.array([1 - sigma for sigma in SAMPLE_SIGMAS[:-1]], np.float32)
    assert_array_equal(np.array(timesteps), expected)
    assert_array_equal(np.array(sigmas), np.r_[np.float32(1) - expected, np.float32(0)])
    positions, tags, images, texts, count = build_image_layout(
        mx.array([1, 0, 1]), [(2, 4), (4, 2)], 4, 4
    )
    assert count == 4
    assert_array_equal(np.array(texts), [0, 1, 2])
    assert_array_equal(np.array(images), np.arange(3, 11))
    assert_array_equal(np.array(tags), [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert_array_equal(np.array(positions[:, 0]), [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5])
    assert_allclose(np.array(positions[-4:, 1:]), [[0, 0], [0, 16], [16, 0], [16, 16]])
    assert canvas_size(512, 512) == (2048, 2048)
    assert canvas_size(512, 512, 1024) == (512, 512)
    assert canvas_size(65, 97, 1024) == (64, 96)
    assert bound_short_edge(2048, 1024, 1024, 2048) == (2048, 1024)
    assert bound_short_edge(400, 200, 1024, 1024) == (2048, 1024)
    with pytest.raises(ValueError, match="32 pixels"):
        bound_short_edge(20, 400, 1024, 1024)


def test_transformer_heads_and_adaln_cache():
    model = MiniMaxH3ImageTransformer(_transformer_config())
    inputs = _inputs()
    live = model(**inputs, pdd_step_index=2)
    cache = model.build_adaln_cache(inputs["timestep"])
    cached = model(**inputs, pdd_step_index=2, adaln_cache=cache)
    assert_allclose(np.array(live), np.array(cached), atol=1e-6)
    assert model.drop_adaln_weights() > 0
    assert not model.adaln_weights_available
    assert_allclose(
        np.array(live),
        np.array(model(**inputs, pdd_step_index=2, adaln_cache=cache)),
        atol=1e-6,
    )
    with pytest.raises(RuntimeError, match="cache"):
        model(**inputs, pdd_step_index=2)
    for step, head in enumerate(model.pdd_proj_out):
        head.weight = mx.zeros_like(head.weight)
        head.bias = mx.full(head.bias.shape, step, dtype=mx.float32)
        result = model(**inputs, pdd_step_index=step, adaln_cache=cache)
        assert result.shape == (1, 4, 8)
        assert_array_equal(np.array(result), np.full((1, 4, 8), step))
    with pytest.raises(ValueError, match="six"):
        model(**inputs, pdd_step_index=6, adaln_cache=cache)


def test_vae_tiling_and_single_frame_encoder():
    vae = MiniMaxH3ImageVAE(_vae_config())
    pixels = mx.random.normal((1, 3, 64, 64))
    assert vae.encode(pixels).mean.shape == (1, 2, 4, 4)
    z = mx.random.normal((1, 2, 4, 4))
    tiled = vae.decode(z).sample
    vae.disable_tiling()
    assert_allclose(np.array(vae.decode(z).sample), np.array(tiled), atol=1e-6)
    vae.enable_tiling(
        tile_sample_min_height=64,
        tile_sample_min_width=64,
        tile_sample_min_overlap_height=16,
        tile_sample_min_overlap_width=16,
    )
    result = vae.decode(mx.random.normal((1, 2, 8, 8))).sample
    assert result.shape == (1, 3, 128, 128)
    assert mx.all(mx.isfinite(result)).item()


@pytest.mark.parametrize(
    "options,match",
    [
        ({"steps": 5}, "six"),
        ({"guidance": 2}, "guidance"),
        ({"width": 0}, "positive"),
        ({"min_generation_area": 1}, "min_generation_area"),
        ({"max_sequence_length": 0}, "max_sequence_length"),
        ({"image_paths": ("x",) * 17}, "16 reference"),
    ],
)
def test_invalid_requests_fail_before_loading(options, match):
    pipeline = MiniMaxH3ImagePipeline("/does-not-exist")
    with pytest.raises(ValueError, match=match):
        pipeline.generate_array("a teapot", **options)


def test_model_api_and_edit_validation(tmp_path):
    pipeline = SimpleNamespace(
        model_path=tmp_path,
        prompt_tokens=12,
        generate_array=Mock(return_value=mx.zeros((32, 64, 3), mx.uint8)),
    )
    model = MiniMaxH3ImageGenerationModel(pipeline, "synthetic")
    request = ImageGenerationRequest("a teapot", width=64, height=32)
    result = model.generate(request)
    assert (result.steps, result.guidance, result.prompt_tokens) == (6, 1, 12)
    assert pipeline.generate_array.call_args.kwargs["steps"] == 6
    edit = ImageEditRequest("make it blue", ("one.png", "two.png"), width=64, height=32)
    result = model.edit(edit)
    assert result.metadata["reference_images"] == 2
    assert pipeline.generate_array.call_args.kwargs["image_paths"] == edit.image_paths
    with pytest.raises(ValueError, match="one to sixteen"):
        model.edit(ImageEditRequest("blue", ()))
    with pytest.raises(ValueError, match="negative_prompt"):
        model.generate(
            ImageGenerationRequest("teapot", extra={"negative_prompt": "bad"})
        )


def test_native_edit_presentation_and_frozen_reference_rows(monkeypatch):
    from mlx_vlm.models.minimax_h3.visual_vae import (
        MiniMaxH3DiagonalGaussianDistribution,
    )
    from mlx_vlm.models.minimax_h3_image import pipeline as module

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [10, 11] if text.startswith("Picture") else [12]

        def convert_tokens_to_ids(self, token):
            return {"<|vision_start|>": 1, "<|image_pad|>": 2, "<|vision_end|>": 3}[
                token
            ]

    presentations = []

    def encode(ids, tags, **kwargs):
        presentations.append((ids, tags))
        return SimpleNamespace(hidden_states=mx.ones((1, len(ids), 32)) * 0.01)

    # Keep the real DiT, packing, VAE decoding and sampler; replace only the
    # large pretrained encoders with deterministic conditioning fixtures.
    monkeypatch.setattr(
        module,
        "process_qwen_images",
        lambda images: (mx.zeros((2, 1536)), mx.array([[1, 2, 2], [1, 2, 2]])),
    )
    pipeline = MiniMaxH3ImagePipeline(
        "/synthetic", evict_text_encoder=False, upsample_prompt=False
    )
    pipeline._conditioner = SimpleNamespace(tokenizer=Tokenizer(), _encode=encode)
    transformer = MiniMaxH3ImageTransformer(_transformer_config())
    calls = []

    class RecordingTransformer:
        def __getattr__(self, name):
            return getattr(transformer, name)

        def __call__(self, *args):
            calls.append((mx.array(args[0]), mx.array(args[3])))
            return transformer(*args)

    pipeline._transformer = RecordingTransformer()
    vae = MiniMaxH3ImageVAE(_vae_config())
    pipeline._vae = SimpleNamespace(
        config=vae.config,
        encode=lambda pixels: MiniMaxH3DiagonalGaussianDistribution(
            mx.zeros((1, 4, 4, 4))
        ),
        decode=vae.decode,
    )
    images = [mx.zeros((64, 64, 3), mx.uint8), mx.full((64, 64, 3), 100, mx.uint8)]
    result = pipeline.generate_array(
        "make both teapots blue",
        image_paths=images,
        width=64,
        height=64,
        min_generation_area=1024,
    )
    assert result.shape == (64, 64, 3)
    assert presentations == [
        ([10, 11, 1, 2, 3, 10, 11, 1, 2, 3, 12], [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    ]
    assert len(calls) == 6
    for rows, indices in calls:
        assert_array_equal(np.array(rows[:, :8]), np.array(calls[0][0][:, :8]))
        assert_array_equal(np.array(indices), [0] * 11 + [1] * 8 + [0] * 4)


@pytest.mark.parametrize("quantize", [False, True])
def test_conversion_strict_reload_and_generation(tmp_path, quantize):
    source, converted = tmp_path / "source", tmp_path / "mlx"
    _write_source(source)
    model = load_image_generation_model(str(source), upsample_prompt=False)
    baseline = generate_image(
        model, "a teapot", width=64, height=64, seed=7, min_generation_area=1024
    )
    assert baseline.array.shape == (64, 64, 3)
    assert model.pipeline._conditioner is None
    assert not model.pipeline.transformer.adaln_weights_available
    repeat = generate_image(
        model, "a teapot", width=64, height=64, seed=7, min_generation_area=1024
    )
    assert_array_equal(np.array(baseline.array), np.array(repeat.array))
    convert_minimax_h3_image(source, converted, quantize=quantize, q_group_size=32)
    assert not (converted / "LICENSE").exists()
    assert not (converted / "processor").exists()
    assert not (converted / "tokenizer" / "unused-source.txt").exists()
    assert not (converted / "tokenizer" / "chat_template.jinja").exists()
    assert "tags:\n- mlx" in model_card()
    assert (
        load_conditioner(converted).model.language_model.model.num_hidden_layers == 50
    )
    assert len(load_transformer(converted).pdd_proj_out) == 6
    assert load_vae(converted).config.latents_mean == (0.2, -0.3)
    loaded = load_image_generation_model(str(converted), upsample_prompt=False)
    result = generate_image(
        loaded, "a teapot", width=64, height=64, seed=7, min_generation_area=1024
    )
    assert result.array.shape == baseline.array.shape
    if not quantize:
        assert_array_equal(np.array(result.array), np.array(baseline.array))
    else:
        assert any(
            key.endswith(".scales")
            for key, _ in tree_flatten(loaded.pipeline.transformer.parameters())
        )
    weights = converted / "transformer" / "model.safetensors"
    values = mx.load(str(weights))
    del values["pdd_proj_out.5.weight"]
    mx.save_safetensors(str(weights), values)
    with pytest.raises(ValueError, match="pdd_proj_out.5.weight"):
        load_transformer(converted)
