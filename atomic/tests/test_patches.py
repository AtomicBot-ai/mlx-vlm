"""What each patch in atomic/patches guarantees, checked through behavior.

The tests run on the upstream tag with the patch series applied (Release
(Atomic)'s build job, and by hand: see atomic/README.md). They go through the
server's own entry points, not through the helpers a patch adds, so they keep
passing when a patch is dropped because upstream now does the same thing, and
fail when a patch is dropped although upstream does not. Each test names the
patch it guards. Only people change this file: the pipeline's agent may not.
"""

import json
from types import SimpleNamespace as NS

import mlx_vlm.server as server
import mlx_vlm.server.generation as generation

# `mlx_vlm.server.app` is the FastAPI application; the module is exported apart.
app_module = server._app_module


# --- 0001-server-single-model-mode ------------------------------------------


def _fake_generators(monkeypatch):
    created = []

    def make(model_path, adapter_path=None, **kwargs):
        resources = (NS(name=model_path), NS(), NS(model_type="qwen2_vl"))
        created.append(model_path)
        return NS(
            model_path=model_path,
            adapter_path=adapter_path,
            model=resources[0],
            processor=resources[1],
            config=resources[2],
            wait_until_ready=lambda: resources,
            stop_and_join=lambda: None,
        )

    monkeypatch.setattr(app_module, "ResponseGenerator", make)
    monkeypatch.setattr(app_module._apc, "from_env", lambda *a, **kw: None)
    monkeypatch.setattr(server.runtime, "model_cache", server.ModelCacheRegistry())
    monkeypatch.setattr(server.runtime, "response_generator", None)
    monkeypatch.setattr(server.runtime, "apc_manager", None)
    return created


def test_single_model_mode_keeps_the_loaded_model(monkeypatch):
    created = _fake_generators(monkeypatch)
    monkeypatch.setenv("MLX_VLM_SINGLE_MODEL", "1")
    model, _, _ = server.get_cached_model("/models/loaded")
    again, _, _ = server.get_cached_model("some-catalog-label")
    assert again is model
    assert created == ["/models/loaded"]


def test_single_model_mode_off_switches_models(monkeypatch):
    created = _fake_generators(monkeypatch)
    monkeypatch.delenv("MLX_VLM_SINGLE_MODEL", raising=False)
    server.get_cached_model("/models/loaded")
    server.get_cached_model("/models/other")
    assert created == ["/models/loaded", "/models/other"]


# --- 0002-server-honor-suppress-tokens --------------------------------------


class _Tokenizer:
    eos_token_id = 1
    unk_token_id = 3
    chat_template = "{{ messages }}"


class _Processor:
    tokenizer = _Tokenizer()
    chat_template = "{{ messages }}"


def _generator(monkeypatch, tmp_path, generation_config):
    """A ResponseGenerator after its model load, with this generation_config.json.

    The generator thread is never started: _initialize_model is the code that
    runs when a model loads, and it is called directly.
    """
    (tmp_path / "generation_config.json").write_text(json.dumps(generation_config))
    config = NS(model_type="gemma4_unified", eos_token_id=1)
    monkeypatch.setattr(
        generation,
        "load_model_resources",
        lambda model_path, adapter_path=None: (NS(), _Processor(), config),
    )
    gen = object.__new__(generation.ResponseGenerator)
    gen.model_path = str(tmp_path)
    gen.adapter_path = None
    gen.draft_model_path = None
    gen.draft_kind_override = None
    gen.apc_manager = None
    monkeypatch.delenv("MLX_VLM_DRAFT_MODEL", raising=False)
    monkeypatch.delenv("MLX_VLM_DRAFT_KIND", raising=False)
    gen._initialize_model()
    return gen


def _logit_bias(monkeypatch, gen, **request_options):
    """The logit_bias the generator hands to make_logits_processors for a request."""
    captured = {}

    def fake(logit_bias, *args, **kwargs):
        captured["logit_bias"] = logit_bias
        return []

    monkeypatch.setattr(generation, "make_logits_processors", fake)
    request = server.ChatRequest(
        model="demo", messages=[{"role": "user", "content": "hi"}], **request_options
    )
    gen._make_logits_processors(server._build_gen_args(request))
    return captured["logit_bias"] or {}


def test_suppress_tokens_become_a_default_logit_bias(monkeypatch, tmp_path):
    gen = _generator(monkeypatch, tmp_path, {"suppress_tokens": [258883, 258882]})
    bias = _logit_bias(monkeypatch, gen)
    assert bias[258883] < -1e6
    assert bias[258882] < -1e6


def test_suppress_tokens_keep_the_request_bias(monkeypatch, tmp_path):
    gen = _generator(monkeypatch, tmp_path, {"suppress_tokens": [258883]})
    bias = _logit_bias(monkeypatch, gen, logit_bias={"258883": 5.0})
    assert float(bias.get(258883, bias.get("258883"))) == 5.0


def test_no_suppress_tokens_no_bias(monkeypatch, tmp_path):
    gen = _generator(monkeypatch, tmp_path, {"eos_token_id": [1, 106]})
    assert not _logit_bias(monkeypatch, gen)
