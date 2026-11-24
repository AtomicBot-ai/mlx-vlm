import json
from pathlib import Path

from huggingface_hub import snapshot_download

SOURCE_REPO_ID = "MiniMaxAI/MiniMax-H3-Image"
SHARED_PATTERNS = (
    "model_index.json",
    "pdd_config.json",
    "text_encoder/*",
    "tokenizer/*",
)


def download_model(model=SOURCE_REPO_ID, *, revision=None, force_download=False):
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    metadata = Path(
        snapshot_download(
            str(model),
            revision=revision,
            force_download=force_download,
            allow_patterns=["model_index.json"],
        )
    )
    index = json.loads((metadata / "model_index.json").read_text())
    converted = index.get("mlx_vlm_format") == "minimax_h3_image"
    components = (
        ("transformer/*", "vae/*")
        if converted
        else ("transformer_diffusers/*", "vae_diffusers/*")
    )
    # Do not fetch duplicate native DiT/VAE weights or the optional prompt expander.
    path = Path(
        snapshot_download(
            str(model),
            revision=revision,
            force_download=force_download,
            allow_patterns=[*SHARED_PATTERNS, *components],
        )
    )
    return path


def text_encoder_path(root):
    root = Path(root)
    path = root / "text_encoder"
    if (path / "config.json").is_file():
        return path
    # Some releases keep the shared H3 conditioner in a separate Hub repo.
    index = json.loads((root / "model_index.json").read_text())
    entry = index.get("text_encoder")
    reference = entry[2] if isinstance(entry, list) and len(entry) > 2 else None
    if not isinstance(reference, dict) or not reference.get(
        "pretrained_model_name_or_path"
    ):
        raise FileNotFoundError(
            f"Missing H3-Image text_encoder/config.json under {root}"
        )
    subfolder = reference.get("subfolder", "text_encoder")
    if subfolder != "text_encoder":
        raise ValueError(f"Unsupported H3 conditioner subfolder: {subfolder}")
    shared = Path(
        snapshot_download(
            reference["pretrained_model_name_or_path"],
            revision=reference.get("revision"),
            allow_patterns=["text_encoder/*"],
        )
    )
    return shared / subfolder
