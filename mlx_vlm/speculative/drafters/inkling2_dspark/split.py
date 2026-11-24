from safetensors import safe_open

from ..mtp_split import MTPSplitter


class Inkling2DSparkSplitter(MTPSplitter):
    output_model_type = "inkling2_dspark"

    def read_text_config(self, source_config):
        text = super().read_text_config(source_config)
        text["mtp_config"] = source_config.get("mtp_config") or text.get("mtp_config")
        if not text["mtp_config"]:
            raise ValueError("Inkling 2 checkpoint has no mtp_config")
        return text

    def iter_selected(self, source_path, text_config):
        # The native sidecar has unprefixed names, including layers.* and norm.*.
        # Never scan the target shards for these names or inherit their quantization.
        path = source_path / "mtp.safetensors"
        if path.exists():
            with safe_open(path, framework="mlx") as f:
                yield path, list(f.keys())

    def depth(self, text_config):
        return int(text_config["mtp_config"]["block_size"]) - 1

    def should_quantize_key(self, key):
        # The recurrent head explicitly splits FP32 products and folds hidden_out.
        return not key.startswith("markov_head.") and super().should_quantize_key(key)
