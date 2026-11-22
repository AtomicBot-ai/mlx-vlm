"""The training skill must describe the trainer that actually ships.

An agent follows the skill literally, so a command, option or dataset format
it names has to exist. These tests read the skill's Markdown and check it
against the real CLI.
"""

import json
import re
from pathlib import Path

import pytest

from mlx_vlm import train as train_cli
from mlx_vlm import train_runs

SKILL = Path(__file__).resolve().parents[2] / "skills" / "skills" / "training"
FILES = [SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]
ALGORITHMS = train_cli.describe()["tasks"]["vlm"]["algorithms"]


def options_of(algorithm):
    """Options a caller may set for an algorithm; managed ones are excluded."""
    return {
        option["name"]: option
        for option in ALGORITHMS[algorithm]["options"]
        if option["group"] != "managed"
    }


ANY_OPTION = set().union(*(options_of(name) for name in ALGORITHMS))


def json_examples():
    for path in FILES:
        for block in re.findall(r"```json\n(.*?)```", path.read_text(), re.DOTALL):
            example = json.loads(block)
            if "algorithm" in example:
                yield pytest.param(example, id=f"{path.name}:{example['algorithm']}")


def test_every_reference_is_linked_and_every_link_resolves():
    text = (SKILL / "SKILL.md").read_text()
    linked = set(re.findall(r"\]\(references/([a-z-]+\.md)\)", text))
    assert linked == {path.name for path in (SKILL / "references").glob("*.md")}
    for path in FILES[1:]:
        for name in re.findall(r"\]\(([a-z-]+\.md)\)", path.read_text()):
            assert (path.parent / name).is_file(), f"{path.name} links to {name}"


def test_documented_commands_are_the_real_commands():
    text = (SKILL / "SKILL.md").read_text()
    table = text.split("## Commands")[1].split("## Playbooks")[0]
    documented = set(re.findall(r"^\| `([a-z-]+)", table, re.MULTILINE))
    parser = train_runs.build_parser()
    real = set(
        next(action for action in parser._actions if action.dest == "command").choices
    )
    assert documented == real


@pytest.mark.parametrize("example", json_examples())
def test_example_options_exist_and_pass_the_training_parser(example, tmp_path):
    allowed = options_of(example["algorithm"])
    unknown = set(example) - set(allowed) - {"algorithm"}
    assert not unknown, f"not options of {example['algorithm']}: {sorted(unknown)}"
    config = tmp_path / "options.json"
    config.write_text(json.dumps(example))
    train_cli.parse_args(["-c", str(config)])  # exits on an invalid option set


def test_settings_written_as_name_and_value_are_real_options():
    for path in FILES:
        for name in re.findall(r"`([a-z_]+): [^`]+`", path.read_text()):
            assert name in ANY_OPTION, f"{path.name} sets unknown option '{name}'"


def test_named_choices_are_offered_by_the_trainer():
    options = options_of("sft")
    text = "\n".join(path.read_text() for path in FILES)
    for name in ("train_type", "optimizer", "lr_schedule"):
        for value in re.findall(rf'`{name}: "?([a-z_]+)"?`', text):
            assert value in options[name]["choices"], f"{name}: {value}"
    assert options["learning_rate"]["default"] == 1e-5  # the skill says to raise it


def test_dataset_formats_match_what_inspect_dataset_reports():
    text = (SKILL / "references" / "datasets.md").read_text()
    documented = dict(re.findall(r"^\| `([a-z_]+)` \| .*? \| (.*?) \|$", text, re.M))
    documented.pop("unknown")
    documented.pop("format")  # the table's header row
    reported = {
        name: set(algorithms) for name, _, algorithms in train_runs._ROW_FORMATS
    }
    assert {
        name: set(re.findall(r"`([a-z]+)`", cell)) for name, cell in documented.items()
    } == reported
    for algorithms in reported.values():
        assert algorithms <= set(ALGORITHMS)


def test_fields_the_skill_reads_are_fields_the_commands_return(tmp_path):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there."},
            ],
            "_meta": {"source": "x" * 500},
        }
    ] * 5
    (tmp_path / "train.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    split = train_runs.inspect_dataset({"data": str(tmp_path)})["splits"]["train"]
    assert split["format"] == "messages" and split["algorithms"] == ["sft"]
    assert split["rows"] == split["sampled_rows"] == 5
    # Bookkeeping columns do not count as training text.
    assert split["text_chars"] == {"median": 27, "p95": 27, "max": 27}
    assert split["rows_with_images"] == 0

    usage = train_runs.RunManager(tmp_path / "runs", train_runs.Limits()).usage()
    assert usage["system"]["memory_gb"] > 0

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "quantization": {"bits": 4}})
    )
    (model / "model.safetensors").write_bytes(b"0" * 2_000_000)
    assert train_runs.validate_model(str(model)) == {
        "valid": True,
        "model": str(model),
        "model_type": "qwen3_5",
        "quantization_bits": 4,
        "weights_gb": 0.0,
    }
