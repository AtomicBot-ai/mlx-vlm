"""Neutral internal representation for the tool-calling pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


class ToolCallParseError(ValueError):
    """Generation produced tool calls, but none could be decoded."""

    code = "tool_call_parse_error"


@dataclass(frozen=True)
class ParseResult:
    """The outcome of extracting tool calls from a model's text output.

    ``calls`` are OpenAI-shaped tool-call dicts ready for the wire;
    ``remaining_text`` is the model output with the tool-call spans removed.
    ``errors`` records rejected or unterminated calls; non-tool messages that
    share the call delimiters do not count as failures.
    """

    calls: List[Dict[str, Any]]
    remaining_text: str
    errors: List[str] = field(default_factory=list)

    def raise_if_failed(self, finish_reason=None):
        """Keep usable calls and distinguish truncation from failed generation."""
        if self.errors and not self.calls and finish_reason != "length":
            raise ToolCallParseError(
                "The model generated tool calls, but none could be parsed."
            )
