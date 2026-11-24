"""Parse an Inkling 2 tool call.

Inkling frames every output segment as a message,
``<|message_model|>[recipient]<|content_KIND|>body<|end_message|>``, and a tool
call is the ``invoke_tool_json`` kind addressed to the function::

    <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>

The recipient sits before the kind marker and thinking closes with the same
``<|end_message|>``, so the markers span a whole message: the extractor hands
each one to ``parse_tool_call``, which returns no call for the other kinds.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

tool_call_start: str = "<|message_model|>"
tool_call_end: str = "<|end_message|>"

_INVOKE_HEADER = re.compile(r"[^<]*<\|content_invoke_tool_json\|>")


def tool_call_payload(text: str) -> Optional[str]:
    """Return the body only when the first content marker declares a tool call."""
    header = _INVOKE_HEADER.match(text)
    return text[header.end() :] if header else None


def parse_tool_call(text: str, tools: Optional[Any] = None) -> List[dict]:
    """Decode one message (markers already stripped) into ``[{name, arguments}]``, or ``[]`` when it is not a call."""
    payload = tool_call_payload(text)
    if payload is None:
        return []
    call = json.loads(payload)
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        raise ValueError(f"No function name in Inkling tool call: {text!r}")
    arguments = call.get("args", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"Inkling tool call arguments must be an object: {text!r}")
    return [{"name": call["name"], "arguments": arguments}]
