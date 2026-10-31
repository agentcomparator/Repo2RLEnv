"""Anthropic Messages ⇄ the OpenAI chat-completions wire, in pure functions.

Pi and OpenCode speak Anthropic's Messages protocol to the local bridge — that
is the matched runtime experiment; an `openai_compat` author route speaks the
OpenAI wire contract to $OPENAI_COMPAT_BASE_URL, which is what every
OpenAI-compatible gateway serves. This module is the entire translation, with
no I/O, so the bridge keeps metering, the tool allowlist and response
validation on the runtime side of the boundary for both routes.

The gateway is called non-streaming and the bridge synthesizes Anthropic's SSE
events from the completed message: one charged completion instead of an event
delta to reconcile, and the runtime sees the exact protocol it negotiated.

Every function mirrors an Anthropic protocol field rather than a call site, so
the wire shape is pinned by tests, not by whatever the runtimes happen to send.
"""

from __future__ import annotations

import json

# OpenAI finish reasons ⇄ Anthropic stop reasons. Anything unmapped is a normal
# completion: an unusable payload is caught by validate_provider_response after
# metering, exactly like the Anthropic route.
STOP_REASONS = {
    "stop": "end_turn",
    "stop_sequence": "end_turn",
    "content_filter": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
}


def plain_text(value) -> str:
    """Flatten Anthropic content (a string or a list of blocks) to text.

    Non-text blocks are dropped here rather than at each call site: thinking,
    signature and cache_control fields have no OpenAI-wire meaning, and only
    `text` blocks ever carried language the model should re-read.
    """
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "\n".join(
        block.get("text") or ""
        for block in value
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _tool_choice(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    kind = value.get("type")
    if kind in {"auto", "any", "none"}:
        return {"auto": "auto", "any": "required", "none": "none"}[kind]
    if kind == "tool" and value.get("name"):
        return {"type": "function", "function": {"name": value["name"]}}
    return None


def _tool_entry(tool: dict) -> dict:
    function = {
        "name": tool.get("name", ""),
        "parameters": tool.get("input_schema")
        or {"type": "object", "properties": {}, "additionalProperties": True},
    }
    if tool.get("description"):
        function["description"] = tool["description"]
    return {"type": "function", "function": function}


def _messages(messages) -> list[dict]:
    output: list[dict] = []
    for message in messages or []:
        role, content = message.get("role"), message.get("content")
        if role == "user":
            if isinstance(content, str):
                output.append({"role": "user", "content": content})
                continue
            blocks = [b for b in content or [] if isinstance(b, dict)]
            results = [b for b in blocks if b.get("type") == "tool_result"]
            for block in results:
                text = plain_text(block.get("content"))
                output.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": ("Tool failed: " + text) if block.get("is_error") else text,
                    }
                )
            # The OpenAI contract follows each tool call immediately with its
            # results, then returns to the user; the user's own text comes after.
            text = plain_text(blocks)
            if text or not results:
                output.append({"role": "user", "content": text})
        elif role == "assistant":
            blocks = (
                content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            )
            calls = [
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(
                            block.get("input") if isinstance(block.get("input"), dict) else {},
                            ensure_ascii=False,
                        ),
                    },
                }
                for block in blocks
                if block.get("type") == "tool_use"
            ]
            entry: dict = {"role": "assistant", "content": plain_text(blocks)}
            if calls:
                entry["tool_calls"] = calls
            output.append(entry)
        else:
            raise ValueError(f"Anthropic messages may only be user or assistant, got {role!r}")
    return output


def to_openai_body(body: dict, *, model: str) -> dict:
    """Translate one Anthropic Messages request onto the OpenAI wire contract.

    The runtime's inference modifiers (temperature, thinking, output_config,
    top_p/top_k) were already stripped by the bridge; cache_control, metadata
    and stop-only-protocol fields have no OpenAI counterpart and never migrate.
    """
    payload: dict = {
        "model": model,
        "messages": _messages(body.get("messages")),
        # The bridge answers the runtime's stream itself, so the gateway call
        # never streams and one completion is metered as one turn.
        "stream": False,
    }
    system = plain_text(body.get("system"))
    if system:
        payload["messages"].insert(0, {"role": "system", "content": system})
    if isinstance(body.get("max_tokens"), int):
        payload["max_tokens"] = body["max_tokens"]
    tools = [tool for tool in body.get("tools") or [] if isinstance(tool, dict)]
    if tools:
        payload["tools"] = [_tool_entry(tool) for tool in tools]
        choice = _tool_choice(body.get("tool_choice"))
        if choice is not None:
            payload["tool_choice"] = choice
    if body.get("stop_sequences"):
        payload["stop"] = [str(value) for value in body["stop_sequences"]]
    return payload


def _usage(value: dict) -> dict:
    """Gateway usage tokens in the Anthropic accounting shape the bridge records."""
    if "prompt_tokens" not in value or "completion_tokens" not in value:
        return {}
    usage = {
        "input_tokens": value["prompt_tokens"],
        "output_tokens": value["completion_tokens"],
    }
    cached = (value.get("prompt_tokens_details") or {}).get("cached_tokens")
    if isinstance(cached, int) and not isinstance(cached, bool) and cached >= 0:
        usage["cache_read_input_tokens"] = cached
    return usage


def from_openai_response(data: dict, *, model: str) -> dict:
    """One OpenAI chat completion as an Anthropic assistant message."""
    choices = data.get("choices") or []
    choice: dict = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = []
    if isinstance(message.get("content"), str) and message["content"].strip():
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            # Keep the turn usable instead of failing a charged completion: the
            # handler receives a well-formed object and its input error — which
            # the schema would have caught anyway — reaches the model as
            # ordinary tool output for one bounded correction.
            parsed = {
                "_malformed_arguments": (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False)
                    if arguments is not None
                    else ""
                )
            }
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": function.get("name", ""),
                "input": parsed,
            }
        )
    return {
        "id": data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": STOP_REASONS.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": _usage(data.get("usage") or {}),
    }


def message_events(message: dict) -> list[dict]:
    """A completed Anthropic message as Anthropic's SSE event sequence.

    Replayed for a runtime that requested `stream: true`: message_start carries
    input usage, each block is start/delta/stop, message_delta carries the stop
    reason and output usage, message_stop closes it — the same shape the
    Anthropic route streams through verbatim (including the `event:` line).
    """
    usage = message.get("usage") or {}
    events: list[dict] = [
        {
            "type": "message_start",
            "message": {
                "id": message.get("id", ""),
                "type": "message",
                "role": "assistant",
                "model": message.get("model", ""),
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {**usage, "output_tokens": 0},
            },
        }
    ]
    blocks = [
        block
        for block in message.get("content") or []
        if isinstance(block, dict) and block.get("type") in {"text", "tool_use"}
    ]
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            seed = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block.get("text", "")}
        else:
            seed = {
                "type": "tool_use",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "input": {},
            }
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
            }
        events.append({"type": "content_block_start", "index": index, "content_block": seed})
        events.append({"type": "content_block_delta", "index": index, "delta": delta})
        events.append({"type": "content_block_stop", "index": index})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": message.get("stop_reason"), "stop_sequence": None},
            "usage": {"output_tokens": usage.get("output_tokens", 0)},
        }
    )
    events.append({"type": "message_stop"})
    return events
