"""Anthropic Messages ⇄ the OpenAI wire — the compat author route's whole translation.

Pi and OpenCode negotiate Anthropic's Messages protocol with the bridge, while
`openai_compat` speaks OpenAI's chat-completions contract upstream. These tests
pin both directions as protocol shapes, so bridge metering, the tool allowlist
and response validation can stay on the runtime side of the boundary.
"""

from __future__ import annotations

import json

from repo2rlenv.tasksmith.author.openai_wire import (
    STOP_REASONS,
    from_openai_response,
    message_events,
    plain_text,
    to_openai_body,
)

# A representative author turn: system, exploration, a tool call and its result.
ANTHROPIC_BODY = {
    "model": "mimo-v2.6-flash",
    "max_tokens": 12000,
    "stream": True,
    "system": [{"type": "text", "text": "Author the artifact.", "cache_control": {}}],
    "messages": [
        {"role": "user", "content": "Start."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "opaque", "signature": "signed"},
                {"type": "text", "text": "Reading the file."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "shell",
                    "input": {"command": "cat src/app.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "contents",
                },
                {"type": "text", "text": "Now submit."},
            ],
        },
    ],
    "tools": [
        {
            "name": "shell",
            "description": "Run remotely",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }
    ],
    "tool_choice": {"type": "auto"},
    "stop_sequences": ["SUBMIT"],
}


def test_plain_text_flattens_text_blocks_only():
    assert plain_text("already text") == "already text"
    assert plain_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert plain_text([{"type": "thinking", "thinking": "x"}]) == ""
    assert plain_text(None) == ""


def test_request_translation_covers_the_persisted_history():
    payload = to_openai_body(ANTHROPIC_BODY, model="mimo-v2.6-flash")

    assert payload["model"] == "mimo-v2.6-flash"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 12000
    assert payload["stop"] == ["SUBMIT"]

    messages = payload["messages"]
    assert messages[0] == {
        "role": "system",
        "content": "Author the artifact.",
    }  # cache_control never migrates
    assert messages[1] == {"role": "user", "content": "Start."}

    assistant = messages[2]
    assert assistant["content"] == "Reading the file."  # thinking drops here
    assert assistant["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": json.dumps({"command": "cat src/app.py"}),
            },
        }
    ]

    # Tool results must trail their assistant call, before the user's own text.
    assert [message["role"] for message in messages[3:5]] == ["tool", "user"]
    assert messages[3]["tool_call_id"] == "toolu_1"
    assert messages[3]["content"] == "contents"
    assert messages[4]["content"] == "Now submit."

    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run remotely",
                "parameters": ANTHROPIC_BODY["tools"][0]["input_schema"],
            },
        }
    ]
    assert payload["tool_choice"] == "auto"


def test_tool_choice_translates_the_shapes_used_by_runtimes():
    with_tools = {"tools": ANTHROPIC_BODY["tools"]}
    assert (
        to_openai_body({**with_tools, "tool_choice": {"type": "any"}}, model="m")["tool_choice"]
        == "required"
    )
    assert (
        to_openai_body({**with_tools, "tool_choice": {"type": "none"}}, model="m")["tool_choice"]
        == "none"
    )
    assert to_openai_body(
        {**with_tools, "tool_choice": {"type": "tool", "name": "shell"}}, model="m"
    )["tool_choice"] == {"type": "function", "function": {"name": "shell"}}
    # The OpenAI contract rejects a choice with nothing to choose from.
    assert "tool_choice" not in to_openai_body({"tool_choice": {"type": "auto"}}, model="m")
    assert "tool_choice" not in to_openai_body({"tools": []}, model="m")


def test_user_message_of_only_failed_results_stays_results():
    payload = to_openai_body(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": "boom",
                            "is_error": True,
                        }
                    ],
                }
            ]
        },
        model="m",
    )
    assert payload["messages"] == [
        {"role": "tool", "tool_call_id": "t", "content": "Tool failed: boom"}
    ]


def test_unknown_role_is_rejected_rather_than_silently_dropped():
    from repo2rlenv.tasksmith.author.openai_wire import _messages

    try:
        _messages([{"role": "system", "content": "x"}])
    except ValueError as exc:
        assert "user or assistant" in str(exc)
    else:  # pragma: no cover - the guard must hold
        raise AssertionError("expected a ValueError for a non-Anthropic role")


COMPLETION = {
    "id": "chatcmpl-1",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "content": "Submitting now.",
                "tool_calls": [
                    {
                        "id": "call_9",
                        "type": "function",
                        "function": {
                            "name": "submit_artifact",
                            "arguments": json.dumps({"profile": "complete"}),
                        },
                    }
                ],
            },
        }
    ],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 400},
    },
}


def test_completion_translates_back_to_an_anthropic_message():
    message = from_openai_response(COMPLETION, model="mimo-v2.6-flash")

    assert message["type"] == "message"
    assert message["role"] == "assistant"
    assert message["stop_reason"] == "tool_use"
    assert message["content"][0] == {"type": "text", "text": "Submitting now."}
    assert message["content"][1] == {
        "type": "tool_use",
        "id": "call_9",
        "name": "submit_artifact",
        "input": {"profile": "complete"},
    }
    # Gateway token accounting in the Anthropic shape the bridge records.
    assert message["usage"] == {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 400,
    }


def test_finished_translations_map_to_the_protocol_stop_reasons():
    message = from_openai_response(
        {"choices": [{"finish_reason": "length", "message": {"content": "x"}}]}, model="m"
    )
    assert message["stop_reason"] == "max_tokens"
    assert message["usage"] == {}
    assert STOP_REASONS["stop"] == "end_turn"
    assert from_openai_response({"choices": []}, model="m")["stop_reason"] == "end_turn"


def test_malformed_tool_arguments_become_an_explicit_input_object():
    """A charged completion is kept usable; the handler's input error reaches the model."""
    message = from_openai_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "c",
                                "type": "function",
                                "function": {"name": "s", "arguments": "{"},
                            },
                            {
                                "id": "d",
                                "type": "function",
                                "function": {"name": "s", "arguments": "[1]"},
                            },
                        ]
                    },
                }
            ]
        },
        model="m",
    )
    assert message["content"][0]["input"] == {"_malformed_arguments": "{"}
    assert message["content"][1]["input"] == {"_malformed_arguments": "[1]"}


def test_completed_message_replays_as_anthropic_sse():
    message = from_openai_response(COMPLETION, model="mimo-v2.6-flash")
    events = [event["type"] for event in message_events(message)]
    assert events == [
        "message_start",
        "content_block_start",  # text
        "content_block_delta",
        "content_block_stop",
        "content_block_start",  # tool_use
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    streamed = message_events(message)
    start = streamed[0]["message"]
    assert start["usage"]["input_tokens"] == 1000
    assert start["usage"]["output_tokens"] == 0
    assert start["content"] == []

    text_delta = streamed[2]["delta"]
    assert text_delta == {"type": "text_delta", "text": "Submitting now."}

    tool_start = streamed[4]["content_block"]
    assert tool_start == {
        "type": "tool_use",
        "id": "call_9",
        "name": "submit_artifact",
        "input": {},
    }
    arguments = json.loads(streamed[5]["delta"]["partial_json"])
    assert arguments == {"profile": "complete"}

    assert streamed[7]["delta"]["stop_reason"] == "tool_use"
    assert streamed[7]["usage"] == {"output_tokens": 500}
    assert streamed[4]["index"] == streamed[5]["index"] == streamed[6]["index"] == 1
