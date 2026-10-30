"""`openai_compat` — the alias for an OpenAI-compatible gateway.

Env triad: `$OPENAI_COMPAT_BASE_URL` / `$OPENAI_COMPAT_API_TOKEN` /
`$OPENAI_COMPAT_MODEL`. No live LLM and no real `litellm` import: `_do_complete`
imports it lazily, so a stub module in `sys.modules` stands in and its
`completion` kwargs are inspected — the whole point is what goes on the wire.
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from repo2rlenv import cli as cli_mod
from repo2rlenv.llm import _do_complete, check_provider
from repo2rlenv.spec.input import LLMSpec

BASE_URL = "https://gateway.internal/v1"
TOKEN = "gw-live-token"
MODEL = "llama-3.3-70b-instruct"

TRIAD = {
    "OPENAI_COMPAT_BASE_URL": BASE_URL,
    "OPENAI_COMPAT_API_TOKEN": TOKEN,
    "OPENAI_COMPAT_MODEL": MODEL,
}


def argparse_namespace(**kwargs) -> argparse.Namespace:
    base = {"llm": None, "llm_fallback": None, "llm_endpoint": None, "llm_key_env": None}
    base.update(kwargs)
    return argparse.Namespace(**base)


def _fake_response(content: str = "PONG") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
    )


@pytest.fixture
def litellm_stub(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    stub = SimpleNamespace(
        completion=mock.MagicMock(return_value=_fake_response()),
        completion_cost=lambda **kw: 0.0,
        get_llm_provider=mock.MagicMock(return_value=("m", "openai", None, None)),
    )
    monkeypatch.setitem(sys.modules, "litellm", stub)
    return stub


def _kwargs(stub: SimpleNamespace) -> dict:
    assert stub.completion.call_count == 1
    return stub.completion.call_args.kwargs


# --- the wire contract ---------------------------------------------------------


@mock.patch.dict(os.environ, TRIAD, clear=True)
def test_env_triad_reaches_litellm_as_openai_wire(litellm_stub):
    spec = LLMSpec(provider="openai_compat", model="")
    assert spec.model == MODEL  # $OPENAI_COMPAT_MODEL filled in
    assert spec.qualified_name == f"openai_compat/{MODEL}"  # receipts keep the alias

    _do_complete(spec, system=None, user="ping", max_tokens=8, temperature=0.0)
    kw = _kwargs(litellm_stub)
    assert kw["model"] == f"openai/{MODEL}"  # what LiteLLM routes on
    assert kw["api_base"] == BASE_URL
    assert kw["api_key"] == TOKEN
    assert kw["api_base"] not in (None, ""), kw
    assert "max_tokens" in kw


@mock.patch.dict(os.environ, {**TRIAD, "OPENAI_COMPAT_BASE_URL": "https://other/v1"}, clear=True)
def test_endpoint_beats_env_base_url(litellm_stub):
    spec = LLMSpec(provider="openai_compat", model=MODEL, endpoint="http://127.0.0.1:9/v1")
    _do_complete(spec, system=None, user="ping", max_tokens=8, temperature=0.0)
    assert _kwargs(litellm_stub)["api_base"] == "http://127.0.0.1:9/v1"


@mock.patch.dict(os.environ, {**TRIAD, "OPENAI_COMPAT_BASE_URL": ""}, clear=True)
def test_missing_base_url_fails_fast_naming_the_var(litellm_stub):
    spec = LLMSpec(provider="openai_compat", model=MODEL)
    with pytest.raises(RuntimeError, match=r"\$OPENAI_COMPAT_BASE_URL is unset"):
        _do_complete(spec, system=None, user="ping", max_tokens=8, temperature=0.0)
    litellm_stub.completion.assert_not_called()


@mock.patch.dict(os.environ, {**TRIAD, "OPENAI_COMPAT_API_TOKEN": ""}, clear=True)
def test_missing_token_fails_fast_naming_the_var(litellm_stub):
    spec = LLMSpec(provider="openai_compat", model=MODEL)
    with pytest.raises(RuntimeError, match=r"\$OPENAI_COMPAT_API_TOKEN is unset"):
        _do_complete(spec, system=None, user="ping", max_tokens=8, temperature=0.0)
    litellm_stub.completion.assert_not_called()


def test_blank_token_never_sends_empty_string(litellm_stub):
    """`.env.example` ships blank values; a blank token must raise, not travel."""
    env = {**TRIAD, "OPENAI_COMPAT_API_TOKEN": ""}
    with mock.patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="OPENAI_COMPAT_API_TOKEN"):
            _do_complete(
                LLMSpec(provider="openai_compat", model=MODEL),
                system=None,
                user="ping",
                max_tokens=8,
                temperature=0.0,
            )
    litellm_stub.completion.assert_not_called()


@mock.patch.dict(os.environ, TRIAD, clear=True)
def test_explicit_api_key_env_wins_over_default_token(litellm_stub):
    with mock.patch.dict(os.environ, {"OTHER_GATEWAY_KEY": "second"}):
        spec = LLMSpec(provider="openai_compat", model=MODEL, api_key_env="OTHER_GATEWAY_KEY")
        _do_complete(spec, system=None, user="ping", max_tokens=8, temperature=0.0)
        assert _kwargs(litellm_stub)["api_key"] == "second"


# --- input contract -------------------------------------------------------------


@mock.patch.dict(
    os.environ, {k: v for k, v in TRIAD.items() if k != "OPENAI_COMPAT_MODEL"}, clear=True
)
def test_alias_without_model_names_the_env_var():
    with pytest.raises(ValueError, match=r"\$OPENAI_COMPAT_MODEL"):
        LLMSpec(provider="openai_compat", model="")


@mock.patch.dict(os.environ, {}, clear=True)
def test_plain_providers_still_require_a_model():
    with pytest.raises(ValueError, match=r"llm\.model is required for provider 'anthropic'"):
        LLMSpec(provider="anthropic")


def test_bare_cli_flag_becomes_empty_model_for_the_spec_to_fill():
    assert cli_mod._split_provider_model("openai_compat", "--llm") == ("openai_compat", "")
    # ...and a slash with an explicit model still wins over the env var
    assert cli_mod._split_provider_model("openai_compat/m", "--llm") == ("openai_compat", "m")


@mock.patch.dict(os.environ, TRIAD, clear=True)
def test_bare_cli_flag_builds_a_valid_spec():
    args = argparse_namespace(llm="openai_compat")
    overrides = cli_mod._llm_overrides(args)
    assert overrides == {"provider": "openai_compat", "model": ""}
    spec = LLMSpec(**overrides)
    assert spec.model == MODEL


# --- preflight + parameters -------------------------------------------------------


@mock.patch.dict(os.environ, TRIAD, clear=True)
def test_check_provider_preflights_the_wire_name(litellm_stub):
    check_provider(LLMSpec(provider="openai_compat", model=MODEL))
    seen = [c.args[0] for c in litellm_stub.get_llm_provider.call_args_list]
    assert seen == [f"openai/{MODEL}"]


@mock.patch.dict(os.environ, {**TRIAD, "OPENAI_COMPAT_MODEL": "gpt-5.6-sol"}, clear=True)
def test_openai_wire_models_inherit_openai_parameter_names(litellm_stub):
    spec = LLMSpec(provider="openai_compat", model="")
    _do_complete(spec, system=None, user="ping", max_tokens=4096, temperature=0.7)
    kw = _kwargs(litellm_stub)
    assert kw["max_completion_tokens"] == 4096
    assert "max_tokens" not in kw
    assert "temperature" not in kw  # gpt-5+ rejects it


@mock.patch.dict(os.environ, TRIAD, clear=True)
def test_fallback_chain_preflights_both_wire_names(litellm_stub):
    fallback = LLMSpec(provider="openai", model="gpt-5.5")
    spec = LLMSpec(provider="openai_compat", model=MODEL, fallback=fallback)
    check_provider(spec)
    seen = [c.args[0] for c in litellm_stub.get_llm_provider.call_args_list]
    assert seen == [f"openai/{MODEL}", "openai/gpt-5.5"]
