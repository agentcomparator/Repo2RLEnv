"""Where `openai_compat` lands outside the plain generate/bootstrap path.

One declaration (`llm.DIRECT_PROVIDERS`) gates blind Harbor trials, the
quality-loop validators/CLI and the FrontierSmith author; `llm.agent_credential_env`
decides the env names a sandboxed harness receives. These tests pin both.
"""

from __future__ import annotations

import argparse
import os
from unittest import mock

import pytest
from pydantic import BaseModel

from repo2rlenv.llm import (
    DIRECT_PROVIDERS,
    agent_credential_env,
    wire_model,
)
from repo2rlenv.quality.loop.cli import model_spec
from repo2rlenv.quality.loop.models import LoopOptions
from repo2rlenv.spec.input import LLMSpec

BASE_URL = "https://gateway.internal/v1"

TRIAD = {
    "OPENAI_COMPAT_BASE_URL": BASE_URL,
    "OPENAI_COMPAT_API_TOKEN": "gw-token",
    "OPENAI_COMPAT_MODEL": "llama-3.3-70b-instruct",
}


# --- the declaration and the credential mapping ---------------------------------


def test_direct_providers_hold_the_three_driveable_routes():
    assert frozenset({"openai", "anthropic", "openai_compat"}) == DIRECT_PROVIDERS


def test_compat_credentials_land_under_openai_names():
    spec = LLMSpec(provider="openai_compat", model="m")
    with mock.patch.dict(os.environ, TRIAD, clear=True):
        assert agent_credential_env(spec, "tok") == {
            "OPENAI_API_KEY": "tok",
            "OPENAI_BASE_URL": BASE_URL,
        }


def test_other_direct_providers_keep_their_own_key_name():
    spec = LLMSpec(provider="anthropic", model="claude-sonnet-4-6")
    assert agent_credential_env(spec, "sk-ant") == {"ANTHROPIC_API_KEY": "sk-ant"}


def test_compat_credentials_require_a_gateway_url_first():
    spec = LLMSpec(provider="openai_compat", model="m")
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="OPENAI_COMPAT_BASE_URL"):
            agent_credential_env(spec, "tok")


def test_wire_model_maps_stored_receipt_names_back():
    assert wire_model("openai_compat/llama-3.3-70b-instruct") == "openai/llama-3.3-70b-instruct"
    assert wire_model("openai_compat/org/model") == "openai/org/model"
    assert wire_model("anthropic/claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"


# --- quality-loop CLI -------------------------------------------------------------


def test_model_spec_accepts_alias_with_and_without_a_model():
    spec = model_spec("openai_compat/meta-llama/llama-3.3-70b")
    assert (spec.provider, spec.model) == ("openai_compat", "meta-llama/llama-3.3-70b")

    with mock.patch.dict(os.environ, TRIAD, clear=True):
        bare = model_spec("openai_compat")
        assert bare.model == "llama-3.3-70b-instruct"  # $OPENAI_COMPAT_MODEL filled in


def test_model_spec_still_rejects_routes_it_cannot_drive():
    with pytest.raises(argparse.ArgumentTypeError, match="openai/MODEL"):
        model_spec("bedrock/claude-sonnet-4-6")
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(argparse.ArgumentTypeError, match="OPENAI_COMPAT_MODEL"):
            model_spec("openai_compat")  # no $OPENAI_COMPAT_MODEL to fall back on


def test_loop_options_accepts_the_alias_without_a_fallback():
    options = LoopOptions(solver_model=LLMSpec(provider="openai_compat", model="m"))
    assert options.solver_model.provider == "openai_compat"


def test_loop_options_still_rejects_unknown_routes_and_silent_fallbacks():
    with pytest.raises(ValueError, match="direct OpenAI/Anthropic routes or openai_compat"):
        LoopOptions(solver_model=LLMSpec(provider="groq", model="m"))
    with pytest.raises(ValueError, match="direct OpenAI/Anthropic routes or openai_compat"):
        LoopOptions(
            solver_model=LLMSpec(
                provider="openai_compat",
                model="m",
                fallback=LLMSpec(provider="anthropic", model="claude-sonnet-4-6"),
            )
        )


# --- FrontierSmith author gate ------------------------------------------------------


class _Schema(BaseModel):
    issue: str


def _author(spec):
    from repo2rlenv.pipelines.recipes.frontiersmith.author import author

    return author(
        spec,
        _Schema,
        prompt="p",
        payload={},
        path=mock.MagicMock(exists=lambda: False),  # no stored receipt
        ledger=mock.MagicMock(),
        operation="op",
        max_tokens=100,
        resume=False,
    )


def test_frontiersmith_admits_the_alias_through_the_gate(monkeypatch):
    """A priced model on the alias passes the gate: the failure we see is the
    missing credential (resolved before any reservation), not a route rejection."""
    monkeypatch.delenv("OPENAI_COMPAT_API_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", BASE_URL)
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        _author(LLMSpec(provider="openai_compat", model="gpt-6-sol"))


def test_frontiersmith_still_rejects_other_providers_and_fallbacks():
    with pytest.raises(ValueError, match="FrontierSmith requires OpenAI"):
        _author(LLMSpec(provider="anthropic", model="claude-sonnet-4-6"))
    with pytest.raises(ValueError, match="without implicit fallbacks"):
        _author(
            LLMSpec(
                provider="openai_compat",
                model="gpt-6-sol",
                fallback=LLMSpec(provider="openai", model="gpt-6-luna"),
            )
        )


def test_frontiersmith_pricing_contract_still_gates_the_model():
    with pytest.raises(ValueError, match="This runtime requires openai/gpt-6"):
        _author(LLMSpec(provider="openai_compat", model="llama-3.3-70b-instruct"))
