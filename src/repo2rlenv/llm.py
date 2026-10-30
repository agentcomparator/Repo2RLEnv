"""LiteLLM wrapper — single entry point across providers, with cost tracking.

The pipelines call `complete(input, prompt)`; we resolve the API key from
either the LLMSpec hint or the provider-default env var, dispatch, then use
LiteLLM's `completion_cost()` to attach a USD estimate.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from repo2rlenv.auth import LLM_KEY_ENV_DEFAULTS, resolve_llm_api_key
from repo2rlenv.spec.input import OPENAI_COMPAT_BASE_URL_ENV, OPENAI_COMPAT_PROVIDER, LLMSpec

logger = logging.getLogger(__name__)

# Sent with `endpoint` to providers whose client insists on *some* key string
# (`openai/<model>` against vLLM, Ollama, llama.cpp — the server ignores it).
# Same value vLLM's docs use.
_PLACEHOLDER_API_KEY = "EMPTY"


# Models that reject any `temperature` value (forced default). Add patterns
# as new releases land. Probed empirically via scripts/probe_llm_routes.py.
_NO_TEMPERATURE_RE = re.compile(
    r"(claude-opus-4-7|claude-opus-4-8|gpt-5(\.|-|$)|gpt-6|o1-|o3-|o4-)",
    re.IGNORECASE,
)
_OPENAI_COMPLETION_LIMIT_RE = re.compile(r"^(gpt-[56](?:[.-]|$)|o[134](?:-|$))")


def _supports_temperature(model: str) -> bool:
    return _NO_TEMPERATURE_RE.search(model) is None


# Providers the direct-dispatch gates accept: credentials we resolve ourselves
# and a model string a Harbor harness can route (provider prefix + env/api_base).
# One declaration for `harbor.run_trial`, `quality.loop` validators/CLI and the
# FrontierSmith author — add a route here only if all of them can drive it.
DIRECT_PROVIDERS: frozenset[str] = frozenset({"openai", "anthropic", OPENAI_COMPAT_PROVIDER})


def agent_credential_env(spec: LLMSpec, key: str) -> dict[str, str]:
    """Credential env for a sandboxed harness.

    `openai_compat` is injected under OpenAI's names (`OPENAI_API_KEY` +
    `OPENAI_BASE_URL`) because Harbor's model connection, LiteLLM's `openai/`
    route and the OpenAI SDK all resolve those two; every other provider keeps
    the historical `<PROVIDER>_API_KEY` shape.
    """
    if is_openai_compat(spec):
        return {"OPENAI_API_KEY": key, "OPENAI_BASE_URL": openai_compat_api_base(spec)}
    return {spec.provider.upper() + "_API_KEY": key}


def is_openai_compat(spec: LLMSpec) -> bool:
    """Whether this spec is our alias for an OpenAI-compatible gateway."""
    return spec.provider.lower() == OPENAI_COMPAT_PROVIDER


def wire_model(name: str) -> str:
    """`wire_model_name` for a rendered `qualified_name` string.

    Receipts and trial records store the alias (`openai_compat/<model>`); recovery
    paths that rebuild a Harness config from a stored record use this.
    """
    if name.lower().startswith(f"{OPENAI_COMPAT_PROVIDER}/"):
        return "openai/" + name.split("/", 1)[1]
    return name


def wire_model_name(spec: LLMSpec) -> str:
    """The `provider/model` string to hand LiteLLM / a Harbor harness.

    `openai_compat/<model>` has no LiteLLM provider prefix, so it goes out as the
    OpenAI wire contract (`openai/<model>`); LiteLLM's route fires against the
    base URL we set from `$OPENAI_COMPAT_BASE_URL`. Everything else passes through
    as `qualified_name` — receipt identity still records the alias.
    """
    return wire_model(spec.qualified_name)


def openai_compat_api_base(spec: LLMSpec) -> str:
    """Base URL for `openai_compat`; `--llm-endpoint` beats `$OPENAI_COMPAT_BASE_URL`."""
    if spec.endpoint:
        return spec.endpoint
    url = os.environ.get(OPENAI_COMPAT_BASE_URL_ENV, "").strip()
    if not url:
        raise RuntimeError(
            f"no base URL for provider {spec.provider!r}: ${OPENAI_COMPAT_BASE_URL_ENV} is unset. "
            "Set it to the gateway root (e.g. https://host/v1), or pass "
            "--llm-endpoint URL (or llm.endpoint in config)."
        )
    return url


def completion_token_limit(spec: LLMSpec, max_tokens: int) -> dict[str, int]:
    """Provider wire parameters shared with externally orchestrated Harbor calls."""
    # `openai_compat` sends the OpenAI contract, so its model ids inherit
    # OpenAI's parameter names whatever the gateway fronts.
    name = (
        "max_completion_tokens"
        if spec.provider in ("openai", OPENAI_COMPAT_PROVIDER)
        and _OPENAI_COMPLETION_LIMIT_RE.match(spec.model)
        else "max_tokens"
    )
    return {name: max_tokens}


@dataclass(slots=True)
class LLMResponse:
    content: str
    usage: dict | None = None
    cost_usd: float = 0.0  # cost of THIS call, in USD (best-effort)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _is_failover_eligible(exc: BaseException) -> bool:
    """True for transient provider errors worth retrying on a fallback model.

    LiteLLM raises specific subclasses; we match on class names so we don't
    have to import the symbols (some live in nested submodules and shift
    between versions).
    """
    name = type(exc).__name__
    # Retry on: 5xx upstream, rate limits, network blips, timeouts.
    # Don't retry on: 4xx bad-request, auth errors, not-found, content filter.
    return name in {
        "InternalServerError",  # 5xx incl. Anthropic 529 Overloaded
        "RateLimitError",  # 429
        "ServiceUnavailableError",
        "APIConnectionError",
        "Timeout",
        "APIError",  # generic upstream error
    }


def _resolve_api_key(spec: LLMSpec) -> str | None:
    """The API key to hand LiteLLM, or None to let LiteLLM resolve it.

    - `api_key_env` given → its value; raise if unset (never falls through)
    - `endpoint` given → the provider-default key is *not* forwarded to a
      custom server. `LLM_KEY_ENV_DEFAULTS` clients (`openai/`, …) need a key
      string, so they get the placeholder; `hosted_vllm/`, `ollama/`, … get
      None and LiteLLM applies its own `HOSTED_VLLM_API_KEY`-style lookup.
    - otherwise → provider-default key; raise naming the var for
      `LLM_KEY_ENV_DEFAULTS` providers, None (LiteLLM's call) for the rest
    """
    provider = spec.provider.lower()
    if provider == OPENAI_COMPAT_PROVIDER:
        # Our alias owns the credential: `$OPENAI_COMPAT_API_TOKEN` (or an explicit
        # `api_key_env`) is always sent to the gateway — never a placeholder, and
        # the `endpoint` branch below must not warn and strip it.
        key = resolve_llm_api_key(provider, spec.api_key_env)
        if key is None:
            raise RuntimeError(
                f"no API key for provider {spec.provider!r}: "
                f"${spec.api_key_env or LLM_KEY_ENV_DEFAULTS[provider]} is unset."
            )
        return key
    if spec.api_key_env:
        key = resolve_llm_api_key(provider, spec.api_key_env)
        if key is None:
            raise RuntimeError(
                f"no API key for provider {spec.provider!r}: ${spec.api_key_env} is unset."
            )
        return key
    if spec.endpoint:
        if provider in LLM_KEY_ENV_DEFAULTS and resolve_llm_api_key(provider):
            key_env = LLM_KEY_ENV_DEFAULTS[provider]
            logger.warning(
                "$%s is set but is not sent to the custom LLM endpoint. "
                "Use --llm-key-env %s (or llm.api_key_env in config) "
                "if this endpoint should receive that key.",
                key_env,
                key_env,
            )
        return _PLACEHOLDER_API_KEY if provider in LLM_KEY_ENV_DEFAULTS else None
    key = resolve_llm_api_key(provider)
    if key is None and provider in LLM_KEY_ENV_DEFAULTS:
        raise RuntimeError(
            f"no API key resolved for provider {spec.provider!r}. Set "
            f"{LLM_KEY_ENV_DEFAULTS[provider]}, or use --llm-endpoint for a self-hosted server."
        )
    return key


def check_provider(spec: LLMSpec) -> None:
    """Raise early if LiteLLM doesn't know a provider in the spec's fallback chain.

    Cheap next to what follows (clone, image pull, agent loop), so the bootstrap
    entry points call it before any of that starts.
    """
    import litellm  # type: ignore[import-untyped]

    cur: LLMSpec | None = spec
    while cur is not None:
        # Validates the wire name, not the alias: `openai_compat/...` is routed as
        # `openai/...` at call time, so preflight must ask about that prefix too.
        wire = wire_model_name(cur)
        try:
            litellm.get_llm_provider(wire)
        except Exception as exc:
            first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            raise RuntimeError(f"unknown LLM provider in {wire!r}: {first_line}") from exc
        cur = cur.fallback


def _do_complete(
    spec: LLMSpec,
    *,
    system: str | None,
    user: str,
    max_tokens: int,
    temperature: float,
    response_schema: dict | None = None,
) -> LLMResponse:
    """One non-fallback chat-completion call. Internal helper for `complete()`."""
    import litellm  # type: ignore[import-untyped]

    api_key = _resolve_api_key(spec)

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    kwargs: dict = {
        "model": wire_model_name(spec),
        "messages": messages,
        "num_retries": 0,
        "timeout": spec.timeout_sec,
    }
    # Unknown-to-LiteLLM model IDs still need the current OpenAI wire contract.
    # Do not rely on the installed SDK's model catalog to rename this parameter.
    kwargs.update(completion_token_limit(spec, max_tokens))
    if api_key is not None:
        kwargs["api_key"] = api_key
    # Newer reasoning-focused models (Opus 4.7+, GPT-5+) reject `temperature`.
    if _supports_temperature(spec.model):
        kwargs["temperature"] = temperature
    if spec.endpoint:
        kwargs["api_base"] = spec.endpoint
    elif is_openai_compat(spec):
        kwargs["api_base"] = openai_compat_api_base(spec)
    if response_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "repo2rlenv_response",
                "strict": True,
                "schema": response_schema,
            },
        }

    if spec.provider == "huggingface" and spec.endpoint is None:
        kwargs.setdefault("api_base", "https://router.huggingface.co/v1")

    response = litellm.completion(**kwargs)
    choice = response.choices[0]
    content = choice.message.content or ""

    usage_obj = getattr(response, "usage", None)
    prompt_tokens = 0
    completion_tokens = 0
    if usage_obj is not None:
        prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage_obj, "completion_tokens", 0) or 0

    cost_usd = 0.0
    try:
        cost_usd = float(litellm.completion_cost(completion_response=response))
    except Exception as exc:
        logger.debug("completion_cost failed for %s: %s", spec.qualified_name, exc)

    return LLMResponse(
        content=content,
        usage=dict(usage_obj) if usage_obj else None,
        cost_usd=cost_usd,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def complete(
    spec: LLMSpec,
    *,
    system: str | None = None,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    response_schema: dict | None = None,
    _depth: int = 0,
) -> LLMResponse:
    """Single chat-completion call with automatic fallback on transient errors.

    Calls `spec.qualified_name`. On 5xx / 429 / network / timeout errors, if
    `spec.fallback` is set, retries with the fallback model recursively (up
    to 3 levels deep, then re-raises). 4xx errors (bad model, auth, etc.)
    are NOT retried — those signal config bugs, not transient failures.

    Honors `LLMSpec.endpoint` for self-hosted backends.
    """
    try:
        return _do_complete(
            spec,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
        )
    except Exception as exc:
        if _depth >= 3 or spec.fallback is None or not _is_failover_eligible(exc):
            raise
        logger.warning(
            "primary LLM %s failed with %s; falling back to %s",
            spec.qualified_name,
            type(exc).__name__,
            spec.fallback.qualified_name,
        )
        return complete(
            spec.fallback,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
            _depth=_depth + 1,
        )
