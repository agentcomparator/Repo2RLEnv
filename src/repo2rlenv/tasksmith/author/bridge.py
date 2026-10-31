from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
from pathlib import Path

from aiohttp import (
    ClientConnectionError,
    ClientError,
    ClientPayloadError,
    ClientSession,
    ClientTimeout,
    web,
)

from repo2rlenv.auth import resolve_llm_api_key
from repo2rlenv.campaigns.budget import BudgetExceeded
from repo2rlenv.llm import openai_compat_api_base, resolve_model_alias
from repo2rlenv.spec.input import (
    OPENAI_COMPAT_API_TOKEN_ENV,
    OPENAI_COMPAT_PROVIDER,
    LLMSpec,
)
from repo2rlenv.tasksmith.author import openai_wire
from repo2rlenv.tasksmith.author.budget import AuthorBudget as Budget
from repo2rlenv.tasksmith.author.inference import (
    MAX_OUTPUT_TOKENS,
    MODEL_TIMEOUT_SEC,
    anthropic_options,
    inference_settings,
)


class ProviderOutputError(RuntimeError):
    """Completed, metered response was unusable; a bounded new attempt is safe."""


# Gateways occasionally hold a request without a single byte for minutes while
# normal completions answer within seconds (observed live). A silent window is
# therefore treated as the transport-class transient `llm.complete` already
# retries: one fresh attempt through full admission, and the silent attempt is
# left RESERVED (never settled at zero — a server-side generation may still be
# billed) for `campaign settle`. The windows stay under MODEL_TIMEOUT_SEC so
# the runtime adapter never cancels during recovery, and under the stage
# deadline (runner.author) so recovery plus completion still fit the stage.
STALL_ATTEMPT_TIMEOUTS = (240, 560)


class AgentBridge:
    """Only model inference and explicitly registered remote tools cross this API.

    Runtime processes receive a short-lived loopback token, never provider keys.
    Requests are metered before forwarding; incomplete streams retain reservations.
    """

    def __init__(self, *, model, budget: Budget, tools, handlers, trace: Path, max_turns, max_cost):
        # The runtime always negotiates Anthropic's Messages protocol with the
        # bridge; this route decides which wire the bridge itself speaks upstream.
        model = resolve_model_alias(model)
        provider, _, name = model.partition("/")
        if provider not in ("anthropic", OPENAI_COMPAT_PROVIDER) or not name:
            raise ValueError(
                "The matched runtime experiment supports anthropic/MODEL and "
                "openai_compat[/MODEL] author models"
            )
        self.model, self.model_id = model, name
        self.compat = provider == OPENAI_COMPAT_PROVIDER
        self.gateway_url = self.gateway_headers = None
        if self.compat:
            spec = LLMSpec(provider=OPENAI_COMPAT_PROVIDER, model=name)
            token = resolve_llm_api_key(spec.provider, spec.api_key_env)
            if token is None:
                raise ValueError(
                    f"The gateway author route needs a token: "
                    f"${OPENAI_COMPAT_API_TOKEN_ENV} is unset"
                )
            self.gateway_url = openai_compat_api_base(spec)
            self.gateway_headers = {"Authorization": "Bearer " + token}
        self.rates = None
        self.budget = budget
        self.tools, self.handlers, self.trace = tools, handlers, trace
        self.max_turns, self.max_cost = max_turns, max_cost
        self.turns, self.cost = 0, 0.0
        self.start_spend = budget.spent
        self.token = secrets.token_urlsafe(32)
        self.failure: BaseException | None = None
        self.failed = asyncio.Event()
        self.tool_lock = asyncio.Lock()
        self.model_lock = asyncio.Lock()
        self._closing = False
        self._tool_tasks: set[asyncio.Task] = set()
        self._model_tasks: set[asyncio.Task] = set()

    def record(self, kind, **data):
        with self.trace.open("a") as f:
            f.write(json.dumps({"kind": kind, **data}, default=str) + "\n")

    async def __aenter__(self):
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        self.http = ClientSession(timeout=ClientTimeout(total=MODEL_TIMEOUT_SEC))
        app = web.Application(client_max_size=8 * 1024 * 1024)
        app.router.add_post("/v1/messages", self.messages)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/tool", self.tool)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    def stop(self, *, cancel_models=False):
        """Reject new work and cancel tool effects before terminating the runtime."""
        self._closing = True
        tasks = self._tool_tasks | (self._model_tasks if cancel_models else set())
        current = asyncio.current_task()
        for task in tasks:
            if task is not current:
                task.cancel()

    async def __aexit__(self, exc_type, exc, tb):
        self.stop(cancel_models=exc_type is not None)
        try:
            # A runtime can exit as soon as it sees message_stop, before the
            # upstream stream reaches EOF and its final charge is recorded.
            await asyncio.gather(*self._tool_tasks, *self._model_tasks, return_exceptions=True)
        finally:
            try:
                await self.runner.cleanup()
            finally:
                await self.http.close()

    def ensure_open(self):
        if self._closing:
            raise web.HTTPServiceUnavailable(text="Runtime bridge is stopping")

    def authenticate(self, request):
        token = request.headers.get("x-api-key") or request.headers.get(
            "Authorization", ""
        ).removeprefix("Bearer ")
        if not secrets.compare_digest(token, self.token):
            raise web.HTTPUnauthorized()

    def fail(self, exc):
        if self.failure is None:
            self.failure = exc
            self.failed.set()
            self.record("runtime_error", error=f"{type(exc).__name__}: {exc}")
        self.stop(cancel_models=True)

    async def models(self, request):
        self.authenticate(request)
        return web.json_response(
            {"data": [{"id": self.model_id, "type": "model"}], "has_more": False}
        )

    async def tool(self, request):
        self.authenticate(request)
        self.ensure_open()
        preceding_models = tuple(self._model_tasks)
        task = asyncio.current_task()
        self._tool_tasks.add(task)
        try:
            data = await request.json()
            name, arguments = data.get("name"), data.get("arguments", {})
            if name not in self.handlers or not isinstance(arguments, dict):
                raise web.HTTPBadRequest(text="Unknown tool or malformed arguments")
            artifact_stage = "submit_artifact" in self.handlers
            remaining = self.max_turns - self.turns
            try:
                # Native runtimes can dispatch tools at message_stop before
                # provider EOF. Match LangGraph: meter and validate that model
                # response before allowing any effect. Cancelling a waiting
                # tool must not cancel a model that still needs settlement.
                await asyncio.gather(
                    *(asyncio.shield(model) for model in preceding_models),
                    return_exceptions=True,
                )
                async with self.tool_lock:
                    self.ensure_open()
                    remaining = self.max_turns - self.turns
                    if artifact_stage and name == "shell" and remaining <= 2:
                        output = (
                            "Command not executed: the final two model calls are reserved for "
                            "artifact submission and schema correction. Submit the best supported "
                            "complete artifact using the evidence already collected."
                        )
                    else:
                        output = await self.handlers[name](**arguments)
            except web.HTTPException:
                raise
            except (ValueError, TypeError, KeyError) as exc:
                output = f"Tool input error: {exc}"
            except Exception as exc:
                self.fail(exc)
                return web.json_response({"error": str(exc)}, status=409)
            if len(output) > 24000:
                output = output[:8000] + "\n[truncated]\n" + output[-16000:]
            if artifact_stage and name == "shell":
                output += (
                    f"\nController budget: {max(0, remaining)} model calls remain. "
                    "Reserve the final two for submit_artifact and any needed revise_artifact."
                )
            self.record("tool", name=name, arguments=arguments, output=output)
            return web.json_response({"output": output})
        finally:
            self._tool_tasks.discard(task)

    async def messages(self, request):
        self.authenticate(request)
        self.ensure_open()
        task = asyncio.current_task()
        self._model_tasks.add(task)
        try:
            return await self._messages(request)
        except asyncio.CancelledError:
            self.fail(RuntimeError("Provider request cancelled before completion"))
            raise
        except web.HTTPException:
            raise
        except Exception as exc:
            self.fail(exc)
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
                status=429 if isinstance(exc, BudgetExceeded) else 502,
            )
        finally:
            self._model_tasks.discard(task)

    async def _messages(self, request):
        body = await request.json()
        if body.get("model") != self.model_id:
            raise ValueError("Runtime requested a different model than the matched experiment")
        allowed = {t["function"]["name"] for t in self.tools}
        if any(t.get("name") not in allowed for t in body.get("tools", [])):
            raise ValueError("Runtime exposed tools outside the matched allowlist")
        body["max_tokens"] = MAX_OUTPUT_TOKENS
        # Runtime defaults cannot change the matched inference policy. Preserve
        # signed/opaque thinking blocks in the conversation exactly as supplied.
        for key in ("thinking", "temperature", "top_p", "top_k", "output_config"):
            body.pop(key, None)
        body.update(anthropic_options(self.model))
        normalize_cache(body)
        stream = bool(body.get("stream"))
        forward = openai_wire.to_openai_body(body, model=self.model_id) if self.compat else body
        turn, reservation, ceiling = await self._admit(body, forward)
        self.record(
            "model_request",
            turn=turn,
            model=self.model_id,
            system=body.get("system"),
            messages=body.get("messages", [])[-1:],
            tools=[t.get("name") for t in body.get("tools", [])],
            tool_choice=body.get("tool_choice"),
            inference=inference_settings(self.model),
            timeout_sec=MODEL_TIMEOUT_SEC,
        )
        if self.compat:
            return await self._gateway(
                request,
                body,
                forward,
                stream=stream,
                turn=turn,
                reservation=reservation,
                ceiling=ceiling,
            )
        return await self._anthropic(
            request, body, stream=stream, turn=turn, reservation=reservation, ceiling=ceiling
        )

    async def _admit(self, body, forward):
        """Reserve the next model turn; the first attempt and a stall retry share it.

        Both go through the same turn, cost and scope checks, so a retry cannot
        outspend the stage allowance — a retained stalled reservation counts
        against `spent` until it is reconciled.
        """
        input_rate, output_rate, _ = await self._rates()
        ceiling = (len(json.dumps(forward).encode()) + 4096) * input_rate * 1.25 + body[
            "max_tokens"
        ] * output_rate
        async with self.model_lock:
            self.ensure_open()
            if self.turns >= self.max_turns:
                raise BudgetExceeded(f"Runtime model-call limit reached: {self.max_turns}")
            spent = self.budget.spent - self.start_spend
            if spent + ceiling > self.max_cost:
                raise BudgetExceeded(
                    f"Runtime agent budget ${self.max_cost}: ${spent:.4f} committed; "
                    f"need ${ceiling:.4f}"
                )
            reservation = self.budget.reserve(ceiling, f"llm:{self.model}")
            self.turns += 1
            turn = self.turns
        return turn, reservation, ceiling

    async def _anthropic(self, request, body, *, stream, turn, reservation, ceiling):
        """The matched experiment's upstream: its own protocol, forwarded verbatim."""
        input_rate, output_rate, _ = await self._rates()
        headers = {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if request.headers.get("anthropic-beta"):
            headers["anthropic-beta"] = request.headers["anthropic-beta"]
        async with self.http.post(
            "https://api.anthropic.com/v1/messages", json=body, headers=headers
        ) as upstream:
            if upstream.status != 200:
                error = await upstream.text()
                if 400 <= upstream.status < 500:
                    self.budget.settle(reservation, 0)
                raise RuntimeError(f"Anthropic HTTP {upstream.status}: {error[:1500]}")
            if not stream:
                result = await upstream.json()
                usage = result.get("usage", {})
                try:
                    cost = usage_cost(usage, input_rate, output_rate)
                except ValueError:
                    self.record(
                        "model",
                        turn=turn,
                        message=result,
                        usage=usage,
                        cost_usd=None,
                        reservation_usd=ceiling,
                    )
                    raise
                self.budget.settle(reservation, cost)
                self.cost += cost
                self.record("model", turn=turn, message=result, usage=usage, cost_usd=cost)
                validate_provider_response(result.get("content", []), result.get("stop_reason"))
                return web.json_response(result)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            events, pending = [], b""
            async for chunk in upstream.content.iter_any():
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line.startswith(b"data: "):
                        with contextlib.suppress(ValueError):
                            events.append(json.loads(line[6:]))
                with contextlib.suppress(ConnectionError):
                    await response.write(chunk)
            usage = {}
            content = []
            stop_reason = None
            for event in events:
                if event.get("type") == "message_start":
                    message = event.get("message", {})
                    usage.update(message.get("usage", {}))
                    content.extend(message.get("content", []))
                    stop_reason = event.get("stop_reason") or stop_reason
                elif event.get("type") == "message_delta":
                    usage.update(event.get("usage", {}))
                    stop_reason = event.get("delta", {}).get("stop_reason") or stop_reason
                elif event.get("type") == "content_block_start":
                    content.append(event.get("content_block", {}))
                elif event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        content.append({"type": "text", "text": delta.get("text", "")})
            try:
                if not any(e.get("type") == "message_stop" for e in events):
                    raise RuntimeError("Incomplete provider stream; reservation retained")
                cost = usage_cost(usage, input_rate, output_rate)
            except (ValueError, RuntimeError):
                self.record(
                    "model",
                    turn=turn,
                    events=events,
                    usage=usage,
                    cost_usd=None,
                    reservation_usd=ceiling,
                )
                raise
            self.budget.settle(reservation, cost)
            self.cost += cost
            self.record("model", turn=turn, events=events, usage=usage, cost_usd=cost)
            validate_provider_response(content, stop_reason)
            with contextlib.suppress(ConnectionError):
                await response.write_eof()
            return response

    async def _gateway(self, request, body, forward, *, stream, turn, reservation, ceiling):
        """One OpenAI-wire turn against the gateway, metered before the runtime sees it.

        The gateway is called non-streaming — one billed completion per turn —
        and the completed message is replayed to the runtime in Anthropic's SSE
        shape, so there are no deltas to reconcile and the turn settles once.
        A silent window (no complete response within one attempt timeout) is
        retried once through full admission; see STALL_ATTEMPT_TIMEOUTS for why
        the stalled operation stays reserved instead of being settled.
        """
        url = self.gateway_url.rstrip("/") + "/chat/completions"
        attempt = 0
        while True:
            attempt += 1
            try:
                async with self.http.post(
                    url,
                    json=forward,
                    headers=self.gateway_headers,
                    timeout=ClientTimeout(total=STALL_ATTEMPT_TIMEOUTS[attempt - 1]),
                ) as upstream:
                    if upstream.status != 200:
                        error = await upstream.text()
                        if 400 <= upstream.status < 500:
                            # A rejected request never reached generation.
                            self.budget.settle(reservation, 0)
                        raise RuntimeError(
                            f"{OPENAI_COMPAT_PROVIDER} HTTP {upstream.status}: {error[:1500]}"
                        )
                    data = await upstream.json()
                break
            except (TimeoutError, ClientConnectionError, ClientPayloadError) as exc:
                self.record(
                    "provider_stall",
                    turn=turn,
                    attempt=attempt,
                    timeout_sec=STALL_ATTEMPT_TIMEOUTS[attempt - 1],
                    error=f"{type(exc).__name__}: {exc}",
                    reserved_operation=reservation,
                )
                if attempt >= len(STALL_ATTEMPT_TIMEOUTS):
                    raise RuntimeError(
                        f"{OPENAI_COMPAT_PROVIDER} gateway produced no complete response in "
                        f"{sum(STALL_ATTEMPT_TIMEOUTS)}s across {attempt} attempts; retained "
                        "reservations need reconciliation"
                    ) from exc
                # The stalled operation keeps its reservation — its real cost is
                # unknown — and the next attempt reserves afresh through the same
                # admission, so the stage allowance covers both.
                turn, reservation, ceiling = await self._admit(body, forward)
                self.record(
                    "model_request",
                    turn=turn,
                    model=self.model_id,
                    system=body.get("system"),
                    messages=body.get("messages", [])[-1:],
                    tools=[tool.get("name") for tool in body.get("tools", [])],
                    tool_choice=body.get("tool_choice"),
                    inference=inference_settings(self.model),
                    timeout_sec=MODEL_TIMEOUT_SEC,
                    attempt=attempt + 1,
                )
        usage = data.get("usage") or {}
        message = openai_wire.from_openai_response(data, model=self.model_id)
        try:
            cost = gateway_usage_cost(usage, self.rates)
        except ValueError:
            self.record(
                "model",
                turn=turn,
                message=message,
                usage=usage,
                cost_usd=None,
                reservation_usd=ceiling,
            )
            raise
        self.budget.settle(reservation, cost)
        self.cost += cost
        self.record(
            "model", turn=turn, message=message, usage=usage, cost_usd=cost, attempt=attempt
        )
        validate_provider_response(message["content"], message["stop_reason"])
        if not stream:
            return web.json_response(message)
        return await self._serve(request, openai_wire.message_events(message))

    async def _serve(self, request, events):
        """A completed turn replayed as Anthropic SSE in the runtime's own format."""
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        payload = b"".join(
            f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n".encode()
            for event in events
        )
        with contextlib.suppress(ConnectionError):
            await response.write(payload)
        with contextlib.suppress(ConnectionError):
            await response.write_eof()
        return response

    async def _rates(self) -> tuple[float, float, float]:
        """USD per token — (input, output, cache read) — for the matched route.

        The gateway prices its own catalog, and that published rate is what
        actually bills us, so it wins; LiteLLM only knows its own ids. A model
        neither can price is rejected here, before its first reserved turn.
        """
        if self.rates is not None:
            return self.rates
        import litellm

        if self.compat:
            rates = await self._catalog_rates()
            if rates is None:
                try:
                    info = litellm.get_model_info(f"openai/{self.model_id}")
                except Exception:
                    info = {}
                rates = (
                    info.get("input_cost_per_token"),
                    info.get("output_cost_per_token"),
                    info.get("cache_read_input_token_cost"),
                )
                if not rates[0] or not rates[1]:
                    raise ValueError(
                        f"Cannot budget {self.model}: the gateway publishes no pricing for "
                        f"{self.model_id!r} and LiteLLM has no price for openai/{self.model_id}"
                    )
            if rates[2] is None:
                rates = (rates[0], rates[1], rates[0] * 0.1)
            self.rates = tuple(float(rate) for rate in rates)
            return self.rates
        pricing = litellm.get_model_info(self.model)
        input_rate, output_rate = pricing["input_cost_per_token"], pricing["output_cost_per_token"]
        if not input_rate or not output_rate:
            raise ValueError("Unknown model price")
        self.rates = (float(input_rate), float(output_rate), float(input_rate) * 0.1)
        return self.rates

    async def _catalog_rates(self) -> tuple[float, float, float] | None:
        """Published per-token rates for this model, or None when none are published.

        Read at most once per bridge. A catalog that cannot be read falls back
        to LiteLLM instead of blocking a route that could otherwise run, and a
        model the catalog omits is simply not this function's answer.
        """
        url = self.gateway_url.rstrip("/") + "/models"
        try:
            async with asyncio.timeout(20):
                async with self.http.get(url, headers=self.gateway_headers) as response:
                    if response.status != 200:
                        return None
                    catalog = await response.json()
        except (TimeoutError, ClientError, ValueError):
            return None
        if not isinstance(catalog, dict):
            return None
        for entry in catalog.get("data") or []:
            if not isinstance(entry, dict) or entry.get("id") != self.model_id:
                continue
            pricing = entry.get("pricing")
            if not isinstance(pricing, dict):
                return None
            input_rate, output_rate = (
                pricing.get("input_usd_per_1m"),
                pricing.get("output_usd_per_1m"),
            )
            if input_rate is None or output_rate is None:
                return None
            cache_rate = pricing.get("cache_read_usd_per_1m") or 0
            return (
                float(input_rate) / 1_000_000,
                float(output_rate) / 1_000_000,
                float(cache_rate) / 1_000_000,
            )
        return None


def validate_provider_response(content, stop_reason):
    """A metered model turn needs usable text or a tool call, not just thinking."""
    if stop_reason in {"max_tokens", "length"}:
        raise ProviderOutputError(f"Provider response truncated: stop_reason={stop_reason}")
    if not any(
        (block.get("type") == "text" and (block.get("text") or "").strip())
        or block.get("type") == "tool_use"
        for block in content
    ):
        raise ProviderOutputError(
            "Provider response contained no text or tool calls (thinking-only or empty)"
        )


def normalize_cache(value):
    """Use the same five-minute cache pricing ceiling as the LangGraph baseline."""
    if isinstance(value, dict):
        if isinstance(value.get("cache_control"), dict):
            value["cache_control"] = {"type": "ephemeral"}
        for child in value.values():
            normalize_cache(child)
    elif isinstance(value, list):
        for child in value:
            normalize_cache(child)


def gateway_usage_cost(usage, rates) -> float:
    """One OpenAI-wire turn at the gateway's own rates, cache reads at their cache rate.

    `usage` and `rates` carry no pieces of either protocol's accounting: token
    counts come straight from the gateway's usage object and prices from the
    resolved route, so nothing about Anthropic's cache multipliers is assumed.
    """
    incoming, outgoing = usage.get("prompt_tokens"), usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    if incoming is None or outgoing is None:
        raise ValueError("Provider returned incomplete usage; retain reservation")
    if any(type(value) is not int or value < 0 for value in (incoming, outgoing, cached)):
        raise ValueError("Provider usage must contain nonnegative integer token counts")
    if cached > incoming:
        raise ValueError("Provider usage reports cache reads beyond its input tokens")
    if rates is None:
        raise ValueError("Gateway pricing was never resolved")
    input_rate, output_rate, cache_rate = rates
    return (incoming - cached) * input_rate + cached * cache_rate + outgoing * output_rate


def usage_cost(usage, input_rate, output_rate):
    if "input_tokens" not in usage or "output_tokens" not in usage:
        raise ValueError("Provider returned incomplete usage; retain reservation")
    cache = usage.get("cache_creation", {})
    long = cache.get("ephemeral_1h_input_tokens", 0)
    short = cache.get(
        "ephemeral_5m_input_tokens", max(0, usage.get("cache_creation_input_tokens", 0) - long)
    )
    return (
        usage["input_tokens"]
        + short * 1.25
        + long * 2
        + usage.get("cache_read_input_tokens", 0) * 0.1
    ) * input_rate + usage["output_tokens"] * output_rate
