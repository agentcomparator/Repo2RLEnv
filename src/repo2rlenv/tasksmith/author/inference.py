from __future__ import annotations

# A design submission can include the complete private verifier plus model reasoning.
# This is an output ceiling; each request still reserves against the same stage budget.
MAX_OUTPUT_TOKENS = 12000
# The runtime's own abort must stay ABOVE the bridge's stall-recovery window
# (bridge.STALL_ATTEMPT_TIMEOUTS = 240+560) so an adapter never cancels a live
# recovery, and below the stage deadline (runner.author) so a recovered turn can
# still finish inside its stage. Live runs died at 300s and then 600s to silent
# gateway windows; normal completions observed under 60s.
MODEL_TIMEOUT_SEC = 1150


def anthropic_options(model: str) -> dict:
    adaptive_models = (
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
    )
    if model.startswith("anthropic/") and model.split("/", 1)[1].startswith(adaptive_models):
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    return {}


def inference_settings(model: str, *, max_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    return {"model": model, "max_tokens": max_tokens, **anthropic_options(model)}
